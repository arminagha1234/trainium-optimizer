# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: bind the on-device-validated Qwen3.5 Gated DeltaNet NKI kernels to
the framework's GENERIC injection hook (``backends.kernel_inject``).

This is the PERF path for the Qwen3.5 / Qwen3-Next ``linear_attention``
(GatedDeltaNet, gated delta-rule) primitive — the canonical registry slot
``DeltaNet`` (harvest identity ``GatedDeltaNet``). It replaces the reference-torch
``chunk_gated_delta_rule`` that the arch-proof graph rewrites emit.

Design mirrors ``kernels/FlashAttention/adapter.py`` and the ``kernel_inject``
contract:

  * **Torch-free / NKI-free at import.** Importing this module pulls in NOTHING
    heavy — no ``nki``, no ``numpy`` — so the registry -> retrieve -> inject
    WIRING stays unit-testable on a plain CPU box. The vendored kernel package
    (``gdn_src/``) is loaded lazily the first time a forward runs on a Trainium
    box.

  * **Forward-factory entry.** ``build_gdn_forward(module)`` is the
    ``kernel_inject`` entry contract: handed the target module, it RETURNS a
    replacement ``forward``. The manifest points ``entry`` at
    ``adapter:build_gdn_forward`` and ``path`` at this file.

## Multi-variant dispatch (decode / prefill-perf / short-prefill)

GatedDeltaNet has several on-device-validated variants. Rather than a manifest
per variant, the single forward dispatches by shape (mirroring
``NeuronGatedDeltaNet.forward`` in the harvested layers.py)::

    seq_len == 1  and cached state -> deltanet_tkg_batched_bh          (DECODE)
    seq_len >= 128 (S % 128 == 0)  -> deltanet_fused_chunked_fwd_batched (PREFILL perf)
    seq_len  < 128 and cached      -> deltanet_recurrent_fwd_state      (short prefill)
    seq_len  < 128 no cache        -> deltanet_recurrent_fwd           (short prefill)

## Vendored kernel source / IP boundary

The kernel package is vendored verbatim next to this adapter under
``gdn_src/`` (Apache-2.0 — original NKI work published under Apache 2.0 per the
source repo README; see ./NOTICE). ``gdn_src/nki_kernels/*.py`` are ``@nki.jit``
kernels callable directly on numpy arrays on a Trainium box, plus the shared
``gdn_src/constants.py`` (P_MAX, _BROADCAST_MASK). The kernels import the
standalone ``nki`` package.

## Input contract (fla-style gated delta-rule)

Per-request batched tensors (all B*H (request, head) slices stacked on axis 0):

    query, key, value : [BH, S, 128]  (RAW query/key; the forward L2-norms both
                        and scales q by 1/sqrt(D))
    g                 : [BH, S]       per-head scalar RAW log-decay (<= 0);
                        the chunked kernel does its own cumsum
    beta              : [BH, S]       per-token write gate (post-sigmoid)
    initial_state     : [BH, 128, 128] or None (cached-state decode/short prefill)

Constraints: head_k_dim == head_v_dim == 128; gate g per-head scalar <= 0;
chunked path requires S % 128 == 0; float32. No group axis — GQA expansion of
q/k happens in the layer BEFORE the kernel. On-device validated on trn2
(nki 0.6.0): out cos 1.0 (max_err ~3e-7 chunked, ~1.5e-5 tkg), final_state cos 1.0.

Honest scope (like FlashAttention's documented gap): the built forward is a
GatedDeltaNet-OP forward over pre-split, pre-batched slices; bridging a full HF
GatedDeltaNet mixer (projections, conv, gating parameterization, GQA, cache
plumbing) is the documented next step. This adapter wires the kernels in with the
same dispatch the harvested layer uses.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

# Provenance stamp (mirrors FlashAttention's KERNEL_SOURCE).
KERNEL_SOURCE = (
    "Qwen3.5 Gated DeltaNet NKI kernels, vendored from HF "
    "kernels/jburtoft/qwen35-deltanet-tkg-full (v4; Apache-2.0). On-device "
    "validated on trn2 (nki 0.6.0): out cos 1.0 (chunked max_err ~3e-7, tkg "
    "~1.5e-5), final_state cos 1.0."
)

_HERE = Path(__file__).resolve().parent
DK = 128  # required head dim (== P_MAX, SBUF partition width)

# module path (under vendored gdn_src package) -> entry symbol, per variant.
_PREFILL = "gdn_src.nki_kernels.deltanet_fused_chunked_fwd_batched"
_DECODE = "gdn_src.nki_kernels.deltanet_tkg_batched_bh"
_RECURRENT = "gdn_src.nki_kernels.deltanet_recurrent_fwd"
_RECURRENT_STATE = "gdn_src.nki_kernels.deltanet_recurrent_fwd_state"


def _load(module_path: str, symbol: str):
    """Import a vendored kernel from the ``gdn_src`` package (relative imports
    resolve because it is a real package). Only ever called on a Trainium box
    (from inside the built forward)."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    mod = importlib.import_module(module_path)
    return getattr(mod, symbol)


def build_gdn_forward(module: Any) -> Callable:
    """``kernel_inject`` forward-factory: return a
    ``forward(query, key, value, g, beta, initial_state=None)`` that runs the
    validated GatedDeltaNet NKI kernel, dispatching by shape.

    The returned forward is the ONLY place ``numpy`` / ``nki`` are touched.
    """
    state: dict[str, Any] = {}

    def _preprocess(np, query, key):
        qn = query / (np.linalg.norm(query, axis=-1, keepdims=True) + 1e-6)
        kn = key / (np.linalg.norm(key, axis=-1, keepdims=True) + 1e-6)
        q = (qn * (DK ** -0.5)).astype(np.float32)
        k = kn.astype(np.float32)
        return q, k

    def forward(query, key, value, g, beta, initial_state=None):
        import numpy as np  # noqa: PLC0415 - device-only

        query = np.asarray(query, np.float32)
        key = np.asarray(key, np.float32)
        value = np.asarray(value, np.float32)
        g = np.asarray(g, np.float32)
        beta = np.asarray(beta, np.float32)
        BH, S = query.shape[0], query.shape[1]
        assert query.shape[-1] == DK, f"GatedDeltaNet requires head_dim == {DK}"
        cached = initial_state is not None

        # ---- DECODE: single token, cached state -> tkg_batched_bh ----------
        if S == 1 and cached:
            entry = state.get("decode") or _load(_DECODE, "deltanet_tkg_batched_bh")
            state["decode"] = entry
            q, k = _preprocess(np, query, key)
            # (BH, 1, D) -> (BH, D, 1); g/beta per-head scalar broadcast to (BH, D, 1)
            q3 = np.ascontiguousarray(np.transpose(q, (0, 2, 1)))
            k3 = np.ascontiguousarray(np.transpose(k, (0, 2, 1)))
            v3 = np.ascontiguousarray(np.transpose(value, (0, 2, 1)))
            g3 = np.ascontiguousarray(np.broadcast_to(
                g.reshape(BH, 1, 1), (BH, DK, 1)).astype(np.float32))
            b3 = np.ascontiguousarray(np.broadcast_to(
                beta.reshape(BH, 1, 1), (BH, DK, 1)).astype(np.float32))
            st = np.ascontiguousarray(np.asarray(initial_state, np.float32))
            o, state_out = entry(q3, k3, v3, g3, b3, st)
            o = np.transpose(np.asarray(o), (0, 2, 1))       # -> (BH, 1, D)
            return o, np.asarray(state_out)

        # ---- PREFILL perf: S % 128 == 0 -> fused_chunked_fwd_batched -------
        if S >= DK and S % DK == 0:
            entry = state.get("prefill")
            masks = state.get("masks")
            if entry is None:
                if str(_HERE) not in sys.path:
                    sys.path.insert(0, str(_HERE))
                kmod = importlib.import_module(_PREFILL)
                entry = kmod.deltanet_fused_chunked_fwd_batched
                masks = (kmod._make_lower_mask(), kmod._make_identity(),
                         kmod._make_lower_mask_diag())
                state["prefill"] = entry
                state["masks"] = masks
            lm, iden, lmd = masks
            q, k = _preprocess(np, query, key)
            g_in = g.reshape(BH, S, 1).astype(np.float32)
            beta_in = beta.reshape(BH, S, 1).astype(np.float32)
            init = (np.zeros((BH, DK, DK), np.float32) if not cached
                    else np.ascontiguousarray(np.asarray(initial_state, np.float32)))
            out = entry(q, k, value, g_in, beta_in, init, lm, iden, lmd)
            return np.asarray(out[0]), np.asarray(out[1])

        # ---- SHORT PREFILL: S < 128 -> recurrent (per-slice loop) ----------
        sym = "deltanet_recurrent_fwd_state" if cached else "deltanet_recurrent_fwd"
        path = _RECURRENT_STATE if cached else _RECURRENT
        entry = state.get(sym) or _load(path, sym)
        state[sym] = entry
        q, k = _preprocess(np, query, key)
        outs, states = [], []
        for i in range(BH):
            g_bc = np.ascontiguousarray(np.broadcast_to(
                g[i].reshape(S, 1), (S, DK)).astype(np.float32))
            b_bc = np.ascontiguousarray(np.broadcast_to(
                beta[i].reshape(S, 1), (S, DK)).astype(np.float32))
            args = [np.ascontiguousarray(q[i]), np.ascontiguousarray(k[i]),
                    np.ascontiguousarray(value[i]), g_bc, b_bc]
            if cached:
                args.append(np.ascontiguousarray(
                    np.asarray(initial_state[i], np.float32)))
                o, so = entry(*args)
                outs.append(np.asarray(o))
                states.append(np.asarray(so))
            else:
                outs.append(np.asarray(entry(*args)))
        out = np.stack(outs, axis=0)
        if cached:
            return out, np.stack(states, axis=0)
        return out, None

    return forward
