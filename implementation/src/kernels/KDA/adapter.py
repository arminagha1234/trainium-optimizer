# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: bind the on-device-validated KDA (Kimi Delta Attention) NKI kernels
to the framework's GENERIC injection hook (``backends.kernel_inject``).

Design mirrors ``kernels/FlashAttention/adapter.py`` and the ``kernel_inject``
contract:

  * **Torch-free / NKI-free at import.** Importing this module pulls in NOTHING
    heavy — no ``nki``, no ``numpy`` — so the registry -> retrieve -> inject
    WIRING stays unit-testable on a plain CPU box. The vendored kernel files
    (``nki_kda.py``, ``nki_kda_chunked_exact.py``) are loaded lazily, from their
    sibling files, the first time a forward runs on a Trainium box.

  * **Forward-factory entry.** ``build_kda_forward(module)`` is the
    ``kernel_inject`` entry contract: handed the target module, it RETURNS a
    replacement ``forward``. The manifest points ``entry`` at
    ``adapter:build_kda_forward`` and ``path`` at this file.

## Multi-variant dispatch (prefill + decode)

KDA has two on-device-validated variants. Rather than two manifests, the single
forward-factory selects the variant from an optional ``kda_mode`` attribute on
the injected module (default ``"prefill"``):

  * ``"prefill"`` -> ``kda_chunk_step_exact`` (nki_kda_chunked_exact.py): exact
    per-channel chunked path, 128-token chunks with cross-chunk state carry.
  * ``"decode"``  -> ``kda_recurrent_fwd_state`` (nki_kda.py): token-serial
    recurrent path.

This mirrors how ``FlashAttention/adapter.py`` selects among flash_fwd /
flash_fwd_fp32 / flash_fwd_causal from module attributes.

## Vendored kernel source / IP boundary

``nki_kda.py`` and ``nki_kda_chunked_exact.py`` are vendored verbatim next to
this adapter (Apache-2.0; see ./LICENSE). They import the standalone ``nki``
package and are ``@nki.jit`` kernels callable directly on numpy arrays on a
Trainium box.

## Input contract (fla-core convention — the forward preprocesses q/k)

  * q, k are L2-normalized along the head dim; q additionally scaled by
    ``1/sqrt(head_dim)`` — done inside the forward from the RAW projections.
  * g is the per-channel log-decay (<= 0), activated, NOT cumulative (the chunk
    kernel does its own cumsum).
  * beta is the per-token scalar write gate.
  * state carries [dk, dv] = [128, 128].

Constraints: head_k_dim == head_v_dim == 128; chunk_size == 128; prefill seqlen
divisible by 128; float32 inputs (gate/exponent path must stay fp32 for
exactness). On-device validated on trn2 (nki 0.6.0): cos 1.0, max_err ~1e-7/1e-8
across seqlens and gate scales incl. g=2.0, with exact cross-chunk state carry.

Honest scope (like FlashAttention's documented gap): the built forward is a
per-(batch, head) KDA-OP forward; the caller loops B*H and handles the fla-core
q/k RoPE / GQA expansion. Bridging a full HF KDA layer is the documented next
step; this adapter wires the kernels in and is exercised at the op level.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

# Provenance stamp (mirrors FlashAttention's KERNEL_SOURCE).
KERNEL_SOURCE = (
    "KDA (Kimi Delta Attention) NKI kernels, vendored from "
    "jburtoft/kda-neuron-kernels (HuggingFace, build/torch-neuron/; Apache-2.0). "
    "On-device validated on trn2 (nki 0.6.0): kda_chunk_step_exact (prefill) and "
    "kda_recurrent_fwd_state (decode) cos 1.0, max_err ~1e-7/1e-8."
)

_HERE = Path(__file__).resolve().parent
DK = 128  # required head dim (NeuronCore SBUF partition width)

# (mode -> (kernel_file, entry_symbol))
_VARIANTS = {
    "prefill": ("nki_kda_chunked_exact.py", "kda_chunk_step_exact"),
    "decode": ("nki_kda.py", "kda_recurrent_fwd_state"),
}


def _load_entry(filename: str, symbol: str):
    """Import a vendored sibling kernel file from disk (real ``__file__`` so the
    NKI tracer can introspect source) and return its entry callable. Only ever
    called on a Trainium box (from inside the built forward)."""
    fpath = _HERE / filename
    spec = importlib.util.spec_from_file_location(
        "kda_kernel_" + Path(filename).stem, str(fpath))
    if spec is None or spec.loader is None:  # pragma: no cover - wiring failure
        raise ImportError(f"cannot load KDA kernel from {fpath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def build_kda_forward(module: Any) -> Callable:
    """``kernel_inject`` forward-factory: return a
    ``forward(q_raw, k_raw, v, g, beta, initial_state=None)`` that runs the
    validated KDA NKI kernel for this module's requested mode.

    The returned forward is the ONLY place ``numpy`` / ``nki`` are touched.
    """
    mode = str(getattr(module, "kda_mode", "prefill"))
    if mode not in _VARIANTS:
        raise ValueError(f"KDA mode must be one of {sorted(_VARIANTS)}; got {mode!r}")
    filename, symbol = _VARIANTS[mode]
    state: dict[str, Any] = {}

    def forward(q_raw, k_raw, v, g, beta, initial_state=None):
        import numpy as np  # noqa: PLC0415 - device-only

        entry = state.get("entry")
        if entry is None:
            entry = _load_entry(filename, symbol)
            state["entry"] = entry

        S = q_raw.shape[0]
        assert q_raw.shape[-1] == DK, f"KDA kernel requires head_dim == {DK}"

        # fla-core preprocess: L2-norm q,k on head dim; scale q by 1/sqrt(dk).
        qn = q_raw / (np.linalg.norm(q_raw, axis=-1, keepdims=True) + 1e-12)
        kn = k_raw / (np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-12)
        q = (qn * (DK ** -0.5)).astype(np.float32)
        k = kn.astype(np.float32)
        v = np.ascontiguousarray(v.astype(np.float32))
        g = np.ascontiguousarray(g.astype(np.float32))
        beta = np.asarray(beta, dtype=np.float32).reshape(S, 1)
        st = (np.zeros((DK, DK), np.float32) if initial_state is None
              else np.asarray(initial_state, np.float32))

        if mode == "decode":
            beta_bc = np.ascontiguousarray(np.broadcast_to(beta, (S, DK)))
            out, final_state = entry(q, k, v, g, beta_bc)
            return np.asarray(out), np.asarray(final_state)

        # prefill: loop 128-token chunks, carrying [k, v] state
        assert S % DK == 0, "prefill seqlen must be divisible by chunk_size=128"
        outs = []
        for c0 in range(0, S, DK):
            sl = slice(c0, c0 + DK)
            co, st = entry(
                np.ascontiguousarray(q[sl]), np.ascontiguousarray(k[sl]),
                np.ascontiguousarray(v[sl]), np.ascontiguousarray(beta[sl]),
                np.ascontiguousarray(g[sl]), np.ascontiguousarray(st))
            outs.append(np.asarray(co))
            st = np.asarray(st)
        return np.concatenate(outs, axis=0), st

    return forward
