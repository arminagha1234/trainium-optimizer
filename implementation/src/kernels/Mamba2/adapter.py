# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: bind the on-device-validated Mamba-2 SSD (state-space duality)
selective-scan kernel to the framework's GENERIC injection hook
(``backends.kernel_inject``).

Design mirrors ``kernels/FlashAttention/adapter.py`` and the ``kernel_inject``
contract:

  * **Torch-free / NKI-free at import.** Importing this module pulls in NOTHING
    heavy — no torch, no ``nki``, no ``nkilib``, no ``numpy``. That is what keeps
    the registry -> retrieve -> inject WIRING unit-testable on a plain CPU box.
    The SSD kernel is loaded lazily (``import nkilib.experimental.scan.ssd``)
    only the first time a forward actually runs on a Trainium box.

  * **Forward-factory entry.** ``build_mamba2_forward(module)`` is the
    ``kernel_inject`` entry contract: handed the target module, it RETURNS a
    replacement ``forward``. The manifest points ``entry`` at
    ``adapter:build_mamba2_forward`` and ``path`` at this file, so
    ``inject_kernel`` / ``load_kernel_entry`` resolve it exactly like any other
    kernel.

## Kernel provenance / IP boundary

Unlike KDA and DeltaNet, the SSD kernel source is NOT vendored here: it ships in
the ``nkilib`` package (``nkilib.experimental.scan.ssd``,
github.com/aws-neuron/nki-library) which is installed on the target box. The
adapter references it by import — the manifest records the origin. This is the
"harvest-source-referenced (installed package)" registration mode.

## Invocation contract (reproduces the on-device validation)

The raw kernel is ``ssd(x, dt, A, B, C, chunk_size=128, D=None,
initial_state=None, causal_mask=None) -> (y, final_state)`` with layouts::

    x[B,H,L,P]  dt[B,H,L]  A[H]  B[B,L,N]  C[B,L,N]  D[H]
    initial_state[B,H,N,P] ;  y[B,H,L,P]  final_state[B,H,N,P]

The built forward accepts model-native Mamba-2 mixer tensors (ngroups=1)::

    x  : [batch, seqlen, nheads, headdim]
    dt : [batch, seqlen, nheads]   (already softplus'd, positive)
    A  : [nheads]                  (negative; = -exp(A_log))
    B,C: [batch, seqlen, dstate]
    D  : [nheads] or None

transposes into the kernel layout, calls the kernel, and returns
``y : [batch, seqlen, nheads, headdim]`` plus ``final_state``.

On-device validated (trn2, neuronx-cc 2.27 / torch_xla 2.9): compiles + matches
``ssd_torch_ref`` to ~1e-7 (fp32-exact); a tiny 2-layer Mamba-2 end-to-end
matched the reference at cosine 1.0.

Honest scope (like FlashAttention's documented gap): the built forward is an
SSD-OP forward taking the mixer tensors already split out. Bridging a full HF
Mamba-2 mixer module (in/out projections, conv1d, dt/A/D parameterization, gated
RMSNorm) to this op is the documented next step; this adapter wires the kernel in
and is exercised at the op level.
"""

from __future__ import annotations

from typing import Any, Callable

# Provenance stamp (recorded alongside the banked lesson / ledger, mirroring
# FlashAttention's KERNEL_SOURCE).
KERNEL_SOURCE = (
    "nkilib.experimental.scan.ssd (Mamba-2 state-space-duality selective scan; "
    "github.com/aws-neuron/nki-library). On-device validated on trn2: matches "
    "ssd_torch_ref ~1e-7 (fp32-exact); 2-layer Mamba-2 end-to-end cosine 1.0. "
    "Constraints: ngroups=1, dstate<=128, chunk<=128, seqlen % chunk == 0."
)

# Kernel constraints (documented; asserted in the forward where cheap).
CHUNK_MAX = 128
DSTATE_MAX = 128


def _load_ssd():
    """Import the SSD kernel from the installed ``nkilib`` package. Only ever
    called on a Trainium box (from inside the built forward), so ``nkilib`` /
    ``nki`` never load on a CPU box."""
    from nkilib.experimental.scan.ssd import ssd  # noqa: PLC0415 - device-only
    return ssd


def build_mamba2_forward(module: Any) -> Callable:
    """``kernel_inject`` forward-factory: return a ``forward(x, dt, A, B, C, ...)``
    that runs the validated SSD NKI kernel.

    The returned forward is the ONLY place ``numpy`` / ``nkilib`` are touched, and
    it reproduces the validated invocation exactly: transpose the mixer tensors
    into the kernel's ``[B,H,L,P]`` layout, build the lower-triangular causal
    mask, call ``ssd``, and transpose ``y`` back to ``[B,L,H,P]``.
    """
    # Allow a caller to override chunk_size via a module attribute; default 128.
    chunk_size = int(getattr(module, "ssd_chunk_size", CHUNK_MAX))
    state: dict[str, Any] = {}

    def forward(x, dt, A, B, C, D=None, initial_state=None):
        import numpy as np  # noqa: PLC0415 - device-only

        ssd = state.get("ssd")
        if ssd is None:
            ssd = _load_ssd()
            state["ssd"] = ssd

        seqlen = x.shape[1]
        dstate = B.shape[-1]
        assert chunk_size <= CHUNK_MAX, f"SSD chunk_size must be <= {CHUNK_MAX}"
        assert dstate <= DSTATE_MAX, f"SSD dstate must be <= {DSTATE_MAX}"
        assert seqlen % chunk_size == 0, (
            f"SSD seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})")

        xk = np.ascontiguousarray(np.transpose(x, (0, 2, 1, 3)).astype(np.float32))
        dtk = np.ascontiguousarray(np.transpose(dt, (0, 2, 1)).astype(np.float32))
        cm = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32))
        kw = dict(x=xk, dt=dtk, A=np.asarray(A, np.float32),
                  B=np.asarray(B, np.float32), C=np.asarray(C, np.float32),
                  chunk_size=chunk_size, causal_mask=cm)
        if D is not None:
            kw["D"] = np.asarray(D, np.float32)
        if initial_state is not None:
            kw["initial_state"] = np.asarray(initial_state, np.float32)

        y, final_state = ssd(**kw)                       # on-device NKI
        y = np.transpose(np.asarray(y), (0, 2, 1, 3))    # -> [B, L, H, P]
        return y, np.asarray(final_state)

    return forward
