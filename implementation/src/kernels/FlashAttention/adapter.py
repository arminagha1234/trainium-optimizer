# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: bind the on-device-validated flash-attention NKI kernel
(``flash_nki_opt.py``) to the framework's GENERIC injection hook
(``backends.kernel_inject``).

Design mirrors ``kernels/moe_fused/adapter.py`` and the ``kernel_inject`` contract:

  * **Torch-free / NKI-free at import.** Importing this module pulls in NOTHING
    heavy — no torch, no neuronxcc, no torch_xla. That is what makes the whole
    registry->retrieve->inject WIRING unit-testable on a plain CPU box (the flash
    kernel itself only imports ``neuronxcc.nki.*`` when the built forward is
    actually CALLED on a Trainium box). ``flash_nki_opt.py`` is loaded lazily,
    from its sibling file, the first time a forward runs.

  * **Forward-factory entry.** ``build_flash_forward(module)`` is the
    ``kernel_inject`` entry contract: it is handed the target module and RETURNS a
    replacement ``forward``. The registry manifest points ``entry`` at
    ``adapter:build_flash_forward`` and ``path`` at this file, so
    ``inject_kernel``/``load_kernel_entry`` resolve it exactly like any other
    kernel — the flash kernel is not a hardcoded special case.

Invocation contract (REPRODUCES how the kernel was validated on-device): the
flash entry points are output-as-arg NKI kernels invoked via
``torch_neuronx.nki_jit`` with the destination tensor passed as the trailing
argument. Layout matches the kernel docstring: ``q,k,v`` are ``(d_head, seqlen)``
and the output is ``(seqlen, d_head)``; scores are UNSCALED. The forward selects
``flash_fwd`` (bf16, non-causal), ``flash_fwd_fp32`` (fp32, non-causal), or
``flash_fwd_causal`` (bf16 + causal) from optional module attributes.

Honest scope (like the moe_fused adapter's documented gap): the built forward is
an ATTENTION-OP forward taking q/k/v already in the kernel's (d_head, seqlen)
layout. Bridging a full HF attention module (q/k/v projections, head reshape,
GQA, scaling, RoPE) to this op is the documented next step — this adapter wires
the kernel in and is exercised on-device through the framework at the op level.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

# Provenance stamp (recorded alongside the banked lesson / ledger, mirroring
# moe_fused's KERNEL_SOURCE). The kernel is banked verbatim from the validated
# artifact staged at /tmp/flash_nki_opt.py.
KERNEL_SOURCE = "flash_nki_opt (streaming online-softmax flash-attention, "\
    "on-device validated S=2048/4096/8192, bf16 max_err ~1e-2 vs bf16 incumbent)"

_HERE = Path(__file__).resolve().parent
_KERNEL_FILE = _HERE / "flash_nki_opt.py"

# Which entry point handles which (dtype, causal) request. Chosen from optional
# attributes on the injected module so a caller can request causal / fp32 without
# a second manifest.
_ENTRY_NONCAUSAL_BF16 = "flash_fwd"
_ENTRY_NONCAUSAL_FP32 = "flash_fwd_fp32"
_ENTRY_CAUSAL_BF16 = "flash_fwd_causal"


def _load_kernel_module():
    """Import the sibling ``flash_nki_opt.py`` from disk (real ``__file__`` so the
    NKI tracer can introspect source), returning the module. Only ever called on
    a Trainium box (from inside the built forward), so the top-level
    ``neuronxcc.nki`` imports in the kernel never load on a CPU box."""
    spec = importlib.util.spec_from_file_location(
        "flash_nki_opt_kernel", str(_KERNEL_FILE))
    if spec is None or spec.loader is None:  # pragma: no cover - wiring failure
        raise ImportError(f"cannot load flash kernel from {_KERNEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_entry(module: Any) -> str:
    """Pick the flash entry-point name for this module's request."""
    causal = bool(getattr(module, "flash_causal", False))
    want_fp32 = bool(getattr(module, "flash_fp32", False))
    if causal:
        # Only a bf16 causal entry exists in the validated kernel.
        return _ENTRY_CAUSAL_BF16
    return _ENTRY_NONCAUSAL_FP32 if want_fp32 else _ENTRY_NONCAUSAL_BF16


def build_flash_forward(module: Any) -> Callable:
    """``kernel_inject`` forward-factory: return a ``forward(q, k, v)`` that runs
    the validated flash-attention NKI kernel.

    The returned forward is the ONLY place torch / neuronxcc are touched, and it
    reproduces the validated invocation exactly: wrap the chosen output-as-arg
    entry with ``torch_neuronx.nki_jit`` and call it with a freshly-allocated
    ``out`` tensor as the trailing argument. ``q,k,v`` are ``(d_head, seqlen)``;
    the returned output is ``(seqlen, d_head)``.
    """
    entry_name = _select_entry(module)
    # Cache the jitted kernel across calls on the closure (built on first use so
    # module import stays torch/nki-free).
    state: dict[str, Any] = {}

    def forward(q, k, v):
        import torch  # noqa: PLC0415 - device-only
        import torch_neuronx  # noqa: PLC0415 - device-only

        jit = state.get("jit")
        if jit is None:
            kmod = _load_kernel_module()
            entry = getattr(kmod, entry_name)
            jit = torch_neuronx.nki_jit(entry)
            state["jit"] = jit

        d_head, seqlen = q.shape
        out = torch.zeros((seqlen, d_head), dtype=q.dtype, device=q.device)
        jit(q, k, v, out)          # output-as-arg (torch_neuronx.nki_jit)
        return out

    return forward
