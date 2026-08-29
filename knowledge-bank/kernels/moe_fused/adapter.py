# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Adapter: bind the vendored fused MoE megakernel to a HuggingFace Qwen3-MoE
model as a Stage-3 BORROW candidate.

This module is the ONE place that decides whether the borrowed kernel may run
for a given model, and — when it may — performs the forward swap. It is
deliberately dependency-light: precondition logic imports nothing heavy, so the
framework core and unit tests can reason about "is this kernel offered / can it
run here?" on any box (no torch, no nki, no Neuron). The actual swap
(`swap_moe_forward`) imports torch lazily and NEVER raises — on any unmet
precondition it returns (False, reason) so the worker cleanly falls back to the
eager HF MoE, and the tournament's equivalence gate then simply sees an
unchanged (correct, non-faster) candidate rather than a crash.

Honest scope (see ./README.md and ./NOTICE for the full gap analysis):

  The vendored kernel is a DECODE-time (TKG) megakernel hardcoded to
  Qwen3-30B-A3B at TP=4 (hidden=2048, experts=128, top_k=8,
  moe_intermediate=768), traced through the NxDI / torch_xla stack, and
  dependent on the private `nki-library` (`nkilib.core.*`) package. The
  framework's native-PyTorch backend runs plain HF models with DTensor TP and
  measures PREFILL. Those two facts mean the kernel self-skips on every model
  except an exact Qwen3-30B-A3B/TP4 deployment on an environment that provides
  `nkilib` and torch_xla. That is expected and documented — the wiring below is
  complete and equivalence-gated; on-device execution for A3B is the deferred
  next step.
"""

from __future__ import annotations

from typing import Any

# Provenance stamp recorded in the ledger `source` column and the bank lesson.
KERNEL_SOURCE = "nki-moe-megakernel@5879c39"

# The vendored kernel is compiled for these EXACT dims (moe_fused_nki._H etc.).
# A model/deployment must match all of them for the kernel to run.
SUPPORTED_CONTRACT = {
    "hidden_size": 2048,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
    "tp_degree": 4,
}

# The config-axis value that requests this borrow. Mirrors the `place:*` axis
# convention: a plain config key the backend/worker recognize.
MOE_KERNEL_KEY = "moe_kernel"
FUSED_NKI = "fused_nki"


def is_moe_arch(hf_config: Any) -> bool:
    """True if the HF config describes a (sparse) MoE causal LM.

    Detection is best-effort off the config, matching how the backend detects
    diffusion components: an `architectures` entry containing 'Moe', or the
    presence of a routed-expert count (`num_experts` / `num_local_experts`).
    """
    archs = " ".join(getattr(hf_config, "architectures", []) or [])
    if "Moe" in archs or "MoE" in archs:
        return True
    for attr in ("num_experts", "num_local_experts"):
        v = getattr(hf_config, attr, None)
        if isinstance(v, int) and v > 1:
            return True
    return False


def _cfg_get(hf_config: Any, name: str) -> Any:
    # top-K is spelled a couple of ways across HF MoE configs.
    if name == "num_experts_per_tok":
        for alt in ("num_experts_per_tok", "moe_topk", "top_k"):
            v = getattr(hf_config, alt, None)
            if isinstance(v, int):
                return v
        return None
    if name == "num_experts":
        for alt in ("num_experts", "num_local_experts"):
            v = getattr(hf_config, alt, None)
            if isinstance(v, int):
                return v
        return None
    return getattr(hf_config, name, None)


def precheck(hf_config: Any, tp_degree: int) -> tuple[bool, str]:
    """Can the vendored kernel run for this model + TP degree? (ok, reason).

    Pure inspection — no imports, no device. Used both to gate the swap and to
    explain (in the ledger) exactly why the kernel was or was not applied.
    """
    if not is_moe_arch(hf_config):
        return False, "not a MoE architecture (no num_experts / *Moe* arch)"
    mismatches = []
    for key, want in SUPPORTED_CONTRACT.items():
        if key == "tp_degree":
            got = tp_degree
        else:
            got = _cfg_get(hf_config, key)
        if got != want:
            mismatches.append(f"{key}={got}!={want}")
    if mismatches:
        return False, (
            "MoE model, but does not match the kernel's compiled contract "
            f"(Qwen3-30B-A3B@TP4): {', '.join(mismatches)}"
        )
    return True, "matches Qwen3-30B-A3B@TP4 contract"


def nkilib_available() -> bool:
    """True if the private `nki-library` (nkilib.core) dependency is importable.

    The fused-slab kernel imports `nkilib.core.utils.{allocator,tensor_view}`.
    Absent it, the kernel cannot even be traced, so we skip rather than crash.
    """
    try:
        import importlib.util as _u
        return _u.find_spec("nkilib") is not None
    except Exception:  # noqa: BLE001
        return False


def swap_moe_forward(model: Any, tp_degree: int, log=print) -> tuple[bool, str]:
    """Swap the HF Qwen3-MoE sparse-MoE block forward with the fused NKI kernel.

    Returns (swapped, reason). NEVER raises: any unmet precondition or import
    failure returns (False, reason) and leaves the model untouched, so the
    worker falls back to eager and the equivalence gate evaluates a correct,
    unchanged candidate (a graceful no-op).

    NOTE (honest status): the vendored `run(...)` entry is an XLA/torch_xla
    decode kernel expecting the A3B weight layout ([E,H,2,I] gate/up, [E,I,H]
    down) and the private nkilib runtime. Bridging HF's `Qwen3MoeSparseMoeBlock`
    (native-PyTorch, prefill, [B,S,H]) to it requires a weight re-pack + an
    XLA trace bridge that is NOT built here (it is the documented next step).
    So this function currently performs the full precondition gauntlet and, on
    the A3B/TP4 + nkilib path, reports that the execution bridge is pending —
    it does not fake a swap it cannot honestly perform.
    """
    try:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) or cfg
        ok, reason = precheck(text_cfg, tp_degree)
        if not ok:
            log(f"moe-borrow: skip ({reason}) -> eager fallback")
            return False, reason
        if not nkilib_available():
            log("moe-borrow: skip (nki-library / nkilib.core not importable) "
                "-> eager fallback")
            return False, "nkilib unavailable"
        # Preconditions all pass (A3B@TP4 + nkilib present). The remaining work
        # is the weight-repack + XLA trace bridge from HF's MoE block to
        # moe_fused_nki.run — deliberately not stubbed with fake output.
        log("moe-borrow: preconditions met (A3B@TP4 + nkilib); execution "
            "bridge to moe_fused_nki.run is the documented next step -> "
            "eager fallback for now")
        return False, "execution-bridge-pending (A3B@TP4 preconditions met)"
    except Exception as e:  # noqa: BLE001 — must never crash the worker
        log(f"moe-borrow: unexpected error ({e!r}) -> eager fallback")
        return False, f"error: {e!r}"
