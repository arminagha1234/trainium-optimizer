# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Neuron-safe MoE router patch — unblocks the MoE *baseline* on the native-PyTorch
backend.

THE CRASH (captured on-device, allenai/OLMoE-1B-7B baseline):

    [NCC_EVRF013] TopK custom operation does not support 32-bit and 64-bit
      integer types.
      %custom-call = (s64[4096], s64[4096]) custom-call(...),
        custom_call_target="AwsNeuronTopK",
        op_name="OlmoeModel/OlmoeDecoderLayer/OlmoeSparseMoeBlock[mlp]/OlmoeExperts"
        source="transformers/integrations/moe.py:393"

HF's generic MoE expert path (`transformers/integrations/moe.py`) groups the
routed token/expert pairs by expert id before the grouped matmul:

    expert_ids = top_k_index.reshape(-1)      # int64, S = num_tokens * top_k
    expert_ids_g, perm = torch.sort(expert_ids)   # <-- moe.py:393

`torch.sort` (like `torch.topk`) lowers on Neuron to the `AwsNeuronTopK`
custom-op, which REJECTS 32-bit and 64-bit *integer value* inputs — hence the
`(s64[4096], s64[4096])` signature in the abort. Older transformers releases
hit the same op one line earlier, in the router's `torch.topk(router_logits,
k)` selection. Either way the crash is: an INTEGER-typed tensor reaches
AwsNeuronTopK. Every HF MoE (OLMoE / Qwen2-MoE / DeepSeek-MoE routes through
this same module) dies here, which is why they all scored 0.000.

THE FIX (approach (a): a dtype-safe router top-k, the minimal unblock):

We wrap `torch.topk` / `torch.sort` / `torch.argsort` so that when the tensor
being sorted is an INTEGER dtype, the op runs on a float32 view and the sorted
*values* are cast back to the original integer dtype. The permutation / index
output is unchanged. This never touches float inputs (router logits, attention
scores, …) — they pass straight through unmodified — so it is a no-op for every
non-integer sort/topk in the model. Routing-scale expert ids (0 .. num_experts,
always < 2^24) are represented EXACTLY in float32, so the sort order and the
returned permutation are bit-identical to the int path: the fix is
correctness-preserving, not an approximation.

Why this over routing through the borrowed `moe_fused` kernel (approach (b)):
the vendored fused-MoE megakernel's `swap_moe_forward` is, by its own honest
docstring, an execution-bridge-pending no-op today (it self-skips on every model
except an exact Qwen3-30B-A3B@TP4 + nkilib deployment and even there returns
`(False, "execution-bridge-pending")`). It therefore cannot unblock the OLMoE /
Qwen1.5-MoE / DeepSeek-MoE *baselines*, which is the goal here. This patch is
~40 lines, version-independent (it keys off the tensor dtype, not moe.py line
numbers or internal function names), and leaves the fused kernel free to be the
Stage-3 optimization on top of a now-working baseline. It mirrors the fused
kernel's own principle — a sort-free / dtype-safe top-k that avoids the int64
AwsNeuronTopK.

Import-safe: no torch at import time. Installed explicitly (and only) for
MoE-family models at model-load in `neuron_worker.py`; dense models never call
it and are untouched.
"""

from __future__ import annotations

from typing import Any, Callable

# The integer dtypes AwsNeuronTopK rejects. We route these through float32.
# (Populated lazily so the module imports with no torch present.)
_INT_DTYPE_NAMES = ("int8", "int16", "int32", "int64", "uint8")

_INSTALLED = False


def _int_dtypes(torch) -> tuple:
    return tuple(
        getattr(torch, n) for n in _INT_DTYPE_NAMES if hasattr(torch, n)
    )


def _is_int_tensor(torch, x: Any) -> bool:
    return isinstance(x, torch.Tensor) and x.dtype in _int_dtypes(torch)


def _wrap_values_and_indices(torch, fn: Callable) -> Callable:
    """Wrap topk/sort: (values, indices) namedtuple return. On an integer input
    tensor, sort a float32 view and cast the sorted VALUES back to the original
    integer dtype; the index/permutation output is returned unchanged."""

    def wrapper(input, *args, **kwargs):  # noqa: A002 — mirror torch's kw name
        if _is_int_tensor(torch, input):
            orig_dtype = input.dtype
            out = fn(input.to(torch.float32), *args, **kwargs)
            # torch.return_types.{topk,sort} is a structseq: (values, indices).
            values = out[0].to(orig_dtype)
            indices = out[1]
            try:  # preserve the exact return_types.* structseq if we can
                return type(out)([values, indices])
            except Exception:  # noqa: BLE001 — fall back to a plain tuple
                return (values, indices)
        return fn(input, *args, **kwargs)

    return wrapper


def _wrap_indices_only(torch, fn: Callable) -> Callable:
    """Wrap argsort: returns only the index tensor. On an integer input, sort a
    float32 view (order-preserving for exact ints); indices returned as-is."""

    def wrapper(input, *args, **kwargs):  # noqa: A002
        if _is_int_tensor(torch, input):
            return fn(input.to(torch.float32), *args, **kwargs)
        return fn(input, *args, **kwargs)

    return wrapper


def install_neuron_safe_moe_topk(log: Callable[[str], None] = print) -> bool:
    """Make `torch.topk` / `torch.sort` / `torch.argsort` Neuron-safe for
    integer inputs, so HF's MoE expert-grouping (`torch.sort(expert_ids)`) and
    router selection (`torch.topk(router_logits, k)`) no longer feed an integer
    tensor to AwsNeuronTopK.

    Idempotent and process-scoped. The neuron_worker is a single-model,
    single-measurement process that hard-exits, so a process-global wrap is safe
    and self-contained. Returns True if it installed the patch, False if it was
    already installed (or torch is unavailable). NEVER raises — a failure to
    patch degrades to the unpatched path (which the caller can still attempt).
    """
    global _INSTALLED
    if _INSTALLED:
        return False
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        log(f"moe-router-patch: torch unavailable ({e!r}); not installed")
        return False
    try:
        torch.topk = _wrap_values_and_indices(torch, torch.topk)
        torch.sort = _wrap_values_and_indices(torch, torch.sort)
        torch.argsort = _wrap_indices_only(torch, torch.argsort)
        _INSTALLED = True
        log("moe-router-patch: installed dtype-safe torch.{topk,sort,argsort} "
            "(integer inputs routed through float32 -> avoids the int64 "
            "AwsNeuronTopK crash in transformers MoE routing)")
        return True
    except Exception as e:  # noqa: BLE001 — must never crash the worker
        log(f"moe-router-patch: install failed ({e!r}); running unpatched")
        return False
