#!/usr/bin/env python
"""Compile-enabling monkeypatches for transformers-native DeepSeek-V4 MoE.

Runbook phase P4. `torch.compile(backend="neuron", dynamic=False)` fails on the
native MoE because both the learned router's `torch.topk` AND the grouped_mm
expert backend's internal sort lower to `AwsNeuronTopK`, which the Neuron
compiler rejects (confirmed on torch_neuronx 2.12.3: "COMPILATION FAILED ...
custom_call_target=AwsNeuronTopK" inside DeepseekV4Experts). These patches
replace those two paths with compile-friendly, static-shape equivalents:

  * iter_argmax_router_forward - top-k via iterative argmax (verified to select
    the same experts as torch.topk).
  * stage_a_experts_forward - dense experts, unrolled over a compile-time-constant
    expert count (Stage A). Numerically equivalent to the grouped_mm experts
    (cos 0.999963). Fine for the shrunk config (8 experts); the real 256-expert
    model needs Stage B fixed-capacity dispatch instead (the unrolled graph would
    be too large to partition), or runs the native grouped_mm eagerly.
"""
import torch
import torch.nn.functional as F


def stage_a_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Dense MoE experts, unrolled over E (compile-time constant). No grouped_mm
    top-k, no data-dependent shapes, no token dropping (=> drop count is zero).

    NOTE (P8): the accumulator is BF16 here (matches what was validated at P4).
    Runbook 4.6 recommends FP32 MoE accumulation for real-weight accuracy; switch
    `final` to float32 and cast back at P8, re-validating the compile.
    """
    T, H = hidden_states.shape
    w = top_k_weights.to(hidden_states.dtype)
    final = torch.zeros(T, H, dtype=hidden_states.dtype, device=hidden_states.device)
    for e in range(self.num_experts):                               # E compile-time const
        we = (w * (top_k_index == e)).sum(dim=1, keepdim=True)      # [T,1]
        ge = self._apply_gate(F.linear(hidden_states, self.gate_up_proj[e]))  # [T,I]
        oe = F.linear(ge, self.down_proj[e])                        # [T,H]
        final = final + we * oe
    return final


def iter_argmax_router_forward(self, hidden_states):
    """Learned top-k router via iterative argmax instead of torch.topk (which
    lowers to AwsNeuronTopK and fails to compile). Selection matches torch.topk."""
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight)
    scores = self.score_fn(logits)
    biased = scores + self.e_score_correction_bias
    neg = torch.finfo(biased.dtype).min
    masked = biased
    picks = []
    for _ in range(self.top_k):                                     # top_k compile-time const
        i = masked.argmax(dim=-1, keepdim=True)                     # [T,1]
        picks.append(i)
        masked = masked.scatter(1, i, neg)                          # functional mask
    indices = torch.cat(picks, dim=1)                               # [T,k]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * self.routed_scaling_factor, indices


def apply_compile_patches(modeling_module=None):
    """Monkeypatch DeepseekV4Experts + DeepseekV4TopKRouter for torch.compile.

    Returns the modeling module. Idempotent. HashRouter needs no patch (it uses a
    tid2eid gather, not top-k).
    """
    if modeling_module is None:
        from transformers.models.deepseek_v4 import modeling_deepseek_v4 as modeling_module
    modeling_module.DeepseekV4Experts.forward = stage_a_experts_forward
    modeling_module.DeepseekV4TopKRouter.forward = iter_argmax_router_forward
    return modeling_module
