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
    top-k, no data-dependent shapes, no token dropping (=> drop count is zero),
    and no torch.where gather (which deadlocks the Neuron runtime at 43L).

    Precision matches the checkpoint's inference/model.py Expert/MoE: gate/up and
    the SiLU run in FP32, and the cross-expert accumulator is FP32 (`y` is
    float32 in the reference). Verified structurally against inference/model.py.
    """
    T, H = hidden_states.shape
    limit = self.limit
    w = top_k_weights.float()
    final = torch.zeros(T, H, dtype=torch.float32, device=hidden_states.device)   # FP32 accum (ref)
    for e in range(self.num_experts):                               # E compile-time const
        we = (w * (top_k_index == e)).sum(dim=1, keepdim=True)      # [T,1] fp32
        gu = F.linear(hidden_states, self.gate_up_proj[e]).float()  # [T,2I] -> fp32
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)                                # swiglu_limit (routed only)
        up = up.clamp(min=-limit, max=limit)
        inter = F.silu(gate) * up                                   # fp32 SiLU (ref)
        oe = F.linear(inter.to(hidden_states.dtype), self.down_proj[e]).float()
        final = final + we * oe
    return final.to(hidden_states.dtype)


def iter_argmax_router_forward(self, hidden_states):
    """Learned top-k router via iterative argmax instead of torch.topk (which
    lowers to AwsNeuronTopK and fails to compile). Selection matches torch.topk."""
    flat = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(flat.float(), self.weight.float())            # FP32 routing (ref)
    scores = self.score_fn(logits)                                  # sqrtsoftplus in fp32
    biased = scores + self.e_score_correction_bias.float()
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
    dt = hidden_states.dtype
    return logits.to(dt), (weights * self.routed_scaling_factor).to(dt), indices


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
