"""Expert parallelism for fused-expert MoE blocks (Qwen3.5/3.6 MoE family).

Why this exists
---------------
The native TP path shards attention and the *dense* MLP, but `shard_model` only
touches `L.mlp` when it has `gate_proj`. A `Qwen3_5MoeSparseMoeBlock` has
`gate`/`experts`/`shared_expert` instead, so MoE layers were skipped entirely and
**every rank held the full expert set**.

That is why no amount of TP ever helped. Measured on a trn2.48xlarge (24 GB/core):

    Qwen3.5-35B-A3B     experts  64.4 GB/rank   (2.7x a core)
    Qwen3.5-122B-A10B   experts 231.9 GB/rank   (9.7x a core)

Both died at ~22 GB allocated despite a 3.5x size difference -- the signature of
filling the core while loading. Raising tp divided a ~0.2 GB dense term and left
the expert term untouched.

Why this is exact, not an approximation
---------------------------------------
`Qwen3_5MoeExperts.forward` loops over the experts that were *hit* and
`index_add_`s each expert's contribution into a zero-initialised buffer. Every
expert therefore contributes to exactly one additive term. Partition the experts
across ranks, have each rank skip the ones it does not own, and sum the buffers
with an all-reduce: the result is bit-comparable to the unsharded computation up
to floating-point summation order.

The shared expert needs no special handling. It is applied in the *block*
forward, after `self.experts(...)`, so it stays local to each rank and is added
once per rank AFTER the all-reduce inside this module -- not multiplied by the
world size. Patching only the experts module is what keeps that true.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ExpertParallelExperts", "shard_moe_experts", "expert_shard_plan"]


def expert_shard_plan(num_experts: int, rank: int, world: int) -> tuple[int, int]:
    """Half-open [lo, hi) expert range owned by ``rank``.

    Remainder experts are spread over the low ranks so the partition always
    covers every expert exactly once -- dropping or double-counting one would
    silently corrupt the output rather than fail loudly.
    """
    if world <= 1:
        return 0, num_experts
    base, extra = divmod(num_experts, world)
    lo = rank * base + min(rank, extra)
    hi = lo + base + (1 if rank < extra else 0)
    return lo, hi


class ExpertParallelExperts(nn.Module):
    """Drop-in for a fused-expert module holding only this rank's experts.

    Keeps ``num_experts`` at the GLOBAL count: routing is global, and the
    one-hot dispatch must be built over all experts so token->expert assignment
    is identical on every rank. Only the *weights* are local.
    """

    def __init__(self, orig: nn.Module, rank: int, world: int):
        super().__init__()
        num_experts = int(getattr(orig, "num_experts", orig.gate_up_proj.shape[0]))
        lo, hi = expert_shard_plan(num_experts, rank, world)
        self.num_experts = num_experts          # global, for the one-hot
        self.lo, self.hi = lo, hi
        self.act_fn = orig.act_fn
        # .clone() so the full tensor can be freed; a view would keep it alive
        # and defeat the entire point of sharding.
        self.gate_up_proj = nn.Parameter(
            orig.gate_up_proj.data[lo:hi].clone(), requires_grad=False)
        self.down_proj = nn.Parameter(
            orig.down_proj.data[lo:hi].clone(), requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            if e < self.lo or e >= self.hi:
                continue                        # owned by another rank
            local = e - self.lo
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[local]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, self.down_proj[local])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        # Sum the per-rank partial sums. Every expert contributed on exactly one
        # rank, so this reconstructs the full mixture.
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(final_hidden_states)
        return final_hidden_states


def shard_moe_experts(model: nn.Module, rank: int, world: int) -> tuple[int, float]:
    """Replace every fused-expert module in ``model`` with a rank-local shard.

    Returns ``(layers_sharded, gb_freed_per_rank)``. A no-op at ``world <= 1``.
    """
    if world <= 1:
        return 0, 0.0
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:  # multimodal wrapper
        lm = getattr(root, "language_model", None)
        layers = getattr(lm, "layers", []) if lm is not None else []
    n, freed = 0, 0.0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        before = (experts.gate_up_proj.numel() + experts.down_proj.numel()) \
            * experts.gate_up_proj.element_size()
        mlp.experts = ExpertParallelExperts(experts, rank, world)
        after = (mlp.experts.gate_up_proj.numel() + mlp.experts.down_proj.numel()) \
            * mlp.experts.gate_up_proj.element_size()
        freed += (before - after) / 1e9
        n += 1
    return n, freed
