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

    def __init__(self, orig: nn.Module, rank: int, world: int,
                 ep_size: int | None = None, ep_rank: int | None = None,
                 ep_group=None):
        super().__init__()
        num_experts = int(getattr(orig, "num_experts", orig.gate_up_proj.shape[0]))
        # EP degree may be SMALLER than world: experts then shard across `ep_size`
        # ranks and replicate across world/ep_size groups, with the mixture summed
        # over the EP subgroup only (self.ep_group). Default ep_size=world,
        # ep_rank=rank, ep_group=None -> reduce over the whole world, the original
        # coupled EP=TP behaviour, byte-identical.
        ep_size = world if ep_size is None else ep_size
        ep_rank = rank if ep_rank is None else ep_rank
        self.ep_group = ep_group
        lo, hi = expert_shard_plan(num_experts, ep_rank, ep_size)
        self.num_experts = num_experts          # global, for the one-hot
        self.lo, self.hi = lo, hi
        self.act_fn = orig.act_fn
        # .clone() so the full tensor can be freed; a view would keep it alive
        # and defeat the entire point of sharding.
        self.gate_up_proj = nn.Parameter(
            orig.gate_up_proj.data[lo:hi].clone(), requires_grad=False)
        self.down_proj = nn.Parameter(
            orig.down_proj.data[lo:hi].clone(), requires_grad=False)

    @classmethod
    def from_sliced(cls, *, num_experts, rank, world, act_fn, gate_up_proj, down_proj,
                    ep_size=None, ep_rank=None, ep_group=None):
        """Build directly from THIS RANK'S already-sliced expert weights.

        The ``__init__`` above slices ``orig.gate_up_proj[lo:hi]`` from a full,
        materialised expert tensor -- which is exactly the host-DRAM peak that
        shard-on-read exists to avoid. This path takes the weights already narrowed
        to ``[lo:hi]`` by ``shard_stream.stream_shard`` (read straight off disk), so
        the full expert set never exists in this process.

        ``lo``/``hi`` are recomputed from ``expert_shard_plan`` rather than inferred
        from the tensor, and asserted against its length, so a mismatch between the
        streamed slice and the expected range is caught here instead of producing a
        model that routes tokens to the wrong expert.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        ep_size = world if ep_size is None else ep_size
        ep_rank = rank if ep_rank is None else ep_rank
        self.ep_group = ep_group
        lo, hi = expert_shard_plan(num_experts, ep_rank, ep_size)
        if gate_up_proj.shape[0] != hi - lo or down_proj.shape[0] != hi - lo:
            raise ValueError(
                f"sliced expert count {gate_up_proj.shape[0]}/{down_proj.shape[0]} "
                f"!= expected {hi - lo} for rank {rank}/{world} of {num_experts}")
        self.num_experts = num_experts          # global, for the one-hot dispatch
        self.lo, self.hi = lo, hi
        self.act_fn = act_fn
        self.gate_up_proj = nn.Parameter(gate_up_proj, requires_grad=False)
        self.down_proj = nn.Parameter(down_proj, requires_grad=False)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # STATIC-SHAPE dense expert forward, BATCHED across this rank's local experts.
        # Mathematically identical to running each local expert separately and summing
        # by router affinity -- a non-routed token has affinity 0, so this equals the
        # top-k mixture (proven bit-for-bit vs the reference in test_moe_ep.py) -- but
        # the per-expert python loop is fused into two batched matmuls. That removes the
        # num_local separate op dispatches that dominate in eager mode and hands the
        # compiler one clean batched matmul instead of a chain of small F.linear calls.
        #
        # Still fully STATIC: E=num_local is fixed and the affinity is built by comparing
        # top_k_index against a fixed arange -- no nonzero()/where()/index_add()
        # data-dependent gather (the Neuron dynamic=False invariant, pinned by
        # test_expert_forward_has_no_data_dependent_shapes). Why not a truly routed
        # (gather only the tokens each expert was assigned) form: that is a
        # data-dependent shape, which the Neuron compiler rejects; the static routed
        # alternative (fixed per-expert capacity) drops overflow tokens and is no longer
        # bit-exact. Batching keeps exactness while cutting the eager dispatch overhead.
        E = self.hi - self.lo
        if E <= 0:                              # this rank owns no experts (ep>experts)
            final_hidden_states = torch.zeros_like(hidden_states)
        else:
            # gate_up_proj: [E, 2I, H]; down_proj: [E, H, I]; hidden_states: [T, H].
            # einsum('th,eoh->eto') == F.linear(x, gate_up_proj[e]) stacked over e.
            gu = torch.einsum("th,eoh->eto", hidden_states, self.gate_up_proj)  # [E,T,2I]
            gate, up = gu.chunk(2, dim=-1)                                      # [E,T,I]
            h = self.act_fn(gate) * up                                         # [E,T,I]
            h = torch.einsum("eti,ehi->eth", h, self.down_proj)                # [E,T,H]
            # affinity[e,t]: sum of the router weights on whichever top-k slots picked
            # global expert (lo+e), else 0. top_k_index/top_k_weights are [T, K].
            global_ids = torch.arange(self.lo, self.hi, device=top_k_index.device)
            match = top_k_index.unsqueeze(0) == global_ids.view(-1, 1, 1)      # [E,T,K]
            aff = (match * top_k_weights.unsqueeze(0)).sum(dim=-1)             # [E,T]
            final_hidden_states = torch.einsum(
                "eth,et->th", h, aff.to(h.dtype))                             # [T,H]
        # Sum the per-rank partial sums over the EP subgroup. Every expert contributed
        # on exactly one rank within the group, so this reconstructs the full mixture;
        # group=None (EP=world) reduces over the whole world, the coupled default.
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(final_hidden_states, group=getattr(self, "ep_group", None))
        return final_hidden_states


def _ep_layout(rank, world, ep_degree):
    """(ep_size, ep_rank, ep_group) for this rank given the requested EP degree.

    ep_degree is the number of ranks the expert set is split across; the remaining
    world/ep_degree ranks each hold a REPLICA of that split (they shard attention via
    TP instead). Ranks are grouped in contiguous blocks of ep_degree, so EP group g is
    [g*ep_degree : (g+1)*ep_degree] and the mixture is summed within it.

    ep_degree None / >=world / non-divisor reproduces the coupled EP=TP=world exactly:
    ep_group=None (the whole world), so the default path is byte-identical.
    """
    if not ep_degree or ep_degree >= world or ep_degree < 1 or world % ep_degree != 0:
        return world, rank, None
    ep_rank = rank % ep_degree
    group = None
    try:
        import torch.distributed as _d
        if _d.is_initialized():
            # new_group is collective: EVERY rank must create EVERY group, even ones
            # it does not join, or the group it does join will hang.
            for g in range(world // ep_degree):
                members = list(range(g * ep_degree, (g + 1) * ep_degree))
                grp = _d.new_group(ranks=members)
                if rank in members:
                    group = grp
    except Exception:  # noqa: BLE001 - fall back to world reduce if grouping fails
        return world, rank, None
    return ep_degree, ep_rank, group


def shard_moe_experts(model: nn.Module, rank: int, world: int,
                      ep_degree=None, ep_plan=None) -> tuple[int, float]:
    """Replace every fused-expert module in ``model`` with a rank-local shard.

    Returns ``(layers_sharded, gb_freed_per_rank)``. A no-op at ``world <= 1``.

    ``ep_degree`` decouples expert parallelism from tensor parallelism: experts shard
    across ``ep_degree`` ranks (replicated across world/ep_degree), while attention
    still shards across the full ``world``. Default (None or ==world) is the original
    coupled EP=TP behaviour.
    """
    if world <= 1:
        return 0, 0.0
    if ep_plan is not None:
        # explicit layout from the orthogonal 2-D mesh (moe_mesh): strided EP column
        # + that column's process group. Overrides the contiguous Design-B layout.
        ep_size, ep_rank, ep_group = ep_plan
    else:
        ep_size, ep_rank, ep_group = _ep_layout(rank, world, ep_degree)
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
        mlp.experts = ExpertParallelExperts(experts, rank, world,
                                            ep_size=ep_size, ep_rank=ep_rank,
                                            ep_group=ep_group)
        after = (mlp.experts.gate_up_proj.numel() + mlp.experts.down_proj.numel()) \
            * mlp.experts.gate_up_proj.element_size()
        freed += (before - after) / 1e9
        n += 1
    return n, freed
