"""Expert-parallel sharding must be NUMERICALLY EXACT, not merely smaller.

There is no on-device reference to check against -- these models cannot load
unsharded, which is the whole problem -- so equivalence is established here on
CPU against the unsharded computation, with no Trainium and no torch.distributed.

The all-reduce is simulated by summing each rank's partial output, which is
exactly what dist.all_reduce computes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backends.moe_ep import ExpertParallelExperts, expert_shard_plan, shard_moe_experts


class RefExperts(nn.Module):
    """The reference: transformers' fused-expert forward, verbatim in structure."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = nn.SiLU()
        g = torch.randn(num_experts, 2 * inter, hidden) * 0.02
        d = torch.randn(num_experts, hidden, inter) * 0.02
        self.gate_up_proj = nn.Parameter(g, requires_grad=False)
        self.down_proj = nn.Parameter(d, requires_grad=False)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for ei in hit:
            e = int(ei[0])
            if e == self.num_experts:
                continue
            pos, tok = torch.where(mask[e])
            cur = hidden_states[tok]
            gate, up = F.linear(cur, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, self.down_proj[e])
            h = h * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, h.to(final.dtype))
        return final


def _routing(tokens: int, num_experts: int, top_k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, num_experts, generator=g)
    w, idx = torch.topk(F.softmax(logits, dim=-1), top_k, dim=-1)
    return idx, w


def _sharded_sum(ref: RefExperts, world: int, hs, idx, w):
    """Run every rank's shard and sum -- what all_reduce would produce."""
    out = None
    for r in range(world):
        shard = ExpertParallelExperts(ref, r, world)   # no dist -> no all_reduce
        o = shard(hs, idx, w)
        out = o if out is None else out + o
    return out


# --- partition correctness ----------------------------------------------------

def test_partition_covers_every_expert_exactly_once():
    for num_experts in (256, 128, 64, 10, 7):
        for world in (1, 2, 4, 8, 16, 3, 5):
            seen: list[int] = []
            for r in range(world):
                lo, hi = expert_shard_plan(num_experts, r, world)
                seen.extend(range(lo, hi))
            assert sorted(seen) == list(range(num_experts)), (num_experts, world)


def test_partition_is_balanced_within_one():
    lo_hi = [expert_shard_plan(256, r, 16) for r in range(16)]
    sizes = {hi - lo for lo, hi in lo_hi}
    assert sizes == {16}
    sizes = {hi - lo for lo, hi in (expert_shard_plan(10, r, 4) for r in range(4))}
    assert max(sizes) - min(sizes) <= 1


# --- the claim that matters ---------------------------------------------------

def test_sharded_equals_unsharded_fp32():
    torch.manual_seed(0)
    ref = RefExperts(num_experts=16, hidden=32, inter=16)
    hs = torch.randn(24, 32)
    idx, w = _routing(24, 16, top_k=2)
    expected = ref(hs, idx, w)
    for world in (2, 4, 8, 16):
        got = _sharded_sum(ref, world, hs, idx, w)
        assert torch.allclose(got, expected, atol=1e-6), f"world={world}"


def test_sharded_equals_unsharded_uneven_partition():
    """Experts not divisible by world size must still be exact."""
    torch.manual_seed(1)
    ref = RefExperts(num_experts=10, hidden=16, inter=8)
    hs = torch.randn(12, 16)
    idx, w = _routing(12, 10, top_k=3, seed=1)
    expected = ref(hs, idx, w)
    for world in (3, 4, 7):
        got = _sharded_sum(ref, world, hs, idx, w)
        assert torch.allclose(got, expected, atol=1e-6), f"world={world}"


def test_sharded_equals_unsharded_bf16():
    torch.manual_seed(2)
    ref = RefExperts(num_experts=8, hidden=32, inter=16).to(torch.bfloat16)
    hs = torch.randn(16, 32).to(torch.bfloat16)
    idx, w = _routing(16, 8, top_k=2, seed=2)
    w = w.to(torch.bfloat16)
    expected = ref(hs, idx, w)
    got = _sharded_sum(ref, 4, hs, idx, w)
    assert torch.allclose(got.float(), expected.float(), atol=2e-2)


def test_top_k_1_and_all_experts_hit():
    torch.manual_seed(3)
    ref = RefExperts(num_experts=8, hidden=16, inter=8)
    hs = torch.randn(32, 16)
    for top_k in (1, 8):
        idx, w = _routing(32, 8, top_k=top_k, seed=top_k)
        assert torch.allclose(_sharded_sum(ref, 4, hs, idx, w),
                              ref(hs, idx, w), atol=1e-6), top_k


# --- memory is actually reduced ------------------------------------------------

def test_each_rank_holds_only_its_slice():
    ref = RefExperts(num_experts=16, hidden=32, inter=16)
    full = ref.gate_up_proj.numel() + ref.down_proj.numel()
    shard = ExpertParallelExperts(ref, 0, 4)
    part = shard.gate_up_proj.numel() + shard.down_proj.numel()
    assert part * 4 == full
    assert shard.num_experts == 16          # routing stays global
    assert (shard.lo, shard.hi) == (0, 4)


def test_shard_is_a_copy_not_a_view():
    """A view keeps the full tensor alive and saves nothing."""
    ref = RefExperts(num_experts=8, hidden=16, inter=8)
    shard = ExpertParallelExperts(ref, 1, 2)
    assert shard.gate_up_proj.data_ptr() != ref.gate_up_proj.data_ptr()
    assert shard.gate_up_proj._base is None


# --- model-level wiring --------------------------------------------------------

class _Blk(nn.Module):
    def __init__(self, e, h, i):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = RefExperts(e, h, i)


class _Model(nn.Module):
    def __init__(self, layers=3, e=8, h=16, i=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Blk(e, h, i) for _ in range(layers)])


def test_shard_moe_experts_replaces_every_layer_and_reports_savings():
    m = _Model(layers=3)
    n, freed = shard_moe_experts(m, rank=0, world=4)
    assert n == 3
    assert freed > 0
    for layer in m.model.layers:
        assert isinstance(layer.mlp.experts, ExpertParallelExperts)


def test_world_one_is_a_noop():
    m = _Model(layers=2)
    n, freed = shard_moe_experts(m, rank=0, world=1)
    assert (n, freed) == (0, 0.0)
    assert isinstance(m.model.layers[0].mlp.experts, RefExperts)


def test_dense_layers_are_left_alone():
    m = _Model(layers=1)
    m.model.layers[0].mlp = nn.Module()          # no .experts
    m.model.layers[0].mlp.gate_proj = nn.Linear(4, 4)
    n, _ = shard_moe_experts(m, rank=0, world=4)
    assert n == 0
