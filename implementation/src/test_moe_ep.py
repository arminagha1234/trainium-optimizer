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


# ---------------------------------------------------------------------------
# EP decoupled from TP: experts shard across ep_degree < world ranks, replicated
# across world/ep_degree groups, and the mixture is summed WITHIN each EP group.
# Same simulation contract as _sharded_sum above: summing a group's partials is
# exactly what dist.all_reduce(group=ep_group) computes. If the group partition ever
# dropped or double-counted an expert, the sum would diverge from the unsharded
# reference -- so this fails instead of a decoupled run reporting a wrong number.
# ---------------------------------------------------------------------------

def _ep_group_sum(ref, world, ep_degree, hs, idx, w, group_id=0):
    """Sum the partials of one EP group's ranks -- the scoped all_reduce's result."""
    out = None
    for ep_rank in range(ep_degree):
        r = group_id * ep_degree + ep_rank
        shard = ExpertParallelExperts(ref, r, world, ep_size=ep_degree,
                                      ep_rank=r % ep_degree, ep_group=None)
        o = shard(hs, idx, w)
        out = o if out is None else out + o
    return out


def test_ep_group_partition_covers_every_expert_exactly_once():
    """Each EP group, on its own, must own every expert exactly once."""
    from backends.moe_ep import expert_shard_plan
    num_experts = 16
    for world in (8, 16):
        for ep_degree in (1, 2, 4, 8):
            if world % ep_degree:
                continue
            for group_id in range(world // ep_degree):
                seen = []
                for ep_rank in range(ep_degree):
                    lo, hi = expert_shard_plan(num_experts, ep_rank, ep_degree)
                    seen += list(range(lo, hi))
                assert sorted(seen) == list(range(num_experts)), (world, ep_degree)


def test_ep_decoupled_group_sum_equals_unsharded_fp32():
    torch.manual_seed(0)
    ref = RefExperts(16, 32, 24).eval()
    hs = torch.randn(12, 32)
    idx, w = _routing(12, 16, top_k=4)
    expected = ref(hs, idx, w)
    # world=8 cores; try every EP split that divides it.
    for ep_degree in (1, 2, 4, 8):
        got = _ep_group_sum(ref, 8, ep_degree, hs, idx, w)
        assert torch.allclose(got, expected, atol=1e-6), \
            f"ep_degree={ep_degree} diverged from unsharded"


def test_ep_decoupled_uneven_experts_still_exact():
    """Experts not divisible by ep_degree must still partition exactly."""
    torch.manual_seed(1)
    ref = RefExperts(13, 32, 24).eval()
    hs = torch.randn(10, 32)
    idx, w = _routing(10, 13, top_k=3)
    expected = ref(hs, idx, w)
    for ep_degree in (2, 4):          # 13 % 2, 13 % 4 both nonzero
        got = _ep_group_sum(ref, 8, ep_degree, hs, idx, w)
        assert torch.allclose(got, expected, atol=1e-6), ep_degree


def test_every_ep_group_computes_the_same_mixture():
    """Replicas across groups are redundant, not divergent -- each group's sum is
    the full mixture, so different groups agree."""
    torch.manual_seed(2)
    ref = RefExperts(16, 32, 24).eval()
    hs = torch.randn(8, 32)
    idx, w = _routing(8, 16, top_k=4)
    g0 = _ep_group_sum(ref, 8, 4, hs, idx, w, group_id=0)
    g1 = _ep_group_sum(ref, 8, 4, hs, idx, w, group_id=1)
    assert torch.allclose(g0, g1, atol=1e-6)


def test_ep_degree_none_or_world_is_the_coupled_default():
    """ep_degree None / ==world must reproduce the original coupled shards exactly,
    so turning the knob off is byte-identical to before."""
    torch.manual_seed(3)
    ref = RefExperts(16, 32, 24).eval()
    for world in (2, 4, 8):
        for r in range(world):
            base = ExpertParallelExperts(ref, r, world)
            for ep in (None, world):
                dec = ExpertParallelExperts(ref, r, world, ep_size=(ep or world),
                                            ep_rank=r, ep_group=None)
                assert torch.equal(base.gate_up_proj, dec.gate_up_proj), (world, r, ep)
                assert (base.lo, base.hi) == (dec.lo, dec.hi)
                assert dec.ep_group is None


def test_expert_forward_has_no_data_dependent_shapes():
    """The Neuron invariant: the expert forward must not gather by nonzero()/where().

    Those produce data-dependent shapes, which the Neuron compiler (dynamic=False)
    rejects -- so a regression to the gather-routed-tokens form would recompile/stall
    per token distribution on device even though it passes the numeric tests on CPU.
    Pin the static-shape form here so that regression is caught in CI, not on a box.
    """
    import inspect
    from backends.moe_ep import ExpertParallelExperts
    src = inspect.getsource(ExpertParallelExperts.forward)
    assert ".nonzero(" not in src, "expert forward regressed to a nonzero() gather"
    assert "torch.where(" not in src, "expert forward regressed to a where() gather"
    assert "index_add_" not in src, "expert forward regressed to a scatter-add gather"


def test_dense_forward_matches_a_hand_computed_case():
    """A tiny explicit case, so equivalence is anchored to arithmetic, not only to
    the reference module (which could share a bug)."""
    torch.manual_seed(0)
    ref = RefExperts(4, 8, 8).eval()
    hs = torch.randn(3, 8)
    # token 0 -> experts [0,1], token 1 -> [2,3], token 2 -> [1,2]
    idx = torch.tensor([[0, 1], [2, 3], [1, 2]])
    w = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
    got = ExpertParallelExperts(ref, 0, 1)(hs, idx, w)   # world=1: all experts local
    # hand form: sum_j w[t,j] * expert_{idx[t,j]}(hs[t])
    def expert(e, x):
        g, u = F.linear(x, ref.gate_up_proj[e]).chunk(2, dim=-1)
        return F.linear(ref.act_fn(g) * u, ref.down_proj[e])
    want = torch.zeros_like(hs)
    for t in range(3):
        for j in range(2):
            want[t] += w[t, j] * expert(int(idx[t, j]), hs[t:t+1])[0]
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()
