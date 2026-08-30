"""The declarative shard plan must agree with the proven imperative sharding.

A wrong offset in a weight loader does not crash. It returns a model that produces
plausible garbage, and a benchmark then reports a number for it. So the plan in
`shard_stream.slice_for` is not trusted on inspection: it is checked against the
existing `qwen38_tp` + `moe_ep` sharding, on a real (tiny) Qwen3.5-MoE, parameter by
parameter, on every rank. If they ever diverge this test fails instead of a run
quietly producing nonsense.
"""

from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from backends.moe_ep import shard_moe_experts  # noqa: E402
from backends.qwen38_tp import shard_model  # noqa: E402
from backends.shard_stream import expert_range, kv_range, slice_for  # noqa: E402


CFG = dict(
    hidden_size=256, num_hidden_layers=2,
    num_attention_heads=16, num_key_value_heads=2, head_dim=16,
    intermediate_size=128, moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=32, num_experts_per_tok=8, vocab_size=256,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=16, linear_num_value_heads=32,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
)


def _model():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )
    cfg = Qwen3_5MoeTextConfig(**CFG)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(cfg).eval()


def _imperative(base, rank, world):
    m = copy.deepcopy(base)
    shard_moe_experts(m, rank, world)
    shard_model(m, rank, world)
    return dict(m.state_dict())


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_the_plan_matches_the_proven_sharding_for_every_covered_tensor(world):
    """Every tensor the plan claims to shard must come out byte-identical."""
    base = _model()
    full = dict(base.state_dict())
    for rank in range(world):
        got = _imperative(base, rank, world)
        for name, ref in full.items():
            sl = slice_for(name, CFG, rank, world, tuple(ref.shape))
            if sl is None:
                continue                      # replicated: covered by its own test
            if name not in got:
                continue                      # module was replaced wholesale
            expected = ref.narrow(sl.dim, sl.lo, sl.hi - sl.lo)
            actual = got[name]
            assert actual.shape == expected.shape, (world, rank, name,
                                                    actual.shape, expected.shape)
            assert torch.equal(actual, expected), (world, rank, name)


@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_replicated_tensors_really_are_unchanged(world):
    """The safe default has to actually be safe: replicate means identical."""
    base = _model()
    full = dict(base.state_dict())
    got = _imperative(base, 0, world)
    checked = 0
    for name, ref in full.items():
        if slice_for(name, CFG, 0, world, tuple(ref.shape)) is not None \
                or name not in got:
            continue
        if got[name].shape != ref.shape:
            continue        # legitimately reshaped by the imperative path
        assert torch.equal(got[name], ref), name
        checked += 1
    assert checked > 0


def test_expert_ranges_partition_every_expert_exactly_once():
    for world in (2, 3, 4, 8, 16, 32):
        seen = []
        for r in range(world):
            lo, hi = expert_range(32, r, world)
            seen += list(range(lo, hi))
        assert sorted(seen) == list(range(32)), world


def test_kv_ranges_replicate_rather_than_vanish_when_world_exceeds_kv_heads():
    """The #127 bug: 16 query heads, 2 KV heads, world=16 gave zero width."""
    for r in range(16):
        lo, hi = kv_range(16, 2, r, 16)
        assert hi > lo
        assert (lo, hi) == (r // 8, r // 8 + 1)


def test_kv_ranges_match_plain_division_when_world_divides_kv_heads():
    for world in (1, 2):
        for r in range(world):
            lo, hi = kv_range(16, 2, r, world)
            kpr = 2 // world
            assert (lo, hi) == (r * kpr, r * kpr + kpr), (world, r)


def test_the_router_is_never_sharded():
    """Routing is global; only expert WEIGHTS are local (moe_ep)."""
    assert slice_for("model.layers.0.mlp.gate.weight", CFG, 0, 8, (32, 256)) is None


def test_experts_are_sharded_on_the_leading_expert_dimension():
    sl = slice_for("model.layers.0.mlp.experts.gate_up_proj", CFG, 1, 4, (32, 64, 256))
    assert sl is not None and sl.dim == 0
    assert (sl.lo, sl.hi) == expert_range(32, 1, 4)


def test_world_one_shards_nothing():
    for n, sh in (("model.layers.0.mlp.experts.down_proj", (32, 256, 32)),
                  ("model.layers.1.self_attn.q_proj.weight", (512, 256))):
        assert slice_for(n, CFG, 0, 1, sh) is None


# --- end to end: read a real checkpoint and get the same shard ------------------

def test_streaming_a_real_checkpoint_reproduces_the_imperative_shard(tmp_path):
    """The whole point, proven against a real .safetensors file on disk.

    If the streamed slice ever differs from what the proven sharding produces, a
    benchmark would report a number for a corrupted model. So this compares the two
    tensor by tensor, on every rank.
    """
    from safetensors.torch import save_file
    from backends.shard_stream import stream_shard

    base = _model()
    full = {k: v.contiguous() for k, v in base.state_dict().items()}
    ckpt = tmp_path / "model.safetensors"
    save_file(full, str(ckpt))

    world = 8
    for rank in range(world):
        got = _imperative(base, rank, world)
        streamed = dict(stream_shard([ckpt], CFG, rank, world))
        assert streamed, "streamed nothing"
        compared = 0
        for name, ref in full.items():
            sl = slice_for(name, CFG, rank, world, tuple(ref.shape))
            if sl is None or name not in got:
                continue
            assert torch.equal(streamed[name], got[name]), (rank, name)
            compared += 1
        assert compared > 0, rank


def test_streaming_reads_only_the_slice_not_the_whole_tensor(tmp_path):
    """Sliced expert tensors must come back smaller than the checkpoint's."""
    from safetensors.torch import save_file
    from backends.shard_stream import stream_shard

    base = _model()
    full = {k: v.contiguous() for k, v in base.state_dict().items()}
    ckpt = tmp_path / "model.safetensors"
    save_file(full, str(ckpt))

    streamed = dict(stream_shard([ckpt], CFG, 0, 8))
    name = "model.layers.0.mlp.experts.gate_up_proj"
    assert streamed[name].shape[0] == 4          # 32 experts / 8 ranks
    assert full[name].shape[0] == 32
    total_full = sum(v.numel() for v in full.values())
    total_shard = sum(v.numel() for v in streamed.values())
    assert total_shard < total_full / 2, (total_shard, total_full)


def test_streaming_reports_what_it_replicated(tmp_path):
    from safetensors.torch import save_file
    from backends.shard_stream import stream_shard

    base = _model()
    ckpt = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in base.state_dict().items()}, str(ckpt))
    logs: list[str] = []
    list(stream_shard([ckpt], CFG, 0, 4, log=logs.append))
    assert logs and "sliced" in logs[0] and "replicated" in logs[0]


# --- coverage on the models that actually hit the host-DRAM wall ---------------

# Field subsets of the published configs, enough to exercise the rules.
REAL_122B = dict(hidden_size=3072, num_attention_heads=32, num_key_value_heads=2,
                 head_dim=256, num_experts=256, moe_intermediate_size=1024)
REAL_35B = dict(hidden_size=2048, num_attention_heads=16, num_key_value_heads=2,
                head_dim=256, num_experts=256, moe_intermediate_size=512)


@pytest.mark.parametrize("cfg,world,name", [
    (REAL_122B, 32, "122B"),
    (REAL_35B, 16, "35B"),
])
def test_the_expert_weights_are_sharded_for_the_real_moe_configs(cfg, world, name):
    """Experts are 231.9 GB of the 122B's 241.4 GB and 64.4 of the 35B's 68.5.

    If a change ever stops sharding them, the host-DRAM wall is back and the model
    stops loading -- so this is the single most important rule to pin.
    """
    for tensor in ("gate_up_proj", "down_proj"):
        n = f"model.layers.0.mlp.experts.{tensor}"
        sl = slice_for(n, cfg, 0, world, (cfg["num_experts"], 64, cfg["hidden_size"]))
        assert sl is not None, (name, tensor)
        assert sl.dim == 0
        assert sl.hi - sl.lo == cfg["num_experts"] // world, (name, tensor)


def test_attention_is_sharded_for_the_real_moe_configs():
    """Small in bytes, but an unsharded q/o_proj breaks the TP contract."""
    heads, hd = REAL_122B["num_attention_heads"], REAL_122B["head_dim"]
    q = slice_for("model.layers.1.self_attn.q_proj.weight", REAL_122B, 3, 32,
                  (heads * hd * 2, REAL_122B["hidden_size"]))
    assert q is not None and q.dim == 0
    assert q.hi - q.lo == hd * 2                    # one head, gated layout
    o = slice_for("model.layers.1.self_attn.o_proj.weight", REAL_122B, 3, 32,
                  (REAL_122B["hidden_size"], heads * hd))
    assert o is not None and o.dim == 1
    assert o.hi - o.lo == hd


def test_kv_heads_replicate_at_the_real_122b_geometry():
    """32 query heads, 2 KV heads, world=32: every rank must get one KV head."""
    hd = REAL_122B["head_dim"]
    seen = set()
    for r in range(32):
        sl = slice_for("model.layers.1.self_attn.k_proj.weight", REAL_122B, r, 32,
                       (2 * hd, REAL_122B["hidden_size"]))
        assert sl is not None and sl.hi - sl.lo == hd, r
        seen.add(sl.lo // hd)
    assert seen == {0, 1}                            # both KV heads used, none dropped


# ---------------------------------------------------------------------------
# Assembling a model from streamed slices must equal loading it whole and sharding.
#
# test above proves the SLICES match. This proves the ASSEMBLY: meta-instantiate,
# swap expert modules to rank-local, stream weights in -> the live model's parameters
# are bit-for-bit identical to full-load + shard_moe_experts. If they diverge, a run
# using shard-on-read would route tokens through the wrong expert weights and report a
# number for a model that is quietly wrong -- so this fails instead.
# ---------------------------------------------------------------------------
import safetensors.torch  # noqa: E402

from backends.shard_stream import load_ep_sharded  # noqa: E402


def _save_checkpoint(model, d):
    """Write the model as a single safetensors file, like a real checkpoint."""
    sd = {k: v.contiguous() for k, v in model.state_dict().items()
          if not v.is_meta}
    safetensors.torch.save_file(sd, str(d / "model.safetensors"))
    return [d / "model.safetensors"]


def _imperative_ep_only(base, rank, world):
    """Reference: full-load, shard ONLY the experts (attention replicated).

    shard-on-read replicates attention for this model, so the matching imperative
    configuration is expert-parallel with attention left whole -- not shard_model,
    which also TP-shards attention. Both are correct; this isolates what the assembly
    actually does.
    """
    m = copy.deepcopy(base)
    shard_moe_experts(m, rank, world)
    return dict(m.named_parameters())


@pytest.mark.parametrize("world", [2, 4, 8])
def test_assembled_model_matches_full_load_plus_expert_shard(tmp_path, world):
    base = _model()
    # A monkeypatch so load_ep_sharded builds the SAME tiny config, not a Hub fetch.
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    import transformers
    orig_from_pretrained = transformers.AutoConfig.from_pretrained
    cfg_obj = Qwen3_5MoeTextConfig(**CFG)
    cfg_obj._attn_implementation = "eager"
    transformers.AutoConfig.from_pretrained = staticmethod(
        lambda *a, **k: cfg_obj)
    try:
        files = _save_checkpoint(base, tmp_path)
        for rank in range(world):
            ref = _imperative_ep_only(base, rank, world)
            got_model = load_ep_sharded("tiny", files, CFG, rank, world)
            got = dict(got_model.named_parameters())
            # every reference parameter must be present and identical
            for name, expected in ref.items():
                assert name in got, (world, rank, "missing", name)
                a = got[name]
                assert a.shape == expected.shape, (world, rank, name,
                                                   a.shape, expected.shape)
                assert torch.equal(a, expected), (world, rank, name)
            assert set(got) == set(ref), (world, rank,
                                          set(got) ^ set(ref))
    finally:
        transformers.AutoConfig.from_pretrained = orig_from_pretrained


@pytest.mark.parametrize("world", [2, 4])
def test_assembled_expert_weights_are_rank_local_not_full(tmp_path, world):
    """The whole point: a rank must hold only its experts, not all of them."""
    base = _model()
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    import transformers
    orig = transformers.AutoConfig.from_pretrained
    cfg_obj = Qwen3_5MoeTextConfig(**CFG); cfg_obj._attn_implementation = "eager"
    transformers.AutoConfig.from_pretrained = staticmethod(lambda *a, **k: cfg_obj)
    try:
        files = _save_checkpoint(base, tmp_path)
        m = load_ep_sharded("tiny", files, CFG, 0, world)
        per_rank = CFG["num_experts"] // world
        for name, prm in m.named_parameters():
            if name.endswith("mlp.experts.gate_up_proj"):
                assert prm.shape[0] == per_rank, (name, prm.shape, per_rank)
    finally:
        transformers.AutoConfig.from_pretrained = orig
