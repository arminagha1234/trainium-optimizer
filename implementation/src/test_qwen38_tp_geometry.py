"""TP sharding geometry tests -- pinned to the REAL HuggingFace modules.

Why this file exists
--------------------
``test_moe_ep.py`` validated expert parallelism against ``RefExperts``, a local
reimplementation of what the HF fused-expert module was believed to do. It was
faithful, so those tests were sound -- but they could not have caught the bug
that actually killed the run, because the bug was in a DIFFERENT module
(attention) and only appears at a head geometry the reimplementation did not
model.

Qwen3.5-35B-A3B has 16 query heads and 2 KV heads. Expert memory forces tp=16.
``nkv // tp`` is then 0, so ``k_proj``/``v_proj`` were sliced to zero width.
Nothing raised at shard time; the model died six minutes later inside attention.

So these tests build a real (tiny) ``Qwen3_5MoeForCausalLM``, shard it with the
real code path, and run it. Geometry is kept at the real head counts -- only
depth and width are shrunk -- because the head counts are the thing under test.
"""

from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip(
    "transformers", reason="real-module geometry tests need transformers installed")

from backends.moe_ep import shard_moe_experts  # noqa: E402
from backends.qwen38_tp import _slice_linear, shard_attention, shard_model  # noqa: E402


def _tiny_qwen35(layers=("linear_attention", "full_attention"), q_heads=16, kv_heads=2):
    """Real Qwen3.5-MoE at real head counts, shrunk in depth and width."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )

    cfg = Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=len(layers),
        num_attention_heads=q_heads, num_key_value_heads=kv_heads, head_dim=16,
        intermediate_size=128, moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=32, num_experts_per_tok=8, vocab_size=256,
        layer_types=list(layers),
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    )
    cfg._attn_implementation = "eager"   # the impl that surfaces bad KV shapes
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(cfg).eval()


# --- the bug that cost a trn2.48xlarge run -----------------------------------

def test_kv_heads_are_never_sliced_to_zero_width():
    """tp > num_key_value_heads must replicate a KV head, not produce none.

    This is the exact Qwen3.5-35B-A3B geometry: 16 q heads, 2 KV heads, tp=16.
    """
    model = _tiny_qwen35()
    attn = model.model.layers[1].self_attn
    hd = attn.head_dim
    shard_attention(attn, r=7, tp=16)
    assert attn.k_proj.weight.shape[0] == hd, attn.k_proj.weight.shape
    assert attn.v_proj.weight.shape[0] == hd, attn.v_proj.weight.shape
    # 1 local q head over 1 local KV head.
    assert attn.q_proj.weight.shape[0] == hd * 2
    assert attn.num_key_value_groups == 1


def test_each_rank_gets_the_kv_head_serving_its_query_heads():
    """Ranks 0-7 own q heads 0-7, all served by KV head 0; ranks 8-15 by KV head 1."""
    hd = 16
    for r in range(16):
        model = _tiny_qwen35()
        attn = model.model.layers[1].self_attn
        full_k = attn.k_proj.weight.data.clone()
        shard_attention(attn, r=r, tp=16)
        expected_head = r // 8
        expected = full_k[expected_head * hd:(expected_head + 1) * hd]
        assert torch.equal(attn.k_proj.weight.data, expected), f"rank {r}"


def test_kv_slicing_is_unchanged_when_tp_divides_the_kv_heads():
    """The replication path must not perturb the case that already worked."""
    hd = 16
    for tp in (1, 2):
        for r in range(tp):
            model = _tiny_qwen35()
            attn = model.model.layers[1].self_attn
            full_k = attn.k_proj.weight.data.clone()
            shard_attention(attn, r=r, tp=tp)
            kpr = 2 // tp
            expected = full_k[r * kpr * hd:(r * kpr + kpr) * hd]
            assert torch.equal(attn.k_proj.weight.data, expected), (tp, r)


@pytest.mark.parametrize("tp", [2, 4, 8, 16])
def test_every_rank_runs_a_forward_after_sharding(tp):
    """The end-to-end contract: shard rank-by-rank and actually execute.

    Single process, so ``AllReduceLinear`` is a no-op and the numbers are partial
    sums -- shapes are what this asserts. Numerical exactness of the expert split
    is covered on CPU in ``test_moe_ep.py``.
    """
    base = _tiny_qwen35()
    ids = torch.randint(0, 256, (2, 16))
    for r in range(tp):
        m = copy.deepcopy(base)
        shard_moe_experts(m, r, tp)
        shard_model(m, r, tp)
        with torch.no_grad():
            out = m(ids).logits
        assert out.shape == (2, 16, 256), (tp, r, out.shape)


# --- fail at the seam, not six layers later ---------------------------------

def test_empty_row_slice_raises_instead_of_returning_a_zero_width_weight():
    lin = torch.nn.Linear(8, 16, bias=False)
    with pytest.raises(ValueError, match="zero width"):
        _slice_linear(lin, rows=(4, 4))


def test_empty_col_slice_raises():
    lin = torch.nn.Linear(8, 16, bias=False)
    with pytest.raises(ValueError, match="zero width"):
        _slice_linear(lin, cols=(3, 3))


def test_non_empty_slice_still_works():
    lin = torch.nn.Linear(8, 16, bias=False)
    out = _slice_linear(lin, rows=(4, 8))
    assert out.weight.shape == (4, 8)
