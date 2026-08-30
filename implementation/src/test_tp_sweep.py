"""TP sweep breadth: what the search is allowed to propose.

Two bugs this pins, both of which left most of a 64-core box unreachable:

1. The axis offered only POWERS OF TWO, so a 24-head model capped at tp=8 and left
   40 cores idle when tp=12 and tp=24 are valid shardings.
2. `config_axes` still capped Qwen3.5 at tp=4 after #121 raised that cap to the head
   count in `_fit_baseline_tp`. The baseline could run at tp=8 or tp=32 while the
   search would never propose above 4 -- for exactly the large MoE models that need
   it. Qwen3.8-27B is `Qwen3_5ForConditionalGeneration`, so it was hit too.
"""

from __future__ import annotations

import pytest

from backends.native_pytorch import tp_candidates, tp_cap_for

CORES_48XL = 64


# --- real geometries, read from the published configs -------------------------
#      model                heads  kv  linear_value_heads
GEOMETRY = {
    "Qwen3.8-27B":        (24, 4, 48),
    "Qwen3.5-35B-A3B":    (16, 2, 32),
    "Qwen3.5-122B-A10B":  (32, 2, 64),
    "DeepSeek-V4-Flash":  (64, 1, None),
}


def test_a_24_head_model_can_now_use_24_cores_not_8():
    """The concrete win: 3x more of the box becomes reachable."""
    heads, _, lvh = GEOMETRY["Qwen3.8-27B"]
    cap = tp_cap_for("Qwen3_5ForConditionalGeneration", heads, lvh, CORES_48XL)
    tps = tp_candidates(heads, cap)
    assert tps == [1, 2, 3, 4, 6, 8, 12, 24]
    assert max(tps) == 24
    powers_of_two_only = [t for t in (1, 2, 4, 8, 16, 32, 64) if heads % t == 0]
    assert max(powers_of_two_only) == 8      # what it used to stop at


def test_qwen3_5_is_no_longer_capped_at_four():
    """#121 raised this in the baseline chooser; the axis kept the old value."""
    for name in ("Qwen3.5-35B-A3B", "Qwen3.5-122B-A10B"):
        heads, _, lvh = GEOMETRY[name]
        cap = tp_cap_for("Qwen3_5MoeForConditionalGeneration", heads, lvh, CORES_48XL)
        assert cap > 4, name
        assert max(tp_candidates(heads, cap)) == heads, name


def test_tp_64_is_reachable_exactly_when_the_head_count_allows_it():
    """TP=64 is not a setting, it is an arithmetic property of the model."""
    reach = {}
    for name, (heads, _, lvh) in GEOMETRY.items():
        cap = tp_cap_for("X", heads, lvh, CORES_48XL)
        reach[name] = max(tp_candidates(heads, cap))
    assert reach["DeepSeek-V4-Flash"] == 64      # 64 heads
    assert reach["Qwen3.5-122B-A10B"] == 32      # 32 heads: 64 is not expressible
    assert reach["Qwen3.5-35B-A3B"] == 16
    assert reach["Qwen3.8-27B"] == 24


def test_gemma4_stays_hard_capped_at_four():
    """head_dim 512 with 4 KV heads is arithmetic, not a validation limit."""
    assert tp_cap_for("Gemma4ForConditionalGeneration", 32, None, CORES_48XL) == 4


def test_deltanet_value_heads_bound_the_cap():
    """Value heads cannot be replicated -- out_proj all-reduces (#135).

    Proposing tp above them would raise inside the shard instead of simply not
    being offered.
    """
    assert tp_cap_for("Qwen3_5Moe", 64, 8, CORES_48XL) == 8
    # ...and the divisor list respects it.
    assert max(tp_candidates(64, tp_cap_for("Qwen3_5Moe", 64, 8, CORES_48XL))) == 8


def test_the_cap_never_exceeds_physical_cores():
    """A 64-head model on a 4-core box must not be offered tp=64."""
    assert tp_cap_for("X", 64, None, 4) == 4
    assert tp_candidates(64, tp_cap_for("X", 64, None, 4)) == [1, 2, 4]


def test_every_proposed_tp_divides_the_head_count():
    """The worker rejects a non-dividing tp, so proposing one wastes a launch."""
    for name, (heads, _, lvh) in GEOMETRY.items():
        cap = tp_cap_for("X", heads, lvh, CORES_48XL)
        for t in tp_candidates(heads, cap):
            assert heads % t == 0, (name, t)


def test_an_unknown_head_count_falls_back_to_the_power_of_two_ladder():
    """Nothing to divide, so offer the old ladder rather than nothing."""
    assert tp_candidates(None, 64) == [1, 2, 4, 8, 16, 32, 64]
    assert tp_candidates(0, 8) == [1, 2, 4, 8]


def test_cap_is_at_least_one():
    assert tp_cap_for("X", 0, None, 0) == 1
    assert tp_candidates(24, 1) == [1]
