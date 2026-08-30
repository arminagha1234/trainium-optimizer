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



def test_a_24_head_model_stops_at_8_because_12_and_24_cannot_form_a_collective():
    """tp=24 is arithmetically valid and physically unrunnable.

    24 divides evenly by 3, 6, 12 and 24, and every one of those world sizes fails
    `init_process_group` with "Failed to execute the device barrier 2". So the
    reachable set is the powers of two dividing 24: tp=8 really is the ceiling for
    this model, and the other 56 cores are reachable by running more REPLICAS, not by
    a wider shard.
    """
    heads, _, lvh = GEOMETRY["Qwen3.8-27B"]
    cap = tp_cap_for("Qwen3_5ForConditionalGeneration", heads, lvh, CORES_48XL)
    tps = tp_candidates(heads, cap)
    assert tps == [1, 2, 4, 8]
    for unformable in (3, 6, 12, 24):
        assert unformable not in tps, f"tp={unformable} cannot form a collective"


def test_no_axis_ever_proposes_a_non_power_of_two():
    """The rule that cost four 55 GB checkpoint loads to learn."""
    for name, (heads, _, lvh) in GEOMETRY.items():
        cap = tp_cap_for("X", heads, lvh, CORES_48XL)
        for tp in tp_candidates(heads, cap):
            assert tp & (tp - 1) == 0, f"{name}: tp={tp} is not a power of two"


def test_minimax_m2_is_capped_at_16_not_48():
    """48 heads divide by 48, and 48 is not a power of two -- so the tp=48 the launch
    plan called for was never runnable."""
    tps = tp_candidates(48, tp_cap_for("MiniMaxM2ForCausalLM", 48, None, CORES_48XL))
    assert 48 not in tps
    assert max(tps) == 16          # 1,2,4,8,16 divide 48; 32 does not


def test_both_filters_are_load_bearing():
    """Neither constraint alone gives the right answer for a 24-head model."""
    both = tp_candidates(24, tp_cap_for("X", 24, None, CORES_48XL))
    assert 24 % 12 == 0 and 12 not in both        # the ladder removes it
    assert 24 % 16 != 0 and 16 not in both        # the divisor check removes it


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
    assert reach["Qwen3.8-27B"] == 8             # 24 heads, but 12/24 cannot form


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
