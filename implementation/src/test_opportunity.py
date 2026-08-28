# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for opportunity.py — autonomous target selection. Pure CPU."""

from __future__ import annotations

from dataclasses import dataclass

from opportunity import (analytic_opportunity, measured_opportunity,
                         rank_targets, select_targets)


@dataclass
class _Spec:
    name: str
    family: str = "dense_causal_lm"
    notes: str = ""


# --- analytic (device-free) --------------------------------------------------

def test_scan_is_the_compiler_weak_family():
    # scan (GDN) stays auto-worth; attention is NOT auto-worth without shape/%SOL
    # (2026-08-28 calibration: single-head dense attention is compiler-strong).
    specs = [_Spec("rmsnorm"), _Spec("gelu_tanh"), _Spec("gated_delta_rule"),
             _Spec("qkv_matmul")]
    verdicts = {t.op: t.worth_authoring for t in rank_targets(specs)}
    assert verdicts["gated_delta_rule"]                      # scan: compiler-weak
    assert not verdicts["rmsnorm"] and not verdicts["gelu_tanh"]


def test_attention_shape_unknown_is_moderate_not_auto_worth():
    # a bare attention spec with no inferable shape -> "measure it", NOT auto-worth
    t = analytic_opportunity(_Spec("flash_attention"))
    assert not t.worth_authoring
    assert "compiler-strong" in t.reason or "measure" in t.reason


def test_select_targets_drops_standard_ops():
    specs = [_Spec("rmsnorm"), _Spec("gelu_tanh"), _Spec("gated_delta_rule")]
    targets = [t.op for t in select_targets(specs)]
    assert set(targets) == {"gated_delta_rule"}   # only the scan op is auto-worth


# --- attention shape-aware calibration (the 2026-08-28 fix) ------------------

def test_single_head_attention_compiler_strong_low_opportunity():
    from invent_kernels import flash_attention_spec
    # single-head S=2048 -> ~4.2M score elems (well below LOW) -> compiler wins
    t = analytic_opportunity(flash_attention_spec(seqlen=2048, d_head=128))
    assert not t.worth_authoring and t.score <= 0.2
    assert "wins" in t.reason


def test_batched_multihead_attention_ranks_high_but_measure_only():
    from invent_kernels import mha_attention_spec
    # 8*32*2048^2 = 1.07e9 >= HIGH: ranks high-priority-to-MEASURE but is NOT
    # auto-worth (on-device: compiler is strong on attention at every size; a flash
    # win is unproven even batched -> only measured %SOL may select it).
    t = analytic_opportunity(mha_attention_spec(batch=8, heads=32, seqlen=2048,
                                                d_head=128))
    assert not t.worth_authoring and t.score >= 0.5
    assert "MEASURE" in t.reason


def test_attention_below_oom_not_auto_selected_analytically():
    from invent_kernels import mha_attention_spec, flash_attention_spec
    # below the OOM regime, no attention op is authored on the analytic signal
    # alone (compiler is competitive -> measured %SOL is the authority)
    specs = [flash_attention_spec(seqlen=8192, d_head=128),          # 6.7e7 elems
             mha_attention_spec(batch=8, heads=32, seqlen=2048, d_head=128)]  # 1.07e9
    assert select_targets(specs) == []


def test_attention_oom_regime_is_certain_worth():
    from invent_kernels import mha_attention_spec
    # B8/H16/S8192 -> 8*16*8192^2 = 8.6e9 elems (~34GB fp32) >= OOM threshold:
    # dense OOMs on trn2 (measured), flash is the ONLY path -> worth WITHOUT a race
    t = analytic_opportunity(mha_attention_spec(batch=8, heads=16, seqlen=8192,
                                                d_head=128))
    assert t.worth_authoring and t.score >= 0.8
    assert "OOM" in t.reason or "ONLY path" in t.reason
    # and it IS selected analytically (the one attention regime that is)
    assert [x.op for x in select_targets(
        [mha_attention_spec(batch=8, heads=16, seqlen=8192, d_head=128)])] \
        == ["mha_attention"]


def test_measured_low_sol_selects_attention():
    from invent_kernels import mha_attention_spec
    spec = mha_attention_spec(batch=8, heads=32, seqlen=2048, d_head=128)
    # a device race showing the compiler FAR from SOL -> attention IS selected
    ranked = rank_targets([spec], sol_fn=lambda s: (0.10, "memory_bound"))
    assert ranked[0].worth_authoring and ranked[0].source == "measured"
    # ...but near-SOL -> skipped
    ranked = rank_targets([spec], sol_fn=lambda s: (0.92, "compute_bound"))
    assert not ranked[0].worth_authoring and ranked[0].source == "measured"


def test_select_targets_max_cap():
    specs = [_Spec("flash_attention"), _Spec("gated_delta_rule"), _Spec("attn_decode")]
    assert len(select_targets(specs, max_targets=1)) == 1


# --- measured (%SOL) authoritative -------------------------------------------

def test_measured_near_sol_is_skipped():
    # a standard op that MEASURES far from SOL becomes worth authoring...
    t_low = measured_opportunity(_Spec("rmsnorm"), sol=0.05, bottleneck="memory_bound")
    assert t_low.worth_authoring and t_low.source == "measured"
    # ...and an op already near SOL is NOT (compiler wins)
    t_hi = measured_opportunity(_Spec("rmsnorm"), sol=0.9, bottleneck="memory_bound")
    assert not t_hi.worth_authoring


def test_measured_sol_overrides_analytic_and_higher_sol_ranks_lower():
    specs = [_Spec("flash_attention"), _Spec("gated_delta_rule")]
    # flash measures NEAR SOL (compiler handles this shape) -> should drop below GDN
    sols = {"flash_attention": (0.85, "compute_bound"),
            "gated_delta_rule": (0.10, "memory_bound")}
    ranked = rank_targets(specs, sol_fn=lambda s: sols[s.name])
    assert ranked[0].op == "gated_delta_rule"          # far-from-SOL wins
    assert ranked[0].source == "measured"
    # flash near-SOL is dropped from select_targets even though it's "attention"
    assert "flash_attention" not in {t.op for t in select_targets(specs, lambda s: sols[s.name])}


def test_sol_fn_failure_falls_back_to_analytic():
    specs = [_Spec("gated_delta_rule")]   # scan: auto-worth on the analytic signal
    def broken(_s):
        raise RuntimeError("no device")
    ranked = rank_targets(specs, sol_fn=broken)
    assert ranked[0].source == "analytic" and ranked[0].worth_authoring


def test_zero_sol_falls_back_to_analytic():
    # a deferred/off-device race returns sol=0 -> analytic (never a fake target)
    specs = [_Spec("rmsnorm")]
    ranked = rank_targets(specs, sol_fn=lambda s: (0.0, ""))
    assert ranked[0].source == "analytic"


def test_empty_and_nothing_worthwhile():
    assert select_targets([]) == []
    # a model of only standard ops -> nothing worth authoring (compiler wins all)
    assert select_targets([_Spec("rmsnorm"), _Spec("gelu_tanh")]) == []
