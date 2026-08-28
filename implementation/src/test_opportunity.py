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

def test_analytic_ranks_compiler_weak_families_first():
    specs = [_Spec("rmsnorm"), _Spec("flash_attention"), _Spec("gelu_tanh"),
             _Spec("gated_delta_rule"), _Spec("qkv_matmul")]
    ranked = rank_targets(specs)  # no sol_fn -> analytic
    top2 = {t.op for t in ranked[:2]}
    assert top2 == {"flash_attention", "gated_delta_rule"}   # attention + scan
    # standard ops are worth_authoring=False (compiler near-SOL)
    verdicts = {t.op: t.worth_authoring for t in ranked}
    assert verdicts["flash_attention"] and verdicts["gated_delta_rule"]
    assert not verdicts["rmsnorm"] and not verdicts["gelu_tanh"]


def test_select_targets_drops_standard_ops():
    specs = [_Spec("rmsnorm"), _Spec("gelu_tanh"), _Spec("flash_attention"),
             _Spec("gated_delta_rule")]
    targets = [t.op for t in select_targets(specs)]
    assert set(targets) == {"flash_attention", "gated_delta_rule"}


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
    specs = [_Spec("flash_attention")]
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
