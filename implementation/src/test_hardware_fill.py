"""
Tests for hardware-aware "fill the box" behavior.

The motivating bug: a 27B model with 4 KV heads is TP-capped at 4, which on a
64-core trn2.48xlarge (LNC=2) left ~94% of the instance idle. These tests lock
in that the fill planner and the proposer now use the whole instance, and that
the guardrail flags under-utilization.
"""

from __future__ import annotations

from pathlib import Path

from backends.mock import MockBackend
from bank import KnowledgeBank
from guardrails import Guardrails
from hardware import ComputeBudget, budget_for, fill_plan
from ledger import Ledger, Status
from orchestrator import ModelSpec, Orchestrator, always_equivalent
from proposer import BeamProposer
from backends.base import Measurements


# -- topology + fill plan ----------------------------------------------------

def test_trn2_48xl_budget():
    b = budget_for("trn2.48xlarge")
    assert b.num_cores == 64        # 16 chips x 8 physical -> 64 logical at LNC=2
    assert b.hbm_gib_per_core == 24.0


def test_tp4_fills_box_with_dp():
    """The exact case from the bug report: 4 KV heads -> TP=4 -> must fill the
    remaining 60 cores with DP replicas, not leave them idle."""
    b = budget_for("trn2.48xlarge")
    plan = fill_plan(b, tp=4, num_kv_heads=4)
    assert plan.dp == 16            # 64 cores / (tp=4) = 16 replicas
    assert plan.cores_used == 64
    assert plan.utilization == 1.0
    assert plan.kv_replication == 1


def test_tp_beyond_kv_heads_flags_replication():
    """TP=8 with 4 KV heads is allowed but must be marked as needing KV
    replication (a testable option, not a hard ban)."""
    b = budget_for("trn2.48xlarge")
    plan = fill_plan(b, tp=8, num_kv_heads=4)
    assert plan.kv_replication == 2
    assert plan.dp == 8          # 8 groups of 8 fill the 64-core box
    assert plan.cores_used == 64


def test_latency_track_does_not_replicate():
    """DP replicas do not cut single-request latency, so the latency track
    keeps dp=1 and fills via tp/cp instead."""
    b = budget_for("trn2.48xlarge")
    plan = fill_plan(b, tp=4, num_kv_heads=4, track="latency")
    assert plan.dp == 1


def test_runtime_core_override():
    """An explicit core count (e.g. from NEURON_RT_NUM_CORES) overrides the
    static table."""
    b = budget_for("trn2.48xlarge", num_cores=8)
    assert b.num_cores == 8
    assert fill_plan(b, tp=4, num_kv_heads=4).dp == 2


# -- proposer fills every candidate -----------------------------------------

def test_proposer_fills_every_candidate():
    b = budget_for("trn2.48xlarge")
    axes = {"tp_degree": [4, 8], "weights_dtype": ["bf16", "fp8"]}
    p = BeamProposer(axes=axes, budget=b, num_kv_heads=4)
    beam = p.seed(baseline={"tp_degree": 4, "weights_dtype": "bf16"},
                  family="hybrid_attention_causal_lm", param_count=27e9,
                  seq_len=1024, batch=1, sdk_version="2.28.0")
    cands = p.expand(beam)
    assert cands, "expected candidates"
    for c in beam + cands:
        # every candidate uses the whole 64-core box
        assert c.config["cores_used"] == 64
        assert c.config["cores_available"] == 64
        assert c.config["dp_degree"] >= 1


def test_proposer_without_budget_is_unchanged():
    """No budget -> no fill fields -> identical to the original behavior."""
    axes = {"tp_degree": [4, 8]}
    p = BeamProposer(axes=axes)
    beam = p.seed(baseline={"tp_degree": 4}, family="dense_causal_lm",
                  param_count=27e9, seq_len=1024, batch=1, sdk_version="2.28.0")
    assert "dp_degree" not in beam[0].config
    assert "cores_used" not in beam[0].config


# -- guardrail ---------------------------------------------------------------

def test_utilization_guardrail():
    g = Guardrails()
    full = Measurements(metric=1.0, cores_used=16, cores_available=16)
    quarter = Measurements(metric=1.0, cores_used=4, cores_available=16)
    unknown = Measurements(metric=1.0)  # cores_available=0
    assert g.utilization_ok(full) is True
    assert g.utilization_ok(quarter) is False
    assert g.utilization_ok(unknown) is True   # unknown occupancy is not a failure


# -- end to end: orchestrator fills the instance ----------------------------

def test_orchestrator_fills_the_instance():
    """With instance_type set, the winning config uses the whole box, and DP
    replicas make it strictly faster than the single-replica baseline."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        orch = Orchestrator(
            backend=MockBackend(seed=7),
            bank=KnowledgeBank(root / "bank"),
            guards=Guardrails(),
            ledger=Ledger(root / "run"),
            equivalence=always_equivalent,
            sdk_version="2.28.0",
            instance_type="trn2.48xlarge",
        )
        orch.ledger.init()
        spec = ModelSpec(
            model_id="qwen/qwen3.8-27b", family="hybrid_attention_causal_lm",
            param_count=27e9, parent="qwen", num_kv_heads=4,
        )
        best = orch.run_stage1_config(spec)
        assert best.config["cores_available"] == 64
        assert best.config["cores_used"] == 64      # the box is filled
        assert best.config["dp_degree"] >= 2        # replicas, not a single group
        # baseline in the mock is ~600 tok/s single-replica; filling the box
        # must beat that comfortably.
        assert best.metric > 600


# -- TP x DP sweep (measure every box-filling partition) ---------------------

def test_tp_axis_sweeps_to_core_count():
    """With a budget, the TP axis must reach the instance core count so the
    full TP x DP grid is searched, not the hardcoded [1,2,4,8]."""
    b = budget_for("trn2.48xlarge")            # 64 cores
    p = BeamProposer(axes={"tp_degree": [1, 2, 4, 8], "weights_dtype": ["bf16"]},
                     budget=b)
    assert set(p.axes["tp_degree"]) >= {1, 2, 4, 8, 16, 32, 64}


def test_tp_dp_grid_every_partition_fills_the_box():
    """Each TP in the sweep pairs with the DP that fills the 64-core box:
    TP=64/DP=1 ... TP=1/DP=64 — the 'complete mix' to compare."""
    b = budget_for("trn2.48xlarge")
    for tp, exp_dp in [(64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64)]:
        plan = fill_plan(b, tp=tp)
        assert plan.dp == exp_dp, f"tp={tp} -> dp should be {exp_dp}, got {plan.dp}"
        assert plan.cores_used == 64


def test_antipattern_is_verify_first_by_backend(tmp_path):
    """An anti-pattern validated only on XLA must NOT pre-prune high TP on the
    native backend — those get measured so the prior is verified there first."""
    from bank import (Applicability, Confidence, KnowledgeBank, Lesson,
                      LessonType, Tier)
    from ledger import Layer

    bank = KnowledgeBank(tmp_path)
    bank.save(Lesson(
        lesson_id="tp16-xla-only", type=LessonType.ANTI_PATTERN,
        applicability=Applicability("dense_causal_lm", (0, 30e9),
                                    neuron_sdk_versions=["2.28.*"]),
        layer=Layer.CONFIG, migration_risk="medium", tier=Tier.VERIFIED,
        matcher={"tp_degree": {"gte": 16}}, reason="spills on XLA",
        confidence=Confidence(n_models_validated=3, human_verified=True),
        last_reverified_sdk="2.28.0", backend_validated=["vllm-neuron-xla"],
    ))
    cfgs = [{"tp_degree": 32}]
    # validated backend -> pruned
    surv, pruned = bank.prune(cfgs, "dense-causal-lm", "2.28.0", backend="vllm-neuron-xla")
    assert pruned and not surv
    # a different backend -> measured, not pruned
    surv, pruned = bank.prune(cfgs, "dense-causal-lm", "2.28.0", backend="native-pytorch-beta3")
    assert surv and not pruned
    # backend unspecified -> backward-compatible always-prune
    surv, pruned = bank.prune(cfgs, "dense-causal-lm", "2.28.0")
    assert pruned and not surv
