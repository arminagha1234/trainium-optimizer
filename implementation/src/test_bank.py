"""
Tests for the knowledge bank: save/load, both retrieval paths, anti-pattern
pruning, staleness, promotion, and confidence scoring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Symptom,
    Tier,
)
from ledger import Layer, Origin


def _config_prior(tier=Tier.VERIFIED) -> Lesson:
    return Lesson(
        lesson_id="dense-32b-tp8-baseline",
        type=LessonType.CONFIG_PRIOR,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            param_count_range=(20e9, 40e9),
            neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.CONFIG,
        migration_risk="medium",
        tier=tier,
        intervention={"spec": {"tp_degree": 8, "weights_dtype": "bf16"}},
        confidence=Confidence(n_models_validated=3, architecture_diversity=1,
                              human_verified=True),
        last_reverified_sdk="2.28.0",
    )


def _antipattern() -> Lesson:
    return Lesson(
        lesson_id="tp16-spill-small-models",
        type=LessonType.ANTI_PATTERN,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            param_count_range=(0, 30e9),
            neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.CONFIG,
        migration_risk="medium",
        tier=Tier.VERIFIED,
        matcher={"tp_degree": {"gte": 16}},
        reason="weight spill under 30B; 3x slower than TP=8",
        confidence=Confidence(n_models_validated=3, human_verified=True),
        last_reverified_sdk="2.28.0",
    )


def _kernel_with_symptom() -> Lesson:
    return Lesson(
        lesson_id="local-q-collective",
        type=LessonType.OP_REWRITE,
        applicability=Applicability(
            architecture_family="dense_causal_lm",
            neuron_sdk_versions=["2.28.*"],
        ),
        layer=Layer.COLLECTIVE,
        migration_risk="low-medium",
        origin=Origin.HARVESTED,
        tier=Tier.VERIFIED,
        intervention={"spec": {"attention_kernel": "kv_parallel"}},
        symptoms_addressed=[
            Symptom(bottleneck="collective_bound",
                    signature="all_gather of hidden states dominates",
                    observed_via="CC engine busy, PE idle"),
        ],
        source="nki-library@7f3a1b2",
        confidence=Confidence(n_models_validated=4, architecture_diversity=2,
                              human_verified=True),
        last_reverified_sdk="2.28.0",
    )


def test_save_load_roundtrip(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_config_prior())
    loaded = bank.load_all(Tier.VERIFIED)
    assert len(loaded) == 1
    assert loaded[0].lesson_id == "dense-32b-tp8-baseline"
    assert loaded[0].intervention["spec"]["tp_degree"] == 8


def test_intervention_query_matches_by_class(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_config_prior())
    hits = bank.query_interventions(
        family="dense_causal_lm", param_count=31e9, seq_len=1024, batch=1,
        sdk_version="2.28.0",
    )
    assert len(hits) == 1
    # out of range -> no hit
    assert bank.query_interventions(
        family="dense_causal_lm", param_count=7e9, seq_len=1024, batch=1,
        sdk_version="2.28.0",
    ) == []
    # wrong sdk -> no hit
    assert bank.query_interventions(
        family="dense_causal_lm", param_count=31e9, seq_len=1024, batch=1,
        sdk_version="2.20.0",
    ) == []


def test_symptom_query(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_kernel_with_symptom())
    hits = bank.query_symptom(
        bottleneck="collective_bound", family="dense_causal_lm",
        param_count=31e9, seq_len=1024, batch=1, sdk_version="2.28.0",
    )
    assert len(hits) == 1
    assert hits[0].lesson_id == "local-q-collective"
    # different bottleneck -> no hit
    assert bank.query_symptom(
        bottleneck="dma_bound", family="dense_causal_lm", param_count=31e9,
        seq_len=1024, batch=1, sdk_version="2.28.0",
    ) == []


def test_antipattern_pruning(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_antipattern())
    candidates = [
        {"tp_degree": 8, "weights_dtype": "bf16"},   # survives
        {"tp_degree": 16, "weights_dtype": "bf16"},  # pruned
        {"tp_degree": 32, "weights_dtype": "bf16"},  # pruned
    ]
    survivors, pruned = bank.prune(candidates, "dense-causal-lm", "2.28.0")
    assert len(survivors) == 1
    assert survivors[0]["tp_degree"] == 8
    assert len(pruned) == 2
    assert all("spill" in reason for _, reason in pruned)


def test_antipattern_does_not_fire_on_absent_axis(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_antipattern())
    # config without tp_degree -> matcher cannot fire
    survivors, pruned = bank.prune([{"weights_dtype": "fp8"}], "dense-causal-lm", "2.28.0")
    assert len(survivors) == 1
    assert pruned == []


def test_confidence_scoring():
    weak = Confidence(n_models_validated=1, human_verified=False)
    strong = Confidence(n_models_validated=5, architecture_diversity=2,
                        human_verified=True)
    assert strong.score() > weak.score()
    # staleness kills confidence past 2 SDK versions
    assert strong.score(sdk_versions_since_verified=3) == 0.0
    assert strong.score(sdk_versions_since_verified=2) < strong.score(0)


def test_promotion(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    prov = _config_prior(tier=Tier.PROVISIONAL)
    prov.confidence.human_verified = False
    bank.save(prov)
    assert len(bank.load_all(Tier.PROVISIONAL)) == 1
    assert len(bank.load_all(Tier.VERIFIED)) == 0

    bank.promote("dense-32b-tp8-baseline")
    assert len(bank.load_all(Tier.PROVISIONAL)) == 0
    verified = bank.load_all(Tier.VERIFIED)
    assert len(verified) == 1
    assert verified[0].confidence.human_verified is True


def test_staleness_flags_old_lessons(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    old = _config_prior()
    old.last_reverified_sdk = "2.24.0"
    old.lesson_id = "old-lesson"
    bank.save(old)
    fresh = _config_prior()
    fresh.lesson_id = "fresh-lesson"
    fresh.last_reverified_sdk = "2.28.0"
    bank.save(fresh)

    stale = bank.stale(current_sdk="2.28.0", max_versions=2)
    stale_ids = {l.lesson_id for l in stale}
    assert "old-lesson" in stale_ids       # 24 -> 28 is 4 versions
    assert "fresh-lesson" not in stale_ids


def _vllm_prior() -> Lesson:
    """A config_prior learned under the vllm-serve stack (distinct lesson_id and
    param range so it can co-exist with the native _config_prior)."""
    l = _config_prior()
    l.lesson_id = "vllm-32b-tp8-baseline"
    l.backend = "vllm-serve"
    return l


def test_query_is_backend_scoped(tmp_path: Path):
    """A native-tagged prior is returned for a native query but NOT a vllm one,
    and vice-versa — lessons don't leak across execution backends."""
    bank = KnowledgeBank(tmp_path)
    bank.save(_config_prior())     # default backend: native-pytorch
    bank.save(_vllm_prior())       # backend: vllm-serve
    qkw = dict(family="dense_causal_lm", param_count=31e9, seq_len=1024,
               batch=1, sdk_version="2.28.0")

    native = bank.query_interventions(backend="native-pytorch-beta3", **qkw)
    assert [l.lesson_id for l in native] == ["dense-32b-tp8-baseline"]

    vllm = bank.query_interventions(backend="vllm-serve", **qkw)
    assert [l.lesson_id for l in vllm] == ["vllm-32b-tp8-baseline"]

    # backend=None (existing callers) never filters: both are returned.
    both = bank.query_interventions(**qkw)
    assert len(both) == 2


def test_untagged_legacy_lesson_defaults_native(tmp_path: Path):
    """A legacy lesson whose YAML has no `backend` key loads as native-pytorch
    and is returned for a native query but not a vllm one."""
    bank = KnowledgeBank(tmp_path)
    p = bank.save(_config_prior())
    # Assert the on-disk YAML never carried the field (byte-for-byte roundtrip
    # for the native default) and simulate an older lesson explicitly.
    assert "backend:" not in p.read_text()

    loaded = bank.load_all(Tier.VERIFIED)[0]
    assert loaded.backend == "native-pytorch"

    qkw = dict(family="dense_causal_lm", param_count=31e9, seq_len=1024,
               batch=1, sdk_version="2.28.0")
    assert len(bank.query_interventions(backend="native-pytorch-beta3", **qkw)) == 1
    assert bank.query_interventions(backend="vllm-serve", **qkw) == []


def test_nondefault_backend_roundtrips_and_mock_matches_all(tmp_path: Path):
    """A non-default backend survives save/load, and the synthetic `mock`
    backend matches every stack (so mock-backed tests seed priors unchanged)."""
    bank = KnowledgeBank(tmp_path)
    p = bank.save(_vllm_prior())
    assert "backend: vllm-serve" in p.read_text()   # non-default IS emitted

    loaded = next(l for l in bank.load_all(Tier.VERIFIED)
                  if l.lesson_id == "vllm-32b-tp8-baseline")
    assert loaded.backend == "vllm-serve"

    # A mock query matches the vllm-tagged lesson (mock matches every stack).
    hits = bank.query_interventions(
        family="dense_causal_lm", param_count=31e9, seq_len=1024, batch=1,
        sdk_version="2.28.0", backend="mock",
    )
    assert loaded.lesson_id in {l.lesson_id for l in hits}


def test_stats(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_config_prior())
    bank.save(_antipattern())
    bank.save(_kernel_with_symptom())
    s = bank.stats(current_sdk="2.28.0")
    assert s["verified"] == 3
    assert s["by_type"]["config_prior"] == 1
    assert s["by_type"]["anti_pattern"] == 1
    assert s["human_verified_ratio"] == 1.0


# --- query_by_op: op-family-keyed retrieval (cross model-family reuse) --------

def _nki_kernel_lesson(op, shape_class, model_family, tier=Tier.VERIFIED,
                       lid=None) -> Lesson:
    """A banked NKI-kernel lesson carrying a structured op identity (as
    invent_engine._bank_win writes it)."""
    return Lesson(
        lesson_id=lid or f"invented-{op}-{shape_class}",
        type=LessonType.NKI_KERNEL,
        applicability=Applicability(
            architecture_family=model_family, neuron_sdk_versions=["2.28.*"]),
        layer=Layer.KERNEL, migration_risk="low", tier=tier,
        intervention={"spec": {"nki_kernel": op, "shape_class": shape_class}},
        confidence=Confidence(n_models_validated=2), last_reverified_sdk="2.28.0",
    )


def test_query_by_op_exact_and_cross_model_family(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    # An rmsnorm kernel learned on a MoE model, and one on a dense model.
    bank.save(_nki_kernel_lesson("rmsnorm", "p-le128", "moe-causal-lm"))
    bank.save(_nki_kernel_lesson("add_rmsnorm", "p-le128", "dense-causal-lm"))
    bank.save(_nki_kernel_lesson("softmax", "p-le128", "dense-causal-lm"))

    # Query for rmsnorm on a DENSE model — must find the MoE-learned rmsnorm
    # (exact, rank 3) AND the add_rmsnorm (same NORMALIZATION family, rank 1),
    # but NOT softmax (different op-family).
    hits = bank.query_by_op("rmsnorm", shape_class="p-le128")
    ops = [h.intervention["spec"]["nki_kernel"] for h in hits]
    assert "rmsnorm" in ops and "add_rmsnorm" in ops
    assert "softmax" not in ops
    # exact match ranks first
    assert ops[0] == "rmsnorm"


def test_query_by_op_shape_class_ranking(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_nki_kernel_lesson("layernorm", "p-le128", "dense-causal-lm",
                                 lid="invented-layernorm-small"))
    bank.save(_nki_kernel_lesson("groupnorm", "p-le128", "dense-causal-lm",
                                 lid="invented-groupnorm-match"))
    bank.save(_nki_kernel_lesson("rmsnorm", "p-gt128", "dense-causal-lm",
                                 lid="invented-rmsnorm-otsc"))
    # Query rmsnorm@p-le128: groupnorm@p-le128 (same family+shape → rank 2) must
    # outrank rmsnorm@p-gt128 (same family, different shape → rank 1).
    hits = bank.query_by_op("rmsnorm", shape_class="p-le128")
    ops = [h.intervention["spec"]["nki_kernel"] for h in hits]
    assert ops.index("groupnorm") < ops.index("rmsnorm")


def test_query_by_op_ignores_non_op_lessons(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    # A config prior (no op identity) must never surface in query_by_op.
    bank.save(_config_prior())
    bank.save(_nki_kernel_lesson("rmsnorm", "p-le128", "dense-causal-lm"))
    hits = bank.query_by_op("rmsnorm")
    assert len(hits) == 1
    assert hits[0].intervention["spec"]["nki_kernel"] == "rmsnorm"


def test_query_by_op_tier_scoped(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_nki_kernel_lesson("rmsnorm", "p-le128", "moe-causal-lm",
                                 tier=Tier.PROVISIONAL))
    # Default reads VERIFIED only — a provisional lesson is not returned...
    assert bank.query_by_op("rmsnorm") == []
    # ...unless the caller explicitly asks for the provisional tier.
    assert len(bank.query_by_op("rmsnorm", tier=Tier.PROVISIONAL)) == 1
