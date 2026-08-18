"""
Tests for autonomous bank promotion (provisional -> verified without a human).

This is the switch that turns the bank from self-logging into self-learning:
with it on, a lesson proven mid-run is trusted by later models in the same run.
The bar must be real, and auto-promoted lessons must stay distinguishable from
human-verified ones.
"""

from __future__ import annotations

from pathlib import Path

from bank import (
    Applicability,
    AutoPromotionPolicy,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Tier,
)
from ledger import Layer, Origin


def _provisional(
    lesson_id="cand", n_models=2, diversity=2, origin=Origin.NONE,
    beat_borrowed=None, correctness=100.0,
) -> Lesson:
    return Lesson(
        lesson_id=lesson_id,
        type=LessonType.CONFIG_PRIOR,
        applicability=Applicability("dense_causal_lm", (20e9, 40e9),
                                    neuron_sdk_versions=["2.28.*"]),
        layer=Layer.CONFIG, migration_risk="medium",
        origin=origin, tier=Tier.PROVISIONAL,
        intervention={"spec": {"tp_degree": 8}},
        confidence=Confidence(n_models_validated=n_models,
                              architecture_diversity=diversity,
                              human_verified=False),
        last_reverified_sdk="2.28.0",
        beat_borrowed_by=beat_borrowed,
        evidence=[{"model": "m1", "correctness": correctness}],
    )


def test_disabled_policy_is_noop(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional())
    results = bank.auto_promote(AutoPromotionPolicy())   # enabled=False
    assert results == []
    assert bank.load_all(Tier.VERIFIED) == []
    assert len(bank.load_all(Tier.PROVISIONAL)) == 1


def test_qualifying_lesson_promotes(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional(n_models=2, diversity=2))
    results = bank.auto_promote(AutoPromotionPolicy(enabled=True))
    assert results and results[0][1] is True
    verified = bank.load_all(Tier.VERIFIED)
    assert len(verified) == 1
    assert len(bank.load_all(Tier.PROVISIONAL)) == 0
    # trusted, but NOT claimed as human-verified
    assert verified[0].auto_promoted is True
    assert verified[0].confidence.human_verified is False
    assert verified[0].promoted_at


def test_too_few_models_stays_provisional(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional(n_models=1))
    results = bank.auto_promote(AutoPromotionPolicy(enabled=True, min_models=2))
    assert results[0][1] is False
    assert len(bank.load_all(Tier.PROVISIONAL)) == 1


def test_single_family_blocked_by_default_but_allowed_overnight(tmp_path: Path):
    # diversity=1 fails the default (min_families=2) ...
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional(n_models=2, diversity=1))
    assert bank.auto_promote(AutoPromotionPolicy(enabled=True))[0][1] is False
    assert len(bank.load_all(Tier.PROVISIONAL)) == 1
    # ... but the overnight preset allows single-family compounding.
    assert bank.auto_promote(AutoPromotionPolicy.overnight())[0][1] is True
    assert len(bank.load_all(Tier.VERIFIED)) == 1


def test_invented_needs_margin(tmp_path: Path):
    # invented with no margin recorded -> blocked
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional("inv-nomargin", origin=Origin.INVENTED, beat_borrowed=None))
    assert bank.auto_promote(AutoPromotionPolicy.overnight())[0][1] is False

    # invented that beat borrowed by only 3% (< 5% bar) -> blocked
    bank2 = KnowledgeBank(tmp_path / "b2")
    bank2.save(_provisional("inv-thin", origin=Origin.INVENTED, beat_borrowed=0.03))
    assert bank2.auto_promote(AutoPromotionPolicy.overnight())[0][1] is False

    # invented that beat borrowed by 12% -> promotes
    bank3 = KnowledgeBank(tmp_path / "b3")
    bank3.save(_provisional("inv-strong", origin=Origin.INVENTED, beat_borrowed=0.12))
    assert bank3.auto_promote(AutoPromotionPolicy.overnight())[0][1] is True


def test_failed_correctness_blocks(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional(correctness=91.0))    # below the 99% gate
    assert bank.auto_promote(AutoPromotionPolicy.overnight())[0][1] is False


def test_auto_promoted_not_counted_as_human_verified(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_provisional(n_models=2, diversity=2))
    bank.auto_promote(AutoPromotionPolicy(enabled=True))
    stats = bank.stats(current_sdk="2.28.0")
    assert stats["verified"] == 1
    assert stats["auto_promoted"] == 1
    assert stats["human_verified_ratio"] == 0.0   # honesty: not human-signed
