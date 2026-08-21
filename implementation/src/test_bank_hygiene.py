"""
Tests for bank hygiene / toolchain re-validation.

The safeguard that keeps the bank improving instead of rotting: when the
toolchain moves off what a verified prior was validated on, the prior is
re-measured on a canary and either re-earns its stamp (revalidated) or is
demoted verified->provisional with a staleness breadcrumb. A stale-wrong prior
is never silently kept.

All harness logic here runs on a plain CPU box with mock lessons: the
re-measurement is injected (a scripted CanaryMeasurement) so the demote/refresh
bookkeeping is deterministic, plus one end-to-end test that the real
Orchestrator measure+equivalence path composes with a MockBackend.
"""

from __future__ import annotations

from pathlib import Path

from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Tier,
)
from bank_hygiene import (
    CanaryMeasurement,
    RevalidationReport,
    canary_from_specs,
    current_toolchain,
    dominant_sdk,
    maybe_revalidate_at_startup,
    revalidate,
    stale_priors,
    toolchain_changed,
)
from guardrails import Guardrails
from ledger import Layer, Origin
from orchestrator import ModelSpec


# --- fixtures ---------------------------------------------------------------

def _prior(
    lesson_id="p1", family="dense_causal_lm", sdk_globs=("2.28.*",),
    metric=1000.0, backend="native-pytorch", config=None,
    human_verified=True,
) -> Lesson:
    return Lesson(
        lesson_id=lesson_id,
        type=LessonType.CONFIG_PRIOR,
        applicability=Applicability(
            architecture_family=family,
            param_count_range=(0.0, 1e15),
            neuron_sdk_versions=list(sdk_globs),
        ),
        layer=Layer.CONFIG, migration_risk="medium",
        origin=Origin.NONE, tier=Tier.VERIFIED,
        intervention={"spec": config or {"tp_degree": 4, "weights_dtype": "bf16"}},
        backend=backend,
        confidence=Confidence(n_models_validated=3, architecture_diversity=2,
                              human_verified=human_verified),
        last_reverified_sdk=sdk_globs[0].replace(".*", ".0"),
        evidence=[{"model": "m1", "metric": metric}],
    )


class _StubBackend:
    """A backend with just a name — enough for the backend-scoping filter when
    measurement is injected."""
    def __init__(self, name):
        self.name = name


def _measure(result: CanaryMeasurement):
    """A measure_fn that always returns the same scripted outcome."""
    return lambda backend, spec, config, guards: result


_CANARIES = {"dense_causal_lm": ModelSpec(
    model_id="Qwen/Qwen3-0.6B", family="dense_causal_lm", param_count=0.6e9)}


# --- current_toolchain / change detection -----------------------------------

def test_current_toolchain_never_raises():
    # No compiler on a CPU box -> "" (best-effort), never an exception.
    assert isinstance(current_toolchain(), str)


def test_dominant_sdk_and_change(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    assert dominant_sdk(bank) is None
    assert toolchain_changed(bank, "2.30.0") is False        # empty bank -> no storm
    bank.save(_prior("a", sdk_globs=("2.28.*",)))
    bank.save(_prior("b", sdk_globs=("2.28.*",)))
    assert dominant_sdk(bank) == "2.28.0"
    assert toolchain_changed(bank, "2.30.0") is True         # 30 != 28
    assert toolchain_changed(bank, "2.28.6360.0") is False   # same minor
    assert toolchain_changed(bank, "garbage") is False       # unparseable -> safe


# --- stale detection --------------------------------------------------------

def test_stale_priors_selects_uncovered_verified(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("old", sdk_globs=("2.28.*",)))
    bank.save(_prior("current", sdk_globs=("2.30.*",)))
    stale = stale_priors(bank, "2.30.0")
    assert [l.lesson_id for l in stale] == ["old"]


def test_stale_priors_covered_by_last_reverified(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    l = _prior("reverified", sdk_globs=("2.28.*",))
    l.last_reverified_sdk = "2.30.1"          # re-verified on the current minor
    bank.save(l)
    assert stale_priors(bank, "2.30.0") == []


# --- revalidate: refresh (still wins) ---------------------------------------

def test_revalidate_refreshes_when_still_wins(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", sdk_globs=("2.28.*",), metric=1000.0))
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=1050.0,
                                              equivalence_passed=True, fits=True)))
    assert report.revalidated == 1 and report.demoted == 0
    # Still verified, SDK coverage extended, re-verify stamp bumped.
    verified = bank.load_all(Tier.VERIFIED)
    assert len(verified) == 1
    l = verified[0]
    assert "2.30.*" in l.applicability.neuron_sdk_versions
    assert l.last_reverified_sdk == "2.30.0"
    assert not bank.load_all(Tier.PROVISIONAL)


def test_revalidate_within_margin_still_wins(tmp_path: Path):
    # 5% slower is within the 10% regression tolerance -> still a win.
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", metric=1000.0))
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=950.0,
                                              equivalence_passed=True, fits=True)))
    assert report.revalidated == 1


# --- revalidate: demote paths -----------------------------------------------

def _demote_setup(tmp_path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", sdk_globs=("2.28.*",), metric=1000.0))
    return bank


def _assert_demoted(bank):
    assert bank.load_all(Tier.VERIFIED) == []
    prov = bank.load_all(Tier.PROVISIONAL)
    # the demoted prior + the staleness breadcrumb
    ids = sorted(l.lesson_id for l in prov)
    demoted = next(l for l in prov if l.lesson_id == "p")
    assert demoted.confidence.human_verified is False   # trust withdrawn
    assert demoted.auto_promoted is False
    assert demoted.demoted_at and demoted.demote_reason
    stale_note = next(l for l in prov if l.lesson_id.startswith("staleness-p-"))
    assert stale_note.type is LessonType.ANTI_PATTERN
    assert stale_note.matcher.get("stale_prior") == "p"
    return demoted, stale_note


def test_revalidate_demotes_on_equivalence_fail(tmp_path: Path):
    bank = _demote_setup(tmp_path)
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=1100.0,
                                              equivalence_passed=False,
                                              notes="top1 drift")))
    assert report.demoted == 1
    _, note = _assert_demoted(bank)
    assert "equivalence" in note.reason


def test_revalidate_demotes_on_regression(tmp_path: Path):
    bank = _demote_setup(tmp_path)
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=500.0,   # 50% slower
                                              equivalence_passed=True, fits=True)))
    assert report.demoted == 1
    _assert_demoted(bank)


def test_revalidate_demotes_on_oom(tmp_path: Path):
    bank = _demote_setup(tmp_path)
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=1100.0,
                                              equivalence_passed=True, fits=False,
                                              notes="OOM 92% HBM")))
    assert report.demoted == 1
    _, note = _assert_demoted(bank)
    assert "fit" in note.reason


# --- revalidate: conservative on transient failures -------------------------

def test_revalidate_inconclusive_on_transient(tmp_path: Path):
    # ok=False (crash / noisy) must NEVER demote — the prior stays verified.
    bank = _demote_setup(tmp_path)
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=False, notes="noisy measurement")))
    assert report.inconclusive == 1 and report.demoted == 0
    assert len(bank.load_all(Tier.VERIFIED)) == 1     # untouched
    assert not bank.load_all(Tier.PROVISIONAL)


# --- revalidate: skips --------------------------------------------------------

def test_revalidate_skips_when_no_canary(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", family="moe_causal_lm"))     # no moe canary
    report = revalidate(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=1.0)))
    assert report.skipped == 1 and report.demoted == 0
    assert len(bank.load_all(Tier.VERIFIED)) == 1


def test_revalidate_scopes_to_backend(tmp_path: Path):
    # A native-pytorch prior is not re-measured on a vllm-serve run.
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", backend="native-pytorch"))
    report = revalidate(
        bank, _StubBackend("vllm-serve"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=500.0,
                                              equivalence_passed=True)))
    assert report.outcomes == []                       # filtered out entirely
    assert len(bank.load_all(Tier.VERIFIED)) == 1


# --- bank.demote unit -------------------------------------------------------

def test_bank_demote_moves_and_withdraws_trust(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", human_verified=True))
    bank.demote("p", reason="regressed under sdk 2.30.0")
    assert bank.load_all(Tier.VERIFIED) == []
    prov = bank.load_all(Tier.PROVISIONAL)
    assert len(prov) == 1
    assert prov[0].confidence.human_verified is False
    assert prov[0].demote_reason == "regressed under sdk 2.30.0"
    # nonexistent verified lesson raises, matching promote()'s contract
    import pytest
    with pytest.raises(KeyError):
        bank.demote("nope")


# --- hook --------------------------------------------------------------------

def test_hook_noop_when_toolchain_unchanged(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", sdk_globs=("2.30.*",)))       # already current
    called = []
    report = maybe_revalidate_at_startup(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=1.0)),
        log=called.append)
    assert report is None                               # no-op
    assert len(bank.load_all(Tier.VERIFIED)) == 1


def test_hook_runs_when_toolchain_changed(tmp_path: Path):
    bank = KnowledgeBank(tmp_path)
    bank.save(_prior("p", sdk_globs=("2.28.*",), metric=1000.0))
    report = maybe_revalidate_at_startup(
        bank, _StubBackend("native-pytorch"), _CANARIES, "2.30.0", Guardrails(),
        measure_fn=_measure(CanaryMeasurement(ok=True, metric=1100.0,
                                              equivalence_passed=True, fits=True)))
    assert report is not None and report.revalidated == 1


def test_canary_from_specs_picks_smallest_per_family():
    specs = {
        "big": ModelSpec("A/big", "dense_causal_lm", 32e9),
        "small": ModelSpec("A/small", "dense_causal_lm", 0.6e9),
        "moe": ModelSpec("A/moe", "moe_causal_lm", 30e9),
    }
    canaries = canary_from_specs(specs)
    assert canaries["dense_causal_lm"].param_count == 0.6e9
    assert canaries["moe_causal_lm"].model_id == "A/moe"


# --- integration: real Orchestrator + MockBackend path ----------------------

def test_default_measure_path_composes_with_mock_backend(tmp_path: Path):
    """End-to-end with the REAL measure+equivalence path (no injected measure_fn).
    Proves bank_hygiene composes with the Orchestrator/backend stack on CPU."""
    from backends.mock import MockBackend

    bank = KnowledgeBank(tmp_path)
    # recorded metric tiny -> mock re-measure easily clears the margin -> wins.
    bank.save(_prior("wins", sdk_globs=("2.28.*",), backend="mock", metric=1.0,
                     config={"tp_degree": 4, "weights_dtype": "bf16"}))
    # recorded metric enormous -> mock re-measure is far below -> regression.
    bank.save(_prior("regresses", sdk_globs=("2.28.*",), backend="mock",
                     metric=1e12,
                     config={"tp_degree": 4, "weights_dtype": "bf16"}))

    report = revalidate(bank, MockBackend(seed=7), _CANARIES, "2.30.0", Guardrails())

    actions = {o.lesson_id: o.action for o in report.outcomes}
    assert actions["wins"] == "revalidated"
    assert actions["regresses"] == "demoted"
    verified_ids = {l.lesson_id for l in bank.load_all(Tier.VERIFIED)}
    assert verified_ids == {"wins"}


def test_report_summary_smoke():
    r = RevalidationReport(current_sdk="2.30.0", backend="mock")
    assert "0 revalidated" in r.summary()
