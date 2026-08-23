"""
Tests for the Stage-4 INVENT engine's COMPOUNDING pieces (CPU-only, no Trainium):

  1. Author-time bank RETRIEVAL — the engine now READS the bank before authoring
     (`_retrieve_lessons`), so a previously-banked anti-pattern / prior win for an
     op becomes load-bearing, and `run_op` records how many relevant lessons it
     consulted (on the InventResult and in the ledger row).

  2. Failure DIAGNOSIS with the rewrite catalog — a failure whose reason carries a
     known compiler error signature (e.g. the `.tril`
     `s2d2_ts_as_valid_elem_count` ISA assertion) gets the matching rewrite's
     name+summary appended to the ledger desc AND the banked anti-pattern reason,
     turning an opaque "failed" into an actionable "failed; known fix: <rewrite>".

Runnable two ways:
    python -m pytest -q test_invent_retrieval.py
    python test_invent_retrieval.py
"""

from __future__ import annotations

from pathlib import Path

from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Origin,
    Symptom,
    Tier,
)
from ledger import Layer, Ledger, Stage
from invent_engine import InventEngine, RaceResult
from invent_kernels import author_kernel, catalog


# -- race fns ----------------------------------------------------------------
def _win_race(_a, _s) -> RaceResult:
    return RaceResult(True, correct=True, correctness_pct=100.0, speedup=1.30,
                      kernel_ms=0.70, baseline_ms=0.91, reason="mock win")


def _deferred_race(_a, _s) -> RaceResult:
    return RaceResult(False, reason="off-device: no nki")


def _tril_wrong_race(_a, _s) -> RaceResult:
    # An on-device failure that carries the EXACT ISA assertion a runtime .tril
    # lowers to — the signature the rewrite catalog keys `tril-to-const-mask` on.
    return RaceResult(
        True, correct=False, correctness_pct=8.0, speedup=0.0,
        reason=("device race error: neuronx-cc exit-70; ISA validation failed: "
                "s2d2_ts_as_valid_elem_count assertion on TensorScalarAffineSelect"))


# -- helpers -----------------------------------------------------------------
def _seed_anti_pattern(bank: KnowledgeBank, spec, tier: Tier = Tier.VERIFIED) -> str:
    """Seed a bank with an anti-pattern keyed to `spec`'s op + family + shape."""
    lesson_id = f"antipattern-invented-{spec.name}-{spec.shape_class}"
    bank.save(Lesson(
        lesson_id=lesson_id,
        type=LessonType.ANTI_PATTERN,
        applicability=Applicability(architecture_family=spec.family,
                                    neuron_sdk_versions=["2.28.*"]),
        layer=Layer.KERNEL, migration_risk="low",
        origin=Origin.INVENTED, tier=tier,
        reason=(f"a prior invented {spec.name} ({spec.shape_class}) kernel was "
                f"correct-but-slow; do not re-try the naive formulation"),
        confidence=Confidence(n_models_validated=1, architecture_diversity=1,
                              human_verified=True),
        last_reverified_sdk="2.28.0",
    ))
    return lesson_id


# ===========================================================================
# (1) author-time bank retrieval
# ===========================================================================
def test_retrieve_lessons_returns_seeded_anti_pattern(tmp_path):
    spec = catalog()["softcap"]
    kb = KnowledgeBank(tmp_path / "kb")
    lid = _seed_anti_pattern(kb, spec)

    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    lessons = eng._retrieve_lessons(spec)
    assert [l.lesson_id for l in lessons] == [lid]


def test_retrieve_lessons_empty_bank_is_zero(tmp_path):
    spec = catalog()["softcap"]
    eng = InventEngine(out_dir=tmp_path / "run")
    assert eng._retrieve_lessons(spec) == []


def test_retrieve_lessons_ignores_other_ops(tmp_path):
    # A lesson banked for a DIFFERENT op must not be retrieved for this op.
    other = catalog()["layernorm"]
    kb = KnowledgeBank(tmp_path / "kb")
    _seed_anti_pattern(kb, other)

    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    assert eng._retrieve_lessons(catalog()["softcap"]) == []


def test_run_op_records_lessons_consulted_on_result_and_ledger(tmp_path):
    spec = catalog()["softcap"]
    kb = KnowledgeBank(tmp_path / "kb")
    _seed_anti_pattern(kb, spec)

    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    res = eng.run_op(spec, race_fn=_deferred_race)
    # The count is surfaced on the result...
    assert res.lessons_consulted == 1
    # ...and stamped into the ledger row description.
    rows = [r for r in Ledger(tmp_path / "run").read() if r.stage is Stage.INVENT]
    assert len(rows) == 1
    assert "[lessons:1]" in rows[0].description


def test_run_op_no_lessons_leaves_ledger_desc_unprefixed(tmp_path):
    # With an empty bank, the ledger desc is byte-for-byte the pre-change shape
    # (no "[lessons:N]" prefix) — the change is additive and defaulted.
    eng = InventEngine(out_dir=tmp_path / "run")
    res = eng.run_op(catalog()["softcap"], race_fn=_deferred_race)
    assert res.lessons_consulted == 0
    rows = [r for r in Ledger(tmp_path / "run").read() if r.stage is Stage.INVENT]
    assert "[lessons:" not in rows[0].description


def test_provisional_win_compounds_into_next_retrieval(tmp_path):
    # A win banked (PROVISIONAL) by one run is retrieved by the next run of the
    # same op — the compounding the framework is built on, without waiting for
    # weekly human promotion (verified-only queries would miss it).
    spec = catalog()["softcap"]
    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    win = eng.run_op(spec, race_fn=_win_race)
    assert win.status == "win"

    eng2 = InventEngine(out_dir=tmp_path / "run2", bank_root=tmp_path / "kb")
    ids = [l.lesson_id for l in eng2._retrieve_lessons(spec)]
    assert win.lesson_id in ids


def test_author_kernel_accepts_lessons_kwarg_without_changing_output(tmp_path):
    # The optional hints seam must be inert for the recipe author (defaulted).
    spec = catalog()["softcap"]
    kb = KnowledgeBank(tmp_path / "kb")
    _seed_anti_pattern(kb, spec)
    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    lessons = eng._retrieve_lessons(spec)

    base = author_kernel(spec)
    with_hints = author_kernel(spec, lessons=lessons)
    assert with_hints.nki_src == base.nki_src
    assert with_hints.entry == base.entry


# ===========================================================================
# (2) diagnose failures with the rewrite catalog
# ===========================================================================
def test_diagnose_failure_matches_known_signature(tmp_path):
    eng = InventEngine(out_dir=tmp_path / "run")
    desc_sfx, reason_sfx = eng._diagnose_failure(
        "ISA validation: s2d2_ts_as_valid_elem_count assertion")
    assert "tril-to-const-mask" in desc_sfx
    assert "tril-to-const-mask" in reason_sfx
    # NCC_EVRF029 (the MoE-router sort reject) is now a REAL catalogued signature
    # (topk-sort-to-argmax, added in the sort->argmax PR) -> it must diagnose, not
    # be treated as "unrelated". (This test previously used NCC_EVRF029 as its
    # negative example, which went stale once the signature was catalogued.)
    d_sort, r_sort = eng._diagnose_failure(
        "NCC_EVRF029: Operation sort is not supported on trn2")
    assert "topk-sort-to-argmax" in d_sort and "topk-sort-to-argmax" in r_sort
    # a GENUINELY unmatched / empty error -> no suffix (opaque stays opaque, not mislabeled)
    assert eng._diagnose_failure("some unrelated compiler error QUX_9999") == ("", "")
    assert eng._diagnose_failure("") == ("", "")


def test_on_device_failure_gets_rewrite_appended_to_ledger_and_bank(tmp_path):
    spec = catalog()["softcap"]
    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    res = eng.run_op(spec, race_fn=_tril_wrong_race)
    assert res.status == "anti_pattern"

    # (a) ledger row desc carries the actionable fix name.
    rows = [r for r in Ledger(tmp_path / "run").read() if r.stage is Stage.INVENT]
    assert "tril-to-const-mask" in rows[0].description

    # (b) result detail carries it too.
    assert "tril-to-const-mask" in res.detail

    # (c) the BANKED anti-pattern reason carries the known fix — so the next
    #     author reads it, not just an opaque "incorrect on device".
    lessons = KnowledgeBank(tmp_path / "kb").load_all(Tier.PROVISIONAL)
    ap = [l for l in lessons if l.lesson_id == res.lesson_id]
    assert len(ap) == 1
    assert "tril-to-const-mask" in ap[0].reason
    assert "known fix" in ap[0].reason.lower()


def test_unknown_failure_reason_is_not_mislabeled(tmp_path):
    # A failure with no known signature banks a plain anti-pattern (no fake fix).
    def _plain_wrong(_a, _s):
        return RaceResult(True, correct=False, correctness_pct=30.0, speedup=0.0,
                          reason="device race error: numerical mismatch, no signature")

    spec = catalog()["softcap"]
    eng = InventEngine(out_dir=tmp_path / "run", bank_root=tmp_path / "kb")
    res = eng.run_op(spec, race_fn=_plain_wrong)
    assert res.status == "anti_pattern"
    assert "known fix" not in res.detail.lower()
    lessons = KnowledgeBank(tmp_path / "kb").load_all(Tier.PROVISIONAL)
    ap = [l for l in lessons if l.lesson_id == res.lesson_id][0]
    assert "known fix" not in ap.reason.lower()


# ===========================================================================
# standalone runner (no pytest required)
# ===========================================================================
def _run_standalone() -> int:
    import inspect
    import tempfile
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(fns)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
