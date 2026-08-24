"""Tests for ``nki_selfimprove`` — the lesson-bank + self-improvement loop
(Pillar 2/3). All CPU-only, no Trainium, no model: authoring is the echo
provider and the on-device race is a deterministic injected ``race_fn``.

Coverage:
  * ``LessonBank`` round-trips a lesson to disk and back; a corrupt/absent file
    yields a fresh lesson (never raises).
  * ``update`` computes the running best among CORRECT kernels, sets
    ``rounds_to_correct`` on the first correct iter, records ``first_iter_compiled``,
    and banks FAILED approaches with their class.
  * ``render_for_prompt`` is EMPTY before anything is banked (so iter-1 is the
    pure static-knowledge baseline) and, once banked, surfaces the best speedup
    ("beat"), the winning template, and each failed approach.
  * RETRIEVAL: ``SelfImproveEngine._retrieve_lessons`` injects the rendered
    self-improve lesson (as a prompt-seam lesson) on top of the base engine's
    lessons — and injects NOTHING on the first attempt.
  * ``approach_fingerprint`` / ``classify_outcome`` are stable and bucket the
    obvious cases.
  * end-to-end ``run_selfimprove`` with a mock race that improves over iters
    produces a monotone best-speedup curve, sets the summary fields, and stops
    honestly on plateau.

Runnable two ways:
    python -m pytest -q test_nki_selfimprove.py
    python test_nki_selfimprove.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import nki_selfimprove as SI
from invent_kernels import resolve_ops
from invent_engine import InventResult, RaceResult, OfflineGate


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rec(it, *, correct, speedup, compiled=True, status="anti_pattern",
         approach="a", cls="correct_but_slow", reason="r"):
    return SI.IterationRecord(
        iteration=it, status=status, compiled=compiled, correct=correct,
        speedup=speedup, correctness_pct=100.0 if correct else 0.0, rounds=1,
        outcome_class=cls, approach_sig=approach,
        approach_changed_from_prev=False, lesson_injected=False,
        lessons_consulted=0, reason=reason)


# ---------------------------------------------------------------------------
# LessonBank persistence + merge
# ---------------------------------------------------------------------------
def test_bank_roundtrip_and_missing_file():
    with tempfile.TemporaryDirectory() as d:
        bank = SI.LessonBank(d)
        # Missing file -> fresh empty lesson, no raise.
        fresh = bank.get("softmax")
        assert fresh.op == "softmax" and fresh.n_attempts == 0
        assert fresh.render_for_prompt() == ""  # nothing banked -> no injection
        # Save + reload.
        fresh.best_speedup = 0.5
        fresh.n_attempts = 1
        bank.save(fresh)
        again = bank.get("softmax")
        assert again.best_speedup == 0.5 and again.n_attempts == 1


def test_bank_corrupt_file_is_survivable():
    with tempfile.TemporaryDirectory() as d:
        bank = SI.LessonBank(d)
        (Path(d) / "softmax.json").write_text("{not valid json")
        lesson = bank.get("softmax")  # must not raise
        assert lesson.op == "softmax" and lesson.n_attempts == 0


def test_update_tracks_best_rounds_and_failures():
    with tempfile.TemporaryDirectory() as d:
        bank = SI.LessonBank(d)
        # iter1: compiled but WRONG -> failed approach, no best, no rounds_to_correct
        bank.update("softmax", "sc", _rec(1, correct=False, speedup=0.0,
                    status="anti_pattern", cls="incorrect", approach="v1"))
        l1 = bank.get("softmax")
        assert l1.first_iter_compiled is True
        assert l1.rounds_to_correct is None
        assert l1.best_speedup is None
        assert len(l1.failed_approaches) == 1
        assert l1.failed_approaches[0]["outcome_class"] == "incorrect"
        # iter2: CORRECT @0.4x -> sets best + rounds_to_correct=2, banks template
        bank.update("softmax", "sc", _rec(2, correct=True, speedup=0.4,
                    approach="v2"), kernel_src="SRC-V2", entry="softmax_kernel")
        l2 = bank.get("softmax")
        assert l2.rounds_to_correct == 2
        assert l2.best_speedup == 0.4 and l2.best_correct is True
        assert l2.best_kernel_src == "SRC-V2"
        assert l2.best_approach_sig == "v2"
        # iter3: CORRECT but SLOWER @0.3x -> best stays 0.4, template unchanged
        bank.update("softmax", "sc", _rec(3, correct=True, speedup=0.3,
                    approach="v3"), kernel_src="SRC-V3")
        l3 = bank.get("softmax")
        assert l3.best_speedup == 0.4 and l3.best_kernel_src == "SRC-V2"
        # iter4: CORRECT + FASTER @0.9x -> new best + new template
        bank.update("softmax", "sc", _rec(4, correct=True, speedup=0.9,
                    approach="v4"), kernel_src="SRC-V4")
        l4 = bank.get("softmax")
        assert l4.best_speedup == 0.9 and l4.best_kernel_src == "SRC-V4"
        assert l4.rounds_to_correct == 2  # unchanged — first correct was iter2
        assert l4.n_attempts == 4


# ---------------------------------------------------------------------------
# render_for_prompt content
# ---------------------------------------------------------------------------
def test_render_surfaces_best_template_and_failures():
    with tempfile.TemporaryDirectory() as d:
        bank = SI.LessonBank(d)
        bank.update("softmax", "sc", _rec(1, correct=False, speedup=0.0,
                    cls="compile:matmul_moving", approach="bad",
                    reason="nc_matmul moving free dimension 4096 exceeds max 512"))
        bank.update("softmax", "sc", _rec(2, correct=True, speedup=0.42,
                    approach="good"),
                    kernel_src="import nki\n@nki.jit\ndef softmax_kernel(x):\n    return x",
                    entry="softmax_kernel")
        text = bank.render_for_prompt("softmax")
        assert "beat 0.420x" in text.lower()
        assert "first correct at iteration 2" in text.lower()
        assert "winning template" in text.lower()
        assert "softmax_kernel" in text  # the template excerpt is present
        assert "FAILED approaches" in text
        assert "compile:matmul_moving" in text


def test_render_empty_before_any_attempt():
    with tempfile.TemporaryDirectory() as d:
        bank = SI.LessonBank(d)
        assert bank.render_for_prompt("rmsnorm") == ""


# ---------------------------------------------------------------------------
# RETRIEVAL — the load-bearing wiring: lesson injected into authoring
# ---------------------------------------------------------------------------
def test_retrieve_injects_selfimprove_lesson_but_not_on_first_attempt():
    from kernel_providers import echo_complete_fn
    from kernel_author import LLMAuthor
    with tempfile.TemporaryDirectory() as d:
        bank = SI.LessonBank(Path(d) / "lessons")
        engine = SI.SelfImproveEngine(Path(d) / "run", lesson_bank=bank,
                                      author=LLMAuthor(echo_complete_fn))
        spec = resolve_ops(["softmax"])[0]
        # First attempt: bank empty -> NO self-improve lesson injected.
        lessons = engine._retrieve_lessons(spec)
        assert engine._last_selfimprove_injected is False
        assert not any(getattr(l, "lesson_id", "").startswith("selfimprove-")
                       for l in lessons)
        # Bank a result, then retrieve again -> the lesson IS injected.
        bank.update("softmax", spec.shape_class,
                    _rec(1, correct=True, speedup=0.5, approach="v1"),
                    kernel_src="SRC", entry="softmax_kernel")
        lessons2 = engine._retrieve_lessons(spec)
        assert engine._last_selfimprove_injected is True
        inj = [l for l in lessons2
               if getattr(l, "lesson_id", "").startswith("selfimprove-")]
        assert len(inj) == 1
        assert "beat 0.500x" in inj[0].reason.lower()


def test_injected_lesson_reaches_the_author_prompt():
    # The rendered lesson must actually appear in build_author_prompt output
    # (it rides the existing `lessons` seam via _fmt_lessons).
    from kernel_author import build_author_prompt
    spec = resolve_ops(["softmax"])[0]
    pl = SI._PromptLesson(lesson_id="selfimprove-softmax",
                          reason="SELF-IMPROVEMENT MEMORY — beat 0.500x")
    prompt = build_author_prompt(spec, [pl], [])
    assert "SELF-IMPROVEMENT MEMORY" in prompt
    assert "beat 0.500x" in prompt


# ---------------------------------------------------------------------------
# fingerprint + classification
# ---------------------------------------------------------------------------
def test_approach_fingerprint_detects_change():
    a = "import nki\nfor i in range(4):\n  e = nl.exp(x)\n  s = nl.sum(e, axis=1, keepdims=True)"
    b = "import nki\ne = nisa.activation(nl.exp, x, reduce_op=nl.add)"
    fa, fb = SI.approach_fingerprint(a), SI.approach_fingerprint(b)
    assert fa != fb
    assert "nl.exp" in fa and "nl.sum" in fa and "loops=1" in fa
    assert "nisa.activation" in fb and "reduce_op" in fb and "loops=0" in fb


def test_classify_outcome_buckets():
    assert SI.classify_outcome("win", "correct=True speedup=1.3x") == "win"
    assert SI.classify_outcome("anti_pattern",
                               "correct but 0.4x (< 5% margin)") == "correct_but_slow"
    assert SI.classify_outcome("anti_pattern",
                               "WRONG on device (12% within tol)") == "incorrect"
    assert SI.classify_outcome("device_deferred", "x") == "device_deferred"
    assert SI.classify_outcome("no_author", "x") == "no_author"


# ---------------------------------------------------------------------------
# end-to-end loop with a mock race (no device, no model)
# ---------------------------------------------------------------------------
def _mk_result(spec, *, correct, speedup, ran=True, status=None, reason=""):
    race = RaceResult(ran, correct=correct, correctness_pct=100.0 if correct else 0.0,
                      speedup=speedup, reason=reason)
    st = status or ("win" if (correct and speedup >= 1.05) else
                    "anti_pattern" if ran else "device_deferred")
    return InventResult(spec.name, spec.shape_class, spec.origin, st,
                        OfflineGate(True, False, 0.0), race, detail=reason)


def test_run_selfimprove_curve_and_plateau():
    from kernel_providers import echo_complete_fn
    from kernel_author import LLMAuthor
    spec = resolve_ops(["softmax"])[0]
    # A scripted race: iter1 wrong, iter2 correct@0.4x, iter3 correct@0.8x,
    # then flat @0.8x -> should plateau-stop after K=2 stale iters.
    plan = {1: (False, 0.0), 2: (True, 0.4), 3: (True, 0.8),
            4: (True, 0.8), 5: (True, 0.8), 6: (True, 0.8)}
    calls = {"n": 0}

    def race_fn(kernel, sp):
        calls["n"] += 1
        c, s = plan[calls["n"]]
        return _mk_result(sp, correct=c, speedup=s).race

    with tempfile.TemporaryDirectory() as d:
        res = SI.run_selfimprove(
            "softmax", iters=6, out_dir=Path(d) / "run",
            author=LLMAuthor(echo_complete_fn), race_fn=race_fn,
            no_improve_stop_k=2, log=lambda *_: None)
    traj = res["trajectory"]
    # best-speedup curve is monotone non-decreasing and never fabricated.
    bests = [r["best_speedup_so_far"] for r in traj]
    assert bests[0] is None            # iter1 wrong -> no speedup
    assert res["summary"]["rounds_to_correct"] == 2
    assert res["summary"]["best_speedup"] == 0.8
    assert res["summary"]["best_iter"] == 3
    # plateau stop: stopped before all 6 iters (2 stale after iter3).
    assert res["iters_run"] == 5
    assert "plateau" in res["stop_reason"]
    assert res["summary"]["first_try_compiled"] is True


def test_run_selfimprove_never_fabricates_on_deferred():
    from kernel_providers import echo_complete_fn
    from kernel_author import LLMAuthor
    spec = resolve_ops(["rmsnorm"])[0]

    def race_fn(kernel, sp):
        return RaceResult(False, reason="no Neuron device — deferred")

    with tempfile.TemporaryDirectory() as d:
        res = SI.run_selfimprove(
            "rmsnorm", iters=3, out_dir=Path(d) / "run",
            author=LLMAuthor(echo_complete_fn), race_fn=race_fn,
            no_improve_stop_k=5, log=lambda *_: None)
    for r in res["trajectory"]:
        assert r["speedup"] is None          # never fabricated
        assert r["compiled"] is False
        assert r["correct"] is False
    assert res["summary"]["best_speedup"] is None


# ---------------------------------------------------------------------------
# runnable without pytest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
