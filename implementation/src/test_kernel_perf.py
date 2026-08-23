"""Tests for the perf optimize loop (kernel_perf) and its wiring into the
invent engine.

Theme (the perf analogue of test_kernel_repair): a measured latency TEACHES the
next attempt. The loop is exercised with mock authors — one that "learns" from
perf feedback and gets faster, one that cannot improve, one that breaks
correctness — to prove it CONVERGES when it can, keeps the running-best correct
kernel, and stops HONESTLY otherwise (never a fabricated speedup).

Runnable two ways:
    python -m pytest -q test_kernel_perf.py     # if pytest is installed
    python test_kernel_perf.py                  # standalone fallback runner
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernel_perf import (
    DMA_BLOCKED,
    MEMORY_BOUND,
    SINGLE_ENGINE,
    KernelPerfLoop,
    PerfFeedback,
    classify_bottleneck,
    guidance_for,
)


# A minimal RaceResult stand-in (duck-typed): the loop only reads these fields,
# so we avoid importing invent_engine's heavy RaceResult here (mirrors how
# test_kernel_repair uses the light CompileResult).
@dataclass
class _Race:
    ran: bool = True
    correct: bool = True
    kernel_ms: float = 0.0
    baseline_ms: float = 0.0
    speedup: float = 0.0
    bottleneck: str = "memory_bound"
    roofline_ratio: float = 0.0
    mfu: float = -1.0
    reason: str = ""


# -- bottleneck -> guidance --------------------------------------------------
def test_classify_routes_analytic_bottlenecks():
    assert classify_bottleneck(_Race(bottleneck="memory_bound")) == MEMORY_BOUND
    # compute-bound but still slow -> one engine serializing / PE under-filled.
    assert classify_bottleneck(_Race(bottleneck="compute_bound")) == SINGLE_ENGINE
    # richer profiler signals (if ever present) route first.
    assert classify_bottleneck(_Race(reason="DMA queue saturated")) == DMA_BLOCKED
    assert classify_bottleneck(_Race(bottleneck="", reason="")) == MEMORY_BOUND


def test_guidance_is_one_dominant_fix_per_label():
    for label, needle in ((MEMORY_BOUND, "FUSE"), (SINGLE_ENGINE, "ACTIVATION-FUSE"),
                          (DMA_BLOCKED, "DOUBLE-BUFFER")):
        g = guidance_for(label)
        assert len(g) == 1, f"{label} must surface exactly one dominant fix"
        assert needle in g[0]
    # unknown -> safe default (fusion).
    assert guidance_for("nonsense") == guidance_for(MEMORY_BOUND)


# -- PerfFeedback.as_prompt --------------------------------------------------
def test_perf_feedback_prompt_surfaces_latency_bottleneck_and_guidance():
    fb = PerfFeedback(round=2, kernel_ms=8.0, baseline_ms=1.0, speedup=0.125,
                      bottleneck=MEMORY_BOUND, roofline_ratio=0.004,
                      guidance=guidance_for(MEMORY_BOUND))
    p = fb.as_prompt()
    assert "8.0000 ms" in p and "1.0000 ms" in p     # measured latency vs baseline
    assert "0.125x" in p                             # the speedup gap
    assert MEMORY_BOUND in p                         # the ONE dominant bottleneck
    assert "roofline_ratio=0.0040" in p
    assert "FUSE" in p and "HOIST" in p              # the targeted guidance


# -- KernelPerfLoop: CONVERGES when the author learns from feedback ----------
class _LearningAuthor:
    """Returns a tag naming the current round; the mock measure maps it to a
    faster latency, the way the real LLM author gets faster once it reads the
    measured bottleneck + fix."""

    def __call__(self, trail):
        return f"kernel-r{len(trail)}"


def test_loop_converges_when_author_gets_faster():
    # latency 10ms -> 2ms across rounds; at 2ms the roofline is saturated
    # (mfu >= min_utilization) -> converged.
    schedule = {
        "kernel-r0": _Race(kernel_ms=10.0, baseline_ms=1.0, bottleneck="memory_bound",
                           roofline_ratio=0.02, mfu=0.1),
        "kernel-r1": _Race(kernel_ms=2.0, baseline_ms=1.0, bottleneck="compute_bound",
                           roofline_ratio=1.1, mfu=0.9),
    }

    def measure(kernel):
        return schedule[kernel]

    loop = KernelPerfLoop(max_rounds=6, min_gain_pct=2.0, stall_patience=2)
    out = loop.run(_LearningAuthor(), measure)
    assert out.reason == "converged"
    assert out.ok
    assert out.rounds == 2
    assert out.kernel == "kernel-r1"
    assert abs(out.best_ms - 2.0) < 1e-9
    assert abs(out.speedup - 0.5) < 1e-9             # baseline 1.0 / 2.0ms
    assert out.trajectory == [10.0, 2.0]             # losses AND wins are data


# -- STOPS no_gain when the author cannot improve ----------------------------
def test_loop_stops_no_gain_when_author_cannot_improve():
    # Every round is correct but the SAME latency -> no adopted gain -> bail at
    # stall_patience, honestly, keeping the (first, best) kernel.
    def author(trail):
        return "same"

    def measure(kernel):
        return _Race(kernel_ms=5.0, baseline_ms=1.0, bottleneck="memory_bound",
                     roofline_ratio=0.01, mfu=0.05)

    loop = KernelPerfLoop(max_rounds=8, min_gain_pct=2.0, stall_patience=2)
    out = loop.run(author, measure)
    assert out.reason == "no_gain"
    # round 1 establishes the best; rounds 2 & 3 stall -> bail at round 3 (< 8).
    assert out.rounds == 3 < 8                        # bailed early, did not burn budget
    assert out.kernel == "same"                      # round-1 result kept as best
    assert abs(out.best_ms - 5.0) < 1e-9
    assert not out.ok or out.kernel is not None      # a best was retained


# -- STOPS regressed_or_broke when a round returns incorrect -----------------
def test_loop_stops_regressed_or_broke_and_keeps_last_good():
    # Round 1 is correct + fast (adopted); round 2 comes back INCORRECT -> stop
    # immediately, keep the round-1 kernel (never adopt a broken rewrite).
    seq = iter([
        _Race(correct=True, kernel_ms=4.0, baseline_ms=1.0, mfu=0.1,
              roofline_ratio=0.02),
        _Race(correct=False, kernel_ms=1.0, baseline_ms=1.0, mfu=0.99,
              roofline_ratio=1.5, reason="allclose failed after rewrite"),
    ])

    def author(trail):
        return f"kernel-r{len(trail)}"

    def measure(kernel):
        return next(seq)

    loop = KernelPerfLoop(max_rounds=6, min_gain_pct=2.0, stall_patience=3)
    out = loop.run(author, measure)
    assert out.reason == "regressed_or_broke"
    assert out.rounds == 2
    assert out.kernel == "kernel-r0"                 # last-good kept
    assert out.race is not None and out.race.correct is True


def test_loop_regressed_on_round1_with_seed_keeps_seed():
    # Seeded with a known-good slow kernel; round 1 breaks correctness -> the seed
    # is preserved (the running-best is correct from round 0).
    seed_race = _Race(correct=True, kernel_ms=9.0, baseline_ms=1.0, mfu=0.05,
                      roofline_ratio=0.01)

    def author(trail):
        return "broken"

    def measure(kernel):
        return _Race(ran=True, correct=False, kernel_ms=0.0, baseline_ms=1.0,
                     reason="wrong")

    loop = KernelPerfLoop(max_rounds=4)
    out = loop.run(author, measure, seed_kernel="seed", seed_race=seed_race)
    assert out.reason == "regressed_or_broke"
    assert out.kernel == "seed"
    assert out.race is seed_race


# -- exhausted (evolving but never converging / never stalling long enough) --
def test_loop_exhausts_rounds_when_it_keeps_improving_but_never_saturates():
    # A memory-bound op that shrinks a little every round but never reaches the
    # roofline knee: the loop keeps adopting gains (so never stalls) and runs to
    # the cap, reporting an HONEST 'exhausted' with the best kept.
    lat = {"ms": 10.0}

    def author(trail):
        return f"k{len(trail)}"

    def measure(kernel):
        lat["ms"] *= 0.5                             # halves each round (>2% gain)
        return _Race(kernel_ms=lat["ms"], baseline_ms=1.0, bottleneck="memory_bound",
                     roofline_ratio=0.02, mfu=0.1)

    loop = KernelPerfLoop(max_rounds=3, min_gain_pct=2.0, stall_patience=2)
    out = loop.run(author, measure)
    assert out.reason == "exhausted"
    assert out.rounds == 3
    assert abs(out.best_ms - 1.25) < 1e-9            # 10 -> 5 -> 2.5 -> 1.25
    assert out.trajectory == [5.0, 2.5, 1.25]


# ===========================================================================
# wiring into the invent engine
# ===========================================================================
from bank import KnowledgeBank, LessonType, Tier            # noqa: E402
from invent_engine import InventEngine, RaceResult          # noqa: E402
from invent_kernels import catalog                            # noqa: E402


def _slow_correct(_a, _s) -> RaceResult:
    # correct but 0.10x the baseline -> single-shot would dead-end as anti-pattern.
    return RaceResult(True, correct=True, correctness_pct=100.0, speedup=0.10,
                      kernel_ms=10.0, baseline_ms=1.0, reason="slow")


def test_wiring_default_max_perf_rounds_1_is_unchanged(tmp_path):
    # max_perf_rounds=1 (default): a correct-but-slow kernel is banked as an
    # anti-pattern exactly as today — the perf loop never runs.
    eng = InventEngine(out_dir=tmp_path)
    assert eng.max_perf_rounds == 1
    res = eng.run_op(catalog()["rmsnorm"], race_fn=_slow_correct)
    assert res.status == "anti_pattern"
    assert "perf loop" not in res.detail


def test_wiring_perf_loop_optimizes_correct_but_slow_kernel(tmp_path):
    # max_perf_rounds>1: the SAME correct-but-slow starting point now enters the
    # perf loop (mock measure_fn) and is optimized into a WIN. The mock measure is
    # stateful: call 1 is the seed race (slow), later calls are the loop's
    # per-round re-measure, getting faster until the roofline saturates.
    calls = {"n": 0}
    latencies = [10.0, 5.0, 0.5]                     # seed, round1, round2(fast)

    def measure(_author, _spec):
        i = min(calls["n"], len(latencies) - 1)
        ms = latencies[i]
        calls["n"] += 1
        saturated = ms <= 0.5
        return RaceResult(
            True, correct=True, correctness_pct=100.0,
            speedup=1.0 / ms, kernel_ms=ms, baseline_ms=1.0,
            reason="mock", bottleneck="memory_bound" if not saturated else "compute_bound",
            roofline_ratio=0.02 if not saturated else 1.2,
            mfu=0.1 if not saturated else 0.9)

    eng = InventEngine(out_dir=tmp_path, max_perf_rounds=4)
    res = eng.run_op(catalog()["rmsnorm"], race_fn=measure)
    assert res.status == "win", res.detail
    assert "perf loop" in res.detail                 # the trajectory note is surfaced
    assert calls["n"] >= 3                            # seed + at least two loop rounds
    # the banked lesson is a real invented NKI_KERNEL win at the improved speed.
    lessons = KnowledgeBank(tmp_path / "knowledge-bank").load_all(Tier.PROVISIONAL)
    wins = [l for l in lessons if l.type is LessonType.NKI_KERNEL]
    assert wins, "the optimized kernel should bank an NKI_KERNEL win"


def test_wiring_perf_loop_banks_trajectory_when_still_slow(tmp_path):
    # If the loop cannot make it fast, the best attempt is banked as an
    # anti-pattern WITH the latency trajectory (losses are data).
    def measure(_author, _spec):
        return RaceResult(True, correct=True, correctness_pct=100.0, speedup=0.10,
                          kernel_ms=10.0, baseline_ms=1.0, reason="stuck",
                          bottleneck="memory_bound", roofline_ratio=0.01, mfu=0.05)

    eng = InventEngine(out_dir=tmp_path, max_perf_rounds=4)
    res = eng.run_op(catalog()["rmsnorm"], race_fn=measure)
    assert res.status == "anti_pattern"
    assert "perf loop" in res.detail and "no_gain" in res.detail
    lessons = KnowledgeBank(tmp_path / "knowledge-bank").load_all(Tier.PROVISIONAL)
    ap = [l for l in lessons if l.type is LessonType.ANTI_PATTERN]
    assert ap, "a still-slow perf loop should bank an anti-pattern"
    assert any("perf loop" in l.reason for l in ap)  # trajectory recorded on the lesson


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
