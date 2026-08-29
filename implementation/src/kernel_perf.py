"""kernel_perf.py — the iterative author -> measure -> read-latency -> re-author
loop that takes a CORRECT-but-slow NKI kernel and makes it FAST.

This is the perf analogue of ``kernel_repair.py``. The repair loop closes the
CORRECTNESS gap (compile-fail -> read the compiler error -> re-author until it
compiles); this loop closes the PERFORMANCE gap (correct-but-slow -> read the
measured latency + the ONE dominant roofline bottleneck -> re-author until it is
fast, or until the loop can honestly say it cannot do better).

Today's invent_engine authors a correct kernel, races it once, and if it is
correct-but-slow it DEAD-ENDS as an anti-pattern (the 0.08x rmsnorm case). The
mature pattern is a bounded optimize loop: measure, and on a slow result feed the
measured latency-vs-baseline + the dominant bottleneck (from the analytic
roofline signal already on ``RaceResult``) back to the author, so the next
attempt knows WHAT to fix — one dominant lever per round, not a scattershot
rewrite.

Design (mirrors ``kernel_repair``): author and measure are INJECTED functions
(interfaces), so the loop is
    (a) unit-testable with a mock author that "learns" from perf feedback, and
    (b) pluggable with a real LLM/agent author + a real ``_device_race`` measure.
The loop owns only the control flow, the bottleneck->guidance diagnosis, the
running-best bookkeeping, and the HONEST stop conditions — never a fabricated
number and never a claimed speedup it did not measure.

    AuthorFn:  (trail: list[PerfFeedback]) -> kernel   # consumes accumulated perf feedback
    MeasureFn: (kernel) -> RaceResult                  # RE-VALIDATES correctness AND re-measures

Honest stops (mirror kernel_repair — never burn N pointless rounds, never fake a
win):
  * converged          — the running-best kernel saturates the roofline
                         (``mfu >= min_utilization`` or ``roofline_ratio`` at/above
                         the ridge): more rounds cannot meaningfully help.
  * no_gain            — latency did not improve by ``min_gain_pct`` for
                         ``stall_patience`` consecutive rounds (the author cannot
                         make it faster / has no lever left).
  * exhausted          — ``max_rounds`` reached without converging.
  * regressed_or_broke — a round came back INCORRECT (or un-run): stop
                         immediately and keep the last-good kernel. A perf rewrite
                         that breaks correctness is not a candidate.

The running-best is always a CORRECT kernel: a candidate is adopted only when it
is correct AND at least ``min_gain_pct`` faster than the current best
(``kernel_ms < best*(1-min_gain_pct/100)``). Losses are DATA — the full latency
trajectory is kept on the outcome so a still-slow result banks an anti-pattern
WITH its trajectory, not an opaque "did not win".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# bottleneck -> ONE dominant fix (the perf analogue of the rewrite catalog)
# ---------------------------------------------------------------------------
# The three diagnosable slow-kernel states and the single highest-ROI lever for
# each (distilled from the author's _PERF_PREAMBLE / the nki-optimization
# playbook). One dominant fix per round: change one thing, re-measure, keep what
# helped — never a scattershot rewrite the next measurement cannot attribute.
MEMORY_BOUND = "memory_bound"      # spilling intermediates through HBM
SINGLE_ENGINE = "single_engine"    # one engine serializes; PE under-filled
DMA_BLOCKED = "dma_blocked"        # bandwidth/packet-rate bound on the load path
GPSIMD_BOUND = "gpsimd_bound"      # indirect/dynamic gather serializing on GpSimd

_GUIDANCE: dict[str, list[str]] = {
    GPSIMD_BOUND: [
        "MAKE THE ADDRESSING STATIC / express the gather as a MATMUL. GpSimd is "
        "serializing an indirect (data-dependent) gather on its slow integrated DMA "
        "(~153 GB/s/dir) while TensorE sits idle. If the access pattern is known at "
        "trace time, compute the indices on the HOST and bake them as compile-time "
        "constants so every DMA is static-addressed; and if the gather is linear "
        "(bilinear/interpolate/upsample/one-hot), rewrite it as `out = W @ x` with a "
        "precomputed mostly-zero weight matrix so the work runs on TensorE (the "
        "strongest engine), not GpSimd. Applies ONLY to STATIC access patterns — "
        "not data-dependent gathers (KV-paging, MoE token dispatch)."],
    MEMORY_BOUND: [
        "FUSE the whole op into ONE kernel and HOIST loop-invariant loads "
        "(gamma/beta/cap/row-max) out of the tile loop: a memory-bound op that "
        "spills a temporary through HBM is pure loss (there is no HW cache). Keep "
        "intermediates in SBUF — ONE load per input, ONE store of the output — and "
        "load each invariant once, resident, reused every tile."],
    SINGLE_ENGINE: [
        "ACTIVATION-FUSE and OVERLAP the engines: collapse the elementwise+reduce "
        "onto the Scalar engine via nisa.activation(op=, bias=, scale=, "
        "reduce_op=nl.add) (one instruction, not a materialized temp + a separate "
        "reduce), then route reduce->Scalar, apply->Vector, and broadcast via a "
        "TensorE matmul-against-ones so no single engine serializes the pipeline."],
    DMA_BLOCKED: [
        "DOUBLE-BUFFER the tile loop so tile n+1's DMA overlaps tile n's compute "
        "(buffer rotation), driving latency toward max(compute, dma) instead of "
        "compute + dma; widen tiles (free dim >= 512, bf16 >= 1024) so every DMA "
        "moves >= 2 KiB/partition and all 16 DMA engines stay busy."],
}


def guidance_for(bottleneck: str) -> list[str]:
    """The ONE dominant fix for a bottleneck label. Unknown/empty -> the
    memory-bound fix (fusion is the safe default lever for the elementwise/norm
    ops this engine authors, which are memory-bound by construction)."""
    return _GUIDANCE.get(bottleneck or "", _GUIDANCE[MEMORY_BOUND])


def specific_guidance(bottleneck: str, *, reason: str = "", op_name: str = "",
                      op_family: str = "", max_hints: int = 2) -> list[str]:
    """The SPECIFIC NKI-Guide optimizations (Opt #5a/#5b/#7/#8/…) for a measured
    symptom, via ``perf_hints`` — the fine-grained lever the compiler misses (fast
    weight load for a short-dim matmul, tensor_tensor_scan for a per-element scan,
    partition vectorization, transpose-swap). Falls back to the coarse
    ``guidance_for`` lever when nothing specific matches, so the loop always has a
    fix. Never raises (a missing perf_hints module degrades to coarse guidance)."""
    try:
        import perf_hints  # noqa: PLC0415 — optional, self-contained
        hints = perf_hints.guidance_from_symptoms(
            bottleneck, reason=reason, op_name=op_name, op_family=op_family,
            max_hints=max_hints)
        if hints:
            return hints
    except Exception:  # noqa: BLE001
        pass
    return guidance_for(bottleneck)


def classify_bottleneck(race: Any) -> str:
    """Reduce a measured ``RaceResult`` to ONE dominant, actionable bottleneck
    label among {memory_bound, single_engine, dma_blocked}.

    The analytic roofline (``race.bottleneck`` in {"memory_bound","compute_bound"}
    + ``race.roofline_ratio``) is the primary signal; a richer profiler signal, if
    ever present in ``race.bottleneck`` / ``race.reason`` (e.g. it names "dma",
    "spill", or "single-engine"), is honored first so the loop routes to the right
    lever without waiting on a real profiler. Never raises; defaults to
    memory_bound (fusion) when nothing is known."""
    hay = f"{getattr(race, 'bottleneck', '')} {getattr(race, 'reason', '')}".lower()
    # gpsimd / indirect-gather FIRST — its reason contains "gpsimd-bound" and would
    # otherwise be swallowed by the "dma"/"engine" checks below (wrong fix). Match
    # the LABEL token "gpsimd-bound" (not bare "gpsimd") so a single-engine reason
    # that merely lists "GPSIMD 0%" does not misroute here.
    if "gpsimd-bound" in hay or "gpsimd_bound" in hay or "indirect" in hay:
        return GPSIMD_BOUND
    if "dma" in hay or "bandwidth" in hay:
        return DMA_BLOCKED
    if "single" in hay or "serial" in hay or "engine" in hay:
        return SINGLE_ENGINE
    if "spill" in hay:
        return MEMORY_BOUND
    # Fall back to the analytic roofline classification.
    bn = (getattr(race, "bottleneck", "") or "").lower()
    if bn == "compute_bound":
        # Compute-bound but still slow -> the compute engine is under-filled /
        # one engine is serializing the pipeline: fuse + overlap engines.
        return SINGLE_ENGINE
    return MEMORY_BOUND


# ---------------------------------------------------------------------------
# feedback + outcome records
# ---------------------------------------------------------------------------
@dataclass
class PerfFeedback:
    """One measured-but-still-slow round, fed back to the author for the next
    attempt. The perf analogue of ``kernel_repair.Feedback``: it surfaces the
    measured latency-vs-baseline, the ONE dominant bottleneck, and the single
    targeted fix — never an opaque 'make it faster'."""

    round: int
    kernel_ms: float
    baseline_ms: float
    speedup: float                 # baseline_ms / kernel_ms; >1 == faster than eager
    bottleneck: str                # dominant label: memory_bound | single_engine | dma_blocked
    roofline_ratio: float          # arithmetic_intensity / bf16 ridge (<1 memory-bound)
    guidance: list[str] = field(default_factory=list)
    prev_src: str = ""             # the source that produced this latency (context for the rewrite)

    def as_prompt(self) -> str:
        """The actionable message the author consumes on the next round: measured
        latency vs baseline + the ONE dominant bottleneck + the targeted fix."""
        gap = (f"{self.speedup:.3f}x the baseline" if self.speedup > 0
               else "no valid speedup measured")
        lines = [
            f"Round {self.round}: measured {self.kernel_ms:.4f} ms vs baseline "
            f"{self.baseline_ms:.4f} ms ({gap}) — still too slow.",
            f"Dominant bottleneck: {self.bottleneck} "
            f"(roofline_ratio={self.roofline_ratio:.4f}).",
            "Apply THIS one dominant fix and re-author (change one lever, then "
            "re-measure — do not rewrite everything at once):",
        ]
        lines += [f"  - {g}" for g in (self.guidance or guidance_for(self.bottleneck))]
        return "\n".join(lines)


@dataclass
class PerfOutcome:
    ok: bool
    rounds: int
    reason: str                    # "converged" | "no_gain" | "exhausted" | "regressed_or_broke"
    kernel: Any = None             # the running-BEST (correct) kernel
    race: Any = None               # its measured RaceResult (the best one)
    baseline_ms: float = 0.0
    best_ms: float = 0.0
    speedup: float = 0.0           # baseline_ms / best_ms for the kept kernel
    trail: list[PerfFeedback] = field(default_factory=list)
    # Per-round measured kernel latency (ms), oldest first, INCLUDING the seed at
    # index 0 when the loop was seeded — the trajectory a still-slow anti-pattern
    # is banked WITH (losses are data).
    trajectory: list[float] = field(default_factory=list)

    @property
    def trajectory_str(self) -> str:
        """Human-readable latency trajectory, e.g. '10.000 -> 5.000 -> 2.000 ms'."""
        if not self.trajectory:
            return "(no measured rounds)"
        return " -> ".join(f"{ms:.4f}" for ms in self.trajectory) + " ms"


AuthorFn = Callable[[list["PerfFeedback"]], Any]
MeasureFn = Callable[[Any], Any]          # kernel -> RaceResult (re-validate + re-measure)


class KernelPerfLoop:
    """Bounded author -> measure -> diagnose -> re-author loop that makes a
    correct kernel fast, keeping the running-best correct kernel and stopping
    HONESTLY (never a fabricated speedup)."""

    def __init__(self, max_rounds: int = 6, min_gain_pct: float = 2.0,
                 stall_patience: int = 2, min_utilization: float = 0.85,
                 op_name: str = "", op_family: str = "") -> None:
        # min_gain_pct: a candidate must be at least this much faster than the
        #   running-best to be ADOPTED (below it is noise — mirrors
        #   guardrails.marginal_improvement_pct).
        # stall_patience: this many consecutive rounds with no adopted gain -> bail.
        # min_utilization: the roofline saturation bar for "converged" (mirrors
        #   guardrails.min_utilization); at/above it, more rounds cannot help.
        # op_name/op_family: OPTIONAL op context so ``diagnose`` can route a
        #   shape-specific NKI-Guide optimization (fast weight load for a decode
        #   matmul, tensor_tensor_scan for a scan) via perf_hints — safe to omit.
        self.max_rounds = max_rounds
        self.min_gain_pct = min_gain_pct
        self.stall_patience = stall_patience
        self.min_utilization = min_utilization
        self.op_name = op_name
        self.op_family = op_family

    def diagnose(self, race: Any) -> tuple[str, list[str]]:
        """Measured result -> (dominant bottleneck, ONE targeted fix). The
        retrieval that turns a bare latency into a named, actionable lever.

        Prefers the SPECIFIC NKI-Guide optimization for the measured symptom
        (``specific_guidance`` via perf_hints — reads the ``reason`` string's
        threshold-breach tokens + the op shape), and falls back to the coarse
        per-bottleneck lever when nothing specific matches."""
        bn = classify_bottleneck(race)
        guidance = specific_guidance(
            bn, reason=str(getattr(race, "reason", "")),
            op_name=self.op_name, op_family=self.op_family)
        return bn, guidance

    def _converged(self, race: Any) -> bool:
        """The running-best saturates the roofline -> optimizing further is not
        worth a round. True when model-FLOPs-utilization clears ``min_utilization``
        OR the measured %SOL clears the NKI-Guide's per-bottleneck 'good' bar (90%
        compute-bound / 60% memory-bound — so a memory-bound op is judged CONVERGED
        at the bar it can actually reach, not held to a 90% bar it never will) OR
        the op is at/above the compute ridge (``roofline_ratio >= 1``)."""
        mfu = getattr(race, "mfu", -1.0)
        if mfu >= self.min_utilization:
            return True
        sol = float(getattr(race, "sol", 0.0) or 0.0)
        bn = (getattr(race, "bottleneck", "") or "").lower()
        if sol > 0.0:
            try:
                import roofline  # noqa: PLC0415
                if roofline.meets_good_bar(sol, bn):
                    return True
            except Exception:  # noqa: BLE001
                pass
        return getattr(race, "roofline_ratio", 0.0) >= 1.0

    def run(self, author_fn: AuthorFn, measure_fn: MeasureFn,
            seed_kernel: Any = None, seed_race: Any = None) -> PerfOutcome:
        """Drive the optimize loop. Optionally SEEDED with a known-good
        (kernel, race) — the correct-but-slow kernel the engine already measured —
        so the running-best is a real correct kernel from round 0 and a round-1
        regression still keeps that seed."""
        trail: list[PerfFeedback] = []
        best_kernel = seed_kernel
        best_race = seed_race
        best_ms = float(getattr(seed_race, "kernel_ms", 0.0)) if seed_race else float("inf")
        if best_ms <= 0.0:
            best_ms = float("inf")
        baseline_ms = float(getattr(seed_race, "baseline_ms", 0.0)) if seed_race else 0.0
        trajectory: list[float] = ([best_ms] if seed_race and best_ms != float("inf")
                                   else [])
        no_gain = 0

        for rnd in range(1, self.max_rounds + 1):
            kernel = author_fn(trail)          # author sees ALL prior perf feedback
            race = measure_fn(kernel)          # RE-VALIDATE correctness + RE-MEASURE

            # regressed_or_broke: a perf rewrite that is no longer correct (or did
            # not even run) is not a candidate — stop and keep the last-good kernel.
            if not getattr(race, "ran", False) or not getattr(race, "correct", False):
                return PerfOutcome(
                    ok=best_kernel is not None, rounds=rnd,
                    reason="regressed_or_broke", kernel=best_kernel, race=best_race,
                    baseline_ms=baseline_ms, best_ms=(0.0 if best_ms == float("inf")
                                                      else best_ms),
                    speedup=_speedup(baseline_ms, best_ms), trail=trail,
                    trajectory=trajectory)

            kernel_ms = float(getattr(race, "kernel_ms", 0.0))
            if baseline_ms <= 0.0:
                baseline_ms = float(getattr(race, "baseline_ms", 0.0))
            trajectory.append(kernel_ms)

            # Keep-gate: adopt ONLY a candidate that is correct AND at least
            # min_gain_pct faster than the running-best.
            improved = kernel_ms > 0.0 and kernel_ms < best_ms * (1.0 - self.min_gain_pct / 100.0)
            if improved:
                best_kernel, best_race, best_ms = kernel, race, kernel_ms
                no_gain = 0
            else:
                no_gain += 1

            # converged: the (best) kernel saturates the roofline — done, honestly.
            if self._converged(race) or (best_race is not None and self._converged(best_race)):
                return PerfOutcome(
                    ok=True, rounds=rnd, reason="converged",
                    kernel=best_kernel, race=best_race, baseline_ms=baseline_ms,
                    best_ms=best_ms, speedup=_speedup(baseline_ms, best_ms),
                    trail=trail, trajectory=trajectory)

            # no_gain: no adopted improvement for stall_patience rounds -> bail
            # rather than burn the budget (the author has no lever left).
            if no_gain >= self.stall_patience:
                return PerfOutcome(
                    ok=best_kernel is not None, rounds=rnd, reason="no_gain",
                    kernel=best_kernel, race=best_race, baseline_ms=baseline_ms,
                    best_ms=(0.0 if best_ms == float("inf") else best_ms),
                    speedup=_speedup(baseline_ms, best_ms), trail=trail,
                    trajectory=trajectory)

            # Still slow but making (or capable of) progress: diagnose the ONE
            # dominant lever and feed it back for the next round.
            bn, guidance = self.diagnose(race)
            trail.append(PerfFeedback(
                round=rnd, kernel_ms=kernel_ms, baseline_ms=baseline_ms,
                speedup=_speedup(baseline_ms, kernel_ms), bottleneck=bn,
                roofline_ratio=float(getattr(race, "roofline_ratio", 0.0)),
                guidance=guidance,
                prev_src=getattr(kernel, "nki_src", "") or ""))

        return PerfOutcome(
            ok=best_kernel is not None, rounds=self.max_rounds, reason="exhausted",
            kernel=best_kernel, race=best_race, baseline_ms=baseline_ms,
            best_ms=(0.0 if best_ms == float("inf") else best_ms),
            speedup=_speedup(baseline_ms, best_ms), trail=trail,
            trajectory=trajectory)


def _speedup(baseline_ms: float, kernel_ms: float) -> float:
    """baseline/kernel, or 0.0 when either side is not a real measurement — never
    a fabricated ratio."""
    if baseline_ms <= 0.0 or kernel_ms <= 0.0 or kernel_ms == float("inf"):
        return 0.0
    return baseline_ms / kernel_ms
