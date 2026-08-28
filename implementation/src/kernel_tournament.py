# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""kernel_tournament.py — TOURNAMENT authoring for the hard ops.

The on-device finding: the from-scratch flash kernel stalled on a SINGLE author
attempt. One shot at a hard op is exactly what fails — the solution space (layout,
decomposition, fusion boundaries) is wide and one framing rarely lands it. The
fix is BREADTH: author N candidates from DIVERSE strategies in parallel, race them
all through the real correctness+speed gate, and keep the winner. Budget is cheap;
a wasted losing candidate costs one compile, and the winner is measurably the best
of N instead of the only one tried.

Diversity comes from the STRATEGY, not from sampling temperature (the Bedrock
Opus-5 config runs ``temperature=None``, so re-sampling the same prompt is not a
reliable diversity source). Each strategy PREPENDS a distinct directive to the
author prompt — "layout-first" vs "memory-first" vs "decompose-first" vs
"recurrence-first" — biasing the model toward a genuinely different approach. This
is deterministic and unit-testable.

Design (composes with every existing seam):
  * ``Strategy``               — a named prompt-directive (the approach bias).
  * ``strategy_authors``       — build one ``LLMAuthor`` per strategy from a shared
                                 ``complete_fn`` (each wired with a directive-prepending
                                 prompt fn; composes with kernel_compose's block).
  * ``Tournament.run``         — race a list of (name, author) through an injected
                                 ``measure_fn`` (the engine's ``_device_race``), pick
                                 the correct + fastest. PURE orchestration, testable
                                 with mock authors + a mock measure.
  * ``TournamentAuthor``       — a ``KernelAuthor`` that runs the tournament on the
                                 FIRST round (feedback empty) to pick the approach,
                                 then delegates repair rounds to the winning
                                 strategy's author (so repair iterates the winner,
                                 not the whole bracket every round).

Honesty: the tournament NEVER fabricates a winner. A candidate is eligible only if
its ``measure_fn`` result ``ran`` AND ``correct``; the winner is the eligible
candidate with the highest measured speedup (ties broken by lower ``kernel_ms``).
If none are correct, the tournament returns the best-effort candidate (the one that
at least ran) so the engine's normal gate still records an honest anti-pattern —
it does not invent a passing kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from invent_kernels import AuthoredKernel, OpSpec

MeasureFn = Callable[[AuthoredKernel, OpSpec], Any]   # (kernel, spec) -> RaceResult-like


@dataclass(frozen=True)
class Strategy:
    """One authoring approach — a named prompt-directive that biases the model
    toward a distinct kernel structure."""
    name: str
    directive: str


# Diverse framings for hard ops. Each is a real, different lever an expert would
# try; together they cover the axes a single prompt cannot explore at once.
DEFAULT_STRATEGIES: list[Strategy] = [
    Strategy("decompose-first",
             "APPROACH: compose the kernel from the smallest verified idioms "
             "(online-softmax step, tiled PSUM matmul, KV-tile loop). Assemble, "
             "do not reinvent. Prefer clarity of the composition over cleverness."),
    Strategy("layout-first",
             "APPROACH: choose the tiling/layout FIRST so the PE (TensorE) stays "
             "busy — partition=128, moving free-dim <=512, feed already-[K,N] "
             "operands into nc_matmul to avoid on-the-fly transposes. Structure "
             "the loops around keeping the systolic array full."),
    Strategy("memory-first",
             "APPROACH: minimize HBM traffic above all — ONE load per input, ONE "
             "store of the output, every intermediate resident in SBUF, invariant "
             "operands hoisted out of the tile loop. Treat a spilled temporary as "
             "a bug. Fuse the whole op into one kernel."),
    Strategy("recurrence-first",
             "APPROACH: for a scan/linear-attention op, carry the recurrent state "
             "in SBUF and iterate the sequence with a SEQUENTIAL range (this beat "
             "the chunked formulation on-device for GatedDeltaNet); update the "
             "state one step/tile at a time — do NOT force a chunked parallel scan."),
]


@dataclass
class Candidate:
    """One tournament entrant: the strategy that produced it, the authored kernel,
    and its measured race (None if it was never measured)."""
    strategy: str
    kernel: AuthoredKernel
    race: Any = None

    @property
    def eligible(self) -> bool:
        """Correct AND actually ran — the only kind of candidate that can win."""
        return bool(getattr(self.race, "ran", False)
                    and getattr(self.race, "correct", False))

    @property
    def speedup(self) -> float:
        return float(getattr(self.race, "speedup", 0.0) or 0.0)

    @property
    def kernel_ms(self) -> float:
        ms = float(getattr(self.race, "kernel_ms", 0.0) or 0.0)
        return ms if ms > 0 else float("inf")


@dataclass
class TournamentResult:
    """The bracket outcome: the chosen winner (best correct candidate, or best-
    effort if none were correct), all candidates, and an honest one-line reason."""
    winner: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)
    reason: str = ""

    @property
    def kernel(self) -> AuthoredKernel | None:
        return self.winner.kernel if self.winner else None

    @property
    def race(self) -> Any:
        return self.winner.race if self.winner else None


def _pick_winner(candidates: list[Candidate]) -> tuple[Candidate | None, str]:
    """The correct candidate with the highest speedup (ties -> lower kernel_ms);
    if none are correct, the best-effort one that at least ran; else None."""
    eligible = [c for c in candidates if c.eligible]
    n_ok = len(eligible)
    if eligible:
        best = max(eligible, key=lambda c: (c.speedup, -c.kernel_ms))
        return best, (f"{n_ok}/{len(candidates)} candidates correct; winner "
                      f"'{best.strategy}' at {best.speedup:.3f}x")
    ran = [c for c in candidates if getattr(c.race, "ran", False)]
    if ran:
        # No correct candidate — hand back one that ran so the engine records an
        # honest anti-pattern (never a fabricated pass).
        return ran[0], (f"0/{len(candidates)} candidates correct — best-effort "
                        f"'{ran[0].strategy}' returned for honest gating")
    if candidates:
        return candidates[0], (f"0/{len(candidates)} candidates measured — "
                               f"'{candidates[0].strategy}' returned unmeasured")
    return None, "no candidates authored"


class Tournament:
    """Race N (name, author) entrants through a shared ``measure_fn`` and pick the
    best. Pure orchestration — the authors and the measure are injected, so this
    is fully unit-testable and the real device work lives in the injected
    ``measure_fn`` (the engine's ``_device_race``)."""

    def run(self, authors: list[tuple[str, Any]], measure_fn: MeasureFn,
            spec: OpSpec, lessons: list | None = None) -> TournamentResult:
        candidates: list[Candidate] = []
        for name, author in authors:
            try:
                kernel = author.author(spec, lessons, [], None)
            except Exception:  # noqa: BLE001 — a broken author just drops out of the bracket
                continue
            race = None
            try:
                race = measure_fn(kernel, spec)
            except Exception:  # noqa: BLE001 — a race that crashes is a losing candidate
                race = None
            candidates.append(Candidate(strategy=name, kernel=kernel, race=race))
        winner, reason = _pick_winner(candidates)
        return TournamentResult(winner=winner, candidates=candidates, reason=reason)


# ---------------------------------------------------------------------------
# strategy-variant authors
# ---------------------------------------------------------------------------
def _strategy_prompt_fn(directive: str, base: Callable[..., str]
                        ) -> Callable[..., str]:
    """A ``build_prompt`` that prepends ``directive`` to the base prompt."""
    def build_prompt(spec, lessons, feedback, perf_feedback=None, **kwargs) -> str:
        body = base(spec, lessons, feedback, perf_feedback, **kwargs)
        return f"{directive}\n\n{body}"
    return build_prompt


def strategy_authors(complete_fn: Callable[[str], str],
                     strategies: list[Strategy] | None = None,
                     *, compose: bool = True) -> list[tuple[str, Any]]:
    """Build one ``LLMAuthor`` per strategy, sharing ``complete_fn``. Each wraps
    the base author prompt (optionally the kernel_compose building-blocks prompt,
    default on) with the strategy's directive prepended. Returns
    ``[(strategy_name, author), ...]`` for ``Tournament.run``."""
    from kernel_author import LLMAuthor, build_author_prompt  # noqa: PLC0415
    strategies = strategies or DEFAULT_STRATEGIES
    if compose:
        try:
            from kernel_compose import make_compose_prompt_fn  # noqa: PLC0415
            base = make_compose_prompt_fn()
        except Exception:  # noqa: BLE001 — compose is a bonus; fall back to plain
            base = build_author_prompt
    else:
        base = build_author_prompt
    out: list[tuple[str, Any]] = []
    for s in strategies:
        out.append((s.name, LLMAuthor(
            complete_fn, build_prompt=_strategy_prompt_fn(s.directive, base))))
    return out


class TournamentAuthor:
    """A ``KernelAuthor`` that runs a tournament on the FIRST round to pick the
    approach, then delegates repair rounds to the winning strategy's author.

    ``measure_fn`` is injected (the engine's ``_device_race``): the tournament
    cannot pick a winner without racing candidates, so this author only works when
    a measure is available. With ``measure_fn=None`` it degrades to the first
    strategy's author (no bracket) so it never breaks a caller that has no device.
    """

    def __init__(self, complete_fn: Callable[[str], str],
                 measure_fn: MeasureFn | None = None,
                 strategies: list[Strategy] | None = None,
                 *, compose: bool = True) -> None:
        self._authors = strategy_authors(complete_fn, strategies, compose=compose)
        self._measure = measure_fn
        self._tournament = Tournament()
        self._winner_author: Any = None
        self.last_result: TournamentResult | None = None

    def with_measure(self, measure_fn: MeasureFn) -> "TournamentAuthor":
        """Set the measure (the engine's ``_device_race``) after construction and
        return self — the engine wires its bound race in via this seam, since it
        is not available when the caller builds the author. Returns self so it
        chains: ``engine.author = TournamentAuthor(fn).with_measure(engine._device_race)``."""
        self._measure = measure_fn
        return self

    def author(self, spec: OpSpec, lessons: list | None = None,
               feedback: list | None = None,
               perf_feedback: list | None = None) -> AuthoredKernel:
        # Repair round (feedback present): iterate the winning approach, not the
        # whole bracket. Falls back to the first author if no winner was picked.
        if feedback and self._winner_author is not None:
            return self._winner_author.author(spec, lessons, feedback, perf_feedback)
        # First round with no measure -> no bracket possible: use the first author.
        if self._measure is None:
            self._winner_author = self._authors[0][1]
            return self._winner_author.author(spec, lessons, feedback, perf_feedback)
        # First round: run the bracket, remember the winning strategy's author.
        result = self._tournament.run(self._authors, self._measure, spec, lessons)
        self.last_result = result
        if result.winner is not None:
            for name, author in self._authors:
                if name == result.winner.strategy:
                    self._winner_author = author
                    break
            return result.winner.kernel
        # No candidates at all — degrade to the first author's single attempt.
        self._winner_author = self._authors[0][1]
        return self._winner_author.author(spec, lessons, feedback, perf_feedback)
