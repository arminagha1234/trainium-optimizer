"""kernel_repair.py — the iterative author -> compile -> read-error -> re-author
loop. This is the mechanism by which the framework gets BETTER at writing
kernels: each compile failure TEACHES the next attempt.

Today's invent_engine authors a kernel ONCE, gates it, and gives up on failure.
The mature pattern (the Neuron AutoFixer does this, up to 8 rounds) is a bounded
repair loop: compile, and on failure feed the EXACT error back — matched against
the symptom-indexed rewrite catalog (kernel_rewrites) — so the next attempt
knows what to fix. The captured Qwen3-Next `TensorScalarAffineSelect` error is a
teacher: round 2 knows to replace `.tril()` with a constant mask.

Design: author and compile are INJECTED functions (interfaces), so the loop is
    (a) unit-testable with a mock author that "learns" from feedback, and
    (b) pluggable with a real LLM/agent author + a real `neuronx-cc` compile.
The loop itself owns only the control flow, the diagnosis, and the honest
stop conditions — never a fabricated success.

    AuthorFn:  (trail: list[Feedback]) -> kernel        # consumes accumulated feedback
    CompileFn: (kernel) -> CompileResult                # ok + error_log

Honest stops (never burn N pointless compiles):
  * success        — compile ok.
  * exhausted      — max_rounds reached without a compile.
  * stalled        — two consecutive rounds produced the IDENTICAL error, i.e.
                     the author is not consuming feedback / the error has no
                     matching rewrite lead; further rounds cannot improve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from kernel_rewrites import Rewrite, describe, match_error


@dataclass
class CompileResult:
    """Outcome of one compile attempt. `error_log` is the raw compiler output on
    failure (the teacher); `artifact` is the NEFF handle on success."""

    ok: bool
    error_log: str = ""
    artifact: str = ""


@dataclass
class Feedback:
    """One failed round, fed back to the author for the next attempt."""

    round: int
    error_log: str
    rewrites: list[Rewrite] = field(default_factory=list)

    def as_prompt(self) -> str:
        """The actionable message the author consumes on the next round."""
        tail = self.error_log.strip().splitlines()[-6:]
        fix = describe(self.rewrites)
        lead = (f"Round {self.round} failed to compile.\n"
                f"Compiler error (tail):\n  " + "\n  ".join(tail) + "\n"
                f"Known fix for this symptom: {fix}")
        if self.rewrites:
            lead += "\nApply that rewrite and re-author.\n" + "\n".join(
                f"[{r.name}]\n{r.fix}" for r in self.rewrites)
        return lead


@dataclass
class RepairOutcome:
    ok: bool
    rounds: int
    reason: str                       # "compiled" | "exhausted rounds" | "stalled: ..."
    kernel: Any = None
    artifact: str = ""
    trail: list[Feedback] = field(default_factory=list)

    @property
    def suggested_rewrites(self) -> list[Rewrite]:
        """The distinct rewrites the loop surfaced — the actionable output even
        when it did not converge (a named work item, not an opaque 'failed')."""
        seen: dict[str, Rewrite] = {}
        for fb in self.trail:
            for r in fb.rewrites:
                seen.setdefault(r.name, r)
        return list(seen.values())


AuthorFn = Callable[[list["Feedback"]], Any]
CompileFn = Callable[[Any], CompileResult]


class KernelRepairLoop:
    """Bounded author -> compile -> diagnose -> re-author loop."""

    def __init__(self, max_rounds: int = 6, stall_patience: int = 2) -> None:
        # stall_patience consecutive IDENTICAL errors -> bail (no progress).
        self.max_rounds = max_rounds
        self.stall_patience = stall_patience

    def diagnose(self, error_log: str) -> list[Rewrite]:
        """Symptom -> known rewrites. The retrieval that turns an opaque compiler
        error into a named, actionable fix."""
        return match_error(error_log)

    def run(self, author_fn: AuthorFn, compile_fn: CompileFn) -> RepairOutcome:
        trail: list[Feedback] = []
        last_error: str | None = None
        stall = 0
        for rnd in range(1, self.max_rounds + 1):
            kernel = author_fn(trail)              # author sees ALL prior feedback
            result = compile_fn(kernel)
            if result.ok:
                return RepairOutcome(True, rnd, "compiled", kernel,
                                     result.artifact, trail)

            # No progress guard: identical error to last round -> the author is
            # not acting on the feedback (or there is no lead), so more rounds
            # can't help. Bail honestly rather than burn the budget.
            if last_error is not None and result.error_log == last_error:
                stall += 1
            else:
                stall = 0
            last_error = result.error_log

            trail.append(Feedback(rnd, result.error_log,
                                  self.diagnose(result.error_log)))

            if stall + 1 >= self.stall_patience:
                return RepairOutcome(
                    False, rnd, "stalled: repeated identical error, no progress",
                    None, "", trail)

        return RepairOutcome(False, self.max_rounds, "exhausted rounds",
                             None, "", trail)
