"""nki_selfimprove.py — Pillar 2/3: the author LEARNS BY DOING.

Pillar 1 (``nki_knowledge``) made the author SKILLED with a *static*, curated
knowledge base: an attention op gets flash/online-softmax examples, a reduction
gets the activation-reduce-fusion idiom, etc. That knowledge never changes — it
cannot learn from what THIS author actually produced on THIS device.

This module closes that loop. After a kernel is authored and MEASURED on-device
(the engine's real ``_compile`` + ``_device_race`` fair-bf16 gate + speedup), we
extract a structured LESSON for that op and PERSIST it per-op:

    {op, best speedup so far, rounds-to-correct, the WINNING kernel
     template/approach that achieved it, and the approaches that FAILED
     (compiler-error class / correctness / slower)}

On the NEXT attempt for the same op, the lesson is RETRIEVED and injected into
the author prompt (alongside ``nki_knowledge``'s static examples) as
"what worked / what failed before — beat your best of Xx". This is the
compounding the framework is built on, but sourced from the author's OWN
measured results rather than a human-curated corpus.

Design — reuse, do not reinvent, the engine's gate/bank/race machinery:

  * ``LessonBank``          — per-op JSON persistence of the structured lesson
    plus every iteration's measured record. Pure I/O + merge logic; no device,
    no model — unit-testable offline.
  * ``_PromptLesson``       — a tiny ``lesson_id``/``reason`` adapter so a
    rendered self-improve lesson flows through the EXISTING
    ``build_author_prompt`` ``lessons`` seam (``_fmt_lessons``) with zero changes
    to the author.
  * ``CapturingAuthor``     — wraps any ``KernelAuthor`` and records the LAST
    authored kernel (source + entry) per op, so after ``run_op`` we can extract
    the winning/failing TEMPLATE (``InventResult`` does not carry the source).
  * ``SelfImproveEngine``   — subclasses ``InventEngine`` and OVERRIDES ONLY
    ``_retrieve_lessons`` to append the current op's rendered self-improve lesson
    to whatever the base engine already retrieves. Every gate/race/bank path is
    the base engine's, unchanged.
  * ``run_selfimprove``     — the loop: author (knowledge + accumulated lessons)
    -> compile+race on-device -> update the lesson bank -> next iter sees it.
    Keeps the BEST correct kernel per op; stops honestly on no-improvement for K
    iters; NEVER fabricates a number (a deferred/errored race is recorded as
    such, not as a speedup).

The measurement is exactly the engine's: correctness is the fair "no worse than
the incumbent bf16 op" gate, speedup is the same-device same-method wallclock
ratio. Nothing here invents a speedup; it only banks and re-surfaces what the
engine measured.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from invent_kernels import AuthoredKernel, OpSpec, resolve_ops
from kernel_author import KernelAuthor
from invent_engine import InventEngine, InventResult

try:  # error-class naming reuses the rewrite catalog when a signature is known
    from kernel_rewrites import match_error
except Exception:  # noqa: BLE001 — classification degrades to keyword buckets
    match_error = None  # type: ignore


# ---------------------------------------------------------------------------
# approach fingerprinting — describe the TEMPLATE an attempt used
# ---------------------------------------------------------------------------
# The specific NKI/ISA calls whose presence characterizes HOW a kernel computes
# the op — the "approach". Two kernels with the same fingerprint took the same
# structural approach; a different fingerprint means the author changed tack
# (which is exactly what a good lesson should provoke after a failure).
_APPROACH_TOKENS = (
    "nisa.activation", "reduce_op", "nl.exp", "nl.max", "nl.sum", "nl.mean",
    "nl.rsqrt", "nl.sqrt", "nisa.reciprocal", "nl.reciprocal", "nl.tanh",
    "nl.sigmoid", "nl.silu", "nisa.nc_matmul", "nisa.nc_transpose",
    "nl.transpose", "nl.mgrid", "nl.arange", "broadcast_to", "keepdims",
    "nl.multiply", "nisa.tensor_reduce",
)


def approach_fingerprint(nki_src: str) -> list[str]:
    """A stable, sorted list of the approach-defining tokens present in the
    source, plus the loop count. Deterministic and cheap — the identity of an
    attempt's TEMPLATE for change-detection and lesson description."""
    if not nki_src:
        return []
    toks = [t for t in _APPROACH_TOKENS if t in nki_src]
    n_loops = len(re.findall(r"\bfor\b", nki_src))
    toks.append(f"loops={n_loops}")
    return sorted(set(toks))


def approach_signature(nki_src: str) -> str:
    """One-line signature of the fingerprint (for prompts / logs)."""
    fp = approach_fingerprint(nki_src)
    return ",".join(fp) if fp else "(empty)"


def source_excerpt(nki_src: str, max_lines: int = 14) -> str:
    """The first ``max_lines`` non-blank source lines — enough for the author to
    recognize the winning template without dumping the whole kernel into every
    future prompt."""
    if not nki_src:
        return ""
    lines = [ln for ln in nki_src.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines])


# ---------------------------------------------------------------------------
# error-class classification — turn an opaque failure into a bankable bucket
# ---------------------------------------------------------------------------
def classify_outcome(status: str, reason: str, *,
                     correct: bool | None = None,
                     ran: bool | None = None) -> str:
    """A stable, human-legible class for an attempt's outcome, so repeated
    failures of the SAME kind are recognizable in the lesson ("do not repeat
    <class>"). MEASURED correctness wins over string matching when provided
    (``correct`` / ``ran`` come straight from the race): a kernel the fair gate
    scored CORRECT but under the 5% margin is ``correct_but_slow``, never
    ``incorrect``. Otherwise buckets by a named rewrite-catalog signature (for
    compile errors) then status + reason keywords. NEVER raises."""
    reason = reason or ""
    if status == "win":
        return "win"
    if status == "no_author":
        return "no_author"
    if status == "device_deferred" or ran is False and status != "anti_pattern":
        return "device_deferred" if status == "device_deferred" else status or "device_deferred"
    # MEASURED correctness is authoritative when we have it.
    if correct is True and status != "win":
        return "correct_but_slow"
    low = reason.lower()
    # Correct-but-slow is the common plateau for these memory-bound ops.
    if status == "anti_pattern" and ("correct but" in low or "< 5%" in low
                                     or "margin" in low):
        return "correct_but_slow"
    # A named compiler signature (the #1 actionable lever) when we can match it.
    if match_error is not None:
        try:
            rw = match_error(reason)
            if rw:
                return f"compile:{rw[0].name}"
        except Exception:  # noqa: BLE001
            pass
    if "offline reject" in low or "offline gate" in low or "lint" in low:
        return "offline_lint"
    if "repair did not converge" in low or "repair failed" in low:
        return "compile_unconverged"
    if "device race error" in low or "compile" in low or "trace" in low \
            or "entry function" in low:
        return "compile_error"
    if status == "anti_pattern":
        # WRONG on device (correctness miss) — reason carries the % within tol.
        return "incorrect"
    return status or "unknown"


# ---------------------------------------------------------------------------
# the persisted lesson
# ---------------------------------------------------------------------------
@dataclass
class IterationRecord:
    """One self-improve iteration's MEASURED outcome (never fabricated)."""

    iteration: int
    status: str                         # engine status: win|anti_pattern|...
    compiled: bool                      # built + ran on device
    correct: bool
    speedup: float | None               # None when not raced (deferred/error)
    correctness_pct: float
    rounds: int                         # author calls inside this run_op (repair)
    outcome_class: str
    approach_sig: str
    approach_changed_from_prev: bool
    lesson_injected: bool               # was a self-improve lesson in the prompt
    lessons_consulted: int              # total banked lessons the engine used
    reason: str


@dataclass
class OpLesson:
    """The structured, persisted per-op lesson — the thing RETRIEVED and injected
    into the next attempt's prompt."""

    op: str
    shape_class: str = ""
    best_speedup: float | None = None   # best among CORRECT kernels
    best_correct: bool = False
    best_kernel_src: str = ""
    best_entry: str = ""
    best_approach_sig: str = ""
    rounds_to_correct: int | None = None    # 1-based iter of FIRST correct kernel
    first_iter_compiled: bool | None = None
    n_attempts: int = 0
    failed_approaches: list[dict] = field(default_factory=list)
    iterations: list[dict] = field(default_factory=list)
    updated: str = ""

    # -- prompt rendering ---------------------------------------------------
    def render_for_prompt(self) -> str:
        """The "what worked / what failed before — beat your best of Xx" block
        injected into the author prompt. Returns "" when there is nothing banked
        yet (so the FIRST attempt is the pure static-knowledge baseline)."""
        if self.n_attempts == 0:
            return ""
        lines: list[str] = [
            "SELF-IMPROVEMENT MEMORY — your OWN measured results for this exact "
            "op on this device (learn from them; do not repeat a failed "
            "approach):",
        ]
        if self.best_correct and self.best_speedup is not None:
            lines.append(
                f"  - Best CORRECT kernel so far: {self.best_speedup:.3f}x vs "
                f"baseline (first correct at iteration {self.rounds_to_correct}). "
                f"YOUR TARGET: beat {self.best_speedup:.3f}x while staying correct."
            )
            if self.best_approach_sig:
                lines.append(f"    winning approach: {self.best_approach_sig}")
            exc = source_excerpt(self.best_kernel_src)
            if exc:
                lines.append("    winning template (reuse its structure, then "
                             "optimize further):")
                lines.append("    ```python")
                for ln in exc.splitlines():
                    lines.append("    " + ln)
                lines.append("    ```")
        else:
            lines.append(
                f"  - No CORRECT kernel yet in {self.n_attempts} attempt(s). A "
                f"CORRECT kernel is the first goal — correctness before speed."
            )
        if self.failed_approaches:
            lines.append("  - FAILED approaches so far — do NOT repeat these; "
                         "change the structural approach:")
            # De-dup by (class, approach) so the list stays short + actionable.
            seen: set[tuple] = set()
            for fa in self.failed_approaches:
                key = (fa.get("outcome_class"), fa.get("approach_sig"))
                if key in seen:
                    continue
                seen.add(key)
                note = fa.get("note", "")
                lines.append(
                    f"      * iter {fa.get('iteration')} "
                    f"[{fa.get('outcome_class')}]: {note} "
                    f"(approach: {fa.get('approach_sig')})"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------
class LessonBank:
    """Per-op JSON lesson store. One file per op under ``root``.

    ``get`` returns the persisted ``OpLesson`` (or a fresh empty one);
    ``update`` merges a measured ``IterationRecord`` in, recomputes the best, and
    persists; ``render_for_prompt`` returns the injection text. All disk I/O is
    defensive — a corrupt/absent file yields a fresh lesson, never an exception,
    so a broken bank can never stop authoring."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, op: str) -> Path:
        safe = re.sub(r"\W+", "_", op)
        return self.root / f"{safe}.json"

    def get(self, op: str) -> OpLesson:
        p = self._path(op)
        if not p.exists():
            return OpLesson(op=op)
        try:
            data = json.loads(p.read_text())
            known = {f for f in OpLesson.__dataclass_fields__}  # type: ignore
            return OpLesson(**{k: v for k, v in data.items() if k in known})
        except Exception:  # noqa: BLE001 — a corrupt bank must not stop authoring
            return OpLesson(op=op)

    def save(self, lesson: OpLesson) -> None:
        lesson.updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self._path(lesson.op).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(lesson), indent=2))
        tmp.replace(self._path(lesson.op))

    def render_for_prompt(self, op: str) -> str:
        return self.get(op).render_for_prompt()

    def update(self, op: str, shape_class: str, rec: IterationRecord,
               kernel_src: str = "", entry: str = "") -> OpLesson:
        """Merge one measured iteration into the op's lesson and persist.

        Recomputes ``best_*`` (best speedup among CORRECT kernels), records the
        winning TEMPLATE (source + approach) when a new best is set, appends
        FAILED approaches, and sets ``rounds_to_correct`` on the first correct
        kernel. Idempotent on the derived fields given the same record stream."""
        lesson = self.get(op)
        lesson.op = op
        lesson.shape_class = shape_class or lesson.shape_class
        lesson.n_attempts += 1
        lesson.iterations.append(asdict(rec))
        if lesson.first_iter_compiled is None:
            lesson.first_iter_compiled = rec.compiled

        if rec.correct:
            if lesson.rounds_to_correct is None:
                lesson.rounds_to_correct = rec.iteration
            # A correct kernel with a measured speedup can set a new best.
            if rec.speedup is not None and (
                lesson.best_speedup is None or rec.speedup > lesson.best_speedup
            ):
                lesson.best_speedup = rec.speedup
                lesson.best_correct = True
                lesson.best_kernel_src = kernel_src
                lesson.best_entry = entry
                lesson.best_approach_sig = rec.approach_sig
        else:
            # A loss is data: bank the failed approach with its class + note.
            note = rec.reason.strip()
            if len(note) > 200:
                note = note[:197] + "..."
            lesson.failed_approaches.append({
                "iteration": rec.iteration,
                "outcome_class": rec.outcome_class,
                "approach_sig": rec.approach_sig,
                "note": note,
            })
        self.save(lesson)
        return lesson


# ---------------------------------------------------------------------------
# prompt-seam adapter
# ---------------------------------------------------------------------------
@dataclass
class _PromptLesson:
    """Minimal object that satisfies ``kernel_author._fmt_lessons`` (needs only
    ``lesson_id`` + ``reason``), so a rendered self-improve lesson rides the
    EXISTING author ``lessons`` seam with no author changes."""

    lesson_id: str
    reason: str
    # The engine's ``_lesson_relevant`` filter is bypassed (we inject directly in
    # the subclass), but provide these so a stray relevance check never crashes.
    intervention: dict = field(default_factory=dict)
    symptoms_addressed: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# capturing author
# ---------------------------------------------------------------------------
class CapturingAuthor:
    """Wraps a ``KernelAuthor`` and records the LAST authored kernel per op (so
    the loop can extract the winning/failed TEMPLATE, which ``InventResult`` does
    not carry) and counts author calls within a ``run_op`` (the repair rounds)."""

    def __init__(self, inner: KernelAuthor) -> None:
        self.inner = inner
        self.last: dict[str, AuthoredKernel] = {}
        self._calls_since_reset = 0

    def reset_calls(self) -> None:
        self._calls_since_reset = 0

    @property
    def calls(self) -> int:
        return self._calls_since_reset

    def author(self, spec: OpSpec, lessons: list | None = None,
               feedback: list | None = None,
               perf_feedback: list | None = None) -> AuthoredKernel:
        self._calls_since_reset += 1
        k = self.inner.author(spec, lessons, feedback, perf_feedback)
        self.last[spec.name] = k
        return k


# ---------------------------------------------------------------------------
# the self-improving engine
# ---------------------------------------------------------------------------
class SelfImproveEngine(InventEngine):
    """``InventEngine`` that injects a per-op self-improve lesson into authoring.

    The ONLY override is ``_retrieve_lessons``: it returns the base engine's
    retrieved lessons (family anti-patterns, symptom index, the engine's own
    provisional bank) PLUS a ``_PromptLesson`` rendering the dedicated
    ``LessonBank`` entry for this op. Every gate/compile/race/bank path is the
    base engine's, unchanged — the self-improvement is entirely in what the
    author is TOLD, and it is measured by the same fair gate as always."""

    def __init__(self, out_dir, lesson_bank: LessonBank,
                 author: KernelAuthor, **kw) -> None:
        self._capturing = CapturingAuthor(author)
        super().__init__(out_dir, author=self._capturing, **kw)
        self.lesson_bank = lesson_bank
        # Observability: set by run_op via _retrieve_lessons so the loop can
        # record whether a self-improve lesson was actually injected this iter.
        self._last_selfimprove_injected = False

    def _retrieve_lessons(self, spec: OpSpec) -> list:
        base = []
        try:
            base = super()._retrieve_lessons(spec)
        except Exception:  # noqa: BLE001 — base retrieval must never stop authoring
            base = []
        text = ""
        try:
            text = self.lesson_bank.render_for_prompt(spec.name)
        except Exception:  # noqa: BLE001
            text = ""
        self._last_selfimprove_injected = bool(text)
        if text:
            base = list(base) + [
                _PromptLesson(lesson_id=f"selfimprove-{spec.name}", reason=text)
            ]
        return base


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def _derive_record(result: InventResult, iteration: int, rounds: int,
                   prev_sig: str | None, lesson_injected: bool,
                   nki_src: str) -> IterationRecord:
    """Turn an ``InventResult`` (+ captured source) into an ``IterationRecord``.

    Correctness/speedup come straight from the engine's measured race — never
    synthesized. ``compiled`` == the kernel built and ran on device
    (``race.ran``). ``speedup`` is None when the race did not run."""
    race = result.race
    ran = bool(getattr(race, "ran", False))
    correct = bool(getattr(race, "correct", False))
    speedup = float(race.speedup) if ran else None
    corr_pct = float(getattr(race, "correctness_pct", 0.0))
    sig = approach_signature(nki_src)
    outcome_class = classify_outcome(
        result.status, result.race.reason or result.detail,
        correct=correct, ran=ran)
    return IterationRecord(
        iteration=iteration,
        status=result.status,
        compiled=ran,
        correct=correct,
        speedup=speedup,
        correctness_pct=corr_pct,
        rounds=rounds,
        outcome_class=outcome_class,
        approach_sig=sig,
        approach_changed_from_prev=(prev_sig is not None and sig != prev_sig),
        lesson_injected=lesson_injected,
        lessons_consulted=result.lessons_consulted,
        reason=(result.race.reason or result.detail or ""),
    )


def run_selfimprove(op: str, iters: int, *,
                    out_dir: Path | str,
                    lesson_bank_root: Path | str | None = None,
                    author: KernelAuthor | None = None,
                    provider: str = "bedrock",
                    model_id: str = "global.anthropic.claude-opus-5",
                    region: str | None = "ap-southeast-4",
                    max_repair_rounds: int = 1,
                    no_improve_stop_k: int = 3,
                    race_fn: Callable | None = None,
                    log: Callable[[str], None] = print) -> dict:
    """Iterate the author on ONE op, banking + retrieving its own lessons.

    Each iteration: author (static knowledge + accumulated lessons) -> engine
    offline gate -> on-device compile+race (fair bf16 gate + speedup) -> update
    the per-op ``LessonBank`` -> the NEXT iteration's prompt carries the updated
    lesson. Keeps the BEST correct kernel. Stops early (honestly) if ``best``
    does not improve for ``no_improve_stop_k`` consecutive iterations AFTER a
    correct kernel exists.

    ``author`` may be injected (an ``LLMAuthor`` with an echo/mock ``complete_fn``
    for offline tests, or any ``KernelAuthor``); when None a real provider-backed
    author is built via ``author_from_provider(provider, ...)``. ``race_fn`` lets
    tests inject a deterministic device outcome; on device the engine's own
    ``_device_race`` is used.

    Returns a JSON-able trajectory + summary. NEVER fabricates a number — an
    iteration that could not be raced records ``speedup=None`` and ``compiled``
    reflects reality."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = LessonBank(lesson_bank_root or (out_dir / "selfimprove-lessons"))

    if author is None:
        from kernel_providers import author_from_provider
        kw: dict[str, Any] = {}
        if provider == "bedrock":
            kw = dict(model_id=model_id, region=region, temperature=None)
        author = author_from_provider(provider, **kw)

    engine = SelfImproveEngine(out_dir, lesson_bank=bank, author=author,
                               max_repair_rounds=max_repair_rounds)
    spec = resolve_ops([op])[0]

    trajectory: list[dict] = []
    best_speedup: float | None = None
    best_iter: int | None = None
    stale = 0
    prev_sig: str | None = None
    stop_reason = f"completed all {iters} iteration(s)"

    for i in range(1, iters + 1):
        engine._capturing.reset_calls()
        t0 = time.time()
        try:
            result = engine.run_op(spec, race_fn=race_fn)
        except Exception as e:  # noqa: BLE001 — an authoring/provider error is data
            log(f"[{op} iter {i}] run_op raised: {e!r}")
            trajectory.append({
                "iteration": i, "status": "engine_error", "compiled": False,
                "correct": False, "speedup": None, "reason": repr(e),
                "best_speedup_so_far": best_speedup,
            })
            stale += 1
            if best_speedup is not None and stale >= no_improve_stop_k:
                stop_reason = (f"no improvement for {no_improve_stop_k} iters "
                               f"(last errors) after a correct kernel")
                break
            continue

        rounds = engine._capturing.calls
        authored = engine._capturing.last.get(op)
        nki_src = authored.nki_src if authored else ""
        entry = authored.entry if authored else ""
        rec = _derive_record(result, i, rounds, prev_sig,
                             engine._last_selfimprove_injected, nki_src)
        bank.update(op, spec.shape_class, rec, kernel_src=nki_src, entry=entry)
        prev_sig = rec.approach_sig

        improved = False
        if rec.correct and rec.speedup is not None:
            if best_speedup is None or rec.speedup > best_speedup:
                best_speedup = rec.speedup
                best_iter = i
                improved = True
        row = asdict(rec)
        row["best_speedup_so_far"] = best_speedup
        row["dt_s"] = round(time.time() - t0, 1)
        trajectory.append(row)
        log(f"[{op} iter {i}] status={rec.status} compiled={rec.compiled} "
            f"correct={rec.correct} speedup={rec.speedup} class={rec.outcome_class} "
            f"lesson_injected={rec.lesson_injected} "
            f"approach_changed={rec.approach_changed_from_prev} "
            f"best={best_speedup} ({row['dt_s']}s)")

        # Honest early stop: only AFTER we have a correct kernel to preserve, and
        # only when best has not moved for K straight iterations.
        if improved:
            stale = 0
        elif best_speedup is not None:
            stale += 1
            if stale >= no_improve_stop_k:
                stop_reason = (f"no improvement in best speedup for "
                               f"{no_improve_stop_k} consecutive iters after a "
                               f"correct kernel (plateau)")
                break

    final_lesson = bank.get(op)
    summary = _summarize(op, spec.shape_class, trajectory, final_lesson,
                         best_speedup, best_iter, stop_reason)
    return {
        "op": op,
        "shape_class": spec.shape_class,
        "iters_run": len(trajectory),
        "iters_requested": iters,
        "stop_reason": stop_reason,
        "trajectory": trajectory,
        "lesson": asdict(final_lesson),
        "summary": summary,
    }


def _summarize(op: str, shape_class: str, trajectory: list[dict],
               lesson: OpLesson, best_speedup: float | None,
               best_iter: int | None, stop_reason: str) -> dict:
    """The Pillar-3 skill-curve summary: iter-1 (static knowledge only) vs the
    best-with-accumulated-lessons, plus the metrics that DO move on memory-bound
    ops even when speedup caps below 1x (rounds-to-correct, first-try-compile,
    approach churn driven by lessons)."""
    iter1 = trajectory[0] if trajectory else {}
    correct_iters = [r for r in trajectory if r.get("correct")]
    compiled_iters = [r for r in trajectory if r.get("compiled")]
    lesson_iters = [r for r in trajectory if r.get("lesson_injected")]
    approach_changes = sum(1 for r in trajectory
                           if r.get("approach_changed_from_prev"))
    speedups = [r["speedup"] for r in trajectory
                if r.get("correct") and r.get("speedup") is not None]
    return {
        "iter1_status": iter1.get("status"),
        "iter1_compiled": iter1.get("compiled"),
        "iter1_correct": iter1.get("correct"),
        "iter1_speedup": iter1.get("speedup"),
        "best_speedup": best_speedup,
        "best_iter": best_iter,
        "n_correct": len(correct_iters),
        "n_compiled": len(compiled_iters),
        "first_try_compiled": lesson.first_iter_compiled,
        "rounds_to_correct": lesson.rounds_to_correct,
        "n_iters_with_lesson_injected": len(lesson_iters),
        "approach_changes": approach_changes,
        "speedup_first_correct": speedups[0] if speedups else None,
        "speedup_last_correct": speedups[-1] if speedups else None,
        # The honest verdict signal: did the best-with-lessons beat iter-1?
        "improved_over_iter1": (
            best_speedup is not None and iter1.get("speedup") is not None
            and best_speedup > iter1.get("speedup")
        ),
        "stop_reason": stop_reason,
    }
