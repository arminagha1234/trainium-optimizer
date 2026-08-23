"""
Stage-4 INVENT engine — authors NEW NKI kernels, gates them, races them, banks
the results. This is the real invention capability the orchestrator's Stage 4
today only stubs out ("no auto-invention (needs NKI-writer agent)").

Standalone by design: it imports the framework's real ``bank`` (KnowledgeBank /
Lesson), ``guardrails`` (the 5% invention margin), and ``ledger`` (append-only
results.tsv), but does NOT touch ``orchestrator.py`` / ``overnight.py`` — those
boxes have diverged and integration is a later step.

The loop, per op (see ../../docs/stage4-invent-design.md):

    op_spec {name, shapes+dtypes, reference fn, baseline to beat}
      -> author_kernel(op_spec) -> AuthoredKernel        (the headline: novel authoring)
      -> OFFLINE gate:   numpy-ref parity @128x128  +  static NKI lint
      -> ON-DEVICE gate: correctness (allclose vs ref, real shape)
                         + speed race (FAIR: kernel and baseline timed by the
                           SAME method on the SAME device, else device_deferred)
      -> keep ONLY if correct AND faster by >= 5% invention margin
      -> bank:  win  -> `invented` NKI_KERNEL lesson (keyed op+arch+shape-class)
                loss -> `anti_pattern` lesson (correct-but-slow, or wrong, or
                        offline-reject) — losses are DATA, logged not hidden.

CPU-mock-testable: the on-device race is behind ``AuthoredKernel.build()``,
which returns None off-device. On a plain CPU box the engine authors, offline-
gates, and (in tests) banks against an INJECTED race — the full harness logic is
exercised without a Trainium. Only the real ``nki.benchmark`` race needs trn2.

Run on device (.73 / .211):

    # FIRST validate the execution path on a proven seed (build+invoke+measure).
    # Success = it EXECUTES + is measured, NOT "entry function not found".
    # Exits non-zero on device if the seed cannot execute.
    python invent_engine.py --self-test --out /path/to/invent_runs/

    # then author + gate + race + bank the novel ops:
    python invent_engine.py \\
        --ops rope_apply,gelu_tanh,softcap,add_rmsnorm,layernorm,attn_decode \\
        --out /path/to/invent_runs/
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Real framework imports — Stage 4 is banked into the SAME store the proposer
# reads, using the SAME lesson schema and the SAME invention margin.
from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Symptom,
    Tier,
    _norm_family,
)
from guardrails import Guardrails
import kernel_rewrites
from kernel_author import KernelAuthor, RecipeAuthor
from kernel_repair import CompileResult, Feedback, KernelRepairLoop
from ledger import Layer, Ledger, Origin, Row, Stage, Status, current_commit
from invent_kernels import (
    AuthoredKernel,
    OpSpec,
    author_kernel,
    nki_available,
    resolve_ops,
    static_lint,
)

# fp32 offline parity is a MATH check (does the clever formulation equal the
# reference?), so it is tight. The on-device allclose uses bf16 tolerances.
_OFFLINE_ATOL = 1e-4
_OFFLINE_RTOL = 1e-4
_BF16_ATOL = 1e-2
_BF16_RTOL = 1e-2

_SDK = "2.28.0"


# ---------------------------------------------------------------------------
# result records
# ---------------------------------------------------------------------------
@dataclass
class OfflineGate:
    passed: bool
    parity_ok: bool                # True ONLY when an INDEPENDENT re-derivation
                                   # matched the reference — never set by a
                                   # function-compared-to-itself tautology.
    parity_max_abs_err: float
    lint_violations: list[str] = field(default_factory=list)
    reason: str = ""
    # Did the recipe supply a numpy_impl that is a genuinely DIFFERENT expression
    # than spec.reference? When False (the recipe reuses spec.reference verbatim),
    # the offline parity comparison is a tautology (f vs f) and validates nothing,
    # so ``parity_ok`` is forced False and the math is left to the on-device gate.
    # Kept as the LAST field with a default so existing positional constructions
    # (incl. tests) are unaffected.
    parity_independent: bool = True


@dataclass
class RaceResult:
    """Outcome of the on-device correctness + speed race.

    ``ran`` is False off-device (kernel could not be built) — an honest
    "deferred", never a fabricated number.
    """

    ran: bool
    correct: bool = False
    correctness_pct: float = 0.0
    speedup: float = 0.0          # baseline_time / kernel_time; >1 == faster
    kernel_ms: float = 0.0
    baseline_ms: float = 0.0
    reason: str = ""


@dataclass
class InventResult:
    op: str
    shape_class: str
    origin: str
    status: str                    # harvested | win | anti_pattern |
                                   # offline_reject | device_deferred | no_author
    offline: OfflineGate
    race: RaceResult
    lesson_id: str = ""
    detail: str = ""
    # How many previously-banked lessons (anti-patterns / prior wins) the engine
    # RETRIEVED as relevant to this op before authoring. Makes the "learn from
    # the bank" step observable. Last field with a default so existing positional
    # constructions (incl. tests) are unaffected.
    lessons_consulted: int = 0


# A race function lets tests inject a deterministic device outcome. On device
# the engine's own ``_device_race`` is used.
RaceFn = Callable[[AuthoredKernel, OpSpec], RaceResult]

# A compile function lets the repair loop (and tests) turn an AuthoredKernel into
# a CompileResult (ok + error_log). On device the engine's own ``_compile`` is
# used; tests inject a deterministic stand-in compiler.
CompileFnT = Callable[[AuthoredKernel], CompileResult]


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class InventEngine:
    """Authors, gates, races, and banks NKI kernels for a set of ops."""

    def __init__(
        self,
        out_dir: Path | str,
        bank_root: Path | str | None = None,
        guards: Guardrails | None = None,
        sdk_version: str = _SDK,
        registry: "Any" = None,
        author: KernelAuthor | None = None,
        max_repair_rounds: int = 1,
    ) -> None:
        # The pluggable authoring seam. Defaults to the recipe table
        # (``RecipeAuthor`` wraps ``invent_kernels.author_kernel``) so behaviour
        # is unchanged; pass an ``LLMAuthor`` (or any ``KernelAuthor``) to drive
        # authoring from a model/agent. ``max_repair_rounds`` is the bound on the
        # author -> compile -> read-error -> re-author loop; the DEFAULT of 1 is
        # today's single-shot authoring (no repair loop), so existing runs and
        # tests are byte-for-byte unchanged. >1 activates the real repair loop.
        self.author: KernelAuthor = author or RecipeAuthor()
        self.max_repair_rounds = max_repair_rounds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Default the bank under the run dir so an experiment never pollutes the
        # curated repo bank unless the caller explicitly points at it.
        self.bank = KnowledgeBank(Path(bank_root) if bank_root
                                  else self.out_dir / "knowledge-bank")
        self.guards = guards or Guardrails()
        self.sdk_version = sdk_version
        self.ledger = Ledger(self.out_dir)
        self.ledger.init()
        # Prior-art / Harvest: consult a kernel registry BEFORE authoring, so a
        # primitive that already has an authored kernel (e.g. DeltaNet for a
        # GatedDeltaNet op) is REUSED, not re-invented. Defaults to an empty
        # registry (nothing available), so behaviour is unchanged unless the
        # caller passes one or $TRN_OPT_KERNEL_DIR is set. Import is local so the
        # engine has no hard dependency on the routing layer.
        if registry is None:
            from kernel_registry import KernelRegistry
            registry = KernelRegistry()
        self.registry = registry

    # -- prior-art / Harvest (search before authoring) -----------------------

    def _prior_art(self, spec: OpSpec):
        """Return a usable, already-authored KernelSpec for this op's primitive,
        or None. This is the Harvest step of Harvest -> Borrow -> Invent, and the
        AutoFixer 'search prior art before authoring' rule: never re-invent a
        kernel the corpus already has. Only kernels at >= simulate-correct rank
        are returned (a failed-compile attempt is not prior art to reuse)."""
        if not getattr(spec, "primitive", ""):
            return None
        try:
            kspec = self.registry.for_primitive(spec.primitive)
        except Exception:  # noqa: BLE001 — a broken registry must not stop authoring
            return None
        return kspec if (kspec and kspec.usable) else None

    # -- learn from the bank (retrieve relevant lessons before authoring) ----

    def _lesson_relevant(self, lesson: Lesson, spec: OpSpec) -> bool:
        """Is a banked lesson relevant to THIS op? By op name / shape_class /
        symptom — the three keys the invent loop banks under (lesson ids are
        ``invented-<op>-<shape_class>`` / ``antipattern-invented-<op>-<shape_class>``;
        symptom signatures name the op)."""
        name = (spec.name or "").lower()
        sc = (spec.shape_class or "").lower()
        hay = f"{lesson.lesson_id} {lesson.reason}".lower()
        if name and name in hay:
            return True
        if sc and sc in hay:
            return True
        iv = lesson.intervention.get("spec", {}) if isinstance(lesson.intervention, dict) else {}
        if isinstance(iv, dict) and (
            iv.get("nki_kernel") == spec.name or iv.get("shape_class") == spec.shape_class
        ):
            return True
        for s in lesson.symptoms_addressed:
            if name and name in (s.signature or "").lower():
                return True
        return False

    def _retrieve_lessons(self, spec: OpSpec) -> list[Lesson]:
        """Query the bank for anti-patterns / prior lessons relevant to this op
        BEFORE authoring, so previously-banked losses and wins become
        load-bearing (today the engine WRITES lessons but never READS them).

        Uses the bank's real retrieval API:
          * ``KnowledgeBank.antipatterns(family, sdk)`` — family anti-patterns
            (verified tier), the same index the pre-compile prune consults;
          * ``KnowledgeBank.query_symptom("compute_bound", ...)`` — the ADIAS
            symptom index the invent NKI_KERNEL / anti-pattern lessons are keyed
            under (verified tier).
        Both read VERIFIED only, so we additionally sweep the PROVISIONAL tier
        the invent loop itself writes to — a loss banked on op A of a run should
        inform a later authoring of the same op/shape-class in the SAME run,
        without waiting on weekly human promotion (the compounding the framework
        is built on). Relevance is filtered by ``_lesson_relevant``. Never
        raises — a broken bank must not stop authoring."""
        sdk = self.sdk_version
        found: dict[str, Lesson] = {}

        def _add(lessons: list[Lesson] | None) -> None:
            for l in lessons or []:
                if l.lesson_id in found:
                    continue
                if self._lesson_relevant(l, spec):
                    found[l.lesson_id] = l

        # (1) family anti-patterns (verified) — the real bank pruning index.
        try:
            _add(self.bank.antipatterns(spec.family, sdk))
        except Exception:  # noqa: BLE001 — a broken bank must not stop authoring
            pass
        # (2) symptom index (verified) — invent lessons are keyed compute_bound.
        try:
            _add(self.bank.query_symptom("compute_bound", spec.family,
                                         0.0, 0, 1, sdk))
        except Exception:  # noqa: BLE001
            pass
        # (3) provisional tier the invent loop itself writes — so lessons compound
        #     within an autonomous run before any human promotion.
        try:
            def _fam_ok(l: Lesson) -> bool:
                af = l.applicability.architecture_family
                if _norm_family(af) != _norm_family(spec.family):
                    return False
                pats = l.applicability.neuron_sdk_versions
                return (not pats) or any(fnmatch.fnmatch(sdk, p) for p in pats)

            _add([l for l in self.bank.load_all(Tier.PROVISIONAL)
                  if l.type in (LessonType.ANTI_PATTERN, LessonType.NKI_KERNEL)
                  and _fam_ok(l)])
        except Exception:  # noqa: BLE001
            pass
        return list(found.values())

    # -- diagnose a failure with the rewrite catalog -------------------------

    def _diagnose_failure(self, error_text: str) -> tuple[str, str]:
        """Match a compiler / error string against the rewrite catalog. Returns
        ``(desc_suffix, reason_suffix)`` — both empty when nothing matches —
        turning an opaque "failed" into an actionable "failed; known fix:
        <rewrite>". The reason_suffix is appended to the banked anti-pattern so
        the next author sees the fix; the desc_suffix lands in the ledger row."""
        try:
            rewrites = kernel_rewrites.match_error(error_text or "")
        except Exception:  # noqa: BLE001 — diagnosis must never break banking
            return "", ""
        if not rewrites:
            return "", ""
        names = ", ".join(r.name for r in rewrites)
        return (f" [known fix: {names}]",
                f" Known fix (rewrite catalog): {kernel_rewrites.describe(rewrites)}")

    # -- offline gate --------------------------------------------------------

    def offline_gate(self, author: AuthoredKernel, spec: OpSpec) -> OfflineGate:
        """numpy-ref parity at the 128x128 shape + static NKI lint.

        Both must pass before ANY device time. Parity validates the math the
        kernel is built on (step 2); lint enforces the mandatory NKI rules
        (partition=128, no arange, no int/tile, DMA rule) on the source text.
        """
        if not author.nki_src or not author.entry:
            return OfflineGate(False, False, float("inf"),
                               reason="no authored kernel source (no recipe)")
        # (1) numpy parity at 128x128 — but a parity check only MEANS something
        # when the kernel's numpy_impl is an INDEPENDENT re-derivation of the
        # reference. Most catalog recipes (all but rope_apply) reuse
        # ``spec.reference`` verbatim as their numpy_impl, so ``numpy_impl(inp)``
        # vs ``reference(inp)`` compares a function to ITSELF — a tautology that
        # trivially passes and validates nothing. We detect that by identity and
        # refuse to report it as a parity PASS. The op is NOT rejected: we still
        # execute the impl once (a real smoke check — it runs, it produces the
        # reference's shape) and defer the actual math check to the on-device
        # gate (allclose vs the reference on the REAL shape), which is the true
        # correctness test for these ops.
        independent = author.numpy_impl is not spec.reference
        inp = spec.offline_inputs()
        try:
            got = np.asarray(author.numpy_impl(inp), dtype=np.float32)
            ref = np.asarray(spec.reference(inp), dtype=np.float32)
        except Exception as e:  # noqa: BLE001 — a math bug is a gate failure, not a crash
            return OfflineGate(False, False, float("inf"),
                               parity_independent=independent,
                               reason=f"numpy_impl raised: {e!r}")
        shape_ok = got.shape == ref.shape
        if independent:
            max_err = float(np.max(np.abs(got - ref))) if shape_ok else float("inf")
            parity_ok = shape_ok and np.allclose(
                got, ref, atol=_OFFLINE_ATOL, rtol=_OFFLINE_RTOL)
        else:
            # Tautological comparison — do NOT claim a verified parity pass. The
            # smoke run above still guarantees the impl executes and is shaped
            # like the reference; that is all the offline stage honestly checked.
            max_err = 0.0 if shape_ok else float("inf")
            parity_ok = False
        # (2) static lint.
        violations = static_lint(author.nki_src)
        # Device time is gated on: lint clean AND the impl ran with the right
        # shape AND (only when an independent check exists) that check passed. A
        # tautological-parity op still advances to the REAL on-device gate — we
        # simply never pretend an offline parity pass occurred.
        passed = (not violations) and shape_ok and (parity_ok if independent else True)
        reason = ""
        if not shape_ok:
            reason = "numpy_impl shape != reference shape"
        elif independent and not parity_ok:
            reason = f"numpy parity fail (max_abs_err={max_err:.3e})"
        elif violations:
            reason = f"lint: {'; '.join(violations)}"
        elif not independent:
            # Passing, but be explicit in the record about what was NOT verified.
            reason = ("parity NOT independently verified: numpy_impl is "
                      "spec.reference (tautology) — math deferred to on-device gate")
        return OfflineGate(passed, parity_ok, max_err, violations, reason,
                           parity_independent=independent)

    # -- on-device race ------------------------------------------------------

    def _device_race(self, author: AuthoredKernel, spec: OpSpec) -> RaceResult:
        """Real trn2 correctness + speed race. No-op (ran=False) off-device.

        Built to run on .73; not exercised on a CPU box. Correctness is
        ``torch.allclose`` at bf16 tolerance vs the reference on the REAL shape;
        speed is ``nki.benchmark(kernel, inputs, n_iterations=100)`` for the
        kernel vs a torch-eager baseline. Any failure degrades to a recorded
        reason, never a crash — an un-compilable invented kernel is the common
        case and must be survivable.
        """
        # Beta-3: shape-keyed trace cache would survive a source fix — force it
        # off in-process too (build() also sets this; belt-and-suspenders in case
        # the caller imported nki before build() ran).
        os.environ["NKI_ENABLE_TRACE_CACHE"] = "0"
        fn = author.build()
        if fn is None:
            return RaceResult(False, reason=(
                "kernel not built (off-device: no nki) — on-device race deferred"
                if not nki_available() else
                "kernel failed to build/trace on device"))
        # A speed race is only meaningful when BOTH contenders are measured the
        # SAME way on the SAME device. Establish the Neuron device handle up
        # front: if we cannot (no torch_xla / not really on a device), then we
        # can only wallclock the torch baseline on CPU while the kernel runs on
        # device — a physically meaningless CPU-vs-device ratio biased toward the
        # kernel. Rather than fabricate that "win", we DEFER (ran=False), exactly
        # as we do when the kernel cannot build. Honesty over a banked artifact.
        device = _neuron_device()
        if device is None:
            return RaceResult(False, reason=(
                "kernel built but no Neuron device handle for a fair "
                "same-device, same-method race — deferred (never a "
                "CPU-baseline-vs-device-kernel speedup)"))
        try:
            import torch  # noqa: PLC0415 — device-only import
            import nki      # noqa: PLC0415
            # NOTE: do NOT import ``torch_neuronx.nki_hop`` here. It was imported
            # eagerly (unused) and on torch-neuronx 2.9 the ``nki_hop`` module no
            # longer exists, so the import raised ImportError and aborted the race
            # before ANY real device work — turning a healthy box into a recorded
            # "device race error". The direct-call invocation path (see
            # _invoke_kernel) needs nothing from nki_hop; the only place that
            # still touches it is a LABELLED fallback, guarded there.

            inp = spec.real_inputs()
            ref = np.asarray(spec.reference(inp), dtype=np.float32)

            def _to_dev(a: np.ndarray):
                # Move onto the SAME device the kernel runs on — the baseline is
                # compared against the kernel there, not on the host.
                return (torch.from_numpy(np.ascontiguousarray(a))
                        .to(torch.bfloat16).to(device))

            # Positional args in the kernel's declared order (see _arg_order),
            # mirroring how the proven moe_fused kernels are invoked
            # (get_multilayer_kernel_jit(L)[2](*args) — a direct positional call
            # of the @nki.jit callable). See _invoke_kernel for why we call the
            # jit'd fn directly instead of wrap_nki(kernel)[1](**kwargs).
            args = [_to_dev(inp[k]) for k in _arg_order(spec.name, inp)]

            out = _invoke_kernel(fn, args)
            got = out.to(torch.float32).cpu().numpy()
            correct = bool(np.allclose(got, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)) \
                if got.shape == ref.shape else False
            # correctness pct: fraction of elements within tolerance.
            if got.shape == ref.shape:
                within = np.isclose(got, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)
                corr_pct = 100.0 * float(np.mean(within))
            else:
                corr_pct = 0.0

            # FAIR race: time BOTH the authored kernel AND the torch baseline
            # with the SAME synchronized on-device wallclock, on tensors resident
            # on the SAME device. (We use on-device wallclock for both rather
            # than nki.benchmark, because nki.benchmark can only time an nki
            # kernel — not the torch-eager baseline — so it cannot be applied
            # symmetrically. Symmetry is the whole point.)
            def _run_kernel():
                _invoke_kernel(fn, args)

            def _run_baseline():
                with torch.no_grad():
                    _torch_baseline(spec.name, inp, device=device)

            kernel_ms = _device_timed_ms(_run_kernel, device)
            baseline_ms = _device_timed_ms(_run_baseline, device)
            speedup = _fair_speedup(kernel_ms, baseline_ms,
                                    "wallclock@device", "wallclock@device")
            if speedup is None:
                # Timings were not comparable (non-positive / not same-method
                # same-device) — defer instead of banking a meaningless ratio.
                return RaceResult(False, reason=(
                    f"fair on-device timing failed (kernel={kernel_ms:.3f}ms, "
                    f"baseline={baseline_ms:.3f}ms) — deferred"))
            return RaceResult(True, correct, corr_pct, speedup,
                              kernel_ms, baseline_ms,
                              reason=f"correct={correct} speedup={speedup:.3f}x")
        except Exception as e:  # noqa: BLE001 — device errors are data
            return RaceResult(True, False, 0.0, 0.0,
                              reason=f"device race error: {e!r}")

    # -- banking -------------------------------------------------------------

    def _bank_win(self, spec: OpSpec, race: RaceResult) -> str:
        """A correct, >=5%-faster invented kernel -> provisional NKI_KERNEL lesson.

        Keyed by op + family + shape-class. Records ``beat_borrowed_by`` (the
        fraction over the raced baseline) so the bank's auto-promotion policy can
        apply the invented-margin gate honestly. Tier is PROVISIONAL: an
        invented kernel is trusted by later models only after promotion.
        """
        lesson_id = f"invented-{spec.name}-{spec.shape_class}"
        lesson = Lesson(
            lesson_id=lesson_id,
            type=LessonType.NKI_KERNEL,
            applicability=Applicability(
                architecture_family=spec.family,
                neuron_sdk_versions=[f"{_minor_glob(self.sdk_version)}"],
            ),
            layer=Layer.KERNEL,
            migration_risk="low",
            origin=Origin.INVENTED,
            tier=Tier.PROVISIONAL,
            intervention={"spec": {"nki_kernel": spec.name,
                                   "shape_class": spec.shape_class}},
            reason=(
                f"Invented NKI kernel for {spec.name} ({spec.shape_class}): "
                f"correct at bf16 tol and {race.speedup:.2f}x the {spec.baseline} "
                f"baseline on device ({race.kernel_ms:.3f}ms vs "
                f"{race.baseline_ms:.3f}ms). Authored from scratch via the 7-step "
                f"pipeline; beat the baseline by "
                f"{(race.speedup - 1.0) * 100:.1f}% (>= 5% invention margin)."),
            symptoms_addressed=[Symptom(
                bottleneck="compute_bound",
                signature=f"{spec.name} op is a hot, fusable site",
                observed_via="op-level benchmark vs eager baseline")],
            source="invent-engine",
            confidence=Confidence(n_models_validated=1, architecture_diversity=1,
                                  human_verified=False),
            last_reverified_sdk=self.sdk_version,
            evidence=[{"op": spec.name, "shape_class": spec.shape_class,
                       "speedup": round(race.speedup, 4),
                       "kernel_ms": round(race.kernel_ms, 4),
                       "baseline_ms": round(race.baseline_ms, 4),
                       "correctness_pct": round(race.correctness_pct, 3),
                       "baseline": spec.baseline}],
            backend_validated=["native-pytorch-beta3"],
            beat_borrowed_by=round(race.speedup - 1.0, 4),
        )
        self.bank.save(lesson)
        return lesson_id

    def _bank_anti_pattern(self, spec: OpSpec, reason: str,
                           race: RaceResult | None = None,
                           diagnosis: str = "") -> str:
        """A wrong / slow / un-buildable invented kernel -> provisional anti-pattern.

        No ``matcher`` on purpose: this is a recorded WARNING ("we tried an
        invented {op} kernel of this shape-class and it did not beat eager"),
        not a hard pre-prune — a future SDK or a better formulation may change
        the answer, so the loss is remembered but does not silently block a
        retry. Losses are data.
        """
        lesson_id = f"antipattern-invented-{spec.name}-{spec.shape_class}"
        detail = reason
        if race is not None and race.ran:
            detail = (f"{reason} (correct={race.correct}, "
                      f"speedup={race.speedup:.3f}x, "
                      f"kernel={race.kernel_ms:.3f}ms, base={race.baseline_ms:.3f}ms)")
        # Append the rewrite-catalog diagnosis (if any) so the banked warning is
        # actionable — the next author reads a known fix, not just "it failed".
        detail = f"{detail}{diagnosis}"
        lesson = Lesson(
            lesson_id=lesson_id,
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability(
                architecture_family=spec.family,
                neuron_sdk_versions=[f"{_minor_glob(self.sdk_version)}"],
            ),
            layer=Layer.KERNEL,
            migration_risk="low",
            origin=Origin.INVENTED,
            tier=Tier.PROVISIONAL,
            reason=(f"Invented NKI kernel for {spec.name} ({spec.shape_class}) "
                    f"did not win: {detail}. Recorded as a warning, not a "
                    f"pre-prune — retry allowed on a new SDK / formulation."),
            confidence=Confidence(n_models_validated=1, architecture_diversity=1,
                                  human_verified=False),
            last_reverified_sdk=self.sdk_version,
            evidence=[{"op": spec.name, "shape_class": spec.shape_class,
                       "reason": reason,
                       "speedup": round(race.speedup, 4) if race else None,
                       "correctness_pct": round(race.correctness_pct, 3)
                       if race else None}],
            backend_validated=["native-pytorch-beta3"],
        )
        self.bank.save(lesson)
        return lesson_id

    # -- ledger --------------------------------------------------------------

    def _record(self, spec: OpSpec, status: Status, metric: float,
                correctness: float, desc: str, origin: Origin = Origin.INVENTED,
                n_lessons: int = 0) -> None:
        # Surface how many banked lessons informed this op (learn-from-the-bank
        # step). Prefix only when >0 so records with no relevant prior are
        # byte-for-byte unchanged.
        prefix = f"[lessons:{n_lessons}] " if n_lessons else ""
        self.ledger.append(Row(
            commit=current_commit(self.out_dir),
            stage=Stage.INVENT, origin=origin, layer=Layer.KERNEL,
            source="invent-engine", metric=metric, mfu=-1.0,
            correctness=correctness, compile_s=0.0, status=status,
            description=f"{spec.name}/{spec.shape_class}: {prefix}{desc}",
        ))

    # -- the loop ------------------------------------------------------------

    def run_op(self, spec: OpSpec, race_fn: RaceFn | None = None,
               compile_fn: "CompileFnT | None" = None) -> InventResult:
        """Learn (retrieve) -> Prior-art (Harvest) -> author -> offline gate ->
        on-device race -> keep/discard -> bank.

        ``compile_fn`` is the seam the repair loop compiles through (only used
        when ``max_repair_rounds > 1``). It defaults to the engine's own
        ``_compile`` (offline gate + on-device build), and is injectable the same
        way ``race_fn`` is, so the repair loop is unit-testable off-device with a
        deterministic stand-in compiler."""
        # LEARN FIRST: retrieve previously-banked lessons (anti-patterns / prior
        # wins) relevant to this op so the bank is READ, not just written. The
        # count is recorded on the ledger row + result; the lessons themselves
        # are handed to the author (which the recipe author ignores today; the
        # LLM author consumes them). This is the "compounding" step.
        lessons = self._retrieve_lessons(spec)
        n = len(lessons)

        # HARVEST FIRST (Harvest -> Borrow -> Invent): if the corpus already has
        # a usable kernel for this op's primitive, REUSE it — do not spend a
        # compile re-inventing what exists. Recorded as a HARVESTED keep so the
        # ledger shows the reuse (and its HW-readiness tier) honestly.
        prior = self._prior_art(spec)
        if prior is not None:
            tier = "on-device" if prior.hw_ready else "simulate"
            self._record(spec, Status.KEEP, 0.0, 100.0,
                         f"harvested existing {prior.name} kernel "
                         f"({prior.status}, {tier}-validated) -> reuse, no authoring",
                         origin=Origin.HARVESTED, n_lessons=n)
            return InventResult(spec.name, spec.shape_class, "harvested",
                                "harvested",
                                OfflineGate(True, False, 0.0,
                                            reason=f"prior art: {prior.name}"),
                                RaceResult(False, reason="harvested (not raced)"),
                                detail=f"reused {prior.name} [{prior.status}]",
                                lessons_consulted=n)

        # REAL repair loop (only when asked). With the default max_repair_rounds=1
        # this branch is skipped entirely and authoring is the single-shot path
        # below — byte-for-byte today's behaviour.
        if self.max_repair_rounds and self.max_repair_rounds > 1:
            return self._run_op_with_repair(spec, lessons, n, race_fn, compile_fn)

        # SINGLE-SHOT (default): author once through the seam with no feedback.
        # RecipeAuthor forwards ``lessons`` to ``author_kernel`` exactly as before.
        author = self.author.author(spec, lessons, [])
        return self._finish(spec, author, n, race_fn)

    def _finish(self, spec: OpSpec, author: AuthoredKernel, n: int,
                race_fn: RaceFn | None) -> InventResult:
        """Shared tail: offline gate -> on-device race -> keep/discard -> bank.

        Extracted verbatim from the original single-shot ``run_op`` so BOTH the
        single-shot and the repaired-kernel paths run the SAME gates (offline
        parity + lint, on-device race, 5% invention margin) and produce the SAME
        honest outcomes. Behaviour for the single-shot path is unchanged."""
        if not author.nki_src:
            self._record(spec, Status.DISCARD, 0.0, 0.0,
                         f"no author available ({author.pipeline_notes})",
                         origin=Origin.NONE, n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "no_author",
                                OfflineGate(False, False, float("inf"),
                                            reason="no author"),
                                RaceResult(False, reason="no author"),
                                detail=author.pipeline_notes, lessons_consulted=n)

        offline = self.offline_gate(author, spec)
        if not offline.passed:
            # Diagnose the offline-reject reason with the rewrite catalog.
            desc_sfx, reason_sfx = self._diagnose_failure(offline.reason)
            lid = self._bank_anti_pattern(
                spec, f"offline gate: {offline.reason}", diagnosis=reason_sfx)
            self._record(spec, Status.DISCARD, 0.0,
                         100.0 if offline.parity_ok else 0.0,
                         f"offline reject: {offline.reason}{desc_sfx}", n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "offline_reject", offline,
                                RaceResult(False, reason="offline reject"),
                                lesson_id=lid, detail=f"{offline.reason}{desc_sfx}",
                                lessons_consulted=n)

        race = (race_fn or self._device_race)(author, spec)

        if not race.ran:
            # Off-device (or un-buildable): offline gate passed, device deferred.
            # NOT a win and NOT an anti-pattern — honestly "not yet raced".
            self._record(spec, Status.DISCARD, 0.0, 100.0,
                         f"offline pass; on-device race deferred ({race.reason})",
                         n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "device_deferred", offline, race,
                                detail=race.reason, lessons_consulted=n)

        if not race.correct:
            # Diagnose the on-device failure (compiler/error string) with the
            # rewrite catalog — an opaque "wrong" becomes an actionable fix.
            desc_sfx, reason_sfx = self._diagnose_failure(race.reason)
            lid = self._bank_anti_pattern(
                spec, "incorrect on device", race, diagnosis=reason_sfx)
            self._record(spec, Status.DISCARD, race.speedup,
                         race.correctness_pct,
                         f"WRONG on device ({race.correctness_pct:.1f}% within tol)"
                         f"{desc_sfx}", n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "anti_pattern", offline, race,
                                lesson_id=lid, detail=f"incorrect on device{desc_sfx}",
                                lessons_consulted=n)

        # Correct — now the speed race with the 5% invention margin.
        is_win = self.guards.is_improvement(race.speedup, 1.0, is_invention=True)
        if is_win:
            lid = self._bank_win(spec, race)
            self._record(spec, Status.KEEP, race.speedup, race.correctness_pct,
                         f"WIN: {race.speedup:.3f}x vs {spec.baseline} "
                         f"(>= 5% margin)", n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "win", offline, race, lesson_id=lid,
                                detail=f"{race.speedup:.3f}x", lessons_consulted=n)
        lid = self._bank_anti_pattern(
            spec, f"correct but only {race.speedup:.3f}x (< 5% margin)", race)
        self._record(spec, Status.DISCARD, race.speedup, race.correctness_pct,
                     f"correct-but-slow: {race.speedup:.3f}x (< 5% margin)",
                     n_lessons=n)
        return InventResult(spec.name, spec.shape_class, spec.origin,
                            "anti_pattern", offline, race, lesson_id=lid,
                            detail=f"correct but {race.speedup:.3f}x < 1.05x",
                            lessons_consulted=n)

    # -- the REAL repair loop (author -> compile -> read-error -> re-author) --

    def _compile(self, kernel: AuthoredKernel, spec: OpSpec) -> CompileResult:
        """Default compile step the repair loop drives, mapping the engine's
        offline + on-device gates onto a ``CompileResult``:

          * Offline gate FIRST (static NKI lint + numpy_impl smoke/parity). A
            lint or shape failure is a compile-blocking error whose reason IS the
            teacher fed back to the next author round — no device time is spent
            on a kernel that cannot even pass the text/shape checks.
          * On device (``nki_available()``): ``build()`` the kernel (import/trace);
            a None result is a real build/trace failure (the "entry function not
            found" class), reported as the error_log. Then run a REAL neuronx-cc
            compile — ``build()`` alone only IMPORTS the module and a ``@nki.jit``
            fn is lowered by neuronx-cc lazily on its FIRST invocation, so a real
            "failed to resolve name"/ISA-validation error would otherwise ESCAPE
            the repair window and die at race time instead of teaching a round-2
            rewrite. ``_device_compile_probe`` forces that lowering and returns
            the compiler error string, which becomes the ``error_log`` the
            ``KernelRepairLoop`` feeds back to the author.
          * Off device: there is no neuronx-cc to run, so an offline-gate PASS is
            the honest best-effort "compiles as far as we can check here" — the
            true device compile is deferred and surfaces downstream as the
            on-device race's ``ran=False`` (device_deferred). Tests inject their
            own ``compile_fn`` to exercise the loop deterministically on CPU.
        """
        offline = self.offline_gate(kernel, spec)
        if not offline.passed:
            return CompileResult(False, error_log=f"offline gate: {offline.reason}")
        if nki_available():
            fn = kernel.build()
            if fn is None:
                return CompileResult(
                    False,
                    error_log=f"device build/trace failed: entry "
                              f"'{kernel.entry}' did not resolve")
            # build() only imported/traced. Force the REAL neuronx-cc lowering so
            # a compile error ("failed to resolve name", ISA validation, ...) is
            # caught INSIDE the repair window and fed back — not at race time.
            compile_err = self._device_compile_probe(fn, spec)
            if compile_err is not None:
                return CompileResult(
                    False, error_log=f"device compile failed: {compile_err}")
            return CompileResult(True, artifact=kernel.entry)
        return CompileResult(True, artifact=f"offline-only:{kernel.entry}")

    def _device_compile_probe(self, fn: Callable, spec: OpSpec) -> str | None:
        """Force a REAL neuronx-cc compile of a built ``@nki.jit`` kernel and
        return the compiler error string (the teacher), or ``None`` if it
        compiled (or could not be probed here).

        Why this exists: a ``@nki.jit`` fn is only lowered by neuronx-cc on its
        FIRST invocation, so ``build()`` (import/trace) succeeds even when the
        kernel will NOT compile. We trigger the lowering by invoking the kernel
        once on device (the SAME proven direct-call path ``_device_race`` uses)
        and capture any compiler error verbatim.

        Returns ``None`` (best-effort "cannot probe — treat build() as far as we
        got") when there is no Neuron device handle or ``torch`` is unavailable:
        without them we cannot compile-invoke, and fabricating an error would be
        dishonest. Device-only, exactly like ``_device_race`` — not exercised on
        a CPU box; tests drive ``_compile`` with a monkeypatched probe.
        """
        device = _neuron_device()
        if device is None:
            return None
        try:
            import torch  # noqa: PLC0415 — device-only import
        except ImportError:
            return None
        try:
            inp = spec.real_inputs()

            def _to_dev(a: np.ndarray):
                return (torch.from_numpy(np.ascontiguousarray(a))
                        .to(torch.bfloat16).to(device))

            args = [_to_dev(inp[k]) for k in _arg_order(spec.name, inp)]
            _invoke_kernel(fn, args)   # forces neuronx-cc lowering (the compile)
        except Exception as e:  # noqa: BLE001 — compiler errors are the teacher
            return repr(e)
        return None

    def _run_op_with_repair(self, spec: OpSpec, lessons: list, n: int,
                            race_fn: RaceFn | None,
                            compile_fn: CompileFnT | None) -> InventResult:
        """Drive authoring through ``KernelRepairLoop`` so a compile failure
        TEACHES the next attempt (the exact error + the matched rewrite fed back
        via the author's ``feedback`` arg). On convergence the compiled kernel
        goes through the SAME ``_finish`` gates as single-shot; on a
        non-converging loop (exhausted / stalled) the failure is banked as an
        anti-pattern (losses are data), diagnosed with the rewrite catalog."""
        loop = KernelRepairLoop(max_rounds=self.max_repair_rounds)
        _compile = compile_fn or (lambda k: self._compile(k, spec))

        def author_fn(trail: list[Feedback]) -> AuthoredKernel:
            return self.author.author(spec, lessons, trail)

        outcome = loop.run(author_fn, _compile)

        if not outcome.ok:
            last_err = outcome.trail[-1].error_log if outcome.trail else ""
            desc_sfx, reason_sfx = self._diagnose_failure(last_err)
            suggested = ", ".join(r.name for r in outcome.suggested_rewrites)
            reason = (f"kernel repair did not converge in {outcome.rounds} "
                      f"round(s) ({outcome.reason})"
                      + (f"; suggested rewrites: {suggested}" if suggested else ""))
            lid = self._bank_anti_pattern(spec, reason, diagnosis=reason_sfx)
            self._record(spec, Status.DISCARD, 0.0, 0.0,
                         f"repair failed: {reason}{desc_sfx}", n_lessons=n)
            return InventResult(
                spec.name, spec.shape_class, spec.origin, "offline_reject",
                OfflineGate(False, False, float("inf"), reason=reason),
                RaceResult(False, reason=outcome.reason),
                lesson_id=lid, detail=f"{reason}{desc_sfx}", lessons_consulted=n)

        # Compiled after N rounds — gate + race + bank the repaired kernel.
        return self._finish(spec, outcome.kernel, n, race_fn)

    def run(self, specs: list[OpSpec],
            race_fn: RaceFn | None = None) -> list[InventResult]:
        results = [self.run_op(s, race_fn=race_fn) for s in specs]
        self._write_summary(results)
        return results

    # -- self-test (validate the EXECUTION path, not authoring quality) ------

    def self_test(self, seed: str = "silu_gate",
                  race_fn: RaceFn | None = None) -> tuple[InventResult, bool, str]:
        """Run a KNOWN-GOOD seed kernel through the full engine to validate the
        on-device EXECUTION path in isolation from authoring quality.

        Returns ``(result, executed, verdict)``. ``executed`` is the pass/fail
        the box cares about: on device it is True iff the kernel actually BUILT,
        INVOKED and was MEASURED (``race.ran`` and no "entry function not found"
        wall); off device it is True as a graceful "deferred" (the CPU-mock path
        cannot run a kernel and that is expected, not a failure).

        A seed (``silu_gate`` / ``rmsnorm`` / ``softmax``) is used on purpose: it
        is a proven-correct formulation, so if IT fails to execute the fault is
        the invocation path, not the authored math. Only once a seed EXECUTES do
        novel kernels have a real shot.
        """
        spec = resolve_ops([seed])[0]
        res = self.run_op(spec, race_fn=race_fn)
        on_device = nki_available()
        if not on_device:
            return res, True, (
                f"OFF-DEVICE: seed {seed!r} authored + offline-gated + "
                f"device-race deferred (status={res.status}); run on trn2 to "
                f"exercise the real build+invoke+measure path")
        executed = _executed_on_device(res)
        if executed:
            verdict = (
                f"ON-DEVICE PASS: seed {seed!r} EXECUTED and was measured "
                f"(status={res.status}, ran={res.race.ran}, "
                f"correct={res.race.correct}, speedup={res.race.speedup:.3f}x) "
                f"— the 'entry function not found' wall is cleared")
        else:
            verdict = (
                f"ON-DEVICE FAIL: seed {seed!r} did NOT execute "
                f"(status={res.status}, ran={res.race.ran}); "
                f"reason={res.race.reason!r}")
        return res, executed, verdict

    # -- reporting -----------------------------------------------------------

    def _write_summary(self, results: list[InventResult]) -> None:
        summary = {
            "on_device": nki_available(),
            "sdk_version": self.sdk_version,
            "invention_margin_pct": self.guards.invention_margin_pct,
            "n_ops": len(results),
            "wins": [r.op for r in results if r.status == "win"],
            "anti_patterns": [r.op for r in results if r.status == "anti_pattern"],
            "offline_rejects": [r.op for r in results
                                if r.status == "offline_reject"],
            "device_deferred": [r.op for r in results
                                if r.status == "device_deferred"],
            "no_author": [r.op for r in results if r.status == "no_author"],
            "results": [
                {
                    "op": r.op, "shape_class": r.shape_class, "origin": r.origin,
                    "status": r.status, "lesson_id": r.lesson_id,
                    "offline_passed": r.offline.passed,
                    # Only report a parity error number when it was an INDEPENDENT
                    # check; a tautological (numpy_impl is reference) comparison
                    # verified no math, so its "0.0" would be misleading -> null.
                    "offline_parity_independent": r.offline.parity_independent,
                    "offline_parity_max_abs_err": (
                        None if (not r.offline.parity_independent
                                 or r.offline.parity_max_abs_err == float("inf"))
                        else r.offline.parity_max_abs_err),
                    "offline_lint": r.offline.lint_violations,
                    "race_ran": r.race.ran,
                    "correct": r.race.correct,
                    "correctness_pct": r.race.correctness_pct,
                    "speedup": r.race.speedup,
                    "detail": r.detail,
                }
                for r in results
            ],
        }
        (self.out_dir / "invent_summary.json").write_text(
            json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# on-device helpers (only reached on trn2)
# ---------------------------------------------------------------------------
def _executed_on_device(res: InventResult) -> bool:
    """True iff the on-device race actually RAN and did not hit the entry wall.

    A win, a correct-but-slow anti-pattern, or a wrong-but-ran anti-pattern all
    count as EXECUTED — the point of the self-test is "did the kernel build,
    invoke and get measured", not "did it win". The one thing that is NOT
    executed is the very failure this fix targets: an "entry function ... not
    found" (or any un-run) race.
    """
    race = res.race
    if not race.ran:
        return False
    reason = (race.reason or "").lower()
    if "entry function" in reason and "not found" in reason:
        return False
    return True


def _arg_order(op: str, inp: dict) -> list[str]:
    """Positional arg order each authored kernel expects (matches nki_src)."""
    order = {
        "rope_apply": ["x", "cos", "sin"],
        "gelu_tanh": ["x"],
        "softcap": ["x", "cap"],
        "add_rmsnorm": ["x", "residual", "gamma"],
        "layernorm": ["x", "gamma", "beta"],
        "attn_decode": ["q", "k", "v"],
        "rmsnorm": ["x", "gamma"],
        "silu_gate": ["x"],
        "softmax": ["x"],
    }.get(op)
    return order if order else list(inp.keys())


def _invoke_kernel(fn: Callable, args: list):
    """Invoke a jitted kernel via the PROVEN beta-3 path: call it directly.

    This routes authored kernels through the SAME mechanism the working
    (moe_fused) kernels use. In this codebase a compiled @nki.jit kernel is run
    by CALLING THE JIT'D CALLABLE DIRECTLY on device tensors and taking element
    ``[0]`` of its (possibly tuple) result — exactly the shape of the proven
    invocation in ``kernels/moe_fused/qwen_with_megakernel.py``:

        kernel_out = get_multilayer_kernel_jit(L)[2](hidden_states, *weights, ...)
        Y = kernel_out[0]

    i.e. the jit builder hands back a callable that is invoked positionally, and
    its output is a sequence whose first element is the result tensor.

    We deliberately do NOT use the previous bespoke
    ``torch_neuronx.nki_hop.wrap_nki(kernel)[1](**kwargs)`` path: an authored
    kernel put through ``wrap_nki`` produced ``entry function
    '<module>.<fn>_kernel' not found`` on every kernel. Calling the ``@nki.jit``
    fn directly (now that ``build()`` gives it a real, importable module + file)
    is the path the compiler can resolve. ``wrap_nki`` is kept ONLY as a labeled
    fallback for the case where a direct call is not supported by the installed
    ``nki`` build; the fallback error (if any) is surfaced verbatim so a real
    "entry not found" is never silently masked. We do NOT use
    ``nki.baremetal`` / ``nki.simulate_kernel`` — those are offline sim only.
    Any failure propagates to ``_device_race``'s handler, which records it as
    data rather than crashing.
    """
    try:
        out = fn(*args)
    except TypeError:
        # Some jit builds return a (spec, meta, callable)-style tuple rather
        # than a directly-callable kernel; try the last callable element, then
        # fall back to the legacy wrap_nki path with a clear provenance tag.
        called = _try_tuple_callable(fn, args)
        if called is not _NO_CALL:
            out = called
        else:
            # Labelled fallback ONLY. ``wrap_nki`` lives in ``torch_neuronx.nki_hop``,
            # which was REMOVED in torch-neuronx 2.9 — guard the optional import so a
            # missing module never aborts the whole race with an ImportError. When
            # it is absent we cannot take this fallback, so surface a clear (non-
            # ImportError) RuntimeError that _device_race records as data.
            try:
                from torch_neuronx.nki_hop import wrap_nki  # noqa: PLC0415
            except ImportError:
                wrap_nki = None
            if wrap_nki is None:
                raise RuntimeError(
                    "authored kernel not directly callable and the wrap_nki "
                    "fallback is unavailable (torch_neuronx.nki_hop removed in "
                    "torch-neuronx 2.9) — no invocation path")
            wrapped = wrap_nki(fn)
            out = wrapped[1](*args)
    return out[0] if isinstance(out, (list, tuple)) else out


_NO_CALL = object()


def _try_tuple_callable(fn: Callable, args: list):
    """If ``fn`` is a jit builder returning a tuple, call its last callable.

    Mirrors the ``get_multilayer_kernel_jit(L)[2](...)`` idiom without hardcoding
    the index: pick the last callable element of the returned tuple. Returns
    ``_NO_CALL`` if ``fn`` is not a tuple-returning builder.
    """
    try:
        maybe = fn
        if isinstance(maybe, (list, tuple)):
            callables = [e for e in maybe if callable(e)]
            if callables:
                return callables[-1](*args)
    except Exception:  # noqa: BLE001
        pass
    return _NO_CALL


def _neuron_device():
    """Return the torch_xla Neuron device handle, or None if unavailable.

    None is the honest "we are not really on a device" signal: without a device
    handle we cannot place the torch baseline on the SAME device the kernel runs
    on, so a fair race is impossible and the caller must defer rather than
    compare a CPU wallclock to a device latency. Off-device this simply returns
    None (no torch_xla); the CPU-mock harness never reaches here because
    ``build()`` already returned None.
    """
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
        return xm.xla_device()
    except Exception:  # noqa: BLE001 — no torch_xla / no device is just "defer"
        return None


def _device_timed_ms(run: Callable, device, iters: int = 100,
                      warmup: int = 5) -> float:
    """Synchronized on-device wallclock, ms/iter — the SAME method used for BOTH
    the authored kernel and the torch baseline so the ratio is apples-to-apples.

    The per-batch ``mark_step`` + ``wait_device_ops`` is what turns an on-device
    wallclock into a real device-latency measurement rather than async-dispatch
    noise: without the barrier the enqueue returns immediately and we would be
    timing Python dispatch, not the device. Returns 0.0 on any failure so the
    caller's ``_fair_speedup`` guard turns a broken measurement into a defer.
    """
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415

        for _ in range(warmup):
            run()
        xm.mark_step()
        xm.wait_device_ops()
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        xm.mark_step()
        xm.wait_device_ops()
        return (time.perf_counter() - t0) / iters * 1000.0
    except Exception:  # noqa: BLE001 — a failed measurement must not fabricate a number
        return 0.0


def _fair_speedup(kernel_ms: float, baseline_ms: float,
                  kernel_timing: str, baseline_timing: str) -> float | None:
    """Speedup = baseline/kernel, but ONLY when the two measurements are
    comparable. Returns None otherwise, forcing the caller to device_deferred.

    Comparable means: measured by the SAME method AND both taken ON the device
    (label convention ``"<method>@device"``). This is the guard against the
    exact pre-fix bug — kernel timed by ``nki.benchmark`` DEVICE latency while
    the baseline was a CPU (``@cpu``) wallclock — which produced a physically
    meaningless ratio biased toward the kernel and could bank a FALSE win. A
    non-positive timing is likewise not a real measurement and yields None.
    """
    if kernel_timing != baseline_timing:
        return None
    if not kernel_timing.endswith("@device"):
        return None
    if kernel_ms <= 0.0 or baseline_ms <= 0.0:
        return None
    return baseline_ms / kernel_ms


def _torch_baseline(op: str, inp: dict, device=None):
    """Torch-eager reference the kernel must beat. Mirrors the numpy reference.

    ``device`` (a torch_xla Neuron device) is REQUIRED for a fair race: the
    baseline tensors are placed on the same device the kernel runs on so both
    sides are timed on-device. It defaults to None only for callers that want
    the pure host computation.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def t(name):
        x = torch.from_numpy(np.ascontiguousarray(inp[name])).to(torch.bfloat16)
        return x.to(device) if device is not None else x

    if op == "rope_apply":
        x, cos, sin = t("x"), t("cos"), t("sin")
        x1, x2 = x[..., 0::2], x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.stack([o1, o2], dim=-1).flatten(-2)
    if op in ("gelu_tanh",):
        x = t("x")
        f = x.shape[-1] // 2
        return F.gelu(x[..., :f], approximate="tanh") * x[..., f:]
    if op == "softcap":
        x = t("x")
        cap = float(inp["cap"][0])
        return torch.tanh(x / cap) * cap
    if op == "add_rmsnorm":
        x, r, g = t("x"), t("residual"), t("gamma")
        h = x + r
        ms = h.pow(2).mean(-1, keepdim=True)
        return h * torch.rsqrt(ms + 1e-6) * g
    if op == "layernorm":
        x, g, b = t("x"), t("gamma"), t("beta")
        return F.layer_norm(x.float(), (x.shape[-1],), g.float(), b.float(),
                            1e-6).to(torch.bfloat16)
    if op == "attn_decode":
        q, k, v = t("q"), t("k"), t("v")
        return F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
    if op == "rmsnorm":
        x, g = t("x"), t("gamma")
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + 1e-6) * g
    if op == "silu_gate":
        x = t("x")
        f = x.shape[-1] // 2
        return F.silu(x[..., :f]) * x[..., f:]
    if op == "softmax":
        return torch.softmax(t("x"), dim=-1)
    raise KeyError(op)


def _minor_glob(sdk: str) -> str:
    """"2.28.0" -> "2.28.*" so the banked lesson is SDK-stamped (bank requires it)."""
    parts = sdk.split(".")
    return f"{parts[0]}.{parts[1]}.*" if len(parts) >= 2 else sdk


# ---------------------------------------------------------------------------
# spec-file loader — point the engine at an arbitrary NEW op over time.
# ---------------------------------------------------------------------------
def load_specs_from_file(path: Path | str) -> list[OpSpec]:
    """Load OpSpecs from a user .py spec file.

    The file may expose either:
      * ``SPECS``      : list[OpSpec], or
      * ``op_specs()`` : callable returning list[OpSpec].

    A new op only needs a small spec (name + reference fn + shapes); to actually
    author a kernel for it, register a recipe in ``invent_kernels`` (or attach an
    author) — otherwise the engine records it honestly as ``no_author``.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location("invent_user_specs", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec file {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "op_specs") and callable(mod.op_specs):
        specs = list(mod.op_specs())
    elif hasattr(mod, "SPECS"):
        specs = list(mod.SPECS)
    else:
        raise AttributeError(
            f"{path} defines neither SPECS nor op_specs(); one is required")
    for s in specs:
        if not isinstance(s, OpSpec):
            raise TypeError(f"spec file yielded a non-OpSpec: {s!r}")
    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(results: list[InventResult], out_dir: Path,
                  on_device: bool) -> None:
    print("\n=== Stage-4 INVENT run ===")
    print(f"mode: {'ON-DEVICE (trn2)' if on_device else 'CPU-mock (offline gate only; device race deferred)'}")
    for r in results:
        line = f"  [{r.status:>15}] {r.op:<12} ({r.shape_class})"
        if r.offline.passed:
            line += "  offline:PASS"
        else:
            line += f"  offline:REJECT ({r.offline.reason})"
        if r.race.ran:
            line += f"  correct={r.race.correct} speedup={r.race.speedup:.3f}x"
        if r.lesson_id:
            line += f"  banked={r.lesson_id}"
        print(line)
    wins = [r for r in results if r.status == "win"]
    anti = [r for r in results if r.status == "anti_pattern"]
    print(f"\nsummary: {len(wins)} win(s), {len(anti)} anti-pattern(s), "
          f"{sum(1 for r in results if r.status == 'device_deferred')} deferred, "
          f"{sum(1 for r in results if r.status == 'offline_reject')} offline-reject, "
          f"{sum(1 for r in results if r.status == 'no_author')} no-author")
    print(f"artifacts: {out_dir}/results.tsv, {out_dir}/invent_summary.json, "
          f"bank lessons under {out_dir}/knowledge-bank/provisional/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage-4 INVENT engine: author + gate + race + bank NKI kernels.")
    ap.add_argument("--ops", default="write-new",
                    help="comma list of ops or groups (all | write-new | seeds), "
                         "e.g. rope_apply,gelu_tanh,softcap,add_rmsnorm,layernorm,attn_decode")
    ap.add_argument("--out", required=True, type=Path,
                    help="run output dir (results.tsv, summary, bank)")
    ap.add_argument("--bank-root", type=Path, default=None,
                    help="knowledge-bank root (default: <out>/knowledge-bank)")
    ap.add_argument("--spec", type=Path, default=None,
                    help="optional .py spec file adding new ops (SPECS or op_specs())")
    ap.add_argument("--sdk", default=_SDK, help="neuron SDK version stamp")
    ap.add_argument(
        "--self-test", nargs="?", const="silu_gate", default=None,
        metavar="SEED",
        help="FIRST validate the on-device EXECUTION path on a KNOWN-GOOD seed "
             "kernel (default: silu_gate) — build + invoke + measure — to prove "
             "the 'entry function not found' wall is cleared, isolated from "
             "authoring quality. On device, exits non-zero if the seed does NOT "
             "execute; then continues to --ops only if the seed executed.")
    a = ap.parse_args(argv)

    import sys as _sys
    raw = list(_sys.argv[1:] if argv is None else argv)
    ops_explicit = any(t == "--ops" or t.startswith("--ops=") for t in raw)

    # Self-test gate: run a proven seed through the full engine first. If it
    # cannot even execute on device, novel ops cannot either — fail fast.
    if a.self_test is not None:
        st_engine = InventEngine(out_dir=a.out, bank_root=a.bank_root,
                                 sdk_version=a.sdk)
        _res, executed, verdict = st_engine.self_test(a.self_test)
        print("\n=== Stage-4 INVENT self-test (execution path) ===")
        print(f"  {verdict}")
        if nki_available() and not executed:
            print("  -> aborting: fix the invocation path before authoring novel "
                  "kernels.")
            return 1
        # A bare --self-test (no explicit --ops) is a pure execution-path check:
        # exit 0 on pass / deferred, so the box can gate on it. If --ops was
        # given, fall through and run those ops after the seed passed.
        if not ops_explicit:
            return 0

    specs: list[OpSpec] = []
    if a.spec:
        specs.extend(load_specs_from_file(a.spec))
    names = [n for n in a.ops.split(",") if n.strip()] if a.ops else []
    if names:
        # Merge catalog/group ops, skipping any already provided by the spec file.
        have = {s.name for s in specs}
        for s in resolve_ops(names):
            if s.name not in have:
                specs.append(s)
    if not specs:
        ap.error("no ops resolved (use --ops and/or --spec)")

    engine = InventEngine(out_dir=a.out, bank_root=a.bank_root, sdk_version=a.sdk)
    results = engine.run(specs)
    _print_report(results, engine.out_dir, nki_available())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
