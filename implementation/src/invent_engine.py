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
                         + speed race (nki.benchmark vs baseline)
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
import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
)
from guardrails import Guardrails
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
    parity_ok: bool
    parity_max_abs_err: float
    lint_violations: list[str] = field(default_factory=list)
    reason: str = ""


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
    status: str                    # win | anti_pattern | offline_reject |
                                   # device_deferred | no_author
    offline: OfflineGate
    race: RaceResult
    lesson_id: str = ""
    detail: str = ""


# A race function lets tests inject a deterministic device outcome. On device
# the engine's own ``_device_race`` is used.
RaceFn = Callable[[AuthoredKernel, OpSpec], RaceResult]


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
    ) -> None:
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
        # (1) numpy parity at 128x128.
        inp = spec.offline_inputs()
        try:
            got = np.asarray(author.numpy_impl(inp), dtype=np.float32)
            ref = np.asarray(spec.reference(inp), dtype=np.float32)
            max_err = float(np.max(np.abs(got - ref))) if got.shape == ref.shape \
                else float("inf")
            parity_ok = (got.shape == ref.shape) and np.allclose(
                got, ref, atol=_OFFLINE_ATOL, rtol=_OFFLINE_RTOL)
        except Exception as e:  # noqa: BLE001 — a math bug is a gate failure, not a crash
            return OfflineGate(False, False, float("inf"),
                               reason=f"numpy_impl raised: {e!r}")
        # (2) static lint.
        violations = static_lint(author.nki_src)
        passed = parity_ok and not violations
        reason = ""
        if not parity_ok:
            reason = f"numpy parity fail (max_abs_err={max_err:.3e})"
        elif violations:
            reason = f"lint: {'; '.join(violations)}"
        return OfflineGate(passed, parity_ok, max_err, violations, reason)

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
        try:
            import torch  # noqa: PLC0415 — device-only import
            import nki      # noqa: PLC0415
            from torch_neuronx import nki_hop  # noqa: PLC0415,F401

            inp = spec.real_inputs()
            ref = np.asarray(spec.reference(inp), dtype=np.float32)

            def _to_dev(a: np.ndarray):
                return torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16)

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

            kernel_ms = _benchmark_kernel(fn, args)
            baseline_ms = _benchmark_baseline(spec, inp)
            speedup = (baseline_ms / kernel_ms) if kernel_ms > 0 else 0.0
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
                           race: RaceResult | None = None) -> str:
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
                correctness: float, desc: str, origin: Origin = Origin.INVENTED) -> None:
        self.ledger.append(Row(
            commit=current_commit(self.out_dir),
            stage=Stage.INVENT, origin=origin, layer=Layer.KERNEL,
            source="invent-engine", metric=metric, mfu=-1.0,
            correctness=correctness, compile_s=0.0, status=status,
            description=f"{spec.name}/{spec.shape_class}: {desc}",
        ))

    # -- the loop ------------------------------------------------------------

    def run_op(self, spec: OpSpec, race_fn: RaceFn | None = None) -> InventResult:
        """Author -> offline gate -> on-device race -> keep/discard -> bank."""
        author = author_kernel(spec)

        if not author.nki_src:
            self._record(spec, Status.DISCARD, 0.0, 0.0,
                         f"no author available ({author.pipeline_notes})",
                         origin=Origin.NONE)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "no_author",
                                OfflineGate(False, False, float("inf"),
                                            reason="no author"),
                                RaceResult(False, reason="no author"),
                                detail=author.pipeline_notes)

        offline = self.offline_gate(author, spec)
        if not offline.passed:
            lid = self._bank_anti_pattern(spec, f"offline gate: {offline.reason}")
            self._record(spec, Status.DISCARD, 0.0,
                         100.0 if offline.parity_ok else 0.0,
                         f"offline reject: {offline.reason}")
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "offline_reject", offline,
                                RaceResult(False, reason="offline reject"),
                                lesson_id=lid, detail=offline.reason)

        race = (race_fn or self._device_race)(author, spec)

        if not race.ran:
            # Off-device (or un-buildable): offline gate passed, device deferred.
            # NOT a win and NOT an anti-pattern — honestly "not yet raced".
            self._record(spec, Status.DISCARD, 0.0, 100.0,
                         f"offline pass; on-device race deferred ({race.reason})")
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "device_deferred", offline, race,
                                detail=race.reason)

        if not race.correct:
            lid = self._bank_anti_pattern(spec, "incorrect on device", race)
            self._record(spec, Status.DISCARD, race.speedup,
                         race.correctness_pct,
                         f"WRONG on device ({race.correctness_pct:.1f}% within tol)")
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "anti_pattern", offline, race,
                                lesson_id=lid, detail="incorrect on device")

        # Correct — now the speed race with the 5% invention margin.
        is_win = self.guards.is_improvement(race.speedup, 1.0, is_invention=True)
        if is_win:
            lid = self._bank_win(spec, race)
            self._record(spec, Status.KEEP, race.speedup, race.correctness_pct,
                         f"WIN: {race.speedup:.3f}x vs {spec.baseline} "
                         f"(>= 5% margin)")
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "win", offline, race, lesson_id=lid,
                                detail=f"{race.speedup:.3f}x")
        lid = self._bank_anti_pattern(
            spec, f"correct but only {race.speedup:.3f}x (< 5% margin)", race)
        self._record(spec, Status.DISCARD, race.speedup, race.correctness_pct,
                     f"correct-but-slow: {race.speedup:.3f}x (< 5% margin)")
        return InventResult(spec.name, spec.shape_class, spec.origin,
                            "anti_pattern", offline, race, lesson_id=lid,
                            detail=f"correct but {race.speedup:.3f}x < 1.05x")

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
                    "offline_parity_max_abs_err": (
                        None if r.offline.parity_max_abs_err == float("inf")
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
            from torch_neuronx.nki_hop import wrap_nki  # noqa: PLC0415
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


def _benchmark_kernel(fn: Callable, args: list) -> float:
    """Milliseconds/iter for the kernel via nki.benchmark (n_iterations=100).

    Falls back to a wallclock of the real (direct-call) invocation if
    nki.benchmark is unavailable or does not accept this call shape.
    """
    try:
        import nki  # noqa: PLC0415
        res = nki.benchmark(fn, args, n_iterations=100)
        # nki.benchmark returns a profile-ish object; pull mean latency in ms.
        for attr in ("latency_ms", "mean_ms", "p50_ms"):
            v = getattr(res, attr, None)
            if isinstance(v, (int, float)):
                return float(v)
        if isinstance(res, (int, float)):
            return float(res)
    except Exception:  # noqa: BLE001
        pass
    return _wallclock_ms(lambda: _invoke_kernel(fn, args))


def _benchmark_baseline(spec: OpSpec, inp: dict) -> float:
    """Milliseconds/iter for the torch-eager baseline on the real shape."""
    try:
        import torch  # noqa: PLC0415

        def run():
            with torch.no_grad():
                _torch_baseline(spec.name, inp)
        return _wallclock_ms(run)
    except Exception:  # noqa: BLE001
        return 0.0


def _torch_baseline(op: str, inp: dict):
    """Torch-eager reference the kernel must beat. Mirrors the numpy reference."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def t(name):
        return torch.from_numpy(np.ascontiguousarray(inp[name])).to(torch.bfloat16)

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


def _wallclock_ms(fn: Callable, iters: int = 100, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


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
