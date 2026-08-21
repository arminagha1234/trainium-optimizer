"""
Bank hygiene — toolchain re-validation, the safeguard that keeps the knowledge
bank *improving* long-term instead of rotting into confidently-wrong.

Every config-prior is stamped with the toolchain that produced it
(`neuron_sdk_versions`, e.g. `2.28.*`). A config that is optimal on neuronx-cc
2.28 can be WRONG on 2.30 — the compiler changed underneath it. Across weekly
capacity blocks and SDK bumps, an un-checked bank silently degrades: the
proposer keeps seeding beams from priors nobody re-measured, and a stale-wrong
prior is worse than no prior at all.

Promotion (bank.auto_promote) only ever *adds* trust. This module is the other
half: it *withdraws* trust when the toolchain moves out from under a prior.

The pass, at a glance:

  1. current_toolchain()  — capture the live neuronx-cc version (best-effort;
     reuses the same importlib path the backends stamp with).
  2. stale_priors()       — VERIFIED priors whose SDK stamp no longer covers the
     live toolchain. These are the re-validation candidates.
  3. revalidate()         — re-apply each stale prior's config on a small canary
     model of the matching family+backend, through the EXISTING measure +
     equivalence gate. Still correct AND within margin of its recorded metric ->
     refresh its SDK stamp (revalidated). No longer wins, or fails equivalence ->
     DEMOTE verified->provisional and drop a `staleness` breadcrumb. A prior is
     never silently kept stale-wrong.

Demotion is deliberately conservative: it fires only on a *deterministic*
regression or equivalence failure (a real, reproducible "this config is now
wrong/slower"), never on a transient crash or a noisy measurement — the same
"deterministic only" judgment the pre-flight gate uses before recording an
anti-pattern.

Import-safe on CPU: nothing here imports torch or touches a device at import
time. The harness logic (stale detection, demote/refresh bookkeeping, the
report) runs against mock lessons on a plain laptop; the actual re-measurement
reuses the Orchestrator's measure+equivalence path and is only touched when a
real (or mock) backend is handed in.

See ../../guardrails.md ("Neuron SDK version tracking" / "Bank staleness
policy") and ../../knowledge-bank.md ("Post-migration re-verification query").
"""

from __future__ import annotations

import fnmatch
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Tier,
    _minor,
    _newest,
    _norm_family,
)
from guardrails import Guardrails

if TYPE_CHECKING:  # pragma: no cover - type hints only, never imported at runtime
    from orchestrator import ModelSpec


# How much slower than its recorded metric a prior may re-measure and still be
# considered "still winning". Mirrors trusted_grader.REPRO_TOL (10%): a prior
# that drops more than this under the new toolchain is a real regression, not
# measurement noise, and is demoted. One-sided by design — a prior that got
# *faster* under the new SDK obviously still wins.
REGRESSION_TOL = 0.10


# ---------------------------------------------------------------------------
# 1. current toolchain
# ---------------------------------------------------------------------------

def current_toolchain() -> str:
    """The live neuronx-cc version, or "" if it cannot be determined.

    Best-effort and dependency-free: reads the installed `neuronx-cc` package
    version via importlib.metadata — the exact path the backends stamp their
    artifacts with (see native_pytorch.toolchain_stamp). On a plain CPU box
    where the compiler is not installed this returns "" rather than raising, so
    the whole pass degrades to a no-op instead of crashing a run.
    """
    try:
        from importlib.metadata import version
        return version("neuronx-cc") or ""
    except Exception:  # noqa: BLE001 — no compiler installed / metadata unreadable
        return ""


def dominant_sdk(bank: KnowledgeBank) -> str | None:
    """The SDK version most of the VERIFIED bank was last validated on.

    The reference point for "did the toolchain change under us?" — computed from
    the bank itself rather than trusting a hand-passed value. Uses each lesson's
    `last_reverified_sdk` when present, else the newest of its stamped
    `neuron_sdk_versions`, and returns the modal stamp. None on an empty bank.
    """
    stamps: list[str] = []
    for l in bank.load_all(Tier.VERIFIED):
        s = l.last_reverified_sdk or _newest(l.applicability.neuron_sdk_versions)
        if s:
            stamps.append(s)
    if not stamps:
        return None
    return Counter(stamps).most_common(1)[0][0]


def toolchain_changed(bank: KnowledgeBank, current_sdk: str) -> bool:
    """Has the live toolchain moved off the bank's dominant stamped SDK?

    Compared on minor-version integers (so `2.28.0`, `2.28.*` and
    `2.28.6360.0` all reduce to 28 and compare equal), matching bank.stale's
    version-distance logic. Conservative on the unknowns: an empty bank, or a
    `current_sdk` we cannot parse, returns False — we never trigger a
    re-validation storm on a signal we cannot read.
    """
    dom = dominant_sdk(bank)
    if dom is None:
        return False
    cur_m, dom_m = _minor(current_sdk), _minor(dom)
    if cur_m is None or dom_m is None:
        return False
    return cur_m != dom_m


# ---------------------------------------------------------------------------
# 2. stale detection
# ---------------------------------------------------------------------------

def _covers(lesson: Lesson, sdk: str) -> bool:
    """Does this lesson's SDK stamp cover `sdk`? True when a stamped glob
    matches, OR when it was last re-verified on the same minor version."""
    if any(fnmatch.fnmatch(sdk, pat) for pat in lesson.applicability.neuron_sdk_versions):
        return True
    lr = lesson.last_reverified_sdk
    if lr and _minor(lr) is not None and _minor(lr) == _minor(sdk):
        return True
    return False


def stale_priors(
    bank: KnowledgeBank,
    current_sdk: str,
    types: tuple[LessonType, ...] = (LessonType.CONFIG_PRIOR,),
) -> list[Lesson]:
    """VERIFIED priors whose `neuron_sdk_versions` don't cover `current_sdk`.

    These are the re-validation candidates: lessons the proposer currently
    trusts (they live in verified/) but that were never measured on the live
    toolchain. Defaults to config-priors — the winning-config lessons the loop
    emits, which carry both a re-appliable config and a recorded metric to gate
    against. Pass a wider `types` to also sweep op-rewrites / NKI kernels.
    """
    return [
        l for l in bank.load_all(Tier.VERIFIED)
        if l.type in types and not _covers(l, current_sdk)
    ]


# ---------------------------------------------------------------------------
# 3. re-validation
# ---------------------------------------------------------------------------

@dataclass
class CanaryMeasurement:
    """The result of re-measuring one prior's config on a canary model.

    `ok=False` marks a TRANSIENT outcome (a crash, or a measurement too noisy to
    trust) — the case demotion must never fire on. `ok=True` is a trustworthy,
    deterministic result: `equivalence_passed`/`fits`/`metric` then decide
    whether the prior still wins.
    """

    ok: bool
    metric: float = 0.0
    correctness: float = 0.0
    equivalence_passed: bool = False
    fits: bool = True            # HBM ceiling — a config that no longer fits regressed
    notes: str = ""


# A measure function re-runs one config on a canary and classifies the outcome.
# Injected so the harness logic unit-tests deterministically with a mock, while
# the default reuses the real Orchestrator measure+equivalence path.
MeasureFn = Callable[[Any, "ModelSpec", "dict[str, Any]", Guardrails], CanaryMeasurement]

# A canary source maps a config-prior to the small model to re-validate it on.
# Accepts a ModelSpec (used for every family), a {family: ModelSpec} dict, or a
# callable family -> ModelSpec | None.
CanarySpec = Any


@dataclass
class RevalidationOutcome:
    lesson_id: str
    family: str
    action: str          # revalidated | demoted | inconclusive | skipped
    recorded_metric: float
    remeasured_metric: float
    equivalence_passed: bool
    reason: str


@dataclass
class RevalidationReport:
    current_sdk: str
    backend: str
    outcomes: list[RevalidationOutcome] = field(default_factory=list)

    def _n(self, action: str) -> int:
        return sum(1 for o in self.outcomes if o.action == action)

    @property
    def revalidated(self) -> int:
        return self._n("revalidated")

    @property
    def demoted(self) -> int:
        return self._n("demoted")

    @property
    def inconclusive(self) -> int:
        return self._n("inconclusive")

    @property
    def skipped(self) -> int:
        return self._n("skipped")

    def summary(self) -> str:
        return (
            f"{len(self.outcomes)} stale prior(s) on backend {self.backend!r} "
            f"vs sdk {self.current_sdk!r}: {self.revalidated} revalidated, "
            f"{self.demoted} DEMOTED, {self.inconclusive} inconclusive "
            f"(transient), {self.skipped} skipped"
        )


def revalidate(
    bank: KnowledgeBank,
    backend: Any,
    canary_spec: CanarySpec,
    current_sdk: str,
    guards: Guardrails,
    *,
    measure_fn: MeasureFn | None = None,
    regression_tol: float = REGRESSION_TOL,
    log: Callable[[str], None] | None = None,
) -> RevalidationReport:
    """Re-validate every stale VERIFIED prior against the live toolchain.

    For each stale config-prior on this `backend`: re-apply its config to a
    canary model of the matching family+backend and re-measure it through the
    EXISTING measure + equivalence gate (nothing here re-implements or weakens a
    gate).

      still correct AND remeasured >= recorded * (1 - regression_tol)
          -> REFRESH: extend `neuron_sdk_versions` to cover current_sdk and bump
             last_reverified_sdk. The prior stays verified, now re-earned.
      equivalence fails, or config no longer fits, or a real regression
          -> DEMOTE verified->provisional and record a `staleness` breadcrumb
             ("prior X regressed under sdk Y"). Never silently kept.
      transient (crash / too-noisy to trust) or no canary/config
          -> leave the prior untouched (inconclusive / skipped). Demotion is
             deterministic-only, mirroring the pre-flight gate.

    Returns a RevalidationReport covering every candidate — refreshed, demoted,
    and skipped alike — so the morning audit shows exactly what the pass did.
    """
    log = log or (lambda _m: None)
    measure = measure_fn or _default_canary_measure
    backend_name = getattr(backend, "name", "") or ""
    report = RevalidationReport(current_sdk=current_sdk, backend=backend_name)

    from bank import _backend_matches  # local: keeps top-level import surface small

    for lesson in stale_priors(bank, current_sdk):
        # Scope to this run's execution stack — don't re-measure a vllm-serve
        # prior on a native-pytorch canary, and vice-versa. mock matches all.
        if backend_name and not _backend_matches(lesson, backend_name):
            continue

        family = lesson.applicability.architecture_family
        config = _prior_config(lesson)
        canary = _resolve_canary(canary_spec, family)
        recorded = _recorded_metric(lesson)

        if config is None or canary is None:
            reason = ("no re-appliable config" if config is None
                      else f"no canary model for family {family!r}")
            report.outcomes.append(RevalidationOutcome(
                lesson.lesson_id, family, "skipped", recorded, 0.0, False, reason))
            log(f"[revalidate] {lesson.lesson_id}: skipped ({reason})")
            continue

        meas = measure(backend, canary, config, guards)

        # --- transient: never demote on a crash or a noisy read ---------------
        if not meas.ok:
            report.outcomes.append(RevalidationOutcome(
                lesson.lesson_id, family, "inconclusive", recorded,
                meas.metric, meas.equivalence_passed,
                f"transient, left verified: {meas.notes}"))
            log(f"[revalidate] {lesson.lesson_id}: inconclusive "
                f"(transient: {meas.notes}) — left verified")
            continue

        # --- deterministic verdict -------------------------------------------
        regressed = (
            recorded > 0.0 and meas.metric < recorded * (1.0 - regression_tol)
        )
        wins = meas.equivalence_passed and meas.fits and meas.metric > 0.0 and not regressed

        if wins:
            _refresh_lesson(bank, lesson, current_sdk)
            report.outcomes.append(RevalidationOutcome(
                lesson.lesson_id, family, "revalidated", recorded,
                meas.metric, True,
                f"still wins (remeasured {meas.metric:.1f} vs recorded {recorded:.1f})"))
            log(f"[revalidate] {lesson.lesson_id}: REVALIDATED on {current_sdk} "
                f"(remeasured {meas.metric:.1f} vs recorded {recorded:.1f})")
        else:
            if not meas.equivalence_passed:
                reason = f"equivalence failed under sdk {current_sdk} ({meas.notes})"
            elif not meas.fits:
                reason = f"no longer fits under sdk {current_sdk} ({meas.notes})"
            else:
                reason = (f"regressed under sdk {current_sdk}: recorded "
                          f"{recorded:.1f} -> remeasured {meas.metric:.1f}")
            bank.demote(lesson.lesson_id, reason=reason)
            _record_staleness_lesson(bank, lesson, current_sdk, recorded,
                                     meas.metric, reason)
            report.outcomes.append(RevalidationOutcome(
                lesson.lesson_id, family, "demoted", recorded,
                meas.metric, meas.equivalence_passed, reason))
            log(f"[revalidate] {lesson.lesson_id}: DEMOTED verified->provisional "
                f"({reason})")

    return report


# ---------------------------------------------------------------------------
# hook — startup entry the overnight loop calls
# ---------------------------------------------------------------------------

def canary_from_specs(specs: dict[str, "ModelSpec"]) -> dict[str, "ModelSpec"]:
    """Build a {family: smallest-model} canary map from a slug->ModelSpec dict.

    The smallest model per family is the cheapest thing to re-measure a prior
    on, which is exactly what a canary should be. Used by overnight.py to feed
    revalidate() its SEED_MODELS as canaries.
    """
    out: dict[str, ModelSpec] = {}
    for spec in specs.values():
        fam = _norm_family(getattr(spec, "family", ""))
        cur = out.get(fam)
        if cur is None or getattr(spec, "param_count", 0.0) < getattr(cur, "param_count", 0.0):
            out[fam] = spec
    return out


def maybe_revalidate_at_startup(
    bank: KnowledgeBank,
    backend: Any,
    canary_spec: CanarySpec,
    current_sdk: str,
    guards: Guardrails,
    *,
    measure_fn: MeasureFn | None = None,
    log: Callable[[str], None] | None = None,
) -> RevalidationReport | None:
    """Run a re-validation pass ONLY when the toolchain has changed.

    The opt-in-safe startup hook: on a stable box (live SDK == the bank's
    dominant stamp) this is a cheap no-op that returns None and never touches a
    prior. On a fresh box / new week / SDK bump it re-validates the stale
    priors before the run loop starts, so the loop never seeds a beam from a
    prior the new toolchain already invalidated.
    """
    log = log or (lambda _m: None)
    if not toolchain_changed(bank, current_sdk):
        log(f"[revalidate] toolchain unchanged (sdk {current_sdk!r} vs bank "
            f"dominant {dominant_sdk(bank)!r}) — skipping re-validation")
        return None
    log(f"[revalidate] toolchain CHANGED (sdk {current_sdk!r} vs bank dominant "
        f"{dominant_sdk(bank)!r}) — re-validating stale verified priors")
    return revalidate(bank, backend, canary_spec, current_sdk, guards,
                      measure_fn=measure_fn, log=log)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prior_config(lesson: Lesson) -> dict[str, Any] | None:
    """The re-appliable config carried by a config-prior. The loop emits
    `intervention={"spec": <config>}`; tolerate a bare intervention dict too.
    None when there is nothing to re-apply."""
    iv = lesson.intervention or {}
    spec = iv.get("spec", iv)
    return dict(spec) if isinstance(spec, dict) and spec else None


def _recorded_metric(lesson: Lesson) -> float:
    """The best metric this prior recorded when it was validated — the number
    the re-measurement is gated against. 0.0 when none was recorded (then the
    margin check is skipped and correctness alone decides)."""
    best = 0.0
    for e in lesson.evidence or []:
        if isinstance(e, dict):
            v = e.get("metric")
            if isinstance(v, (int, float)) and v > best:
                best = float(v)
    return best


def _resolve_canary(canary_spec: CanarySpec, family: str) -> "ModelSpec | None":
    """Pick the canary model for a prior's family from the flexible canary_spec
    (a single ModelSpec, a {family: ModelSpec} map, or a callable)."""
    if canary_spec is None:
        return None
    if callable(canary_spec):
        return canary_spec(family)
    if isinstance(canary_spec, dict):
        fam = _norm_family(family)
        for k, v in canary_spec.items():
            if _norm_family(k) == fam:
                return v
        return None
    # A single ModelSpec: only a canary for its own family.
    if _norm_family(getattr(canary_spec, "family", "")) == _norm_family(family):
        return canary_spec
    return None


def _sdk_glob(sdk: str) -> str:
    """A `major.minor.*` glob covering `sdk` (e.g. "2.30.6360.0" -> "2.30.*"),
    matching how the loop stamps freshly-emitted lessons."""
    parts = (sdk or "").split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}.*"
    return sdk or "*"


def _refresh_lesson(bank: KnowledgeBank, lesson: Lesson, current_sdk: str) -> Path:
    """Extend a re-validated prior's SDK coverage and bump its re-verify stamp,
    then re-save it in place (still verified)."""
    glob = _sdk_glob(current_sdk)
    if glob not in lesson.applicability.neuron_sdk_versions:
        lesson.applicability.neuron_sdk_versions.append(glob)
    lesson.last_reverified_sdk = current_sdk
    return bank.save(lesson)


def _record_staleness_lesson(
    bank: KnowledgeBank, prior: Lesson, current_sdk: str,
    recorded: float, remeasured: float, reason: str,
) -> Path:
    """Drop a provisional `staleness` breadcrumb when a prior is demoted.

    Recorded as a provisional ANTI_PATTERN so it shows up in the bank metrics
    and the weekly triage. Its matcher uses non-config keys (`stale_prior` /
    `sdk`), so — exactly like the pre-flight anti-patterns — it never fires
    against a real candidate config in the Stage-1 tournament; it is purely the
    "prior X regressed under sdk Y" record, so the demotion is auditable and the
    regression is never silently forgotten.
    """
    from ledger import Layer, Origin

    lesson = Lesson(
        lesson_id=f"staleness-{prior.lesson_id}-{_minor(current_sdk) or 'x'}",
        type=LessonType.ANTI_PATTERN,
        applicability=Applicability(
            architecture_family=prior.applicability.architecture_family,
            param_count_range=prior.applicability.param_count_range,
            neuron_sdk_versions=[_sdk_glob(current_sdk)],
        ),
        layer=Layer.CONFIG,
        migration_risk="high",
        origin=Origin.NONE,
        tier=Tier.PROVISIONAL,
        backend=prior.backend,
        matcher={"stale_prior": prior.lesson_id, "sdk": current_sdk},
        reason=(f"prior {prior.lesson_id} regressed under sdk {current_sdk} "
                f"(recorded {recorded:.1f} -> remeasured {remeasured:.1f}): {reason}"),
        confidence=Confidence(n_models_validated=1, human_verified=False),
        last_reverified_sdk=current_sdk,
        evidence=[{
            "stale_prior": prior.lesson_id,
            "sdk": current_sdk,
            "recorded_metric": recorded,
            "remeasured_metric": remeasured,
            "outcome": "demoted_stale",
        }],
    )
    return bank.save(lesson)


def _default_canary_measure(
    backend: Any, spec: "ModelSpec", config: dict[str, Any], guards: Guardrails,
) -> CanaryMeasurement:
    """Re-measure `config` on `spec` through the real Orchestrator gate.

    Reuses the EXACT measure + equivalence path the search uses (baseline
    tokens as the correctness reference, backend.measure, orch._equivalence, the
    Guardrails gates) — it does not re-implement or weaken any of them. It only
    *classifies* the outcome for the demotion decision, keeping the
    deterministic-vs-transient distinction the pre-flight gate relies on:

      - noisy / crash            -> ok=False (transient — never demote)
      - metric=0 / equiv fail    -> ok=True, equivalence_passed=False (demote)
      - HBM OOM                  -> ok=True, fits=False (deterministic regression)
      - correct + fits + faster  -> ok=True, equivalence_passed=True (still wins)

    Heavy imports are local so the module stays import-safe on a bare CPU box.
    """
    from ledger import Ledger
    from orchestrator import Orchestrator

    run_dir = Path(tempfile.mkdtemp(prefix="revalidate-"))
    try:
        ledger = Ledger(run_dir)
        ledger.init()
        orch = Orchestrator(
            backend=backend, bank=KnowledgeBank(run_dir / "bank"),
            guards=guards, ledger=ledger,
        )
        # Baseline the UNMODIFIED canary — this sets the top-1-token equivalence
        # reference the prior's config is then checked against.
        orch.establish_baseline(spec)

        artifact = backend.apply_config(backend.build_baseline(spec.model_id), dict(config))
        neff = backend.compile(artifact)
        m = backend.measure(neff, spec.probe_shape, spec.probe_batch)

        # Deterministic no-throughput: a silent backend failure, treated as a
        # loss (not a win) — same as the orchestrator's metric<=0 gate.
        if m.metric <= 0.0:
            return CanaryMeasurement(ok=True, metric=0.0, equivalence_passed=False,
                                     notes="metric=0 (no throughput)")

        # Noisy measurement is TRANSIENT — a re-run might read clean; never demote.
        if not guards.measurement_trustworthy(m):
            return CanaryMeasurement(ok=False, metric=m.metric,
                                     notes="noisy measurement (transient)")

        eq = orch._equivalence(m, spec, neff)
        if not eq.passed:
            return CanaryMeasurement(ok=True, metric=m.metric,
                                     equivalence_passed=False,
                                     correctness=eq.correctness_pct,
                                     notes=f"equivalence fail: {eq.notes}")

        if not guards.hbm_ok(m):
            return CanaryMeasurement(ok=True, metric=m.metric, equivalence_passed=True,
                                     correctness=eq.correctness_pct, fits=False,
                                     notes=f"OOM {m.hbm_utilization:.0%} HBM")

        return CanaryMeasurement(ok=True, metric=m.metric, equivalence_passed=True,
                                 correctness=eq.correctness_pct, fits=True, notes="ok")
    except Exception as e:  # noqa: BLE001 — a crash is transient, must not demote
        return CanaryMeasurement(ok=False, notes=f"transient error: {e}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
