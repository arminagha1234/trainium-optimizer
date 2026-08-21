"""
Knowledge bank — the memory that makes model N+1 cheaper to optimize than N.

A lesson is a structured record of something that worked (or explicitly did
not). Lessons are YAML files on disk, one per file, so `git blame` gives every
lesson provenance for free and humans can author them by hand.

The bank answers two different questions, via two indexes:

  1. Intervention query (Stage 1) — "this is a 30B MoE, what configs are good
     for that class?" Answered from applicability predicates before we have any
     profile data.

  2. Symptom query (Stages 3-5) — "the profile says I'm collective-bound with
     the CC engine at 40% and PE idle, what has fixed *that* before?" Answered
     from the symptom index. This is the ADIAS insight: intervention-only
     stores leave "what problem was being solved" implicit.

Two tiers:
  - verified/    — proposer reads ONLY this. Human-authored or human-promoted.
  - provisional/ — optimizer writes here automatically; humans triage weekly.

See ../../knowledge-bank.md for the full schema and rationale.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from ledger import Layer, Origin


class LessonType(StrEnum):
    CONFIG_PRIOR = "config_prior"
    OP_REWRITE = "op_rewrite"
    NKI_KERNEL = "nki_kernel"
    ANTI_PATTERN = "anti_pattern"
    REFERENCE_TRANSLATION = "reference_translation"
    EQUIVALENCE_TOLERANCE = "equivalence_tolerance"


class Tier(StrEnum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"


# Family folder names, mirroring the on-disk layout under knowledge-bank/.
FAMILIES = (
    "dense-causal-lm",
    "hybrid-attention-causal-lm",
    "moe-causal-lm",
    "encoder-only",
    "diffusion",
    "speech",
)

# Lesson-type -> folder name.
TYPE_DIR = {
    LessonType.CONFIG_PRIOR: "config-priors",
    LessonType.OP_REWRITE: "op-rewrites",
    LessonType.NKI_KERNEL: "nki-kernels",
    LessonType.ANTI_PATTERN: "anti-patterns",
    LessonType.REFERENCE_TRANSLATION: "reference-translations",
    LessonType.EQUIVALENCE_TOLERANCE: "equivalence-tolerances",
}


@dataclass
class Applicability:
    """The predicate deciding whether a lesson applies to the model at hand."""

    architecture_family: str
    param_count_range: tuple[float, float] = (0.0, 1e15)
    seq_len_range: tuple[int, int] = (0, 10_000_000)
    batch_range: tuple[int, int] = (1, 4096)
    parents_ok: list[str] = field(default_factory=list)  # e.g. [llama, qwen]
    neuron_sdk_versions: list[str] = field(default_factory=list)  # e.g. ["2.28.*"]

    def matches(
        self,
        family: str,
        param_count: float,
        seq_len: int,
        batch: int,
        sdk_version: str,
        parent: str | None = None,
    ) -> bool:
        if _norm_family(self.architecture_family) != _norm_family(family):
            return False
        if not (self.param_count_range[0] <= param_count <= self.param_count_range[1]):
            return False
        if not (self.seq_len_range[0] <= seq_len <= self.seq_len_range[1]):
            return False
        if not (self.batch_range[0] <= batch <= self.batch_range[1]):
            return False
        if self.parents_ok and parent is not None and parent not in self.parents_ok:
            return False
        # SDK gate: empty means "unstamped" — treated as non-matching, because
        # an un-versioned lesson is a lesson we cannot trust across releases.
        if not self.neuron_sdk_versions:
            return False
        if not any(fnmatch.fnmatch(sdk_version, pat) for pat in self.neuron_sdk_versions):
            return False
        return True


@dataclass
class Confidence:
    n_models_validated: int = 1
    architecture_diversity: int = 1     # distinct families this held on
    human_verified: bool = False

    def score(self, sdk_versions_since_verified: int = 0) -> float:
        """0..1. Rewards breadth and human sign-off; decays with staleness.

        Deliberately simple and tunable. The point is ordering, not calibration.
        """
        base = 0.2
        if self.n_models_validated >= 3:
            base = 0.5
        if self.n_models_validated >= 5 and self.architecture_diversity >= 2:
            base = 0.8
        if self.human_verified:
            base = max(base, 0.6)

        # Staleness decay — see ../../guardrails.md#bank-staleness-policy
        if sdk_versions_since_verified >= 3:
            return 0.0                       # not used until re-verified
        if sdk_versions_since_verified == 2:
            base *= 0.5
        return round(base, 3)


@dataclass
class Symptom:
    """A bottleneck this lesson addresses. The ADIAS symptom index. """

    bottleneck: str          # e.g. "collective_bound", "dma_bound", "compute_bound"
    signature: str           # human-readable description of the profile shape
    observed_via: str = ""   # how it shows up in a profile


@dataclass
class Lesson:
    lesson_id: str
    type: LessonType
    applicability: Applicability
    layer: Layer
    migration_risk: str
    origin: Origin = Origin.NONE
    tier: Tier = Tier.PROVISIONAL

    # Type-specific payloads. Only the relevant ones are populated.
    intervention: dict[str, Any] = field(default_factory=dict)   # config_prior/op_rewrite
    matcher: dict[str, Any] = field(default_factory=dict)        # anti_pattern
    reason: str = ""                                             # anti_pattern
    symptoms_addressed: list[Symptom] = field(default_factory=list)

    confidence: Confidence = field(default_factory=Confidence)
    source: str = ""            # repo@commit for harvested/borrowed
    last_reverified_sdk: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    # Backends this lesson has actually been validated on. EMPTY = applies
    # everywhere (backward-compatible default). A NON-empty list means an
    # anti-pattern only prunes on those backends — on any other backend the
    # candidate is measured instead, so an unverified prior (e.g. "TP>=16
    # spills", proven on XLA) does not silently block exploration on a new
    # backend like native PyTorch.
    backend_validated: list[str] = field(default_factory=list)

    # Execution backend this lesson was LEARNED under, normalized to a stack key
    # (build suffix stripped — see _backend_stack). Distinct from
    # backend_validated above: that is the verify-first prune allowlist for
    # anti-patterns; THIS just partitions priors by stack so a lesson learned on
    # one backend (e.g. config priors from native-pytorch) never seeds a beam on
    # a different one (e.g. vllm-serve). Defaults to native-pytorch so legacy,
    # untagged lessons load as native. See _backend_matches for the match rule.
    backend: str = "native-pytorch"

    # Promotion audit trail. auto_promoted is deliberately distinct from
    # confidence.human_verified: an auto-promoted lesson is trusted by the
    # proposer (it lives in verified/) but was NOT signed off by a human, and
    # human_verified_ratio must keep reflecting that.
    auto_promoted: bool = False
    promoted_at: str = ""
    beat_borrowed_by: float | None = None   # fraction; required for origin=invented

    # Demotion audit trail (verified->provisional). Set by KnowledgeBank.demote
    # when toolchain re-validation finds a verified prior regressed under a new
    # SDK (see bank_hygiene.revalidate). Additive: emitted only when a lesson
    # was actually demoted, so untouched lessons round-trip byte-for-byte.
    demoted_at: str = ""
    demote_reason: str = ""

    # ---- serialization -----------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any], tier: Tier) -> Lesson:
        app = d["applicability"]
        return cls(
            lesson_id=d["lesson_id"],
            type=LessonType(d["type"]),
            applicability=Applicability(
                architecture_family=app["architecture_family"],
                param_count_range=tuple(app.get("param_count_range", (0.0, 1e15))),
                seq_len_range=tuple(app.get("seq_len_range", (0, 10_000_000))),
                batch_range=tuple(app.get("batch_range", (1, 4096))),
                parents_ok=app.get("parents_ok", []),
                neuron_sdk_versions=app.get("neuron_sdk_versions", []),
            ),
            layer=Layer(d.get("layer", "")),
            migration_risk=d.get("migration_risk", ""),
            origin=Origin(d.get("origin", "")),
            tier=tier,
            intervention=d.get("intervention", {}),
            matcher=d.get("matcher", {}),
            reason=d.get("reason", ""),
            symptoms_addressed=[
                Symptom(**s) for s in d.get("symptoms_addressed", [])
            ],
            confidence=Confidence(**d.get("confidence", {})),
            source=d.get("source", ""),
            last_reverified_sdk=d.get("last_reverified_sdk", ""),
            evidence=d.get("evidence", []),
            backend_validated=d.get("backend_validated", []),
            backend=d.get("backend", "native-pytorch"),
            auto_promoted=d.get("auto_promoted", False),
            promoted_at=d.get("promoted_at", ""),
            beat_borrowed_by=d.get("beat_borrowed_by", None),
            demoted_at=d.get("demoted_at", ""),
            demote_reason=d.get("demote_reason", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "lesson_id": self.lesson_id,
            "type": self.type.value,
            "layer": self.layer.value,
            "migration_risk": self.migration_risk,
            "origin": self.origin.value,
            "applicability": {
                "architecture_family": self.applicability.architecture_family,
                "param_count_range": list(self.applicability.param_count_range),
                "seq_len_range": list(self.applicability.seq_len_range),
                "batch_range": list(self.applicability.batch_range),
                "parents_ok": self.applicability.parents_ok,
                "neuron_sdk_versions": self.applicability.neuron_sdk_versions,
            },
            "confidence": {
                "n_models_validated": self.confidence.n_models_validated,
                "architecture_diversity": self.confidence.architecture_diversity,
                "human_verified": self.confidence.human_verified,
            },
        }
        # Only emit populated type-specific fields, to keep files readable.
        if self.intervention:
            out["intervention"] = self.intervention
        if self.matcher:
            out["matcher"] = self.matcher
        if self.reason:
            out["reason"] = self.reason
        if self.symptoms_addressed:
            out["symptoms_addressed"] = [
                {"bottleneck": s.bottleneck, "signature": s.signature,
                 "observed_via": s.observed_via}
                for s in self.symptoms_addressed
            ]
        if self.source:
            out["source"] = self.source
        if self.last_reverified_sdk:
            out["last_reverified_sdk"] = self.last_reverified_sdk
        if self.evidence:
            out["evidence"] = self.evidence
        if self.backend_validated:
            out["backend_validated"] = self.backend_validated
        # Emit only when non-default, so existing native-pytorch lesson YAML
        # (which never carried this field) round-trips byte-for-byte.
        if self.backend != "native-pytorch":
            out["backend"] = self.backend
        if self.auto_promoted:
            out["auto_promoted"] = self.auto_promoted
        if self.promoted_at:
            out["promoted_at"] = self.promoted_at
        if self.beat_borrowed_by is not None:
            out["beat_borrowed_by"] = self.beat_borrowed_by
        if self.demoted_at:
            out["demoted_at"] = self.demoted_at
        if self.demote_reason:
            out["demote_reason"] = self.demote_reason
        return out

    # ---- anti-pattern matching --------------------------------------------

    def prunes(self, config: dict[str, Any]) -> bool:
        """True if this anti-pattern's matcher fires against a candidate config.

        Supports: exact value, {gte: N}, {lte: N}, {in: [...]}. Multiple keys
        are ANDed. A matcher key absent from the config never fires (we cannot
        prune on an axis the candidate does not set).
        """
        if self.type is not LessonType.ANTI_PATTERN or not self.matcher:
            return False
        for key, pred in self.matcher.items():
            if key not in config:
                return False
            val = config[key]
            if isinstance(pred, dict):
                if "gte" in pred and not (val >= pred["gte"]):
                    return False
                if "lte" in pred and not (val <= pred["lte"]):
                    return False
                if "in" in pred and val not in pred["in"]:
                    return False
            else:
                if val != pred:
                    return False
        return True


@dataclass(frozen=True)
class AutoPromotionPolicy:
    """When may a provisional lesson become verified WITHOUT a human?

    This is the switch that turns the bank from "self-logging" into
    "self-learning": with it on, a lesson proven during an overnight run is
    trusted by later models in the *same* run, so the 3 seeds compound instead
    of each re-deriving the same wins.

    Deliberately conservative — a bad auto-promotion pollutes every future
    model's search, so the bar is real. It is expressed as explicit,
    interpretable criteria rather than the opaque 0-0.8 `Confidence.score()`
    (which is calibrated for *ranking* verified lessons, not as a promotion
    gate — it returns 0.2 for a solid 2-model lesson, which would fight
    min_models here):
      - validated on >= min_models distinct models,
      - across >= min_families architecture families,
      - every recorded measurement cleared min_correctness,
      - invented kernels additionally beat their borrowed alternative by
        >= invented_min_margin percent (no "invented" freebies).

    Off by default (humans-only). `overnight()` is the preset for the
    autonomous run: it allows single-family compounding (min_families=1) so a
    lesson learned on seed 1 can help seed 2 of the same family.
    """

    enabled: bool = False
    min_models: int = 2
    min_families: int = 2
    min_correctness: float = 99.0
    invented_min_margin: float = 5.0     # percent; mirrors Guardrails.invention_margin_pct

    @classmethod
    def overnight(cls) -> AutoPromotionPolicy:
        """Preset for a no-human-in-the-loop run across the seed set."""
        return cls(enabled=True, min_models=2, min_families=1,
                   min_correctness=99.0, invented_min_margin=5.0)


class KnowledgeBank:
    """YAML-backed lesson store with intervention and symptom retrieval.

    Layout on disk (see knowledge-bank/):
        <root>/<tier>/<family>/<type-dir>/<lesson_id>.yaml
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # ---- io ----------------------------------------------------------------

    def _lesson_path(self, lesson: Lesson) -> Path:
        family_dir = lesson.applicability.architecture_family.replace("_", "-")
        return (
            self.root / lesson.tier.value / family_dir
            / TYPE_DIR[lesson.type] / f"{lesson.lesson_id}.yaml"
        )

    def save(self, lesson: Lesson) -> Path:
        p = self._lesson_path(lesson)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            yaml.safe_dump(lesson.to_dict(), fh, sort_keys=False, default_flow_style=False)
        return p

    def load_all(self, tier: Tier | None = None) -> list[Lesson]:
        tiers = [tier] if tier else list(Tier)
        out: list[Lesson] = []
        for t in tiers:
            base = self.root / t.value
            if not base.exists():
                continue
            for path in base.rglob("*.yaml"):
                with path.open() as fh:
                    d = yaml.safe_load(fh)
                if d:
                    out.append(Lesson.from_dict(d, tier=t))
        return out

    # ---- retrieval ---------------------------------------------------------

    def query_interventions(
        self,
        family: str,
        param_count: float,
        seq_len: int,
        batch: int,
        sdk_version: str,
        parent: str | None = None,
        types: tuple[LessonType, ...] = (
            LessonType.CONFIG_PRIOR,
            LessonType.OP_REWRITE,
            LessonType.NKI_KERNEL,
        ),
        backend: str | None = None,
    ) -> list[Lesson]:
        """Stage-1 style query: what applies to this model class, ranked by
        confidence. Reads verified tier only — the proposer never sees
        provisional lessons in v0.

        When `backend` is given, lessons learned on a different execution stack
        are dropped (see _backend_matches), so native-pytorch priors don't seed
        a vllm-serve beam and vice-versa. `backend=None` never filters, keeping
        existing callers unchanged.
        """
        hits = [
            l for l in self.load_all(Tier.VERIFIED)
            if l.type in types
            and l.applicability.matches(
                family, param_count, seq_len, batch, sdk_version, parent
            )
            and (backend is None or _backend_matches(l, backend))
        ]
        return sorted(hits, key=lambda l: l.confidence.score(), reverse=True)

    def query_symptom(
        self,
        bottleneck: str,
        family: str,
        param_count: float,
        seq_len: int,
        batch: int,
        sdk_version: str,
        backend: str | None = None,
    ) -> list[Lesson]:
        """Stages 3-5 query: given a profiled bottleneck, what has fixed it?

        The ADIAS symptom index. More useful once we have profile data,
        because by then the problem is known. `backend` filters by execution
        stack exactly as in query_interventions; None never filters.
        """
        hits = [
            l for l in self.load_all(Tier.VERIFIED)
            if any(s.bottleneck == bottleneck for s in l.symptoms_addressed)
            and l.applicability.matches(family, param_count, seq_len, batch, sdk_version)
            and (backend is None or _backend_matches(l, backend))
        ]
        return sorted(hits, key=lambda l: l.confidence.score(), reverse=True)

    def antipatterns(
        self, family: str, sdk_version: str,
    ) -> list[Lesson]:
        """All applicable anti-patterns, for pre-compile pruning."""
        return [
            l for l in self.load_all(Tier.VERIFIED)
            if l.type is LessonType.ANTI_PATTERN
            and _norm_family(l.applicability.architecture_family) == _norm_family(family)
            and (
                not l.applicability.neuron_sdk_versions
                or any(fnmatch.fnmatch(sdk_version, p)
                       for p in l.applicability.neuron_sdk_versions)
            )
        ]

    def preflight_antipatterns(
        self, family: str, sdk_version: str,
    ) -> list[Lesson]:
        """Architecture-level anti-patterns for the pre-flight gate.

        Unlike `antipatterns()` (config-matcher pruning, verified tier only),
        these are keyed by architecture *signature* / model_id and record a
        model or arch that failed the EXPENSIVE way — a compile-abort, an NRT
        device-abort, or a 0-metric "unverified" run. BOTH tiers are consulted
        on purpose: a class learned once (written provisional by the loop)
        should fail fast the very next time, so the learning compounds without
        waiting on weekly human promotion. Selected by the `arch_signature` /
        `model_id` matcher keys the pre-flight writer uses, so config-only
        anti-patterns (tp/dtype matchers) are never returned here.
        """
        out: list[Lesson] = []
        for l in self.load_all():
            if l.type is not LessonType.ANTI_PATTERN:
                continue
            if _norm_family(l.applicability.architecture_family) != _norm_family(family):
                continue
            pats = l.applicability.neuron_sdk_versions
            if pats and not any(fnmatch.fnmatch(sdk_version, p) for p in pats):
                continue
            if "arch_signature" in l.matcher or "model_id" in l.matcher:
                out.append(l)
        return out

    def prune(
        self, candidates: list[dict[str, Any]], family: str, sdk_version: str,
        backend: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[tuple[dict, str]]]:
        """Drop candidates matching a known anti-pattern, before any compile.

        Returns (survivors, [(pruned_config, reason), ...]). The pruned list
        feeds the ledger as zero-cost negative results.

        Verify-first: an anti-pattern only prunes when it is *active* on the
        current backend. An anti-pattern with a non-empty `backend_validated`
        that does not include `backend` is NOT applied — its candidates are
        measured instead, so an unverified prior can't silently block
        exploration on a new backend. Empty `backend_validated`, or backend
        left None, keeps the old always-prune behavior.
        """
        aps = self.antipatterns(family, sdk_version)

        def active(ap: Lesson) -> bool:
            if not ap.backend_validated or backend is None:
                return True
            return backend in ap.backend_validated

        survivors, pruned = [], []
        for cfg in candidates:
            hit = next((ap for ap in aps if active(ap) and ap.prunes(cfg)), None)
            if hit is None:
                survivors.append(cfg)
            else:
                pruned.append((cfg, f"{hit.lesson_id}: {hit.reason}"))
        return survivors, pruned

    # ---- staleness ---------------------------------------------------------

    def stale(self, current_sdk: str, max_versions: int = 2) -> list[Lesson]:
        """Verified lessons whose SDK stamp is too old to trust.

        Simplistic version-distance: compares minor version integers. Real
        implementation would parse the full SDK version; this is enough to
        flag the obvious cases for re-verification.
        """
        out = []
        cur = _minor(current_sdk)
        for l in self.load_all(Tier.VERIFIED):
            stamp = l.last_reverified_sdk or _newest(l.applicability.neuron_sdk_versions)
            if stamp is None:
                out.append(l)                        # unstamped == suspect
                continue
            if cur is not None and _minor(stamp) is not None:
                if cur - _minor(stamp) > max_versions:
                    out.append(l)
        return out

    # ---- promotion ---------------------------------------------------------

    def promote(self, lesson_id: str) -> Path:
        """Move a provisional lesson to verified. The weekly-triage action."""
        for l in self.load_all(Tier.PROVISIONAL):
            if l.lesson_id == lesson_id:
                old = self._lesson_path(l)
                l.tier = Tier.VERIFIED
                l.confidence.human_verified = True
                new = self.save(l)
                if old != new:
                    old.unlink(missing_ok=True)
                return new
        raise KeyError(f"no provisional lesson {lesson_id!r}")

    def demote(self, lesson_id: str, reason: str = "") -> Path:
        """Move a verified lesson back to provisional — the inverse of promote.

        The toolchain-hygiene action (see bank_hygiene.revalidate): a verified
        prior that regressed or failed equivalence under a new SDK is no longer
        trusted by the proposer, so it drops to provisional for re-triage. Trust
        is fully withdrawn — human_verified and auto_promoted are cleared — so a
        stale-wrong prior can never be silently re-trusted without fresh
        evidence, and the demotion is stamped for the audit trail.
        """
        for l in self.load_all(Tier.VERIFIED):
            if l.lesson_id == lesson_id:
                old = self._lesson_path(l)
                l.tier = Tier.PROVISIONAL
                l.confidence.human_verified = False
                l.auto_promoted = False
                l.demoted_at = _utcnow()
                l.demote_reason = reason
                new = self.save(l)
                if old != new:
                    old.unlink(missing_ok=True)
                return new
        raise KeyError(f"no verified lesson {lesson_id!r}")

    def _auto_promotion_reason(
        self, l: Lesson, policy: AutoPromotionPolicy, current_sdk: str,
    ) -> tuple[bool, str]:
        """Evaluate one provisional lesson against the policy. Returns
        (qualifies, human-readable reason) — the reason is logged either way so
        the morning audit shows why each lesson did or didn't cross."""
        c = l.confidence
        if c.n_models_validated < policy.min_models:
            return False, f"only {c.n_models_validated} model(s) < {policy.min_models}"
        if c.architecture_diversity < policy.min_families:
            return False, f"only {c.architecture_diversity} family/ies < {policy.min_families}"
        for e in l.evidence:
            corr = e.get("correctness", e.get("correctness_pct"))
            if corr is not None and corr < policy.min_correctness:
                return False, f"a measurement's correctness {corr} < {policy.min_correctness}"
        if l.origin is Origin.INVENTED:
            if l.beat_borrowed_by is None:
                return False, "invented but no beat_borrowed_by recorded"
            if l.beat_borrowed_by * 100.0 < policy.invented_min_margin:
                return False, (f"invented beat borrowed by only "
                               f"{l.beat_borrowed_by*100:.1f}% < {policy.invented_min_margin}%")
        return True, (f"n_models={c.n_models_validated}, "
                      f"families={c.architecture_diversity}, "
                      f"correctness gate + invented margin cleared")

    def auto_promote(
        self, policy: AutoPromotionPolicy, current_sdk: str = "",
    ) -> list[tuple[str, bool, str]]:
        """Promote every qualifying provisional lesson to verified, no human.

        Returns [(lesson_id, promoted, reason), ...] for the whole provisional
        set — promoted and skipped alike — so the run log is auditable. A no-op
        (returns []) when the policy is disabled.
        """
        if not policy.enabled:
            return []
        results: list[tuple[str, bool, str]] = []
        for l in self.load_all(Tier.PROVISIONAL):
            ok, reason = self._auto_promotion_reason(l, policy, current_sdk)
            if ok:
                old = self._lesson_path(l)
                l.tier = Tier.VERIFIED
                l.auto_promoted = True          # trusted, but NOT human_verified
                l.promoted_at = _utcnow()
                new = self.save(l)
                if old != new:
                    old.unlink(missing_ok=True)
            results.append((l.lesson_id, ok, reason))
        return results

    # ---- metrics -----------------------------------------------------------

    def stats(self, current_sdk: str = "") -> dict[str, Any]:
        verified = self.load_all(Tier.VERIFIED)
        provisional = self.load_all(Tier.PROVISIONAL)
        by_type = {t.value: 0 for t in LessonType}
        for l in verified:
            by_type[l.type.value] += 1
        return {
            "verified": len(verified),
            "provisional": len(provisional),
            "by_type": by_type,
            "stale": len(self.stale(current_sdk)) if current_sdk else None,
            "human_verified_ratio": (
                sum(1 for l in verified if l.confidence.human_verified) / len(verified)
                if verified else 0.0
            ),
            # verified lessons that got there by auto-promotion, not a human.
            # Watch this against human_verified_ratio to see how much of the
            # bank the autonomous loop is now responsible for.
            "auto_promoted": sum(1 for l in verified if l.auto_promoted),
        }


# -- helpers -----------------------------------------------------------------

def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _backend_stack(name: str) -> str:
    """Normalize a backend name to its STACK key by stripping the build suffix.

    A backend `.name` carries a build tag (e.g. native-pytorch-beta3); the stack
    is what actually decides whether a lesson transfers, so beta3 and a future
    beta4 must share priors. Only the `-beta<N>` suffix is stripped —
    vllm-serve, diffusion-native, and mock are already stack keys and pass
    through unchanged. Empty/None normalizes to the native default.
    """
    return re.sub(r"-beta\d+$", "", name or "native-pytorch")


def _backend_matches(lesson: Lesson, backend: str) -> bool:
    """Does `lesson` apply on execution backend `backend`?

    Compared on stack keys (build suffix stripped). The synthetic `mock` backend
    matches EVERY stack — on either side — so mock-backed tests keep seeding
    priors for any target unchanged. A lesson with no backend tag defaults to
    native-pytorch (see Lesson.backend).
    """
    want = _backend_stack(backend)
    have = _backend_stack(lesson.backend)
    if want == "mock" or have == "mock":
        return True
    return want == have


def _norm_family(name: str) -> str:
    """Canonicalize a family name. The lesson schema uses underscores
    (dense_causal_lm) while the on-disk layout uses dashes
    (dense-causal-lm); callers pass either. Compare on a single form so the
    two never silently fail to match."""
    return name.replace("-", "_")


def _minor(sdk: str) -> int | None:
    """Extract the minor version integer from e.g. "2.28.0" -> 28."""
    m = re.match(r"\d+\.(\d+)", sdk or "")
    return int(m.group(1)) if m else None


def _newest(patterns: list[str]) -> str | None:
    """Best-effort newest concrete version from a list of glob patterns."""
    concrete = [p.replace(".*", ".0") for p in patterns]
    minors = [(_minor(p), p) for p in concrete]
    minors = [(m, p) for m, p in minors if m is not None]
    return max(minors)[1] if minors else None
