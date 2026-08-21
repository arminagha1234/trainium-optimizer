"""
Pre-flight gate — Rule 4 ("anti-patterns prune before compile") at the
model/architecture level.

The stage pipeline already prunes known-bad *configs* before compiling them
(bank.prune, called from the orchestrator). This module moves the same idea one
level up: some whole *architectures* fail the EXPENSIVE way — they burn a 30-90
min compile, or a launch + partial compile, only to abort with an ISA
validation error, an NRT device abort, or a silent 0-throughput "success". A
pre-flight check reads only the HF `config.json` (config, never weights) and
consults the bank's anti-patterns, so a class that failed once is skipped
instantly the next time instead of re-burning the compute.

Two, deliberately conservative, signals:

  1. Static architecture detection. Linear-attention / GatedDeltaNet models
     (e.g. Qwen3.5 GatedDeltaNet) abort neuronx-cc with an ISA-validation
     assertion (`s2d2_ts_as_valid_elem_count` in `TensorScalarAffineSelect`)
     inside the `[linear_attn]` graph — a real compiler limitation that needs a
     dedicated adapter, not a config tweak. These are pruned on the FIRST
     encounter, from the config alone.

  2. Bank consultation. If this model_id — or, more usefully, this
     architecture *signature* — already produced a compile-abort, an NRT
     device-abort, or a 0-metric "unverified" result, the recorded anti-pattern
     fires and the whole class is skipped fast. Keyed by arch-signature, so a
     sibling model of a known-bad arch is pruned too.

The check is cheap by construction: config inspection + a bank query, never a
compile and never a weight load. It only ever skips *known-bad* arches, so it
is safe on by default (working dense models — Qwen3 0.6B/4B, Qwen2.5-0.5B — are
never gated).

See ../../optimization-stages.md (Rule 4) and ../../guardrails.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from orchestrator import ModelSpec


# A loader turns a model_id into a plain config dict (architectures / model_type
# / attention fields). Injected so this module never hard-depends on
# transformers and stays unit-testable with a mock config.
ConfigLoader = Callable[[str], "dict[str, Any] | None"]


# Skip reason for the statically-detected linear-attention class. Kept as a
# module constant so the orchestrator/tests reference one string.
LINEAR_ATTN_REASON = (
    "linear-attention/GatedDeltaNet: neuronx-cc ISA validation "
    "(TensorScalarAffineSelect) unsupported — needs adapter"
)

# Architecture / model_type markers for the linear-attention family. Matched
# against a separator-stripped, lower-cased haystack (so "gated_delta",
# "GatedDelta" and "gated-delta" all hit the same marker). Deliberately
# specific: a dense "Qwen3ForCausalLM" must NOT match any of these.
_LINEAR_ATTN_MARKERS = (
    "gateddelta",        # Qwen3.5 GatedDeltaNet, *GatedDeltaNet*
    "deltanet",          # DeltaNet / delta_net
    "linearattn",        # linear_attn / linear-attn
    "linearattention",   # linear_attention
)


def _norm(s: str) -> str:
    """Lower-case and strip separators, so `Qwen3_5-GatedDeltaNet` and
    `qwen35gateddeltanet` compare equal. Used for both marker matching and the
    architecture signature."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


# ---------------------------------------------------------------------------
# config loading (cheap — config only, never weights)
# ---------------------------------------------------------------------------

def _default_config_loader(model_id: str) -> dict[str, Any] | None:
    """Best-effort, weight-free config read. Tries a local `config.json` first
    (so a on-disk model dir needs no deps), then transformers.AutoConfig. Any
    failure (no file, transformers absent, unloadable config) returns None —
    the caller then treats the arch as UNKNOWN and does NOT gate it, so a
    config we cannot read never blocks a model that might work."""
    # Local model dir: <model_id>/config.json — no imports needed.
    try:
        p = Path(model_id) / "config.json"
        if p.is_file():
            return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        pass
    # HF hub id (or local repo): read config metadata only, never the weights.
    try:
        from transformers import AutoConfig  # local import: optional dependency
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return cfg.to_dict()
    except Exception:  # noqa: BLE001
        return None


def load_hf_config(
    model_id: str, config_loader: ConfigLoader | None = None,
) -> dict[str, Any] | None:
    """Load a model's HF config as a plain dict, or None if unavailable."""
    return (config_loader or _default_config_loader)(model_id)


# ---------------------------------------------------------------------------
# architecture inspection
# ---------------------------------------------------------------------------

def _architectures(config: dict[str, Any]) -> list[str]:
    """Architecture class names, unwrapping a nested `text_config` (multimodal
    configs put the LM there) so a wrapped decoder is still inspected."""
    archs = list(config.get("architectures") or [])
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        archs += list(text_cfg.get("architectures") or [])
    return [a for a in archs if a]


def _linear_attn_haystack(config: dict[str, Any]) -> str:
    """The normalized string linear-attention markers are matched against:
    architecture names, model_type, and the handful of config fields that name
    an attention flavor. Nested `text_config` is folded in."""
    parts: list[str] = list(_architectures(config))
    for cfg in (config, config.get("text_config")):
        if not isinstance(cfg, dict):
            continue
        for key in ("model_type", "attention_type", "attn_type",
                    "linear_attention", "layer_types"):
            val = cfg.get(key)
            if isinstance(val, (list, tuple)):
                parts.extend(str(v) for v in val)
            elif val is not None:
                parts.append(str(val))
    return _norm(" ".join(parts))


def is_linear_attention_arch(config: dict[str, Any] | None) -> bool:
    """True if the config describes a linear-attention / (gated-)DeltaNet
    architecture — the class that aborts neuronx-cc ISA validation. Conservative:
    an unreadable (None) config returns False (unknown != known-bad)."""
    if not config:
        return False
    hay = _linear_attn_haystack(config)
    return any(marker in hay for marker in _LINEAR_ATTN_MARKERS)


def arch_signature(spec: ModelSpec, config: dict[str, Any] | None = None) -> str:
    """A stable key for the model's architecture, so a lesson learned on one
    model prunes its siblings too. Prefers the HF architecture class name, then
    model_type, and finally falls back to the model_id leaf when no config is
    available. Normalized so casing / separators never split the class."""
    if config:
        archs = _architectures(config)
        if archs:
            return _norm(archs[0])
        for cfg in (config, config.get("text_config")):
            if isinstance(cfg, dict) and cfg.get("model_type"):
                return _norm(str(cfg["model_type"]))
    return _norm(spec.model_id.split("/")[-1])


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def _lesson_matches(lesson, sig: str, model_id: str) -> bool:
    """Does a recorded pre-flight anti-pattern apply to this model? Matches on
    the architecture signature (so siblings are pruned) OR the exact model_id."""
    m = getattr(lesson, "matcher", {}) or {}
    if m.get("arch_signature") and _norm(str(m["arch_signature"])) == sig:
        return True
    if m.get("model_id") and m["model_id"] == model_id:
        return True
    # model_id may also live in evidence rows.
    return any(
        isinstance(e, dict) and e.get("model_id") == model_id
        for e in (getattr(lesson, "evidence", []) or [])
    )


def preflight_check(
    spec: ModelSpec,
    bank: Any = None,
    sdk_version: str = "",
    *,
    config_loader: ConfigLoader | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Cheap, no-compile gate run before `establish_baseline`.

    Returns (ok, reason):
      - (True, None)          -> proceed, nothing predicts an expensive failure.
      - (False, "<reason>")   -> skip; the caller records the ledger row + an
                                 anti-pattern lesson so the class fails fast next
                                 time (see overnight._record_preflight_skip).

    Never compiles and never loads weights: it reads the HF config (config only)
    and queries the bank's pre-flight anti-patterns.
    """
    cfg = config if config is not None else load_hf_config(spec.model_id, config_loader)

    # 1. Static detection — fires on the FIRST encounter, from config alone.
    if is_linear_attention_arch(cfg):
        return False, LINEAR_ATTN_REASON

    # 2. Bank consultation — a class that already failed the expensive way.
    if bank is not None and hasattr(bank, "preflight_antipatterns"):
        sig = arch_signature(spec, cfg)
        for ap in bank.preflight_antipatterns(spec.family, sdk_version):
            if _lesson_matches(ap, sig, spec.model_id):
                return False, ap.reason or f"previously failed the expensive way ({ap.lesson_id})"

    return True, None


def make_anti_pattern_lesson(
    spec: ModelSpec, sig: str, reason: str, sdk_version: str = "",
):
    """Build the provisional anti-pattern lesson emitted on a pre-flight skip.

    Keyed by arch-signature (not just model_id) so a sibling of a known-bad
    architecture is pruned too — the whole point is that the class fails fast
    the second time. The matcher uses `arch_signature`/`model_id` keys, which
    are NOT config axes, so this lesson never accidentally prunes a real config
    candidate in the Stage-1 tournament (bank.prune's matcher would never fire).
    """
    from bank import Applicability, Confidence, Lesson, LessonType, Tier
    from ledger import Layer, Origin

    sdk_globs = [f"{sdk_version.rsplit('.', 1)[0]}.*"] if sdk_version else ["*"]
    return Lesson(
        lesson_id=f"preflight-{sig}",
        type=LessonType.ANTI_PATTERN,
        applicability=Applicability(
            architecture_family=spec.family,
            param_count_range=(0.0, 1e15),
            neuron_sdk_versions=sdk_globs,
        ),
        layer=Layer.CONFIG,
        migration_risk="high",
        origin=Origin.NONE,
        tier=Tier.PROVISIONAL,
        matcher={"arch_signature": sig, "model_id": spec.model_id},
        reason=reason,
        confidence=Confidence(n_models_validated=1, human_verified=False),
        last_reverified_sdk=sdk_version,
        evidence=[{"model_id": spec.model_id, "arch_signature": sig,
                   "outcome": "preflight_skip"}],
    )
