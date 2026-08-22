"""
Tests for the pre-flight arch gate (Rule 4 at the model/arch level).

Mock-only, no hardware and no transformers: HF configs are injected via a
`config_loader`, and the bank is a real on-disk KnowledgeBank under tmp_path.

Covers:
  - a linear-attention / GatedDeltaNet spec is skipped with the right reason,
  - a normal dense spec is NOT skipped,
  - a previously-recorded anti-pattern (in a mock bank) causes a fast skip,
    including a *sibling* model of the same arch (keyed by arch-signature),
  - the emitted lesson round-trips through the bank's pre-flight query,
  - the orchestrator records a 0-metric result as an explicit FAIL, not a
    benign/kept 0.
"""

from __future__ import annotations

from pathlib import Path

from backends.mock import MockBackend
from bank import KnowledgeBank, LessonType, Tier
from guardrails import Guardrails
from ledger import Layer, Ledger, Stage, Status
from orchestrator import ModelSpec, Orchestrator, always_equivalent
from preflight import (
    LINEAR_ATTN_REASON,
    arch_signature,
    is_linear_attention_arch,
    make_anti_pattern_lesson,
    preflight_check,
)


# -- fixtures / helpers ------------------------------------------------------

DELTANET_SPEC = ModelSpec(
    model_id="Qwen/Qwen3.5-4B-GatedDeltaNet",
    family="hybrid_attention_causal_lm", param_count=4e9, parent="qwen",
)

DENSE_SPEC = ModelSpec(
    model_id="Qwen/Qwen3-4B", family="dense_causal_lm",
    param_count=4e9, parent="qwen",
)


def _loader(mapping: dict[str, dict]):
    """A config_loader that serves canned configs and returns None otherwise
    (mirroring a config that cannot be read)."""
    return lambda model_id: mapping.get(model_id)


DELTANET_CFG = {"architectures": ["Qwen3_5GatedDeltaNetForCausalLM"],
                "model_type": "qwen3_5_gated_deltanet"}
DENSE_CFG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}


# -- static detection --------------------------------------------------------

def test_linear_attn_arch_detected():
    assert is_linear_attention_arch(DELTANET_CFG)
    assert not is_linear_attention_arch(DENSE_CFG)
    assert not is_linear_attention_arch(None)          # unknown != known-bad


def test_deltanet_spec_is_skipped_with_reason():
    loader = _loader({DELTANET_SPEC.model_id: DELTANET_CFG})
    ok, reason = preflight_check(DELTANET_SPEC, config_loader=loader)
    assert ok is False
    assert reason == LINEAR_ATTN_REASON


def test_dense_spec_is_not_skipped():
    loader = _loader({DENSE_SPEC.model_id: DENSE_CFG})
    ok, reason = preflight_check(DENSE_SPEC, config_loader=loader)
    assert ok is True
    assert reason is None


def test_unreadable_config_is_not_gated():
    """A config we cannot read is UNKNOWN, not known-bad — never skip on it."""
    ok, reason = preflight_check(DENSE_SPEC, config_loader=lambda _m: None)
    assert ok is True and reason is None


# -- emitted lesson + bank consultation --------------------------------------

def test_skip_emits_anti_pattern_lesson(tmp_path: Path):
    """The lesson emitted on a skip is a well-formed anti-pattern that the
    bank's pre-flight query returns."""
    bank = KnowledgeBank(tmp_path / "bank")
    sig = arch_signature(DELTANET_SPEC, DELTANET_CFG)
    bank.save(make_anti_pattern_lesson(DELTANET_SPEC, sig, LINEAR_ATTN_REASON, "2.28.0"))

    aps = bank.preflight_antipatterns(DELTANET_SPEC.family, "2.28.0")
    assert len(aps) == 1
    assert aps[0].type is LessonType.ANTI_PATTERN
    assert aps[0].tier is Tier.PROVISIONAL
    assert aps[0].matcher.get("arch_signature") == sig


# A diffusion arch that is NOT statically flagged as linear-attention, but which
# produced a 0-metric ("no throughput") run last time — the class the bank
# learns to skip after one expensive attempt. Static detection can't catch this;
# only the recorded lesson does.
FLUX_SPEC = ModelSpec(model_id="black-forest-labs/FLUX.1-schnell",
                      family="diffusion", param_count=12e9, parent="black-forest-labs")
FLUX_CFG = {"architectures": ["FluxPipeline"], "model_type": "flux"}


def test_recorded_anti_pattern_causes_fast_skip(tmp_path: Path):
    """A previously-recorded anti-pattern makes the SAME model skip fast from
    the bank — even though its config is NOT statically flagged (the diffusion
    0-metric case)."""
    bank = KnowledgeBank(tmp_path / "bank")
    loader = _loader({FLUX_SPEC.model_id: FLUX_CFG})
    # Static detection alone does NOT skip it.
    assert preflight_check(FLUX_SPEC, config_loader=loader) == (True, None)

    sig = arch_signature(FLUX_SPEC, FLUX_CFG)
    bank.save(make_anti_pattern_lesson(
        FLUX_SPEC, sig, "metric=0 -> backend produced no throughput", "2.28.0"))

    ok, reason = preflight_check(
        FLUX_SPEC, bank=bank, sdk_version="2.28.0", config_loader=loader)
    assert ok is False
    assert "no throughput" in reason


def test_recorded_anti_pattern_prunes_sibling_by_arch(tmp_path: Path):
    """The lesson is keyed by ARCH signature, so a sibling model (different
    model_id, same arch) is pruned too."""
    bank = KnowledgeBank(tmp_path / "bank")
    sig = arch_signature(FLUX_SPEC, FLUX_CFG)
    bank.save(make_anti_pattern_lesson(FLUX_SPEC, sig, "arch produces 0 img/s", "2.28.0"))

    sibling = ModelSpec(model_id="black-forest-labs/FLUX.1-dev",
                        family="diffusion", param_count=12e9,
                        parent="black-forest-labs")
    sibling_cfg = {"architectures": ["FluxPipeline"], "model_type": "flux"}
    ok, reason = preflight_check(
        sibling, bank=bank, sdk_version="2.28.0",
        config_loader=_loader({sibling.model_id: sibling_cfg}))
    assert ok is False
    assert "0 img/s" in reason


def test_dense_not_gated_by_unrelated_anti_pattern(tmp_path: Path):
    """A recorded linear-attn anti-pattern must NOT gate a working dense model
    (different family + different arch signature)."""
    bank = KnowledgeBank(tmp_path / "bank")
    sig = arch_signature(DELTANET_SPEC, DELTANET_CFG)
    bank.save(make_anti_pattern_lesson(DELTANET_SPEC, sig, "arch aborts", "2.28.0"))

    ok, reason = preflight_check(
        DENSE_SPEC, bank=bank, sdk_version="2.28.0",
        config_loader=_loader({DENSE_SPEC.model_id: DENSE_CFG}))
    assert ok is True and reason is None


def test_preflight_lesson_never_prunes_a_real_config(tmp_path: Path):
    """The arch-signature matcher must not accidentally prune a Stage-1 config
    candidate: bank.prune keys on config axes, which never include
    `arch_signature`/`model_id`."""
    bank = KnowledgeBank(tmp_path / "bank")
    sig = arch_signature(DELTANET_SPEC, DELTANET_CFG)
    lesson = make_anti_pattern_lesson(DELTANET_SPEC, sig, "arch aborts", "2.28.0")
    lesson.tier = Tier.VERIFIED          # even promoted, it must not prune configs
    bank.save(lesson)

    candidates = [{"tp_degree": 8, "weights_dtype": "bf16"},
                  {"tp_degree": 4, "attn_implementation": "sdpa"}]
    survivors, pruned = bank.prune(
        candidates, "hybrid_attention_causal_lm", "2.28.0")
    assert survivors == candidates
    assert pruned == []


# -- 0-metric is a real failure, not a benign 0 ------------------------------

class _ZeroMetricBackend(MockBackend):
    """A backend whose *config candidates* report 0 throughput — the diffusion
    "0 img/s" silent failure. The Stage-0 baseline measures normally (positive),
    so the run has a valid incumbent; only the searched candidates come back at
    0, exercising the Stage-1 config gate. (A 0-metric BASELINE is now its own
    honest FAIL_NO_BASELINE case — see test_moe_baseline_fix.)"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._measured = 0

    def measure(self, neff, shape, batch):
        m = super().measure(neff, shape, batch)
        self._measured += 1
        if self._measured == 1:
            return m                       # first call = baseline: keep positive
        m.metric = 0.0
        m.metric_p50 = 0.0
        m.metric_p99 = 0.0
        return m


def test_zero_metric_recorded_as_failure_not_benign(tmp_path: Path):
    """A 0-metric candidate that 'passes' equivalence (the always-equivalent
    mock checker) is recorded as an explicit FAIL and discarded — never kept."""
    orch = Orchestrator(
        backend=_ZeroMetricBackend(seed=3), bank=KnowledgeBank(tmp_path / "b"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "r"),
        equivalence=always_equivalent, sdk_version="2.28.0",
    )
    orch.ledger.init()
    spec = ModelSpec("m", "dense_causal_lm", 8e9, "qwen")
    orch.run_stage1_config(spec)

    rows = orch.ledger.read()
    zero_fails = [r for r in rows
                  if "produced no throughput" in r.description]
    assert zero_fails, "a 0-metric candidate must be recorded as an explicit FAIL"
    assert all(r.status is Status.DISCARD for r in zero_fails)
    # And no 0-metric row was ever kept.
    assert not any(r.status is Status.KEEP and r.metric <= 0.0
                   and r.stage is Stage.CONFIG for r in rows)
