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

# A realistic Qwen3-Next / Qwen3.5 GatedDeltaNet-MoE config: linear-attention
# (layer_types names it) AND detectably qwen3-next (model_type + linear markers),
# so the graph-rewrite bundle applies.
QWEN3_NEXT_SPEC = ModelSpec(
    model_id="Qwen/Qwen3.5-0.8B", family="hybrid_attention_causal_lm",
    param_count=8e8, parent="qwen",
)
QWEN3_NEXT_CFG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "model_type": "qwen3_next",
    "layer_types": ["linear_attention", "full_attention"],
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 16,
    "linear_conv_kernel_dim": 4,
}


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


# -- qwen3-next graph-rewrite path (rewrites_wired, opt-in) ------------------

def test_is_qwen3_next_arch_detects_dict_config():
    # preflight works with DICT configs; the detector must handle them (the
    # backend's object-based detector would miss a dict). model_type OR markers.
    from preflight import _is_qwen3_next_arch
    assert _is_qwen3_next_arch(QWEN3_NEXT_CFG)                       # model_type
    assert _is_qwen3_next_arch(                                      # markers only
        {"linear_key_head_dim": 128, "linear_num_value_heads": 16,
         "linear_conv_kernel_dim": 4})
    assert not _is_qwen3_next_arch(DENSE_CFG)
    assert not _is_qwen3_next_arch(DELTANET_CFG)   # not qwen3-next-signalled
    assert not _is_qwen3_next_arch(None)


def test_qwen3_next_proceeds_when_rewrites_wired():
    # The fix: a qwen3-next model is allowed through when the backend installs
    # the graph-rewrite bundle — the rewrites (not a DeltaNet kernel) are the
    # supported correctness path at these scales. No registry / kernel needed.
    loader = _loader({QWEN3_NEXT_SPEC.model_id: QWEN3_NEXT_CFG})
    ok, reason = preflight_check(QWEN3_NEXT_SPEC, config_loader=loader,
                                 rewrites_wired=True)
    assert ok is True, reason
    assert reason is None


def test_qwen3_next_still_skips_without_rewrites_wired():
    # Opt-in: default (rewrites_wired=False) is unchanged — still a linear-attn
    # skip, so existing callers/behaviour are untouched.
    loader = _loader({QWEN3_NEXT_SPEC.model_id: QWEN3_NEXT_CFG})
    ok, reason = preflight_check(QWEN3_NEXT_SPEC, config_loader=loader)
    assert ok is False
    assert reason == LINEAR_ATTN_REASON


def test_rewrites_wired_does_not_admit_non_qwen3_next_linear_attn():
    # rewrites_wired must ONLY admit qwen3-next (the arch the bundle handles) —
    # a different linear-attn arch the detector does not classify as qwen3-next
    # must still skip (the rewrites don't apply to it).
    loader = _loader({DELTANET_SPEC.model_id: DELTANET_CFG})
    ok, reason = preflight_check(DELTANET_SPEC, config_loader=loader,
                                 rewrites_wired=True)
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


# --- ordering: feasibility is asked BEFORE architecture support ---------------
#
# The capability gate used to sit after the architecture routing, which returns
# early. `if rewrites_wired and _is_qwen3_next_arch(cfg): return True, None` fires
# first, so EVERY hybrid linear-attention model bypassed the gate. On a real run
# Qwen3.5-122B-A10B walked straight past it into a baseline it could not load and
# crashed 32 ranks, while the gate had the answer ready (8006 GB of host DRAM
# needed against 2147). DeepSeek-V4-Flash, not a linear-attention arch, fell
# through to the gate on the SAME run and was correctly skipped in 10 seconds --
# which is what makes this an ordering bug rather than a gate bug.

# Qwen3.5-122B-A10B's real shape. linear_num_value_heads matters: value heads cannot
# be replicated (out_proj all-reduces), so they bound tp -- the small fixture this
# derives from carries 16, which would cap a 32-head model at tp=16 and is not what
# the real model does.
_HUGE_QWEN3_NEXT_CFG = dict(QWEN3_NEXT_CFG, **{
    "hidden_size": 3072, "num_hidden_layers": 48, "num_attention_heads": 32,
    "num_key_value_heads": 2, "head_dim": 256, "moe_intermediate_size": 1024,
    "intermediate_size": 1024, "num_experts": 256, "vocab_size": 151936,
    "linear_num_key_heads": 16, "linear_num_value_heads": 64,
})


def test_an_unfittable_model_is_skipped_even_when_rewrites_are_wired():
    """Being SUPPORTED is not being FEASIBLE. 122B's real shape, 250 GB measured."""
    from capability import TRN2_48XLARGE

    loader = _loader({QWEN3_NEXT_SPEC.model_id: _HUGE_QWEN3_NEXT_CFG})
    ok, reason = preflight_check(QWEN3_NEXT_SPEC, config_loader=loader,
                                 rewrites_wired=True,
                                 hardware=TRN2_48XLARGE, weight_gb=250.2)
    assert ok is False, "a model that cannot be loaded must not reach a baseline"
    assert reason and reason.startswith("capability:")
    assert "host DRAM" in reason


def test_the_same_model_proceeds_once_the_loader_makes_it_fit(monkeypatch):
    """And the gate must not become a blanket ban on large hybrids."""
    from capability import TRN2_48XLARGE

    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", "4")
    loader = _loader({QWEN3_NEXT_SPEC.model_id: _HUGE_QWEN3_NEXT_CFG})
    ok, reason = preflight_check(QWEN3_NEXT_SPEC, config_loader=loader,
                                 rewrites_wired=True,
                                 hardware=TRN2_48XLARGE, weight_gb=250.2)
    assert ok is True, reason


def test_a_fitting_hybrid_still_proceeds_with_hardware_supplied():
    """The ordering change must not disturb the small-model path."""
    from capability import TRN2_48XLARGE

    loader = _loader({QWEN3_NEXT_SPEC.model_id: QWEN3_NEXT_CFG})
    ok, reason = preflight_check(QWEN3_NEXT_SPEC, config_loader=loader,
                                 rewrites_wired=True,
                                 hardware=TRN2_48XLARGE, weight_gb=1.6)
    assert ok is True, reason


def test_without_hardware_the_gate_is_skipped_and_routing_is_unchanged():
    """No hardware profile means no feasibility opinion -- fail open, as before."""
    loader = _loader({QWEN3_NEXT_SPEC.model_id: _HUGE_QWEN3_NEXT_CFG})
    ok, reason = preflight_check(QWEN3_NEXT_SPEC, config_loader=loader,
                                 rewrites_wired=True)
    assert ok is True, reason


# --- a fixed limitation must not stay banked forever --------------------------
#
# DeepSeek-V4-Flash was correctly skipped as HOST_LIMITED, which banked an
# anti-pattern. Staggered loading then made it loadable and the fresh gate said
# RUNNABLE -- but the run was skipped anyway, replaying the stale reason verbatim
# ("64 full copies, one per rank...") long after that stopped being what the loader
# does. The compounding bank had compounded a false memory.

class _FakeLesson:
    def __init__(self, reason, matcher=None, lesson_id="preflight-x"):
        self.reason = reason
        self.matcher = matcher if matcher is not None else {}
        self.lesson_id = lesson_id
        self.evidence = []


class _FakeBank:
    def __init__(self, lessons):
        self._lessons = lessons

    def preflight_antipatterns(self, family, sdk_version):
        return self._lessons


def _cap_lesson(spec, reason="capability: loading 319 GB on 64 ranks needs 20431 GB"):
    from preflight import arch_signature
    sig = arch_signature(spec, DENSE_CFG)
    return _FakeLesson(reason, {"arch_signature": sig, "model_id": spec.model_id})


def test_a_banked_capability_verdict_is_not_replayed():
    """It is re-derived for free at step 0, so a stored copy can only be stale."""
    loader = _loader({DENSE_SPEC.model_id: DENSE_CFG})
    bank = _FakeBank([_cap_lesson(DENSE_SPEC)])
    ok, reason = preflight_check(DENSE_SPEC, bank=bank, config_loader=loader)
    assert ok is True, reason


def test_a_banked_expensive_failure_is_still_replayed():
    """The bank's actual purpose: do not rediscover a compile abort at full price."""
    loader = _loader({DENSE_SPEC.model_id: DENSE_CFG})
    bank = _FakeBank([_FakeLesson(
        "neuronx-cc aborted after 3h on this architecture",
        {"arch_signature": __import__("preflight").arch_signature(DENSE_SPEC, DENSE_CFG),
         "model_id": DENSE_SPEC.model_id})])
    ok, reason = preflight_check(DENSE_SPEC, bank=bank, config_loader=loader)
    assert ok is False
    assert "aborted" in reason


def test_a_tagged_capability_lesson_is_recognised_without_the_prefix():
    """Newly banked entries carry matcher['source'], so the tag is authoritative."""
    from preflight import arch_signature

    loader = _loader({DENSE_SPEC.model_id: DENSE_CFG})
    sig = arch_signature(DENSE_SPEC, DENSE_CFG)
    tagged = _FakeLesson("some reworded capability message",
                         {"arch_signature": sig, "model_id": DENSE_SPEC.model_id,
                          "source": "capability"})
    ok, _ = preflight_check(DENSE_SPEC, bank=_FakeBank([tagged]),
                            config_loader=loader)
    assert ok is True


def test_capability_skips_are_tagged_when_banked():
    """So a future run can tell a cheap verdict from an expensive failure."""
    from preflight import arch_signature, make_anti_pattern_lesson

    sig = arch_signature(DENSE_SPEC, DENSE_CFG)
    cap = make_anti_pattern_lesson(DENSE_SPEC, sig, "capability: 8006 GB of host DRAM")
    assert cap.matcher.get("source") == "capability"

    expensive = make_anti_pattern_lesson(DENSE_SPEC, sig, "neuronx-cc ISA failure")
    assert "source" not in expensive.matcher
