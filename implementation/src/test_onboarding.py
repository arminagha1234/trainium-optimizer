"""
Tests for Tier-0 auto-architecture onboarding (Phase 1).

CPU-mock only: NO weights, NO device, NO transformers. HF configs are injected
as plain dicts (either passed straight to `fingerprint`, or served through a
`config_loader` exactly like test_preflight.py does). Every assertion is about
the STRUCTURE->verdict decision and the ModelSpec a MAP produces.

Covers the five shapes the design calls out plus Yi-1.5-9B's real config:
  - Llama-shaped dense (GQA)              -> MAP dense_causal_lm
  - Qwen-shaped GQA                       -> MAP dense_causal_lm
  - out-of-range dimension (head_dim 512) -> TIER1_SYNTHESIZE
  - GatedDeltaNet / linear-attn           -> TIER2_DIAGNOSE
  - MoE                                   -> TIER1_SYNTHESIZE
  - Yi-1.5-9B (real Llama-arch shape)     -> MAP dense_causal_lm
  - GPT-2-shaped (LayerNorm, no RoPE)     -> TIER2_DIAGNOSE (structural)
"""

from __future__ import annotations

from onboarding import (
    FAMILY_DENSE,
    Verdict,
    auto_onboard,
    auto_spec,
    fingerprint,
    make_needs_onboarding_lesson,
    match_family,
    needs_auto_onboard,
    resolve_onboarding,
)
from orchestrator import ModelSpec


# -- sample configs (real-shaped) --------------------------------------------

# Llama-3-8B-shaped: GQA 32/8, head_dim 128, silu (SwiGLU), rms_norm, rope.
LLAMA_CFG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
    "hidden_act": "silu", "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
    "max_position_embeddings": 8192,
}

# Qwen3-shaped GQA: 32/8, head_dim 128, silu, rms_norm, rope.
QWEN_CFG = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
    "hidden_size": 4096, "num_hidden_layers": 36, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 12288, "vocab_size": 151936,
    "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "max_position_embeddings": 32768,
}

# Yi-1.5-9B real shape (LlamaForCausalLM): 48 layers, hidden 4096, 32 heads /
# 4 KV (head_dim 128), intermediate 11008, vocab 64000, rope_theta 5e6, silu.
YI_15_9B_CFG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 4096, "num_hidden_layers": 48, "num_attention_heads": 32,
    "num_key_value_heads": 4, "intermediate_size": 11008, "vocab_size": 64000,
    "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 5000000.0,
    "max_position_embeddings": 4096,
}

# Structure matches dense, but head_dim=512 (Gemma-4 Global-style) is out of the
# validated range -> Tier-1 synthesize.
BIG_HEADDIM_CFG = {
    "architectures": ["SomeNewForCausalLM"], "model_type": "somenew",
    "hidden_size": 4096, "num_hidden_layers": 40, "num_attention_heads": 8,
    "num_key_value_heads": 4, "head_dim": 512, "intermediate_size": 16384,
    "vocab_size": 128000, "hidden_act": "silu", "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "max_position_embeddings": 8192,
}

# MoE (Mixtral-shaped): dense structure but routed experts -> Tier-1 synthesize.
MOE_CFG = {
    "architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
    "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 32000,
    "hidden_act": "silu", "rms_norm_eps": 1e-5, "rope_theta": 1000000.0,
    "max_position_embeddings": 32768, "num_local_experts": 8,
    "num_experts_per_tok": 2,
}

# Linear-attention / GatedDeltaNet -> Tier-2 diagnose (novel primitive).
DELTANET_CFG = {
    "architectures": ["Qwen3_5GatedDeltaNetForCausalLM"],
    "model_type": "qwen3_5_gated_deltanet",
    "hidden_size": 4096, "num_hidden_layers": 48, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 12288, "vocab_size": 151936,
    "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
    "max_position_embeddings": 32768,
}

# GPT-2-shaped: LayerNorm + learned positions (no RoPE), non-gated GELU -> the
# structure is novel to our family set -> Tier-2 diagnose.
GPT2_CFG = {
    "architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
    "n_embd": 1600, "n_layer": 48, "n_head": 25, "n_inner": 6400,
    "vocab_size": 50257, "activation_function": "gelu_new",
    "layer_norm_epsilon": 1e-5, "n_positions": 1024,
}


def _loader(mapping):
    """A config_loader that serves canned configs (mirrors test_preflight)."""
    return lambda model_id: mapping.get(model_id)


# -- fingerprint -------------------------------------------------------------

def test_fingerprint_llama_structure():
    fp = fingerprint(LLAMA_CFG)
    assert fp is not None
    assert fp.attention == "gqa"          # 32 q / 8 kv
    assert fp.has_rope is True
    assert fp.norm == "rmsnorm"
    assert fp.activation == "swiglu"
    assert fp.is_moe is False
    assert fp.is_linear_attn is False
    assert fp.head_dim == 128             # 4096 / 32
    assert fp.num_key_value_heads == 8
    assert fp.max_position_embeddings == 8192


def test_fingerprint_full_attention_when_kv_equals_heads():
    cfg = dict(LLAMA_CFG, num_key_value_heads=32)
    assert fingerprint(cfg).attention == "full"


def test_fingerprint_gpt2_layernorm_no_rope():
    fp = fingerprint(GPT2_CFG)
    assert fp.norm == "layernorm"
    assert fp.has_rope is False
    assert fp.activation == "gelu"
    assert fp.head_dim == 1600 // 25


def test_fingerprint_none_on_unreadable_config():
    assert fingerprint("no/such/model", config_loader=lambda _id: None) is None


# -- match_family verdicts ---------------------------------------------------

def test_llama_maps_to_dense():
    v = match_family(fingerprint(LLAMA_CFG), "meta-llama/Llama-3-8B")
    assert v.verdict is Verdict.MAP
    assert v.family == FAMILY_DENSE
    assert v.spec_kwargs["num_kv_heads"] == 8


def test_qwen_gqa_maps_to_dense():
    v = match_family(fingerprint(QWEN_CFG), "Qwen/Qwen3-something")
    assert v.verdict is Verdict.MAP
    assert v.family == FAMILY_DENSE


def test_out_of_range_head_dim_is_tier1():
    v = match_family(fingerprint(BIG_HEADDIM_CFG))
    assert v.verdict is Verdict.TIER1_SYNTHESIZE
    assert "head_dim" in v.reason


def test_huge_vocab_is_tier1():
    v = match_family(fingerprint(dict(LLAMA_CFG, vocab_size=400_000)))
    assert v.verdict is Verdict.TIER1_SYNTHESIZE
    assert "vocab" in v.reason


def test_irregular_gqa_is_tier1():
    # 30 q heads not divisible by 8 kv heads.
    cfg = dict(LLAMA_CFG, num_attention_heads=30, hidden_size=30 * 128)
    v = match_family(fingerprint(cfg))
    assert v.verdict is Verdict.TIER1_SYNTHESIZE


def test_moe_is_tier1():
    v = match_family(fingerprint(MOE_CFG))
    assert v.verdict is Verdict.TIER1_SYNTHESIZE
    assert "MoE" in v.reason
    assert v.fingerprint.is_moe is True


def test_deltanet_is_tier2():
    v = match_family(fingerprint(DELTANET_CFG))
    assert v.verdict is Verdict.TIER2_DIAGNOSE
    assert "linear-attention" in v.reason.lower()


def test_gpt2_layernorm_is_tier2():
    v = match_family(fingerprint(GPT2_CFG))
    assert v.verdict is Verdict.TIER2_DIAGNOSE
    # first structural gate hit is the norm (RMSNorm required)
    assert "rmsnorm" in v.reason.lower() or "norm" in v.reason.lower()


def test_none_fingerprint_is_tier2():
    assert match_family(None).verdict is Verdict.TIER2_DIAGNOSE


# -- Yi-1.5-9B end to end ----------------------------------------------------

def test_yi_15_9b_maps_to_dense_with_sane_spec():
    fp = fingerprint(YI_15_9B_CFG)
    assert fp.attention == "gqa" and fp.head_dim == 128 and fp.norm == "rmsnorm"

    v = match_family(fp, "01-ai/Yi-1.5-9B")
    assert v.verdict is Verdict.MAP
    assert v.family == FAMILY_DENSE

    spec = auto_spec("01-ai/Yi-1.5-9B",
                     config_loader=_loader({"01-ai/Yi-1.5-9B": YI_15_9B_CFG}))
    assert isinstance(spec, ModelSpec)
    assert spec.family == FAMILY_DENSE
    assert spec.num_kv_heads == 4
    assert spec.probe_shape == "chat 512/256"       # min(512, 4096)/2
    # param-count estimate should land in a ~9-11B band (bf16 closed form).
    assert 8e9 <= spec.param_count <= 12e9
    # Yi-1.5 uses LlamaForCausalLM, so shape-based detection yields "llama"
    # (recognized by shape, not name) — the org tag "yi" is also acceptable.
    assert spec.parent in ("llama", "yi")


# -- auto_spec returns None for non-MAP --------------------------------------

def test_auto_spec_none_for_tier1():
    ld = _loader({"x/moe": MOE_CFG})
    assert auto_spec("x/moe", config_loader=ld) is None
    assert auto_onboard("x/moe", config_loader=ld).verdict is Verdict.TIER1_SYNTHESIZE


def test_auto_spec_none_for_tier2():
    ld = _loader({"x/dn": DELTANET_CFG})
    assert auto_spec("x/dn", config_loader=ld) is None


# -- integration hook --------------------------------------------------------

def test_needs_auto_onboard_only_for_familyless_specs():
    assert needs_auto_onboard(ModelSpec(model_id="a/b", family="", param_count=0))
    assert needs_auto_onboard(ModelSpec(model_id="a/b", family="auto",
                                        param_count=0))
    # An explicit-family seed is never auto-onboarded.
    assert not needs_auto_onboard(
        ModelSpec(model_id="Qwen/Qwen3-0.6B", family=FAMILY_DENSE,
                  param_count=0.6e9))


def test_resolve_onboarding_map_preserves_intent():
    ld = _loader({"01-ai/Yi-1.5-9B": YI_15_9B_CFG})
    queued = ModelSpec(model_id="01-ai/Yi-1.5-9B", family="auto",
                       param_count=0.0, track="latency", long_context=True)
    resolved, verdict = resolve_onboarding(queued, config_loader=ld)
    assert verdict.verdict is Verdict.MAP
    assert resolved is not None
    assert resolved.family == FAMILY_DENSE
    assert resolved.track == "latency"          # non-structural intent kept
    assert resolved.long_context is True
    assert resolved.num_kv_heads == 4


def test_resolve_onboarding_tier2_returns_none_and_verdict():
    ld = _loader({"x/dn": DELTANET_CFG})
    queued = ModelSpec(model_id="x/dn", family="auto", param_count=0.0)
    resolved, verdict = resolve_onboarding(queued, config_loader=ld)
    assert resolved is None
    assert verdict.verdict is Verdict.TIER2_DIAGNOSE


def test_needs_onboarding_lesson_is_bankable():
    # Reuses the pre-flight anti-pattern path; keyed by arch-signature so a
    # sibling of the same unseen shape is flagged too.
    ld = _loader({"x/dn": DELTANET_CFG})
    queued = ModelSpec(model_id="x/dn", family="auto", param_count=0.0)
    _, verdict = resolve_onboarding(queued, config_loader=ld)
    lesson = make_needs_onboarding_lesson(queued, verdict, sdk_version="2.28.0")
    assert "needs-onboarding" in lesson.reason
    assert lesson.matcher.get("arch_signature")
