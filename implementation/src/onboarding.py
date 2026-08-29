"""
Auto-architecture onboarding — Phase 1: Tier-0 config-driven family mapping.

Given only an HF model id the framework has never seen, read its `config.json`
(cheap, NO weights), recognize its architecture *by shape* (not by name), and —
if the load-bearing STRUCTURE matches a known family exactly — emit the
`ModelSpec` the existing Stage 0->6 loop already consumes, plus the resolved
family label. The loop can then baseline + optimize it with no human.

This is Tier 0 of the three-tier design in ../../docs/auto-onboarding-design.md.
Only Tier 0 is IMPLEMENTED here; Tier 1 (adapter synthesis for near-misses) and
Tier 2 (attempt-and-diagnose for novel primitives) are returned as *verdicts*
(flags with a precise reason) so the caller can record a `needs-onboarding`
lesson instead of burning a compile — they are NOT built in Phase 1.

Matching threshold (locked decision Q1): **strict-on-structure,
flexible-on-dimensions.** Tier-0 auto-map ONLY if the structure matches a known
family exactly — attention type (full vs GQA + RoPE presence), norm type
(RMSNorm vs LayerNorm), activation (gated-GLU: SwiGLU/GeGLU vs plain GELU), and
MoE-or-not. If the structure matches but a DIMENSION is out of the known
adapter's validated range (new KV ratio, new head_dim, larger vocab) ->
`TIER1_SYNTHESIZE`. If a STRUCTURAL axis is novel (linear-attn / GatedDeltaNet,
unknown norm/attention) -> `TIER2_DIAGNOSE`.

Correctness reference for an onboarded baseline (locked decision Q2) is HF
CPU-eager — but that lives in the equivalence gate / normal ladder, not here.
This module never loads weights and never touches a device.

Import-safe on CPU: the only heavy dependency (transformers) is reached lazily
through `preflight.load_hf_config`, and the bank/ledger imports used by the
integration hook are done inside the function that needs them.

See ../../docs/auto-onboarding-design.md, preflight.py (the gate this extends),
and backends/qwen38_tp.py (the hand-written adapter pattern this automates for
the common case).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from orchestrator import ModelSpec
from preflight import (
    ConfigLoader,
    _architectures,
    _norm,
    is_linear_attention_arch,
    load_hf_config,
)


# ---------------------------------------------------------------------------
# structural taxonomy (the load-bearing axes we match on, per Q1)
# ---------------------------------------------------------------------------

# Activations that form a GATED-GLU MLP (gate_proj + up_proj + down_proj) — the
# structural shape the dense adapter handles. SwiGLU (silu/swish, Llama/Qwen/
# Mistral/Yi) and GeGLU-tanh (gelu_pytorch_tanh, Gemma-2) are both accepted:
# the specific nonlinearity is a within-family variation the standard HF path
# handles. A plain, non-gated GELU (GPT-2/BERT-shaped) is a DIFFERENT structure.
_SWIGLU_ACTS = frozenset({"silu", "swish"})
_GEGLU_ACTS = frozenset({"gelu_pytorch_tanh", "gelu_tanh", "geglu"})

# Known family: the dense causal-LM shape (RMSNorm + RoPE + GQA/MHA + gated-GLU,
# not MoE). This is Llama / Qwen / Mistral / Yi / Gemma-2-shaped.
FAMILY_DENSE = "dense_causal_lm"

# Dimension ranges the dense adapter is VALIDATED for (flexible-on-dimensions,
# but bounded — outside these a Tier-1 plan must be synthesized, not assumed).
#   head_dim: standard transformer head sizes. Gemma-4's 512 (with 4 KV heads)
#     needs the special capped-TP adapter, so it is intentionally out of range.
#   vocab: up to Gemma-2's 256k. Larger embeddings need a vocab-parallel plan.
#   GQA ratio: query-heads / kv-heads must be an integer <= 16 (else KV
#     replication / a bespoke head plan is needed).
_HEAD_DIM_IN_RANGE = frozenset({64, 72, 80, 96, 112, 128, 160, 192, 256})
_VOCAB_MAX_IN_RANGE = 262_144
_GQA_RATIO_MAX = 16


class Verdict(str, Enum):
    """The three onboarding outcomes (Phase 1 implements MAP; the other two are
    flagged for later phases)."""

    MAP = "MAP"                              # Tier 0: structure + dims known -> ModelSpec
    TIER1_SYNTHESIZE = "TIER1_SYNTHESIZE"    # structure known, dim out of range
    TIER2_DIAGNOSE = "TIER2_DIAGNOSE"        # a structural axis is novel


@dataclass
class ArchFingerprint:
    """A cheap, weights-free structural + dimensional fingerprint of a model,
    extracted from its HF config. STRUCTURE is what Tier-0 matches on;
    DIMENSIONS are checked against the matched family's validated ranges."""

    # --- structure (the load-bearing axes) ---
    attention: str              # "full" (MHA) | "gqa"
    has_rope: bool
    norm: str                   # "rmsnorm" | "layernorm" | "unknown"
    activation: str             # "swiglu" | "geglu" | "gelu" | "other"
    is_moe: bool
    is_linear_attn: bool
    # --- dimensions ---
    num_layers: int
    hidden: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    max_position_embeddings: int
    num_experts: int | None = None
    top_k: int | None = None
    # provenance / debug
    arch_name: str = ""
    model_type: str = ""
    raw_activation: str = ""


@dataclass
class OnboardVerdict:
    """The result of matching a fingerprint to the known families. For MAP,
    `family` + `spec_kwargs` are populated (ready to build a ModelSpec); for the
    Tier-1/2 verdicts only `reason` is meaningful (a precise, actionable flag)."""

    verdict: Verdict
    reason: str
    family: str | None = None
    spec_kwargs: dict[str, Any] | None = None
    fingerprint: ArchFingerprint | None = None


# ---------------------------------------------------------------------------
# small config accessors (tolerate nested text_config; multiple spellings)
# ---------------------------------------------------------------------------

def _text_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """The decoder config, unwrapping a multimodal `text_config` if present.
    Falls back to the top-level config (the common single-tower case)."""
    tc = config.get("text_config")
    return tc if isinstance(tc, dict) else config


def _first_int(cfg: dict[str, Any], *names: str, default: int = 0) -> int:
    for n in names:
        v = cfg.get(n)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
    return default


def _classify_activation(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r in _SWIGLU_ACTS:
        return "swiglu"
    if r in _GEGLU_ACTS:
        return "geglu"
    if r.startswith("gelu"):
        return "gelu"
    return "other" if r else "swiglu"  # HF Llama omits it; silu/SwiGLU is the default


def _classify_norm(cfg: dict[str, Any]) -> str:
    if any(k in cfg for k in ("rms_norm_eps",)):
        return "rmsnorm"
    if any(k in cfg for k in ("layer_norm_epsilon", "layer_norm_eps",
                              "layernorm_epsilon", "ln_eps")):
        return "layernorm"
    # Fall back to model_type / arch for configs that omit an explicit eps key.
    hay = _norm(f"{cfg.get('model_type', '')} "
                f"{' '.join(_architectures(cfg))}")
    if any(m in hay for m in ("llama", "qwen", "mistral", "yi", "gemma",
                              "phi3", "mixtral")):
        return "rmsnorm"
    if any(m in hay for m in ("gpt2", "gptj", "gptneox", "bert", "falcon",
                              "opt", "bloom")):
        return "layernorm"
    return "unknown"


def _has_rope(cfg: dict[str, Any]) -> bool:
    if "rope_theta" in cfg or "rope_scaling" in cfg:
        return True
    pet = str(cfg.get("position_embedding_type", "")).lower()
    return pet in ("rotary", "rope")


# ---------------------------------------------------------------------------
# 1. fingerprint
# ---------------------------------------------------------------------------

def fingerprint(
    model_id_or_config: str | dict[str, Any],
    config_loader: ConfigLoader | None = None,
) -> ArchFingerprint | None:
    """Cheap, weights-free fingerprint of a model's architecture.

    Accepts either a model id (loaded via `preflight.load_hf_config`, honoring an
    injected `config_loader` for tests) or an already-loaded config dict. Returns
    None if the config cannot be read — the caller then treats the model as
    UNKNOWN (never auto-maps it).
    """
    if isinstance(model_id_or_config, dict):
        config: dict[str, Any] | None = model_id_or_config
    else:
        config = load_hf_config(model_id_or_config, config_loader)
    if not config:
        return None

    cfg = _text_cfg(config)

    heads = _first_int(cfg, "num_attention_heads", "n_head", "num_heads", default=0)
    kv = _first_int(cfg, "num_key_value_heads", "num_kv_heads", default=0) or heads
    hidden = _first_int(cfg, "hidden_size", "n_embd", "d_model", default=0)
    head_dim = _first_int(cfg, "head_dim", default=0) or (
        hidden // heads if heads else 0)

    raw_act = str(cfg.get("hidden_act")
                  or cfg.get("hidden_activation")
                  or cfg.get("activation_function") or "")

    archs = _architectures(config)
    return ArchFingerprint(
        attention="gqa" if (kv and heads and kv < heads) else "full",
        has_rope=_has_rope(cfg),
        norm=_classify_norm(cfg),
        activation=_classify_activation(raw_act),
        is_moe=_is_moe(config),
        is_linear_attn=is_linear_attention_arch(config),
        num_layers=_first_int(cfg, "num_hidden_layers", "n_layer", "num_layers",
                              default=0),
        hidden=hidden,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        intermediate_size=_first_int(cfg, "intermediate_size", "ffn_dim",
                                     "n_inner", default=4 * hidden),
        vocab_size=_first_int(cfg, "vocab_size", default=0),
        rope_theta=float(cfg.get("rope_theta") or 0.0),
        max_position_embeddings=_first_int(
            cfg, "max_position_embeddings", "n_positions", "max_seq_len",
            default=0),
        num_experts=_moe_experts(cfg),
        top_k=_moe_top_k(cfg),
        arch_name=archs[0] if archs else "",
        model_type=str(cfg.get("model_type", "")),
        raw_activation=raw_act,
    )


def _is_moe(config: dict[str, Any]) -> bool:
    """MoE-or-not, off the config only (mirrors kernels.moe_fused.is_moe_arch,
    but on a plain dict). An `architectures` entry naming Moe/MoE, or a routed-
    expert count > 1, marks a sparse MoE causal LM."""
    if any("moe" in a.lower() for a in _architectures(config)):
        return True
    cfg = _text_cfg(config)
    return (_moe_experts(cfg) or 0) > 1


def _moe_experts(cfg: dict[str, Any]) -> int | None:
    v = _first_int(cfg, "num_experts", "num_local_experts", "n_routed_experts",
                   default=0)
    return v or None


def _moe_top_k(cfg: dict[str, Any]) -> int | None:
    v = _first_int(cfg, "num_experts_per_tok", "moe_topk", "top_k",
                   "num_experts_per_token", default=0)
    return v or None


# ---------------------------------------------------------------------------
# 2. match_family
# ---------------------------------------------------------------------------

def _parent_of(fp: ArchFingerprint, model_id: str = "") -> str:
    """Best-effort informational parent tag (llama|qwen|mistral|yi|gemma|...),
    from model_type / arch name / model-id org. Never load-bearing."""
    hay = _norm(f"{fp.model_type} {fp.arch_name} {model_id}")
    for tag in ("qwen", "llama", "mistral", "mixtral", "yi", "gemma", "phi",
                "deepseek", "falcon"):
        if tag in hay:
            return tag
    return (fp.model_type or "unknown").lower()


def _estimate_param_count(fp: ArchFingerprint) -> float:
    """bf16 param-count estimate from dims — the same closed form the native
    backend uses to size the instance: (4 h^2 + 3 h I) per layer + 2 V h."""
    h, inter, L, V = (fp.hidden, fp.intermediate_size, fp.num_layers,
                      fp.vocab_size)
    if not (h and L):
        return 0.0
    return float((4 * h * h + 3 * h * inter) * L + 2 * V * h)


def _probe_shape(fp: ArchFingerprint) -> str:
    """Probe shape from max_position_embeddings, in the loop's existing
    "chat <in>/<out>" convention. Caps the input at the 512 seed convention but
    never exceeds the model's own context window."""
    mp = fp.max_position_embeddings or 1024
    in_len = min(512, mp)
    out_len = max(1, in_len // 2)
    return f"chat {in_len}/{out_len}"


def match_family(fp: ArchFingerprint | None, model_id: str = "") -> OnboardVerdict:
    """Match a fingerprint to a known family. Strict-on-structure (Q1):

      - a novel STRUCTURAL axis (linear-attn, non-RMSNorm, no-RoPE, non-gated
        activation) -> TIER2_DIAGNOSE (needs a new adapter/kernel; Phase 3),
      - structure matches the dense family but a DIMENSION is out of the
        validated range (head_dim / vocab / GQA ratio) -> TIER1_SYNTHESIZE
        (needs a synthesized parallelism plan; Phase 2),
      - structure matches AND dims in range -> MAP (Tier 0: emit ModelSpec).
    """
    if fp is None:
        return OnboardVerdict(
            Verdict.TIER2_DIAGNOSE,
            "config unavailable/unreadable — cannot fingerprint (no auto-map)",
        )

    # --- structural gate (strict) ------------------------------------------
    # Linear attention / GatedDeltaNet is a genuinely novel primitive (this is
    # the pre-flight gate's ISA-fail class) — Tier 2, hand off to Stage 4/human.
    if fp.is_linear_attn:
        return OnboardVerdict(
            Verdict.TIER2_DIAGNOSE,
            f"linear-attention/GatedDeltaNet (arch={fp.arch_name or fp.model_type}): "
            "novel attention primitive, no known family adapter — needs a "
            "dedicated kernel/adapter (Tier-2 diagnose)",
            fingerprint=fp,
        )
    if fp.norm != "rmsnorm":
        return OnboardVerdict(
            Verdict.TIER2_DIAGNOSE,
            f"norm={fp.norm!r} is not RMSNorm — outside the known (dense_causal_lm) "
            "family's structure (Tier-2 diagnose)",
            fingerprint=fp,
        )
    if not fp.has_rope:
        return OnboardVerdict(
            Verdict.TIER2_DIAGNOSE,
            "no RoPE detected (absolute/ALiBi/other positional scheme) — "
            "outside the known family's structure (Tier-2 diagnose)",
            fingerprint=fp,
        )
    if fp.activation not in ("swiglu", "geglu"):
        return OnboardVerdict(
            Verdict.TIER2_DIAGNOSE,
            f"activation={fp.raw_activation!r} is not a gated-GLU MLP "
            "(SwiGLU/GeGLU) — outside the known family's structure "
            "(Tier-2 diagnose)",
            fingerprint=fp,
        )

    # MoE: the structure is recognizable, but routing + expert-parallel layout
    # is a parameterizable plan the dense adapter does not carry. Tier-1
    # synthesis (Phase 2), not a Tier-0 map. (Kept a distinct branch so a future
    # moe_causal_lm Tier-0 family can slot in here.)
    # HOW TO BUILD THE EXPERT LAYOUT: see docs/large-model-playbook.md — experts
    # are the memory bulk and the dense TP path does NOT shard them (that is the
    # Qwen3.5-30B OOM); the fix is expert-TP in backends/qwen38_tp.py:shard_moe.
    if fp.is_moe:
        return OnboardVerdict(
            Verdict.TIER1_SYNTHESIZE,
            f"MoE causal LM (num_experts={fp.num_experts}, top_k={fp.top_k}): "
            "dense structure recognized but needs a synthesized expert-parallel/"
            "routing plan (Tier-1 synthesize)",
            fingerprint=fp,
        )

    # --- structure MATCHES the dense family. Now check DIMENSIONS in range. ---
    # (flexible-on-dimensions, but bounded — out of range -> Tier-1 synthesize.)
    reasons: list[str] = []
    if fp.head_dim not in _HEAD_DIM_IN_RANGE:
        reasons.append(
            f"head_dim={fp.head_dim} outside validated range "
            f"{sorted(_HEAD_DIM_IN_RANGE)}")
    if fp.vocab_size > _VOCAB_MAX_IN_RANGE:
        reasons.append(
            f"vocab_size={fp.vocab_size} > {_VOCAB_MAX_IN_RANGE} "
            "(needs vocab-parallel plan)")
    if fp.num_key_value_heads <= 0 or fp.num_attention_heads <= 0:
        reasons.append("missing/zero head counts")
    elif fp.num_attention_heads % fp.num_key_value_heads != 0:
        reasons.append(
            f"num_attention_heads={fp.num_attention_heads} not divisible by "
            f"num_key_value_heads={fp.num_key_value_heads} (irregular GQA)")
    else:
        ratio = fp.num_attention_heads // fp.num_key_value_heads
        if ratio > _GQA_RATIO_MAX:
            reasons.append(
                f"GQA ratio={ratio} > {_GQA_RATIO_MAX} (needs KV-replication plan)")

    if reasons:
        return OnboardVerdict(
            Verdict.TIER1_SYNTHESIZE,
            f"{FAMILY_DENSE} structure matches but dimension(s) out of range: "
            + "; ".join(reasons) + " (Tier-1 synthesize)",
            fingerprint=fp,
        )

    # --- Tier-0 MAP: produce the ModelSpec kwargs the loop consumes. ---------
    spec_kwargs: dict[str, Any] = {
        "family": FAMILY_DENSE,
        "param_count": _estimate_param_count(fp),
        "num_kv_heads": fp.num_key_value_heads,
        "probe_shape": _probe_shape(fp),
        "parent": _parent_of(fp, model_id),
    }
    return OnboardVerdict(
        Verdict.MAP,
        f"{FAMILY_DENSE}: RMSNorm + RoPE + "
        f"{'GQA' if fp.attention == 'gqa' else 'MHA'} + "
        f"{fp.activation.upper()} (Llama/Qwen/Mistral/Yi/Gemma-2-shaped)",
        family=FAMILY_DENSE,
        spec_kwargs=spec_kwargs,
        fingerprint=fp,
    )


# ---------------------------------------------------------------------------
# 3. auto_spec — fingerprint -> match -> ModelSpec (or None for Tier-1/2)
# ---------------------------------------------------------------------------

def auto_onboard(
    model_id: str, config_loader: ConfigLoader | None = None,
) -> OnboardVerdict:
    """Full Tier-0 pipeline for a model id: fingerprint -> match_family.
    Returns the verdict (MAP carries the ModelSpec kwargs; Tier-1/2 carry the
    reason). Never loads weights, never touches a device."""
    fp = fingerprint(model_id, config_loader)
    return match_family(fp, model_id)


def auto_spec(
    model_id: str, config_loader: ConfigLoader | None = None,
) -> ModelSpec | None:
    """Orchestrate fingerprint->match->ModelSpec. Returns a ready-to-run
    `ModelSpec` for a Tier-0 MAP, or None for a Tier-1/2 verdict (the reason is
    available via `auto_onboard(model_id)` for the caller to record)."""
    v = auto_onboard(model_id, config_loader)
    if v.verdict is not Verdict.MAP or not v.spec_kwargs:
        return None
    return ModelSpec(model_id=model_id, **v.spec_kwargs)


# ---------------------------------------------------------------------------
# 4. integration hook — plugs into the pre-flight gate / overnight.run_one
# ---------------------------------------------------------------------------

# Sentinel families that mean "no explicit family — please auto-onboard".
# Kept additive: SEED_MODELS all carry an explicit family, so this never fires
# for them. A discovery/queue entry sets family to one of these to request it.
AUTO_FAMILIES = frozenset({"", "auto", "unknown", None})


def needs_auto_onboard(spec: ModelSpec) -> bool:
    """True if this queued spec has no explicit family and should be auto-mapped
    before the loop runs. Explicit-family specs (all seeds) are untouched."""
    return getattr(spec, "family", None) in AUTO_FAMILIES


def resolve_onboarding(
    spec: ModelSpec, config_loader: ConfigLoader | None = None,
) -> tuple[ModelSpec | None, OnboardVerdict]:
    """The hook the loop calls for a family-less queued model.

    Returns (resolved_spec, verdict):
      - Tier-0 MAP -> (a fully-populated ModelSpec, verdict) : proceed into the
        loop exactly like an explicit-family model.
      - Tier-1/2  -> (None, verdict) : the caller records a `needs-onboarding`
        lesson (reuse the anti-pattern/bank path) instead of FAIL_NO_BASELINE.

    The returned MAP spec preserves the queued spec's non-structural knobs
    (track / long_context / probe_batch) so a discovery entry can still express
    intent; structure-derived fields come from the fingerprint.
    """
    verdict = auto_onboard(spec.model_id, config_loader)
    if verdict.verdict is not Verdict.MAP or not verdict.spec_kwargs:
        return None, verdict
    kw = dict(verdict.spec_kwargs)
    resolved = ModelSpec(
        model_id=spec.model_id,
        track=getattr(spec, "track", "throughput"),
        long_context=getattr(spec, "long_context", False),
        probe_batch=getattr(spec, "probe_batch", 1),
        **kw,
    )
    return resolved, verdict


def make_needs_onboarding_lesson(
    spec: ModelSpec, verdict: OnboardVerdict, sdk_version: str = "",
):
    """Build the `needs-onboarding` lesson for a Tier-1/2 verdict, REUSING the
    pre-flight anti-pattern path (keyed by arch-signature so a sibling of the
    same unseen shape is flagged too). Recorded instead of a spurious
    FAIL_NO_BASELINE: the model is *queued for onboarding* (Phase 2/3), not
    dropped. Imports are local so this module stays import-safe on CPU."""
    from preflight import arch_signature, load_hf_config, make_anti_pattern_lesson

    sig = arch_signature(spec, load_hf_config(spec.model_id))
    reason = f"needs-onboarding [{verdict.verdict.value}]: {verdict.reason}"
    return make_anti_pattern_lesson(spec, sig, reason, sdk_version)
