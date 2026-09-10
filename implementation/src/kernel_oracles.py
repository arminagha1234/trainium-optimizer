"""kernel_oracles.py — the CPU-oracle registry: a numpy ground-truth per kernel
primitive, so a kernel's numerics can be validated on ANY box (no Trainium) via
`kernel_validation.verdict`.

An "oracle" is a ``(reference, sim, make_inputs)`` triple:

  * ``reference``   — the numpy GROUND TRUTH the kernel must match. The math,
    derived independently of the kernel.
  * ``sim``         — a SECOND, independently-written numpy implementation of the
    same math (typically kernel-shaped: the layout/algorithm the NKI kernel uses,
    e.g. scatter-free RoPE, or an einsum accumulation of a delta-rule recurrence).
    Comparing ``sim`` against ``reference`` is what makes the offline parity check
    MEAN something.
  * ``make_inputs`` — a deterministic input factory (fixed seed) so the check is
    reproducible.

## Why the "not vacuous" guard exists (the AutoFixer orphan-oracle bug)

The failure this module is built to prevent: an oracle whose ``sim IS its
reference`` (the exact same function object) — or a primitive with NO oracle at
all. Then "sim allclose reference" is comparing a value to ITSELF: it passes
trivially, 100% of the time, and a whole class of kernels sails through the gate
UN-validated. (`invent_engine.offline_gate` documents exactly this: most catalog
recipes reuse ``spec.reference`` verbatim as ``numpy_impl``, so their parity is
vacuous.) So ``Oracle.vacuous`` flags any oracle where ``sim is reference`` or
``sim is None``, and ``audit_oracles()`` surfaces every vacuous/missing primitive
so a silently-skipped class is caught, not shipped.

## Alias resolution

Oracles are keyed by CANONICAL kernel name (matching the corpus / registry:
"DeltaNet", "Mamba2", ...). ``get_oracle(name)`` resolves any primitive spelling
through ``kernel_registry.PRIMITIVE_TO_KERNEL`` (so "gated_delta_net",
"GatedDeltaNet", "gated-delta" all reach the DeltaNet oracle) plus any explicit
aliases a registration declares — the same normalization the registry uses, so
spellings never fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kernel_registry import PRIMITIVE_TO_KERNEL, _norm

RefFn = Callable[[dict], np.ndarray]
InputFn = Callable[[], dict]


@dataclass
class Oracle:
    """A CPU ground-truth triple for one primitive. See module doc."""

    name: str                       # canonical kernel name, e.g. "DeltaNet"
    reference: RefFn
    sim: RefFn | None
    make_inputs: InputFn
    notes: str = ""

    @property
    def vacuous(self) -> bool:
        """True if this oracle cannot actually validate anything: it has no
        independent ``sim`` (``None``) or its ``sim`` IS its ``reference`` (the
        same object), so a parity check would compare a value to itself and pass
        trivially. The orphan-oracle bug this whole module guards against."""
        return self.sim is None or self.sim is self.reference


# canonical-key -> Oracle, and normalized-alias -> canonical-key.
_ORACLES: dict[str, Oracle] = {}
_ALIASES: dict[str, str] = {}


def register_oracle(name: str, reference: RefFn, sim: RefFn | None,
                    make_inputs: InputFn, *, aliases: tuple[str, ...] = (),
                    notes: str = "") -> None:
    """Register an oracle under a canonical ``name`` plus optional ``aliases``.

    All keys are normalized (``kernel_registry._norm``) so spellings collapse the
    same way the registry's do. Registering a vacuous oracle is ALLOWED (it is
    still recorded) — ``audit_oracles()`` is what flags it; we do not silently
    drop it, because a dropped oracle looks identical to a missing one.
    """
    key = _norm(name)
    _ORACLES[key] = Oracle(name=name, reference=reference, sim=sim,
                           make_inputs=make_inputs, notes=notes)
    _ALIASES[key] = key
    for a in aliases:
        _ALIASES[_norm(a)] = key


def get_oracle(name: str) -> Oracle | None:
    """Resolve a primitive/kernel name (any spelling) to its Oracle, or None.

    Resolution order: direct canonical key -> explicit alias -> the
    PRIMITIVE_TO_KERNEL primitive->kernel map (so a primitive descriptor like
    "gated_delta_net" reaches the DeltaNet oracle). None if nothing is registered
    for the resolved kernel (the caller then routes to AUTHOR)."""
    n = _norm(name)
    if n in _ORACLES:
        return _ORACLES[n]
    if n in _ALIASES:
        return _ORACLES.get(_ALIASES[n])
    kname = PRIMITIVE_TO_KERNEL.get(n)      # primitive spelling -> canonical kernel
    if kname:
        kk = _norm(kname)
        if kk in _ORACLES:
            return _ORACLES[kk]
        if kk in _ALIASES:
            return _ORACLES.get(_ALIASES[kk])
    return None


def audit_oracles() -> dict[str, list[str]]:
    """Health report: which registered oracles are VACUOUS, and which kernels
    named in PRIMITIVE_TO_KERNEL have NO oracle at all.

    Returns ``{"vacuous": [...], "missing": [...]}``. Both lists being empty is
    the only "fully covered" state. This is the check that stops a whole primitive
    class from being silently skipped — call it in CI / preflight, not per-run."""
    vacuous = sorted(o.name for o in _ORACLES.values() if o.vacuous)
    covered = set(_ORACLES) | set(_ALIASES)
    missing = sorted({
        kname for kname in PRIMITIVE_TO_KERNEL.values()
        if _norm(kname) not in covered
    })
    return {"vacuous": vacuous, "missing": missing}


# ---------------------------------------------------------------------------
# Built-in oracles. REAL (independent reference/sim) ones reuse the proven
# numpy pairs from invent_kernels so we do not fork the math.
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- gated delta rule (DeltaNet) --------------------------------------------
# A genuine, independently-written pair: the reference builds the recurrence with
# explicit outer products + matmuls; the sim expresses the SAME math with einsum.
# Different code, same result -> a non-vacuous parity check.
def _delta_reference(inp: dict) -> np.ndarray:
    q, k, v, g = inp["q"], inp["k"], inp["v"], inp["g"]     # q,k,v [T,d]; g [T,1]
    T, d = q.shape
    S = np.zeros((d, d), dtype=np.float64)
    out = np.zeros((T, d), dtype=np.float64)
    for t in range(T):
        S = g[t, 0] * S + np.outer(k[t], v[t])              # decay + rank-1 update
        out[t] = q[t] @ S
    return out.astype(np.float32)


def _delta_sim(inp: dict) -> np.ndarray:
    q, k, v, g = inp["q"], inp["k"], inp["v"], inp["g"]
    T, d = q.shape
    S = np.zeros((d, d), dtype=np.float64)
    out = np.zeros((T, d), dtype=np.float64)
    for t in range(T):
        S = g[t, 0] * S + np.einsum("i,j->ij", k[t], v[t])  # einsum vs np.outer
        out[t] = np.einsum("i,ij->j", q[t], S)              # einsum vs @
    return out.astype(np.float32)


def _delta_inputs() -> dict:
    g = _rng(20260822)
    T, d = 32, 16
    return {
        "q": g.standard_normal((T, d)).astype(np.float32),
        "k": g.standard_normal((T, d)).astype(np.float32),
        "v": g.standard_normal((T, d)).astype(np.float32),
        # decay in (0,1) so the recurrence stays bounded.
        "g": (0.5 + 0.4 * g.random((T, 1))).astype(np.float32),
    }


# --- MLA (Multi-head Latent Attention) — DeepSeek-V2/V3, GLM-family ----------
# The two implementations of MLA decode that MUST agree, computed by GENUINELY
# different algorithms (so the parity check is non-vacuous):
#
#   reference — MATERIALIZE: reconstruct the per-head, per-position keys and
#     values from the shared compressed latent (k_nope[h,s] = W_UK[h] @ c[s],
#     v[h,s] = W_UV[h] @ c[s]), then run standard scaled-dot-product attention.
#     A decoupled-RoPE key `kr` (shared across heads) adds a positional score
#     term. This is the "naive" form — O(S * H * d_h) reconstructed KV.
#
#   sim — ABSORB (the form the efficient MLA decode kernel actually uses):
#     fold W_UK into the query (q_abs[h] = q_nope[h] @ W_UK[h]) so attention runs
#     directly against the compressed latent `c`, and fold W_UV into the output
#     (out[h] = (a @ c) @ W_UV[h]^T). The per-position K/V are NEVER
#     materialized. Algebraically identical to `reference` by associativity of
#     the up-projection matmuls — the "matrix absorption" trick.
#
# Because one path reconstructs KV and the other never does, `sim is not
# reference` in every sense that matters: agreement is real evidence.
def _softmax_lastaxis(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _mla_reference(inp: dict) -> np.ndarray:
    c, kr = inp["c"], inp["kr"]          # c [S,d_c] latent cache; kr [S,d_r] rope key
    qn, qr = inp["qn"], inp["qr"]        # qn [H,d_h] query-nope; qr [H,d_r] query-rope
    w_uk, w_uv = inp["w_uk"], inp["w_uv"]  # [H,d_h,d_c], [H,d_v,d_c]
    scale = inp["scale"]
    c, kr, qn, qr = (a.astype(np.float64) for a in (c, kr, qn, qr))
    w_uk, w_uv = w_uk.astype(np.float64), w_uv.astype(np.float64)
    # reconstruct per-head keys/values from the latent (the expensive part)
    k_nope = np.einsum("hxc,sc->hsx", w_uk, c)     # [H,S,d_h]
    v = np.einsum("hvc,sc->hsv", w_uv, c)          # [H,S,d_v]
    nope = np.einsum("hx,hsx->hs", qn, k_nope)     # [H,S]
    rope = np.einsum("hr,sr->hs", qr, kr)          # [H,S] (shared rope key)
    a = _softmax_lastaxis((nope + rope) * scale)   # [H,S]
    out = np.einsum("hs,hsv->hv", a, v)            # [H,d_v]
    return out.astype(np.float32)


def _mla_sim(inp: dict) -> np.ndarray:
    c, kr = inp["c"], inp["kr"]
    qn, qr = inp["qn"], inp["qr"]
    w_uk, w_uv = inp["w_uk"], inp["w_uv"]
    scale = inp["scale"]
    c, kr, qn, qr = (a.astype(np.float64) for a in (c, kr, qn, qr))
    w_uk, w_uv = w_uk.astype(np.float64), w_uv.astype(np.float64)
    # absorb W_UK into the query -> attend directly in the d_c latent space
    q_abs = np.einsum("hx,hxc->hc", qn, w_uk)      # [H,d_c]
    nope = np.einsum("hc,sc->hs", q_abs, c)        # [H,S]  (no K reconstructed)
    rope = np.einsum("hr,sr->hs", qr, kr)          # [H,S]
    a = _softmax_lastaxis((nope + rope) * scale)   # [H,S]
    ctx = np.einsum("hs,sc->hc", a, c)             # [H,d_c] (weighted latent)
    out = np.einsum("hc,hvc->hv", ctx, w_uv)       # [H,d_v] (absorb W_UV)
    return out.astype(np.float32)


def _mla_inputs() -> dict:
    g = _rng(20260909)
    H, S, d_c, d_h, d_r, d_v = 8, 64, 128, 32, 16, 32
    sc = 1.0 / np.sqrt(d_h + d_r)          # DeepSeek scales over the full q dim
    # up-projections scaled by 1/sqrt(d_c) so reconstructed K/V stay O(1).
    return {
        "c":  g.standard_normal((S, d_c)).astype(np.float32),
        "kr": g.standard_normal((S, d_r)).astype(np.float32),
        "qn": g.standard_normal((H, d_h)).astype(np.float32),
        "qr": g.standard_normal((H, d_r)).astype(np.float32),
        "w_uk": (g.standard_normal((H, d_h, d_c)) / np.sqrt(d_c)).astype(np.float32),
        "w_uv": (g.standard_normal((H, d_v, d_c)) / np.sqrt(d_c)).astype(np.float32),
        "scale": sc,
    }


# --- Mamba2 / SSD (state-space duality) -------------------------------------
# The selective state-space scan at the heart of Mamba2, and the primitive the
# harvested mamba2_ssd kernel implements. Two independent derivations:
#
#   reference — SEQUENTIAL RECURRENCE: carry the state h[P,N] step by step,
#     h_t = a_t * h_{t-1} + x_t (outer) B_t, and read out y_t = h_t @ C_t. This
#     is the "linear/recurrent" side of the duality.
#
#   sim — MATERIALIZED SEMISEPARABLE (the "quadratic"/attention side of SSD):
#     build the T x T 1-semiseparable mixing matrix M[t,s] = (C_t . B_s) *
#     prod_{r=s+1..t} a_r for s<=t (0 above the diagonal) and apply y = M @ x.
#     No state is carried; the whole sequence mixes at once.
#
# Same output by the state-space duality, computed by opposite algorithms — a
# non-vacuous parity check that also pins the causal decay (the part that makes
# it a state-space scan and not plain causal linear attention).
def _mamba2_reference(inp: dict) -> np.ndarray:
    x, a, B, C = inp["x"], inp["a"], inp["B"], inp["C"]   # x[T,P] a[T] B[T,N] C[T,N]
    x, a, B, C = (t.astype(np.float64) for t in (x, a, B, C))
    T, P = x.shape
    N = B.shape[1]
    h = np.zeros((P, N), dtype=np.float64)
    y = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        h = a[t] * h + np.outer(x[t], B[t])              # decay + rank-1 input
        y[t] = h @ C[t]                                  # [P,N]@[N] -> [P]
    return y.astype(np.float32)


def _mamba2_sim(inp: dict) -> np.ndarray:
    x, a, B, C = inp["x"], inp["a"], inp["B"], inp["C"]
    x, a, B, C = (t.astype(np.float64) for t in (x, a, B, C))
    pcum = np.cumprod(a)                                 # [T], prod_{0..t} a_r
    # L[t,s] = prod_{r=s+1..t} a_r = pcum[t]/pcum[s] for s<=t, else 0
    ratio = pcum[:, None] / pcum[None, :]                # [T,T]
    L = np.tril(ratio)                                   # causal (incl diagonal)
    scores = C @ B.T                                     # [T,T], C_t . B_s
    M = L * scores                                       # 1-semiseparable mixer
    y = M @ x                                            # [T,P]
    return y.astype(np.float32)


def _mamba2_inputs() -> dict:
    g = _rng(20260910)
    T, P, N = 32, 8, 16
    return {
        "x": g.standard_normal((T, P)).astype(np.float32),
        "B": g.standard_normal((T, N)).astype(np.float32),
        "C": g.standard_normal((T, N)).astype(np.float32),
        # per-step decay in (0.8,0.99): bounded, and not so small the semisep
        # cumulative-product ratios underflow at T=32.
        "a": (0.8 + 0.19 * g.random(T)).astype(np.float32),
    }


# --- RoPE (dense-attention primitive, reused independent pair) --------------
# invent_kernels already ships an independent reference (strided scatter) vs
# kernel-shaped impl (scatter-free stack/flatten). Reuse them verbatim.
def _register_rope() -> None:
    try:
        from invent_kernels import _rope_impl, _rope_inputs, _rope_reference
    except Exception:  # noqa: BLE001 — optional; DeltaNet oracle is enough alone
        return
    register_oracle(
        "RoPE", _rope_reference, _rope_impl,
        lambda: _rope_inputs(128, 128, 1),
        aliases=("rope", "rope_apply", "rotary"),
        notes="strided-scatter reference vs scatter-free stack/flatten sim")


# --- attention sink (decode) — reused independent pair from invent_kernels ---
# full-softmax-with-sink reference vs online/blocked-softmax-with-sink impl: the
# same math by two different algorithms, so the parity check is non-vacuous.
def _register_attn_sink() -> None:
    try:
        from invent_kernels import (_attn_sink_impl, _attn_sink_inputs,
                                     _attn_sink_reference)
    except Exception:  # noqa: BLE001 — optional; the other oracles stand alone
        return
    register_oracle(
        "AttentionSink", _attn_sink_reference, _attn_sink_impl,
        lambda: _attn_sink_inputs(512, 128, 41),
        aliases=("attention_sink", "attn_sink", "sink_attention",
                 "gpt_oss", "gptoss", "sink"),
        notes="attention-sink decode: full-softmax+sink reference vs "
              "online-softmax+sink impl (independent algorithms)")


# --- head_dim=256 decode attention — validates the harvested decode_hd256 ----
# full-head_dim softmax reference vs split-K/split-V online-softmax impl (what
# the customer_armin decode_hd256 kernels do). Independent -> non-vacuous.
def _register_attn_hd256() -> None:
    try:
        from invent_kernels import (_attn_hd256_impl, _attn_hd256_inputs,
                                     _attn_hd256_reference)
    except Exception:  # noqa: BLE001 — optional; the other oracles stand alone
        return
    register_oracle(
        "AttnDecodeHD256", _attn_hd256_reference, _attn_hd256_impl,
        lambda: _attn_hd256_inputs(512, 51),
        aliases=("decode_hd256", "attn_hd256", "hd256", "head_dim_256",
                 "attention_decode_hd256"),
        notes="head_dim=256 decode: full-head_dim softmax reference vs "
              "split-K/split-V online-softmax impl (independent algorithms)")


# --- FlashAttention (long-context dense attention) --------------------------
# The registered, on-device-validated FlashAttention kernel was UNCOVERED by an
# oracle (audit "missing"). Two genuinely independent algorithms for the SAME
# math: the reference MATERIALIZES the full [S,S] score matrix and softmaxes it;
# the sim never materializes it — it streams over K/V blocks with a running max +
# running denominator (the online-softmax rescale flash attention is built on).
# Same result, opposite memory pattern -> a non-vacuous parity check that also
# pins the online-softmax rescale (the part a flash kernel gets wrong).
def _flash_attn_sim(inp: dict) -> np.ndarray:
    q, k, v = inp["q"], inp["k"], inp["v"]           # each [d, S]
    qT = q.T.astype(np.float64)                      # [S, d] (queries as rows)
    kT = k.T.astype(np.float64)                      # [S, d] (keys as rows)
    vT = v.T.astype(np.float64)                      # [S, d] (values as rows)
    S, d = qT.shape
    m = np.full(S, -np.inf)                          # running max per query
    l = np.zeros(S)                                  # running denom per query
    acc = np.zeros((S, d))                           # running weighted sum
    block = 17                                       # NOT a divisor of S: exercise rescale
    for j0 in range(0, S, block):
        kb = kT[j0:j0 + block]                       # [b, d]
        vb = vT[j0:j0 + block]                       # [b, d]
        s = qT @ kb.T                                # [S, b] unscaled scores
        m_new = np.maximum(m, s.max(axis=1))
        alpha = np.exp(m - m_new)                    # rescale prior stats
        p = np.exp(s - m_new[:, None])               # [S, b]
        l = l * alpha + p.sum(axis=1)
        acc = acc * alpha[:, None] + p @ vb
        m = m_new
    return (acc / l[:, None]).astype(np.float32)


def _register_flash() -> None:
    try:
        from invent_kernels import (_flash_attention_inputs,
                                     _flash_attention_reference)
    except Exception:  # noqa: BLE001 — optional; the other oracles stand alone
        return
    register_oracle(
        "FlashAttention", _flash_attention_reference, _flash_attn_sim,
        lambda: _flash_attention_inputs(64, 32, 43),
        aliases=("flash", "flashattn", "flash_attention", "flashattention",
                 "attention_long_context", "long_context_attention",
                 "sliding_window_attention", "gemma4_attention",
                 "hetero_attention"),
        notes="long-context flash attention: full-[S,S]-softmax reference vs "
              "streaming online-softmax (blocked, running max/denom) sim "
              "(independent algorithms)")


# ---------------------------------------------------------------------------
# Known-uncovered primitives — the audit CI gate's explicit, SHRINKING allowlist.
# ---------------------------------------------------------------------------
# These kernels are named in PRIMITIVE_TO_KERNEL but have NO oracle yet, on
# PURPOSE: a non-vacuous oracle needs TWO independent implementations of the same
# math, and for these primitives we do not yet have a GROUNDED reference (a
# published/harvested spec or a _torch twin). Writing the recurrence from memory
# risks a WRONG ground truth — strictly worse than no oracle, since every kernel
# would then be validated against a bug. So they are tracked here, not faked.
#
# The audit gate (test_audit_oracles_is_a_ci_gate) enforces:
#   * ZERO vacuous oracles (the orphan-oracle bug) — always.
#   * every uncovered kernel is IN this set — so adding a NEW primitive without
#     an oracle fails CI until it is either given an oracle or consciously listed.
#   * this set contains NO already-covered kernel — so the allowlist can only
#     SHRINK as oracles land (R3 harvests the _torch twins that ground them).
KNOWN_UNCOVERED: frozenset[str] = frozenset({
    "KDA",            # Kimi Delta Attention — fine-grained (per-channel) gated
                      # delta rule; exact gate form needs a grounded ref (R3).
    "LightningAttn",  # MiniMax lightning attention — per-head decay + norm
                      # convention varies by version; needs a grounded ref.
    "RWKV6",          # RWKV-6 (Finch) WKV recurrence — data-dependent decay.
    "RWKV7",          # RWKV-7 (Goose) delta-rule-with-vector-gating recurrence.
    "mLSTM",          # xLSTM matrix-memory LSTM (parallel form) — specific gates.
    "sLSTM",          # xLSTM scalar LSTM — new exponential gating + normalizer.
    "RGLRU",          # Griffin/RecurrentGemma real-gated linear recurrent unit.
    "PowerRetention", # power-retention variant — grounded ref needed.
    "GlmMoeDsa",      # GLM MoE + sparse attention — composite/model-specific;
                      # a single numeric oracle may not be the right shape.
})


register_oracle(
    "DeltaNet", _delta_reference, _delta_sim, _delta_inputs,
    # aliases beyond what PRIMITIVE_TO_KERNEL already routes; get_oracle also
    # resolves any PRIMITIVE_TO_KERNEL primitive spelling that maps to DeltaNet.
    aliases=("gated_delta_net", "gated-delta", "delta_rule"),
    notes="gated delta rule: outer/matmul reference vs einsum sim")

register_oracle(
    "MLA", _mla_reference, _mla_sim, _mla_inputs,
    aliases=("mla", "multi_head_latent_attention", "multihead_latent_attention",
             "latent_attention", "mla_decode", "mla_attention", "deepseek_mla"),
    notes="multi-head latent attention decode: reconstruct-KV reference vs "
          "absorbed-projection sim (attend in latent space; independent algorithms)")

register_oracle(
    "Mamba2", _mamba2_reference, _mamba2_sim, _mamba2_inputs,
    aliases=("mamba2", "mamba_2", "ssd", "ssm", "mamba2_ssd", "state_space_dual",
             "selective_scan"),
    notes="Mamba2/SSD: sequential state-recurrence reference vs materialized "
          "1-semiseparable (quadratic-dual) sim (state-space duality)")

_register_rope()
_register_attn_sink()
_register_attn_hd256()
_register_flash()
