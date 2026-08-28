# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""opportunity.py — autonomous TARGET SELECTION: decide WHICH ops of a model are
worth authoring a kernel for, so a human no longer has to.

This is the last piece of human judgment in the loop. The framework can author,
mutate, gate correctness, %SOL-profile, and compound — but a human still picks
which op/model to optimize. The on-device finding that motivates this: a
hand-written NKI kernel LOSES to the compiler on standard ops (elementwise, norm,
GEMM — the compiler is already ~80% of speed-of-light there) and only WINS in the
compiler-WEAK regime (long-context attention, linear-attention/scan, sparse,
IO-bound). So the right targets are exactly the ops far from SOL.

Two signals, best-first:
  1. MEASURED %SOL (authoritative) — when a device-timed race is available, use
     ``roofline.classify``: only an op at "opportunity"/"marginal"/"unknown" is
     worth authoring; a "near_sol" op is skipped (the compiler already wins).
  2. ANALYTIC op-family heuristic (device-free fallback) — rank by op family:
     the compiler-weak families (attention, scan) are high-opportunity; standard
     families (elementwise, norm, matmul, softmax, reduction) are low. This lets
     the sweep pre-rank a model's ops WITHOUT paying a compile for each one, then
     the loop measures the top candidates for the authoritative %SOL call.

Pure-python; composes with roofline.py (measured) and nki_knowledge.classify_op
(analytic). No device dependency — the ``sol_fn`` seam supplies measured %SOL
when on-box; off-box the analytic ranking stands alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Op families where a hand/LLM-authored NKI kernel can beat the compiler (it is
# weak / cannot lower / OOMs) — from nki_knowledge.classify_op labels. Everything
# else the compiler already does at/near speed-of-light (measured on-device).
#
# ATTENTION IS DELIBERATELY NOT HERE (2026-08-28 calibration). On-device
# measurement (single trn2 core, hd=128, bf16, fresh-input steady-state) found the
# COMPILER'S dense attention BEATS/ties the harvested flash kernel across S=8k-32k
# and does not OOM through 65k — single-head attention is compiler-STRONG, not
# weak. So attention is ranked by the ACTUAL score-matrix pressure (see
# ``_attention_opportunity``) and, authoritatively, by MEASURED %SOL — never a
# blanket "attention → author it" assumption. ``scan`` (linear-attention / SSM /
# GatedDeltaNet) STAYS: its sequential recurrence genuinely can't be parallelized
# by the compiler (validated separately).
_COMPILER_WEAK_FAMILIES = frozenset({"scan"})
# A default analytic opportunity score per family (0..1, higher = more worth
# authoring). scan high; standard families low; attention handled specially by
# ``_attention_opportunity`` (shape-aware), so its entry here is only the
# unknown-shape fallback.
_FAMILY_OPPORTUNITY = {
    "attention": 0.40, "scan": 0.90, "moe_router": 0.55,
    "softmax": 0.35, "reduction": 0.30, "normalization": 0.25,
    "matmul": 0.15, "elementwise": 0.15,
}
_DEFAULT_ANALYTIC = 0.20  # unknown family -> mild, low-priority opportunity

# Attention score-matrix pressure thresholds, in TOTAL score elements (B*H*S*S) —
# the quantity that decides whether the compiler's dense [S,S] materialization
# stays cheap (it WINS) or grows costly (flash's regime). Calibrated to the
# on-device crossover: single-head S=8k (6.7e7 elems) -> compiler 1.45x faster;
# S=32k (1.07e9) -> ~tie. So below ~2.5e8 the compiler clearly wins (low
# opportunity), above ~1e9 the score work is large enough that a streaming flash
# kernel has real headroom (higher opportunity), with a marginal band between.
_ATTN_ELEMS_LOW = 2.5e8
_ATTN_ELEMS_HIGH = 1.0e9


@dataclass(frozen=True)
class OpTarget:
    """One op's authoring-worthiness verdict."""
    op: str
    score: float                 # 0..1, higher = more worth authoring
    worth_authoring: bool
    source: str                  # "measured" | "analytic"
    reason: str


def _op_family(spec: Any) -> str:
    """The op-family label for a spec, via nki_knowledge.classify_op (falls back
    to a bare name if that module is unavailable)."""
    try:
        from nki_knowledge import classify_op
        return classify_op(getattr(spec, "name", "") or "",
                           getattr(spec, "family", None),
                           getattr(spec, "notes", None))
    except Exception:  # noqa: BLE001
        return ""


import re as _re

# Tokens the attention specs encode in shape_class: b<batch> h<heads> s<seq>
# hd<head_dim> (e.g. "mha-b8-h32-s2048-hd128", "flash-s2048-hd128"). Parsed to get
# the REAL shape WITHOUT allocating inputs (a batched spec's real_inputs is
# multi-GB — reading its shape by materializing it is a non-starter).
_SHAPE_TOKEN = {"b": _re.compile(r"(?:^|[^a-z])b(\d+)"),
                "h": _re.compile(r"(?:^|[^a-z])h(\d+)"),
                "s": _re.compile(r"(?:^|[^a-z])s(\d+)"),
                "hd": _re.compile(r"hd(\d+)")}


def _attention_score_elems(spec: Any) -> float:
    """Estimate an attention op's TOTAL score-matrix elements (B*H*S*S) — the
    quantity the compiler must materialize for dense attention (small => compiler
    wins; large => flash's regime). Returns 0.0 when it can't infer (caller uses
    the moderate default). Never raises. Never allocates.

    Prefers parsing ``shape_class`` (the specs encode ``b/h/s/hd`` there), so a
    batched spec whose real inputs are multi-GB is read cheaply. Falls back to the
    (small) ``offline_inputs`` array shapes when the shape_class carries no tokens:
    S is the largest dim, d the smallest dim >= 8, and B*H = total_q / (S*d)."""
    sc = (getattr(spec, "shape_class", "") or "").lower()
    m_s = _SHAPE_TOKEN["s"].search(sc)
    m_hd = _SHAPE_TOKEN["hd"].search(sc)
    if m_s and m_hd:
        S = float(m_s.group(1))
        mb, mh = _SHAPE_TOKEN["b"].search(sc), _SHAPE_TOKEN["h"].search(sc)
        B = float(mb.group(1)) if mb else 1.0
        H = float(mh.group(1)) if mh else 1.0
        return B * H * S * S
    # Fallback: infer from the SMALL offline inputs (never real_inputs — may be
    # multi-GB). This is a lower-bound proxy but avoids a huge allocation.
    try:
        import numpy as np  # noqa: PLC0415
        inp = spec.offline_inputs()
        if not isinstance(inp, dict) or not inp:
            return 0.0
        named = {k: np.asarray(v) for k, v in inp.items()}
        qkv = [a for k, a in named.items()
               if k.lower() in ("q", "k", "v", "query", "key", "value")] \
            or list(named.values())
        qkv = [a for a in qkv if getattr(a, "ndim", 0) >= 2]
        if not qkv:
            return 0.0
        dims = [d for a in qkv for d in a.shape]
        S = float(max(dims))
        d = float(min(x for x in dims if x >= 8) if any(x >= 8 for x in dims)
                  else min(dims))
        n_mats = max(1.0, round(float(qkv[0].size) / (S * d)))
        return n_mats * S * S
    except Exception:  # noqa: BLE001 — inference is best-effort
        return 0.0


def _attention_opportunity(spec: Any) -> OpTarget:
    """Shape-aware attention verdict (2026-08-28 calibration). Ranks by the
    materialized score-matrix pressure instead of a blanket assumption: small
    scores -> compiler wins (low opportunity, measured); large scores (long S,
    batched-multi-head) -> a streaming flash kernel has headroom. MEASURED %SOL
    still overrides this in ``rank_targets`` when a device race is available."""
    elems = _attention_score_elems(spec)
    op = getattr(spec, "name", "?")
    if elems <= 0.0:
        # Couldn't infer shape — moderate "measure it" prior, not worth-authoring
        # on the analytic signal alone.
        return OpTarget(op, _FAMILY_OPPORTUNITY["attention"], False, "analytic",
                        "attention (shape unknown) — measure %SOL before authoring; "
                        "single-head dense attention is compiler-strong on trn2")
    if elems < _ATTN_ELEMS_LOW:
        return OpTarget(op, 0.15, False, "analytic",
                        f"attention scores ~{elems:.2e} elems (< {_ATTN_ELEMS_LOW:.0e}) "
                        "— compiler's dense attention wins here (measured); skip")
    if elems >= _ATTN_ELEMS_HIGH:
        return OpTarget(op, 0.70, True, "analytic",
                        f"attention scores ~{elems:.2e} elems (>= {_ATTN_ELEMS_HIGH:.0e}) "
                        "— large score matrix (long-context / batched-multi-head); a "
                        "streaming flash kernel has real headroom")
    return OpTarget(op, 0.40, False, "analytic",
                    f"attention scores ~{elems:.2e} elems — marginal; measure %SOL")


def analytic_opportunity(spec: Any) -> OpTarget:
    """Device-FREE opportunity verdict from the op family — no compile. Used to
    pre-rank a model's ops before spending any device time. Attention is
    shape-aware (``_attention_opportunity``); other families use the flat
    per-family prior."""
    fam = _op_family(spec)
    if fam == "attention":
        return _attention_opportunity(spec)
    score = _FAMILY_OPPORTUNITY.get(fam, _DEFAULT_ANALYTIC)
    weak = fam in _COMPILER_WEAK_FAMILIES
    return OpTarget(
        op=getattr(spec, "name", "?"), score=score,
        worth_authoring=weak or score >= 0.5,
        source="analytic",
        reason=(f"op-family '{fam}' is compiler-weak — a kernel can win"
                if weak else
                f"op-family '{fam}' — compiler is usually near-SOL; low priority"))


def measured_opportunity(spec: Any, sol: float, bottleneck: str = "") -> OpTarget:
    """Authoritative verdict from a DEVICE-TIMED %SOL (via roofline.classify).
    Only a positive near-SOL reading skips an op (fail-open on unknown)."""
    try:
        import roofline
        prof = roofline.classify(sol, bottleneck or "memory_bound",
                                 measured=(sol > 0.0))
        verdict = prof.verdict
        worth = prof.worth_authoring
    except Exception:  # noqa: BLE001 — fall back to a simple threshold
        worth = not (sol >= 0.80)
        verdict = "near_sol" if sol >= 0.80 else "opportunity"
    # score: far-from-SOL == high opportunity (1 - sol), clamped.
    score = max(0.0, min(1.0, 1.0 - sol)) if sol > 0 else 0.5
    return OpTarget(
        op=getattr(spec, "name", "?"), score=score, worth_authoring=worth,
        source="measured",
        reason=f"%SOL={sol*100:.0f}% -> {verdict}")


def rank_targets(specs: list, sol_fn: Callable[[Any], tuple] | None = None
                 ) -> list[OpTarget]:
    """Rank a model's ops by authoring-worthiness, most-worth first.

    ``sol_fn(spec) -> (sol, bottleneck)`` supplies a DEVICE-TIMED %SOL for an op
    (authoritative → ``measured_opportunity``); if it is None or raises/returns a
    non-positive sol for an op, that op falls back to the device-free
    ``analytic_opportunity``. So the sweep works fully off-device (analytic only)
    and upgrades seamlessly on-box. Sorted by score desc; worth-authoring ops
    first within ties."""
    out: list[OpTarget] = []
    for spec in specs:
        target = None
        if sol_fn is not None:
            try:
                sol, bn = sol_fn(spec)
                if sol and sol > 0.0:
                    target = measured_opportunity(spec, float(sol), str(bn or ""))
            except Exception:  # noqa: BLE001 — a broken measure falls back to analytic
                target = None
        if target is None:
            target = analytic_opportunity(spec)
        out.append(target)
    out.sort(key=lambda t: (t.worth_authoring, t.score), reverse=True)
    return out


def select_targets(specs: list, sol_fn: Callable[[Any], tuple] | None = None,
                   max_targets: int | None = None) -> list[OpTarget]:
    """The ops the framework should author kernels for: the worth-authoring ops
    (compiler-weak / far-from-SOL), highest-opportunity first, optionally capped
    at ``max_targets`` (single-chip budget). Near-SOL ops are dropped — the
    compiler already wins there, so authoring is wasted. NEVER returns a near-SOL
    op; returns [] if nothing is worth authoring (an honest 'compiler already
    wins everything here')."""
    ranked = [t for t in rank_targets(specs, sol_fn) if t.worth_authoring]
    return ranked[:max_targets] if max_targets is not None else ranked
