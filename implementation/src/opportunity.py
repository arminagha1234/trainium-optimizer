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
_COMPILER_WEAK_FAMILIES = frozenset({"attention", "scan"})
# A default analytic opportunity score per family (0..1, higher = more worth
# authoring). Attention/scan high; standard families low. Tunable.
_FAMILY_OPPORTUNITY = {
    "attention": 0.90, "scan": 0.90, "moe_router": 0.55,
    "softmax": 0.35, "reduction": 0.30, "normalization": 0.25,
    "matmul": 0.15, "elementwise": 0.15,
}
_DEFAULT_ANALYTIC = 0.20  # unknown family -> mild, low-priority opportunity


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


def analytic_opportunity(spec: Any) -> OpTarget:
    """Device-FREE opportunity verdict from the op family alone — no compile.
    Used to pre-rank a model's ops before spending any device time."""
    fam = _op_family(spec)
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
