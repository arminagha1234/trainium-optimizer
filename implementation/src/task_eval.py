# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""task_eval.py — a real TASK-LEVEL correctness gate for the trusted grader.

The seam (``trusted_grader.verify_winner(..., task_eval=...)``, added earlier)
takes any ``task_eval(backend, spec, winner) -> {"ok","score","metric"}``. This
module supplies the production one: **logprob / KL agreement** between the
optimized model and its baseline, which is strictly stronger than the grader's
existing top-1-argmax equivalence.

Why stronger: top-1 argmax only checks the WINNING token per position. A kernel
(or config) can preserve every argmax while distorting the underlying
distribution — flattening it, shifting mass between near-ties — which changes
sampling, beam search, and downstream logprob-based decoding. That is precisely
the reward-hack surface once Stage-4 generates kernels. Comparing the top-k
DISTRIBUTIONS (KL divergence) catches it; comparing only the argmax does not.

The score: per position, KL(P_base || P_cand) computed over the BASELINE's top-k
tokens (the candidate's logprob for a baseline token is read from its own top-k,
or floored when the candidate dropped that token from its top-k — a dropped
high-probability baseline token is exactly the divergence we want to penalize).
Averaged over all positions across a held-out prompt set. The gate passes iff
mean KL <= ``max_kl`` AND top-1 agreement >= ``min_top1`` (belt and suspenders:
the coarse check must also hold).

The scoring is PURE (numpy) and unit-testable off-device. The only device work
is obtaining the top-k logprobs, which the backend returns on
``Measurements.top_logprobs`` (per-position ``{"ids","logprobs"}``); a backend
that does not populate it yields no positions -> the gate FAILS CLOSED (a
task-eval that cannot measure agreement must not certify a winner).
"""

from __future__ import annotations

import math
from typing import Any, Callable

# Floor logprob assigned to a baseline top-k token that is ABSENT from the
# candidate's top-k. Very low (≈ prob 1e-9) so "the candidate dropped a token the
# baseline ranked highly" registers as large divergence, not a free pass.
_ABSENT_LOGPROB = -20.7  # ln(1e-9)


def _kl_over_baseline_topk(base_ids, base_lp, cand_ids, cand_lp) -> float:
    """KL(P_base || P_cand) over the baseline's top-k support (nats).

    Both sides are renormalized over the SAME support (the baseline's top-k
    tokens) so the divergence is a proper KL — and KL(P||P) == 0 exactly, even
    though a real model's top-k logprobs do NOT sum to 1 (they're a subset of the
    full vocab). The candidate's raw prob for each baseline token comes from its
    own top-k, or an absent-floor when the candidate dropped that token (which
    then shows up as real divergence after renormalization). Returns 0.0 for
    empty/degenerate input (missing data is handled fail-closed by the caller)."""
    if not base_ids or not base_lp:
        return 0.0
    cand = {int(i): float(lp) for i, lp in zip(cand_ids or [], cand_lp or [])}
    # Raw probs for the baseline top-k tokens on BOTH sides (cand floored if the
    # token is absent from its top-k), then renormalize each over this support.
    bp_raw = [math.exp(float(x)) for x in base_lp]
    cp_raw = [math.exp(cand.get(int(tok), _ABSENT_LOGPROB)) for tok in base_ids]
    zb, zc = sum(bp_raw), sum(cp_raw)
    if zb <= 0 or zc <= 0:
        return 0.0
    kl = 0.0
    for b, c in zip(bp_raw, cp_raw):
        pb = b / zb
        pc = c / zc
        if pb > 0 and pc > 0:
            kl += pb * math.log(pb / pc)
    return max(0.0, kl)


def agreement(base_positions: list, cand_positions: list) -> dict:
    """Score distribution agreement between two aligned lists of per-position
    top-k records (each ``{"ids":[...], "logprobs":[...]}``).

    Returns ``{"mean_kl", "top1_match", "n"}``: mean per-position KL over the
    positions present in BOTH, and the fraction of those positions whose top-1
    token matches. ``n`` is the compared-position count (0 -> no signal)."""
    n = min(len(base_positions), len(cand_positions))
    if n == 0:
        return {"mean_kl": float("inf"), "top1_match": 0.0, "n": 0}
    kls, matches = [], 0
    for i in range(n):
        b, c = base_positions[i] or {}, cand_positions[i] or {}
        b_ids, b_lp = b.get("ids", []), b.get("logprobs", [])
        c_ids, c_lp = c.get("ids", []), c.get("logprobs", [])
        kls.append(_kl_over_baseline_topk(b_ids, b_lp, c_ids, c_lp))
        if b_ids and c_ids and int(b_ids[0]) == int(c_ids[0]):
            matches += 1
    return {"mean_kl": sum(kls) / len(kls), "top1_match": matches / n, "n": n}


def make_task_eval(*, max_kl: float = 0.10, min_top1: float = 0.90,
                   log: Callable[[str], None] = lambda _m: None) -> Callable:
    """Build a ``task_eval(backend, spec, winner)`` for
    ``trusted_grader.verify_winner``.

    It re-runs the BASELINE and the WINNING config through the backend (same
    build/apply/compile/measure calls the grader uses), reads each side's
    ``Measurements.top_logprobs``, and scores logprob/KL agreement. Passes iff
    ``mean_kl <= max_kl`` AND ``top1_match >= min_top1``. FAILS CLOSED (ok=False)
    if either side yields no top_logprobs (cannot certify what it cannot
    measure) — never raises (the grifter wraps it, but we also guard here)."""

    def task_eval(backend: Any, spec: Any, winner: Any) -> dict:
        try:
            base_art = backend.compile(backend.build_baseline(spec.model_id))
            base_m = backend.measure(base_art, spec.probe_shape, spec.probe_batch)
            cand_art = backend.compile(
                backend.apply_config(backend.build_baseline(spec.model_id),
                                     dict(winner.config)))
            cand_m = backend.measure(cand_art, spec.probe_shape, spec.probe_batch)
            base_lp = list(getattr(base_m, "top_logprobs", []) or [])
            cand_lp = list(getattr(cand_m, "top_logprobs", []) or [])
            if not base_lp or not cand_lp:
                return {"ok": False, "score": float("inf"),
                        "metric": "topk_logprob_kl",
                        "detail": "no top_logprobs from backend (fail-closed)"}
            a = agreement(base_lp, cand_lp)
            ok = a["mean_kl"] <= max_kl and a["top1_match"] >= min_top1
            log(f"task_eval: mean_kl={a['mean_kl']:.4f} (<= {max_kl}) "
                f"top1={a['top1_match']:.3f} (>= {min_top1}) n={a['n']} -> "
                f"{'PASS' if ok else 'FAIL'}")
            return {"ok": bool(ok), "score": float(a["mean_kl"]),
                    "metric": "topk_logprob_kl",
                    "detail": f"top1_match={a['top1_match']:.3f} n={a['n']}"}
        except Exception as e:  # noqa: BLE001 — a broken eval fails closed, never raises
            return {"ok": False, "score": float("inf"),
                    "metric": "topk_logprob_kl", "detail": f"task_eval error: {e!r}"}

    return task_eval
