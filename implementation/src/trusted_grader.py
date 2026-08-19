"""
TRUSTED GRADER — independent verification of a winning config before it is
published (borrowed from the NeurIPS-Trainium-Competition scorer, which
recomputes the metric with organizer-owned code and NEVER trusts the
participant's self-reported number).

The search promotes a config on a SINGLE probe measurement, which can be a noise
fluke (or, once Stage 4 generates kernels, a reward-hack). Before a winner goes
on the leaderboard, this re-runs it independently and requires it to:
  1. REPRODUCE its metric within tolerance (kills noise flukes), and
  2. re-pass the top-1-token equivalence check vs the Stage-0 baseline.

A winner that fails either is marked `unverified` — it is still recorded, but the
leaderboard can flag it. This never trusts the number the search reported; it
measures again.
"""

from __future__ import annotations

REPRO_TOL = 0.10          # re-measured metric must be within 10% of the claimed one
EQUIV_MIN = 0.75          # top-1 token match floor vs baseline (same as the loop gate)


def verify_winner(backend, spec, winner, baseline_tokens, log) -> dict:
    """Re-measure the winning config from scratch and verify reproduction +
    equivalence. Never raises — verification must not crash a run."""
    try:
        artifact = backend.apply_config(
            backend.build_baseline(spec.model_id), dict(winner.config))
        neff = backend.compile(artifact)
        m = backend.measure(neff, spec.probe_shape, spec.probe_batch)

        claimed = float(getattr(winner, "metric", 0.0) or 0.0)
        remeasured = float(m.metric)
        drift = abs(remeasured - claimed) / claimed if claimed > 0 else 1.0
        reproduced = (remeasured > 0.0) and (drift <= REPRO_TOL)

        eq_ok = True
        toks = list(getattr(m, "top1_tokens", []) or [])
        if baseline_tokens and toks:
            n = min(len(baseline_tokens), len(toks))
            match = sum(1 for i in range(n) if baseline_tokens[i] == toks[i]) / n
            eq_ok = match >= EQUIV_MIN
        elif baseline_tokens and not toks:
            eq_ok = False  # winner produced no tokens on re-run

        verdict = "verified" if (reproduced and eq_ok) else "unverified"
        name = spec.model_id.split("/")[-1]
        log(f"[{name}] trusted grader: {verdict} "
            f"(claimed={claimed:,.0f} remeasured={remeasured:,.0f} "
            f"drift={drift:.1%} equivalence={'ok' if eq_ok else 'FAIL'})")
        return {
            "verdict": verdict,
            "reproduced": reproduced,
            "remeasured_tok_s": remeasured,
            "drift_pct": drift * 100.0,
            "equivalence_ok": eq_ok,
        }
    except Exception as e:  # noqa: BLE001 — verification must never end a run
        log(f"trusted grader failed (non-fatal): {e}")
        return {"verdict": "ungraded", "reproduced": False,
                "remeasured_tok_s": 0.0, "drift_pct": 0.0, "equivalence_ok": False}
