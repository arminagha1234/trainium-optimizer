"""
TRUSTED GRADER — independent verification of a winning config before it is
published (borrowed from the NeurIPS-Trainium-Competition scorer, which
recomputes the metric with organizer-owned code and NEVER trusts the
participant's self-reported number).

The search promotes a config on a SINGLE probe measurement, which can be a noise
fluke (or, once Stage 4 generates kernels, a reward-hack). Before a winner goes
on the leaderboard, this re-runs it independently and requires it to:
  1. REPRODUCE its metric within tolerance (kills noise flukes),
  2. re-pass the top-1-token equivalence check vs the Stage-0 baseline, and
  3. (optional) clear a TASK-LEVEL correctness check.

A winner that fails any is marked `unverified` — it is still recorded, but the
leaderboard can flag it, and `publish(require_verified=True)` refuses to make it
the canonical recipe. This never trusts the number the search reported; it
measures again.

TASK-LEVEL GATE (the `task_eval` seam). Reproduction + top-1-token equivalence
catch noise and gross divergence, but top-1 argmax is a coarse correctness proxy
— a kernel can preserve the argmax while distorting the distribution (a reward-
hack surface once kernels are generated). A real task/accuracy check (held-out
logprob agreement, a small eval set, perplexity within tolerance) is the honest
gate, but it needs on-device model outputs. So `verify_winner` takes an OPTIONAL
`task_eval` callable: when supplied it is REQUIRED to pass for a `verified`
verdict; when omitted (the default) behavior is byte-identical to before, so this
is a seam the on-device eval plugs into without changing any current caller.

    task_eval(backend, spec, winner) -> {"ok": bool, "score": float,
                                         "metric": str}   # never trusted to not raise
"""

from __future__ import annotations

from typing import Any, Callable

REPRO_TOL = 0.10          # re-measured metric must be within 10% of the claimed one
EQUIV_MIN = 0.75          # top-1 token match floor vs baseline (same as the loop gate)

# A task-level correctness check. Returns a dict with at least {"ok": bool};
# "score"/"metric" are recorded for the audit trail. See the module docstring.
TaskEval = Callable[[Any, Any, Any], dict]


def verify_winner(backend, spec, winner, baseline_tokens, log,
                  task_eval: "TaskEval | None" = None) -> dict:
    """Re-measure the winning config from scratch and verify reproduction +
    equivalence (+ an optional task-level check via ``task_eval``). Never raises
    — verification must not crash a run."""
    try:
        artifact = backend.apply_config(
            backend.build_baseline(spec.model_id), dict(winner.config))
        neff = backend.compile(artifact)
        m = backend.measure(neff, spec.probe_shape, spec.probe_batch)

        claimed = float(getattr(winner, "metric", 0.0) or 0.0)
        remeasured = float(m.metric)
        drift = abs(remeasured - claimed) / claimed if claimed > 0 else 1.0
        reproduced = (remeasured > 0.0) and (drift <= REPRO_TOL)
        # WHY the re-measurement produced nothing. `measure` already knows -- it puts
        # the classified cause in `failure_reason` -- and this function used to throw
        # it away, logging a bare "remeasured=0" that is indistinguishable between a
        # genuinely slow model, a wedged box, a missing cache file and a compiler
        # crash. Two full trn2.48xlarge runs ended in guesswork because of it: the
        # 35B completed every stage, reported 368 tok/s of box throughput, and then
        # failed grading with `remeasured=0` seventeen seconds after Stage 6 -- far
        # too fast to have measured a 72 GB model at all, which the reason would have
        # said outright.
        why = str(getattr(m, "failure_reason", "") or "").strip()
        if remeasured <= 0.0 and not why:
            why = ("re-measurement returned 0 with no failure_reason -- the backend "
                   "reported neither a metric nor a cause")

        eq_ok = True
        toks = list(getattr(m, "top1_tokens", []) or [])
        if baseline_tokens and toks:
            n = min(len(baseline_tokens), len(toks))
            match = sum(1 for i in range(n) if baseline_tokens[i] == toks[i]) / n
            eq_ok = match >= EQUIV_MIN
        elif baseline_tokens and not toks:
            eq_ok = False  # winner produced no tokens on re-run

        # Optional task-level correctness gate. Absent -> task_ok True (verdict
        # unchanged from before). Present -> REQUIRED to pass for "verified".
        # Never trusted to not raise; a raising eval fails closed (task_ok False).
        task_ok, task_score, task_metric = True, 0.0, ""
        if task_eval is not None:
            try:
                tr = task_eval(backend, spec, winner) or {}
                task_ok = bool(tr.get("ok", False))
                task_score = float(tr.get("score", 0.0) or 0.0)
                task_metric = str(tr.get("metric", ""))
            except Exception as te:  # noqa: BLE001 — a bad eval fails closed, never crashes
                task_ok = False
                task_metric = f"task_eval error: {te!r}"

        verdict = "verified" if (reproduced and eq_ok and task_ok) else "unverified"
        name = spec.model_id.split("/")[-1]
        task_note = "" if task_eval is None else (
            f" task={'ok' if task_ok else 'FAIL'}"
            + (f"({task_metric}={task_score:.4f})" if task_metric else ""))
        log(f"[{name}] trusted grader: {verdict} "
            f"(claimed={claimed:,.0f} remeasured={remeasured:,.0f} "
            f"drift={drift:.1%} equivalence={'ok' if eq_ok else 'FAIL'}{task_note})"
            + (f" -- re-measure failed: {why}" if remeasured <= 0.0 and why else ""))
        return {
            "verdict": verdict,
            "reproduced": reproduced,
            "remeasured_tok_s": remeasured,
            "drift_pct": drift * 100.0,
            "equivalence_ok": eq_ok,
            # Carried so an `unverified` row explains ITSELF. A verdict without a
            # cause is a dead end for whoever reads it next.
            "remeasure_failure": why if remeasured <= 0.0 else "",
            "task_ok": task_ok,
            "task_score": task_score,
            "task_metric": task_metric,
        }
    except Exception as e:  # noqa: BLE001 — verification must never end a run
        log(f"trusted grader failed (non-fatal): {e!r}")
        return {"verdict": "ungraded", "reproduced": False,
                "remeasured_tok_s": 0.0, "drift_pct": 0.0, "equivalence_ok": False,
                "task_ok": False, "task_score": 0.0, "task_metric": "",
                "remeasure_failure": f"grader raised: {e!r}"}
