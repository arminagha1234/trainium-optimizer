# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for task_eval — the logprob/KL task-level correctness gate. Pure CPU:
synthetic top-k logprob distributions + a fake backend. No device, no model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from task_eval import agreement, make_task_eval, _kl_over_baseline_topk


def _pos(ids, probs):
    """A per-position top-k record from (token ids, PROBABILITIES)."""
    return {"ids": list(ids), "logprobs": [math.log(p) for p in probs]}


# --- KL primitive ------------------------------------------------------------

def test_identical_distributions_zero_kl():
    p = _pos([1, 2, 3], [0.7, 0.2, 0.1])
    kl = _kl_over_baseline_topk(p["ids"], p["logprobs"], p["ids"], p["logprobs"])
    assert kl < 1e-9


def test_divergent_distribution_positive_kl():
    b = _pos([1, 2, 3], [0.7, 0.2, 0.1])
    c = _pos([1, 2, 3], [0.34, 0.33, 0.33])   # flattened — same argmax, diff dist
    kl = _kl_over_baseline_topk(b["ids"], b["logprobs"], c["ids"], c["logprobs"])
    assert kl > 0.1


def test_dropped_baseline_token_penalized():
    # candidate dropped baseline's #1 token from its top-k -> large divergence
    b = _pos([1, 2, 3], [0.7, 0.2, 0.1])
    c = _pos([2, 3, 4], [0.5, 0.3, 0.2])       # token 1 absent from candidate
    kl = _kl_over_baseline_topk(b["ids"], b["logprobs"], c["ids"], c["logprobs"])
    assert kl > 1.0


# --- agreement over positions ------------------------------------------------

def test_agreement_identical():
    seq = [_pos([1, 2], [0.9, 0.1]), _pos([5, 6], [0.8, 0.2])]
    a = agreement(seq, seq)
    assert a["mean_kl"] < 1e-9 and a["top1_match"] == 1.0 and a["n"] == 2


def test_agreement_top1_preserved_but_distribution_shifted():
    base = [_pos([1, 2], [0.9, 0.1]), _pos([5, 6], [0.9, 0.1])]
    cand = [_pos([1, 2], [0.55, 0.45]), _pos([5, 6], [0.55, 0.45])]
    a = agreement(base, cand)
    assert a["top1_match"] == 1.0        # argmax unchanged...
    assert a["mean_kl"] > 0.1            # ...but the distribution diverged


def test_agreement_no_positions_is_infinite():
    a = agreement([], [])
    assert a["n"] == 0 and a["mean_kl"] == float("inf")


# --- factory / backend wiring ------------------------------------------------

@dataclass
class _Spec:
    model_id: str = "acme/tiny"
    probe_shape: Any = 1024
    probe_batch: int = 1


@dataclass
class _Winner:
    config: dict = field(default_factory=lambda: {"tp_degree": 8})


@dataclass
class _Meas:
    top_logprobs: list = field(default_factory=list)


class _FakeBackend:
    """Returns baseline vs candidate top_logprobs the test controls."""
    def __init__(self, base_lp, cand_lp):
        self._base, self._cand, self._n = base_lp, cand_lp, 0

    def build_baseline(self, mid):
        return {"m": mid}

    def apply_config(self, art, cfg):
        art = dict(art); art["cfg"] = cfg
        return art

    def compile(self, art):
        return art

    def measure(self, art, shape, batch):
        # first measure() call is baseline, second is candidate
        lp = self._cand if art.get("cfg") else self._base
        return _Meas(top_logprobs=lp)


def test_task_eval_passes_on_matching_model():
    seq = [_pos([1, 2], [0.9, 0.1]), _pos([5, 6], [0.8, 0.2])]
    te = make_task_eval(max_kl=0.05, min_top1=0.9)
    r = te(_FakeBackend(seq, seq), _Spec(), _Winner())
    assert r["ok"] is True and r["score"] < 1e-6 and r["metric"] == "topk_logprob_kl"


def test_task_eval_fails_on_distribution_distortion():
    base = [_pos([1, 2], [0.9, 0.1]), _pos([5, 6], [0.9, 0.1])]
    cand = [_pos([1, 2], [0.55, 0.45]), _pos([5, 6], [0.55, 0.45])]
    te = make_task_eval(max_kl=0.05, min_top1=0.9)
    r = te(_FakeBackend(base, cand), _Spec(), _Winner())
    assert r["ok"] is False and r["score"] > 0.05   # top-1 preserved, KL too high


def test_task_eval_fails_closed_without_logprobs():
    te = make_task_eval()
    r = te(_FakeBackend([], []), _Spec(), _Winner())
    assert r["ok"] is False and "fail-closed" in r["detail"]


def test_task_eval_never_raises_on_broken_backend():
    class _Broken:
        def build_baseline(self, mid): raise RuntimeError("device gone")
    r = make_task_eval()(_Broken(), _Spec(), _Winner())
    assert r["ok"] is False and "error" in r["detail"]


def test_task_eval_composes_with_trusted_grader():
    """The whole point: a distribution-distorting winner that PASSES repro +
    top-1 equivalence is still rejected by verify_winner once task_eval is
    supplied."""
    from trusted_grader import verify_winner

    class _GraderBackend:
        def build_baseline(self, mid): return {"m": mid}
        def apply_config(self, art, cfg): art = dict(art); art["cfg"] = cfg; return art
        def compile(self, art): return art
        def measure(self, art, shape, batch):
            class M:
                metric = 100.0
                top1_tokens = [1, 5]
                # distorted distribution on the candidate; identical top-1
                top_logprobs = ([_pos([1, 2], [0.55, 0.45]), _pos([5, 6], [0.55, 0.45])]
                                if art.get("cfg") else
                                [_pos([1, 2], [0.9, 0.1]), _pos([5, 6], [0.9, 0.1])])
            return M()

    @dataclass
    class _W:
        metric: float = 100.0
        config: dict = field(default_factory=lambda: {"tp_degree": 8})

    te = make_task_eval(max_kl=0.05, min_top1=0.9)
    verdict = verify_winner(_GraderBackend(), _Spec(), _W(), [1, 5],
                            lambda _m: None, task_eval=te)
    assert verdict["reproduced"] and verdict["equivalence_ok"]   # coarse checks pass
    assert verdict["task_ok"] is False                           # task gate rejects
    assert verdict["verdict"] == "unverified"


# --- regression: real-model top-k does NOT sum to 1 (bug caught on-device) ---
def _raw_pos(ids, logprobs):
    """A per-position top-k with RAW logprobs that do NOT sum to 1 (as a real
    model's top-k subset of the full vocab)."""
    return {"ids": list(ids), "logprobs": list(logprobs)}


def test_identical_nonnormalized_topk_is_zero_kl():
    # top-5 logprobs summing to ~0.38 prob mass (realistic) — KL(P||P) must be 0
    p = _raw_pos([1, 2, 3, 4, 5], [-1.0, -2.0, -2.5, -3.0, -3.5])
    kl = _kl_over_baseline_topk(p["ids"], p["logprobs"], p["ids"], p["logprobs"])
    assert kl < 1e-9, kl                       # regression: was ~0.98 before the fix
    a = agreement([p], [p])
    assert a["mean_kl"] < 1e-9 and a["top1_match"] == 1.0


def test_nonnormalized_flattened_is_divergent():
    base = _raw_pos([1, 2, 3, 4, 5], [-1.0, -2.0, -2.5, -3.0, -3.5])
    flat = _raw_pos([1, 2, 3, 4, 5], [-1.5, -1.6, -1.7, -1.8, -1.9])  # same top-1, flat
    kl = _kl_over_baseline_topk(base["ids"], base["logprobs"], flat["ids"], flat["logprobs"])
    assert kl > 0.05
