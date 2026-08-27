# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trusted_grader.verify_winner, including the optional task-level
correctness gate (task_eval seam). Pure CPU — a tiny fake backend stands in for
the on-device re-measure; no NeuronCore needed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trusted_grader import verify_winner


# --- fakes --------------------------------------------------------------------

@dataclass
class _Spec:
    model_id: str = "acme/tiny-1b"
    probe_shape: Any = 1024
    probe_batch: int = 1


@dataclass
class _Winner:
    metric: float = 100.0
    config: dict = field(default_factory=lambda: {"tp_degree": 8})


@dataclass
class _Meas:
    metric: float
    top1_tokens: list = field(default_factory=list)


class _FakeBackend:
    """Re-measures to a controllable value + token stream. Mirrors the Backend
    surface verify_winner touches (build_baseline/apply_config/compile/measure)."""

    def __init__(self, remeasured=100.0, tokens=None):
        self._m = remeasured
        self._toks = tokens if tokens is not None else [1, 2, 3, 4]

    def build_baseline(self, model_id):
        return {"model": model_id}

    def apply_config(self, baseline, config):
        return {"artifact": baseline, "config": config}

    def compile(self, artifact):
        return {"neff": artifact}

    def measure(self, neff, shape, batch):
        return _Meas(metric=self._m, top1_tokens=list(self._toks))


def _log(_msg):  # silent sink
    pass


BASE_TOKENS = [1, 2, 3, 4]


# --- baseline behavior (no task_eval) ----------------------------------------

def test_verified_when_reproduces_and_equivalent():
    r = verify_winner(_FakeBackend(remeasured=100.0, tokens=BASE_TOKENS),
                      _Spec(), _Winner(metric=100.0), BASE_TOKENS, _log)
    assert r["verdict"] == "verified"
    assert r["reproduced"] and r["equivalence_ok"]
    assert r["task_ok"] is True          # no task_eval -> defaults True


def test_unverified_on_drift():
    # remeasured 130 vs claimed 100 -> 30% drift > REPRO_TOL
    r = verify_winner(_FakeBackend(remeasured=130.0, tokens=BASE_TOKENS),
                      _Spec(), _Winner(metric=100.0), BASE_TOKENS, _log)
    assert r["verdict"] == "unverified"
    assert r["reproduced"] is False


def test_unverified_on_token_divergence():
    r = verify_winner(_FakeBackend(remeasured=100.0, tokens=[9, 9, 9, 9]),
                      _Spec(), _Winner(metric=100.0), BASE_TOKENS, _log)
    assert r["verdict"] == "unverified"
    assert r["equivalence_ok"] is False


# --- the task_eval seam -------------------------------------------------------

def test_task_eval_pass_keeps_verified():
    calls = {"n": 0}

    def task_eval(backend, spec, winner):
        calls["n"] += 1
        return {"ok": True, "score": 0.991, "metric": "logprob_agreement"}

    r = verify_winner(_FakeBackend(tokens=BASE_TOKENS), _Spec(),
                      _Winner(metric=100.0), BASE_TOKENS, _log, task_eval=task_eval)
    assert calls["n"] == 1
    assert r["verdict"] == "verified"
    assert r["task_ok"] is True
    assert r["task_metric"] == "logprob_agreement"
    assert abs(r["task_score"] - 0.991) < 1e-9


def test_task_eval_fail_blocks_verified_even_when_repro_and_equiv_pass():
    """A kernel that reproduces the metric AND preserves top-1 tokens but
    distorts the distribution (task_eval fails) must NOT be verified — the seam's
    whole point."""
    def task_eval(backend, spec, winner):
        return {"ok": False, "score": 0.42, "metric": "logprob_agreement"}

    r = verify_winner(_FakeBackend(remeasured=100.0, tokens=BASE_TOKENS),
                      _Spec(), _Winner(metric=100.0), BASE_TOKENS, _log,
                      task_eval=task_eval)
    assert r["reproduced"] and r["equivalence_ok"]   # the coarse gates pass...
    assert r["task_ok"] is False                     # ...but the task gate fails
    assert r["verdict"] == "unverified"


def test_task_eval_raising_fails_closed():
    def task_eval(backend, spec, winner):
        raise RuntimeError("eval harness exploded")

    r = verify_winner(_FakeBackend(tokens=BASE_TOKENS), _Spec(),
                      _Winner(metric=100.0), BASE_TOKENS, _log, task_eval=task_eval)
    assert r["task_ok"] is False
    assert r["verdict"] == "unverified"
    assert "task_eval error" in r["task_metric"]


def test_backend_crash_is_ungraded_with_task_fields():
    class _BrokenBackend:
        def build_baseline(self, mid):
            raise RuntimeError("device offline")

    r = verify_winner(_BrokenBackend(), _Spec(), _Winner(), BASE_TOKENS, _log,
                      task_eval=lambda *a: {"ok": True})
    assert r["verdict"] == "ungraded"
    assert r["task_ok"] is False        # ungraded path still carries the field
