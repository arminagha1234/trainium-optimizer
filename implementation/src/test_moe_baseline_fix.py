"""
Tests for the MoE *baseline* unblock (moe-baseline-fix):

  1. The dtype-safe router top-k shim (backends/moe_router_patch): integer
     inputs to topk/sort/argsort are routed through float32 (so they never reach
     the int64-rejecting AwsNeuronTopK), while float inputs pass through
     untouched and the returned indices/permutation are unchanged. This is the
     surgical fix for the captured OLMoE crash
     ("TopK ... does not support 32/64-bit integer types", moe.py:393).

  2. install_neuron_safe_moe_topk degrades gracefully when torch is absent
     (import-safe / CPU-mockable — this box has no torch).

  3. The harness-honesty gate: a Stage-0 baseline that produces NO throughput
     (worker crash / 0 tok/s) now raises NoBaselineError and records a
     FAIL_NO_BASELINE row — it is NEVER kept as a benign "ok, 0.000" incumbent.
     A positive baseline is still kept (regression guard).

All CPU-only; no torch, no Neuron, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backends.base import Measurements
from backends.mock import MockBackend
from bank import KnowledgeBank
from guardrails import Guardrails
from ledger import Ledger, Stage, Status
from orchestrator import (
    ModelSpec,
    NoBaselineError,
    Orchestrator,
    always_equivalent,
)

import backends.moe_router_patch as mrp


# --------------------------------------------------------------------------- #
# 1. dtype-safe shim logic (exercised with a tiny fake torch — no real torch)  #
# --------------------------------------------------------------------------- #

class _FakeTensor:
    def __init__(self, data, dtype):
        self.data = list(data)
        self.dtype = dtype

    def to(self, dtype):
        return _FakeTensor(self.data, dtype)


class _Ret(tuple):
    """Mimics torch.return_types.{sort,topk}: a (values, indices) structseq that
    is constructible from an iterable (as torch's are)."""
    def __new__(cls, seq):
        return super().__new__(cls, seq)


class _FakeTorch:
    Tensor = _FakeTensor
    int8 = "int8"; int16 = "int16"; int32 = "int32"; int64 = "int64"
    uint8 = "uint8"; float32 = "float32"


def _make_sort(record):
    def fake_sort(inp, *a, **k):
        record["seen_dtype"] = inp.dtype
        order = sorted(range(len(inp.data)), key=lambda i: inp.data[i])
        values = _FakeTensor([inp.data[i] for i in order], inp.dtype)
        indices = _FakeTensor(order, "int64")
        return _Ret([values, indices])
    return fake_sort


def _make_argsort(record):
    def fake_argsort(inp, *a, **k):
        record["seen_dtype"] = inp.dtype
        order = sorted(range(len(inp.data)), key=lambda i: inp.data[i])
        return _FakeTensor(order, "int64")
    return fake_argsort


def test_shim_casts_integer_input_through_float_and_back():
    """An int64 tensor is sorted via a float32 view (dodging AwsNeuronTopK), and
    the sorted VALUES are cast back to the original int dtype; indices unchanged.
    Result is order-identical to a native int sort (exact for routing-scale ids)."""
    torch = _FakeTorch()
    rec = {}
    wrapped = mrp._wrap_values_and_indices(torch, _make_sort(rec))

    expert_ids = _FakeTensor([3, 1, 2, 0, 1], torch.int64)
    out = wrapped(expert_ids)

    # The underlying sort saw a FLOAT tensor — never an integer one.
    assert rec["seen_dtype"] == "float32"
    values, indices = out[0], out[1]
    # Values are cast BACK to the original integer dtype.
    assert values.dtype == "int64"
    # Order is correct (sorted expert ids) and the permutation is preserved.
    assert values.data == [0, 1, 1, 2, 3]
    assert indices.data == [3, 1, 4, 2, 0]


def test_shim_passthrough_for_float_input():
    """Float inputs (router logits, attention scores, …) are untouched — the
    shim is a pure no-op for them, so it changes nothing outside integer sorts."""
    torch = _FakeTorch()
    rec = {}
    wrapped = mrp._wrap_values_and_indices(torch, _make_sort(rec))

    logits = _FakeTensor([0.3, 0.9, 0.1], torch.float32)
    out = wrapped(logits)

    assert rec["seen_dtype"] == "float32"        # passed straight through
    assert out[0].dtype == "float32"             # no dtype round-trip
    assert out[0].data == [0.1, 0.3, 0.9]


def test_shim_argsort_indices_only():
    """argsort returns only indices; an integer input still routes via float32."""
    torch = _FakeTorch()
    rec = {}
    wrapped = mrp._wrap_indices_only(torch, _make_argsort(rec))

    out = wrapped(_FakeTensor([5, 2, 9], torch.int64))
    assert rec["seen_dtype"] == "float32"
    assert out.data == [1, 0, 2]


def test_install_is_graceful_without_torch():
    """On a box with no torch (this one), install must NOT raise — it reports
    False and leaves the process unpatched. Import-safe by construction."""
    # torch is genuinely absent here, so the real install exercises the fallback.
    assert mrp.install_neuron_safe_moe_topk(log=lambda *_: None) is False


# --------------------------------------------------------------------------- #
# 2. Harness-honesty: a crashed / 0-throughput baseline is FAIL_NO_BASELINE    #
# --------------------------------------------------------------------------- #

_SPEC = ModelSpec(
    model_id="mock/olmoe", family="moe_causal_lm", param_count=7e9,
    parent="olmoe", probe_shape="chat 1k/512", probe_batch=1,
)


class _ZeroThroughputBackend(MockBackend):
    """Simulates a worker that CRASHED: the measurement comes back with metric=0
    (exactly what native_pytorch.measure returns when the neuron_worker dies or
    writes no result — the OLMoE int64-topk crash)."""

    def measure(self, neff, shape, batch):
        return Measurements(metric=0.0, shape=shape, batch=batch)


def _orch(tmp_path: Path, backend) -> Orchestrator:
    orch = Orchestrator(
        backend=backend, bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "run"),
        equivalence=always_equivalent, sdk_version="2.28.0",
    )
    orch.ledger.init()
    return orch


def test_zero_throughput_baseline_raises_and_records_fail(tmp_path: Path):
    """A crashed baseline (metric=0) must raise NoBaselineError and record a
    FAIL_NO_BASELINE row — never a KEEP with metric=0. This is what turns the
    MoE "ok, 0.000" mis-report into an honest FAIL."""
    orch = _orch(tmp_path, _ZeroThroughputBackend(seed=1))

    with pytest.raises(NoBaselineError):
        orch.establish_baseline(_SPEC)

    rows = orch.ledger.read()
    base = [r for r in rows if r.stage is Stage.BASELINE]
    assert len(base) == 1
    assert base[0].status is Status.FAIL_NO_BASELINE
    assert base[0].status is not Status.KEEP
    assert base[0].metric == 0.0
    # No incumbent was ever set — the run is void.
    assert orch.incumbent is None


def test_positive_baseline_still_kept(tmp_path: Path):
    """Regression guard: a normal baseline (metric>0) is still recorded KEEP and
    becomes the incumbent — the honesty gate only fires on a hard 0."""
    orch = _orch(tmp_path, MockBackend(seed=1))
    base = orch.establish_baseline(_SPEC)

    assert orch.incumbent is base
    base_rows = [r for r in orch.ledger.read() if r.stage is Stage.BASELINE]
    assert len(base_rows) == 1
    assert base_rows[0].status is Status.KEEP
    assert base_rows[0].metric > 0.0
