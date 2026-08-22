"""
Regression tests for two deep-stage fixes:

  1. Real compile time is threaded into the ledger. Native PyTorch compiles
     lazily on the first forward INSIDE the worker, so compile() returns
     compile_seconds=0 and the real cost is only known after measure(). Before
     the fix the orchestrator only ever read neff.compile_seconds (== 0.0), so
     every deep-stage row logged compile_s=0.0 and the compile-timeout guardrail
     was dead. Now measure() carries Measurements.compile_seconds and the
     orchestrator records/gates on it.

  2. The Stage-6 profile loop re-enters run_deep_stages repeatedly. Without a
     dedup guard it recompiled the SAME fixed flag candidates against an
     unchanged incumbent every round — guaranteed "no gain" + wasted compiles.
     run_deep_stages now skips config-keys it already tried.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backends.base import Measurements, Neff
from backends.mock import MockBackend
from guardrails import Guardrails
from ledger import Ledger, Stage, Status
from orchestrator import ModelSpec, Orchestrator, always_equivalent


class _LazyCompileBackend(MockBackend):
    """Models native PyTorch: compile() is lazy (reports 0s); the REAL compile
    time surfaces only in measure() (the worker's compile_s). Returns a constant
    metric so no deep-stage flag candidate ever beats the incumbent — which makes
    profile-loop re-entry a clean dedup case (the incumbent never moves)."""

    MEASURED_COMPILE_S = 123.0
    CONST_METRIC = 1000.0

    def compile(self, artifact) -> Neff:
        return Neff(artifact=artifact, path="lazy", compile_seconds=0.0)

    def measure(self, neff, shape, batch) -> Measurements:
        m = super().measure(neff, shape, batch)
        m.metric = self.CONST_METRIC
        m.metric_p50 = self.CONST_METRIC
        m.compile_seconds = self.MEASURED_COMPILE_S   # the worker-reported value
        return m


SPEC = ModelSpec(
    model_id="acme/tiny-dense-1b", family="dense_causal_lm",
    param_count=1e9, parent="llama", probe_shape="chat 1k/512", probe_batch=1,
)


def _orch(tmp_path: Path) -> Orchestrator:
    from bank import KnowledgeBank
    return Orchestrator(
        backend=_LazyCompileBackend(seed=1),
        bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(),
        ledger=Ledger(tmp_path / "run"),
        equivalence=always_equivalent,
        sdk_version="2.28.0",
    )


def test_deep_stage_rows_record_real_compile_time(tmp_path: Path):
    """Every deep-stage row must carry the measured compile time, not 0.0."""
    orch = _orch(tmp_path)
    orch.ledger.init()
    orch.establish_baseline(SPEC)
    orch.run_deep_stages(SPEC)

    rows = orch.ledger.read()
    deep = [r for r in rows
            if r.stage in (Stage.KNOWN_KERNEL, Stage.BORROW, Stage.GRAPH_REWRITE)]
    assert deep, "expected deep-stage rows to be recorded"
    # The fix: these rows now show the worker-reported compile time.
    assert all(r.compile_s == _LazyCompileBackend.MEASURED_COMPILE_S for r in deep), (
        "deep-stage rows must record the measured compile_s, not 0.0 — "
        f"got {[r.compile_s for r in deep]}")


def test_graph_rewrite_sweeps_multiple_optlevels(tmp_path: Path):
    """Stage 5 must try more than one optlevel (O1/O2/O3), not just optlevel-3."""
    orch = _orch(tmp_path)
    orch.ledger.init()
    orch.establish_baseline(SPEC)
    orch.run_deep_stages(SPEC)

    gr = [r for r in orch.ledger.read() if r.stage is Stage.GRAPH_REWRITE]
    # The widened sweep is O1, O2, O3, O2+nocast, O3+transformer+nocast.
    assert len(gr) >= 3, f"expected an optlevel sweep, got {len(gr)} graph-rewrite rows"


def test_profile_loop_reentry_does_not_recompile_identical_candidates(tmp_path: Path):
    """Re-entering the deep stages against an unchanged incumbent must not
    recompile the same flag candidates — the dedup guard skips tried configs."""
    orch = _orch(tmp_path)
    orch.ledger.init()
    orch.establish_baseline(SPEC)

    orch.run_deep_stages(SPEC)
    after_first = sum(
        1 for r in orch.ledger.read()
        if r.stage in (Stage.KNOWN_KERNEL, Stage.BORROW, Stage.GRAPH_REWRITE))
    assert after_first >= 3, "first pass should compile the flag candidates"

    orch.run_deep_stages(SPEC)   # simulate a profile-loop re-entry
    after_second = sum(
        1 for r in orch.ledger.read()
        if r.stage in (Stage.KNOWN_KERNEL, Stage.BORROW, Stage.GRAPH_REWRITE))
    assert after_second == after_first, (
        "re-entry recompiled identical candidates — dedup guard not working "
        f"({after_first} -> {after_second})")


def test_stage4_stub_recorded_once_across_reentries(tmp_path: Path):
    """The 'Stage 4 not integrated' row is recorded once per run, not once per
    profile-loop round (no ledger spam)."""
    orch = _orch(tmp_path)
    orch.ledger.init()
    orch.establish_baseline(SPEC)
    orch.run_deep_stages(SPEC)
    orch.run_deep_stages(SPEC)
    orch.run_deep_stages(SPEC)

    invent = [r for r in orch.ledger.read() if r.stage is Stage.INVENT]
    assert len(invent) == 1, f"Stage-4 stub should record once, got {len(invent)}"
    assert invent[0].status is Status.DISCARD
