"""Tests for the Stage-4 INVENT wiring in orchestrator.run_deep_stages.

The real InventEngine (author -> offline-gate -> on-device race -> bank) runs
only on a Trainium box; here we inject a FAKE engine (a run_op stand-in) to prove
the ORCHESTRATOR wiring: Stage 4 is a no-op discard when disabled (unchanged
behaviour), authors the compiler-weak targets when enabled, records each outcome
as a Stage.INVENT row, runs once per model-run (not per profile-loop round),
dedupes across models sharing one engine, and NEVER lets an invent failure break
the model's config/borrow/rewrite result.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backends.mock import MockBackend
from bank import KnowledgeBank
from guardrails import Guardrails
from ledger import Ledger, Origin, Stage, Status
from orchestrator import ModelSpec, Orchestrator, _seed_op_specs, always_equivalent


DENSE = ModelSpec(model_id="mock/dense", family="dense_causal_lm",
                  param_count=8e9, parent="qwen", probe_shape="chat 1k/512",
                  probe_batch=1)


class _FakeEngine:
    """Stands in for InventEngine.run_op. Records calls; returns a chosen
    InventResult-shaped outcome (status + race.speedup/correctness_pct + detail)."""

    def __init__(self, status="win", raise_on_run=False, max_targets=2):
        self.status = status
        self.raise_on_run = raise_on_run
        self.calls: list[str] = []
        self._invent_max_targets = max_targets
        self._attempted_shape_classes: set[str] = set()

    def run_op(self, op_spec):
        self.calls.append(op_spec.name)
        if self.raise_on_run:
            raise RuntimeError("boom in run_op")
        race = SimpleNamespace(speedup=1.30, correctness_pct=100.0)
        return SimpleNamespace(status=self.status, race=race,
                               detail=f"{self.status} detail for {op_spec.name}",
                               op=op_spec.name)


def _orch(tmp_path: Path, engine=None, run_slug="run") -> Orchestrator:
    orch = Orchestrator(
        backend=MockBackend(seed=1), bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(tmp_path / run_slug),
        equivalence=always_equivalent, sdk_version="2.28.0",
        invent_engine=engine)
    orch.ledger.init()
    return orch


def _invent_rows(orch: Orchestrator):
    return [r for r in orch.ledger.read() if r.stage is Stage.INVENT]


# -- disabled: unchanged behaviour -------------------------------------------

def test_stage4_disabled_records_not_enabled(tmp_path: Path):
    orch = _orch(tmp_path, engine=None)
    orch.establish_baseline(DENSE)
    orch.run_deep_stages(DENSE)
    rows = _invent_rows(orch)
    assert len(rows) == 1
    assert rows[0].status is Status.DISCARD
    assert "not enabled" in rows[0].description


# -- enabled: authors + records the outcome ----------------------------------

def test_stage4_enabled_authors_and_records_win(tmp_path: Path):
    eng = _FakeEngine(status="win")
    orch = _orch(tmp_path, engine=eng)
    orch.establish_baseline(DENSE)
    orch.run_deep_stages(DENSE)
    # the dense model's sole compiler-weak target is the gated-delta scan
    assert eng.calls == ["gated_delta_rule"]
    rows = _invent_rows(orch)
    assert any(r.status is Status.KEEP and r.origin is Origin.INVENTED
               and "gated_delta_rule" in r.description for r in rows)


def test_stage4_records_anti_pattern_on_loss(tmp_path: Path):
    eng = _FakeEngine(status="anti_pattern")
    orch = _orch(tmp_path, engine=eng)
    orch.establish_baseline(DENSE)
    orch.run_deep_stages(DENSE)
    rows = _invent_rows(orch)
    assert rows and all(r.status is Status.DISCARD for r in rows)
    assert any("anti_pattern" in r.description for r in rows)


def test_stage4_device_deferred_is_discard_not_win(tmp_path: Path):
    eng = _FakeEngine(status="device_deferred")
    orch = _orch(tmp_path, engine=eng)
    orch.establish_baseline(DENSE)
    orch.run_deep_stages(DENSE)
    rows = _invent_rows(orch)
    assert rows and all(r.status is Status.DISCARD for r in rows)


# -- robustness: an invent failure never breaks the pipeline -----------------

def test_stage4_engine_exception_never_breaks_pipeline(tmp_path: Path):
    eng = _FakeEngine(raise_on_run=True)
    orch = _orch(tmp_path, engine=eng)
    orch.establish_baseline(DENSE)
    incumbent_before = orch.incumbent
    result = orch.run_deep_stages(DENSE)          # must NOT raise
    assert result is orch.incumbent               # pipeline intact
    # the raised invent op is recorded honestly as a discard
    rows = _invent_rows(orch)
    assert any("raised" in r.description for r in rows)
    # a Stage-4 crash never touches the model's config incumbent
    assert orch.incumbent.metric == incumbent_before.metric


# -- once per model-run (profile-loop re-entry must not re-author) -----------

def test_stage4_runs_once_across_reentry(tmp_path: Path):
    eng = _FakeEngine(status="win")
    orch = _orch(tmp_path, engine=eng)
    orch.establish_baseline(DENSE)
    orch.run_deep_stages(DENSE)
    orch.run_deep_stages(DENSE)                   # simulate a profile-loop re-entry
    assert eng.calls == ["gated_delta_rule"]      # authored exactly once


# -- cross-model dedupe (one shared engine across models) --------------------

def test_stage4_cross_model_dedupe(tmp_path: Path):
    eng = _FakeEngine(status="device_deferred")   # a non-win must not re-author per model
    for i in range(3):
        orch = _orch(tmp_path / f"m{i}", engine=eng, run_slug=f"run{i}")
        orch.establish_baseline(DENSE)
        orch.run_deep_stages(DENSE)
    assert eng.calls == ["gated_delta_rule"]       # attempted once across 3 models


# -- seed selection ----------------------------------------------------------

def test_seed_op_specs_dense_returns_the_scan(tmp_path: Path):
    ops = _seed_op_specs(DENSE, max_targets=2)
    names = [o.name for o in ops]
    assert "gated_delta_rule" in names
    # memory-bound catalog ops (rmsnorm/gelu/...) are pruned by the opportunity gate
    assert "rmsnorm" not in names and "gelu_tanh" not in names
