"""
Tests for the "keep working" fixes:
  - the proposer tries high-impact axes first (compile_mode before tp/dtype),
  - the search explores every axis before the soft stop can fire (so it never
    halts before compile_mode again),
  - the overnight driver runs multiple cycles and compounds via auto-promotion,
  - the latency-track seed lesson exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

from backends.mock import MockBackend
from bank import KnowledgeBank
from guardrails import Guardrails
from ledger import Ledger
from orchestrator import ModelSpec, Orchestrator, always_equivalent
from proposer import BeamProposer


def test_proposer_tries_compile_mode_first():
    axes = {"tp_degree": [1, 2, 4], "weights_dtype": ["bf16", "fp8"],
            "compile_mode": ["eager", "compile-default"]}
    p = BeamProposer(axes=axes)
    # compile_mode is the biggest lever, so it must be first in the search order
    assert list(p.axes)[0] == "compile_mode"
    assert list(p.axes).index("tp_degree") > list(p.axes).index("compile_mode")


def test_search_explores_every_axis_before_stopping(tmp_path: Path):
    """Regression for the overnight bug: the greedy stop fired before reaching
    compile_mode. Every config axis must appear in the ledger before the search
    terminates on a soft criterion."""
    orch = Orchestrator(
        backend=MockBackend(seed=1), bank=KnowledgeBank(tmp_path / "b"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "r"),
        equivalence=always_equivalent, sdk_version="2.28.0",
    )
    orch.ledger.init()
    orch.run_stage1_config(ModelSpec("m", "dense_causal_lm", 8e9, "qwen"))
    descs = " ".join(r.description for r in orch.ledger.read())
    for axis in MockBackend().config_axes():
        assert f"{axis}=" in descs, f"axis {axis!r} was never explored before stop"


def test_latency_and_fill_lessons_seeded():
    import seed_bank
    ids = {l.lesson_id for l in seed_bank.seed_lessons()}
    assert "latency-track-fill-tp-cp-not-dp" in ids
    assert "hybrid-attn-27b-tp-by-kvheads-then-fill-dp" in ids


def test_overnight_runs_multiple_cycles(tmp_path: Path, monkeypatch):
    """The driver must keep working across cycles (not run once and exit), write
    a leaderboard each cycle, and invoke auto-promotion so cycles compound."""
    import overnight
    art = tmp_path / "art"
    argv = [
        "overnight", "--backend", "mock", "--models", "qwen3-8b",
        "--out-root", str(art), "--bank-root", str(tmp_path / "bank"),
        "--cycles", "2", "--auto-promote", "--instance-type", "trn2.48xlarge",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    overnight.main()

    assert (art / "LEADERBOARD.md").exists()
    log = (art / "OVERNIGHT_LOG.md").read_text()
    assert "cycle 1 done" in log
    assert "cycle 2 done" in log            # it did NOT stop after one pass
    # cycle 2 used a distinct run dir (no ledger mixing)
    assert (art / "optimization_runs" / "qwen3-8b" / "cycle2").exists()


def test_overnight_stop_file_halts(tmp_path: Path, monkeypatch):
    """A STOP file ends the run cleanly instead of requiring a kill."""
    import overnight
    art = tmp_path / "art"
    art.mkdir(parents=True)
    (art / "STOP").touch()                  # stop before the first cycle
    argv = [
        "overnight", "--backend", "mock", "--models", "qwen3-8b",
        "--out-root", str(art), "--bank-root", str(tmp_path / "bank"),
        "--forever",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    overnight.main()                         # must return, not loop forever
    log = (art / "OVERNIGHT_LOG.md").read_text()
    assert "STOP file seen" in log
