"""
Tests for the Stage-3 (BORROW) fused-MoE-megakernel candidate, against the mock
backend (no hardware). Mirrors the placement-axis test style in
test_stage6_and_placement.py.

What the wiring must guarantee:
  - the MoE-borrow candidate is OFFERED (evaluated as a Stage.BORROW row, with
    the borrowed-kernel source) for a MoE-family model,
  - it is NOT offered for a dense LLM — a graceful no-op, exactly like the
    placement axis degrades to nothing for a model with no separable
    components,
  - a fused-kernel candidate that is FASTER but fails equivalence is discarded
    and never becomes the incumbent (the equivalence gate, not the metric,
    decides — nothing is forced on-device),
  - the dependency-light adapter precheck correctly self-skips off-contract
    models (tiny-random-qwen3moe) and accepts the exact A3B/TP4 contract.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from backends.mock import MockBackend
from bank import KnowledgeBank
from guardrails import Guardrails
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _P

# knowledge-bank/kernels/ cannot be an import package (the directory name has a
# hyphen), so load moe_fused by path and register it under a synthetic name that
# the import below can resolve.
_moe_init = (_P(__file__).resolve().parents[2] / "knowledge-bank" / "kernels"
             / "moe_fused" / "__init__.py")
if "_kb_moe_fused" not in _sys.modules and _moe_init.is_file():
    _spec = _ilu.spec_from_file_location("_kb_moe_fused", _moe_init,
                                         submodule_search_locations=[str(_moe_init.parent)])
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["_kb_moe_fused"] = _mod
    _spec.loader.exec_module(_mod)

from _kb_moe_fused import (
    FUSED_NKI,
    KERNEL_SOURCE,
    MOE_KERNEL_KEY,
    SUPPORTED_CONTRACT,
    is_moe_arch,
    precheck,
)
from ledger import Ledger, Origin, Stage, Status
from orchestrator import EquivalenceResult, ModelSpec, Orchestrator, always_equivalent


MOE_SPEC = ModelSpec(
    model_id="mock/qwen3-moe", family="moe_causal_lm", param_count=30e9, parent="qwen",
    probe_shape="chat 1k/512", probe_batch=1,
)
DENSE_SPEC = ModelSpec(
    model_id="mock/dense", family="dense_causal_lm", param_count=8e9, parent="qwen",
    probe_shape="chat 1k/512", probe_batch=1,
)

_MOE_LABEL = "moe:fused-nki-megakernel"


class _DeterministicMock(MockBackend):
    """Jitter pinned off so re-measuring an identical config is identical."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)

        class _FixedRng:
            def uniform(self, _a, _b):
                return 1.0

            def random(self):
                return 0.5
        self._rng = _FixedRng()


class _MoEMock(_DeterministicMock):
    """A backend that (like native_pytorch on a MoE model) offers the fused-MoE
    borrow candidate."""

    def moe_kernel_candidates(self, artifact):
        return [(_MOE_LABEL, {MOE_KERNEL_KEY: FUSED_NKI})]


def _orch(tmp_path: Path, backend, equivalence=always_equivalent) -> Orchestrator:
    orch = Orchestrator(
        backend=backend, bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "run"),
        equivalence=equivalence, sdk_version="2.28.0",
    )
    orch.ledger.init()
    return orch


def _moe_rows(orch: Orchestrator):
    return [r for r in orch.ledger.read() if _MOE_LABEL in r.description]


# -- offering logic ----------------------------------------------------------

def test_moe_borrow_offered_for_moe_model(tmp_path: Path):
    """A MoE model gets the fused-MoE megakernel evaluated as a Stage.BORROW
    candidate, tagged BORROWED with the vendored-kernel source."""
    orch = _orch(tmp_path, _MoEMock(seed=1))
    orch.establish_baseline(MOE_SPEC)
    orch.run_stage1_config(MOE_SPEC)
    orch.run_deep_stages(MOE_SPEC)

    rows = _moe_rows(orch)
    assert rows, "expected the fused-MoE borrow candidate to be evaluated"
    r = rows[0]
    assert r.stage is Stage.BORROW
    assert r.origin is Origin.BORROWED
    assert r.source == KERNEL_SOURCE


def test_moe_borrow_not_offered_for_dense_model(tmp_path: Path):
    """A dense LLM (backend offers no MoE candidate) sees no fused-MoE borrow —
    the graceful no-op contract, mirroring placement_axes([]) for LLMs."""
    orch = _orch(tmp_path, _DeterministicMock(seed=1))   # base mock: no candidates
    orch.establish_baseline(DENSE_SPEC)
    orch.run_stage1_config(DENSE_SPEC)
    orch.run_deep_stages(DENSE_SPEC)

    assert _moe_rows(orch) == []


# -- equivalence gate --------------------------------------------------------

class _FastButWrongMoEMock(_MoEMock):
    """Reports the fused-MoE kernel as FASTER, but its output (in reality) would
    drift — so the equivalence gate, not the metric, must reject it."""

    def _throughput(self, cfg, shape, batch):
        base = super()._throughput(cfg, shape, batch)
        if cfg.get(MOE_KERNEL_KEY) == FUSED_NKI:
            base *= 1.5   # looks like a big win...
        return base


def _fused_moe_drifts(neff, _spec) -> EquivalenceResult:
    if neff.artifact.config.get(MOE_KERNEL_KEY) == FUSED_NKI:
        return EquivalenceResult(passed=False, correctness_pct=58.0,
                                 notes="fused MoE output drift past tolerance")
    return EquivalenceResult(passed=True, correctness_pct=100.0)


def test_fast_but_wrong_moe_kernel_is_discarded(tmp_path: Path):
    orch = _orch(tmp_path, _FastButWrongMoEMock(seed=1),
                 equivalence=_fused_moe_drifts)
    orch.establish_baseline(MOE_SPEC)
    orch.run_stage1_config(MOE_SPEC)
    orch.run_deep_stages(MOE_SPEC)

    rows = _moe_rows(orch)
    assert rows, "the fused-MoE candidate should have been proposed + evaluated"
    # It was tried and reached the equivalence gate, which failed it...
    assert any("equivalence fail" in r.description for r in rows)
    # ...so despite looking faster, it was never kept as the incumbent.
    assert not any(r.status is Status.KEEP for r in rows)
    assert orch.incumbent.config.get(MOE_KERNEL_KEY) != FUSED_NKI


# -- ledger honesty: swapped vs fell-back-to-eager ---------------------------

class _EagerFallbackMoEMock(_MoEMock):
    """Like native_pytorch when the borrow's precondition is unmet: the run is
    correct (unchanged, eager) and the worker reports the fallback. measure()
    surfaces that via Measurements.moe_kernel_swap, exactly as the real worker's
    JSON does."""

    def measure(self, neff, shape, batch):
        import dataclasses
        m = super().measure(neff, shape, batch)
        if neff.artifact.config.get(MOE_KERNEL_KEY) == FUSED_NKI:
            return dataclasses.replace(
                m, moe_kernel_swap="eager-fallback: off-contract dims")
        return m


def test_moe_borrow_ledger_notes_eager_fallback(tmp_path: Path):
    """When the fused MoE kernel silently falls back to eager, the borrow row
    must SAY so — a reader can tell 'kernel ran' from 'fell back to eager'."""
    orch = _orch(tmp_path, _EagerFallbackMoEMock(seed=1))
    orch.establish_baseline(MOE_SPEC)
    orch.run_stage1_config(MOE_SPEC)
    orch.run_deep_stages(MOE_SPEC)

    rows = _moe_rows(orch)
    assert rows, "expected the fused-MoE borrow candidate to be evaluated"
    assert any("[moe-kernel: eager-fallback" in r.description for r in rows), \
        "borrow row must disclose the eager fallback, not just the provenance"


# -- dependency-light adapter precheck ---------------------------------------

def _fake_cfg(**kw):
    return SimpleNamespace(**kw)


def test_fused_moe_kernel_uses_broadcast_to_not_broadcast():
    """nki 0.6.0 renamed the tile method .broadcast(dim, n) -> .broadcast_to(shape);
    the old form raises AttributeError on-device. The vendored fused-MoE kernel
    source must use the new API. (Text check: importing needs nkilib, absent in
    the unit-test env.)"""
    src = (Path(__file__).resolve().parents[2] / "knowledge-bank" / "kernels" / "moe_fused"
           / "moe_fused_nki.py").read_text()
    assert ".broadcast_to((" in src, "kernel should use the nki 0.6.0 broadcast_to API"
    # No bare .broadcast(dim=... / .broadcast( call survives (comments describing
    # the rename mention it, so match the call form specifically).
    assert ".broadcast(dim" not in src, "old .broadcast(dim, n) form must be gone"


def test_is_moe_arch_detection():
    assert is_moe_arch(_fake_cfg(architectures=["Qwen3MoeForCausalLM"]))
    assert is_moe_arch(_fake_cfg(num_experts=128))
    assert is_moe_arch(_fake_cfg(num_local_experts=8))
    assert not is_moe_arch(_fake_cfg(architectures=["Qwen3ForCausalLM"]))
    assert not is_moe_arch(_fake_cfg(num_attention_heads=32))


def test_precheck_accepts_a3b_tp4_and_skips_off_contract():
    a3b = _fake_cfg(architectures=["Qwen3MoeForCausalLM"], **{
        k: v for k, v in SUPPORTED_CONTRACT.items() if k != "tp_degree"})
    ok, reason = precheck(a3b, tp_degree=4)
    assert ok, reason

    # Wrong TP: MoE but off-contract -> skip.
    ok_tp, _ = precheck(a3b, tp_degree=8)
    assert not ok_tp

    # tiny-random-qwen3moe: MoE arch, but tiny dims -> skip (documented no-op).
    tiny = _fake_cfg(architectures=["Qwen3MoeForCausalLM"], hidden_size=64,
                     num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32)
    ok_tiny, reason_tiny = precheck(tiny, tp_degree=4)
    assert not ok_tiny
    assert "contract" in reason_tiny

    # Dense model -> skip on arch.
    dense = _fake_cfg(architectures=["Qwen3ForCausalLM"], hidden_size=2048)
    ok_dense, _ = precheck(dense, tp_degree=4)
    assert not ok_dense
