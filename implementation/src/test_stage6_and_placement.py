"""
Tests for the two added features, against the mock backend (no hardware):

  Feature 1 — Stage 6, bounded profile-guided re-entry loop:
    - the loop stops after K consecutive no-improvement rounds (patience),
    - the max_rounds cap bounds it independently of patience,
    - it exits immediately when the profile shows no dominant bottleneck,
    - each round is recorded to the ledger as a PROFILE_LOOP row.

  Feature 2 — correctness-gated component placement axis:
    - placement_axes() degrades to a no-op for a model with no separable
      components (the LLM case),
    - a placement candidate that is FASTER but fails equivalence (the Wan 2.2
      device-scheduler drift) is discarded and never becomes the incumbent —
      i.e. the equivalence gate, not the metric, decides.
"""

from __future__ import annotations

from pathlib import Path

from backends.base import OpSite, Profile, placement_axes
from backends.mock import MockBackend
from bank import KnowledgeBank
from guardrails import Guardrails
from ledger import Ledger, Stage, Status
from orchestrator import EquivalenceResult, ModelSpec, Orchestrator, always_equivalent


SPEC = ModelSpec(
    model_id="mock/model", family="dense_causal_lm", param_count=8e9, parent="qwen",
    probe_shape="chat 1k/512", probe_batch=1,
)


class _DeterministicMock(MockBackend):
    """MockBackend with the throughput jitter pinned off, so re-measuring an
    identical config yields an identical metric. The profile loop's
    no-improvement bound is only meaningful against a deterministic backend."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)

        class _FixedRng:
            def uniform(self, _a, _b):  # midpoint / no jitter
                return 1.0

            def random(self):
                return 0.5
        self._rng = _FixedRng()


class _FlatProfileMock(_DeterministicMock):
    """A backend whose profile shows no dominant bottleneck (all ops < 30%)."""

    def profile(self, neff, shape):
        return Profile(
            op_sites=[OpSite("attention_prefill", 0.20), OpSite("mlp", 0.15)],
            bottleneck="balanced",
        )


def _orch(tmp_path: Path, backend) -> Orchestrator:
    orch = Orchestrator(
        backend=backend, bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "run"),
        equivalence=always_equivalent, sdk_version="2.28.0",
    )
    orch.ledger.init()
    return orch


def _reentry_rows(orch: Orchestrator):
    return [r for r in orch.ledger.read()
            if r.stage is Stage.PROFILE_LOOP and "re-entered" in r.description]


# -- Feature 1: Stage 6 loop -------------------------------------------------

def test_profile_loop_stops_after_patience(tmp_path: Path):
    """With a dominant bottleneck but no achievable gain, the loop must stop
    after exactly `patience` no-improvement rounds — well before max_rounds."""
    orch = _orch(tmp_path, _DeterministicMock(seed=1))
    orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)
    orch.run_deep_stages(SPEC)
    before = orch.incumbent.metric

    orch.run_profile_loop(SPEC, max_rounds=5, patience=2)

    reentries = _reentry_rows(orch)
    assert len(reentries) == 2, f"expected patience=2 re-entry rounds, got {len(reentries)}"
    # No spurious win: the deterministic backend can't improve on re-entry.
    assert orch.incumbent.metric == before
    assert all(r.status is Status.DISCARD for r in reentries)


def test_profile_loop_respects_max_rounds(tmp_path: Path):
    """A generous patience must not let the loop run past the max_rounds cap."""
    orch = _orch(tmp_path, _DeterministicMock(seed=2))
    orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)
    orch.run_deep_stages(SPEC)

    orch.run_profile_loop(SPEC, max_rounds=2, patience=99)

    assert len(_reentry_rows(orch)) == 2


def test_profile_loop_exits_when_no_dominant_bottleneck(tmp_path: Path):
    """A flat profile (no op >= 30% of step time) means nothing to re-attack —
    the loop records that and exits without re-entering the deep stages."""
    orch = _orch(tmp_path, _FlatProfileMock(seed=3))
    orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)
    orch.run_deep_stages(SPEC)

    orch.run_profile_loop(SPEC, max_rounds=5, patience=2)

    assert _reentry_rows(orch) == []
    loop_rows = [r for r in orch.ledger.read() if r.stage is Stage.PROFILE_LOOP]
    assert len(loop_rows) == 1
    assert "no dominant bottleneck" in loop_rows[0].description


# -- Feature 2: correctness-gated placement axis -----------------------------

def test_placement_axes_noop_without_components():
    """A model exposing no separable components emits no placement axis — the
    graceful-degradation contract for the LLM family."""
    assert placement_axes([]) == {}
    assert placement_axes(["scheduler"]) == {"place:scheduler": ["cpu", "device"]}


class _PlaceableMock(_DeterministicMock):
    """A diffusion-like mock that exposes scheduler + text_encoder placement and
    (adversarially) reports the on-device scheduler as *faster*. It is not — in
    reality it drifts (Wan 2.2: PSNR 34.7 vs 56.2 dB) — so the equivalence gate,
    not the metric, must reject it."""

    def build_baseline(self, model_id):
        art = super().build_baseline(model_id)
        art.config.update({"place:scheduler": "cpu", "place:text_encoder": "device"})
        return art

    def config_axes(self):
        axes = super().config_axes()
        axes.update(placement_axes(["scheduler", "text_encoder"]))
        return axes

    def _throughput(self, cfg, shape, batch):
        base = super()._throughput(cfg, shape, batch)
        if cfg.get("place:scheduler") == "device":
            base *= 1.25   # looks like a win — but it is numerically wrong
        return base


def _scheduler_drifts_on_device(neff, _spec) -> EquivalenceResult:
    cfg = neff.artifact.config
    if cfg.get("place:scheduler") == "device":
        return EquivalenceResult(
            passed=False, correctness_pct=61.9,
            notes="bf16 scheduler drift over sequential steps (Wan: 34.7 vs 56.2 dB)")
    return EquivalenceResult(passed=True, correctness_pct=100.0)


def test_fast_but_wrong_placement_is_discarded(tmp_path: Path):
    orch = Orchestrator(
        backend=_PlaceableMock(seed=1), bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "run"),
        equivalence=_scheduler_drifts_on_device, sdk_version="2.28.0",
    )
    orch.ledger.init()
    orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)

    rows = orch.ledger.read()
    # The device-scheduler placement WAS tried (and reached measurement)...
    tried = [r for r in rows if "place:scheduler=device" in r.description]
    assert tried, "expected the device-scheduler placement to be proposed and evaluated"
    # ...and every such row failed equivalence and was discarded — the metric
    # made it look faster, so only the correctness gate can be what rejected it.
    assert any("equivalence fail" in r.description for r in tried)
    assert not any(r.status is Status.KEEP and "place:scheduler=device" in r.description
                   for r in rows)
    # The incumbent kept the safe (CPU) scheduler placement.
    assert orch.incumbent.config.get("place:scheduler") == "cpu"
