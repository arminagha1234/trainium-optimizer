"""
Tests for the text-to-image diffusion backend (family="diffusion"), against
the import/routing surface only — no Trainium, no torch, no diffusers required
(native_diffusion.py imports none of them at module load; the heavy deps live in
the worker subprocess it shells out to). Mirrors the mock-only style of the rest
of the suite.

Covers:
  1. _make_backend("diffusion-native") / ("diffusion") returns the DiffusionBackend.
  2. run_one routes a spec with family="diffusion" to the diffusion backend,
     leaves every other family on the requested backend, and never overrides
     the 'mock' backend (laptop smoke-runs stay synthetic).
  3. (ties to PR #5) the correctness-gated placement axis is offered for the
     diffusion components `scheduler` and `text_encoder`, and the baseline is
     seeded with their known-safe default placement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import overnight
from backends.base import placement_axis_key
from backends.native_diffusion import DiffusionBackend
from bank import KnowledgeBank
from orchestrator import ModelSpec


# -- 1. backend construction -------------------------------------------------

@pytest.mark.parametrize("name", ["diffusion-native", "diffusion"])
def test_make_backend_returns_diffusion_backend(name):
    """Both aliases resolve to the DiffusionBackend, named 'diffusion-native'."""
    backend = overnight._make_backend(name, instance_type="trn2.3xlarge")
    assert isinstance(backend, DiffusionBackend)
    assert backend.name == "diffusion-native"


# -- 2. family-driven routing in run_one -------------------------------------

def _diffusion_spec() -> ModelSpec:
    return ModelSpec(
        model_id="/home/ubuntu/sd-turbo", family="diffusion",
        param_count=0.9e9, parent="stabilityai",
        probe_shape="512x512 x1step", probe_batch=1,
    )


def _dense_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen3-0.6B", family="dense_causal_lm",
        param_count=0.6e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
    )


def _requested_backend(monkeypatch, spec: ModelSpec, backend_name: str,
                       tmp_path: Path) -> str:
    """Run run_one just far enough to observe which backend name it asks
    _make_backend for, then abort. run_one swallows the abort (its broad
    except returns a failed ModelResult), so the captured name is the assertion."""
    captured: dict[str, str] = {}

    def _spy(name, instance_type=None):
        captured["name"] = name
        raise RuntimeError("stop-after-selection")   # abort before any measurement

    monkeypatch.setattr(overnight, "_make_backend", _spy)
    result = overnight.run_one(
        slug="probe", spec=spec, backend_name=backend_name,
        out_root=tmp_path, bank=KnowledgeBank(tmp_path / "bank"),
        sdk_version="2.28.0", log=lambda *_a, **_k: None,
        instance_type="trn2.3xlarge",
    )
    assert result.ok is False   # the abort was caught, not raised
    return captured["name"]


def test_run_one_routes_diffusion_family_to_diffusion_backend(monkeypatch, tmp_path):
    """A diffusion spec is routed to diffusion-native even when the driver asks
    for the causal-LM backend (the continuous driver always passes that)."""
    got = _requested_backend(monkeypatch, _diffusion_spec(),
                             "native-pytorch-beta3", tmp_path)
    assert got == "diffusion-native"


def test_run_one_leaves_non_diffusion_family_on_requested_backend(monkeypatch, tmp_path):
    got = _requested_backend(monkeypatch, _dense_spec(),
                             "native-pytorch-beta3", tmp_path)
    assert got == "native-pytorch-beta3"


def test_run_one_does_not_override_mock(monkeypatch, tmp_path):
    """The mock backend is never overridden — laptop smoke-runs stay synthetic
    even for a diffusion spec."""
    got = _requested_backend(monkeypatch, _diffusion_spec(), "mock", tmp_path)
    assert got == "mock"


# -- 3. PR #5 placement axis for diffusion components ------------------------

def test_placement_axis_offered_for_scheduler_and_text_encoder():
    """The diffusion backend activates the #5 placement axis for exactly the two
    components a pipeline exposes: scheduler and text_encoder (cpu vs device)."""
    backend = DiffusionBackend()
    assert backend._placeable_components() == ["scheduler", "text_encoder"]

    axes = backend.config_axes()
    assert axes[placement_axis_key("scheduler")] == ["cpu", "device"]
    assert axes[placement_axis_key("text_encoder")] == ["cpu", "device"]


def test_baseline_seeds_known_safe_placement():
    """Stage 0 seeds the known-safe (validated) placement — both components on
    CPU — so the search proposes the device alternative from a correct incumbent."""
    backend = DiffusionBackend()
    art = backend.build_baseline("/home/ubuntu/sd-turbo")
    assert art.config[placement_axis_key("scheduler")] == "cpu"
    assert art.config[placement_axis_key("text_encoder")] == "cpu"
