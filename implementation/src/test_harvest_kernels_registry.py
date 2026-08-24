"""Tests for the three HARVESTED differentiator kernels banked into the corpus:
Mamba-2 SSD (slot ``Mamba2``), KDA (slot ``KDA``), and Qwen3.5 Gated DeltaNet
(slot ``DeltaNet``).

Theme (matching test_kernel_registry_and_routing.py): each novel primitive the
compiler can't auto-lower is now a REGISTERED, on-device-validated kernel — not a
blocker. These checks run on a plain CPU box: they resolve the registry, confirm
the manifest is rank-4 (passed-on-device), and load each adapter's forward-factory
through the SAME on-disk loader kernel_inject uses — WITHOUT importing torch / nki
/ numpy (the adapters keep those lazy, inside the built forward).
"""

from __future__ import annotations

from pathlib import Path

from kernel_registry import KernelRegistry, kernel_for_primitive
from backends.kernel_inject import load_kernel_entry

# The banked kernels live in implementation/src/kernels/ (next to this test's dir).
_KERNEL_DIR = Path(__file__).resolve().parent / "kernels"


class _MockModule:
    """A stand-in target module for a forward-factory. Optional attributes let a
    test request a variant (kda_mode / ssd_chunk_size) without torch."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def _registry() -> KernelRegistry:
    return KernelRegistry(kernel_dir=_KERNEL_DIR)


# -- primitive routing: the new aliases resolve to the right canonical slot ---

def test_mamba2_primitives_route_to_mamba2():
    for p in ("mamba2_ssd", "ssm_scan", "mamba2", "mamba", "ssm",
              "selective_scan"):
        assert kernel_for_primitive(p) == "Mamba2", p


def test_kda_primitives_route_to_kda():
    for p in ("kda", "kimi_delta_attention", "gated_delta_linear_attention"):
        assert kernel_for_primitive(p) == "KDA", p


def test_gated_delta_net_primitives_route_to_deltanet():
    for p in ("gated_delta_net", "deltanet", "linear_attention",
              "gated_delta", "delta_rule"):
        assert kernel_for_primitive(p) == "DeltaNet", p


# -- registry resolves each to a rank-4 (passed-on-device) KernelSpec ---------

def test_mamba2_spec_is_rank4():
    spec = _registry().for_primitive("mamba2_ssd")
    assert spec is not None and spec.name == "Mamba2"
    assert spec.status == "passed-on-device" and spec.rank == 4
    assert spec.usable and spec.hw_ready
    assert spec.entry == "adapter:build_mamba2_forward"
    assert spec.path == "Mamba2/adapter.py"


def test_kda_spec_is_rank4():
    spec = _registry().for_primitive("kimi_delta_attention")
    assert spec is not None and spec.name == "KDA"
    assert spec.status == "passed-on-device" and spec.rank == 4
    assert spec.usable and spec.hw_ready
    assert spec.entry == "adapter:build_kda_forward"
    assert spec.path == "KDA/adapter.py"


def test_deltanet_spec_is_rank4():
    spec = _registry().for_primitive("gated_delta_net")
    assert spec is not None and spec.name == "DeltaNet"
    assert spec.status == "passed-on-device" and spec.rank == 4
    assert spec.usable and spec.hw_ready
    assert spec.entry == "adapter:build_gdn_forward"
    assert spec.path == "DeltaNet/adapter.py"


# -- each adapter loads + builds a forward, CPU-safe (no torch/nki/numpy) -----

def _load_factory(spec):
    """Load the forward-factory exactly as backends.kernel_inject would, from the
    manifest's (path, entry) resolved against the kernel dir."""
    fpath = _KERNEL_DIR / spec.path
    return load_kernel_entry(str(fpath), spec.entry)


def test_mamba2_adapter_builds_forward():
    spec = _registry().for_primitive("mamba2_ssd")
    factory = _load_factory(spec)
    assert callable(factory)
    forward = factory(_MockModule(ssd_chunk_size=128))
    assert callable(forward)          # built without importing nkilib / numpy


def test_kda_adapter_builds_forward_both_modes():
    spec = _registry().for_primitive("kda")
    factory = _load_factory(spec)
    assert callable(factory)
    for mode in ("prefill", "decode"):
        forward = factory(_MockModule(kda_mode=mode))
        assert callable(forward)      # built without importing nki / numpy


def test_deltanet_adapter_builds_forward():
    spec = _registry().for_primitive("linear_attention")
    factory = _load_factory(spec)
    assert callable(factory)
    forward = factory(_MockModule())
    assert callable(forward)          # built without importing nki / numpy


# -- pre-existing routing is preserved (no regression to canonical slots) -----

def test_existing_canonical_routing_unchanged():
    assert kernel_for_primitive("mamba2") == "Mamba2"
    assert kernel_for_primitive("GatedDeltaNet") == "DeltaNet"
    assert kernel_for_primitive("kda") == "KDA"
    assert kernel_for_primitive("plain_dense_attention") is None
