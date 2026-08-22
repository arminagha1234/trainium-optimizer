"""Tests for the kernel-routing layer: kernel_registry + preflight.kernel_route +
invent_engine's prior-art / Harvest step.

The theme: a novel primitive the compiler can't auto-lower (linear-attention /
GatedDeltaNet) is a NAMED KERNEL NEED, not a permanent blocker — and an authored
kernel that already exists is HARVESTED, not re-invented.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernel_registry import (
    KernelRegistry,
    kernel_for_primitive,
)
from preflight import (
    LINEAR_ATTN_REASON,
    kernel_route,
    preflight_check,
)
from orchestrator import ModelSpec


DELTANET_CFG = {"architectures": ["Qwen3NextForCausalLM"],
                "model_type": "qwen3_next", "linear_attention": "gated_delta"}
DENSE_CFG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}
GDN_SPEC = ModelSpec(model_id="Qwen/Qwen3-Next-80B-A3B", family="hybrid_causal_lm",
                     param_count=80e9)
DENSE_SPEC = ModelSpec(model_id="Qwen/Qwen3-4B", family="dense_causal_lm",
                       param_count=4e9)


def _registry_with_deltanet(tmp_path: Path, status: str) -> KernelRegistry:
    kdir = tmp_path / "kernels" / "DeltaNet"
    kdir.mkdir(parents=True)
    (kdir / "kernel.json").write_text(json.dumps({
        "name": "DeltaNet", "status": status,
        "entry": "gdn_chunked_prefill_nki:gdn_chunked_prefill_kernel",
        "variants": ["gated_deltanet"],
        "tolerances": {"bf16": 8.3e-4, "fp32": 2.3e-5},
    }))
    return KernelRegistry(tmp_path / "kernels")


# -- kernel_registry ---------------------------------------------------------

def test_primitive_maps_to_deltanet():
    for p in ("linear_attention", "GatedDeltaNet", "gated_delta_net",
              "delta_rule", "deltanet", "hybrid_ssm_attn"):
        assert kernel_for_primitive(p) == "DeltaNet", p
    assert kernel_for_primitive("mamba2") == "Mamba2"
    assert kernel_for_primitive("mla") == "MLA"
    assert kernel_for_primitive("plain_dense_attention") is None


def test_empty_registry_finds_nothing():
    reg = KernelRegistry(kernel_dir=None)
    assert reg.lookup("DeltaNet") is None
    assert reg.available("DeltaNet") is False
    assert reg.for_primitive("linear_attention") is None


def test_registry_reads_manifest_and_ranks_status(tmp_path: Path):
    reg = _registry_with_deltanet(tmp_path, status="passed-on-device")
    spec = reg.for_primitive("linear_attention")
    assert spec is not None and spec.name == "DeltaNet"
    assert spec.usable and spec.hw_ready
    assert spec.tolerances["bf16"] == 8.3e-4


def test_failed_compile_kernel_is_not_usable(tmp_path: Path):
    reg = _registry_with_deltanet(tmp_path, status="failed-compile")
    spec = reg.lookup("DeltaNet")
    assert spec is not None and not spec.usable        # rank 1 < usable
    assert reg.available("DeltaNet") is False


def test_simulate_pass_is_usable_but_not_hw_ready(tmp_path: Path):
    reg = _registry_with_deltanet(tmp_path, status="passed")
    spec = reg.lookup("DeltaNet")
    assert spec.usable and not spec.hw_ready           # simulate != on-device


def test_malformed_manifest_is_not_available(tmp_path: Path):
    kdir = tmp_path / "k" / "DeltaNet"
    kdir.mkdir(parents=True)
    (kdir / "kernel.json").write_text("{ not json")
    reg = KernelRegistry(tmp_path / "k")
    assert reg.lookup("DeltaNet") is None


# -- preflight.kernel_route --------------------------------------------------

def test_kernel_route_names_deltanet_when_unavailable():
    need = kernel_route(GDN_SPEC, DELTANET_CFG)          # empty registry
    assert need is not None
    assert need.kernel_name == "DeltaNet"
    assert need.available is False and need.hw_ready is False
    assert "needs the DeltaNet kernel" in need.reason


def test_kernel_route_routes_when_available(tmp_path: Path):
    reg = _registry_with_deltanet(tmp_path, status="passed-on-device")
    need = kernel_route(GDN_SPEC, DELTANET_CFG, registry=reg)
    assert need.available and need.hw_ready
    assert "route to the DeltaNet kernel" in need.reason
    assert "on-device-validated" in need.reason


def test_kernel_route_none_for_dense():
    assert kernel_route(DENSE_SPEC, DENSE_CFG) is None


# -- preflight_check integration ---------------------------------------------

def test_preflight_default_reason_unchanged():
    # No registry, kernels not wired: byte-for-byte the previous behaviour.
    ok, reason = preflight_check(GDN_SPEC, config=DELTANET_CFG)
    assert ok is False and reason == LINEAR_ATTN_REASON


def test_preflight_names_kernel_when_registry_passed(tmp_path: Path):
    reg = _registry_with_deltanet(tmp_path, status="passed-on-device")
    ok, reason = preflight_check(GDN_SPEC, config=DELTANET_CFG, registry=reg)
    assert ok is False                                   # not wired -> still skip
    assert "DeltaNet" in reason


def test_preflight_proceeds_when_kernel_available_and_wired(tmp_path: Path):
    reg = _registry_with_deltanet(tmp_path, status="passed-on-device")
    ok, reason = preflight_check(GDN_SPEC, config=DELTANET_CFG,
                                 registry=reg, kernels_wired=True)
    assert ok is True and reason is None                 # kernel path unblocks it


def test_preflight_dense_never_gated():
    ok, reason = preflight_check(DENSE_SPEC, config=DENSE_CFG)
    assert ok is True and reason is None


# -- invent_engine prior-art / Harvest ---------------------------------------

def _op(primitive: str = ""):
    from invent_kernels import OpSpec
    return OpSpec(
        name="gdn_chunk_op", family="linear_attn", shape_class="C16_K128",
        dtype="bf16", reference=lambda **k: None,
        offline_inputs=lambda: {}, real_inputs=lambda: {},
        primitive=primitive,
    )


def test_invent_harvests_existing_kernel(tmp_path: Path):
    from invent_engine import InventEngine
    reg = _registry_with_deltanet(tmp_path, status="passed-on-device")
    eng = InventEngine(out_dir=tmp_path / "run", registry=reg)
    res = eng.run_op(_op(primitive="linear_attention"))
    assert res.status == "harvested"
    assert "DeltaNet" in res.detail
    # A HARVESTED keep row is recorded (reuse is visible in the ledger).
    assert any("harvested existing DeltaNet" in r.description for r in eng.ledger.read())


def test_invent_without_primitive_does_not_harvest(tmp_path: Path):
    from invent_engine import InventEngine
    reg = _registry_with_deltanet(tmp_path, status="passed-on-device")
    eng = InventEngine(out_dir=tmp_path / "run2", registry=reg)
    res = eng.run_op(_op(primitive=""))       # no primitive -> no prior-art lookup
    assert res.status != "harvested"


def test_invent_does_not_harvest_failed_compile_kernel(tmp_path: Path):
    from invent_engine import InventEngine
    reg = _registry_with_deltanet(tmp_path, status="failed-compile")   # not usable
    eng = InventEngine(out_dir=tmp_path / "run3", registry=reg)
    res = eng.run_op(_op(primitive="linear_attention"))
    assert res.status != "harvested"          # a failed attempt is not prior art
