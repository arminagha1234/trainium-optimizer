# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for op_signature.resolve_kernel_name (robust primitive -> kernel-name
resolution) and its wiring into KernelRegistry.for_signature. Pure CPU/stdlib."""

from __future__ import annotations

import json

from op_signature import resolve_kernel_name
from kernel_registry import KernelRegistry


# --- exact resolution is unchanged -------------------------------------------

def test_exact_hits_unchanged():
    assert resolve_kernel_name("gated_delta_net") == "DeltaNet"
    assert resolve_kernel_name("GatedDeltaNet") == "DeltaNet"
    assert resolve_kernel_name("mamba2") == "Mamba2"
    assert resolve_kernel_name("mamba2_ssd") == "Mamba2"   # the PR #50 SSD kernel dir
    assert resolve_kernel_name("flash") == "FlashAttention"
    assert resolve_kernel_name("mla") == "MLA"             # short, but exact -> hit


# --- near-miss (embedded descriptor) fallback --------------------------------

def test_embedded_descriptor_resolves():
    # a model whose primitive string embeds a known descriptor but is not itself
    # a map key — the exact lookup misses, the substring fallback catches it.
    assert resolve_kernel_name("qwen3next_gated_delta") == "DeltaNet"
    assert resolve_kernel_name("Mamba2SSDMixer") == "Mamba2"
    assert resolve_kernel_name("my_flash_attention_layer") == "FlashAttention"


def test_longest_descriptor_wins():
    # "gated_delta_linear_attention" contains BOTH "gateddelta" (-> DeltaNet) and
    # the longer "gateddeltalinearattention" (-> KDA); the longest must win.
    assert resolve_kernel_name("model_gated_delta_linear_attention") == "KDA"


def test_op_name_fallback_when_primitive_empty():
    # primitive is empty/unhelpful, but the op name carries the family.
    assert resolve_kernel_name("", op_name="flash_attention_fwd") == "FlashAttention"
    assert resolve_kernel_name("mixer", op_name="selective_scan") == "Mamba2"


# --- safety: short descriptors never match as a loose substring --------------

def test_short_descriptors_only_match_exactly():
    # "mla"/"kda"/"ssm" (len < 6) must NOT match as a substring of an unrelated
    # word — only an exact normalized hit resolves them.
    assert resolve_kernel_name("xmlax") is None          # embeds "mla" -> NOT matched
    assert resolve_kernel_name("assembler") is None      # embeds "ssm" -> NOT matched
    assert resolve_kernel_name("mla") == "MLA"            # exact still works


def test_no_match_returns_none():
    assert resolve_kernel_name("gelu") is None
    assert resolve_kernel_name("layernorm") is None
    assert resolve_kernel_name("") is None
    assert resolve_kernel_name("", op_name="") is None


# --- registry integration -----------------------------------------------------

def _kernel_dir(tmp_path, name="DeltaNet", status="passed-on-device"):
    d = tmp_path / "kernels" / name
    d.mkdir(parents=True)
    (d / "kernel.json").write_text(json.dumps({
        "name": name, "status": status,
        "entry": "deltanet_fwd:kernel", "path": f"{name}/kernel.py",
    }))
    return tmp_path / "kernels"


def test_registry_for_signature_resolves_near_miss(tmp_path):
    reg = KernelRegistry(kernel_dir=_kernel_dir(tmp_path))
    # near-miss primitive still harvests the DeltaNet kernel via the fallback
    spec = reg.for_signature("qwen3next_gated_delta")
    assert spec is not None and spec.name == "DeltaNet" and spec.usable
    # exact primitive resolves identically (back-compat via for_primitive shim)
    assert reg.for_primitive("gated_delta_net").name == "DeltaNet"
    # op-name fallback when primitive is empty
    assert reg.for_signature("", op_name="gated_delta_rule").name == "DeltaNet"
    # an unrelated op resolves to nothing (no spurious harvest)
    assert reg.for_signature("gelu") is None


def test_registry_empty_when_no_kernel_dir():
    # default (no kernel dir, no library) -> every lookup None, so the framework
    # behaves exactly as before and never spuriously harvests.
    reg = KernelRegistry()
    assert reg.for_signature("qwen3next_gated_delta") is None
    assert reg.for_primitive("gated_delta_net") is None
