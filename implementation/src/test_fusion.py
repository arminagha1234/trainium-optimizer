# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fusion — cross-op fusion group detection + megakernel materialization.
Pure CPU: minimal OpSpecs with real references so the fused reference composes."""

from __future__ import annotations

import numpy as np

from invent_kernels import OpSpec
from fusion import (
    FUSABLE_PAIRS, detect_fusion_groups, fused_spec, rank_fusion_groups,
    select_fusion_targets,
)


def _spec(name, ref):
    ins = lambda: {"x": np.ones((4, 4), dtype=np.float32)}
    return OpSpec(name=name, family="m", shape_class="s", dtype="bf16",
                  reference=ref, offline_inputs=ins, real_inputs=ins)


# ops with names that classify to the intended families
def _rmsnorm():  # -> normalization
    return _spec("rmsnorm", lambda inp: inp["x"] / (np.abs(inp["x"]).mean() + 1e-6))
def _attention():  # -> attention
    return _spec("flash_attention", lambda inp: inp["x"] * 2.0)
def _residual():  # -> elementwise
    return _spec("residual_add", lambda inp: inp["x"] + 1.0)
def _matmul():  # -> matmul
    return _spec("dense_matmul", lambda inp: inp["x"] @ inp["x"])


# --- detection ---------------------------------------------------------------

def test_detects_norm_attention_chain():
    groups = detect_fusion_groups([_rmsnorm(), _attention()])
    assert len(groups) == 1
    assert groups[0].names == ["rmsnorm", "flash_attention"]


def test_detects_maximal_run_capped_at_max_group():
    # rmsnorm -> attention -> residual is a 3-chain (norm->attn, attn->elementwise)
    specs = [_rmsnorm(), _attention(), _residual()]
    groups = detect_fusion_groups(specs, max_group=3)
    assert len(groups) == 1 and groups[0].size == 3
    # cap at 2 -> only the first boundary fuses
    groups2 = detect_fusion_groups(specs, max_group=2)
    assert groups2[0].size == 2


def test_non_adjacent_or_unfusable_not_grouped():
    # two matmuls in a row: (matmul, matmul) is not in FUSABLE_PAIRS
    groups = detect_fusion_groups([_matmul(), _matmul()])
    assert groups == []


def test_single_op_is_not_a_group():
    assert detect_fusion_groups([_attention()]) == []


def test_run_consumed_then_continues():
    # norm->attn (fuse), then a lone matmul, then norm->attn (fuse) again
    specs = [_rmsnorm(), _attention(), _matmul(), _rmsnorm(), _attention()]
    groups = detect_fusion_groups(specs)
    assert len(groups) == 2
    assert all(g.size == 2 for g in groups)


# --- fused_spec: honest composed reference -----------------------------------

def test_fused_reference_composes_members_in_sequence():
    g = detect_fusion_groups([_rmsnorm(), _attention()])[0]
    fs = fused_spec(g)
    assert fs.name == "fused_rmsnorm_flash_attention"
    inp = {"x": np.ones((4, 4), dtype=np.float32)}
    # expected: attention(rmsnorm(x)) = (x/(mean|x|+eps)) * 2
    y0 = inp["x"] / (np.abs(inp["x"]).mean() + 1e-6)
    expected = y0 * 2.0
    assert np.allclose(fs.reference(inp), expected)


def test_fused_spec_inherits_head_inputs_and_notes():
    g = detect_fusion_groups([_rmsnorm(), _attention()])[0]
    fs = fused_spec(g)
    assert "FUSED megakernel" in fs.notes
    assert fs.primitive == "fused"
    assert set(fs.offline_inputs().keys()) == {"x"}


# --- ranking / selection -----------------------------------------------------

def test_chain_feeding_attention_scores_higher_than_plain():
    # norm->attention (weak family) vs matmul->elementwise (standard)
    weak = detect_fusion_groups([_rmsnorm(), _attention()])
    plain = detect_fusion_groups([_matmul(), _residual()])
    from fusion import _group_score
    assert _group_score(weak[0]) > _group_score(plain[0])


def _scan():  # -> scan; used to BREAK a chain (its boundaries are unfusable here)
    return _spec("ssd_scan", lambda inp: inp["x"] + 0.5)


def test_select_fusion_targets_returns_specs_ranked():
    # two distinct groups separated by a scan whose boundaries are unfusable:
    #   [matmul->residual]  |  ssd_scan  |  [rmsnorm->attention]
    specs = [_matmul(), _residual(), _scan(), _rmsnorm(), _attention()]
    targets = select_fusion_targets(specs)
    assert len(targets) == 2
    assert all(t.name.startswith("fused_") for t in targets)
    # the attention-feeding group should rank first (weak-family bonus)
    assert "attention" in targets[0].name


def test_select_fusion_targets_empty_when_nothing_fusable():
    assert select_fusion_targets([_matmul(), _matmul()]) == []


def test_max_targets_cap():
    specs = [_rmsnorm(), _attention(), _matmul(), _rmsnorm(), _attention()]
    assert len(select_fusion_targets(specs, max_targets=1)) == 1
