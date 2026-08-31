# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for roofline.py — the %SOL profitability signal. Pure CPU/stdlib.

Numbers are anchored to the on-device measurement (2026-08-27): a single-core
streaming copy sustained ~385 GB/s, and a torch-wallclock rmsnorm reported
~40-65 GB/s (host-bound). Those two regimes are the fixtures below."""

from __future__ import annotations

import roofline as rf


def test_peak_constants_are_the_measured_ones():
    assert rf.PEAK_HBM_BW_PER_CORE == 385e9
    # dense bf16 per NeuronCore-v3, corrected from 380e12 (arch doc + first principles)
    assert rf.PEAK_TFLOPS_BF16_PER_CORE == 79e12
    assert rf.PEAK_TFLOPS_FP8_PER_CORE == 158e12
    assert rf.PEAK_TFLOPS_FP32_PER_CORE == 20e12
    assert rf.PEAK_TFLOPS_BF16_SPARSE_PER_CORE == 316e12
    # dtype-aware ceiling picker
    assert rf.peak_tflops("bf16") == 79e12
    assert rf.peak_tflops("fp8") == 158e12
    assert rf.peak_tflops("fp32") == 20e12
    assert rf.peak_tflops("bf16", sparse=True) == 316e12


def test_memory_bound_sol_at_peak_is_one():
    # a kernel that moves bytes at exactly the measured ceiling -> ~100% SOL
    bytes_moved = 50.3e6              # ~the 49152-F sweep point
    device_s = bytes_moved / rf.PEAK_HBM_BW_PER_CORE
    assert abs(rf.sol_memory_bound(bytes_moved, device_s) - 1.0) < 1e-9


def test_wallclock_regime_reads_as_opportunity():
    # the host-bound torch-wallclock rmsnorm (~55 GB/s) is ~14% of the 385 GB/s
    # ceiling -> a clear opportunity (this is exactly the far-from-SOL case).
    bytes_moved = 67.1e6
    achieved_bw = 55e9
    device_s = bytes_moved / achieved_bw
    sol = rf.sol_memory_bound(bytes_moved, device_s)
    assert 0.10 < sol < 0.20
    p = rf.classify(sol, "memory_bound")
    assert p.verdict == "opportunity" and p.worth_authoring


def test_near_sol_reads_as_skip():
    sol = rf.sol_memory_bound(100e6, 100e6 / (0.85 * rf.PEAK_HBM_BW_PER_CORE))
    p = rf.classify(sol, "memory_bound")
    assert p.verdict == "near_sol"
    assert p.worth_authoring is False       # compiler already near roofline -> skip


def test_marginal_band():
    sol = 0.6
    p = rf.classify(sol, "compute_bound")
    assert p.verdict == "marginal" and p.worth_authoring


def test_compute_bound_sol():
    flops = 380e9
    device_s = flops / rf.PEAK_TFLOPS_BF16_PER_CORE      # -> 100% SOL
    assert abs(rf.sol_compute_bound(flops, device_s) - 1.0) < 1e-9


def test_unknown_is_fail_open_never_pruned():
    # no device measurement -> unknown -> NOT pruned (never skip on missing data)
    p = rf.classify(0.0, "memory_bound", measured=False)
    assert p.verdict == "unknown" and p.worth_authoring
    # a non-positive device latency routes to the same fail-open verdict
    p2 = rf.profitability(bytes_moved=1e6, flops=0.0, device_s=0.0,
                          bottleneck="memory_bound")
    assert p2.verdict == "unknown" and p2.worth_authoring


def test_profitability_picks_right_ceiling():
    # memory-bound path uses bytes; compute-bound path uses flops
    pm = rf.profitability(bytes_moved=385e6, flops=0.0, device_s=385e6 / 385e9,
                          bottleneck="memory_bound")
    assert pm.bottleneck == "memory_bound" and abs(pm.sol - 1.0) < 1e-9
    pc = rf.profitability(bytes_moved=0.0, flops=79e9,
                          device_s=79e9 / rf.PEAK_TFLOPS_BF16_PER_CORE,
                          bottleneck="compute_bound")
    assert pc.bottleneck == "compute_bound" and abs(pc.sol - 1.0) < 1e-9


def test_zero_inputs_never_fabricate_a_ratio():
    assert rf.sol_memory_bound(0, 1.0) == 0.0
    assert rf.sol_memory_bound(1e6, 0.0) == 0.0
    assert rf.sol_compute_bound(0, 1.0) == 0.0


def test_model_mfu_percent_known_value():
    # 1e9 params, 1e6 tok/s over 1 core, bf16 -> 2*1e9*1e6 / 79e12 = 2e15/79e12
    # = 25.3% (as a fraction 0.0253 -> *100).
    mfu = rf.model_mfu_percent(1e9, 1e6, 1, "bf16")
    assert abs(mfu - 100.0 * 2e15 / 79e12) < 1e-6
    # spreading the same work over 2 cores halves the per-core MFU
    assert abs(rf.model_mfu_percent(1e9, 1e6, 2, "bf16")
               - rf.model_mfu_percent(1e9, 1e6, 1, "bf16") / 2) < 1e-9


def test_model_mfu_is_dtype_aware():
    # fp8 has 2x the bf16 FLOP ceiling, so the SAME work is HALF the %SOL of fp8's
    # roofline; fp32 has 1/4 the ceiling, so 4x the %SOL. Holds each run to its own
    # roofline instead of the bf16 one.
    base = rf.model_mfu_percent(1e9, 1e6, 1, "bf16")
    assert abs(rf.model_mfu_percent(1e9, 1e6, 1, "fp8") - base / 2) < 1e-6
    assert abs(rf.model_mfu_percent(1e9, 1e6, 1, "fp32") - base * (79.0 / 20.0)) < 1e-3


def test_model_mfu_percent_bad_inputs_are_zero():
    assert rf.model_mfu_percent(0, 1e6, 1) == 0.0
    assert rf.model_mfu_percent(1e9, 0, 1) == 0.0
    assert rf.model_mfu_percent(1e9, 1e6, 0) == 0.0


def test_is_implausible_mfu_boundary():
    assert rf.is_implausible_mfu(100.0001) is True     # over the ceiling -> void
    assert rf.is_implausible_mfu(100.0) is False        # exactly at the ceiling is allowed
    assert rf.is_implausible_mfu(37.0) is False         # a normal, healthy MFU
    # a tolerance band lets small measurement noise through
    assert rf.is_implausible_mfu(105.0, tol=0.10) is False
    assert rf.is_implausible_mfu(111.0, tol=0.10) is True


def test_implausible_mfu_catches_a_fake_speedup():
    # the ~788x-fake class: a tiny latency implies an impossible model FLOP/s.
    # 35e9 params, but "measured" 5e5 tok/s on ONE core -> way over the ceiling.
    fake = rf.model_mfu_percent(35e9, 5e5, 1, "bf16")
    assert fake > 100.0 and rf.is_implausible_mfu(fake)
