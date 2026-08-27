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
    assert rf.PEAK_TFLOPS_BF16_PER_CORE == 380e12


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
    pc = rf.profitability(bytes_moved=0.0, flops=380e9, device_s=380e9 / 380e12,
                          bottleneck="compute_bound")
    assert pc.bottleneck == "compute_bound" and abs(pc.sol - 1.0) < 1e-9


def test_zero_inputs_never_fabricate_a_ratio():
    assert rf.sol_memory_bound(0, 1.0) == 0.0
    assert rf.sol_memory_bound(1e6, 0.0) == 0.0
    assert rf.sol_compute_bound(0, 1.0) == 0.0
