# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for host_path — the host-vs-device axis router + host levers. Pure CPU."""

from __future__ import annotations

from host_path import (
    HostProfile, analyze, from_measurement, optimize_axis,
)


# --- axis routing ------------------------------------------------------------

def test_idle_device_is_host_bound():
    # the bs=1 finding: device 99% idle -> host-bound
    v = analyze(HostProfile(device_busy_frac=0.01, batch_size=1,
                            dispatch_ms=5.0, device_ms=0.05))
    assert v.host_bound and v.axis == "host"
    assert v.device_idle_frac > 0.9
    assert v.recommendations                    # concrete levers offered


def test_busy_device_is_device_bound_defers_to_kernel_optimizer():
    v = analyze(HostProfile(device_busy_frac=0.85, device_ms=2.0, dispatch_ms=0.3))
    assert v.axis == "device" and not v.host_bound
    assert v.recommendations == []              # no host levers; defer to kernels
    assert "kernel optimizer" in v.reason


def test_unknown_when_no_timing():
    v = analyze(HostProfile())
    assert v.axis == "unknown"


def test_busy_derived_from_times_when_frac_absent():
    # device 0.1ms, dispatch 9.9ms -> busy ~1% -> host-bound
    p = HostProfile(device_ms=0.1, dispatch_ms=9.9, batch_size=1)
    assert abs(p.busy - 0.01) < 0.005
    assert analyze(p).axis == "host"


def test_dispatch_dominates_is_host_bound_even_if_busy_moderate():
    # host dispatch >= device time -> host-bound regardless
    v = analyze(HostProfile(device_busy_frac=0.6, device_ms=1.0, dispatch_ms=1.5))
    assert v.axis == "host"


def test_optimize_axis_router():
    assert optimize_axis(HostProfile(device_busy_frac=0.02, dispatch_ms=1.0)) == "host"
    assert optimize_axis(HostProfile(device_busy_frac=0.9, device_ms=1.0)) == "device"
    assert optimize_axis(HostProfile()) == "unknown"


# --- recommendations: the right lever for the condition ----------------------

def test_recompiles_lever_ranked_first():
    v = analyze(HostProfile(device_busy_frac=0.1, dispatch_ms=2.0, recompiles=3))
    assert v.recommendations[0].lever == "eliminate graph recompiles"


def test_marksteps_lever_when_barrier_heavy():
    v = analyze(HostProfile(device_busy_frac=0.1, dispatch_ms=2.0,
                            n_ops=100, n_mark_steps=50))
    assert any("mark_step" in r.lever for r in v.recommendations)


def test_batching_lever_for_bs1_idle_device():
    v = analyze(HostProfile(device_busy_frac=0.05, batch_size=1, dispatch_ms=2.0))
    assert any("batch" in r.lever for r in v.recommendations)


def test_no_batching_lever_when_already_batched():
    v = analyze(HostProfile(device_busy_frac=0.05, batch_size=32, dispatch_ms=2.0))
    assert not any("batch size" in r.lever for r in v.recommendations)


def test_async_lever_when_dispatch_dominates():
    v = analyze(HostProfile(device_busy_frac=0.3, device_ms=1.0, dispatch_ms=2.0))
    assert any("async" in r.lever for r in v.recommendations)


# --- from_measurement: tolerant parse ----------------------------------------

def test_from_measurement_reads_common_keys():
    p = from_measurement({"device_busy": 0.02, "host_ms": 8.0, "device_ms": 0.1,
                          "batch": 1, "recompiles": 2})
    assert p.batch_size == 1 and p.recompiles == 2 and p.dispatch_ms == 8.0
    assert analyze(p).axis == "host"


def test_from_measurement_percentage_busy_normalized():
    p = from_measurement({"device_utilization": 85.0, "device_ms": 2.0})
    assert abs(p.busy - 0.85) < 1e-9


def test_from_measurement_garbage_is_unknown():
    assert analyze(from_measurement("not a dict")).axis == "unknown"
    assert analyze(from_measurement({})).axis == "unknown"
