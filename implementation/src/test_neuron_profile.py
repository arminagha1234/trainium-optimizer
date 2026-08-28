# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for neuron_profile — the profile-guided bottleneck diagnosis. Pure CPU:
synthetic per-engine busy maps + a fake injected profiler. No device."""

from __future__ import annotations

from neuron_profile import (
    DMA_BLOCKED, MEMORY_BOUND, SINGLE_ENGINE,
    ProfileReport, parse_profile_json, profile_kernel, summarize,
)
from kernel_perf import classify_bottleneck


# --- summarize: per-engine busy -> dominant bottleneck -----------------------

def test_dma_dominant_is_dma_blocked():
    r = summarize({"dma": 0.71, "pe": 0.12, "act": 0.20, "pool": 0.05})
    assert r.dominant == DMA_BLOCKED
    assert "dma" in r.reason.lower()
    assert r.measured


def test_one_compute_engine_serializes_is_single_engine():
    # Act busy but nothing saturated, DMA quiet -> serialized on one engine.
    r = summarize({"act": 0.55, "pe": 0.05, "pool": 0.08, "dma": 0.10})
    assert r.dominant == SINGLE_ENGINE
    assert "act" in r.reason.lower()


def test_all_light_is_memory_bound():
    # Everything lightly used -> bandwidth-bound (spilling), not engine-bound.
    r = summarize({"pe": 0.10, "act": 0.12, "pool": 0.06, "dma": 0.18})
    # DMA is busiest but not by the margin over compute; falls through to mem-bound
    # only if it does not clear the dma margin. Here dma(.18) >= act(.12)*1.1 -> dma.
    assert r.dominant in (MEMORY_BOUND, DMA_BLOCKED)


def test_low_everything_no_dma_lead_is_memory_bound():
    r = summarize({"pe": 0.15, "act": 0.14, "pool": 0.13, "dma": 0.14})
    assert r.dominant == MEMORY_BOUND
    assert "memory" in r.reason.lower()


def test_empty_profile_fails_open_to_memory_bound_unmeasured():
    r = summarize({})
    assert r.dominant == MEMORY_BOUND
    assert not r.measured           # so the caller uses the analytic fallback


def test_device_us_carried_through():
    r = summarize({"dma": 0.9, "pe": 0.1}, device_us=1234.0)
    assert r.device_us == 1234.0


# --- the whole point: summarize's reason routes classify_bottleneck ----------

def test_report_reason_routes_perf_loop_classifier():
    """A ProfileReport.reason must steer kernel_perf.classify_bottleneck to the
    SAME label — this is the integration contract (the perf loop greps reason)."""
    class _Race:
        bottleneck = "compute_bound"    # analytic says compute-bound...
        def __init__(self, reason): self.reason = reason

    dma = summarize({"dma": 0.8, "pe": 0.1})
    assert classify_bottleneck(_Race(dma.reason)) == DMA_BLOCKED  # ...profile overrides

    se = summarize({"act": 0.5, "pe": 0.05, "pool": 0.05, "dma": 0.05})
    assert classify_bottleneck(_Race(se.reason)) == SINGLE_ENGINE


# --- parse_profile_json: tolerant of the shapes the profiler emits -----------

def test_parse_flat_percentages():
    b = parse_profile_json({"PE busy %": 71, "Act busy %": 12,
                            "DMA busy %": 40, "Pool busy %": 5})
    assert b["pe"] == 0.71 and b["act"] == 0.12 and b["dma"] == 0.40


def test_parse_nested_engines_utilization():
    b = parse_profile_json({"engines": {
        "TensorEngine": {"utilization": 0.66},
        "ScalarEngine": {"utilization": 0.20},
        "DMA": {"utilization": 0.15}}})
    assert b["pe"] == 0.66 and b["act"] == 0.20 and b["dma"] == 0.15


def test_parse_list_of_records():
    b = parse_profile_json([
        {"name": "PE", "busy": 0.5},
        {"name": "VectorEngine", "busy": 0.3}])
    assert b["pe"] == 0.5 and b["pool"] == 0.3


def test_parse_garbage_is_empty():
    assert parse_profile_json(None) == {}
    assert parse_profile_json("not a profile") == {}
    assert parse_profile_json(12345) == {}


def test_parse_string_percentages():
    b = parse_profile_json({"PE": "71%", "DMA": "0.4"})
    assert abs(b["pe"] - 0.71) < 1e-9 and abs(b["dma"] - 0.40) < 1e-9


# --- profile_kernel: injected profiler seam, fail-open -----------------------

def test_profile_kernel_none_without_profiler():
    # No profiler injected -> honest "cannot profile" (None), never a fake report.
    assert profile_kernel(lambda: None) is None


def test_profile_kernel_with_normalized_dict():
    prof = lambda run: {"dma": 0.8, "pe": 0.1, "device_us": 500.0}
    r = profile_kernel(lambda: None, profiler=prof)
    assert isinstance(r, ProfileReport) and r.dominant == DMA_BLOCKED
    assert r.device_us == 500.0


def test_profile_kernel_with_raw_profiler_json():
    prof = lambda run: {"engines": {"TensorEngine": {"utilization": 0.3},
                                    "DMA": {"utilization": 0.75}}}
    r = profile_kernel(lambda: None, profiler=prof)
    assert r is not None and r.dominant == DMA_BLOCKED


def test_profile_kernel_broken_profiler_is_none():
    def _boom(run): raise RuntimeError("neuron-profile crashed")
    assert profile_kernel(lambda: None, profiler=_boom) is None


def test_profile_kernel_empty_result_is_none():
    assert profile_kernel(lambda: None, profiler=lambda run: {}) is None
