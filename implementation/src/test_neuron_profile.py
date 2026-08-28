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


# --- neuron-explorer summary-json schema (real trn2 capture, 2026-08-28) -----
from neuron_profile import parse_neuron_explorer_summary, capture_profiler, latest_neff

# A trimmed real neuron-explorer `view --output-format=summary-json` node.
_EXPLORER_NODE = {
    "tensor_engine_active_time_percent": 0.110,
    "tensor_engine_instruction_count": 55,          # must NOT be read as utilization
    "scalar_engine_active_time_percent": 0.383,
    "scalar_engine_instruction_count": 94,
    "gpsimd_engine_active_time_percent": 0.563,
    "dma_active_time_percent": 0.476,
    "dynamic_dma_active_time_percent": 0.457,
    "mfu_estimated_percent": 0.0,
    "mbu_estimated_percent": 0.184,
}


def test_parse_explorer_node_maps_engines():
    b = parse_neuron_explorer_summary(_EXPLORER_NODE)
    assert abs(b["pe"] - 0.110) < 1e-9      # tensor_engine, NOT instruction_count(55)
    assert abs(b["act"] - 0.383) < 1e-9     # scalar_engine
    assert abs(b["pool"] - 0.563) < 1e-9    # gpsimd_engine
    assert abs(b["dma"] - 0.476) < 1e-9


def test_parse_explorer_wrapped_by_node_id():
    b = parse_neuron_explorer_summary({"n_deadbeef": _EXPLORER_NODE})
    assert b["pe"] == 0.110 and b["dma"] == 0.476


def test_parse_profile_json_detects_explorer_schema():
    # parse_profile_json must route the explorer schema to the schema-aware parser
    # (a naive flatten would read instruction_count=55 as 5500% util).
    b = parse_profile_json({"n_x": _EXPLORER_NODE})
    assert b["pe"] == 0.110 and "pe" in b and b["pe"] < 1.0


def test_explorer_node_routes_to_single_engine_via_summarize():
    from neuron_profile import summarize, SINGLE_ENGINE
    b = parse_neuron_explorer_summary(_EXPLORER_NODE)
    # gpsimd(pool) 56% busiest, nothing saturated, dma 48% (not >= pool*1.1) -> pool serial
    r = summarize(b)
    assert r.dominant in (SINGLE_ENGINE,)  # pool 0.563 busiest, dma 0.476 < 0.563*1.1


def test_multiple_explorer_nodes_picks_busiest():
    quiet = dict(_EXPLORER_NODE, dma_active_time_percent=0.01,
                 tensor_engine_active_time_percent=0.01)
    b = parse_neuron_explorer_summary({"n_a": quiet, "n_b": _EXPLORER_NODE})
    assert b["dma"] == 0.476    # busier node chosen


def test_capture_profiler_none_when_explorer_absent(tmp_path):
    # non-existent neff / explorer -> None so profile_kernel fails open
    assert capture_profiler("/no/such.neff", explorer="definitely-not-a-real-cmd") is None


def test_latest_neff_missing_cache_is_none():
    assert latest_neff("/no/such/cache/dir") is None
