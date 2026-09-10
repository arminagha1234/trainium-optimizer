"""Tests for the silent-perf-regression detectors (silent_perf.py, R25).

Pure functions over a best-effort measurement dict; every detector must (a) fire
with the right severity when its inputs cross the threshold, (b) stay silent when
the kernel is clean, and (c) NOT fire (never crash, never fabricate) when its
inputs are absent from a partial profile. Plus the routing integration: a
finding's route_token, appended to a RaceResult.reason, must make
kernel_perf.classify_bottleneck pick the matching one-dominant-fix lever.
"""

from __future__ import annotations

from types import SimpleNamespace

from silent_perf import (
    REREAD_CRITICAL,
    REREAD_WARN,
    SilentPerfFinding,
    code_size_blowup,
    detect_silent_perf,
    operand_reread,
    relayout_dma,
    roofline_gap,
    route_tokens,
    summary,
    worst_severity,
)


# -- operand re-read multiplier ----------------------------------------------

def test_operand_reread_clean_is_none():
    assert operand_reread(bytes_moved=1.0e6, bytes_needed=1.0e6) is None
    assert operand_reread(bytes_moved=1.4e6, bytes_needed=1.0e6) is None  # < warn


def test_operand_reread_warn_and_critical():
    warn = operand_reread(bytes_moved=2.0e6, bytes_needed=1.0e6)      # 2.0x
    assert warn is not None and warn.severity == "warn" and not warn.critical
    assert abs(warn.metric - 2.0) < 1e-9

    crit = operand_reread(bytes_moved=32.0e6, bytes_needed=1.0e6)     # 32x (the TPU KV bug)
    assert crit is not None and crit.critical and crit.kind == "operand_reread"
    assert "re-read" in crit.detail


def test_operand_reread_nonpositive_is_none():
    assert operand_reread(0.0, 1.0) is None
    assert operand_reread(1.0, 0.0) is None


def test_reread_thresholds_are_ordered():
    assert REREAD_WARN < REREAD_CRITICAL


# -- relayout / transpose DMA share ------------------------------------------

def test_relayout_from_ns_warn_and_critical():
    warn = relayout_dma({"relayout_dma_ns": 15.0, "total_dma_ns": 100.0})   # 15%
    assert warn is not None and warn.severity == "warn"
    crit = relayout_dma({"transpose_dma_ns": 30.0, "total_dma_ns": 100.0})  # 30%
    assert crit is not None and crit.critical
    assert "dma" in crit.route_token          # routes to the DMA lever


def test_relayout_from_counts():
    f = relayout_dma({"relayout_dma_count": 5, "total_dma_count": 20})       # 25%
    assert f is not None and f.critical


def test_relayout_clean_and_missing():
    assert relayout_dma({"relayout_dma_ns": 2.0, "total_dma_ns": 100.0}) is None  # 2% < warn
    assert relayout_dma({}) is None                                          # no inputs
    assert relayout_dma({"relayout_dma_ns": 5.0}) is None                    # no total


# -- code-size / instruction-memory blowup -----------------------------------

def test_code_size_needs_an_explicit_limit():
    # No baked HW constant: without a limit the detector never fires.
    assert code_size_blowup({"code_bytes": 8e6}) is None


def test_code_size_fires_over_limit():
    f = code_size_blowup({"code_bytes": 8e6}, limit_bytes=4e6)               # 2x over
    assert f is not None and f.critical and f.kind == "code_size"
    # limit can also come from the measurement dict
    g = code_size_blowup({"code_bytes": 5e6, "imem_limit_bytes": 4e6})
    assert g is not None and g.critical


def test_code_size_under_limit_is_none():
    assert code_size_blowup({"code_bytes": 2e6}, limit_bytes=4e6) is None


# -- roofline gap ------------------------------------------------------------

def test_roofline_gap_flags_large_gap_only():
    assert roofline_gap({"sol": 0.85}) is None            # near SOL -> no finding
    f = roofline_gap({"sol": 0.10})                       # 90% gap
    assert f is not None and f.kind == "roofline_gap" and f.severity == "warn"
    assert roofline_gap({"sol": 0.0}) is None             # unmeasured -> not a bug
    assert roofline_gap({}) is None


# -- umbrella + helpers ------------------------------------------------------

def test_detect_silent_perf_umbrella_multiple_findings():
    m = {
        "bytes_moved": 10.0e6, "bytes_needed": 1.0e6,     # 10x re-read (critical)
        "relayout_dma_ns": 30.0, "total_dma_ns": 100.0,   # 30% relayout (critical)
        "sol": 0.10,                                      # big roofline gap (warn)
        "code_bytes": 8.0e6, "imem_limit_bytes": 4.0e6,   # 2x code (critical)
    }
    findings = detect_silent_perf(m)
    kinds = {f.kind for f in findings}
    assert kinds == {"operand_reread", "relayout_dma", "roofline_gap", "code_size"}
    assert worst_severity(findings) == "critical"
    assert summary(findings).startswith("silent-perf:")


def test_detect_silent_perf_clean_and_partial():
    assert detect_silent_perf({}) == []                   # nothing to say
    assert worst_severity([]) == "" and summary([]) == ""
    # a partial profile fires only the applicable detector
    only_reread = detect_silent_perf({"bytes_moved": 4e6, "bytes_needed": 1e6})
    assert [f.kind for f in only_reread] == ["operand_reread"]


def test_detect_never_crashes_on_garbage_values():
    # non-numeric / NaN / inf values are ignored, not raised on
    m = {"bytes_moved": "oops", "bytes_needed": None, "sol": float("nan"),
         "relayout_dma_ns": float("inf"), "total_dma_ns": 100.0}
    assert detect_silent_perf(m) == []


# -- routing integration with kernel_perf.classify_bottleneck ----------------

def test_route_tokens_drive_the_perf_loop_lever():
    from kernel_perf import classify_bottleneck, DMA_BLOCKED, MEMORY_BOUND

    relayout = relayout_dma({"relayout_dma_ns": 30.0, "total_dma_ns": 100.0})
    reread = operand_reread(bytes_moved=10e6, bytes_needed=1e6)

    # a race whose reason carries the relayout token -> DMA lever
    race_relayout = SimpleNamespace(bottleneck="", reason=route_tokens([relayout]))
    assert classify_bottleneck(race_relayout) == DMA_BLOCKED

    # a race whose reason carries the re-read token -> MEMORY_BOUND (fuse/resident)
    race_reread = SimpleNamespace(bottleneck="", reason=route_tokens([reread]))
    assert classify_bottleneck(race_reread) == MEMORY_BOUND


def test_route_tokens_empty_when_nothing_routes():
    gap = roofline_gap({"sol": 0.1})            # a finding, but no routing lever
    assert route_tokens([gap]) == ""
    assert route_tokens([]) == ""


def test_finding_is_frozen():
    f = SilentPerfFinding(kind="k", severity="warn", metric=1.0, detail="d")
    try:
        f.metric = 2.0            # frozen dataclass -> should raise
        raised = False
    except Exception:
        raised = True
    assert raised
