"""Tests for the symptom-indexed error->fix catalog (kernel_rewrites.py)."""

from __future__ import annotations

from kernel_rewrites import (
    REWRITES,
    Rewrite,
    describe,
    match_error,
    match_ops,
)


def _names(rewrites) -> set[str]:
    return {r.name for r in rewrites}


# -- basic matching ----------------------------------------------------------

def test_match_error_routes_sort_unsupported_to_argmax_rewrite():
    hits = match_error("... [NCC_EVRF029] Operation sort is not supported on trn2 ...")
    assert "topk-sort-to-argmax" in _names(hits)


def test_match_error_empty_log_is_no_match():
    assert match_error("") == []
    assert match_error("a perfectly happy compile, status PASS") == []


def test_match_ops_finds_hostile_op():
    assert "tril-to-const-mask" in _names(match_ops(["aten::tril"]))


def test_describe_is_actionable():
    assert describe([]) == "no known rewrite matched this failure"
    txt = describe(match_error("s2d2_ts_as_valid_elem_count"))
    assert "tril-to-const-mask" in txt


# -- R7: the HBM-OOM entry (grounded in the live soak failure) ---------------

def test_oom_signatures_route_to_the_footprint_rewrite():
    for log in (
        "[measure] worker produced no result (rc=1): OOM / HBM pressure: rc=1",
        "Warning: Neuron OOM: Segment pool state: segments=43 ...",
        "... (function RecordOOMFailure)",
    ):
        assert "hbm-oom-shrink-footprint" in _names(match_error(log)), log


def test_oom_rewrite_does_not_cross_match_the_compiler_bug():
    """NCC_INLA001 'Allocated memory out of bound' is a compiler bug to ESCALATE,
    not the resource OOM — the OOM rewrite must NOT claim it."""
    log = ("[INTERNAL_ERROR] [NCC_INLA001] Unhandled exception ... Allocated "
           "memory out of bound {slice.3.636_sub0}@SB<0,0>(144x960)")
    assert "hbm-oom-shrink-footprint" not in _names(match_error(log))


def test_oom_rewrite_is_model_graph_level():
    oom = next(r for r in REWRITES if r.name == "hbm-oom-shrink-footprint")
    assert oom.applies_at == "model-graph"
    assert "TP" in oom.fix and "bucket" in oom.fix.lower()


# -- catalog integrity -------------------------------------------------------

def test_every_rewrite_is_frozen_and_well_formed():
    for r in REWRITES:
        assert isinstance(r, Rewrite)
        assert r.name and r.summary and r.fix
        assert r.applies_at in ("model-graph", "nki-kernel")
        assert r.confidence in ("low", "medium", "high")
        # error_signatures are the routing key — non-lint entries must have some
        # signal (a signature or a hostile op) to be reachable.
        assert r.error_signatures or r.hostile_ops


def test_rewrite_names_are_unique():
    names = [r.name for r in REWRITES]
    assert len(names) == len(set(names))
