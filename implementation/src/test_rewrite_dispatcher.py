# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for backends/rewrite_dispatcher — the compile-error -> graph-rewrite
autopilot. Pure CPU; no torch / transformers. Every test restores the global
installer registry so cases do not leak into each other."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backends"))
sys.path.insert(0, os.path.dirname(__file__))

from backends import rewrite_dispatcher as rd  # noqa: E402


# --- fixtures -----------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the module-global installer registry around each
    test so registrations/replacements never leak."""
    saved = dict(rd._INSTALLERS)
    try:
        yield
    finally:
        rd._INSTALLERS.clear()
        rd._INSTALLERS.update(saved)


def _log_sink():
    lines: list[str] = []
    return lines, lines.append


# --- registry -----------------------------------------------------------------

def test_builtins_registered():
    """The three qwen3_next signatures + the int64 patch are wired at import."""
    names = set(rd.registered())
    assert {"topk-sort-to-argmax", "tril-to-const-mask",
            "dense-moe-static-dispatch", "int64-topk-to-float-view"} <= names


def test_qwen3_bundle_is_one_installer_under_three_names():
    """The bundle is the SAME callable under three keys (dedup relies on this)."""
    a = rd._INSTALLERS["topk-sort-to-argmax"]
    b = rd._INSTALLERS["tril-to-const-mask"]
    c = rd._INSTALLERS["dense-moe-static-dispatch"]
    assert a is b is c


def test_register_duplicate_raises_without_replace():
    rd.register_installer("x-mock", lambda log: True)
    with pytest.raises(ValueError):
        rd.register_installer("x-mock", lambda log: True)
    # replace=True overwrites cleanly
    rd.register_installer("x-mock", lambda log: False, replace=True)
    assert rd._INSTALLERS["x-mock"](print) is False


def test_unregister_returns_and_removes():
    rd.register_installer("y-mock", lambda log: True)
    fn = rd.unregister_installer("y-mock")
    assert callable(fn)
    assert rd.unregister_installer("y-mock") is None


# --- pre-emptive fan-out ------------------------------------------------------

def test_preemptive_dedupes_same_installer():
    """An installer registered under N names runs ONCE in the fan-out."""
    calls = {"n": 0}

    def inst(log):
        calls["n"] += 1
        return True

    rd._INSTALLERS.clear()
    rd.register_installer("a", inst)
    rd.register_installer("b", inst)
    res = rd.preemptive_install_all(print)
    assert calls["n"] == 1
    # exactly one of the two names is credited as applied; the run is dedup'd
    assert len(res.applied) == 1


def test_preemptive_classifies_applied_skipped_errored():
    rd._INSTALLERS.clear()
    rd.register_installer("applies", lambda log: True)
    rd.register_installer("skips", lambda log: False)

    def boom(log):
        raise RuntimeError("nope")

    rd.register_installer("errors", boom)
    lines, sink = _log_sink()
    res = rd.preemptive_install_all(sink)
    assert res.applied == ["applies"]
    assert res.skipped == ["skips"]
    assert "errors" in res.errored and "nope" in res.errored["errors"]
    assert any("errors" in ln for ln in lines)


# --- retry loop ---------------------------------------------------------------

def test_compile_succeeds_first_try_no_attempts():
    ok, attempts = rd.compile_with_rewrite_retry(lambda: "NEFF", print)
    assert ok == "NEFF"
    assert attempts == []


def test_matched_installer_fires_then_compile_succeeds():
    """First compile raises with a known signature; installer fires; retry ok."""
    rd._INSTALLERS.clear()
    fired = {"n": 0}

    def inst(log):
        fired["n"] += 1
        return True

    rd.register_installer("topk-sort-to-argmax", inst)

    state = {"tries": 0}

    def compile_fn():
        state["tries"] += 1
        if state["tries"] == 1:
            raise RuntimeError("NCC_EVRF029: Operation sort is not supported on trn2")
        return "NEFF"

    ok, attempts = rd.compile_with_rewrite_retry(compile_fn, print, max_rounds=3)
    assert ok == "NEFF"
    assert fired["n"] == 1
    assert len(attempts) == 1
    assert "topk-sort-to-argmax" in attempts[0].applied


def test_no_match_reraises_immediately():
    """An unknown compile error is re-raised without retry."""
    state = {"tries": 0}

    def compile_fn():
        state["tries"] += 1
        raise RuntimeError("some totally novel error nobody catalogued")

    with pytest.raises(RuntimeError, match="novel error"):
        rd.compile_with_rewrite_retry(compile_fn, print, max_rounds=3)
    assert state["tries"] == 1  # no retry — nothing matched


def test_match_but_no_installer_is_a_lead_not_a_retry():
    """A signature that matches the catalog but has NO registered installer
    is recorded as pending and does NOT cause an infinite retry."""
    rd._INSTALLERS.clear()  # strip all installers -> every match is a lead
    state = {"tries": 0}

    def compile_fn():
        state["tries"] += 1
        raise RuntimeError("NCC_EVRF029: Operation sort is not supported on trn2")

    with pytest.raises(RuntimeError, match="EVRF029"):
        rd.compile_with_rewrite_retry(compile_fn, print, max_rounds=5)
    assert state["tries"] == 1
    # (the attempt list isn't returned on re-raise, but the single try proves
    #  we did not spin: no installer fired -> break -> re-raise)


def test_fake_fix_cannot_loop_forever():
    """An installer that claims to patch (returns True) but never fixes the
    error fires ONCE, then is skipped -> bounded by 'no new applied' break."""
    rd._INSTALLERS.clear()
    fired = {"n": 0}

    def liar(log):
        fired["n"] += 1
        return True  # claims success but compile keeps failing

    rd.register_installer("topk-sort-to-argmax", liar)
    state = {"tries": 0}

    def compile_fn():
        state["tries"] += 1
        raise RuntimeError("NCC_EVRF029: Operation sort is not supported on trn2")

    with pytest.raises(RuntimeError, match="EVRF029"):
        rd.compile_with_rewrite_retry(compile_fn, print, max_rounds=5)
    assert fired["n"] == 1           # installer fired exactly once
    assert state["tries"] == 2       # initial + 1 retry, then break (no new fire)


def test_max_rounds_cap_with_multiple_installers():
    """Two distinct matching installers -> at most 2 retries even if each keeps
    'succeeding', capped by max_rounds."""
    rd._INSTALLERS.clear()

    # craft an error that matches two DIFFERENT catalog signatures
    err = ("AwsNeuronTopK reject AND NCC_EVRF029: Operation sort is not "
           "supported on trn2")
    fired = []

    rd.register_installer("int64-topk-to-float-view",
                          lambda log: (fired.append("int64"), True)[1])
    rd.register_installer("topk-sort-to-argmax",
                          lambda log: (fired.append("sort"), True)[1])

    state = {"tries": 0}

    def compile_fn():
        state["tries"] += 1
        raise RuntimeError(err)

    with pytest.raises(RuntimeError):
        rd.compile_with_rewrite_retry(compile_fn, print, max_rounds=3)
    # both installers fire in round 1 (both match), so round 2 has nothing new
    # to apply -> break. initial + 1 retry = 2 compile attempts.
    assert set(fired) == {"int64", "sort"}
    assert state["tries"] == 2


def test_installer_raise_in_retry_is_swallowed_and_marked_fired():
    """A raising installer during retry is caught, marked fired (won't re-run),
    and since nothing was 'applied' the loop breaks and re-raises."""
    rd._INSTALLERS.clear()
    fired = {"n": 0}

    def boom(log):
        fired["n"] += 1
        raise RuntimeError("installer internal error")

    rd.register_installer("topk-sort-to-argmax", boom)
    state = {"tries": 0}

    def compile_fn():
        state["tries"] += 1
        raise RuntimeError("NCC_EVRF029: sort is not supported on trn2")

    lines, sink = _log_sink()
    with pytest.raises(RuntimeError, match="sort is not supported"):
        rd.compile_with_rewrite_retry(compile_fn, sink, max_rounds=5)
    assert fired["n"] == 1
    # a raising installer contributes nothing to `applied`, so the loop breaks
    # right after the first failure -> compile_fn ran once (no retry).
    assert state["tries"] == 1
    assert any("internal error" in ln for ln in lines)


# --- coverage introspection ---------------------------------------------------

def test_catalog_coverage_flags_leads():
    """Every catalog rewrite reports has-installer true/false. The four with
    built-in installers are True; the lint-* / dynamic-slice ones are leads."""
    cov = rd.catalog_coverage()
    assert cov["topk-sort-to-argmax"] is True
    assert cov["int64-topk-to-float-view"] is True
    # a model-graph lead with no auto-fix installer yet
    assert cov["dynamic-slice-to-static-bucket"] is False
    # nki-kernel lint rewrites are not graph installers
    assert cov["lint-arange-to-mgrid"] is False
