# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the structural-mutator wire-in of ``InventEngine._optimize_perf``.

Verifies the seam added by the perf-loop-mutator change:
  * with ``perf_use_mutator=True`` (default), the perf loop is driven by the
    STRUCTURAL mutator (kernel_mutator.MutatingAuthor) and the LLM author is
    NOT called during optimization;
  * with ``perf_use_mutator=False``, it falls back to the legacy LLM-author path;
  * with an empty-source seed kernel (a NO-AUTHOR op), it falls back regardless
    of the flag (nothing to mutate);
  * the ledger note records which driver ran.

Pure CPU; the perf loop's measure step is a mock ``race_fn`` (no device)."""

from __future__ import annotations

from invent_engine import InventEngine, RaceResult
from invent_kernels import AuthoredKernel, catalog

# A kernel source with BOTH a bare ``512`` tile literal and a square-sum reduce,
# so kernel_mutator.mutate() has real levers to pull (widen/narrow tile,
# fuse-square-reduce). Must parse as Python (the mutator ast-validates variants).
_MUTATABLE_SRC = '''
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

def k(x, out):
    CHUNK = 512
    for i in nl.static_range(0, x.shape[0], CHUNK):
        t = nl.load(x[i:i + CHUNK])
        ms = nl.sum(t * t, axis=1, keepdims=True)
        out[i:i + CHUNK] = t * nl.reciprocal(ms)
'''


def _seed_kernel(src: str = _MUTATABLE_SRC, entry: str = "k") -> AuthoredKernel:
    spec = catalog()["rmsnorm"]
    return AuthoredKernel(
        op=spec.name, origin="invented", numpy_impl=spec.reference,
        nki_src=src, entry=entry, pipeline_notes="seed")


class _SpyLLM:
    """KernelAuthor-shaped spy: counts .author() calls so a test can prove the
    LLM was (or was not) invoked during the perf loop. Returns the seed kernel
    unchanged so the loop terminates cleanly if it IS used."""

    def __init__(self, seed: AuthoredKernel) -> None:
        self.seed = seed
        self.calls = 0

    def author(self, spec, lessons=None, feedback=None, perf_feedback=None):
        self.calls += 1
        return self.seed


def _faster_each_change(tmp_state):
    """A race_fn that returns a strictly faster latency every time it sees a
    NEW source, so an adopted variant keeps the loop progressing (and a repeated
    source stalls it). Correctness always True. Never converges by roofline, so
    the stop is driven by the mutator exhausting variants / no_gain."""
    def race_fn(kernel, spec):
        src = getattr(kernel, "nki_src", "")
        if src not in tmp_state["seen"]:
            tmp_state["seen"][src] = 1.0 - 0.05 * len(tmp_state["seen"])
        ms = tmp_state["seen"][src]
        return RaceResult(True, correct=True, correctness_pct=100.0,
                          speedup=1.0 / ms, kernel_ms=ms, baseline_ms=1.0,
                          bottleneck="memory_bound", roofline_ratio=0.3, mfu=0.3)
    return race_fn


def test_mutator_drives_perf_loop_and_llm_not_called(tmp_path):
    spec = catalog()["rmsnorm"]
    seed = _seed_kernel()
    spy = _SpyLLM(seed)
    eng = InventEngine(out_dir=tmp_path, author=spy,
                       max_perf_rounds=6, perf_use_mutator=True)
    seed_race = RaceResult(True, correct=True, correctness_pct=100.0,
                           speedup=1.0, kernel_ms=1.0, baseline_ms=1.0,
                           bottleneck="memory_bound", roofline_ratio=0.3, mfu=0.3)
    state = {"seen": {}}
    best_k, best_r, note = eng._optimize_perf(
        spec, seed, seed_race, lessons=None, race_fn=_faster_each_change(state))

    # The LLM author was NOT called during optimization — the mutator drove it.
    assert spy.calls == 0, "LLM author should not run when the mutator is enabled"
    assert "structural-mutator" in note
    # The mutator produced at least one distinct variant that was measured
    # (more than just the seed source raced).
    assert len(state["seen"]) >= 2
    # The kept kernel is a structural mutation (its notes are stamped by
    # MutatingAuthor) OR the seed — never an LLM rewrite.
    assert best_r.correct


def test_flag_off_falls_back_to_llm_author(tmp_path):
    spec = catalog()["rmsnorm"]
    seed = _seed_kernel()
    spy = _SpyLLM(seed)
    eng = InventEngine(out_dir=tmp_path, author=spy,
                       max_perf_rounds=4, perf_use_mutator=False)
    seed_race = RaceResult(True, correct=True, correctness_pct=100.0,
                           speedup=1.0, kernel_ms=1.0, baseline_ms=1.0,
                           bottleneck="memory_bound", roofline_ratio=0.3, mfu=0.3)
    state = {"seen": {}}
    _, _, note = eng._optimize_perf(
        spec, seed, seed_race, lessons=None, race_fn=_faster_each_change(state))
    # Legacy path: the LLM author WAS called (at least once).
    assert spy.calls >= 1
    assert "llm-author" in note


def test_empty_source_seed_falls_back_even_when_flag_on(tmp_path):
    """A NO-AUTHOR op (empty nki_src) has nothing to mutate -> the wire-in falls
    back to the LLM author even with perf_use_mutator=True."""
    spec = catalog()["rmsnorm"]
    seed = _seed_kernel(src="", entry="")
    spy = _SpyLLM(seed)
    eng = InventEngine(out_dir=tmp_path, author=spy,
                       max_perf_rounds=4, perf_use_mutator=True)
    seed_race = RaceResult(True, correct=True, correctness_pct=100.0,
                           speedup=1.0, kernel_ms=1.0, baseline_ms=1.0,
                           bottleneck="memory_bound", roofline_ratio=0.3, mfu=0.3)
    state = {"seen": {}}
    _, _, note = eng._optimize_perf(
        spec, seed, seed_race, lessons=None, race_fn=_faster_each_change(state))
    assert spy.calls >= 1
    assert "llm-author" in note


def test_mutator_default_is_on():
    """The constructor default enables the structural mutator (the on-device
    finding: refinement must be structural, not LLM-prompted)."""
    eng = InventEngine(out_dir="/tmp/_perf_mutator_default_probe")
    assert eng.perf_use_mutator is True


def test_broken_variant_does_not_get_kept(tmp_path):
    """If every mutated variant races INCORRECT, the loop stops
    (regressed_or_broke) and keeps the correct seed — a bad mutation is never
    banked as the best kernel."""
    spec = catalog()["rmsnorm"]
    seed = _seed_kernel()
    spy = _SpyLLM(seed)
    eng = InventEngine(out_dir=tmp_path, author=spy,
                       max_perf_rounds=6, perf_use_mutator=True)
    seed_race = RaceResult(True, correct=True, correctness_pct=100.0,
                           speedup=1.0, kernel_ms=1.0, baseline_ms=1.0,
                           bottleneck="memory_bound", roofline_ratio=0.3, mfu=0.3)

    def _break_on_change(kernel, spec):
        # The seed source stays correct; any mutation races incorrect.
        if getattr(kernel, "nki_src", "") == _MUTATABLE_SRC:
            return RaceResult(True, correct=True, correctness_pct=100.0,
                              speedup=1.0, kernel_ms=1.0, baseline_ms=1.0,
                              bottleneck="memory_bound", roofline_ratio=0.3, mfu=0.3)
        return RaceResult(True, correct=False, correctness_pct=10.0,
                          speedup=2.0, kernel_ms=0.5, baseline_ms=1.0)

    best_k, best_r, note = eng._optimize_perf(
        spec, seed, seed_race, lessons=None, race_fn=_break_on_change)
    # The kept kernel is the correct seed, never the faster-but-broken variant.
    assert best_r.correct
    assert best_k.nki_src == _MUTATABLE_SRC
