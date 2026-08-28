# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for kernel_tournament — tournament authoring. Pure CPU: mock authors +
a mock measure returning controlled RaceResult-likes. No model, no device."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from invent_kernels import AuthoredKernel, OpSpec
from kernel_tournament import (
    DEFAULT_STRATEGIES, Strategy, Tournament, TournamentAuthor,
    strategy_authors,
)


def _spec(name="flash_attention"):
    ref = lambda inp: inp["x"]
    ins = lambda: {"x": np.zeros((8, 8), dtype=np.float32)}
    return OpSpec(name=name, family="x", shape_class="s", dtype="bf16",
                  reference=ref, offline_inputs=ins, real_inputs=ins)


@dataclass
class _Race:
    ran: bool = True
    correct: bool = True
    speedup: float = 1.0
    kernel_ms: float = 1.0


class _FixedAuthor:
    """Authors a kernel tagged with its strategy name so we can trace the winner."""
    def __init__(self, tag): self.tag = tag
    def author(self, spec, lessons=None, feedback=None, perf_feedback=None):
        return AuthoredKernel(op=spec.name, origin="invented",
                              numpy_impl=spec.reference,
                              nki_src=f"# {self.tag}", entry="k",
                              pipeline_notes=self.tag)


# --- Tournament.run: picks correct + fastest ---------------------------------

def test_picks_fastest_correct_candidate():
    authors = [("a", _FixedAuthor("a")), ("b", _FixedAuthor("b")),
               ("c", _FixedAuthor("c"))]
    races = {"# a": _Race(speedup=2.0), "# b": _Race(speedup=5.0),
             "# c": _Race(speedup=3.0)}
    result = Tournament().run(authors, lambda k, s: races[k.nki_src], _spec())
    assert result.winner.strategy == "b"          # highest speedup wins
    assert "3 candidates correct" in result.reason


def test_incorrect_candidates_ineligible():
    authors = [("a", _FixedAuthor("a")), ("b", _FixedAuthor("b"))]
    races = {"# a": _Race(correct=False, speedup=99.0),  # fast but WRONG
             "# b": _Race(correct=True, speedup=1.5)}
    result = Tournament().run(authors, lambda k, s: races[k.nki_src], _spec())
    assert result.winner.strategy == "b"          # the correct one, not the fast-wrong one
    assert "1/2 candidates correct" in result.reason


def test_no_correct_returns_best_effort_not_fabricated():
    authors = [("a", _FixedAuthor("a"))]
    result = Tournament().run(
        authors, lambda k, s: _Race(ran=True, correct=False), _spec())
    assert result.winner is not None              # best-effort for honest gating...
    assert not result.winner.eligible             # ...but NOT marked correct
    assert "0/1" in result.reason


def test_ties_broken_by_lower_latency():
    authors = [("a", _FixedAuthor("a")), ("b", _FixedAuthor("b"))]
    races = {"# a": _Race(speedup=2.0, kernel_ms=3.0),
             "# b": _Race(speedup=2.0, kernel_ms=1.0)}
    result = Tournament().run(authors, lambda k, s: races[k.nki_src], _spec())
    assert result.winner.strategy == "b"          # same speedup, lower ms


def test_measure_crash_is_losing_candidate():
    def measure(k, s):
        if k.nki_src == "# a":
            raise RuntimeError("device blew up")
        return _Race(speedup=2.0)
    authors = [("a", _FixedAuthor("a")), ("b", _FixedAuthor("b"))]
    result = Tournament().run(authors, measure, _spec())
    assert result.winner.strategy == "b"          # crashed candidate cannot win


def test_broken_author_drops_out():
    class _Boom:
        def author(self, *a, **k): raise RuntimeError("author died")
    authors = [("bad", _Boom()), ("good", _FixedAuthor("good"))]
    result = Tournament().run(authors, lambda k, s: _Race(), _spec())
    assert result.winner.strategy == "good"
    assert len(result.candidates) == 1            # the broken one never entered


# --- strategy_authors: diverse directives in the prompt ----------------------

def test_strategy_authors_inject_distinct_directives():
    prompts = {}
    def complete(prompt, **kw):
        # record whichever strategy directive is present
        for s in DEFAULT_STRATEGIES:
            if s.directive[:30] in prompt:
                prompts[s.name] = True
        return "```python\ndef k(x): return x\n```"
    authors = strategy_authors(complete, compose=False)
    assert len(authors) == len(DEFAULT_STRATEGIES)
    for name, author in authors:
        author.author(_spec())
    assert set(prompts) == {s.name for s in DEFAULT_STRATEGIES}


# --- TournamentAuthor: bracket round 1, delegate repair ----------------------

def test_tournament_author_first_round_runs_bracket():
    def complete(prompt, **kw):
        # tag the kernel by which strategy directive is in the prompt
        tag = next((s.name for s in DEFAULT_STRATEGIES if s.directive[:30] in prompt),
                   "none")
        return f"```python\n# {tag}\ndef k(x): return x\n```"
    # measure: make 'memory-first' the fastest
    def measure(k, s):
        return _Race(speedup=(9.0 if "memory-first" in k.nki_src else 1.0))
    ta = TournamentAuthor(complete, measure_fn=measure, compose=False)
    k = ta.author(_spec())
    assert "memory-first" in k.nki_src
    assert ta.last_result and "winner 'memory-first'" in ta.last_result.reason


def test_tournament_author_repair_delegates_to_winner():
    calls = {"n": 0}
    def complete(prompt, **kw):
        calls["n"] += 1
        tag = next((s.name for s in DEFAULT_STRATEGIES if s.directive[:30] in prompt),
                   "none")
        return f"```python\n# {tag}\ndef k(x): return x\n```"
    def measure(k, s):
        return _Race(speedup=(9.0 if "layout-first" in k.nki_src else 1.0))
    ta = TournamentAuthor(complete, measure_fn=measure, compose=False)
    ta.author(_spec())                       # round 1: full bracket
    n_after_bracket = calls["n"]
    # repair round (feedback present) -> ONE call (the winner), not the whole bracket
    from kernel_repair import Feedback
    ta.author(_spec(), feedback=[Feedback(round=1, error_log="some compiler error")])
    assert calls["n"] == n_after_bracket + 1


def test_tournament_author_no_measure_degrades_to_single():
    def complete(prompt, **kw):
        return "```python\ndef k(x): return x\n```"
    ta = TournamentAuthor(complete, measure_fn=None, compose=False)
    k = ta.author(_spec())
    assert k.entry == "k"                    # single author, no bracket, still works


def test_tournament_author_is_kernel_author_protocol():
    from kernel_author import KernelAuthor
    ta = TournamentAuthor(lambda p, **kw: "```python\ndef k(x): return x\n```")
    assert isinstance(ta, KernelAuthor)
