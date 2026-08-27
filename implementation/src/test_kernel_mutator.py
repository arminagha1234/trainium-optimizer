"""Tests for ``kernel_mutator`` — in-code structural refinement of a winner.

All CPU-only, no Trainium, no model. Covers:
  * each mutation fires only when its pattern is present, changes the source,
    and yields valid Python;
  * every proposed variant is a REFINEMENT of the input (high refinement_ratio)
    — the property the on-device A/B showed the LLM prompt could NOT deliver;
  * ``mutate`` orders variants by the diagnosed bottleneck and dedups;
  * ``MutatingAuthor`` cycles through variants, routes on the trail bottleneck,
    and returns the seed unchanged once exhausted;
  * a ``KernelPerfLoop.run`` smoke test adopts a mutation a mock measure rewards.

Runnable two ways:
    python -m pytest -q test_kernel_mutator.py
    python test_kernel_mutator.py
"""

from __future__ import annotations

import ast
import difflib

import kernel_mutator as KM

# Local template-overlap ratio so this suite does not depend on any other
# branch's module. A mutation is a REFINEMENT (keeps the template) iff overlap
# with the winner is high; a from-scratch rewrite scores low. Threshold matches
# the mutator's design intent.
_REFINEMENT_MIN_RATIO = 0.6


def refinement_ratio(candidate: str, best: str) -> float:
    a, b = (candidate or "").strip(), (best or "").strip()
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.splitlines(), b.splitlines()).ratio()


# A realistic RMSNorm winner: has the square-sum pattern, a delayed-division
# opportunity, and a 512 tile — so all mutation families have something to bite.
_RMSNORM = """import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def rmsnorm_kernel(a, gamma):
    P, F = a.shape
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    t   = nl.load(a[:, :])
    g   = nl.load(gamma[:, :])
    ms  = nl.sum(t * t, axis=1, keepdims=True) * (1.0 / F)
    inv = nl.rsqrt(ms + 1e-6)
    o   = t * nl.broadcast_to(inv, shape=(P, F))
    nl.store(out[:, :], value=o * nl.broadcast_to(g, shape=(P, F)))
    return out"""

_SOFTMAX = """import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def softmax_kernel(x):
    P, F = x.shape
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    t   = nl.load(x[:, :])
    mx  = nl.max(t, axis=1, keepdims=True)
    e   = nisa.activation(nl.exp, t - mx)
    den = nl.sum(e, axis=1, keepdims=True)
    o   = e / den
    n0 = 0
    while n0 < F:
        n1 = min(n0 + 512, F)
        n0 = n1
    nl.store(out[:, :], value=o)
    return out"""


# ---------------------------------------------------------------------------
# individual mutations
# ---------------------------------------------------------------------------
def test_fuse_square_reduce_fires_on_pattern():
    out = KM._fuse_square_reduce(_RMSNORM)
    assert out is not None
    # the validated fused form: activation_reduce with a reduce_res out-param
    assert "nisa.activation_reduce(op=nl.square, data=t, reduce_op=nl.add" in out
    assert "reduce_res=ms_sq[...]" in out
    assert "ms_sq = nl.ndarray((t.shape[0], 1)" in out
    assert "nl.sum(t * t" not in out
    # the surrounding expression (* (1.0 / F)) is preserved, now over the temp
    assert "ms_sq * (1.0 / F)" in out
    ast.parse(out)


def test_fuse_square_reduce_skips_without_nisa_import():
    # a kernel that does not import nisa must NOT get an activation_reduce (would
    # NameError at exec) — the mutation declines rather than emit a broken kernel.
    no_nisa = _RMSNORM.replace("import neuronxcc.nki.isa as nisa\n", "")
    assert KM._fuse_square_reduce(no_nisa) is None


def test_fuse_square_reduce_none_without_pattern():
    assert KM._fuse_square_reduce(_SOFTMAX) is None   # softmax has no t*t sum


def test_delayed_division_fires_on_sum_denominator():
    out = KM._delayed_division(_SOFTMAX)
    assert out is not None
    assert "e * nl.reciprocal(den)" in out
    assert "e / den" not in out
    ast.parse(out)


def test_delayed_division_leaves_scalar_divide_untouched():
    # `1.0 / F` in the rmsnorm mean-square is a SCALAR divide and must NOT be
    # rewritten (the conservative denominator pattern only matches sum/den tiles).
    out = KM._delayed_division(_RMSNORM)
    # rmsnorm has no `X / den`-style tensor divide -> no mutation at all.
    assert out is None


def test_widen_and_narrow_tile():
    wide = KM._widen_tile(_SOFTMAX)
    narrow = KM._narrow_tile(_SOFTMAX)
    assert wide is not None and "1024" in wide and "512" not in wide
    assert narrow is not None and "256" in narrow and "512" not in narrow
    ast.parse(wide); ast.parse(narrow)
    # A source with no 512 tile -> no tile mutation.
    assert KM._widen_tile(_RMSNORM) is None


# ---------------------------------------------------------------------------
# mutate(): every variant is a REFINEMENT (the property the prompt couldn't give)
# ---------------------------------------------------------------------------
def test_every_variant_is_a_refinement_not_a_rewrite():
    for winner in (_RMSNORM, _SOFTMAX):
        variants = KM.mutate(winner)
        assert variants, "expected at least one mutation to apply"
        for mk in variants:
            r = refinement_ratio(mk.nki_src, winner)
            assert r >= _REFINEMENT_MIN_RATIO, (mk.label, r)   # ~1.0 by construction
            assert mk.nki_src != winner                        # but it DID change
            ast.parse(mk.nki_src)                              # and still parses


def test_mutate_dedups_and_skips_noops():
    # A source with nothing to mutate yields no variants (not a list of no-ops).
    trivial = "import neuronxcc.nki as nki\n@nki.jit\ndef k(x):\n    return x"
    assert KM.mutate(trivial) == []
    assert KM.mutate("") == []
    # No duplicate source across variants.
    variants = KM.mutate(_SOFTMAX)
    srcs = [v.nki_src for v in variants]
    assert len(srcs) == len(set(srcs))


def test_mutate_orders_by_bottleneck():
    # With DMA_BLOCKED diagnosed, a tile mutation (its lever) must be proposed
    # before the memory-bound fusion mutations.
    variants = KM.mutate(_SOFTMAX, bottleneck=KM.DMA_BLOCKED)
    labels = [v.label for v in variants]
    assert labels, labels
    first_tile = next((i for i, l in enumerate(labels) if "tile" in l), None)
    first_mem = next((i for i, l in enumerate(labels)
                      if "tile" not in l), len(labels))
    assert first_tile is not None and first_tile < first_mem, labels


# ---------------------------------------------------------------------------
# MutatingAuthor — the perf-loop seam
# ---------------------------------------------------------------------------
def test_author_cycles_then_returns_seed_when_exhausted():
    a = KM.MutatingAuthor(_SOFTMAX, entry="softmax_kernel", op="softmax")
    n_variants = len(KM.mutate(_SOFTMAX))
    assert n_variants >= 2
    served = set()
    for _ in range(n_variants):
        k = a.author_fn([])
        assert k.entry == "softmax_kernel"
        assert k.nki_src != _SOFTMAX          # each is a real variant
        served.add(k.nki_src)
    assert len(served) == n_variants          # no repeats within the queue
    # Exhausted -> seed unchanged (loop reads this as "no lever left").
    last = a.author_fn([])
    assert last.nki_src == _SOFTMAX
    assert "exhausted" in last.pipeline_notes


def test_author_routes_on_trail_bottleneck():
    class _FB:
        bottleneck = KM.DMA_BLOCKED
    a = KM.MutatingAuthor(_SOFTMAX, entry="softmax_kernel")
    k = a.author_fn([_FB()])
    # First proposal under a DMA bottleneck is a tile mutation.
    assert "tile" in k.pipeline_notes, k.pipeline_notes


def test_author_kernelauthor_shim_drives_off_perf_feedback():
    # The KernelAuthor-style .author(...) entrypoint ignores spec/lessons and
    # drives off perf_feedback, mirroring author_fn — so it can be dropped into
    # either seam.
    a = KM.MutatingAuthor(_RMSNORM, entry="rmsnorm_kernel")
    k = a.author(spec=None, perf_feedback=[])
    assert k.entry == "rmsnorm_kernel"
    assert k.nki_src != _RMSNORM
    assert refinement_ratio(k.nki_src, _RMSNORM) >= _REFINEMENT_MIN_RATIO


# ---------------------------------------------------------------------------
# integration: KernelPerfLoop adopts a mutation a mock measure rewards
# ---------------------------------------------------------------------------
def test_perf_loop_adopts_a_rewarded_mutation():
    from kernel_perf import KernelPerfLoop

    class _Race:
        def __init__(self, ms, correct=True):
            self.ran = True
            self.correct = correct
            self.kernel_ms = ms
            self.baseline_ms = 1.0
            self.bottleneck = "memory_bound"
            self.roofline_ratio = 0.2
            self.reason = ""

    seed_race = _Race(0.80)
    author = KM.MutatingAuthor(_RMSNORM, entry="rmsnorm_kernel", op="rmsnorm")
    seed_kernel = author._as_kernel(_RMSNORM, "seed")

    # Mock measure: the fused-square variant is FAST (0.40 ms), everything else
    # is no better than the seed. The loop must adopt the fused variant.
    def measure(kernel):
        if "nisa.activation_reduce(op=nl.square" in kernel.nki_src:
            return _Race(0.40)
        return _Race(0.80)

    loop = KernelPerfLoop(max_rounds=6, min_gain_pct=2.0, stall_patience=3)
    outcome = loop.run(author.author_fn, measure,
                       seed_kernel=seed_kernel, seed_race=seed_race)

    assert outcome.ok
    assert "nisa.activation_reduce(op=nl.square" in outcome.kernel.nki_src   # refined winner
    assert outcome.best_ms == 0.40
    # The adopted kernel is a refinement of the seed, not a rewrite.
    assert refinement_ratio(outcome.kernel.nki_src, _RMSNORM) >= _REFINEMENT_MIN_RATIO


# ===========================================================================
# standalone runner (no pytest required)
# ===========================================================================
def _run_standalone() -> int:
    import inspect
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(fns)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
