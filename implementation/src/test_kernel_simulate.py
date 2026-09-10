"""Tests for the CPU simulate correctness gate (kernel_simulate.py, R1).

The real ``nki.simulate_kernel`` path only executes on a box where the NKI
simulator is importable (not this laptop / CI), so here we exercise the module's
PLUMBING with an injected fake ``sim_fn`` + fake ``load_entry`` — the same
dependency-injection seam the engine uses for ``race_fn`` / ``compile_fn``. The
honest-deferral contract (no simulator -> ran=False, never a crash or a fake
pass) is verified against this box's real (absent-simulator) state.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from kernel_simulate import (
    SimulateResult,
    simulate_available,
    simulate_kernel_cpu,
)


# A fake loader: the real kernel object is irrelevant to the fake sim_fn, which
# computes from the numpy args directly. Returns a callable so the None-check
# passes.
def _fake_load(nki_src, entry, op):
    return lambda *a, **k: None


def _arr(seed=0):
    return np.random.default_rng(seed).standard_normal((8, 4)).astype(np.float32)


# -- honest deferral (this box has no simulator) -----------------------------

def test_simulate_available_is_false_without_nki():
    """On a box where neither ``nki`` nor ``neuronxcc.nki`` is importable (this
    laptop / CI), the probe is False so every caller cleanly no-ops."""
    have = (importlib.util.find_spec("nki") is not None
            or _spec_ok("neuronxcc.nki"))
    assert simulate_available() == have
    if not have:
        assert simulate_available() is False


def _spec_ok(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def test_defers_when_no_simulator_resolvable():
    """No injected sim_fn + no real simulator on this box => ran=False with a
    self-describing reason (deferred to the on-device gate), never a crash."""
    if simulate_available():
        return  # on a real simulator box this path DOES run; covered on-device
    res = simulate_kernel_cpu(nki_src="src", entry="k", op="op",
                              inputs={"x": _arr()}, reference_out=_arr())
    assert res.ran is False and not res.correct
    assert "off-simulator" in res.reason


def test_empty_source_defers():
    res = simulate_kernel_cpu(nki_src="", entry="", op="op",
                              inputs={"x": _arr()}, reference_out=_arr())
    assert res.ran is False and "no authored kernel source" in res.reason


def test_loader_returns_none_defers():
    res = simulate_kernel_cpu(nki_src="src", entry="k", op="op",
                              inputs={"x": _arr()}, reference_out=_arr(),
                              load_entry=lambda *a: None, sim_fn=lambda *a, **k: None)
    assert res.ran is False and "not loadable" in res.reason


# -- injected-fake plumbing (the CPU-testable path) --------------------------

def test_correct_kernel_passes_at_precise_fp_0():
    x = _arr()
    # identity kernel; reference is the same array => exact match at =0.
    good = lambda kernel, *args, out=None: args[0]
    res = simulate_kernel_cpu(nki_src="s", entry="k", op="identity",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=good)
    assert res.ran and res.correct
    assert res.precise_fp is False and res.max_abs_err == 0.0
    assert res.nonfinite is False and res.precision_artifact is False


def test_algorithm_bug_missed_at_both_precisions():
    x = _arr()
    # off by a constant far above tol at BOTH precisions => algorithm bug.
    bad = lambda kernel, *args, out=None: args[0] + 0.5
    res = simulate_kernel_cpu(nki_src="s", entry="k", op="bug",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=bad)
    assert res.ran and not res.correct
    assert res.precise_fp is True                    # re-ran bit-accurate to confirm
    assert "algorithm bug" in res.reason
    assert res.max_abs_err > 0.4


def test_precision_artifact_passes_only_at_precise_fp_1():
    """Misses at NKI_PRECISE_FP=0 but clears at =1 => flagged a precision
    artifact (a rounding ghost), NOT an algorithm bug — so the repair loop is
    not sent chasing it."""
    x = _arr()

    def sim(kernel, *args, out=None):
        # exact only when the bit-accurate mode is set (the module sets the env).
        if os.environ.get("NKI_PRECISE_FP") == "1":
            return args[0]
        return args[0] + 1e-2            # > 1e-4 tol => miss at =0

    res = simulate_kernel_cpu(nki_src="s", entry="k", op="prec",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=sim)
    assert res.ran and res.correct
    assert res.precise_fp is True and res.precision_artifact is True
    assert "precision" in res.reason


def test_nonfinite_output_is_rejected():
    x = _arr()

    def sim(kernel, *args, out=None):
        y = args[0].copy()
        y.flat[0] = np.nan
        return y

    res = simulate_kernel_cpu(nki_src="s", entry="k", op="nan",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=sim)
    assert res.ran and not res.correct
    assert res.nonfinite is True and "NaN/Inf" in res.reason


def test_out_arg_convention_fallback():
    """A simulator that rejects a returned value (the documented out-arg style)
    is handled by the out= fallback path."""
    x = _arr()

    def sim(kernel, *args, out=None):
        if out is None:
            raise RuntimeError(
                "Returning value from top level nki kernel is not supported")
        out[...] = args[0]
        return None

    res = simulate_kernel_cpu(nki_src="s", entry="k", op="outarg",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=sim)
    assert res.ran and res.correct


def test_missing_arg_in_order_defers():
    x = _arr()
    res = simulate_kernel_cpu(nki_src="s", entry="k", op="op",
                              inputs={"x": x}, reference_out=x,
                              arg_order=["x", "does_not_exist"],
                              load_entry=_fake_load, sim_fn=lambda *a, **k: x)
    assert res.ran is False and "arg missing" in res.reason


def test_simulator_error_is_deferral_not_failure():
    """A simulator that raises a NON-return error => honest deferral (ran=False),
    not a spurious correctness failure."""
    x = _arr()

    def boom(kernel, *args, out=None):
        raise RuntimeError("simulator internal explosion")

    res = simulate_kernel_cpu(nki_src="s", entry="k", op="boom",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=boom)
    assert res.ran is False and "simulate raised" in res.reason


def test_shape_mismatch_is_incorrect():
    x = _arr()
    wrong_shape = lambda kernel, *args, out=None: args[0][:2]
    res = simulate_kernel_cpu(nki_src="s", entry="k", op="shape",
                              inputs={"x": x}, reference_out=x,
                              load_entry=_fake_load, sim_fn=wrong_shape)
    assert res.ran and not res.correct


def test_simulate_result_defaults():
    r = SimulateResult(False)
    assert not r.correct and r.max_abs_err == float("inf")
    assert r.precise_fp is False and r.precision_artifact is False
    assert r.nonfinite is False and r.reason == ""
