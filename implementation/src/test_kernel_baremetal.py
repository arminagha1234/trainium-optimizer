"""Tests for the native-DLC device path (kernel_baremetal.py, #1).

The real nki.baremetal/benchmark calls run only on a native-DLC box; here we
exercise the plumbing with injected fakes (the same DI pattern as R1's
kernel_simulate) and verify the honest-deferral contract against this laptop's
real (no-neuronxcc) state.
"""

from __future__ import annotations

import numpy as np

from kernel_baremetal import (
    NativeRaceResult,
    _latency_ms_from,
    native_dlc,
    run_baremetal,
    torch_xla_present,
)


def _load(nki_src, entry, op):
    return lambda *a, **k: None          # a stand-in kernel object


def _arr():
    return np.zeros((8, 4), np.float32)


# -- honest deferral on this box ---------------------------------------------

def test_native_dlc_false_off_device():
    # this laptop has no neuronxcc.nki, so the native path does not apply
    assert native_dlc() is False
    assert isinstance(torch_xla_present(), bool)


def test_run_baremetal_defers_without_baremetal_fn():
    res = run_baremetal(nki_src="s", entry="k", op="op",
                        inputs={"x": _arr()}, load_entry=_load)
    assert res.ran is False and "baremetal" in res.reason


def test_empty_source_defers():
    assert run_baremetal(nki_src="", entry="", op="op",
                         inputs={"x": _arr()}).ran is False


# -- injected-fake plumbing --------------------------------------------------

def test_compiles_and_benchmarks():
    calls = {}

    def fake_baremetal(kernel):
        def run(*args):
            calls["ran"] = True
        return run

    def fake_benchmark(kernel):
        def run(*args):
            return {"latency_us": 72.0}
        return run

    res = run_baremetal(nki_src="s", entry="k", op="op", inputs={"x": _arr()},
                        load_entry=_load, baremetal_fn=fake_baremetal,
                        benchmark_fn=fake_benchmark)
    assert res.ran and res.compiled
    assert abs(res.kernel_ms - 0.072) < 1e-9      # 72us -> 0.072ms
    assert calls.get("ran") is True


def test_compile_failure_is_ran_but_not_compiled():
    def bad_baremetal(kernel):
        def run(*args):
            raise RuntimeError("NCC_IBCG901 too many strides")
        return run

    res = run_baremetal(nki_src="s", entry="k", op="op", inputs={"x": _arr()},
                        load_entry=_load, baremetal_fn=bad_baremetal)
    assert res.ran is True and res.compiled is False
    assert "did NOT compile" in res.reason


def test_benchmark_unavailable_still_device_viable():
    def fake_baremetal(kernel):
        return lambda *args: None

    res = run_baremetal(nki_src="s", entry="k", op="op", inputs={"x": _arr()},
                        load_entry=_load, baremetal_fn=fake_baremetal,
                        benchmark_fn=None)
    # compiled/ran proven; latency just unmeasured (no benchmark on this box)
    assert res.ran and res.compiled and res.kernel_ms == 0.0


def test_missing_arg_defers():
    res = run_baremetal(nki_src="s", entry="k", op="op", inputs={"x": _arr()},
                        arg_order=["x", "missing"], load_entry=_load,
                        baremetal_fn=lambda k: (lambda *a: None))
    assert res.ran is False and "arg missing" in res.reason


# -- latency unit parsing ----------------------------------------------------

def test_latency_ms_parsing():
    assert abs(_latency_ms_from({"latency_ms": 1.5}) - 1.5) < 1e-9
    assert abs(_latency_ms_from({"latency_us": 72.0}) - 0.072) < 1e-9
    assert abs(_latency_ms_from({"latency_ns": 72000.0}) - 0.072) < 1e-9
    assert abs(_latency_ms_from(72.0) - 0.072) < 1e-9     # bare number = microseconds
    assert _latency_ms_from("unparseable") == 0.0
    assert _latency_ms_from(None) == 0.0


def test_native_result_defaults():
    r = NativeRaceResult(False)
    assert not r.compiled and r.kernel_ms == 0.0 and r.reason == ""
