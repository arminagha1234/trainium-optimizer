# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""kernel_baremetal.py — the NATIVE-DLC on-device path: compile-proof (nki.baremetal)
+ latency (nki.benchmark), for boxes that have ``neuronxcc`` but NO ``torch_xla``.

The invent engine's default ``_device_race`` runs the authored ``@nki.jit`` kernel
through the torch_xla bridge (``xm.xla_device`` + ``mark_step``). A native-PyTorch
DLC (TorchNeuron, eager, NO XLA) has no torch_xla, so that path can only DEFER —
which is exactly what we observed on the soak box (torch_xla absent). But such
boxes DO have ``neuronxcc.nki``, whose kernels can run on the NeuronCore via
``nki.baremetal`` (the NEFF builds + executes) and be timed via ``nki.benchmark``.

Per the project knowledge-bank, on the native DLC the device signals split three
ways — and this module supplies the two the CPU simulator cannot:

  * CORRECTNESS          -> ``nki.simulate_kernel`` (CPU, authoritative). That is
                            R1 / ``kernel_simulate`` — NOT here.
  * COMPILE + can-it-run -> ``nki.baremetal``: the NEFF builds and executes on the
                            core. Its host READBACK returns zeros on the native
                            image, so we use it ONLY as a device-VIABILITY proof
                            (it clears neuronx-cc walls like NCC_IBCG901), NEVER
                            for numerics.
  * DEVICE LATENCY       -> ``nki.benchmark``: real NeuronCore latency, needs no
                            readback.

So the honest native-DLC "race" is: correctness from R1 simulate, device-viability
from baremetal, latency (and thus %SOL / a speedup vs a native torch-eager
baseline) from benchmark. The engine's R8 physical-plausibility veto still guards
the speedup, so the mild benchmark-vs-wallclock timing asymmetry cannot bank a
faster-than-physics win.

Honest-deferral contract (mirrors ``kernel_simulate`` / ``_device_race``): off a
native device (no ``neuronxcc.nki``, or torch has no ``neuron`` device), or on ANY
error, every function returns a ``ran=False`` result with a self-describing reason
— never a fabricated number, never a crash. Injectable seams (``baremetal_fn`` /
``benchmark_fn`` / ``load_entry``) make the plumbing unit-testable off-device; the
real ``nki`` calls are validated on the box (this laptop has no nki).
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class NativeRaceResult:
    """Outcome of the native-DLC device path. ``ran`` is False off a native
    device or on any error (an honest deferral). When ``ran`` is True,
    ``compiled`` reports whether ``nki.baremetal`` built + ran the NEFF on the
    core (device-viability), and ``kernel_ms`` is the ``nki.benchmark`` latency
    (0.0 when unmeasured)."""

    ran: bool
    compiled: bool = False
    kernel_ms: float = 0.0
    reason: str = ""


def torch_xla_present() -> bool:
    try:
        return importlib.util.find_spec("torch_xla") is not None
    except (ImportError, ValueError):
        return False


def _neuronxcc_nki_present() -> bool:
    try:
        return importlib.util.find_spec("neuronxcc.nki") is not None
    except (ImportError, ValueError):
        return False


def native_dlc() -> bool:
    """True on a native-PyTorch DLC where the baremetal/benchmark path applies:
    ``neuronxcc.nki`` is importable AND ``torch_xla`` is NOT (so the default
    xla-bridge race can only defer). Off-device (this laptop) -> False, so callers
    cleanly fall through to their existing behaviour."""
    return _neuronxcc_nki_present() and not torch_xla_present()


def _resolve(attr: str) -> Callable | None:
    """Resolve ``nki.<attr>`` (baremetal / benchmark) across the known stacks
    (``neuronxcc.nki`` first, then top-level ``nki``). None if unimportable."""
    for mod_name in ("neuronxcc.nki", "nki"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 — absent/broken import is data, not a crash
            continue
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None


def _native_neuron_device():
    """A native ``torch.device('neuron')`` handle (NOT torch_xla), or None. This
    is how the native-PyTorch backend places tensors on the core; unlike
    ``xm.xla_device()`` it needs no torch_xla."""
    try:
        import torch  # noqa: PLC0415
        # Constructing the device is cheap; a real op later proves it works.
        return torch.device("neuron")
    except Exception:  # noqa: BLE001
        return None


def _call_kernel_variants(fn: Callable, args: list):
    """Invoke a resolved baremetal/benchmark-wrapped callable across the call
    forms different nki versions use, returning its result. Tries the direct
    positional call first (the proven moe_fused invocation shape), then a
    decorator-style ``fn(kernel)(*args)`` is handled by the caller. Raises on a
    genuine failure so the caller can record the reason."""
    return fn(*args)


def run_baremetal(
    *,
    nki_src: str,
    entry: str,
    op: str,
    inputs: dict,
    arg_order: list[str] | None = None,
    load_entry: Callable[[str, str, str], Any] | None = None,
    baremetal_fn: Callable | None = None,
    benchmark_fn: Callable | None = None,
    do_benchmark: bool = True,
) -> NativeRaceResult:
    """Prove the authored kernel compiles + runs on the NeuronCore (nki.baremetal)
    and, optionally, time it (nki.benchmark). Dependency-free and injectable so it
    is unit-testable off-device:
      * ``load_entry(nki_src, entry, op)`` materializes + imports the kernel
        (defaults to ``invent_kernels._load_entry_from_file``).
      * ``baremetal_fn`` / ``benchmark_fn`` default to the resolved ``nki``
        callables; tests inject deterministic fakes.

    Returns a ``NativeRaceResult``; ``ran=False`` (with a reason) whenever the
    native device is unavailable or anything goes wrong — the honest deferral.
    Correctness is NOT judged here (baremetal readback is unreliable on the native
    image) — that stays with R1 ``nki.simulate`` (kernel_simulate)."""
    if not nki_src or not entry:
        return NativeRaceResult(False, reason="no authored kernel source")

    baremetal_fn = baremetal_fn or _resolve("baremetal")
    if baremetal_fn is None:
        return NativeRaceResult(
            False, reason="off native-DLC: no nki.baremetal importable "
                          "(native device race deferred)")

    if load_entry is None:
        try:
            from invent_kernels import _load_entry_from_file as load_entry  # type: ignore
        except Exception as e:  # noqa: BLE001
            return NativeRaceResult(False, reason=f"no kernel loader: {e!r}")
    try:
        kernel = load_entry(nki_src, entry, op)
    except Exception as e:  # noqa: BLE001
        return NativeRaceResult(False, reason=f"kernel load raised: {e!r}")
    if kernel is None:
        return NativeRaceResult(False, reason="kernel entry not loadable")

    order = arg_order or list(inputs.keys())
    try:
        args = [np.asarray(inputs[k]) for k in order]
    except KeyError as e:
        return NativeRaceResult(False, reason=f"baremetal arg missing: {e!r}")

    # (1) COMPILE + RUN proof via nki.baremetal. baremetal is applied to the
    # kernel (decorator/wrapper form) and then invoked; a clean run (no compiler
    # abort) is the device-viability signal. Its returned buffer is IGNORED
    # (readback is zeros on the native image).
    try:
        wrapped = baremetal_fn(kernel)          # decorator/wrapper form
        _call_kernel_variants(wrapped, args)
        compiled = True
    except TypeError:
        # Some versions take (kernel, *args) directly rather than a wrapper.
        try:
            baremetal_fn(kernel, *args)
            compiled = True
        except Exception as e:  # noqa: BLE001
            return NativeRaceResult(False, reason=f"baremetal run failed: {e!r}")
    except Exception as e:  # noqa: BLE001 — a compiler abort is data (kernel not device-viable)
        return NativeRaceResult(True, compiled=False,
                                reason=f"kernel did NOT compile/run on device "
                                       f"(baremetal): {e!r}")

    if not do_benchmark:
        return NativeRaceResult(True, compiled=True, reason="baremetal ran (no benchmark)")

    # (2) LATENCY via nki.benchmark.
    benchmark_fn = benchmark_fn or _resolve("benchmark")
    if benchmark_fn is None:
        return NativeRaceResult(True, compiled=True, kernel_ms=0.0,
                                reason="baremetal ran; nki.benchmark unavailable "
                                       "(latency unmeasured)")
    try:
        kernel_ms = _benchmark_ms(benchmark_fn, kernel, args)
    except Exception as e:  # noqa: BLE001 — a benchmark failure => viable but untimed
        return NativeRaceResult(True, compiled=True, kernel_ms=0.0,
                                reason=f"baremetal ran; benchmark raised: {e!r}")
    return NativeRaceResult(True, compiled=True, kernel_ms=kernel_ms,
                            reason=f"native device: compiled+ran; "
                                   f"benchmark={kernel_ms:.4f}ms")


def _benchmark_ms(benchmark_fn: Callable, kernel: Callable, args: list) -> float:
    """Run nki.benchmark and extract a per-iteration latency in milliseconds,
    tolerant of the return shapes different nki versions use (a float ns/us/ms, a
    dict with a latency/p50 key, or an object with a ``.latency`` attribute).
    Returns 0.0 when nothing parseable comes back."""
    try:
        res = benchmark_fn(kernel)(*args)          # decorator/wrapper form
    except TypeError:
        res = benchmark_fn(kernel, *args)          # direct form
    return _latency_ms_from(res)


def _latency_ms_from(res: Any) -> float:
    """Best-effort ms extraction. nki.benchmark commonly reports microseconds
    (the knowledge-bank quotes 'L50=72µs'); we treat a bare number as µs unless a
    key/attr names the unit. Never raises; 0.0 if unparseable."""
    # dict-like
    if isinstance(res, dict):
        for k in ("latency_ms", "p50_ms", "ms"):
            if k in res:
                return float(res[k])
        for k, div in (("latency_us", 1e3), ("p50_us", 1e3),
                       ("latency_ns", 1e6), ("latency", 1e3), ("p50", 1e3)):
            if k in res:
                return float(res[k]) / div
        return 0.0
    # attr-like
    for attr, div in (("latency_ms", 1.0), ("latency_us", 1e3),
                      ("latency_ns", 1e6), ("latency", 1e3), ("p50", 1e3)):
        v = getattr(res, attr, None)
        if v is not None:
            try:
                return float(v) / div
            except (TypeError, ValueError):
                continue
    # bare number -> assume microseconds (nki.benchmark's usual unit)
    if isinstance(res, (int, float)):
        return float(res) / 1e3
    return 0.0
