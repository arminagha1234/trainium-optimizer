"""kernel_simulate.py — the OFFLINE (CPU) correctness gate via ``nki.simulate_kernel``.

R1 of docs/nki-stage-improvement-plan.md. Today ``invent_engine.offline_gate``
validates the math by comparing an op's ``numpy_impl`` to its ``reference`` — but
almost every catalog recipe sets ``numpy_impl = spec.reference``, so that check
is a tautology (a function compared to itself) that validates NOTHING, and ALL
real math validation is deferred to a scarce on-device compile. The result: a
kernel with a math bug is only caught AFTER it has spent Trainium time.

This module closes that gap by running the **actual authored NKI source** on CPU
through the NKI simulator (``nki.simulate_kernel``) and comparing ITS output to
the op's reference. Simulating the real kernel is never a tautology, so it is the
genuine offline math check — and it runs on ANY box, for the price of a CPU
interpret, before any device time is spent.

Honest-deferral contract (mirrors ``invent_engine._device_race``):
  * OFF the simulator box (``neuronxcc.nki`` / ``nki`` not importable, i.e. this
    laptop and CI), or on ANY simulator error, this returns ``ran=False`` with a
    self-describing ``reason`` — NEVER a fabricated pass, NEVER a crash. So on a
    plain CPU box the gate is a transparent no-op and the engine's existing
    tautology-deferral behaviour is byte-for-byte unchanged.
  * The simulate path only truly executes where the NKI simulator is importable.
    There it is bit-exact and authoritative for CORRECTNESS (per the project
    knowledge-bank: ``simulate_kernel`` is the correctness oracle; ``baremetal``
    readback is broken on the native DLC; ``benchmark`` gives only latency).

Because the real execution needs the simulator package (absent on a dev laptop),
the device-activation path here is validated ON-DEVICE, once, separately from the
CPU-testable plumbing (which is exercised via an injected fake in the tests). Any
convention detail that turns out wrong on the simulator box degrades to a
self-describing ``ran=False`` deferral — the same safe fallback as "no nki".

NKI_PRECISE_FP triage (per the public nki_simulator guide): the simulator honours
``NKI_PRECISE_FP``. With ``=0`` (default, fast) it approximates FP; with ``=1`` it
is bit-accurate. We run the algorithm-level pass first and, only if it misses,
re-run bit-accurate — so an ALGORITHM bug (misses in both) is separated from a
PRECISION artifact (misses at =0, clears at =1). The distinction routes the
repair loop to the right fix instead of chasing a rounding ghost.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# fp32 offline parity is a MATH check (does the clever formulation equal the
# reference?), so it is tight — the SAME tolerance invent_engine uses offline.
_SIM_ATOL = 1e-4
_SIM_RTOL = 1e-4


@dataclass
class SimulateResult:
    """Outcome of the CPU simulate correctness gate.

    ``ran`` is False off-simulator (package absent) or on ANY simulate error —
    an honest "deferred to the on-device gate", never a fabricated number. When
    ``ran`` is True, ``correct`` is the authoritative offline math verdict for
    the REAL kernel source (not a numpy_impl tautology).
    """

    ran: bool
    correct: bool = False
    max_abs_err: float = float("inf")
    # Which NKI_PRECISE_FP mode produced the verdict: False == algorithm-level
    # (fast, =0), True == bit-accurate (=1). When a kernel misses at =0 but
    # clears at =1, the miss was a PRECISION artifact, not an algorithm bug.
    precise_fp: bool = False
    # True when the =0 pass missed AND the =1 pass cleared it — i.e. the fast
    # simulation flagged a divergence that is purely a floating-point-precision
    # artifact. Surfaced so the repair loop does NOT chase a rounding ghost.
    precision_artifact: bool = False
    # Any NaN/Inf the kernel produced where the reference is finite — an instant
    # correctness reject (a correct op never manufactures a non-finite value).
    nonfinite: bool = False
    reason: str = ""


def simulate_available() -> bool:
    """True iff an NKI simulator (``simulate_kernel``) is importable on this box.

    Checks both stacks the project has seen: the top-level ``nki`` package and
    ``neuronxcc.nki`` (per knowledge-bank, ``neuronxcc.nki.jit`` produces the
    ``GenericKernel`` that ``simulate_kernel`` consumes). Off-device — this
    laptop, CI — both are absent and every caller cleanly no-ops (``ran=False``).
    """
    for mod in ("neuronxcc.nki", "nki"):
        try:
            if importlib.util.find_spec(mod) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def _resolve_simulate_kernel() -> Callable | None:
    """Resolve a ``simulate_kernel`` callable across the known import paths.

    Order mirrors the knowledge-bank guidance (``neuronxcc.nki`` is the stack
    that carries the simulator) with a top-level ``nki`` fallback. Returns None
    if none is importable — the caller then reports an honest ``ran=False``.
    """
    # (module path, attribute) candidates, most-specific first.
    candidates = (
        ("neuronxcc.nki", "simulate_kernel"),
        ("nki", "simulate_kernel"),
    )
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 — absent/broken import is data, not a crash
            continue
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None


@contextmanager
def _precise_fp(enabled: bool):
    """Temporarily set ``NKI_PRECISE_FP`` around a simulate call, restoring the
    prior value afterwards (so we never leak the bit-accurate mode into an
    unrelated later call)."""
    key = "NKI_PRECISE_FP"
    prev = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _invoke_simulate(sim_fn: Callable, kernel: Callable, args: list,
                     out_template: np.ndarray) -> np.ndarray:
    """Run one ``simulate_kernel`` call, handling both invocation conventions.

    Per knowledge-bank the CPU simulator uses ``simulate_kernel(fn, *inputs,
    out=<preallocated>)`` and the kernel fills ``out`` in place (a top-level
    ``return`` is rejected). But the framework's authored kernels are RETURN-
    style (``_device_race`` takes the returned tensor). So we try the return
    convention first and, only if the simulator rejects a returned value, fall
    back to the ``out=`` convention — mirroring how ``_invoke_kernel`` tolerates
    multiple proven calling shapes. Whichever produces a real array wins.
    """
    # (1) return-style: out = simulate_kernel(fn, *inputs)
    try:
        res = sim_fn(kernel, *args)
        if res is not None:
            return np.asarray(res)
    except Exception as e:  # noqa: BLE001
        # The documented signal that this kernel is out-arg style, not return.
        if "Returning value" not in str(e) and "return" not in str(e).lower():
            raise
    # (2) out-arg style: simulate_kernel(fn, *inputs, out=buf); buf filled.
    buf = np.zeros_like(out_template)
    sim_fn(kernel, *args, out=buf)
    return buf


def _compare(got: np.ndarray, ref: np.ndarray,
             atol: float, rtol: float) -> tuple[bool, float, bool]:
    """(correct, max_abs_err, nonfinite). A NaN/Inf where the reference is finite
    is an instant reject regardless of tolerance."""
    got = np.asarray(got, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if got.shape != ref.shape:
        return False, float("inf"), False
    nonfinite = bool(np.any(~np.isfinite(got) & np.isfinite(ref)))
    if nonfinite:
        return False, float("inf"), True
    max_abs = float(np.max(np.abs(got - ref))) if got.size else 0.0
    ok = bool(np.allclose(got, ref, atol=atol, rtol=rtol))
    return ok, max_abs, False


def simulate_kernel_cpu(
    *,
    nki_src: str,
    entry: str,
    op: str,
    inputs: dict,
    reference_out: np.ndarray,
    arg_order: list[str] | None = None,
    load_entry: Callable[[str, str, str], Any] | None = None,
    sim_fn: Callable | None = None,
    atol: float = _SIM_ATOL,
    rtol: float = _SIM_RTOL,
) -> SimulateResult:
    """Run the authored NKI source on CPU and compare to ``reference_out``.

    Dependency-free by construction (no import of ``invent_engine`` / ``nki`` at
    module load) so it is importable and unit-testable on any box:
      * ``load_entry(nki_src, entry, op) -> callable`` materializes + imports the
        kernel (defaults to ``invent_kernels._load_entry_from_file``, the proven
        file-backed loader). Injectable so tests can supply a plain Python fn.
      * ``sim_fn`` is the ``simulate_kernel`` callable (defaults to the resolved
        real one). Injectable so tests drive a deterministic fake — the CPU-
        testable plumbing path.
      * ``arg_order`` is the positional order the kernel expects (caller passes
        ``invent_engine._arg_order`` to avoid a circular import); falls back to
        ``inputs`` insertion order.

    Returns a ``SimulateResult``; ``ran=False`` (with a reason) whenever the
    simulator is unavailable or anything goes wrong — the honest deferral.
    """
    if not nki_src or not entry:
        return SimulateResult(False, reason="no authored kernel source to simulate")

    sim_fn = sim_fn or _resolve_simulate_kernel()
    if sim_fn is None:
        return SimulateResult(
            False, reason="off-simulator: no nki.simulate_kernel importable "
                          "(CPU simulate deferred to on-device gate)")

    # Load the entry callable from the REAL source (default: the proven
    # file-backed loader, so @nki.jit source introspection works on-device).
    if load_entry is None:
        try:
            from invent_kernels import _load_entry_from_file as load_entry  # type: ignore
        except Exception as e:  # noqa: BLE001
            return SimulateResult(False, reason=f"no kernel loader available: {e!r}")
    try:
        kernel = load_entry(nki_src, entry, op)
    except Exception as e:  # noqa: BLE001 — a load failure is data, not a crash
        return SimulateResult(False, reason=f"kernel load raised: {e!r}")
    if kernel is None:
        return SimulateResult(False, reason="kernel entry not loadable for simulate")

    order = arg_order or list(inputs.keys())
    try:
        args = [np.asarray(inputs[k]) for k in order]
    except KeyError as e:  # arg_order names a missing input — a spec/authoring bug
        return SimulateResult(False, reason=f"simulate arg missing: {e!r}")

    ref = np.asarray(reference_out, dtype=np.float32)

    # Pass 1: algorithm-level (NKI_PRECISE_FP=0, fast). Pass 2 (bit-accurate)
    # runs ONLY if pass 1 missed, to tell an algorithm bug from a precision one.
    def _run(precise: bool) -> tuple[bool, float, bool, str]:
        try:
            with _precise_fp(precise):
                got = _invoke_simulate(sim_fn, kernel, args, ref)
        except Exception as e:  # noqa: BLE001 — any simulate error => honest deferral
            return False, float("inf"), False, f"simulate raised: {e!r}"
        ok, max_abs, nonfinite = _compare(got, ref, atol, rtol)
        return ok, max_abs, nonfinite, ""

    ok0, err0, nf0, why0 = _run(precise=False)
    if why0:
        # The simulator itself errored (not a numeric miss) — deferral, not a fail.
        return SimulateResult(False, reason=why0)
    if ok0:
        return SimulateResult(True, correct=True, max_abs_err=err0, precise_fp=False)
    if nf0:
        # NaN/Inf is an algorithm/uninitialized-read reject; bit-accuracy won't help.
        return SimulateResult(True, correct=False, max_abs_err=err0,
                              precise_fp=False, nonfinite=True,
                              reason="kernel produced NaN/Inf where reference is finite")

    # Missed at =0 → is it precision or algorithm? Re-run bit-accurate.
    ok1, err1, nf1, why1 = _run(precise=True)
    if why1:
        # =1 errored; report the =0 miss honestly (it did run).
        return SimulateResult(True, correct=False, max_abs_err=err0, precise_fp=False,
                              reason=f"simulate miss (max_abs_err={err0:.3e}); "
                                     f"bit-accurate re-run errored: {why1}")
    if ok1:
        return SimulateResult(True, correct=True, max_abs_err=err1, precise_fp=True,
                              precision_artifact=True,
                              reason=f"passed only at NKI_PRECISE_FP=1 "
                                     f"(=0 max_abs_err={err0:.3e}) — precision "
                                     f"artifact, not an algorithm bug")
    return SimulateResult(True, correct=False, max_abs_err=err1, precise_fp=True,
                          nonfinite=nf1,
                          reason=f"algorithm bug: simulate miss at both "
                                 f"NKI_PRECISE_FP=0 ({err0:.3e}) and =1 ({err1:.3e})")
