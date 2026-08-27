# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
rewrite_dispatcher.py — the compile-error -> graph-rewrite autopilot.

The step past hardcoded per-arch gates in ``neuron_worker``: turn the
``kernel_rewrites`` catalog into a live dispatcher that

  1. runs registered installers pre-emptively (each installer is idempotent
     and self-gates on arch, so calling every one is safe and gives us a
     single fan-out replacing the ad-hoc if-cascade in the worker);
  2. wraps a compile in a bounded retry loop — on failure, matches the
     compiler-log signature against ``REWRITES`` via ``kernel_rewrites.match_error``,
     fires any matching installer that has not yet run, and retries;
  3. leaves the installer registry OPEN — every new rewrite added to
     ``kernel_rewrites.REWRITES`` gets a matching entry here (name -> callable)
     and both the pre-emptive fan-out and the retry loop pick it up
     automatically without any worker edit.

This is Phase A of the autonomy plan: any HF model whose compile trips a
signature already in the catalog is fixed without a human in the loop, and
new signatures compound as installers are registered.

INSTALLER CONTRACT. An installer is ``Callable[[log_fn], bool]``. It MUST be

  * idempotent (safe to call twice — return ``False`` the second time),
  * scoped (patches a specific ``transformers.models.*`` module or similar;
    calling it on an unrelated arch is a no-op that returns ``False``),
  * exception-tolerant at its own boundary (any raise here is caught by the
    dispatcher and treated as "installer refused"; the run continues).

The dispatcher itself NEVER raises out of installer logic — it only re-raises
the original compile error when no installer fixed it. That preserves the
"framework never crashes a run" invariant the worker relies on.

Pure Python; no torch / transformers at import time. Unit-testable on a CPU
box; the two built-in installers guard their torch/transformers imports so
this module is importable without the Neuron stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# The catalog lives at implementation/src/kernel_rewrites.py; the worker's
# fallback pattern (package-relative first, src-on-path second) mirrors here.
try:
    from kernel_rewrites import REWRITES, match_error  # type: ignore
except Exception:  # noqa: BLE001
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from kernel_rewrites import REWRITES, match_error  # type: ignore


# --- installer registry -------------------------------------------------------

# Log sink: matches the worker's ``_log`` signature.
LogFn = Callable[[str], None]

# Installer: (log_fn) -> patched?  True iff this call actually monkey-patched
# something. Idempotent installers return False on the second call.
Installer = Callable[[LogFn], bool]

_INSTALLERS: dict[str, Installer] = {}


def register_installer(rewrite_name: str, fn: Installer, *, replace: bool = False) -> None:
    """Bind a rewrite name (from ``kernel_rewrites.REWRITES``) to a concrete
    installer function. Duplicates raise unless ``replace=True`` (used by tests
    to swap in a mock)."""
    if rewrite_name in _INSTALLERS and not replace:
        raise ValueError(f"installer already registered for {rewrite_name!r}")
    _INSTALLERS[rewrite_name] = fn


def unregister_installer(rewrite_name: str) -> Installer | None:
    """Remove and return an installer, or None if none was registered. Used by
    tests to restore a clean registry between cases."""
    return _INSTALLERS.pop(rewrite_name, None)


def registered() -> list[str]:
    """Sorted list of installer names currently registered."""
    return sorted(_INSTALLERS.keys())


# --- pre-emptive fan-out ------------------------------------------------------

@dataclass
class PreemptiveResult:
    """Return of ``preemptive_install_all``. ``applied`` is the installers that
    actually patched (returned True); ``skipped`` is the ones that ran cleanly
    but no-oped (arch mismatch or already installed); ``errored`` is the ones
    that raised (never fatal — the dispatcher swallows and logs)."""
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errored: dict[str, str] = field(default_factory=dict)


def preemptive_install_all(log_fn: LogFn = print) -> PreemptiveResult:
    """Run every registered installer once, in name-sorted order. Each
    installer self-gates on arch (so calling one for the "wrong" model is a
    silent no-op), which means this is a strict superset of the worker's
    current per-arch if-cascade — same rewrites fire, plus any new registered
    installer picks up on the next model without a worker edit.

    Deduped: installers registered under multiple rewrite names (e.g. the
    qwen3_next bundle is registered under three names because it clears three
    signatures) run once here."""
    result = PreemptiveResult()
    seen: set[int] = set()
    for name in registered():
        inst = _INSTALLERS[name]
        if id(inst) in seen:
            continue
        seen.add(id(inst))
        try:
            if inst(log_fn):
                result.applied.append(name)
            else:
                result.skipped.append(name)
        except Exception as e:  # noqa: BLE001 — never crash a run from an installer
            result.errored[name] = repr(e)
            log_fn(f"rewrite-dispatch: preemptive installer {name} raised {e!r}; skipped")
    return result


# --- retry loop ---------------------------------------------------------------

@dataclass
class RewriteAttempt:
    """One round of the retry loop. Records what the compile said, which
    catalog rewrites matched the error signature, which of those had installers
    that fired this round, and which had no installer registered (a lead)."""
    error: str
    matched: list[str]
    applied: list[str]
    pending: list[str]


def compile_with_rewrite_retry(
    compile_fn: Callable[[], Any],
    log_fn: LogFn = print,
    *,
    max_rounds: int = 3,
) -> tuple[Any, list[RewriteAttempt]]:
    """Run ``compile_fn``, retrying on failures whose error message matches a
    known rewrite signature.

    Loop terminates when
      (a) ``compile_fn`` returns  -> return (result, attempts);
      (b) no rewrite matches the error  -> re-raise the compile error;
      (c) rewrites match but no new installer fires this round  -> re-raise;
      (d) ``max_rounds`` retries exhausted  -> re-raise the last error.

    The bound is deliberately tight (default 3 total retries): an installer
    that CLAIMS to patch but does not fix its symptom would otherwise loop
    forever. The ``fired`` set skips already-run installers on subsequent
    rounds so the same fake fix can never re-fire.

    Args:
      compile_fn: nullary callable that performs the compile / first forward.
        Raises on compile failure; return value is passed through.
      log_fn: worker log sink.
      max_rounds: total attempts CAP (1 initial + up to max_rounds-1 retries).
    """
    attempts: list[RewriteAttempt] = []
    fired: set[str] = set()
    last_error: Exception | None = None
    for rnd in range(max_rounds):
        try:
            result = compile_fn()
            return result, attempts
        except Exception as e:  # noqa: BLE001 — compile errors are opaque
            last_error = e
            err = str(e)
            matches = match_error(err)
            names = [r.name for r in matches]
            applied: list[str] = []
            pending: list[str] = []
            for name in names:
                if name in fired:
                    continue
                inst = _INSTALLERS.get(name)
                if inst is None:
                    pending.append(name)
                    continue
                fired.add(name)  # mark BEFORE call so a raising installer isn't retried
                try:
                    if inst(log_fn):
                        applied.append(name)
                except Exception as ie:  # noqa: BLE001
                    log_fn(f"rewrite-dispatch: installer {name} raised {ie!r}; skipping")
            attempts.append(RewriteAttempt(
                error=err[:400], matched=names, applied=applied, pending=pending,
            ))
            log_fn(
                f"rewrite-dispatch round {rnd + 1}/{max_rounds}: "
                f"matched={names} applied={applied} pending={pending}"
            )
            if not applied:
                # Nothing new fired this round; retrying would run the same graph.
                break
    if last_error is not None:
        raise last_error
    raise RuntimeError("rewrite-dispatch: compile_fn never ran (max_rounds<1)")


# --- built-in installers ------------------------------------------------------

def _install_qwen3_next_bundle(log_fn: LogFn) -> bool:
    """The Qwen3-Next / Qwen3.5 rewrite bundle: router (sort-free argmax) +
    GatedDeltaNet (tril->const-mask) + experts (dense-MoE dispatch). Registered
    under all THREE rewrite names it clears, so any of those signatures
    triggers this once (dedup in the retry loop's ``fired`` set + in
    ``preemptive_install_all``'s ``seen`` set)."""
    try:
        from backends.qwen3_next_rewrites import install_qwen3_next_neuron_rewrites  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            from qwen3_next_rewrites import install_qwen3_next_neuron_rewrites  # type: ignore
        except Exception as e:  # noqa: BLE001
            log_fn(f"rewrite-dispatch: qwen3_next installer import failed ({e!r})")
            return False
    return bool(install_qwen3_next_neuron_rewrites(log_fn))


def _install_moe_router_int64(log_fn: LogFn) -> bool:
    """The generic MoE int64-topk float-view patch. The underlying installer
    self-gates on MoE arch, so this is a no-op for dense models."""
    try:
        from backends.moe_router_patch import install_neuron_safe_moe_topk  # type: ignore
    except Exception:  # noqa: BLE001
        try:
            from moe_router_patch import install_neuron_safe_moe_topk  # type: ignore
        except Exception as e:  # noqa: BLE001
            log_fn(f"rewrite-dispatch: moe_router_patch import failed ({e!r})")
            return False
    try:
        install_neuron_safe_moe_topk(log_fn)
        return True
    except Exception as e:  # noqa: BLE001
        log_fn(f"rewrite-dispatch: moe router int64 patch raised ({e!r})")
        return False


# Register built-ins. The qwen3_next bundle is registered under each of the
# three catalog names it clears — same installer, three keys, deduped by id().
for _bundle_key in ("topk-sort-to-argmax", "tril-to-const-mask", "dense-moe-static-dispatch"):
    register_installer(_bundle_key, _install_qwen3_next_bundle)
register_installer("int64-topk-to-float-view", _install_moe_router_int64)


# --- introspection ------------------------------------------------------------

def catalog_coverage() -> dict[str, bool]:
    """Return {rewrite_name: has_installer?} across the whole catalog. A False
    entry is a lead: a known compile-hostile pattern with no auto-fix yet."""
    return {r.name: r.name in _INSTALLERS for r in REWRITES}
