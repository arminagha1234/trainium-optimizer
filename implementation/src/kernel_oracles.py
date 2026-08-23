"""kernel_oracles.py — the CPU-oracle registry: a numpy ground-truth per kernel
primitive, so a kernel's numerics can be validated on ANY box (no Trainium) via
`kernel_validation.verdict`.

An "oracle" is a ``(reference, sim, make_inputs)`` triple:

  * ``reference``   — the numpy GROUND TRUTH the kernel must match. The math,
    derived independently of the kernel.
  * ``sim``         — a SECOND, independently-written numpy implementation of the
    same math (typically kernel-shaped: the layout/algorithm the NKI kernel uses,
    e.g. scatter-free RoPE, or an einsum accumulation of a delta-rule recurrence).
    Comparing ``sim`` against ``reference`` is what makes the offline parity check
    MEAN something.
  * ``make_inputs`` — a deterministic input factory (fixed seed) so the check is
    reproducible.

## Why the "not vacuous" guard exists (the AutoFixer orphan-oracle bug)

The failure this module is built to prevent: an oracle whose ``sim IS its
reference`` (the exact same function object) — or a primitive with NO oracle at
all. Then "sim allclose reference" is comparing a value to ITSELF: it passes
trivially, 100% of the time, and a whole class of kernels sails through the gate
UN-validated. (`invent_engine.offline_gate` documents exactly this: most catalog
recipes reuse ``spec.reference`` verbatim as ``numpy_impl``, so their parity is
vacuous.) So ``Oracle.vacuous`` flags any oracle where ``sim is reference`` or
``sim is None``, and ``audit_oracles()`` surfaces every vacuous/missing primitive
so a silently-skipped class is caught, not shipped.

## Alias resolution

Oracles are keyed by CANONICAL kernel name (matching the corpus / registry:
"DeltaNet", "Mamba2", ...). ``get_oracle(name)`` resolves any primitive spelling
through ``kernel_registry.PRIMITIVE_TO_KERNEL`` (so "gated_delta_net",
"GatedDeltaNet", "gated-delta" all reach the DeltaNet oracle) plus any explicit
aliases a registration declares — the same normalization the registry uses, so
spellings never fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kernel_registry import PRIMITIVE_TO_KERNEL, _norm

RefFn = Callable[[dict], np.ndarray]
InputFn = Callable[[], dict]


@dataclass
class Oracle:
    """A CPU ground-truth triple for one primitive. See module doc."""

    name: str                       # canonical kernel name, e.g. "DeltaNet"
    reference: RefFn
    sim: RefFn | None
    make_inputs: InputFn
    notes: str = ""

    @property
    def vacuous(self) -> bool:
        """True if this oracle cannot actually validate anything: it has no
        independent ``sim`` (``None``) or its ``sim`` IS its ``reference`` (the
        same object), so a parity check would compare a value to itself and pass
        trivially. The orphan-oracle bug this whole module guards against."""
        return self.sim is None or self.sim is self.reference


# canonical-key -> Oracle, and normalized-alias -> canonical-key.
_ORACLES: dict[str, Oracle] = {}
_ALIASES: dict[str, str] = {}


def register_oracle(name: str, reference: RefFn, sim: RefFn | None,
                    make_inputs: InputFn, *, aliases: tuple[str, ...] = (),
                    notes: str = "") -> None:
    """Register an oracle under a canonical ``name`` plus optional ``aliases``.

    All keys are normalized (``kernel_registry._norm``) so spellings collapse the
    same way the registry's do. Registering a vacuous oracle is ALLOWED (it is
    still recorded) — ``audit_oracles()`` is what flags it; we do not silently
    drop it, because a dropped oracle looks identical to a missing one.
    """
    key = _norm(name)
    _ORACLES[key] = Oracle(name=name, reference=reference, sim=sim,
                           make_inputs=make_inputs, notes=notes)
    _ALIASES[key] = key
    for a in aliases:
        _ALIASES[_norm(a)] = key


def get_oracle(name: str) -> Oracle | None:
    """Resolve a primitive/kernel name (any spelling) to its Oracle, or None.

    Resolution order: direct canonical key -> explicit alias -> the
    PRIMITIVE_TO_KERNEL primitive->kernel map (so a primitive descriptor like
    "gated_delta_net" reaches the DeltaNet oracle). None if nothing is registered
    for the resolved kernel (the caller then routes to AUTHOR)."""
    n = _norm(name)
    if n in _ORACLES:
        return _ORACLES[n]
    if n in _ALIASES:
        return _ORACLES.get(_ALIASES[n])
    kname = PRIMITIVE_TO_KERNEL.get(n)      # primitive spelling -> canonical kernel
    if kname:
        kk = _norm(kname)
        if kk in _ORACLES:
            return _ORACLES[kk]
        if kk in _ALIASES:
            return _ORACLES.get(_ALIASES[kk])
    return None


def audit_oracles() -> dict[str, list[str]]:
    """Health report: which registered oracles are VACUOUS, and which kernels
    named in PRIMITIVE_TO_KERNEL have NO oracle at all.

    Returns ``{"vacuous": [...], "missing": [...]}``. Both lists being empty is
    the only "fully covered" state. This is the check that stops a whole primitive
    class from being silently skipped — call it in CI / preflight, not per-run."""
    vacuous = sorted(o.name for o in _ORACLES.values() if o.vacuous)
    covered = set(_ORACLES) | set(_ALIASES)
    missing = sorted({
        kname for kname in PRIMITIVE_TO_KERNEL.values()
        if _norm(kname) not in covered
    })
    return {"vacuous": vacuous, "missing": missing}


# ---------------------------------------------------------------------------
# Built-in oracles. REAL (independent reference/sim) ones reuse the proven
# numpy pairs from invent_kernels so we do not fork the math.
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- gated delta rule (DeltaNet) --------------------------------------------
# A genuine, independently-written pair: the reference builds the recurrence with
# explicit outer products + matmuls; the sim expresses the SAME math with einsum.
# Different code, same result -> a non-vacuous parity check.
def _delta_reference(inp: dict) -> np.ndarray:
    q, k, v, g = inp["q"], inp["k"], inp["v"], inp["g"]     # q,k,v [T,d]; g [T,1]
    T, d = q.shape
    S = np.zeros((d, d), dtype=np.float64)
    out = np.zeros((T, d), dtype=np.float64)
    for t in range(T):
        S = g[t, 0] * S + np.outer(k[t], v[t])              # decay + rank-1 update
        out[t] = q[t] @ S
    return out.astype(np.float32)


def _delta_sim(inp: dict) -> np.ndarray:
    q, k, v, g = inp["q"], inp["k"], inp["v"], inp["g"]
    T, d = q.shape
    S = np.zeros((d, d), dtype=np.float64)
    out = np.zeros((T, d), dtype=np.float64)
    for t in range(T):
        S = g[t, 0] * S + np.einsum("i,j->ij", k[t], v[t])  # einsum vs np.outer
        out[t] = np.einsum("i,ij->j", q[t], S)              # einsum vs @
    return out.astype(np.float32)


def _delta_inputs() -> dict:
    g = _rng(20260822)
    T, d = 32, 16
    return {
        "q": g.standard_normal((T, d)).astype(np.float32),
        "k": g.standard_normal((T, d)).astype(np.float32),
        "v": g.standard_normal((T, d)).astype(np.float32),
        # decay in (0,1) so the recurrence stays bounded.
        "g": (0.5 + 0.4 * g.random((T, 1))).astype(np.float32),
    }


# --- RoPE (dense-attention primitive, reused independent pair) --------------
# invent_kernels already ships an independent reference (strided scatter) vs
# kernel-shaped impl (scatter-free stack/flatten). Reuse them verbatim.
def _register_rope() -> None:
    try:
        from invent_kernels import _rope_impl, _rope_inputs, _rope_reference
    except Exception:  # noqa: BLE001 — optional; DeltaNet oracle is enough alone
        return
    register_oracle(
        "RoPE", _rope_reference, _rope_impl,
        lambda: _rope_inputs(128, 128, 1),
        aliases=("rope", "rope_apply", "rotary"),
        notes="strided-scatter reference vs scatter-free stack/flatten sim")


register_oracle(
    "DeltaNet", _delta_reference, _delta_sim, _delta_inputs,
    # aliases beyond what PRIMITIVE_TO_KERNEL already routes; get_oracle also
    # resolves any PRIMITIVE_TO_KERNEL primitive spelling that maps to DeltaNet.
    aliases=("gated_delta_net", "gated-delta", "delta_rule"),
    notes="gated delta rule: outer/matmul reference vs einsum sim")

_register_rope()
