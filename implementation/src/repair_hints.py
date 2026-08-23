"""repair_hints.py — a TARGETED error-signature -> imperative-fix map for the
repair loop, aimed at *SDK-API misuse* the LLM author repeats under repair
pressure.

Why this exists (distinct from ``kernel_rewrites``):
  * ``kernel_rewrites`` routes COMPILER-level, model-GRAPH failures (a hostile
    aten/HLO op like ``torch.tril`` or a sort-based top-k) to a graph rewrite, or
    an offline-LINT symptom to a kernel edit. Those are *categories of failure*.
  * THIS map handles a narrower, higher-frequency problem seen in the LIVE
    author loop: the model calls a real NKI SDK function with the WRONG
    signature — e.g. ``nisa.nc_matmul(x)`` with no ``moving`` operand — the
    compiler emits the SAME "missing required argument" error every round, and
    re-feeding the raw stack trace does not correct it. The author preamble
    already documents the correct signatures, but the model ignores prose under
    repair pressure. So when a KNOWN error signature appears we PREPEND a loud,
    imperative correction ("COMPILER SAID X — DO THIS: ...") to the repair
    prompt, in ADDITION to the raw error. This is teaching the real SDK API, not
    reward-hacking: every fix is the *correct* documented call.

Design notes:
  * Matching is robust: each hint carries one or more regex ``patterns`` matched
    (case-insensitive) with ``re.search`` against the raw error text, so it fires
    on the compiler string however it is wrapped (``repr(TypeError(...))``,
    ``device compile failed: ...``, an offline-gate reason, etc.).
  * The map is module-level and trivially extensible: append a ``RepairHint`` to
    ``HINTS``. Keep signatures SPECIFIC enough not to cross-fire, and keep the
    ``fix`` text imperative and concrete (the exact correct call).
  * Seeded ONLY with signatures actually hit on this SDK (nki 0.6.0 /
    neuronx-cc 2.27). Extend as new ones are captured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RepairHint:
    """One known compiler/SDK error signature and the imperative fix to inject.

    ``key``       — stable id (used by the repair loop to tell a NEW hint from a
                    previously-surfaced one when relaxing the stall guard).
    ``patterns``  — regex strings; the hint fires if ANY matches (re.search,
                    IGNORECASE) the raw error text. Robust to wrapping/reordering.
    ``fix``       — the loud, imperative correction. Prepended to the repair
                    prompt as "COMPILER SAID <title> — DO THIS:\\n<fix>".
    ``title``     — short human label of the symptom for the banner line.
    """

    key: str
    title: str
    patterns: tuple[str, ...]
    fix: str


# The map. Ordered most-specific first. Each entry is a REAL signature hit on
# this SDK; the fix is the *correct documented call* the model must use instead.
HINTS: tuple[RepairHint, ...] = (
    RepairHint(
        key="nc_matmul-missing-moving",
        title="nc_matmul called without its required `moving` operand",
        patterns=(
            r"nc_matmul\(\).*missing.*moving",          # exact 0.6.0 TypeError
            r"missing (?:value for )?required (?:positional )?argument.*'moving'",
            r"nc_matmul.*'moving'",
        ),
        fix=(
            "nisa.nc_matmul's REAL signature on this SDK is\n"
            "    nisa.nc_matmul(dst, stationary, moving, ...)\n"
            "`dst` is the FIRST (required) argument: a pre-allocated PSUM tile it\n"
            "WRITES IN PLACE — it computes `dst = stationary.T @ moving` and\n"
            "RETURNS NOTHING (do NOT assign its result). You are missing `moving`\n"
            "because your two positional args bound to (dst, stationary). Fix by\n"
            "allocating a PSUM dst and passing ALL THREE by keyword:\n"
            "    psum = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)\n"
            "    nisa.nc_matmul(dst=psum, stationary=stat, moving=mov)  # writes psum\n"
            "    # ...then use `psum` below; nc_matmul returned None.\n"
            "Shapes: stationary [K, M], moving [K, N] -> dst [M, N]; contraction K\n"
            "is the PARTITION axis (K, M <= 128; moving free dim N <= 512)."
        ),
    ),
    RepairHint(
        key="nc_transpose-missing-data",
        title="nc_transpose signature / size-1-partition transpose failure",
        # Require the nc_transpose token: the bare "missing required argument
        # 'data'" string is NOT specific (nisa.activation also has a `data`
        # arg and emits the identical phrase) — matching it would mis-route.
        patterns=(
            r"nc_transpose\(\).*missing.*data",
            r"nc_transpose.*missing.*required.*argument.*'data'",
            r"nc_transpose.*'data'",
        ),
        fix=(
            "nisa.nc_transpose's REAL signature on this SDK is\n"
            "    nisa.nc_transpose(dst, data, ...)\n"
            "`dst` is the FIRST (required) arg: a pre-allocated tile it WRITES IN\n"
            "PLACE with the transpose of `data`. It RETURNS NOTHING (do NOT assign\n"
            "its result). Allocate dst and pass both by keyword:\n"
            "    dst = nl.ndarray((F, P), dtype=data.dtype, buffer=nl.sbuf)\n"
            "    nisa.nc_transpose(dst=dst, data=src)   # data [P,F] -> dst [F,P]\n"
            "    # ...then use `dst`.\n"
            "P, F each <= 128. ROBUST ALTERNATIVE: the HIGH-LEVEL `nl.transpose(x)`\n"
            "RETURNS a tile (no dst) and is often simpler; or feed an already-[K, N]\n"
            "`moving` operand into nc_matmul so no on-the-fly transpose is needed."
        ),
    ),
    RepairHint(
        key="activation-signature",
        title="nisa.activation called with the wrong signature (dst/op/data, no dtype)",
        patterns=(
            r"activation\(\).*unexpected keyword argument 'dtype'",
            r"activation\(\).*missing.*required argument 'data'",
            r"activation\(\).*(?:unexpected keyword|missing.*required)",
        ),
        fix=(
            "nisa.activation's REAL signature on this SDK is\n"
            "    nisa.activation(dst, op, data, bias=None, scale=1.0,\n"
            "                    reduce_op=None, reduce_res=None, ...)\n"
            "`dst` is FIRST and `data` (the input tile) is the THIRD positional\n"
            "arg — it WRITES `op(scale*data + bias)` INTO dst and RETURNS NOTHING.\n"
            "There is NO `dtype` keyword (cast on the host after .cpu() if needed).\n"
            "Allocate dst and pass by keyword:\n"
            "    dst = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)\n"
            "    nisa.activation(dst=dst, op=nl.square, data=x, reduce_op=nl.add)\n"
            "    # ...then use `dst`; do NOT write `nisa.activation(op=..., ...)`\n"
            "    # without dst/data, and do NOT pass dtype=."
        ),
    ),
    RepairHint(
        key="simplifier-ismp902-host-cast",
        title="NCC_ISMP902 / is_subset Simplifier crash from an in-graph dtype cast",
        patterns=(
            r"NCC_ISMP902",
            r"is_subset",
            r"Simplifier",
        ),
        fix=(
            "This is the neuronx-cc Simplifier crash triggered by fusing a dtype\n"
            "CONVERT into the kernel graph. Do the output dtype cast on the HOST\n"
            "AFTER `.cpu()`, NEVER inside the kernel / on-device:\n"
            "    out = kernel(...)          # keep the on-device dtype\n"
            "    host = out.cpu()           # bare .cpu(), NO dtype cast on device\n"
            "    host = host.to(torch.bfloat16)   # cast on the HOST if needed\n"
            "Remove any `.to(dtype)` / `astype` that runs on the device tile."
        ),
    ),
    RepairHint(
        key="reduction-collapse-1d",
        title="a reduction collapsed a tile to 1-D (tiles must stay >= 2-D)",
        patterns=(
            r"at least 2 dimensions?",
            r"must be (?:2|two)[- ]?dimensional",
            r"1-?D tensor",
            r"rank[ -]?1\b",
            r"collaps\w*",
            r"expected.*2D",
        ),
        fix=(
            "A reduction collapsed a tile to 1-D. SBUF/PSUM tiles must stay >= 2-D.\n"
            "Keep the reduced axis with keepdims so the result is [P, 1], never 1-D:\n"
            "    ms = nl.sum(sq, axis=1, keepdims=True)   # [P, 1], not [P]\n"
            "    mx = nl.max(x,  axis=1, keepdims=True)   # same for max\n"
            "Then broadcast the [P, 1] result back over the free axis."
        ),
    ),
    RepairHint(
        key="broadcast-to-freefn",
        title="tensor-method .broadcast_to(...) does not resolve in 0.6.0",
        patterns=(
            r"has no attribute 'broadcast_to'",
            r"\.broadcast_to\b",
            r"broadcast_to",
        ),
        fix=(
            "The tensor-METHOD `tile.broadcast_to(...)` does not resolve in NKI\n"
            "0.6.0. Use the FREE-FUNCTION form, and note `shape` is KEYWORD-ONLY\n"
            "(a positional shape raises TypeError):\n"
            "    b = nl.broadcast_to(tile, shape=(P, F))   # shape= is required"
        ),
    ),
    RepairHint(
        key="unexpected-partition-broadcast",
        title="Unexpected partition broadcast! (implicit [1,F]/[P,1] broadcast)",
        patterns=(
            r"Unexpected partition broadcast",
            r"partition broadcast",
        ),
        fix=(
            "This fires when arithmetic IMPLICITLY partition-broadcasts a tile —\n"
            "e.g. `t[P,F] + iota[1,F]` or `t[P,F] * col[P,1]`. NKI does NOT allow an\n"
            "implicit partition (first-dim) broadcast. Broadcast the [1,F]/[P,1]\n"
            "operand to the FULL [P,F] shape EXPLICITLY before the op:\n"
            "    iota = nisa.iota(..., dtype=nl.int32)          # nisa.iota, not nl.iota\n"
            "    ib   = nl.broadcast_to(iota, shape=(P, F))     # shape= keyword-only\n"
            "    y    = t + ib                                  # now shapes match\n"
            "Or build a full [P,F] index grid with `nl.mgrid[0:P, 0:F]` directly.\n"
            "(nl.mgrid and nl.arange DO lower fine on their own — the failure is the\n"
            "implicit partition broadcast in the arithmetic, not the index op.)"
        ),
    ),
)


def match_hints(error_log: str) -> list[RepairHint]:
    """Return the hints whose signature appears in ``error_log`` (case-insensitive
    regex search), most-specific (map-order) first. Empty for an empty/None log or
    an unrecognized error — the caller then falls back to raw-error-only feedback
    (unchanged behaviour)."""
    if not error_log:
        return []
    hits: list[RepairHint] = []
    for h in HINTS:
        if any(re.search(p, error_log, re.IGNORECASE) for p in h.patterns):
            hits.append(h)
    return hits


def format_hints(hints: list[RepairHint]) -> str:
    """Render matched hints as a loud, delimited imperative block to PREPEND to
    the repair prompt. Empty string when nothing matched (caller prepends
    nothing)."""
    if not hints:
        return ""
    blocks = []
    for h in hints:
        blocks.append(
            f">>> COMPILER SAID: {h.title} — DO THIS:\n"
            f"{h.fix}"
        )
    return "\n".join(blocks)
