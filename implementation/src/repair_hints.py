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
            "nisa.nc_matmul RETURNS the result tile — there is NO `dst`/`out` arg.\n"
            "Signature: nisa.nc_matmul(stationary, moving, *, ...) -> tile.\n"
            "`stationary` and `moving` are the ONLY positionals. ASSIGN the return:\n"
            "    psum = nisa.nc_matmul(stat, mov)   # -> [M, N] tile; then use psum\n"
            "You hit 'missing moving' because you passed only one operand — pass\n"
            "BOTH stationary and moving. Do NOT pass a `dst` (a 3rd positional\n"
            "errors 'too many positional arguments'). Shapes: stationary [K, M],\n"
            "moving [K, N] -> [M, N]; contraction K is the PARTITION axis\n"
            "(K, M <= 128; moving free dim N <= 512)."
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
            "nisa.nc_transpose RETURNS the transposed tile — there is NO `dst` arg.\n"
            "Signature: nisa.nc_transpose(data, *, mask=None, dtype=None, ...) -> tile.\n"
            "ASSIGN the return value:\n"
            "    t = nisa.nc_transpose(data=src)   # data [P,F] -> t [F,P]; use `t`\n"
            "Pass `data` (the only positional). Do NOT pass a `dst` (a 2nd positional\n"
            "errors 'too many positional arguments'). P, F each <= 128. The HIGH-LEVEL\n"
            "`nl.transpose(x)` also RETURNS a tile and is often simpler; or feed an\n"
            "already-[K, N] `moving` operand into nc_matmul so no on-the-fly\n"
            "transpose is needed."
        ),
    ),
    RepairHint(
        key="activation-signature",
        title="nisa.activation called with the wrong signature (op/data return-form)",
        patterns=(
            r"activation\(\).*missing.*required argument 'data'",
            r"activation\(\).*(?:unexpected keyword|missing.*required)",
            r"activation\(\).*too many positional",
        ),
        fix=(
            "nisa.activation RETURNS a tile — there is NO `dst`/`out` arg.\n"
            "Signature: nisa.activation(op, data, *, bias=None, scale=1.0,\n"
            "                           reduce_op=None, dtype=None, ...) -> tile.\n"
            "`op` is FIRST and `data` (the input tile) is SECOND (the only two\n"
            "positionals; the rest are keyword-only). It RETURNS `op(scale*data +\n"
            "bias)` — ASSIGN it. `dtype=` IS a valid kwarg. Example:\n"
            "    e = nisa.activation(nl.exp, x, bias=neg_rowmax)   # use `e`\n"
            "Pass op and data (do NOT pass a `dst`). For a FUSED free-axis reduce do\n"
            "NOT use activation(reduce_op=) — that return-form does NOT return the\n"
            "reduction on trn2; use nisa.activation_reduce with a reduce_res OUT-param:\n"
            "    ms = nl.ndarray((P,1), dtype=nl.float32, buffer=nl.sbuf)\n"
            "    nisa.activation_reduce(op=nl.square, data=x, reduce_op=nl.add,\n"
            "                           reduce_res=ms[...])   # ms gets the [P,1] reduce"
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
    RepairHint(
        key="infer-tile-partition-dim",
        title="a tile's first dim is not the partition axis (tile inference failed)",
        patterns=(
            r"Failed to infer tile",
            r"first dimension of the tile is not the partition",
        ),
        fix=(
            "Every SBUF/PSUM tile's FIRST dim is the PARTITION axis (<=128). Shape\n"
            "your loads so dim0 is the partition dim; do NOT put a large feature dim\n"
            "(e.g. H=4096) on dim0 - tile it onto the FREE axis (dim1). E.g. load x as\n"
            "[P<=128, free] and iterate/tile the free axis, keep the partition dim first."
        ),
    ),
    RepairHint(
        key="unexpected-output-dependencies",
        title="output tile has unwritten/partially-indexed dst access",
        patterns=(
            r"Unexpected output dependencies",
            r"missing indices in the dst access",
        ),
        fix=(
            "Every index of the output tile must be written each iteration. If you\n"
            "store into out[..., j] inside a loop over j, ensure ALL j are covered (or\n"
            "store the full 2-D slice at once via nl.store(out[:, a:b], value=tile)).\n"
            "Do not leave dst indices unwritten or partially indexed."
        ),
    ),
    RepairHint(
        key="too-many-positional-return-form",
        title="too many positional arguments (return-form contract / no dst)",
        patterns=(
            r"too many positional arguments",
        ),
        fix=(
            "RETURN-FORM contract: (a) the @nki.jit entry takes EXACTLY the input\n"
            "tensors and RETURNS the output - NO out=/dst= parameter; allocate the\n"
            "output inside with nl.ndarray(shape, dtype, buffer=nl.shared_hbm) and\n"
            "return it. (b) nisa.nc_matmul(stationary, moving) and\n"
            "nisa.nc_transpose(data=...) take NO dst - a 3rd/2nd positional overflows.\n"
            "ASSIGN their RETURN value."
        ),
    ),
    RepairHint(
        key="activation-reduce-return-form-inic902",
        title="NCC_INIC902 NeuronInstComb crash from activation(reduce_op=) return-form",
        patterns=(
            r"NCC_INIC902",
            r"NeuronInstComb",
            r"use_empty",
        ),
        fix=(
            "This neuronx-cc NeuronInstComb crash (NCC_INIC902 / 'use_empty') is the\n"
            "fused-reduce return-form trap: `x = nisa.activation(op, data, reduce_op=)`\n"
            "does NOT return the reduction on trn2 (it returns the FULL activation\n"
            "tile), and the dangling reduce fails to combine. Use activation_REDUCE\n"
            "with a reduce_res OUT-param instead (on-device validated):\n"
            "    acc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)\n"
            "    nisa.activation_reduce(op=nl.square, data=x, reduce_op=nl.add,\n"
            "                           reduce_res=acc[...])   # acc gets [P,1] sum\n"
            "For softmax use op=nl.exp (+ bias=neg_rowmax). If you only need the\n"
            "elementwise activation (no reduce), drop reduce_op entirely:\n"
            "    e = nisa.activation(nl.exp, x, bias=neg_rowmax)   # no reduce_op"
        ),
    ),
    RepairHint(
        key="sfkvectorizer-gist-isfv902",
        title="NCC_ISFV902 SFKVectorizer gist() internal crash (degenerate partition)",
        patterns=(
            r"NCC_ISFV902",
            r"SFKVectorizer",
            r"gist\(\)",
        ),
        fix=(
            "neuronx-cc internal vectorizer crash (NCC_ISFV902) - usually triggered by\n"
            "a SIZE-1-PARTITION tile or a degenerate/trivial reduction. Restructure to\n"
            "carry a GENUINE partition dim (P>=2); avoid [1,F] tiles and single-\n"
            "iteration loops; tile the work so every tile has partition dim >=2."
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
