# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""perf_hints.py — a TARGETED perf-symptom -> specific-optimization map for the
PERF loop, the analogue of ``repair_hints`` for SPEED (not correctness).

Why this exists (distinct from ``kernel_perf._GUIDANCE``):
  * ``kernel_perf._GUIDANCE`` maps the THREE coarse bottleneck labels
    (memory_bound / single_engine / dma_blocked) to ONE fix each — the safe
    default lever. That is enough to route a slow kernel, but it is generic
    ("fuse", "overlap", "double-buffer").
  * THIS map is the finer grain: it turns a measured symptom (a profile-threshold
    breach + the op shape) into the SPECIFIC named optimization from the NKI
    Performance Guide — e.g. "matmul with a short dimension and low MFU -> Opt #7
    fast weight load: map the short tensor to MOVING (up to 4x)", "per-element
    scan -> Opt #5a nisa.tensor_tensor_scan", "two sub-128-partition reduces ->
    Opt #5b partition vectorization (2x)". These are the levers that BEAT the
    compiler on the ops where it is weak — the compiler already does the coarse
    fusion/tiling/DMA-sizing, so the win is the low-level `nki.isa` trick it does
    not find.

    Source: NKI Kernel Performance Guide (Opt #1-#10) —
    awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki_perf_guide.html
    banked at docs/nki-perf-guide.md.

How it plugs in (mirrors ``repair_hints``):
  * ``symptoms_from(...)`` builds a lowercase symptom haystack from the coarse
    bottleneck label + the profiler's threshold breaches (``ProfileReport`` tokens
    via ``neuron_profile.perf_symptoms``) + the op name/family. It is PURE.
  * ``match_perf_hints(haystack)`` returns the hints whose trigger tokens appear —
    most-specific first — and ``format_perf_hints`` renders the loud imperative
    block the perf loop PREPENDS to the re-author guidance. When nothing specific
    matches, the caller falls back to ``kernel_perf.guidance_for`` (the coarse
    lever), so behaviour degrades gracefully — never worse than today.

Everything here is CPU-pure and unit-testable; no device, no profiler dependency
(the tokens are strings). Extend by appending a ``PerfHint`` to ``HINTS``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PerfHint:
    """One perf symptom and the specific NKI-Guide optimization to apply.

    ``key``      — stable id (lets the perf loop tell a NEW hint from an
                   already-surfaced one when relaxing its stall guard).
    ``opt``      — the NKI-Guide optimization number this teaches (e.g. "Opt #7").
    ``title``    — short human label of the symptom + lever (banner line).
    ``triggers`` — regex strings; the hint fires if ANY matches (re.search,
                   IGNORECASE) the symptom haystack. Keep specific enough not to
                   cross-fire.
    ``fix``      — the loud, imperative correction with the exact call, prepended
                   to the perf re-author prompt.
    ``technique``— the ``nki_knowledge.TECHNIQUES`` key this corresponds to (so a
                   test can assert the knowledge stays DRY / the key exists).
    """

    key: str
    opt: str
    title: str
    triggers: tuple[str, ...]
    fix: str
    technique: str = ""


# Ordered most-specific first. Each fix is a REAL lever from the NKI Perf Guide;
# the numbers (up to 4x, 2x, ~100-cycle overhead, >=32 KiB, ~1024 free-dim) are
# the guide's own.
HINTS: tuple[PerfHint, ...] = (
    PerfHint(
        key="indirect-gather-static-or-matmul",
        opt="GpSimd/indirect",
        title="GpSimd-bound: an indirect/dynamic gather serializes while TensorE idles",
        triggers=(
            r"gpsimd[-_ ]?bound", r"indirect[-_ ]?dma", r"indirect[-_ ]?gather",
            r"dynamic[-_ ]?dma", r"\binterpolate\b", r"grid[-_ ]?sample",
            r"\bgather\b", r"\bupsample\b", r"\bresample\b",
        ),
        fix=(
            "MAKE THE ADDRESSING STATIC, then EXPRESS THE GATHER AS A MATMUL. GpSimd\n"
            "is serializing an indirect/dynamic gather on its slow integrated DMA\n"
            "(~153 GB/s/dir) while TensorE sits idle — proven on UniVR: 758.9->106 ms\n"
            "(7x) from static addressing, then ->55.6 ms (1.92x, TensorE 2.4%->67%)\n"
            "from the matmul form, 13x cumulative. Two levers:\n"
            "  1. If the sampling indices are DATA-INDEPENDENT at trace time, compute\n"
            "     them on the HOST and bake them as COMPILE-TIME CONSTANTS so every\n"
            "     DMA is static-addressed (frees GpSimd, streams at HBM peak).\n"
            "  2. If the resample is LINEAR (bilinear/interpolate/upsample/one-hot),\n"
            "     rewrite it as `out = W @ x` with a precomputed mostly-ZERO weight\n"
            "     matrix (bilinear = 4 corner taps/row) — one static matmul on\n"
            "     TensorE instead of an indirect gather on GpSimd.\n"
            "ONLY for STATIC access patterns — NOT data-dependent gathers (KV-cache\n"
            "paging, MoE token dispatch), whose indices depend on runtime values."
        ),
        technique="gather-as-matmul",
    ),
    PerfHint(
        key="fast-weight-load-matvec",
        opt="Opt #7",
        title="matmul with a SHORT dimension underfills the PE array (matrix-vector / decode)",
        # A matmul/decode op that is compute-bound-but-slow or low-MFU, especially
        # the bs=1 auto-regressive matrix-vector product.
        triggers=(
            r"short[-_ ]?(?:matmul|mm|dim)",
            r"mat(?:rix)?[-_ ]?vec",
            r"\bdecode\b", r"auto[-_ ]?regress", r"single[-_ ]?query", r"bs[=_ ]?1",
            r"low[-_ ]?mfu",
        ),
        fix=(
            "FAST WEIGHT LOAD (up to 4x on a short-dim matmul). When ONE matmul\n"
            "dimension is much smaller than 128 (the classic case: an autoregressive\n"
            "decode step's [1,K]x[K,N] matrix-vector product), map the SHORT tensor to\n"
            "the MOVING position, NOT stationary: LoadStationary is up to 4x faster\n"
            "than MultiplyMoving of the same free size, so you want the SHORT operand\n"
            "moving and the LARGE operand stationary.\n"
            "  * Short-Moving: LS_II ~= 128/4 = 32 cyc, MM issued ~every 64 cyc.\n"
            "  * Short-Stationary (WRONG): MM issued ~every 128 cyc — 2-4x slower.\n"
            "Swap the args to nisa.nc_matmul (stationary, moving) accordingly; use the\n"
            "identity A@B = (B^T @ A^T)^T (swapping operands transposes the output) so\n"
            "the result layout still matches. This is THE lever for bs=1 decode."
        ),
        technique="fast-weight-load",
    ),
    PerfHint(
        key="partition-vectorize-reduce",
        opt="Opt #5b",
        title="reduce/matmul spans < 128 partitions — partition axis under-vectorized (2x left)",
        triggers=(
            r"sub[-_ ]?128", r"under[-_ ]?128", r"partition[-_ ]?under",
            r"64[-_ ]?partition", r"partition[-_ ]?vector",
            r"serial[-_ ]?reduce", r"two[-_ ]?reduces?",
        ),
        fix=(
            "PARTITION VECTORIZATION (2x). Two independent ops each spanning < 128\n"
            "partitions run SERIALLY (they cannot parallelize when each uses only\n"
            "half the lanes). Pack them onto disjoint partitions of ONE 128-partition\n"
            "tile, then do a SINGLE full-width op:\n"
            "  * write nc_matmul output A to partitions 0:63 and B to 64:127 of the\n"
            "    same PSUM tile, then ONE nl.max/nl.sum(..., axis=1, keepdims=True)\n"
            "    over the full [128, F] tile — 2x faster than two [64, F] reduces.\n"
            "Always fill all 128 partitions: a [128,96] operand uses only 96 PE\n"
            "columns; widening to 128 costs the SAME time. Never leave lanes idle."
        ),
        technique="partition-vectorize",
    ),
    PerfHint(
        key="tensor-tensor-scan-perelement",
        opt="Opt #5a",
        title="scan/prefix-sum emitted as per-element ops (~100-cycle overhead each)",
        triggers=(
            r"per[-_ ]?element", r"element[-_ ]?wise[-_ ]?scan",
            r"tiny[-_ ]?instruction", r"back[-_ ]?to[-_ ]?back",
            r"tensor[-_ ]?scalar[-_ ]?scan", r"prefix[-_ ]?sum",
            r"instruction[-_ ]?overhead", r"one[-_ ]?element[-_ ]?per[-_ ]?partition",
        ),
        fix=(
            "USE THE SCAN PRIMITIVE + BIG FREE-DIM TILES (Opt #5a). A scan written as\n"
            "seq_len back-to-back single-element nisa.tensor_scalar ops pays the ~100-\n"
            "cycle STATIC instruction overhead every step (measured 189 ns / 264 cyc\n"
            "of overhead vs 1 cycle of useful work) — the overhead, not the math,\n"
            "dominates.\n"
            "  * Replace the per-element recurrence with the fused VectorE primitive\n"
            "    nisa.tensor_tensor_scan (one instruction over the whole free axis).\n"
            "  * More generally: make each instruction touch a FREE dim >= 128\n"
            "    elements (ideally larger) so the static overhead amortizes. Read-\n"
            "    after-write chains make tiny tiles even worse — batch them."
        ),
        technique="tensor-tensor-scan",
    ),
    PerfHint(
        key="transpose-swap-layout",
        opt="Opt #8",
        title="an intermediate transpose exists only to satisfy a downstream layout",
        triggers=(
            r"transpose", r"nc_transpose", r"load_transpose", r"layout[-_ ]?mismatch",
            r"re[-_ ]?layout", r"wrong[-_ ]?layout",
        ),
        fix=(
            "MITIGATE THE TRANSPOSE (Opt #8) — do not pay for a re-layout the matmul\n"
            "can absorb. Two levers:\n"
            "  * SWAP stationary/moving in nc_matmul so the OUTPUT layout already\n"
            "    matches what the NEXT op needs (e.g. feeding a layernorm/rmsnorm's\n"
            "    bn_stats which wants the feature dim on the FREE axis: map the WEIGHT\n"
            "    to moving instead of stationary — the output comes out pre-transposed,\n"
            "    no nc_transpose needed).\n"
            "  * Or move the reduce to the engine whose native layout you already have\n"
            "    (TensorE cross-partition reduce via nc_matmul-against-ones vs VectorE\n"
            "    free-axis tensor_reduce). Pick the engine that avoids the transpose.\n"
            "If a transpose is truly unavoidable and the kernel is MEMORY-bound, use\n"
            "nl.load() + nisa.nc_transpose() in a tiled loop, NOT nl.load_transpose2d\n"
            "(the DMA-transpose has much lower bandwidth) — best when TensorE is idle."
        ),
        technique="transpose-swap-for-layout",
    ),
    PerfHint(
        key="instruction-combine-activation",
        opt="Opt #6",
        title="a multiply/add/exp chain touches the data multiple times (combine it, 3x)",
        triggers=(
            r"single[-_ ]?engine", r"serial", r"multi(?:ple)?[-_ ]?pass",
            r"touch(?:es)?[-_ ]?data[-_ ]?(?:twice|multiple)", r"separate[-_ ]?ops",
        ),
        fix=(
            "COMBINE INSTRUCTIONS (Opt #6, ~3x latency). A scale->bias->exp chain done\n"
            "as three ops touches the input three times and serializes the pipeline.\n"
            "Fuse into ONE deep-pipelined ScalarE instruction:\n"
            "    e = nisa.activation(nl.exp, data, bias=b, scale=s)   # exp(s*data+b)\n"
            "and for a fused free-axis reduce use nisa.activation_reduce(op, data,\n"
            "reduce_op=nl.add, reduce_res=acc[...]) so the reduce rides the SAME pass.\n"
            "Touch the data ONCE. Then OVERLAP engines (Opt #3): route exp->ScalarE,\n"
            "sum->VectorE, masking->GpSimdE, matmuls->TensorE so no one engine\n"
            "serializes while the others sit idle."
        ),
        technique="engine-overlap",
    ),
    PerfHint(
        key="large-dma-transfers",
        opt="Opt #9",
        title="DMA-bound with small transfers (< 32 KiB) — packet-rate bound, not bandwidth",
        triggers=(
            r"small[-_ ]?dma", r"tiny[-_ ]?transfer", r"dma[-_ ]?blocked",
            r"bandwidth", r"low[-_ ]?mbu", r"4b[-_ ]?transfer", r"packet[-_ ]?rate",
        ),
        fix=(
            "LARGER DMA TRANSFERS + OVERLAP (Opt #9 / #4). Small loads leave the DMA\n"
            "engines idle between transfers (per-transfer overhead > active time).\n"
            "  * Maximize BOTH dims of each nl.load/nl.store so a transfer is >= 32 KiB\n"
            "    (the ideal minimum); target free dim ~1024 (beyond 1024 has\n"
            "    diminishing return and can hurt pipelining). Example: a 128x1024 fp32\n"
            "    load = 16 transfers (one per DMA engine), each 8 part x 1024 x 4B =\n"
            "    32 KiB. A 4B transfer is pathological.\n"
            "  * DOUBLE-BUFFER (Opt #4): structure the tile loop so tile n+1's DMA\n"
            "    overlaps tile n's compute, driving latency to max(compute, dma), not\n"
            "    compute + dma. All 16 DMA engines should stay busy."
        ),
        technique="wide-aligned-tiles",
    ),
    PerfHint(
        key="fuse-spill",
        opt="Opt #1/#2",
        title="spilling an intermediate through HBM (spill traffic > 30% of SBUF traffic)",
        triggers=(
            r"spill", r"memory[-_ ]?bound", r"reload", r"intermediate[-_ ]?hbm",
            r"temporal[-_ ]?locality", r"round[-_ ]?trip",
        ),
        fix=(
            "RAISE ARITHMETIC INTENSITY (Opt #1/#2) — the memory-bound baseline lever.\n"
            "  * FUSE the whole op into one kernel: producer->consumer tiles stay in\n"
            "    SBUF; one load per input, one store of the output. There is NO HW\n"
            "    cache — a spilled intermediate round-tripped through HBM is pure loss.\n"
            "  * KEEP REUSED DATA resident (temporal locality): hoist loop-invariant\n"
            "    loads (gamma/beta/scale/row-max) OUT of the tile loop, load once.\n"
            "  * GOTCHA: declare buffers INSIDE the inner loop, not outside — a buffer\n"
            "    hoisted above its use can force the compiler to spill it. If spill\n"
            "    traffic exceeds ~30% of SBUF<->device traffic, this is your bottleneck.\n"
            "  * Trade-off vs Opt #3: caching too much raises SBUF pressure and causes\n"
            "    its OWN spills — fuse to the point where spill traffic drops, no more."
        ),
        technique="loop-fusion",
    ),
)


# Symptom tokens that ``symptoms_from`` may emit (documented so callers/tests
# share the vocabulary). These are the profile-threshold breaches + shape cues the
# hints trigger on. Kept in sync with neuron_profile.perf_symptoms.
SYMPTOM_TOKENS = (
    "memory-bound", "single-engine", "dma-blocked", "gpsimd-bound",  # coarse bottleneck labels
    "low-mfu", "low-mbu", "spill-high", "small-dma",      # profile-threshold breaches
    "short-matmul-dim", "per-element-scan", "transpose",  # shape / structure cues
    "indirect-dma", "indirect-gather",                    # GpSimd / static-addressing cues
)


def symptoms_from(bottleneck: str = "", *, reason: str = "", op_name: str = "",
                  op_family: str = "", profile_tokens: tuple[str, ...] = ()) -> str:
    """Build the lowercase symptom haystack ``match_perf_hints`` scans.

    Combines the coarse bottleneck label + the measured ``reason`` string + the op
    name/family + any profile-threshold tokens (``neuron_profile.perf_symptoms``).
    PURE — just string assembly; never raises. The op name/family let a shape cue
    (a matmul named ``*_decode`` / ``qkv_proj``, a ``scan``/``mamba`` op) trigger
    the right shape-specific Opt without a profiler."""
    parts = [bottleneck or "", reason or "", op_name or "", op_family or ""]
    parts.extend(profile_tokens or ())
    # Derive a couple of structural cues from the op name so shape-specific hints
    # (fast-weight-load, tensor_tensor_scan) fire even with only analytic signal.
    hay = " ".join(p for p in parts if p).lower()
    if op_family == "scan" or any(k in hay for k in ("scan", "mamba", "deltanet", "recurr")):
        hay += " per-element-scan"
    if op_family == "matmul" or any(k in hay for k in ("decode", "matvec", "gemv")):
        hay += " short-matmul-dim"
    if op_family == "indirect_gather" or any(k in hay for k in (
            "interpolate", "grid_sample", "grid-sample", "resample", "upsample",
            "gather", "scatter", "bilinear", "warp", "remap")):
        hay += " indirect-gather indirect-dma"
    return hay


def match_perf_hints(haystack: str) -> list[PerfHint]:
    """Return the hints whose trigger tokens appear in ``haystack`` (case-
    insensitive regex search), most-specific (map-order) first. Empty for an
    empty/unrecognized haystack — the caller then falls back to
    ``kernel_perf.guidance_for`` (the coarse lever)."""
    if not haystack:
        return []
    hits: list[PerfHint] = []
    for h in HINTS:
        if any(re.search(p, haystack, re.IGNORECASE) for p in h.triggers):
            hits.append(h)
    return hits


def format_perf_hints(hints: list[PerfHint], max_hints: int = 2) -> str:
    """Render matched hints as a loud imperative block to PREPEND to the perf
    re-author guidance. Capped at ``max_hints`` (one dominant lever per round is
    the guide's discipline — do not dump all ten). Empty string when nothing
    matched."""
    if not hints:
        return ""
    blocks = []
    for h in hints[:max_hints]:
        blocks.append(f">>> PERF ({h.opt}): {h.title} — DO THIS:\n{h.fix}")
    return "\n".join(blocks)


def guidance_from_symptoms(bottleneck: str = "", *, reason: str = "",
                           op_name: str = "", op_family: str = "",
                           profile_tokens: tuple[str, ...] = (),
                           max_hints: int = 2) -> list[str]:
    """One-liner the perf loop calls: build the haystack, match the specific
    NKI-Guide optimizations, and return them as guidance strings (each "Opt #N:
    <fix>"). Empty list when nothing specific matched — the caller keeps its coarse
    ``guidance_for`` lever. Never raises."""
    hay = symptoms_from(bottleneck, reason=reason, op_name=op_name,
                        op_family=op_family, profile_tokens=profile_tokens)
    return [f"{h.opt}: {h.fix}" for h in match_perf_hints(hay)[:max_hints]]
