# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""silent_perf.py — detectors for the perf bugs a compiler HIDES (R25).

The single most-endorsed lesson from the cross-hardware research (docs/nki-stage-
improvement-plan.md §6, T3): both TPU (XLA) and GPU stacks routinely produce a
CORRECT program while silently leaving large performance on the table — and,
worse, silently INSERTING work (copies, relayout transposes, instruction-overlay
stalls) rather than erroring. The SAIL Gemma-v6e write-up spent most of its
52%→63% MFU grind hunting exactly these hidden failures. On Trainium the analogues
are: neuronx-cc quietly inserting **relayout DMAs**, a kernel **re-reading** the
same operand from HBM many times (the KV-re-read that cost 6×), and an
**unrolled** matmul whose code overflows instruction memory.

This module turns those into explicit, machine-readable findings so the perf loop
and the bank stop trusting a "correct" kernel's silent perf losses — the same way
the design already distrusts silent correctness wins (the mock/real labelling +
the MFU-implausibility guard in ``roofline.py``). These detectors are the WASTE
half; ``roofline.py`` owns the %SOL/MFU-ceiling half (and the fake-speedup guard),
and this module reuses it rather than duplicating.

Pure functions over a best-effort ``measurement`` dict (every key optional, so a
partial profile degrades gracefully — a detector that lacks its inputs simply
does not fire, never crashes, never fabricates). Each finding carries a
``route_token`` chosen so ``kernel_perf.classify_bottleneck`` routes it to the
right one-dominant-fix lever (relayout → DMA path, re-read → fuse/keep-resident).

None of the thresholds is a hardware constant this codebase cannot measure: the
re-read and relayout-share bars are dimensionless ratios; the code-size bar is
NOT baked (there is no measured Trn2 instruction-memory figure in-repo), so that
detector only fires when the caller passes a real limit — honesty over a guessed
constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- thresholds (dimensionless; see module doc for why none is a HW constant) --
# Operand re-read multiplier = bytes actually moved through HBM / bytes the op
# MUST move (each input read once, output written once). A perfectly-tiled
# memory-bound kernel is ~1.0; the TPU RPA bug re-read KV ~32×. Warn early, since
# even 1.5× is real bandwidth waste on a memory-bound op.
REREAD_WARN = 1.5
REREAD_CRITICAL = 3.0

# Share of DMA time (or DMA-op count) spent on relayout/transpose moves the
# compiler inserted to fix a layout it silently disagreed with. A little is
# normal; a lot means an operand is stored in the wrong orientation and every
# call pays a copy (the TPU "free transpose that wasn't", ~173 µs/layer).
RELAYOUT_SHARE_WARN = 0.10
RELAYOUT_SHARE_CRITICAL = 0.25

# Roofline gap = 1 - %SOL. Only a LARGE gap is a finding here (the profitability
# verdict in roofline.py is the primary rank signal; this is the human-readable
# companion). Mirrors roofline.OPPORTUNITY_MAX_SOL (0.40) -> gap 0.60.
ROOFLINE_GAP_WARN = 0.60

_CRITICAL = "critical"
_WARN = "warn"


@dataclass(frozen=True)
class SilentPerfFinding:
    """One detected silent-perf issue. ``route_token`` is a phrase safe to append
    to a ``RaceResult.reason`` so ``kernel_perf.classify_bottleneck`` picks the
    matching one-dominant-fix lever without any change to the loop."""

    kind: str          # "operand_reread" | "relayout_dma" | "code_size" | "roofline_gap"
    severity: str      # "warn" | "critical"
    metric: float      # the measured multiplier / share / gap
    detail: str
    route_token: str = ""

    @property
    def critical(self) -> bool:
        return self.severity == _CRITICAL


def _num(measurement: dict, *keys: str) -> float | None:
    """First present, finite, numeric value among ``keys`` (else None). Tolerant
    of a partial profile: a detector whose inputs are absent simply won't fire."""
    for k in keys:
        if k in measurement and measurement[k] is not None:
            try:
                v = float(measurement[k])
            except (TypeError, ValueError):
                continue
            if v == v and v not in (float("inf"), float("-inf")):  # finite, not NaN
                return v
    return None


def operand_reread(bytes_moved: float, bytes_needed: float) -> SilentPerfFinding | None:
    """Flag HBM traffic amplification: ``bytes_moved`` far exceeding the
    ``bytes_needed`` floor (each input read once + output written once) means the
    kernel is re-fetching an operand — the KV-re-read class of bug. Returns None
    when inputs are non-positive or the ratio is within tolerance."""
    if bytes_moved <= 0 or bytes_needed <= 0:
        return None
    mult = bytes_moved / bytes_needed
    if mult < REREAD_WARN:
        return None
    sev = _CRITICAL if mult >= REREAD_CRITICAL else _WARN
    return SilentPerfFinding(
        kind="operand_reread", severity=sev, metric=mult,
        detail=(f"HBM traffic is {mult:.1f}x the minimum ({bytes_moved:.3g} moved "
                f"vs {bytes_needed:.3g} needed) — an operand is being re-read; keep "
                f"it SBUF-resident / retile so each byte is fetched once"),
        # "spill" -> classify_bottleneck -> MEMORY_BOUND (fuse + keep resident).
        route_token="operand re-read / spill through HBM")


def relayout_dma(measurement: dict) -> SilentPerfFinding | None:
    """Flag compiler-inserted relayout/transpose DMAs consuming a large share of
    the DMA path (the 'free transpose that wasn't'). Accepts either time
    (``relayout_dma_ns`` + ``total_dma_ns``) or counts (``relayout_dma_count`` +
    ``total_dma_count``). Returns None when the inputs are absent/non-positive."""
    relayout = _num(measurement, "relayout_dma_ns", "transpose_dma_ns",
                    "relayout_dma_count", "transpose_dma_count")
    total = _num(measurement, "total_dma_ns", "total_dma_count", "dma_ns")
    if relayout is None or total is None or total <= 0 or relayout < 0:
        return None
    share = relayout / total
    if share < RELAYOUT_SHARE_WARN:
        return None
    sev = _CRITICAL if share >= RELAYOUT_SHARE_CRITICAL else _WARN
    return SilentPerfFinding(
        kind="relayout_dma", severity=sev, metric=share,
        detail=(f"{share*100:.0f}% of the DMA path is relayout/transpose the "
                f"compiler inserted — an operand is stored in the wrong "
                f"orientation; pre-transpose it once at load, or emit it in the "
                f"layout the consumer wants (transposed_out/in)"),
        # "relayout"/"dma" -> classify_bottleneck -> DMA_BLOCKED (fix the load path).
        route_token="relayout dma on the load path")


def code_size_blowup(measurement: dict,
                     limit_bytes: float | None = None) -> SilentPerfFinding | None:
    """Flag an instruction-footprint blowup (the unrolled-matmul → instruction-
    memory-overflow → silent fetch-stall class). Fires ONLY when the caller
    supplies a real ``limit_bytes`` (or ``measurement['imem_limit_bytes']``):
    there is no measured Trn2 instruction-memory constant in this repo, so we do
    NOT bake one — a guessed hardware limit would be worse than no detector.
    Reads ``code_bytes`` (preferred) or ``instr_count`` against the limit."""
    limit = limit_bytes if limit_bytes is not None else _num(
        measurement, "imem_limit_bytes")
    if limit is None or limit <= 0:
        return None
    code = _num(measurement, "code_bytes", "instr_bytes", "neff_code_bytes")
    if code is None or code <= 0:
        return None
    ratio = code / limit
    if ratio < 1.0:
        return None
    return SilentPerfFinding(
        kind="code_size", severity=_CRITICAL, metric=ratio,
        detail=(f"kernel code is {code:.3g} B, {ratio:.1f}x the {limit:.3g} B "
                f"instruction-memory limit — it will stream instruction overlays "
                f"from HBM and stall; emit a COMPACT loop instead of unrolling "
                f"(smaller block sizes)"),
        route_token="instruction-memory overflow (unrolled code)")


def roofline_gap(measurement: dict) -> SilentPerfFinding | None:
    """A human-readable companion to roofline.py's profitability verdict: flag a
    large gap to the roofline (measured %SOL well below the ceiling). ``sol`` is a
    fraction 0..1; None/non-positive -> no finding (unmeasured is not a bug)."""
    sol = _num(measurement, "sol")
    if sol is None or sol <= 0:
        return None
    gap = 1.0 - sol
    if gap < ROOFLINE_GAP_WARN:
        return None
    return SilentPerfFinding(
        kind="roofline_gap", severity=_WARN, metric=gap,
        detail=(f"measured {sol*100:.0f}% of SOL ({gap*100:.0f}% gap to the "
                f"roofline) — real headroom; treat the gap as a bug to localize, "
                f"not a fixed cost (roofline-as-bug-detector)"),
        route_token="")   # roofline gap alone does not pick a lever; leave routing to the rest


def detect_silent_perf(measurement: dict,
                       *, code_limit_bytes: float | None = None
                       ) -> list[SilentPerfFinding]:
    """Run every applicable detector over a best-effort ``measurement`` dict and
    return the findings (empty when the kernel is clean or the profile lacks the
    inputs). Never raises. Recognized keys (all optional):
      bytes_moved, bytes_needed            -> operand_reread
      relayout_dma_ns|_count, total_dma_ns|_count -> relayout_dma
      code_bytes|instr_bytes, imem_limit_bytes    -> code_size
      sol                                  -> roofline_gap
    """
    findings: list[SilentPerfFinding] = []
    bm = _num(measurement, "bytes_moved")
    bn = _num(measurement, "bytes_needed")
    if bm is not None and bn is not None:
        f = operand_reread(bm, bn)
        if f:
            findings.append(f)
    for detector in (relayout_dma, roofline_gap):
        f = detector(measurement)
        if f:
            findings.append(f)
    f = code_size_blowup(measurement, limit_bytes=code_limit_bytes)
    if f:
        findings.append(f)
    return findings


def route_tokens(findings: list[SilentPerfFinding]) -> str:
    """Join the findings' ``route_token``s into a phrase safe to append to a
    ``RaceResult.reason`` so ``kernel_perf.classify_bottleneck`` routes to the
    matching one-dominant-fix lever. Empty when nothing routes."""
    toks = [f.route_token for f in findings if f.route_token]
    return "; ".join(toks)


def worst_severity(findings: list[SilentPerfFinding]) -> str:
    """"critical" if any finding is critical, else "warn" if any, else ""."""
    if any(f.critical for f in findings):
        return _CRITICAL
    return _WARN if findings else ""


def summary(findings: list[SilentPerfFinding]) -> str:
    """One-line human summary for the ledger/bank (empty when clean)."""
    if not findings:
        return ""
    parts = [f"{f.kind}[{f.severity}]={f.metric:.2f}" for f in findings]
    return "silent-perf: " + ", ".join(parts)
