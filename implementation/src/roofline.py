# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""roofline.py — the %-of-speed-of-light PROFITABILITY signal for kernel authoring.

The strategy pivot (validated on-device 2026-08-27): a hand-written NKI kernel
does NOT beat the compiler everywhere — on standard compute-bound GEMM the
compiler is already ~80% of hardware speed-of-light, so authoring there is
wasted effort. The win is WHERE THE COMPILER IS WEAK — far from SOL. So %SOL is a
PROFITABILITY GATE: low %SOL (far from the roofline) == a real opportunity worth
an authored kernel; near-SOL == leave it to the compiler.

This module turns a measured latency into a %SOL number against the two roofline
ceilings, and a coarse verdict the gate reads.

    memory-bound op:  %SOL = achieved_bytes_per_s  / PEAK_HBM_BW_PER_CORE
    compute-bound op: %SOL = achieved_flops_per_s   / PEAK_TFLOPS_BF16_PER_CORE

CRITICAL — LATENCY MUST BE DEVICE-TIMED. The peak constants below are single-core
DEVICE ceilings. A %SOL computed from a torch WALLCLOCK latency is meaningless:
wallclock is host/launch-overhead-bound (measured 2026-08-27: a torch-wallclock
rmsnorm reported ~40-65 GB/s while a device-timed streaming copy on the SAME box
sustained ~385 GB/s — the wallclock number was 6-10x low, pure host overhead).
Feed these helpers a latency from ``nki.benchmark`` / ``neuron-profile`` (the
NeuronCore-latency column), NOT a Python ``time.perf_counter`` loop around
``mark_step``. ``%sol_from_wallclock`` is deliberately absent for this reason.

Peak constants are per SINGLE NeuronCore (what the framework authors), trn2:
  * ``PEAK_HBM_BW_PER_CORE`` — MEASURED on-device 2026-08-27 via a streaming-copy
    ``nki.benchmark`` sweep (F=16384/32768/49152 fp32 -> 317/365/384 GB/s,
    asymptoting). 385 GB/s is the sustained single-core ceiling used as the
    memory-bound SOL reference. This is an ACHIEVED ceiling (what a perfect
    single-core streaming kernel gets), which is the honest bar for a memory-bound
    op — not a spec-sheet aggregate-device number a single core can never reach.
  * ``PEAK_TFLOPS_BF16_PER_CORE`` — 79 TFLOP/s DENSE bf16. CORRECTED 2026-08-29
    from the official Trainium2 arch doc (NeuronCore-v3 Tensor Engine: 79 BF16/
    FP16/TF32 dense TFLOPS) + first principles (128x128 array x 2.4 GHz x 2
    flop/MAC = 78.6 TFLOPS). The prior 380e12 was ~4.8x too high — it matched no
    dense figure (fp8 dense 158, bf16 SPARSE 316, fp32 20) and made every
    compute-bound op read ~4.8x BELOW its true %SOL, so the profitability gate
    wrongly flagged near-SOL GEMMs as "worth authoring". Dense per-dtype ceilings
    below (``peak_tflops``); pick the ceiling for the op's actual compute dtype.
    NOTE: doc-derived — worth one on-device compute-bound race to confirm.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- single-core roofline ceilings (trn2 / NeuronCore-v3) --------------------
PEAK_HBM_BW_PER_CORE = 385e9          # bytes/s — MEASURED (streaming-copy, 2026-08-27);
                                      # matches doc 3 TB/s / 8 cores = 375 GB/s/core (~3%)
# Tensor-engine DENSE peak per NeuronCore-v3, by input dtype (arch doc, 2.4 GHz,
# 128x128 array). bf16 is the default authoring dtype; fp8 doubles the datapath
# (256x128), fp32 quarters it, and 2:4 SPARSE quadruples bf16.
PEAK_TFLOPS_BF16_PER_CORE = 79e12     # FLOP/s — DENSE bf16/fp16/tf32 (was 380e12; corrected)
PEAK_TFLOPS_FP32_PER_CORE = 20e12     # FLOP/s — dense fp32
PEAK_TFLOPS_FP8_PER_CORE = 158e12     # FLOP/s — dense fp8 (e4m3/e5m2)
PEAK_TFLOPS_BF16_SPARSE_PER_CORE = 316e12  # FLOP/s — 2:4 structured-sparse bf16/fp8

# The bf16 arithmetic-intensity ridge point (FLOP per byte): above it an op is
# compute-bound, below it memory-bound. Recomputed from the two peak constants so
# the module is self-contained (79e12 / 385e9 ~= 205 flop/byte for dense bf16).
RIDGE_FLOPS_PER_BYTE = PEAK_TFLOPS_BF16_PER_CORE / PEAK_HBM_BW_PER_CORE


def peak_tflops(dtype: str = "bf16", *, sparse: bool = False) -> float:
    """The Tensor-engine dense (or 2:4-sparse) FLOP/s ceiling for a compute
    dtype — so ``sol_compute_bound`` can be held to the RIGHT roofline (an fp8
    matmul has 2x the bf16 ceiling; an fp32 op has 1/4). Defaults to dense bf16.
    Unknown dtype -> dense bf16 (the safe default)."""
    d = (dtype or "").lower()
    if sparse:
        return PEAK_TFLOPS_BF16_SPARSE_PER_CORE
    if "fp8" in d or "e4m3" in d or "e5m2" in d or "float8" in d:
        return PEAK_TFLOPS_FP8_PER_CORE
    if "fp32" in d or "float32" in d or d == "f32":
        return PEAK_TFLOPS_FP32_PER_CORE
    return PEAK_TFLOPS_BF16_PER_CORE

# Profitability thresholds (fractions of SOL). Tunable; the pivot's guidance is
# "low %SOL = opportunity, ~80% = don't bother".
OPPORTUNITY_MAX_SOL = 0.40   # <= 40% of SOL -> a real authoring opportunity
NEAR_SOL_MIN = 0.80          # >= 80% of SOL -> compiler already ~wins; skip

# --- NKI Performance Guide "good utilization" bars (docs/nki-perf-guide.md) ---
# The guide states different SOL bars per bottleneck: a compute-bound kernel is
# "good" at >= 90% engine-active/MFU, a memory-bound kernel at >= 60% MBU. These
# are finer than the single NEAR_SOL_MIN and let the perf loop declare a kernel
# CONVERGED against the RIGHT bar for its bottleneck (so a memory-bound op is not
# held to a 90% bar it can never hit, nor a compute-bound op let off at 60%).
COMPUTE_BOUND_GOOD_SOL = 0.90   # compute-bound: >= 90% of SOL / MFU is good
MEMORY_BOUND_GOOD_SOL = 0.60    # memory-bound:  >= 60% of SOL / MBU is good
# Static shape bars the author should meet BEFORE measuring (Opt #5/#9): every
# instruction needs >= 128 elements/partition to be efficient; DMA transfers want
# >= 32 KiB; the ideal DMA free-dim is ~1024 (beyond it has diminishing return).
MIN_ELEMS_PER_PARTITION = 128
IDEAL_DMA_TRANSFER_KIB = 32
IDEAL_DMA_FREE_DIM = 1024


def good_sol_bar(bottleneck: str) -> float:
    """The guide's 'good utilization' SOL bar for a bottleneck: 90% for
    compute-bound, 60% for memory-bound (the default — the ops this framework
    authors are mostly memory-bound). Use this as the per-bottleneck convergence
    target instead of a single flat threshold."""
    return (COMPUTE_BOUND_GOOD_SOL if bottleneck == "compute_bound"
            else MEMORY_BOUND_GOOD_SOL)


def meets_good_bar(sol: float, bottleneck: str) -> bool:
    """True when a measured %SOL clears the guide's good bar for its bottleneck —
    i.e. the kernel is as close to the roofline as the guide calls 'good' and more
    optimization rounds are unlikely to pay off."""
    return sol > 0.0 and sol >= good_sol_bar(bottleneck)


def sol_memory_bound(bytes_moved: float, device_s: float) -> float:
    """%SOL (0..1+) for a memory-bound op: achieved HBM bandwidth / peak.
    ``device_s`` MUST be a device-timed latency in seconds (see module doc).
    Returns 0.0 for a non-positive/again-unmeasured latency (never a fabricated
    ratio)."""
    if bytes_moved <= 0 or device_s <= 0:
        return 0.0
    achieved_bw = bytes_moved / device_s
    return achieved_bw / PEAK_HBM_BW_PER_CORE


def sol_compute_bound(flops: float, device_s: float,
                      peak_flops: float = PEAK_TFLOPS_BF16_PER_CORE) -> float:
    """%SOL (0..1+) for a compute-bound op: achieved FLOP/s / peak FLOP/s.
    ``device_s`` MUST be a device-timed latency in seconds. ``peak_flops`` is the
    Tensor-engine ceiling for the op's compute dtype (default dense bf16; pass
    ``peak_tflops("fp8")`` / ``peak_tflops("fp32")`` for those paths so the op is
    held to the RIGHT roofline)."""
    if flops <= 0 or device_s <= 0:
        return 0.0
    achieved = flops / device_s
    return achieved / (peak_flops if peak_flops > 0 else PEAK_TFLOPS_BF16_PER_CORE)


@dataclass(frozen=True)
class Profitability:
    """The gate's read on whether an op is worth authoring a kernel for."""
    sol: float                 # measured %SOL against the relevant ceiling (0..1+)
    bottleneck: str            # "memory_bound" | "compute_bound"
    verdict: str               # "opportunity" | "marginal" | "near_sol" | "unknown"
    reason: str

    @property
    def worth_authoring(self) -> bool:
        """True when there is real headroom below the roofline. ``unknown`` (no
        device measurement yet) is treated as worth-authoring: we do NOT skip an
        op just because it has not been device-timed — the gate only PRUNES on a
        POSITIVE near-SOL signal, never on missing data."""
        return self.verdict in ("opportunity", "marginal", "unknown")


def classify(sol: float, bottleneck: str, *, measured: bool = True) -> Profitability:
    """Turn a measured %SOL into a profitability verdict.

    ``measured=False`` (or a non-positive sol) -> "unknown": we could not
    device-time the op, so the gate must NOT prune it (fail-open — never skip an
    op on missing data, only on a positive near-SOL reading)."""
    if not measured or sol <= 0.0:
        return Profitability(
            sol=0.0, bottleneck=bottleneck, verdict="unknown",
            reason="no device-timed measurement — not pruned (fail-open)")
    if sol >= NEAR_SOL_MIN:
        return Profitability(
            sol, bottleneck, "near_sol",
            f"{sol*100:.0f}% of SOL >= {NEAR_SOL_MIN*100:.0f}% — compiler already "
            "near the roofline; authoring is unlikely to beat it")
    if sol <= OPPORTUNITY_MAX_SOL:
        return Profitability(
            sol, bottleneck, "opportunity",
            f"{sol*100:.0f}% of SOL <= {OPPORTUNITY_MAX_SOL*100:.0f}% — far from "
            "the roofline; a well-tiled kernel has real headroom here")
    return Profitability(
        sol, bottleneck, "marginal",
        f"{sol*100:.0f}% of SOL — some headroom, lower priority than a "
        "clear opportunity")


def profitability(bytes_moved: float, flops: float, device_s: float,
                  bottleneck: str) -> Profitability:
    """Convenience: pick the right ceiling from ``bottleneck`` and classify.
    ``device_s`` MUST be device-timed; a non-positive latency yields an
    ``unknown`` (fail-open) verdict."""
    if device_s <= 0:
        return classify(0.0, bottleneck, measured=False)
    if bottleneck == "compute_bound":
        return classify(sol_compute_bound(flops, device_s), bottleneck)
    return classify(sol_memory_bound(bytes_moved, device_s), bottleneck)
