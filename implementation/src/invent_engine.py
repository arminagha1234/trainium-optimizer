"""
Stage-4 INVENT engine — authors NEW NKI kernels, gates them, races them, banks
the results. This is the real invention capability the orchestrator's Stage 4
today only stubs out ("no auto-invention (needs NKI-writer agent)").

Standalone by design: it imports the framework's real ``bank`` (KnowledgeBank /
Lesson), ``guardrails`` (the 5% invention margin), and ``ledger`` (append-only
results.tsv), but does NOT touch ``orchestrator.py`` / ``overnight.py`` — those
boxes have diverged and integration is a later step.

The loop, per op (see ../../docs/stage4-invent-design.md):

    op_spec {name, shapes+dtypes, reference fn, baseline to beat}
      -> author_kernel(op_spec) -> AuthoredKernel        (the headline: novel authoring)
      -> OFFLINE gate:   numpy-ref parity @128x128  +  static NKI lint
      -> ON-DEVICE gate: correctness (allclose vs ref, real shape)
                         + speed race (FAIR: kernel and baseline timed by the
                           SAME method on the SAME device, else device_deferred)
      -> keep ONLY if correct AND faster by >= 5% invention margin
      -> bank:  win  -> `invented` NKI_KERNEL lesson (keyed op+arch+shape-class)
                loss -> `anti_pattern` lesson (correct-but-slow, or wrong, or
                        offline-reject) — losses are DATA, logged not hidden.

CPU-mock-testable: the on-device race is behind ``AuthoredKernel.build()``,
which returns None off-device. On a plain CPU box the engine authors, offline-
gates, and (in tests) banks against an INJECTED race — the full harness logic is
exercised without a Trainium. Only the real ``nki.benchmark`` race needs trn2.

Run on device (.73 / .211):

    # FIRST validate the execution path on a proven seed (build+invoke+measure).
    # Success = it EXECUTES + is measured, NOT "entry function not found".
    # Exits non-zero on device if the seed cannot execute.
    python invent_engine.py --self-test --out /path/to/invent_runs/

    # then author + gate + race + bank the novel ops:
    python invent_engine.py \\
        --ops rope_apply,gelu_tanh,softcap,add_rmsnorm,layernorm,attn_decode \\
        --out /path/to/invent_runs/
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Real framework imports — Stage 4 is banked into the SAME store the proposer
# reads, using the SAME lesson schema and the SAME invention margin.
from bank import (
    Applicability,
    Confidence,
    KnowledgeBank,
    Lesson,
    LessonType,
    Symptom,
    Tier,
    _norm_family,
)
from guardrails import Guardrails
import kernel_rewrites
from kernel_anticheat import require_reproducible, run_candidate_before_reference
from kernel_author import KernelAuthor, RecipeAuthor
from kernel_perf import KernelPerfLoop, PerfFeedback, PerfOutcome
from kernel_repair import CompileResult, Feedback, KernelRepairLoop
from ledger import Layer, Ledger, Origin, Row, Stage, Status, current_commit
from invent_kernels import (
    AuthoredKernel,
    OpSpec,
    author_kernel,
    nki_available,
    resolve_ops,
    static_lint,
)

# fp32 offline parity is a MATH check (does the clever formulation equal the
# reference?), so it is tight. The on-device allclose uses bf16 tolerances.
_OFFLINE_ATOL = 1e-4
_OFFLINE_RTOL = 1e-4
_BF16_ATOL = 1e-2
_BF16_RTOL = 1e-2

# --- bf16-fairness correctness gate ----------------------------------------
# The old WIN gate was ``allclose(kernel, fp32_numpy_ref)`` at 1e-2. That is
# UNFAIR to a bf16 kernel: a bf16 kernel provably cannot match the fp32 ideal to
# 1e-2, and NEITHER DOES THE INCUMBENT bf16 op it replaces — measured on-device,
# torch-eager bf16 itself fails allclose-vs-fp32 at 1e-2 for add_rmsnorm (47
# elems of 2.1M), rmsnorm (7), silu_gate (1), softcap (190), rope_apply (3). So a
# genuine near-miss kernel (add_rmsnorm: only 8 fails vs fp32 — i.e. MORE accurate
# than the incumbent's 47) was scored correct=False by a single boundary element.
#
# The FAIR, non-reward-hacking gate: a bf16 kernel is correct iff it tracks the
# fp32 ideal AT LEAST AS WELL as the incumbent bf16 op it replaces — same 1e-2
# tol, measured against the fp32 ideal (exact host math: this deliberately does
# NOT compare the kernel to a host-side bf16 tensor, which on trn2 differs from
# the device kernel by benign device-vs-host bf16 rounding on a few large-value
# elements — a comparison artifact, not a kernel error, and the true same-DEVICE
# bf16 tensor cannot be read back: neuronx-cc NCC_ISMP902 crashes on the host-read
# of the reduction graph). "No worse than the op it replaces" is the definition of
# an acceptable drop-in, NOT a tolerance loosening: the tol is UNCHANGED, a genuinely
# broken kernel (rope/gelu at 0-2% agreement) misses by ORDERS OF MAGNITUDE more
# elements than the incumbent and fails hard, and the anti-cheat protections
# (candidate-before-reference + reproducibility) are untouched.
#
# The incumbent's bf16 output (from _bf16_oracle) supplies its own fp32-miss count
# as the yardstick. The kernel may miss the fp32 ideal on at most
# ``_CORRECT_FAIL_FACTOR`` x that count (device/host + summation-order bf16
# tie-breaking can roughly double it), OR ``_CORRECT_PPM_FLOOR`` of all elements
# when the incumbent is exact (softmax/layernorm/gelu) — whichever is larger.
_CORRECT_FAIL_FACTOR = 2.0
_CORRECT_PPM_FLOOR = 5e-5      # 50 ppm absolute bf16-boundary budget

# --- Magnitude guard (companion to the miss-COUNT budget above) ------------
# The count budget above bounds HOW MANY elements miss the fp32 ideal, not HOW
# FAR each one misses. On count alone a kernel that is perfect everywhere except
# a handful of elements it fills with NaN/Inf or a catastrophically wrong value
# (e.g. 1e6) would still PASS, as long as those few stay under the count budget.
# So we ALSO bound the MAGNITUDE of the kernel's misses, judged — like the count
# — against the incumbent bf16 op, which defines what a tolerable bf16 error is:
#   * ANY NaN/Inf in the kernel where the fp32 ideal is finite is an instant
#     reject (a correct op never manufactures a non-finite value the ideal lacks).
#   * the kernel's WORST missed-element abs error may exceed the incumbent's OWN
#     worst missed-element abs error by at most ``_CORRECT_MAG_FACTOR`` (device/
#     host + summation-order bf16 tie-breaking can inflate a benign miss a few x),
#     OR ``_CORRECT_MAG_FLOOR`` x the bf16 tolerance band at the data's peak
#     magnitude when the incumbent is exact (softmax/layernorm) — whichever is
#     larger. add_rmsnorm's benign bf16-boundary misses top out ~0.055 and clear
#     this with wide margin; a 1e6 / NaN excursion is rejected. BOTH this and the
#     count budget must pass. This only ever TIGHTENS the gate, never loosens it.
_CORRECT_MAG_FACTOR = 8.0
_CORRECT_MAG_FLOOR = 8.0       # x (atol + rtol*peak|ref|) tol band, oracle-exact case

_SDK = "2.28.0"


# ---------------------------------------------------------------------------
# result records
# ---------------------------------------------------------------------------
@dataclass
class OfflineGate:
    passed: bool
    parity_ok: bool                # True ONLY when an INDEPENDENT re-derivation
                                   # matched the reference — never set by a
                                   # function-compared-to-itself tautology.
    parity_max_abs_err: float
    lint_violations: list[str] = field(default_factory=list)
    reason: str = ""
    # Did the recipe supply a numpy_impl that is a genuinely DIFFERENT expression
    # than spec.reference? When False (the recipe reuses spec.reference verbatim),
    # the offline parity comparison is a tautology (f vs f) and validates nothing,
    # so ``parity_ok`` is forced False and the math is left to the on-device gate.
    # Kept as the LAST field with a default so existing positional constructions
    # (incl. tests) are unaffected.
    parity_independent: bool = True


@dataclass
class RaceResult:
    """Outcome of the on-device correctness + speed race.

    ``ran`` is False off-device (kernel could not be built) — an honest
    "deferred", never a fabricated number.
    """

    ran: bool
    correct: bool = False
    correctness_pct: float = 0.0
    speedup: float = 0.0          # baseline_time / kernel_time; >1 == faster
    kernel_ms: float = 0.0
    baseline_ms: float = 0.0
    reason: str = ""
    # ANALYTIC roofline signal (no profiler — derived from spec shapes+dtype by
    # ``_analytic_roofline``). These tell the author WHY a kernel is slow, which a
    # bare speedup ratio cannot: a memory-bound op that spills is a fusion/traffic
    # problem, a compute-bound op that under-fills the PE is a tiling problem.
    # Trailing + defaulted so every existing positional/keyword RaceResult
    # construction (incl. tests + the deferred paths) is unchanged.
    arithmetic_intensity: float = 0.0   # Flops per byte moved (HBM traffic)
    bottleneck: str = ""                # "memory_bound" | "compute_bound" | ""
    roofline_ratio: float = 0.0         # arithmetic_intensity / bf16 ridge point
    mfu: float = -1.0                   # best-effort model-FLOPs-utilization ceiling; -1 == unknown
    # MEASURED %SOL against the real trn2 single-core roofline (roofline.py peak
    # constants), computed from the device-timed kernel latency + the op's
    # bytes/flops. 0.0 when off-device / unmeasured. ``profit_verdict`` is the
    # profitability read: "opportunity" | "marginal" | "near_sol" | "unknown".
    # Trailing + defaulted so every existing RaceResult construction is unchanged.
    sol: float = 0.0
    profit_verdict: str = ""


@dataclass
class InventResult:
    op: str
    shape_class: str
    origin: str
    status: str                    # harvested | win | anti_pattern |
                                   # offline_reject | device_deferred | no_author
    offline: OfflineGate
    race: RaceResult
    lesson_id: str = ""
    detail: str = ""
    # How many previously-banked lessons (anti-patterns / prior wins) the engine
    # RETRIEVED as relevant to this op before authoring. Makes the "learn from
    # the bank" step observable. Last field with a default so existing positional
    # constructions (incl. tests) are unaffected.
    lessons_consulted: int = 0


# A race function lets tests inject a deterministic device outcome. On device
# the engine's own ``_device_race`` is used.
RaceFn = Callable[[AuthoredKernel, OpSpec], RaceResult]

# A compile function lets the repair loop (and tests) turn an AuthoredKernel into
# a CompileResult (ok + error_log). On device the engine's own ``_compile`` is
# used; tests inject a deterministic stand-in compiler.
CompileFnT = Callable[[AuthoredKernel], CompileResult]


# ---------------------------------------------------------------------------
# analytic roofline — the PERF signal, computed from shapes (no profiler)
# ---------------------------------------------------------------------------
# bf16 TensorE ridge point on NC-v3/trn2: ~222 Flops/Byte (playbook §10 — below
# this arithmetic intensity the op is memory-bound, above it compute-bound). We
# stay device-independent on purpose: this classifies a kernel from its SPEC
# (shapes + dtype), so it is a pure, unit-testable function and needs no run.
_BF16_RIDGE_FLOPS_PER_BYTE = 222.0

# Bytes per element by dtype string (best-effort; defaults to bf16=2 — the
# engine's on-device dtype). Substring match keeps it robust to prefixes like
# "torch." / "np.".
_DTYPE_BYTES = {
    "fp8": 1, "float8": 1, "int8": 1, "e4m3": 1, "e5m2": 1,
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "half": 2, "int16": 2,
    "fp32": 4, "float32": 4, "float": 4, "int32": 4,
    "fp64": 8, "float64": 8, "int64": 8,
}

# Flops per input element for the non-matmul ops (elementwise + norm/reduction).
# These are the small-constant "a few ops per element" shapes: an activation, a
# norm, a reduction all move O(N) bytes and do O(N) flops, so their arithmetic
# intensity is O(1) << ridge — memory-bound by construction. The exact constant
# does not change the classification (all are << 222); it only makes the reported
# AI plausible. Matched by op-name substring; default 4.0.
_FLOPS_PER_ELEM = {
    "rmsnorm": 5.0, "layernorm": 7.0, "softmax": 5.0, "gelu": 8.0,
    "silu": 4.0, "softcap": 5.0, "rope": 6.0, "add": 2.0,
}

# Op-name / family tokens that mean "the work is a matmul" — arithmetic intensity
# can be high (compute-bound), so we estimate MACs from the contracting shapes
# rather than the elementwise per-element constant.
_MATMUL_TOKENS = ("matmul", "mm", "linear", "attn", "attention", "gemm")


def _dtype_bytes(dtype: str) -> int:
    """Bytes-per-element for a dtype string; defaults to 2 (bf16, the on-device
    dtype the engine casts inputs to for the race)."""
    d = (dtype or "").lower()
    for token, nbytes in _DTYPE_BYTES.items():
        if token in d:
            return nbytes
    return 2


def _flops_per_elem(name: str) -> float:
    """Per-input-element flop count for a non-matmul op, by name substring."""
    n = (name or "").lower()
    for token, f in _FLOPS_PER_ELEM.items():
        if token in n:
            return f
    return 4.0


def _analytic_roofline(spec: OpSpec) -> tuple[float, str, float]:
    """Classify an op against the bf16 roofline from its SPEC alone.

    Returns ``(arithmetic_intensity, bottleneck, roofline_ratio)`` where
    ``arithmetic_intensity`` is Flops per byte moved, ``bottleneck`` is
    ``"memory_bound"`` (below the ~222 Flops/Byte bf16 ridge) or
    ``"compute_bound"`` (at/above it), and ``roofline_ratio`` is
    ``arithmetic_intensity / ridge`` (<1 memory-bound, >=1 compute-bound).

    Pure and device-free: it reads the input shapes (``spec.real_inputs`` with an
    ``offline_inputs`` fallback) and the output shape (``spec.reference``), counts
    ONE load per input + ONE store of the output (the fused ideal — playbook §11),
    and estimates flops. For an elementwise/norm/reduction op that is O(1) flops
    per byte (well under the ridge -> memory-bound, which is why fusion is the
    #1 lever for them); for a matmul-family op it estimates MACs from the two
    largest 2-D operands (M*N*K) so a genuine GEMM can land compute-bound. Never
    raises — any shape/reference failure degrades to a memory-bound default with
    a zero AI rather than crashing the race.
    """
    nbytes = _dtype_bytes(spec.dtype)
    try:
        inp = spec.real_inputs()
        if not isinstance(inp, dict):
            inp = {}
    except Exception:  # noqa: BLE001 — a shape-gen failure must not crash the race
        inp = {}
    arrays = [np.asarray(v) for v in inp.values()]
    in_elems = int(sum(a.size for a in arrays))
    try:
        out = np.asarray(spec.reference(inp))
        out_elems = int(out.size)
    except Exception:  # noqa: BLE001 — fall back to the largest input as the store size
        out_elems = max((a.size for a in arrays), default=0)

    bytes_moved = float((in_elems + out_elems) * nbytes)
    if bytes_moved <= 0:
        return 0.0, "memory_bound", 0.0

    name = (spec.name or "").lower()
    if any(tok in name for tok in _MATMUL_TOKENS):
        # MACs ~= M*N*K from the two largest 2-D operands (2 flops/MAC). This is a
        # coarse estimate — enough to let a real GEMM cross the ridge; decode-shape
        # "attn" ops with a thin operand stay low (correctly memory-bound).
        twod = sorted((a for a in arrays if a.ndim >= 2),
                      key=lambda a: a.size, reverse=True)
        if twod:
            m, k = twod[0].shape[0], twod[0].shape[-1]
            n = twod[1].shape[-1] if len(twod) > 1 else 1
            flops = 2.0 * float(m) * float(n) * float(k)
        else:
            flops = _flops_per_elem(name) * float(in_elems)
    else:
        flops = _flops_per_elem(name) * float(in_elems)

    ai = flops / bytes_moved
    ratio = ai / _BF16_RIDGE_FLOPS_PER_BYTE
    bottleneck = "memory_bound" if ai < _BF16_RIDGE_FLOPS_PER_BYTE else "compute_bound"
    return ai, bottleneck, ratio


def _op_bytes_flops(spec: OpSpec) -> tuple[float, float]:
    """The op's HBM traffic (bytes moved: one load per input + one store of the
    output) and flop count, from shapes alone — the raw inputs a %SOL computation
    needs (``roofline.sol_memory_bound``/``sol_compute_bound``). Same accounting
    as ``_analytic_roofline`` (which only returns the ratio); factored out here so
    the achieved-vs-peak %SOL can be computed from a device-timed latency. Never
    raises — returns (0.0, 0.0) on any shape/reference failure, which routes the
    roofline gate to its fail-open ``unknown`` verdict."""
    nbytes = _dtype_bytes(spec.dtype)
    try:
        inp = spec.real_inputs()
        if not isinstance(inp, dict):
            inp = {}
    except Exception:  # noqa: BLE001
        inp = {}
    arrays = [np.asarray(v) for v in inp.values()]
    in_elems = int(sum(a.size for a in arrays))
    try:
        out_elems = int(np.asarray(spec.reference(inp)).size)
    except Exception:  # noqa: BLE001
        out_elems = max((a.size for a in arrays), default=0)
    bytes_moved = float((in_elems + out_elems) * nbytes)
    if bytes_moved <= 0:
        return 0.0, 0.0
    name = (spec.name or "").lower()
    if any(tok in name for tok in _MATMUL_TOKENS):
        twod = sorted((a for a in arrays if a.ndim >= 2),
                      key=lambda a: a.size, reverse=True)
        if twod:
            m, k = twod[0].shape[0], twod[0].shape[-1]
            n = twod[1].shape[-1] if len(twod) > 1 else 1
            flops = 2.0 * float(m) * float(n) * float(k)
        else:
            flops = _flops_per_elem(name) * float(in_elems)
    else:
        flops = _flops_per_elem(name) * float(in_elems)
    return bytes_moved, flops


# ---------------------------------------------------------------------------
# adversarial measure-path protocol — anti-reward-hacking (FIX C)
# ---------------------------------------------------------------------------
def _measure_candidate(candidate_fn: Callable[[], Any],
                       reference_fn: Callable[[], Any],
                       *, scrub_fn: Callable[[], None] | None = None,
                       repro_runs: int = 2) -> tuple[Any, Any, bool, str]:
    """Run the correctness measurement under the two anti-reward-hack protocols
    from ``kernel_anticheat``, so a kernel cannot pass by gaming the harness.

    1. **Candidate before reference** (``run_candidate_before_reference``): the
       candidate is run FIRST, so no reference-produced output exists for a
       do-nothing kernel to alias/return (Kevin-32B's recycled-output exploit).
    2. **Run twice, require identical** (``require_reproducible``): the candidate
       must produce the SAME result across repeats — a candidate that reads a
       racy/uninitialized/aliased buffer drifts and is rejected (Sakana's
       "it's really the reference" measurement). ``scrub_fn`` is called between
       runs to zero/scrub any reusable output/scratch buffer so a stale value
       cannot be recycled and masquerade as deterministic.

    Returns ``(candidate_out, reference_out, reproducible, repro_reason)``. This
    is pure sequencing of INJECTED callables — no device — so the whole protocol
    is unit-testable on CPU mocks, which is the point (the real device wiring in
    ``_device_race`` supplies the concrete run/scrub callables).
    """
    candidate_out, reference_out = run_candidate_before_reference(
        candidate_fn, reference_fn)

    def _repeat() -> Any:
        if scrub_fn is not None:
            scrub_fn()           # scrub reusable buffers BEFORE each repeat run
        return candidate_fn()

    reproducible, repro_reason = require_reproducible(_repeat, n=repro_runs)
    return candidate_out, reference_out, reproducible, repro_reason


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class InventEngine:
    """Authors, gates, races, and banks NKI kernels for a set of ops."""

    def __init__(
        self,
        out_dir: Path | str,
        bank_root: Path | str | None = None,
        guards: Guardrails | None = None,
        sdk_version: str = _SDK,
        registry: "Any" = None,
        author: KernelAuthor | None = None,
        max_repair_rounds: int = 1,
        max_perf_rounds: int = 1,
        perf_use_mutator: bool = True,
        kernel_library: "Any" = None,
        arch: str = "trn2",
    ) -> None:
        # The pluggable authoring seam. Defaults to the recipe table
        # (``RecipeAuthor`` wraps ``invent_kernels.author_kernel``) so behaviour
        # is unchanged; pass an ``LLMAuthor`` (or any ``KernelAuthor``) to drive
        # authoring from a model/agent. ``max_repair_rounds`` is the bound on the
        # author -> compile -> read-error -> re-author loop; the DEFAULT of 1 is
        # today's single-shot authoring (no repair loop), so existing runs and
        # tests are byte-for-byte unchanged. >1 activates the real repair loop.
        self.author: KernelAuthor = author or RecipeAuthor()
        self.max_repair_rounds = max_repair_rounds
        # Bound on the author -> measure -> read-latency -> re-author PERF loop
        # (see ``kernel_perf.KernelPerfLoop``). The DEFAULT of 1 is today's
        # behaviour: a correct kernel is raced ONCE and gated on the 5% margin
        # (a correct-but-slow kernel dead-ends as an anti-pattern) — existing runs
        # and tests are byte-for-byte unchanged. >1 activates the optimize loop:
        # a correct-but-slow kernel is re-authored with measured PerfFeedback
        # until it is fast, converges, or the loop honestly gives up.
        self.max_perf_rounds = max_perf_rounds
        # Who drives the re-author step of that PERF loop. Default True routes it
        # to the STRUCTURAL mutator (``kernel_mutator.MutatingAuthor``): keep the
        # winning template and change ONE mechanical lever per round. This is the
        # on-device finding (2026-08-25) — a Bedrock LLM author RE-WRITES the
        # template every call (source overlap 0.13-0.24 with the winner), so
        # "refine, don't rewrite" is not a promptable behavior; it must be done
        # structurally. Set False to keep the legacy LLM-author-in-loop path (the
        # author re-authored from scratch with perf feedback each round). Only
        # affects runs with ``max_perf_rounds > 1`` (the perf loop is off by
        # default), so the default single-race path is byte-for-byte unchanged.
        self.perf_use_mutator = perf_use_mutator
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Default the bank under the run dir so an experiment never pollutes the
        # curated repo bank unless the caller explicitly points at it.
        self.bank = KnowledgeBank(Path(bank_root) if bank_root
                                  else self.out_dir / "knowledge-bank")
        self.guards = guards or Guardrails()
        self.sdk_version = sdk_version
        self.ledger = Ledger(self.out_dir)
        self.ledger.init()
        # Prior-art / Harvest: consult a kernel registry BEFORE authoring, so a
        # primitive that already has an authored kernel (e.g. DeltaNet for a
        # GatedDeltaNet op) is REUSED, not re-invented. Defaults to an empty
        # registry (nothing available), so behaviour is unchanged unless the
        # caller passes one or $TRN_OPT_KERNEL_DIR is set. Import is local so the
        # engine has no hard dependency on the routing layer.
        if registry is None:
            from kernel_registry import KernelRegistry
            # Default the kernel registry to the IN-REPO validated kernels dir
            # (implementation/src/kernels/) when $TRN_OPT_KERNEL_DIR is unset, so
            # harvest-before-invent fires WITHOUT any manual env: a model whose
            # primitive matches a banked kernel (e.g. flash_attention ->
            # kernels/FlashAttention, on-device-validated) reuses it automatically
            # instead of re-authoring. $TRN_OPT_KERNEL_DIR still overrides (point
            # it at an external/proprietary kernel tree). Falls back to no dir
            # (empty registry, unchanged behavior) if the in-repo dir is absent.
            env_kdir = os.environ.get("TRN_OPT_KERNEL_DIR")
            if env_kdir:
                kdir: str | None = env_kdir
            else:
                _repo_kernels = Path(__file__).resolve().parent / "kernels"
                kdir = str(_repo_kernels) if _repo_kernels.is_dir() else None
            registry = KernelRegistry(kernel_dir=kdir)
        self.registry = registry
        # BANK-ON-WIN: the in-repo validated kernel library (kernel_library.
        # KernelLibrary). When set, an on-device WIN also stores the kernel SOURCE
        # here (keep-winner), so a kernel we authored + validated is durably kept
        # and reused by future runs — not lost. Default None => no library write
        # (behaviour unchanged); the win still banks its NKI_KERNEL lesson as before.
        self.kernel_library = kernel_library
        self.arch = arch

    # -- prior-art / Harvest (search before authoring) -----------------------

    def _prior_art(self, spec: OpSpec):
        """Return a usable, already-authored KernelSpec for this op's primitive,
        or None. This is the Harvest step of Harvest -> Borrow -> Invent, and the
        AutoFixer 'search prior art before authoring' rule: never re-invent a
        kernel the corpus already has. Only kernels at >= simulate-correct rank
        are returned (a failed-compile attempt is not prior art to reuse).

        Resolution is by primitive descriptor AND (as a fallback) the op name,
        via the registry's signature-aware lookup — so a model whose ``primitive``
        field is empty or a near-miss spelling ("qwen3next_gated_delta") still
        harvests the corpus kernel it should, instead of silently re-inventing."""
        prim = getattr(spec, "primitive", "") or ""
        op_name = getattr(spec, "name", "") or ""
        if not prim and not op_name:
            return None
        try:
            # Prefer the signature-aware lookup (primitive + op-name fallback);
            # degrade to the primitive-only API for a registry/mock without it.
            if hasattr(self.registry, "for_signature"):
                kspec = self.registry.for_signature(prim, op_name)
            else:
                kspec = self.registry.for_primitive(prim) if prim else None
        except Exception:  # noqa: BLE001 — a broken registry must not stop authoring
            return None
        return kspec if (kspec and kspec.usable) else None

    # -- learn from the bank (retrieve relevant lessons before authoring) ----

    def _lesson_relevant(self, lesson: Lesson, spec: OpSpec) -> bool:
        """Is a banked lesson relevant to THIS op? By op name / shape_class /
        symptom — the three keys the invent loop banks under (lesson ids are
        ``invented-<op>-<shape_class>`` / ``antipattern-invented-<op>-<shape_class>``;
        symptom signatures name the op)."""
        name = (spec.name or "").lower()
        sc = (spec.shape_class or "").lower()
        hay = f"{lesson.lesson_id} {lesson.reason}".lower()
        if name and name in hay:
            return True
        if sc and sc in hay:
            return True
        iv = lesson.intervention.get("spec", {}) if isinstance(lesson.intervention, dict) else {}
        if isinstance(iv, dict) and (
            iv.get("nki_kernel") == spec.name or iv.get("shape_class") == spec.shape_class
        ):
            return True
        for s in lesson.symptoms_addressed:
            if name and name in (s.signature or "").lower():
                return True
        return False

    def _retrieve_lessons(self, spec: OpSpec) -> list[Lesson]:
        """Query the bank for anti-patterns / prior lessons relevant to this op
        BEFORE authoring, so previously-banked losses and wins become
        load-bearing (today the engine WRITES lessons but never READS them).

        Uses the bank's real retrieval API:
          * ``KnowledgeBank.antipatterns(family, sdk)`` — family anti-patterns
            (verified tier), the same index the pre-compile prune consults;
          * ``KnowledgeBank.query_symptom("compute_bound", ...)`` — the ADIAS
            symptom index the invent NKI_KERNEL / anti-pattern lessons are keyed
            under (verified tier).
        Both read VERIFIED only, so we additionally sweep the PROVISIONAL tier
        the invent loop itself writes to — a loss banked on op A of a run should
        inform a later authoring of the same op/shape-class in the SAME run,
        without waiting on weekly human promotion (the compounding the framework
        is built on). Relevance is filtered by ``_lesson_relevant``.

        Finally, steps (1)-(3) are all keyed by the MODEL architecture family, so
        a kernel/rewrite learned for the SAME op on a DIFFERENT model family is
        never surfaced — the compounding leaks across families. Step (4) closes
        that with ``KnowledgeBank.query_by_op`` (op-family + shape_class keyed,
        model-family agnostic); its results are added DIRECTLY, since it has
        already filtered by op-family and the ``_lesson_relevant`` substring test
        would wrongly drop a same-family-different-op-name match (e.g. a
        ``layernorm`` lesson for an ``rmsnorm`` query). Never raises — a broken
        bank must not stop authoring."""
        sdk = self.sdk_version
        found: dict[str, Lesson] = {}

        def _add(lessons: list[Lesson] | None) -> None:
            for l in lessons or []:
                if l.lesson_id in found:
                    continue
                if self._lesson_relevant(l, spec):
                    found[l.lesson_id] = l

        def _add_direct(lessons: list[Lesson] | None) -> None:
            # Trust the caller's own relevance filter (query_by_op's op-family
            # match); do NOT re-apply _lesson_relevant, which is op-NAME based.
            for l in lessons or []:
                found.setdefault(l.lesson_id, l)

        # (1) family anti-patterns (verified) — the real bank pruning index.
        try:
            _add(self.bank.antipatterns(spec.family, sdk))
        except Exception:  # noqa: BLE001 — a broken bank must not stop authoring
            pass
        # (2) symptom index (verified) — invent lessons are keyed compute_bound.
        try:
            _add(self.bank.query_symptom("compute_bound", spec.family,
                                         0.0, 0, 1, sdk))
        except Exception:  # noqa: BLE001
            pass
        # (3) provisional tier the invent loop itself writes — so lessons compound
        #     within an autonomous run before any human promotion.
        try:
            def _fam_ok(l: Lesson) -> bool:
                af = l.applicability.architecture_family
                if _norm_family(af) != _norm_family(spec.family):
                    return False
                pats = l.applicability.neuron_sdk_versions
                return (not pats) or any(fnmatch.fnmatch(sdk, p) for p in pats)

            _add([l for l in self.bank.load_all(Tier.PROVISIONAL)
                  if l.type in (LessonType.ANTI_PATTERN, LessonType.NKI_KERNEL)
                  and _fam_ok(l)])
        except Exception:  # noqa: BLE001
            pass
        # (4) op-family-keyed retrieval ACROSS model families — a kernel/rewrite
        #     for the SAME op (or same op-family + shape-class) learned on ANY
        #     model family is relevant to authoring this op. Both tiers, since an
        #     invented kernel is banked provisional first. Added directly (see
        #     _add_direct): query_by_op already filtered by op-family.
        try:
            for _tier in (Tier.VERIFIED, Tier.PROVISIONAL):
                _add_direct(self.bank.query_by_op(
                    spec.name, spec.shape_class, tier=_tier))
        except Exception:  # noqa: BLE001
            pass
        return list(found.values())

    # -- diagnose a failure with the rewrite catalog -------------------------

    def _diagnose_failure(self, error_text: str) -> tuple[str, str]:
        """Match a compiler / error string against the rewrite catalog. Returns
        ``(desc_suffix, reason_suffix)`` — both empty when nothing matches —
        turning an opaque "failed" into an actionable "failed; known fix:
        <rewrite>". The reason_suffix is appended to the banked anti-pattern so
        the next author sees the fix; the desc_suffix lands in the ledger row."""
        try:
            rewrites = kernel_rewrites.match_error(error_text or "")
        except Exception:  # noqa: BLE001 — diagnosis must never break banking
            return "", ""
        if not rewrites:
            return "", ""
        names = ", ".join(r.name for r in rewrites)
        return (f" [known fix: {names}]",
                f" Known fix (rewrite catalog): {kernel_rewrites.describe(rewrites)}")

    # -- offline gate --------------------------------------------------------

    def offline_gate(self, author: AuthoredKernel, spec: OpSpec) -> OfflineGate:
        """numpy-ref parity at the 128x128 shape + static NKI lint.

        Both must pass before ANY device time. Parity validates the math the
        kernel is built on (step 2); lint enforces the mandatory NKI rules
        (partition=128, no arange, no int/tile, DMA rule) on the source text.
        """
        if not author.nki_src or not author.entry:
            return OfflineGate(False, False, float("inf"),
                               reason="no authored kernel source (no recipe)")
        # (1) numpy parity at 128x128 — but a parity check only MEANS something
        # when the kernel's numpy_impl is an INDEPENDENT re-derivation of the
        # reference. Most catalog recipes (all but rope_apply) reuse
        # ``spec.reference`` verbatim as their numpy_impl, so ``numpy_impl(inp)``
        # vs ``reference(inp)`` compares a function to ITSELF — a tautology that
        # trivially passes and validates nothing. We detect that by identity and
        # refuse to report it as a parity PASS. The op is NOT rejected: we still
        # execute the impl once (a real smoke check — it runs, it produces the
        # reference's shape) and defer the actual math check to the on-device
        # gate (allclose vs the reference on the REAL shape), which is the true
        # correctness test for these ops.
        independent = author.numpy_impl is not spec.reference
        inp = spec.offline_inputs()
        try:
            got = np.asarray(author.numpy_impl(inp), dtype=np.float32)
            ref = np.asarray(spec.reference(inp), dtype=np.float32)
        except Exception as e:  # noqa: BLE001 — a math bug is a gate failure, not a crash
            return OfflineGate(False, False, float("inf"),
                               parity_independent=independent,
                               reason=f"numpy_impl raised: {e!r}")
        shape_ok = got.shape == ref.shape
        if independent:
            max_err = float(np.max(np.abs(got - ref))) if shape_ok else float("inf")
            parity_ok = shape_ok and np.allclose(
                got, ref, atol=_OFFLINE_ATOL, rtol=_OFFLINE_RTOL)
        else:
            # Tautological comparison — do NOT claim a verified parity pass. The
            # smoke run above still guarantees the impl executes and is shaped
            # like the reference; that is all the offline stage honestly checked.
            max_err = 0.0 if shape_ok else float("inf")
            parity_ok = False
        # (2) static lint.
        violations = static_lint(author.nki_src)
        # Device time is gated on: lint clean AND the impl ran with the right
        # shape AND (only when an independent check exists) that check passed. A
        # tautological-parity op still advances to the REAL on-device gate — we
        # simply never pretend an offline parity pass occurred.
        passed = (not violations) and shape_ok and (parity_ok if independent else True)
        reason = ""
        if not shape_ok:
            reason = "numpy_impl shape != reference shape"
        elif independent and not parity_ok:
            reason = f"numpy parity fail (max_abs_err={max_err:.3e})"
        elif violations:
            reason = f"lint: {'; '.join(violations)}"
        elif not independent:
            # Passing, but be explicit in the record about what was NOT verified.
            reason = ("parity NOT independently verified: numpy_impl is "
                      "spec.reference (tautology) — math deferred to on-device gate")
        return OfflineGate(passed, parity_ok, max_err, violations, reason,
                           parity_independent=independent)

    # -- on-device race ------------------------------------------------------

    def _device_race(self, author: AuthoredKernel, spec: OpSpec) -> RaceResult:
        """Real trn2 correctness + speed race. No-op (ran=False) off-device.

        Built to run on .73; not exercised on a CPU box. Correctness is
        ``torch.allclose`` at bf16 tolerance vs the reference on the REAL shape;
        speed is ``nki.benchmark(kernel, inputs, n_iterations=100)`` for the
        kernel vs a torch-eager baseline. Any failure degrades to a recorded
        reason, never a crash — an un-compilable invented kernel is the common
        case and must be survivable.
        """
        # Beta-3: shape-keyed trace cache would survive a source fix — force it
        # off in-process too (build() also sets this; belt-and-suspenders in case
        # the caller imported nki before build() ran).
        os.environ["NKI_ENABLE_TRACE_CACHE"] = "0"
        # ANALYTIC roofline signal — shape-derived and device-independent, so we
        # compute it up front and attach it to EVERY RaceResult below (including
        # the deferred / errored ones): the perf classification is a property of
        # the op, not of whether this particular box could run it.
        ai, bottleneck, ratio = _analytic_roofline(spec)
        mfu = min(1.0, ratio) if ratio > 0 else -1.0
        _perf = dict(arithmetic_intensity=ai, bottleneck=bottleneck,
                     roofline_ratio=ratio, mfu=mfu)
        fn = author.build()
        if fn is None:
            return RaceResult(False, reason=(
                "kernel not built (off-device: no nki) — on-device race deferred"
                if not nki_available() else
                "kernel failed to build/trace on device"), **_perf)
        # A speed race is only meaningful when BOTH contenders are measured the
        # SAME way on the SAME device. Establish the Neuron device handle up
        # front: if we cannot (no torch_xla / not really on a device), then we
        # can only wallclock the torch baseline on CPU while the kernel runs on
        # device — a physically meaningless CPU-vs-device ratio biased toward the
        # kernel. Rather than fabricate that "win", we DEFER (ran=False), exactly
        # as we do when the kernel cannot build. Honesty over a banked artifact.
        device = _neuron_device()
        if device is None:
            return RaceResult(False, reason=(
                "kernel built but no Neuron device handle for a fair "
                "same-device, same-method race — deferred (never a "
                "CPU-baseline-vs-device-kernel speedup)"), **_perf)
        try:
            import torch  # noqa: PLC0415 — device-only import
            import nki      # noqa: PLC0415
            # NOTE: do NOT import ``torch_neuronx.nki_hop`` here. It was imported
            # eagerly (unused) and on torch-neuronx 2.9 the ``nki_hop`` module no
            # longer exists, so the import raised ImportError and aborted the race
            # before ANY real device work — turning a healthy box into a recorded
            # "device race error". The direct-call invocation path (see
            # _invoke_kernel) needs nothing from nki_hop; the only place that
            # still touches it is a LABELLED fallback, guarded there.

            inp = spec.real_inputs()

            def _to_dev(a: np.ndarray):
                # Move onto the SAME device the kernel runs on — the baseline is
                # compared against the kernel there, not on the host.
                return (torch.from_numpy(np.ascontiguousarray(a))
                        .to(torch.bfloat16).to(device))

            # Positional args in the kernel's declared order (see _arg_order),
            # mirroring how the proven moe_fused kernels are invoked
            # (get_multilayer_kernel_jit(L)[2](*args) — a direct positional call
            # of the @nki.jit callable). See _invoke_kernel for why we call the
            # jit'd fn directly instead of wrap_nki(kernel)[1](**kwargs).
            args = [_to_dev(inp[k]) for k in _arg_order(spec.name, inp)]

            def _candidate():
                out = _invoke_kernel(fn, args)
                # Copy to host FIRST, then cast on the host. Casting the still-on-
                # device tensor (out.to(fp32).cpu()) fuses a `convert` op into the
                # SAME XLA graph as the NKI custom-call, which trips a neuronx-cc
                # Simplifier crash (NCC_ISMP902 "is_subset()") on the kernel module
                # — isolated on real silicon: out.to(fp32).cpu() -> compiler crash;
                # out.cpu().to(fp32) -> PASS + correct. No NEURON_CC_FLAG works
                # around it; the fix is graph structure (keep the custom-call
                # compiling alone; do the dtype cast on the host).
                return out.cpu().to(torch.float32).numpy()

            def _reference():
                return np.asarray(spec.reference(inp), dtype=np.float32)

            # ADVERSARIAL correctness (FIX C, anti-reward-hacking): run the
            # CANDIDATE before the REFERENCE — so a do-nothing kernel has no
            # reference output to alias (Kevin-32B) — and require the candidate to
            # be REPRODUCIBLE across repeats (a racy/aliased/uninitialized read
            # drifts; Sakana). In this direct-call contract the kernel RETURNS a
            # fresh output tensor each invocation (there is no passed-in out=
            # buffer to recycle), so each repeat already produces a fresh buffer
            # and there is no persistent output to scrub between runs — the
            # candidate-before-reference ordering + the run-twice check are the
            # load-bearing protections here.
            got, ref, reproducible, repro_reason = _measure_candidate(
                _candidate, _reference, scrub_fn=None)
            if not reproducible:
                # Nondeterministic / gamed output is NOT correct, regardless of
                # whether a single run happened to match the reference.
                return RaceResult(True, correct=False, correctness_pct=0.0,
                                  reason=f"non-reproducible candidate — {repro_reason}",
                                  **_perf)
            # --- FAIR correctness: no worse than the incumbent bf16 op ---------
            # (bf16-fairness fix — see the _CORRECT_* module notes.) ``ref`` is the
            # fp32 numpy ideal, which a bf16 kernel provably cannot meet at 1e-2
            # (and neither does the incumbent bf16 op). Instead of a strict allclose
            # vs fp32, _bf16_correct scores the kernel correct iff it misses the
            # fp32 ideal on no more elements than the incumbent bf16 op does (the
            # op it replaces). The anti-cheat protections above (candidate-before-
            # reference ordering + reproducibility) are untouched.
            oracle, oracle_src = _bf16_oracle(spec.name, inp, device)
            correct, corr_pct, oracle_note = _bf16_correct(got, ref, oracle,
                                                           oracle_src)

            # FAIR race: time BOTH the authored kernel AND the torch baseline
            # with the SAME synchronized on-device wallclock, on tensors resident
            # on the SAME device. (We use on-device wallclock for both rather
            # than nki.benchmark, because nki.benchmark can only time an nki
            # kernel — not the torch-eager baseline — so it cannot be applied
            # symmetrically. Symmetry is the whole point.)
            def _run_kernel():
                _invoke_kernel(fn, args)

            def _run_baseline():
                with torch.no_grad():
                    _torch_baseline(spec.name, inp, device=device)

            kernel_ms = _device_timed_ms(_run_kernel, device)
            baseline_ms = _device_timed_ms(_run_baseline, device)
            speedup = _fair_speedup(kernel_ms, baseline_ms,
                                    "wallclock@device", "wallclock@device")
            if speedup is None:
                # Timings were not comparable (non-positive / not same-method
                # same-device) — defer instead of banking a meaningless ratio.
                return RaceResult(False, reason=(
                    f"fair on-device timing failed (kernel={kernel_ms:.3f}ms, "
                    f"baseline={baseline_ms:.3f}ms) — deferred"), **_perf)
            # %SOL against the real trn2 roofline (roofline.py measured peaks),
            # from the device-timed kernel latency + the op's bytes/flops. This is
            # the PROFITABILITY signal: a kernel already near SOL leaves no room to
            # optimize further (the perf gate reads profit_verdict). device_s uses
            # the SAME synchronized on-device latency as the speedup, not a naive
            # host loop. Never raises — a roofline failure just leaves sol at 0.0.
            sol, profit_verdict = 0.0, ""
            try:
                import roofline  # noqa: PLC0415 — optional, self-contained
                _bytes, _flops = _op_bytes_flops(spec)
                _prof = roofline.profitability(
                    _bytes, _flops, kernel_ms / 1000.0, bottleneck)
                sol, profit_verdict = _prof.sol, _prof.verdict
            except Exception:  # noqa: BLE001 — %SOL is advisory; never break the race
                pass
            return RaceResult(True, correct, corr_pct, speedup,
                              kernel_ms, baseline_ms,
                              reason=f"correct={correct} speedup={speedup:.3f}x [{oracle_note}]",
                              sol=sol, profit_verdict=profit_verdict, **_perf)
        except Exception as e:  # noqa: BLE001 — device errors are data
            return RaceResult(True, False, 0.0, 0.0,
                              reason=f"device race error: {e!r}", **_perf)

    # -- banking -------------------------------------------------------------

    def _bank_win(self, spec: OpSpec, race: RaceResult) -> str:
        """A correct, >=5%-faster invented kernel -> provisional NKI_KERNEL lesson.

        Keyed by op + family + shape-class. Records ``beat_borrowed_by`` (the
        fraction over the raced baseline) so the bank's auto-promotion policy can
        apply the invented-margin gate honestly. Tier is PROVISIONAL: an
        invented kernel is trusted by later models only after promotion.
        """
        lesson_id = f"invented-{spec.name}-{spec.shape_class}"
        lesson = Lesson(
            lesson_id=lesson_id,
            type=LessonType.NKI_KERNEL,
            applicability=Applicability(
                architecture_family=spec.family,
                neuron_sdk_versions=[f"{_minor_glob(self.sdk_version)}"],
            ),
            layer=Layer.KERNEL,
            migration_risk="low",
            origin=Origin.INVENTED,
            tier=Tier.PROVISIONAL,
            intervention={"spec": {"nki_kernel": spec.name,
                                   "shape_class": spec.shape_class}},
            reason=(
                f"Invented NKI kernel for {spec.name} ({spec.shape_class}): "
                f"correct at bf16 tol and {race.speedup:.2f}x the {spec.baseline} "
                f"baseline on device ({race.kernel_ms:.3f}ms vs "
                f"{race.baseline_ms:.3f}ms). Authored from scratch via the 7-step "
                f"pipeline; beat the baseline by "
                f"{(race.speedup - 1.0) * 100:.1f}% (>= 5% invention margin)."),
            symptoms_addressed=[Symptom(
                bottleneck="compute_bound",
                signature=f"{spec.name} op is a hot, fusable site",
                observed_via="op-level benchmark vs eager baseline")],
            source="invent-engine",
            confidence=Confidence(n_models_validated=1, architecture_diversity=1,
                                  human_verified=False),
            last_reverified_sdk=self.sdk_version,
            evidence=[{"op": spec.name, "shape_class": spec.shape_class,
                       "speedup": round(race.speedup, 4),
                       "kernel_ms": round(race.kernel_ms, 4),
                       "baseline_ms": round(race.baseline_ms, 4),
                       "correctness_pct": round(race.correctness_pct, 3),
                       "baseline": spec.baseline}],
            backend_validated=["native-pytorch-beta3"],
            beat_borrowed_by=round(race.speedup - 1.0, 4),
        )
        self.bank.save(lesson)
        return lesson_id

    def _bank_kernel_to_library(self, spec: OpSpec, author, race: RaceResult) -> bool:
        """On an on-device WIN, store the kernel SOURCE in the in-repo kernel
        library (keep-winner) so it is durably kept + reused, not re-invented next
        run. No-op when no library is configured or there is no real source.
        Never raises into the run — a library failure must not fail the win."""
        if self.kernel_library is None or not getattr(author, "nki_src", ""):
            return False
        try:
            primitive = getattr(spec, "primitive", "") or spec.name
            manifest = {
                "name": f"{spec.name}",
                "primitive": primitive,
                "arch": self.arch,
                "shape_class": spec.shape_class,
                "entry": author.entry,
                "status": "passed-on-device",
                "dtype": getattr(spec, "dtype", "fp32"),
                "provenance": (f"auto-banked by invent-engine on-device win: "
                               f"{spec.name} {race.speedup:.2f}x vs {spec.baseline}"),
                "correctness": {"cosine": 1.0 if race.correct else 0.0,
                                "correctness_pct": round(race.correctness_pct, 3)},
                "performance": {"speedup": round(race.speedup, 4),
                                "baseline": spec.baseline,
                                "kernel_ms": round(race.kernel_ms, 4),
                                "baseline_ms": round(race.baseline_ms, 4)},
                "sdk": {"neuronxcc": self.sdk_version, "backend": "native-pytorch-beta3"},
            }
            return bool(self.kernel_library.bank(manifest, author.nki_src))
        except Exception:  # noqa: BLE001 — a library write must never fail the win
            return False

    def _bank_anti_pattern(self, spec: OpSpec, reason: str,
                           race: RaceResult | None = None,
                           diagnosis: str = "") -> str:
        """A wrong / slow / un-buildable invented kernel -> provisional anti-pattern.

        No ``matcher`` on purpose: this is a recorded WARNING ("we tried an
        invented {op} kernel of this shape-class and it did not beat eager"),
        not a hard pre-prune — a future SDK or a better formulation may change
        the answer, so the loss is remembered but does not silently block a
        retry. Losses are data.
        """
        lesson_id = f"antipattern-invented-{spec.name}-{spec.shape_class}"
        detail = reason
        if race is not None and race.ran:
            detail = (f"{reason} (correct={race.correct}, "
                      f"speedup={race.speedup:.3f}x, "
                      f"kernel={race.kernel_ms:.3f}ms, base={race.baseline_ms:.3f}ms)")
        # Append the rewrite-catalog diagnosis (if any) so the banked warning is
        # actionable — the next author reads a known fix, not just "it failed".
        detail = f"{detail}{diagnosis}"
        lesson = Lesson(
            lesson_id=lesson_id,
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability(
                architecture_family=spec.family,
                neuron_sdk_versions=[f"{_minor_glob(self.sdk_version)}"],
            ),
            layer=Layer.KERNEL,
            migration_risk="low",
            origin=Origin.INVENTED,
            tier=Tier.PROVISIONAL,
            reason=(f"Invented NKI kernel for {spec.name} ({spec.shape_class}) "
                    f"did not win: {detail}. Recorded as a warning, not a "
                    f"pre-prune — retry allowed on a new SDK / formulation."),
            confidence=Confidence(n_models_validated=1, architecture_diversity=1,
                                  human_verified=False),
            last_reverified_sdk=self.sdk_version,
            evidence=[{"op": spec.name, "shape_class": spec.shape_class,
                       "reason": reason,
                       "speedup": round(race.speedup, 4) if race else None,
                       "correctness_pct": round(race.correctness_pct, 3)
                       if race else None}],
            backend_validated=["native-pytorch-beta3"],
        )
        self.bank.save(lesson)
        return lesson_id

    # -- ledger --------------------------------------------------------------

    def _record(self, spec: OpSpec, status: Status, metric: float,
                correctness: float, desc: str, origin: Origin = Origin.INVENTED,
                n_lessons: int = 0) -> None:
        # Surface how many banked lessons informed this op (learn-from-the-bank
        # step). Prefix only when >0 so records with no relevant prior are
        # byte-for-byte unchanged.
        prefix = f"[lessons:{n_lessons}] " if n_lessons else ""
        self.ledger.append(Row(
            commit=current_commit(self.out_dir),
            stage=Stage.INVENT, origin=origin, layer=Layer.KERNEL,
            source="invent-engine", metric=metric, mfu=-1.0,
            correctness=correctness, compile_s=0.0, status=status,
            description=f"{spec.name}/{spec.shape_class}: {prefix}{desc}",
        ))

    # -- the loop ------------------------------------------------------------

    def run_op(self, spec: OpSpec, race_fn: RaceFn | None = None,
               compile_fn: "CompileFnT | None" = None) -> InventResult:
        """Learn (retrieve) -> Prior-art (Harvest) -> author -> offline gate ->
        on-device race -> keep/discard -> bank.

        ``compile_fn`` is the seam the repair loop compiles through (only used
        when ``max_repair_rounds > 1``). It defaults to the engine's own
        ``_compile`` (offline gate + on-device build), and is injectable the same
        way ``race_fn`` is, so the repair loop is unit-testable off-device with a
        deterministic stand-in compiler."""
        # LEARN FIRST: retrieve previously-banked lessons (anti-patterns / prior
        # wins) relevant to this op so the bank is READ, not just written. The
        # count is recorded on the ledger row + result; the lessons themselves
        # are handed to the author (which the recipe author ignores today; the
        # LLM author consumes them). This is the "compounding" step.
        lessons = self._retrieve_lessons(spec)
        n = len(lessons)

        # HARVEST FIRST (Harvest -> Borrow -> Invent): if the corpus already has
        # a usable kernel for this op's primitive, REUSE it — do not spend a
        # compile re-inventing what exists. Recorded as a HARVESTED keep so the
        # ledger shows the reuse (and its HW-readiness tier) honestly.
        prior = self._prior_art(spec)
        if prior is not None:
            tier = "on-device" if prior.hw_ready else "simulate"
            self._record(spec, Status.KEEP, 0.0, 100.0,
                         f"harvested existing {prior.name} kernel "
                         f"({prior.status}, {tier}-validated) -> reuse, no authoring",
                         origin=Origin.HARVESTED, n_lessons=n)
            return InventResult(spec.name, spec.shape_class, "harvested",
                                "harvested",
                                OfflineGate(True, False, 0.0,
                                            reason=f"prior art: {prior.name}"),
                                RaceResult(False, reason="harvested (not raced)"),
                                detail=f"reused {prior.name} [{prior.status}]",
                                lessons_consulted=n)

        # REAL repair loop (only when asked). With the default max_repair_rounds=1
        # this branch is skipped entirely and authoring is the single-shot path
        # below — byte-for-byte today's behaviour.
        if self.max_repair_rounds and self.max_repair_rounds > 1:
            return self._run_op_with_repair(spec, lessons, n, race_fn, compile_fn)

        # SINGLE-SHOT (default): author once through the seam with no feedback.
        # RecipeAuthor forwards ``lessons`` to ``author_kernel`` exactly as before.
        author = self.author.author(spec, lessons, [])
        return self._finish(spec, author, n, race_fn, lessons=lessons)

    def _finish(self, spec: OpSpec, author: AuthoredKernel, n: int,
                race_fn: RaceFn | None,
                lessons: list | None = None) -> InventResult:
        """Shared tail: offline gate -> on-device race -> keep/discard -> bank.

        Extracted verbatim from the original single-shot ``run_op`` so BOTH the
        single-shot and the repaired-kernel paths run the SAME gates (offline
        parity + lint, on-device race, 5% invention margin) and produce the SAME
        honest outcomes. Behaviour for the single-shot path is unchanged."""
        if not author.nki_src:
            self._record(spec, Status.DISCARD, 0.0, 0.0,
                         f"no author available ({author.pipeline_notes})",
                         origin=Origin.NONE, n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "no_author",
                                OfflineGate(False, False, float("inf"),
                                            reason="no author"),
                                RaceResult(False, reason="no author"),
                                detail=author.pipeline_notes, lessons_consulted=n)

        offline = self.offline_gate(author, spec)
        if not offline.passed:
            # Diagnose the offline-reject reason with the rewrite catalog.
            desc_sfx, reason_sfx = self._diagnose_failure(offline.reason)
            lid = self._bank_anti_pattern(
                spec, f"offline gate: {offline.reason}", diagnosis=reason_sfx)
            self._record(spec, Status.DISCARD, 0.0,
                         100.0 if offline.parity_ok else 0.0,
                         f"offline reject: {offline.reason}{desc_sfx}", n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "offline_reject", offline,
                                RaceResult(False, reason="offline reject"),
                                lesson_id=lid, detail=f"{offline.reason}{desc_sfx}",
                                lessons_consulted=n)

        race = (race_fn or self._device_race)(author, spec)

        if not race.ran:
            # Off-device (or un-buildable): offline gate passed, device deferred.
            # NOT a win and NOT an anti-pattern — honestly "not yet raced".
            self._record(spec, Status.DISCARD, 0.0, 100.0,
                         f"offline pass; on-device race deferred ({race.reason})",
                         n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "device_deferred", offline, race,
                                detail=race.reason, lessons_consulted=n)

        if not race.correct:
            # Diagnose the on-device failure (compiler/error string) with the
            # rewrite catalog — an opaque "wrong" becomes an actionable fix.
            desc_sfx, reason_sfx = self._diagnose_failure(race.reason)
            lid = self._bank_anti_pattern(
                spec, "incorrect on device", race, diagnosis=reason_sfx)
            self._record(spec, Status.DISCARD, race.speedup,
                         race.correctness_pct,
                         f"WRONG on device ({race.correctness_pct:.1f}% within tol)"
                         f"{desc_sfx}", n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "anti_pattern", offline, race,
                                lesson_id=lid, detail=f"incorrect on device{desc_sfx}",
                                lessons_consulted=n)

        # PERF LOOP (only when asked): a kernel can be CORRECT but slow (the 0.08x
        # rmsnorm case, which otherwise dead-ends here as an anti-pattern). When
        # ``max_perf_rounds > 1`` we try to make it FAST *before* the invention
        # margin gate: re-author with measured PerfFeedback (latency vs baseline +
        # the ONE dominant bottleneck + a targeted fix), re-measuring each round,
        # keeping the running-best correct kernel. The loop's BEST result then
        # flows into the SAME win/anti-pattern gate below (5% margin unchanged).
        # With the default max_perf_rounds=1 this branch is skipped entirely and
        # behaviour is byte-for-byte today's single race.
        perf_note = ""
        if self.max_perf_rounds and self.max_perf_rounds > 1:
            # PROFITABILITY GATE: if the correct kernel is ALREADY near the
            # hardware roofline (>=80% of measured single-core SOL), further
            # optimization cannot meaningfully help — skip the perf loop and its
            # authoring cost. Fail-open: only a POSITIVE near_sol reading skips;
            # "opportunity"/"marginal"/"unknown" all still optimize (we never skip
            # an op on a missing/low measurement). See roofline.py.
            if getattr(race, "profit_verdict", "") == "near_sol":
                perf_note = (f" [perf loop skipped: near_sol "
                             f"(sol={race.sol*100:.0f}% of roofline) — already "
                             f"near the hardware ceiling, no headroom to optimize]")
            else:
                author, race, perf_note = self._optimize_perf(
                    spec, author, race, lessons, race_fn)

        # Correct — now the speed race with the 5% invention margin.
        is_win = self.guards.is_improvement(race.speedup, 1.0, is_invention=True)
        if is_win:
            lid = self._bank_win(spec, race)
            self._bank_kernel_to_library(spec, author, race)   # keep-winner source store
            self._record(spec, Status.KEEP, race.speedup, race.correctness_pct,
                         f"WIN: {race.speedup:.3f}x vs {spec.baseline} "
                         f"(>= 5% margin){perf_note}", n_lessons=n)
            return InventResult(spec.name, spec.shape_class, spec.origin,
                                "win", offline, race, lesson_id=lid,
                                detail=f"{race.speedup:.3f}x{perf_note}",
                                lessons_consulted=n)
        # Correct-but-slow -> anti-pattern. When a perf loop ran, bank the BEST
        # attempt WITH its latency trajectory (losses are data — a future author /
        # SDK sees how far the optimize loop got and where it stalled).
        lid = self._bank_anti_pattern(
            spec, f"correct but only {race.speedup:.3f}x (< 5% margin){perf_note}",
            race)
        self._record(spec, Status.DISCARD, race.speedup, race.correctness_pct,
                     f"correct-but-slow: {race.speedup:.3f}x (< 5% margin){perf_note}",
                     n_lessons=n)
        return InventResult(spec.name, spec.shape_class, spec.origin,
                            "anti_pattern", offline, race, lesson_id=lid,
                            detail=f"correct but {race.speedup:.3f}x < 1.05x{perf_note}",
                            lessons_consulted=n)

    # -- the PERF loop (author -> measure -> read-latency -> re-author) -------

    def _optimize_perf(self, spec: OpSpec, author: AuthoredKernel,
                       race: RaceResult, lessons: list | None,
                       race_fn: RaceFn | None) -> tuple[AuthoredKernel, RaceResult, str]:
        """Drive ``KernelPerfLoop`` so a measured slow latency TEACHES the next
        attempt (measured latency + dominant bottleneck + one targeted fix fed
        back via the author's ``perf_feedback`` arg). Seeded with the already-
        measured correct (kernel, race) so the running-best is real from round 0
        and a round-1 regression keeps it. Returns the BEST (kernel, race) plus a
        short note (loop reason + latency trajectory) appended to the ledger/bank.

        The measure step is the SAME race the engine already uses (``race_fn`` or
        ``_device_race``) so it RE-VALIDATES correctness AND re-measures each
        round — a perf rewrite that breaks correctness is caught and stops the
        loop (``regressed_or_broke``), never banked as a win."""
        loop = KernelPerfLoop(
            max_rounds=self.max_perf_rounds,
            min_gain_pct=self.guards.marginal_improvement_pct,
            min_utilization=self.guards.min_utilization)
        measure = race_fn or self._device_race

        # WHO re-authors each round. Default: the STRUCTURAL mutator — keep the
        # winning template, change ONE mechanical lever (wider tile, delayed
        # division, activation-reduce fusion), routed by the loop's diagnosed
        # bottleneck. This is the on-device finding: an LLM author re-derives from
        # scratch every call, so refinement must be structural (see kernel_mutator
        # docstring + self.perf_use_mutator). The mutator only PROPOSES; the
        # loop's measure_fn re-validates correct+faster, so a bad variant is
        # cheaply rejected with NO wasted LLM round, and an exhausted variant
        # queue hands back the seed unchanged -> the loop stops honestly (no_gain).
        # Falls back to the LLM author when disabled OR when the seed kernel has no
        # source to mutate (a NO-AUTHOR from-scratch op).
        use_mutator = self.perf_use_mutator and bool(getattr(author, "nki_src", ""))
        if use_mutator:
            from kernel_mutator import MutatingAuthor
            _mutator = MutatingAuthor(
                seed_src=author.nki_src, entry=author.entry,
                op=spec.name, reference=author.numpy_impl)

            def author_fn(perf_trail: list[PerfFeedback]) -> AuthoredKernel:
                # Next template-preserving variant, prioritized by the latest
                # diagnosed bottleneck in perf_trail. No LLM call.
                return _mutator.author_fn(perf_trail)
        else:
            def author_fn(perf_trail: list[PerfFeedback]) -> AuthoredKernel:
                # Legacy path: re-author with the accumulated perf feedback
                # (analogous to the repair loop's ``feedback``). No repair feedback
                # here — the kernel already compiled and ran correctly; the open
                # problem is speed.
                return self.author.author(spec, lessons, [], perf_feedback=perf_trail)

        def measure_fn(kernel: AuthoredKernel) -> RaceResult:
            return measure(kernel, spec)

        outcome: PerfOutcome = loop.run(
            author_fn, measure_fn, seed_kernel=author, seed_race=race)
        driver = "structural-mutator" if use_mutator else "llm-author"
        note = (f" [perf loop ({driver}): {outcome.reason} in "
                f"{outcome.rounds} round(s); latency {outcome.trajectory_str}]")
        best_kernel = outcome.kernel if outcome.kernel is not None else author
        best_race = outcome.race if outcome.race is not None else race
        return best_kernel, best_race, note

    # -- the REAL repair loop (author -> compile -> read-error -> re-author) --

    def _compile(self, kernel: AuthoredKernel, spec: OpSpec) -> CompileResult:
        """Default compile step the repair loop drives, mapping the engine's
        offline + on-device gates onto a ``CompileResult``:

          * Offline gate FIRST (static NKI lint + numpy_impl smoke/parity). A
            lint or shape failure is a compile-blocking error whose reason IS the
            teacher fed back to the next author round — no device time is spent
            on a kernel that cannot even pass the text/shape checks.
          * On device (``nki_available()``): ``build()`` the kernel (import/trace);
            a None result is a real build/trace failure (the "entry function not
            found" class), reported as the error_log. Then run a REAL neuronx-cc
            compile — ``build()`` alone only IMPORTS the module and a ``@nki.jit``
            fn is lowered by neuronx-cc lazily on its FIRST invocation, so a real
            "failed to resolve name"/ISA-validation error would otherwise ESCAPE
            the repair window and die at race time instead of teaching a round-2
            rewrite. ``_device_compile_probe`` forces that lowering and returns
            the compiler error string, which becomes the ``error_log`` the
            ``KernelRepairLoop`` feeds back to the author.
          * Off device: there is no neuronx-cc to run, so an offline-gate PASS is
            the honest best-effort "compiles as far as we can check here" — the
            true device compile is deferred and surfaces downstream as the
            on-device race's ``ran=False`` (device_deferred). Tests inject their
            own ``compile_fn`` to exercise the loop deterministically on CPU.
        """
        offline = self.offline_gate(kernel, spec)
        if not offline.passed:
            return CompileResult(False, error_log=f"offline gate: {offline.reason}")
        if nki_available():
            fn = kernel.build()
            if fn is None:
                return CompileResult(
                    False,
                    error_log=f"device build/trace failed: entry "
                              f"'{kernel.entry}' did not resolve")
            # build() only imported/traced. Force the REAL neuronx-cc lowering so
            # a compile error ("failed to resolve name", ISA validation, ...) is
            # caught INSIDE the repair window and fed back — not at race time.
            compile_err = self._device_compile_probe(fn, spec)
            if compile_err is not None:
                return CompileResult(
                    False, error_log=f"device compile failed: {compile_err}")
            return CompileResult(True, artifact=kernel.entry)
        return CompileResult(True, artifact=f"offline-only:{kernel.entry}")

    def _device_compile_probe(self, fn: Callable, spec: OpSpec) -> str | None:
        """Force a REAL neuronx-cc compile of a built ``@nki.jit`` kernel and
        return the compiler error string (the teacher), or ``None`` if it
        compiled (or could not be probed here).

        Why this exists: a ``@nki.jit`` fn is only lowered by neuronx-cc on its
        FIRST invocation, so ``build()`` (import/trace) succeeds even when the
        kernel will NOT compile. We trigger the lowering by invoking the kernel
        once on device (the SAME proven direct-call path ``_device_race`` uses)
        and capture any compiler error verbatim.

        Returns ``None`` (best-effort "cannot probe — treat build() as far as we
        got") when there is no Neuron device handle or ``torch`` is unavailable:
        without them we cannot compile-invoke, and fabricating an error would be
        dishonest. Device-only, exactly like ``_device_race`` — not exercised on
        a CPU box; tests drive ``_compile`` with a monkeypatched probe.
        """
        device = _neuron_device()
        if device is None:
            return None
        try:
            import torch  # noqa: PLC0415 — device-only import
        except ImportError:
            return None
        try:
            inp = spec.real_inputs()

            def _to_dev(a: np.ndarray):
                return (torch.from_numpy(np.ascontiguousarray(a))
                        .to(torch.bfloat16).to(device))

            args = [_to_dev(inp[k]) for k in _arg_order(spec.name, inp)]
            out = _invoke_kernel(fn, args)
            # Force the lazy neuronx-cc lowering to actually FIRE here, in the repair
            # window, by materializing the output to host — a BARE .cpu() with NO dtype
            # cast. (A dtype cast on-device would re-trip the NCC_ISMP902 Simplifier
            # crash; see _device_race.) Without this materialization the lowering stays
            # lazy and a real compile error escapes the probe, only surfacing later at
            # the race readback instead of being fed back to the author this round.
            out.cpu()
        except Exception as e:  # noqa: BLE001 — compiler errors are the teacher
            return repr(e)
        return None

    def _run_op_with_repair(self, spec: OpSpec, lessons: list, n: int,
                            race_fn: RaceFn | None,
                            compile_fn: CompileFnT | None) -> InventResult:
        """Drive authoring through ``KernelRepairLoop`` so a compile failure
        TEACHES the next attempt (the exact error + the matched rewrite fed back
        via the author's ``feedback`` arg). On convergence the compiled kernel
        goes through the SAME ``_finish`` gates as single-shot; on a
        non-converging loop (exhausted / stalled) the failure is banked as an
        anti-pattern (losses are data), diagnosed with the rewrite catalog."""
        loop = KernelRepairLoop(max_rounds=self.max_repair_rounds)
        _compile = compile_fn or (lambda k: self._compile(k, spec))

        def author_fn(trail: list[Feedback]) -> AuthoredKernel:
            return self.author.author(spec, lessons, trail)

        outcome = loop.run(author_fn, _compile)

        if not outcome.ok:
            last_err = outcome.trail[-1].error_log if outcome.trail else ""
            desc_sfx, reason_sfx = self._diagnose_failure(last_err)
            suggested = ", ".join(r.name for r in outcome.suggested_rewrites)
            reason = (f"kernel repair did not converge in {outcome.rounds} "
                      f"round(s) ({outcome.reason})"
                      + (f"; suggested rewrites: {suggested}" if suggested else ""))
            lid = self._bank_anti_pattern(spec, reason, diagnosis=reason_sfx)
            self._record(spec, Status.DISCARD, 0.0, 0.0,
                         f"repair failed: {reason}{desc_sfx}", n_lessons=n)
            return InventResult(
                spec.name, spec.shape_class, spec.origin, "offline_reject",
                OfflineGate(False, False, float("inf"), reason=reason),
                RaceResult(False, reason=outcome.reason),
                lesson_id=lid, detail=f"{reason}{desc_sfx}", lessons_consulted=n)

        # Compiled after N rounds — gate + race + bank the repaired kernel.
        return self._finish(spec, outcome.kernel, n, race_fn, lessons=lessons)

    def run(self, specs: list[OpSpec],
            race_fn: RaceFn | None = None) -> list[InventResult]:
        results = [self.run_op(s, race_fn=race_fn) for s in specs]
        self._write_summary(results)
        return results

    # -- self-test (validate the EXECUTION path, not authoring quality) ------

    def self_test(self, seed: str = "silu_gate",
                  race_fn: RaceFn | None = None) -> tuple[InventResult, bool, str]:
        """Run a KNOWN-GOOD seed kernel through the full engine to validate the
        on-device EXECUTION path in isolation from authoring quality.

        Returns ``(result, executed, verdict)``. ``executed`` is the pass/fail
        the box cares about: on device it is True iff the kernel actually BUILT,
        INVOKED and was MEASURED (``race.ran`` and no "entry function not found"
        wall); off device it is True as a graceful "deferred" (the CPU-mock path
        cannot run a kernel and that is expected, not a failure).

        A seed (``silu_gate`` / ``rmsnorm`` / ``softmax``) is used on purpose: it
        is a proven-correct formulation, so if IT fails to execute the fault is
        the invocation path, not the authored math. Only once a seed EXECUTES do
        novel kernels have a real shot.
        """
        spec = resolve_ops([seed])[0]
        res = self.run_op(spec, race_fn=race_fn)
        on_device = nki_available()
        if not on_device:
            return res, True, (
                f"OFF-DEVICE: seed {seed!r} authored + offline-gated + "
                f"device-race deferred (status={res.status}); run on trn2 to "
                f"exercise the real build+invoke+measure path")
        executed = _executed_on_device(res)
        if executed:
            verdict = (
                f"ON-DEVICE PASS: seed {seed!r} EXECUTED and was measured "
                f"(status={res.status}, ran={res.race.ran}, "
                f"correct={res.race.correct}, speedup={res.race.speedup:.3f}x) "
                f"— the 'entry function not found' wall is cleared")
        else:
            verdict = (
                f"ON-DEVICE FAIL: seed {seed!r} did NOT execute "
                f"(status={res.status}, ran={res.race.ran}); "
                f"reason={res.race.reason!r}")
        return res, executed, verdict

    # -- reporting -----------------------------------------------------------

    def _write_summary(self, results: list[InventResult]) -> None:
        summary = {
            "on_device": nki_available(),
            "sdk_version": self.sdk_version,
            "invention_margin_pct": self.guards.invention_margin_pct,
            "n_ops": len(results),
            "wins": [r.op for r in results if r.status == "win"],
            "anti_patterns": [r.op for r in results if r.status == "anti_pattern"],
            "offline_rejects": [r.op for r in results
                                if r.status == "offline_reject"],
            "device_deferred": [r.op for r in results
                                if r.status == "device_deferred"],
            "no_author": [r.op for r in results if r.status == "no_author"],
            "results": [
                {
                    "op": r.op, "shape_class": r.shape_class, "origin": r.origin,
                    "status": r.status, "lesson_id": r.lesson_id,
                    "offline_passed": r.offline.passed,
                    # Only report a parity error number when it was an INDEPENDENT
                    # check; a tautological (numpy_impl is reference) comparison
                    # verified no math, so its "0.0" would be misleading -> null.
                    "offline_parity_independent": r.offline.parity_independent,
                    "offline_parity_max_abs_err": (
                        None if (not r.offline.parity_independent
                                 or r.offline.parity_max_abs_err == float("inf"))
                        else r.offline.parity_max_abs_err),
                    "offline_lint": r.offline.lint_violations,
                    "race_ran": r.race.ran,
                    "correct": r.race.correct,
                    "correctness_pct": r.race.correctness_pct,
                    "speedup": r.race.speedup,
                    "detail": r.detail,
                }
                for r in results
            ],
        }
        (self.out_dir / "invent_summary.json").write_text(
            json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# on-device helpers (only reached on trn2)
# ---------------------------------------------------------------------------
def _executed_on_device(res: InventResult) -> bool:
    """True iff the on-device race actually RAN and did not hit the entry wall.

    A win, a correct-but-slow anti-pattern, or a wrong-but-ran anti-pattern all
    count as EXECUTED — the point of the self-test is "did the kernel build,
    invoke and get measured", not "did it win". The one thing that is NOT
    executed is the very failure this fix targets: an "entry function ... not
    found" (or any un-run) race.
    """
    race = res.race
    if not race.ran:
        return False
    reason = (race.reason or "").lower()
    if "entry function" in reason and "not found" in reason:
        return False
    return True


def _arg_order(op: str, inp: dict) -> list[str]:
    """Positional arg order each authored kernel expects (matches nki_src)."""
    order = {
        "rope_apply": ["x", "cos", "sin"],
        "gelu_tanh": ["x"],
        "softcap": ["x", "cap"],
        "add_rmsnorm": ["x", "residual", "gamma"],
        "layernorm": ["x", "gamma", "beta"],
        "attn_decode": ["q", "k", "v"],
        "rmsnorm": ["x", "gamma"],
        "silu_gate": ["x"],
        "softmax": ["x"],
    }.get(op)
    return order if order else list(inp.keys())


def _invoke_kernel(fn: Callable, args: list):
    """Invoke a jitted kernel via the PROVEN beta-3 path: call it directly.

    This routes authored kernels through the SAME mechanism the working
    (moe_fused) kernels use. In this codebase a compiled @nki.jit kernel is run
    by CALLING THE JIT'D CALLABLE DIRECTLY on device tensors and taking element
    ``[0]`` of its (possibly tuple) result — exactly the shape of the proven
    invocation in ``kernels/moe_fused/qwen_with_megakernel.py``:

        kernel_out = get_multilayer_kernel_jit(L)[2](hidden_states, *weights, ...)
        Y = kernel_out[0]

    i.e. the jit builder hands back a callable that is invoked positionally, and
    its output is a sequence whose first element is the result tensor.

    We deliberately do NOT use the previous bespoke
    ``torch_neuronx.nki_hop.wrap_nki(kernel)[1](**kwargs)`` path: an authored
    kernel put through ``wrap_nki`` produced ``entry function
    '<module>.<fn>_kernel' not found`` on every kernel. Calling the ``@nki.jit``
    fn directly (now that ``build()`` gives it a real, importable module + file)
    is the path the compiler can resolve. ``wrap_nki`` is kept ONLY as a labeled
    fallback for the case where a direct call is not supported by the installed
    ``nki`` build; the fallback error (if any) is surfaced verbatim so a real
    "entry not found" is never silently masked. We do NOT use
    ``nki.baremetal`` / ``nki.simulate_kernel`` — those are offline sim only.
    Any failure propagates to ``_device_race``'s handler, which records it as
    data rather than crashing.
    """
    try:
        out = fn(*args)
    except TypeError as exc:
        # Some jit builds return a (spec, meta, callable)-style tuple rather
        # than a directly-callable kernel; try the last callable element, then
        # fall back to the legacy wrap_nki path with a clear provenance tag.
        called = _try_tuple_callable(fn, args)
        if called is not _NO_CALL:
            out = called
        else:
            # Labelled fallback ONLY. ``wrap_nki`` lives in ``torch_neuronx.nki_hop``,
            # which was REMOVED in torch-neuronx 2.9 — guard the optional import so a
            # missing module never aborts the whole race with an ImportError. When
            # it is absent we cannot take this fallback, so surface a clear (non-
            # ImportError) RuntimeError that _device_race records as data.
            try:
                from torch_neuronx.nki_hop import wrap_nki  # noqa: PLC0415
            except ImportError:
                wrap_nki = None
            if wrap_nki is None:
                # FIX 1(b): the #1 masked cause here is an INVOCATION-CONTRACT
                # mismatch — the authored kernel took an ``out=``/destination
                # parameter (an extra required positional arg), so calling it
                # with only the op's input tensors raised a "missing positional
                # argument" TypeError. The old message ("no invocation path")
                # threw that actionable signal away, so the repair loop could
                # not learn to drop the out-param. Surface the REAL TypeError
                # verbatim and name the contract the harness enforces.
                raise RuntimeError(
                    f"authored kernel not directly callable: {exc!s}. The "
                    f"harness invokes the kernel as `out = kernel(*inputs)` with "
                    f"{len(args)} positional input tensor(s) and expects the "
                    f"output tensor to be RETURNED. A kernel that declares an "
                    f"`out=`/destination parameter (an extra required arg) or "
                    f"otherwise mismatches this arity fails here: got "
                    f"{len(args)} positional args but the kernel expects a "
                    f"different count. Re-author the kernel to take exactly the "
                    f"positional inputs and RETURN the output tensor — do NOT "
                    f"take an out param. (The wrap_nki fallback is also "
                    f"unavailable: torch_neuronx.nki_hop was removed in "
                    f"torch-neuronx 2.9 — no invocation path.)"
                ) from exc
            wrapped = wrap_nki(fn)
            out = wrapped[1](*args)
    return out[0] if isinstance(out, (list, tuple)) else out


_NO_CALL = object()


def _try_tuple_callable(fn: Callable, args: list):
    """If ``fn`` is a jit builder returning a tuple, call its last callable.

    Mirrors the ``get_multilayer_kernel_jit(L)[2](...)`` idiom without hardcoding
    the index: pick the last callable element of the returned tuple. Returns
    ``_NO_CALL`` if ``fn`` is not a tuple-returning builder.
    """
    try:
        maybe = fn
        if isinstance(maybe, (list, tuple)):
            callables = [e for e in maybe if callable(e)]
            if callables:
                return callables[-1](*args)
    except Exception:  # noqa: BLE001
        pass
    return _NO_CALL


def _neuron_device():
    """Return the torch_xla Neuron device handle, or None if unavailable.

    None is the honest "we are not really on a device" signal: without a device
    handle we cannot place the torch baseline on the SAME device the kernel runs
    on, so a fair race is impossible and the caller must defer rather than
    compare a CPU wallclock to a device latency. Off-device this simply returns
    None (no torch_xla); the CPU-mock harness never reaches here because
    ``build()`` already returned None.
    """
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
        return xm.xla_device()
    except Exception:  # noqa: BLE001 — no torch_xla / no device is just "defer"
        return None


def _device_timed_ms(run: Callable, device, iters: int = 100,
                      warmup: int = 5) -> float:
    """Synchronized on-device wallclock, ms/iter — the SAME method used for BOTH
    the authored kernel and the torch baseline so the ratio is apples-to-apples.

    The per-batch ``mark_step`` + ``wait_device_ops`` is what turns an on-device
    wallclock into a real device-latency measurement rather than async-dispatch
    noise: without the barrier the enqueue returns immediately and we would be
    timing Python dispatch, not the device. Returns 0.0 on any failure so the
    caller's ``_fair_speedup`` guard turns a broken measurement into a defer.
    """
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415

        for _ in range(warmup):
            run()
        xm.mark_step()
        xm.wait_device_ops()
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        xm.mark_step()
        xm.wait_device_ops()
        return (time.perf_counter() - t0) / iters * 1000.0
    except Exception:  # noqa: BLE001 — a failed measurement must not fabricate a number
        return 0.0


def _fair_speedup(kernel_ms: float, baseline_ms: float,
                  kernel_timing: str, baseline_timing: str) -> float | None:
    """Speedup = baseline/kernel, but ONLY when the two measurements are
    comparable. Returns None otherwise, forcing the caller to device_deferred.

    Comparable means: measured by the SAME method AND both taken ON the device
    (label convention ``"<method>@device"``). This is the guard against the
    exact pre-fix bug — kernel timed by ``nki.benchmark`` DEVICE latency while
    the baseline was a CPU (``@cpu``) wallclock — which produced a physically
    meaningless ratio biased toward the kernel and could bank a FALSE win. A
    non-positive timing is likewise not a real measurement and yields None.
    """
    if kernel_timing != baseline_timing:
        return None
    if not kernel_timing.endswith("@device"):
        return None
    if kernel_ms <= 0.0 or baseline_ms <= 0.0:
        return None
    return baseline_ms / kernel_ms


def _torch_baseline(op: str, inp: dict, device=None):
    """Torch-eager reference the kernel must beat. Mirrors the numpy reference.

    ``device`` (a torch_xla Neuron device) is REQUIRED for a fair race: the
    baseline tensors are placed on the same device the kernel runs on so both
    sides are timed on-device. It defaults to None only for callers that want
    the pure host computation.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    def t(name):
        x = torch.from_numpy(np.ascontiguousarray(inp[name])).to(torch.bfloat16)
        return x.to(device) if device is not None else x

    if op == "rope_apply":
        x, cos, sin = t("x"), t("cos"), t("sin")
        x1, x2 = x[..., 0::2], x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.stack([o1, o2], dim=-1).flatten(-2)
    if op in ("gelu_tanh",):
        x = t("x")
        f = x.shape[-1] // 2
        return F.gelu(x[..., :f], approximate="tanh") * x[..., f:]
    if op == "softcap":
        x = t("x")
        cap = float(inp["cap"][0])
        return torch.tanh(x / cap) * cap
    if op == "add_rmsnorm":
        x, r, g = t("x"), t("residual"), t("gamma")
        h = x + r
        ms = h.pow(2).mean(-1, keepdim=True)
        return h * torch.rsqrt(ms + 1e-6) * g
    if op == "layernorm":
        x, g, b = t("x"), t("gamma"), t("beta")
        return F.layer_norm(x.float(), (x.shape[-1],), g.float(), b.float(),
                            1e-6).to(torch.bfloat16)
    if op == "attn_decode":
        q, k, v = t("q"), t("k"), t("v")
        return F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
    if op == "flash_attention":
        # Dense (materialized-scores) attention matching _flash_attention_reference:
        # q,k,v are [d_head, S]; scores = qᵀ@k [S,S] UNSCALED; out = softmax(scores)@vᵀ
        # [S, d_head]. On torch_xla this lowers to the COMPILER's fused dense
        # attention — the honest bar a streaming/flash NKI kernel must beat (the
        # dense form materializes [S,S], so at long S it is far from SOL / OOMs,
        # which is exactly where a flash kernel wins).
        q, k, v = t("q"), t("k"), t("v")
        scores = q.transpose(-1, -2) @ k                  # [S, S]
        p = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
        return p @ v.transpose(-1, -2)                    # [S, d_head]
    if op == "rmsnorm":
        x, g = t("x"), t("gamma")
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + 1e-6) * g
    if op == "silu_gate":
        x = t("x")
        f = x.shape[-1] // 2
        return F.silu(x[..., :f]) * x[..., f:]
    if op == "softmax":
        return torch.softmax(t("x"), dim=-1)
    raise KeyError(op)


def _bf16_correct(got: np.ndarray, ref: np.ndarray,
                  oracle: np.ndarray | None, oracle_src: str = "") -> tuple:
    """Decide on-device correctness the FAIR way, returning
    ``(correct: bool, correctness_pct: float, note: str)``.

    Pure numpy so the whole gate is unit-testable off-device. Inputs:
      * ``got``    -- the kernel output (fp32, host).
      * ``ref``    -- the fp32 numpy IDEAL (``spec.reference``). Used as the
        YARDSTICK because it is exact host math (no device/host-bf16 artifact),
        NOT as an absolute pass/fail bar (a bf16 kernel cannot meet it at 1e-2).
      * ``oracle`` -- the incumbent bf16 op output (torch-eager bf16 -> fp32
        host), or None. Supplies the incumbent's OWN fp32-miss count as the bar.

    Gate (see the _CORRECT_* module notes): count how many elements the KERNEL
    misses vs the fp32 ideal at the UNCHANGED _BF16_ tol, and how many the
    INCUMBENT bf16 op misses the same way. Correct iff the kernel misses no more
    than ``_CORRECT_FAIL_FACTOR`` x the incumbent's miss-count (bf16 tie-breaking
    can roughly double it), or ``_CORRECT_PPM_FLOOR`` of all elements when the
    incumbent is exact -- whichever is larger. This is "no worse than the op it
    replaces" (an acceptable drop-in), fair to bf16, and not a loosening: a broken
    kernel misses orders of magnitude more and fails. With no oracle we fall back
    to strict ``allclose`` vs fp32 (old behaviour), never a silent pass."""
    shp_ref = got.shape == ref.shape
    # NaN/Inf reject (magnitude guard, part 1): a correct op never introduces a
    # non-finite value where the fp32 ideal is finite. Instant fail, independent
    # of the count/magnitude budgets below.
    nonfinite_bad = (shp_ref and
                     bool((np.isfinite(ref) & ~np.isfinite(got)).any()))
    k_close = (np.isclose(got, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)
               if shp_ref else None)
    k_fail = int((~k_close).sum()) if shp_ref else int(got.size)
    corr_pct = 100.0 * float(k_close.mean()) if shp_ref else 0.0
    # worst abs error over the elements the KERNEL misses (0.0 if it misses none;
    # NaN/Inf here propagates to a failing comparison below, and is also caught by
    # nonfinite_bad — belt and braces).
    k_max_miss_err = (float(np.abs(got[~k_close] - ref[~k_close]).max())
                      if shp_ref and (~k_close).any() else 0.0)
    have_oracle = oracle is not None and oracle.shape == ref.shape
    if have_oracle and shp_ref:
        o_close = np.isclose(oracle, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)
        o_fail = int((~o_close).sum())
        budget = max(int(_CORRECT_FAIL_FACTOR * o_fail),
                     int(_CORRECT_PPM_FLOOR * got.size) + 1)
        count_ok = k_fail <= budget
        # Magnitude guard, part 2: the incumbent's OWN worst miss defines a
        # tolerable bf16 error; when it is exact (no miss) fall back to a
        # data-scaled bf16 tol band so the floor tracks the magnitude of the data.
        o_max_miss_err = (float(np.abs(oracle[~o_close] - ref[~o_close]).max())
                          if (~o_close).any() else 0.0)
        peak = float(np.abs(ref).max()) if ref.size else 0.0
        mag_floor = _CORRECT_MAG_FLOOR * (_BF16_ATOL + _BF16_RTOL * peak)
        mag_bar = max(_CORRECT_MAG_FACTOR * o_max_miss_err, mag_floor)
        mag_ok = (not nonfinite_bad) and (k_max_miss_err <= mag_bar)
        correct = count_ok and mag_ok
        note = (f"oracle={oracle_src}; kernel_no_worse_than_incumbent={correct} "
                f"(k_fail={k_fail} vs incumbent o_fail={o_fail}, budget={budget}; "
                f"k_max_miss_err={k_max_miss_err:.4g} vs mag_bar={mag_bar:.4g} "
                f"[o_max_miss_err={o_max_miss_err:.4g}], nonfinite={nonfinite_bad})")
    else:
        # No incumbent bf16 op to compare against -> strict fp32 gate (old
        # behaviour) PLUS the NaN/Inf reject. Never a silent pass.
        correct = (shp_ref and not nonfinite_bad and
                   bool(np.allclose(got, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)))
        note = (f"oracle={oracle_src}(none); strict-fp32 fallback; "
                f"k_fail={k_fail}; nonfinite={nonfinite_bad}")
    return correct, corr_pct, note


def _bf16_oracle(op: str, inp: dict, device=None):
    """The FAIR correctness reference for a bf16 kernel: the SAME-PRECISION
    incumbent op (torch-eager bf16 — the op the kernel replaces), returned as a
    host fp32 ndarray, plus a provenance label. Returns ``(None, reason)`` if it
    cannot be computed.

    Why bf16, not the fp32 numpy ``spec.reference``: a bf16 kernel cannot match
    fp32 to 1e-2 and neither does the incumbent bf16 op (see the _CORRECT_* module
    notes). The incumbent's own fp32-miss count is the fair bar the kernel is held
    to in _bf16_correct; judging the kernel against fp32 alone spuriously fails
    genuine bf16 kernels.

    Why the value is taken on the HOST (CPU bf16) by default: reading the
    incumbent's ON-DEVICE bf16 output back to host crashes neuronx-cc on this SDK
    (NCC_ISMP902 "is_subset()" Simplifier error on the reduction+broadcast graph
    when a copy-to-host op is present) AND — verified on-silicon — poisons the XLA
    runtime so the SUBSEQUENT on-device speed timing of the same op then returns
    0.0ms and the fair race defers. So an on-device host-read would both fail and
    break the speed race. CPU bf16 is the same LEAF precision (identical bf16
    rounding of the inputs and of h=x+r, h*h; the reduction accumulates in fp32 on
    both CPU torch and the Trainium vector engine) — the property that makes the
    comparison fair. Device residency was only ever a convenience, never the point
    of the gate. Set ``INVENT_ORACLE_ON_DEVICE=1`` to force the on-device host-read
    path on an SDK where that compiler bug is fixed (off by default so it can never
    regress the speed race)."""
    try:
        import torch  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001 — no torch: no oracle, caller falls back
        return None, f"torch unavailable: {e!r}"
    dev_err = ""
    if device is not None and os.environ.get("INVENT_ORACLE_ON_DEVICE") == "1":
        try:
            with torch.no_grad():
                tb = _torch_baseline(op, inp, device=device)
            # HOST cast (never fuse the convert into the graph — NCC_ISMP902).
            return (np.asarray(tb.cpu().to(torch.float32).numpy()),
                    "torch-bf16@device")
        except Exception as e:  # noqa: BLE001 — fall through to CPU bf16
            dev_err = f"; device host-read failed: {type(e).__name__}"
    try:
        with torch.no_grad():
            tb = _torch_baseline(op, inp, device=None)
        return (np.asarray(tb.to(torch.float32).cpu().numpy()),
                f"torch-bf16@cpu{dev_err}")
    except Exception as e:  # noqa: BLE001 — no oracle; caller keeps the fp32 ref
        return None, f"bf16 oracle unavailable: {e!r}"


def _minor_glob(sdk: str) -> str:
    """"2.28.0" -> "2.28.*" so the banked lesson is SDK-stamped (bank requires it)."""
    parts = sdk.split(".")
    return f"{parts[0]}.{parts[1]}.*" if len(parts) >= 2 else sdk


# ---------------------------------------------------------------------------
# spec-file loader — point the engine at an arbitrary NEW op over time.
# ---------------------------------------------------------------------------
def load_specs_from_file(path: Path | str) -> list[OpSpec]:
    """Load OpSpecs from a user .py spec file.

    The file may expose either:
      * ``SPECS``      : list[OpSpec], or
      * ``op_specs()`` : callable returning list[OpSpec].

    A new op only needs a small spec (name + reference fn + shapes); to actually
    author a kernel for it, register a recipe in ``invent_kernels`` (or attach an
    author) — otherwise the engine records it honestly as ``no_author``.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location("invent_user_specs", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec file {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "op_specs") and callable(mod.op_specs):
        specs = list(mod.op_specs())
    elif hasattr(mod, "SPECS"):
        specs = list(mod.SPECS)
    else:
        raise AttributeError(
            f"{path} defines neither SPECS nor op_specs(); one is required")
    for s in specs:
        if not isinstance(s, OpSpec):
            raise TypeError(f"spec file yielded a non-OpSpec: {s!r}")
    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(results: list[InventResult], out_dir: Path,
                  on_device: bool) -> None:
    print("\n=== Stage-4 INVENT run ===")
    print(f"mode: {'ON-DEVICE (trn2)' if on_device else 'CPU-mock (offline gate only; device race deferred)'}")
    for r in results:
        line = f"  [{r.status:>15}] {r.op:<12} ({r.shape_class})"
        if r.offline.passed:
            line += "  offline:PASS"
        else:
            line += f"  offline:REJECT ({r.offline.reason})"
        if r.race.ran:
            line += f"  correct={r.race.correct} speedup={r.race.speedup:.3f}x"
        if r.lesson_id:
            line += f"  banked={r.lesson_id}"
        print(line)
    wins = [r for r in results if r.status == "win"]
    anti = [r for r in results if r.status == "anti_pattern"]
    print(f"\nsummary: {len(wins)} win(s), {len(anti)} anti-pattern(s), "
          f"{sum(1 for r in results if r.status == 'device_deferred')} deferred, "
          f"{sum(1 for r in results if r.status == 'offline_reject')} offline-reject, "
          f"{sum(1 for r in results if r.status == 'no_author')} no-author")
    print(f"artifacts: {out_dir}/results.tsv, {out_dir}/invent_summary.json, "
          f"bank lessons under {out_dir}/knowledge-bank/provisional/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage-4 INVENT engine: author + gate + race + bank NKI kernels.")
    ap.add_argument("--ops", default="write-new",
                    help="comma list of ops or groups (all | write-new | seeds), "
                         "e.g. rope_apply,gelu_tanh,softcap,add_rmsnorm,layernorm,attn_decode")
    ap.add_argument("--out", required=True, type=Path,
                    help="run output dir (results.tsv, summary, bank)")
    ap.add_argument("--bank-root", type=Path, default=None,
                    help="knowledge-bank root (default: <out>/knowledge-bank)")
    ap.add_argument("--spec", type=Path, default=None,
                    help="optional .py spec file adding new ops (SPECS or op_specs())")
    ap.add_argument("--sdk", default=_SDK, help="neuron SDK version stamp")
    ap.add_argument(
        "--self-test", nargs="?", const="silu_gate", default=None,
        metavar="SEED",
        help="FIRST validate the on-device EXECUTION path on a KNOWN-GOOD seed "
             "kernel (default: silu_gate) — build + invoke + measure — to prove "
             "the 'entry function not found' wall is cleared, isolated from "
             "authoring quality. On device, exits non-zero if the seed does NOT "
             "execute; then continues to --ops only if the seed executed.")
    a = ap.parse_args(argv)

    import sys as _sys
    raw = list(_sys.argv[1:] if argv is None else argv)
    ops_explicit = any(t == "--ops" or t.startswith("--ops=") for t in raw)

    # Self-test gate: run a proven seed through the full engine first. If it
    # cannot even execute on device, novel ops cannot either — fail fast.
    if a.self_test is not None:
        st_engine = InventEngine(out_dir=a.out, bank_root=a.bank_root,
                                 sdk_version=a.sdk)
        _res, executed, verdict = st_engine.self_test(a.self_test)
        print("\n=== Stage-4 INVENT self-test (execution path) ===")
        print(f"  {verdict}")
        if nki_available() and not executed:
            print("  -> aborting: fix the invocation path before authoring novel "
                  "kernels.")
            return 1
        # A bare --self-test (no explicit --ops) is a pure execution-path check:
        # exit 0 on pass / deferred, so the box can gate on it. If --ops was
        # given, fall through and run those ops after the seed passed.
        if not ops_explicit:
            return 0

    specs: list[OpSpec] = []
    if a.spec:
        specs.extend(load_specs_from_file(a.spec))
    names = [n for n in a.ops.split(",") if n.strip()] if a.ops else []
    if names:
        # Merge catalog/group ops, skipping any already provided by the spec file.
        have = {s.name for s in specs}
        for s in resolve_ops(names):
            if s.name not in have:
                specs.append(s)
    if not specs:
        ap.error("no ops resolved (use --ops and/or --spec)")

    engine = InventEngine(out_dir=a.out, bank_root=a.bank_root, sdk_version=a.sdk)
    results = engine.run(specs)
    _print_report(results, engine.out_dir, nki_available())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
