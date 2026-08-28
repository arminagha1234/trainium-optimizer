"""
Backend interface — isolates the ~20% of the framework that is
backend-specific, so the XLA -> native-PyTorch migration is an added file
rather than a rewrite.

Everything above this line in the stack (bank, search loop, guardrails,
ledger, reporting, and the NKI kernels themselves) is backend-independent.

Concrete backends:
  vllm_neuron_xla.py   — V1 primary. Production-representative, proven at a large (multiple-x).
  nxdi_xla.py          — Stage 0 baseline producer (what autoport targets).
  native_pytorch.py    — added when the beta is viable and TP=8 works on Trn2.

See ../../../architecture.md#backend-adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# -- component placement -----------------------------------------------------
# A "placement" config axis moves one separable component (a diffusion
# scheduler, a text-encoder, ...) between the device and the CPU. It is a normal
# config axis: the search proposes both placements and the EXISTING measure() +
# equivalence gate keep whichever is faster AND still correct. Placement is
# NEVER hard-forced onto the device — we have direct evidence that backfires:
#
#   Wan 2.2 port, 50-step video diffusion:
#     - scheduler on device: 72.3 s (NOT faster than 71.2 s on CPU) AND output
#       degraded to PSNR 34.7 dB vs the CPU scheduler's 56.2 dB — a bf16
#       reduction/solver drifting over 50 sequential steps.
#     - text-encoder on device: 65 s -> 0.7 s, a large win with no drift.
#
# So placement is PER-COMPONENT and must be gated by both speed and correctness,
# not assumed. See the "placement-device-scheduler-bf16-drift" anti-pattern.
PLACEMENT_PREFIX = "place:"


def placement_axis_key(component: str) -> str:
    """The config-axis key for a component's device-vs-CPU placement."""
    return f"{PLACEMENT_PREFIX}{component}"


def placement_axes(components: list[str]) -> dict[str, list[str]]:
    """Device-vs-CPU placement axes for the given separable components.

    Returns {} for an empty component list, so a backend/model that exposes no
    separable components (e.g. a dense causal LM, which runs entirely on-device)
    contributes no placement candidates — the axis degrades to a no-op rather
    than fabricating knobs the model does not have. The values are ordered
    CPU-first so the safe placement is tried before the device one; the
    equivalence gate, not the ordering, is what actually rejects a bad move.
    """
    return {placement_axis_key(c): ["cpu", "device"] for c in components}


@dataclass
class Artifact:
    """A configured, not-yet-compiled model implementation."""

    model_id: str
    backend: str
    config: dict[str, Any] = field(default_factory=dict)
    workdir: str = ""            # path to the fork/checkout this artifact edits


@dataclass
class Neff:
    """A compiled artifact ready to run. `path` may be empty for eager backends."""

    artifact: Artifact
    path: str = ""
    compile_seconds: float = 0.0


@dataclass
class Measurements:
    """One (shape, batch) measurement. Metric units are track-specific."""

    metric: float                 # tok/s for track A; images/s, RTF, etc. elsewhere
    metric_p50: float = 0.0
    metric_p99: float = 0.0
    ttft_ms_p50: float = 0.0
    ttft_ms_p99: float = 0.0
    hbm_peak_gb: float = 0.0
    hbm_available_gb: float = 0.0
    mfu_percent: float = -1.0
    shape: str = ""
    batch: int = 1
    warmup_iters: int = 0
    measured_iters: int = 0
    # Real compile time observed during this measurement. For lazy/eager
    # backends (native PyTorch) the compile happens on the first forward INSIDE
    # the worker, so it is only known after measure() runs — not at compile().
    # The worker reports it (compile_s in its JSON); measure() threads it here so
    # the orchestrator can enforce the compile-timeout guardrail on the REAL
    # value and record a non-zero compile_s in the ledger. 0.0 means "unknown /
    # not a compiled run" (e.g. eager). Backends that know compile time upfront
    # still set Neff.compile_seconds; the orchestrator uses whichever is > 0.
    compile_seconds: float = 0.0
    # Instance occupancy. cores_used = tp*cp*dp for this deployment;
    # cores_available = the whole instance. A model pinned to its TP group on a
    # bigger box reports low device_utilization here — the signal that made
    # "27B at TP=4 on a 64-core trn2.48xlarge" visible. mfu_percent should be
    # reported against the FULL instance so idle cores drag it down.
    cores_used: int = 0
    cores_available: int = 0
    # Equivalence signature: top-1 predicted token id at the last K positions on
    # a fixed deterministic prompt. The orchestrator compares a candidate's
    # signature against the Stage-0 baseline's to gate correctness for real.
    top1_tokens: list = field(default_factory=list)
    # Per-position top-k (token_id, logprob) at the last K positions on the same
    # deterministic prompt — the DISTRIBUTION, not just the argmax. Enables a
    # task-level correctness gate (logprob/KL agreement vs baseline) that catches
    # a kernel which preserves top-1 but distorts the distribution — the reward-
    # hack surface top1_tokens alone misses. See task_eval.py. Each element is a
    # position: {"ids": [int,...], "logprobs": [float,...]} (aligned, descending).
    # EMPTY by default: additive, and a backend/worker that does not populate it
    # simply yields no task-eval signal (the gate fails closed). Filled by the
    # worker from the same last-K logits it uses for top1_tokens.
    top_logprobs: list = field(default_factory=list)
    # Serving-latency fields (populated by the vllm-serve backend; left at their
    # defaults by every other backend, so this is additive and changes no
    # existing behavior). TTFT already has a home above (ttft_ms_*); these make
    # the rest of a latency-SLA measurement first-class on the measurement type
    # itself, not just in the worker JSON:
    #   tpot_ms   — time-per-output-token (decode), inversely tracks decode tok/s
    #   e2e_seconds — total end-to-end for the target (input_len -> output_len)
    #   hits_sla  — whether e2e met the caller's SLA (e.g. <= 2.0 s)
    tpot_ms_p50: float = 0.0
    e2e_seconds: float = 0.0
    hits_sla: bool = False
    # Stage-3 MoE borrow audit trail. The worker records whether the fused NKI
    # megakernel actually swapped in ("swapped: ...") or silently fell back to
    # eager because a precondition was unmet ("eager-fallback: ..."); "" / the
    # default means no swap was attempted (dense model, or non-borrow candidate).
    # Threaded to the ledger so a reader can tell "kernel ran" from "fell back".
    moe_kernel_swap: str = ""
    # Why this measurement produced metric == 0.0. Empty when the measurement
    # succeeded. Purely additive and defaulted, so a backend that never sets it
    # behaves exactly as before -- but when a worker dies, this is the only place
    # the reason can survive. Without it every failure (OOM, unsupported arch,
    # missing dep, import error) collapses to an indistinguishable metric=0.0,
    # which is what made large-model bring-up undebuggable.
    failure_reason: str = ""

    @property
    def hbm_utilization(self) -> float:
        if self.hbm_available_gb <= 0:
            return 0.0
        return self.hbm_peak_gb / self.hbm_available_gb

    @property
    def device_utilization(self) -> float:
        """Fraction of the instance's cores this deployment actually uses."""
        if self.cores_available <= 0:
            return 0.0
        return self.cores_used / self.cores_available


@dataclass
class OpSite:
    """A point in the graph where a kernel can be substituted (Stages 2-4)."""

    op_name: str                  # e.g. "attention_prefill", "rmsnorm"
    cost_share: float = 0.0       # fraction of step time, from the profile
    current_kernel: str = ""      # what is there now
    shape_signature: dict[str, Any] = field(default_factory=dict)


@dataclass
class Profile:
    """Parsed profiler output. The 'what to fix' signal for Stages 2-5."""

    op_sites: list[OpSite] = field(default_factory=list)
    bottleneck: str = ""          # "compute_bound" | "dma_bound" | "collective_bound" | ...
    engine_utilization: dict[str, float] = field(default_factory=dict)  # PE, DMA, CC, ...
    raw_path: str = ""

    def hottest(self, n: int = 3) -> list[OpSite]:
        return sorted(self.op_sites, key=lambda s: s.cost_share, reverse=True)[:n]


@runtime_checkable
class Backend(Protocol):
    """What every serving backend must provide.

    Kept deliberately small. Anything a stage needs that is not here is, by
    definition, backend-independent and belongs in the core.
    """

    name: str

    def build_baseline(self, model_id: str) -> Artifact:
        """Stage 0. Produce a runnable, correct implementation from a model id."""
        ...

    def config_axes(self) -> dict[str, list[Any]]:
        """Stage 1. Backend-specific knob names and their legal values."""
        ...

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        ...

    def compile(self, artifact: Artifact) -> Neff:
        """No-op (returns immediately, compile_seconds=0) for eager backends."""
        ...

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        ...

    def profile(self, neff: Neff, shape: str) -> Profile:
        ...

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        """Stages 2-4. Where a NKI kernel can be substituted."""
        ...

    def toolchain_stamp(self) -> dict[str, str]:
        """Full version capture for reproducibility. neuronx_cc especially."""
        ...
