"""kernel_registry.py — route a novel architecture *primitive* to the NKI kernel
that implements it, and look up which authored kernels are actually AVAILABLE on
this install.

The philosophy (borrowed from the Neuron team's AutoFixer kernel-authoring loop):
**never label a novel primitive a permanent "blocker".** A model whose primitive
the compiler can't auto-lower — linear-attention / GatedDeltaNet, Mamba / SSM,
MLA, RWKV, ... — is not impossible; it needs a *kernel*. neuronx-cc failing to
lower the naive PyTorch (the in-place scatter + O(T) scan in
`torch_chunk_gated_delta_rule` is the canonical example) means "route to a
kernel", not "drop the model". This module is that routing layer:

    primitive  --PRIMITIVE_TO_KERNEL-->  kernel name  --KernelRegistry-->  is it available?

## IP / public-repo boundary (load-bearing)

This module ships ONLY the name map + the registry INTERFACE. Actual NKI kernel
source is proprietary and lives OUTSIDE this (public) repo, in a directory
pointed to by ``$TRN_OPT_KERNEL_DIR`` (or passed explicitly). The registry reads
a tiny per-kernel ``kernel.json`` manifest (name, status, entry point,
tolerances) — never the source — so "is a DeltaNet kernel available and how good
is it" can be answered without a proprietary kernel ever entering this repo. The
public framework carries the orchestration; the kernels plug in privately.

## Outcome ladder

A kernel is only worth *reusing* once it is at least numerically correct. The
ladder mirrors AutoFixer's, and encodes the single most important lesson from
that corpus: **`nki.simulate` passing OVERSTATES hardware readiness** — a Mamba
selective-scan simulated to 2e-7 ran ~67 max_abs_diff off on real Trn2. So an
on-device pass is a strictly higher tier than a simulate-only pass, and only an
on-device kernel may be reused as HW-ready without re-verification.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Primitive descriptor -> canonical kernel-corpus name. CLASS-level domain
# knowledge (not a per-model hardcode): any model exposing the descriptor reuses
# the same kernel. Descriptors are normalized (lower-case, separators stripped)
# before lookup, so "gated_delta_net", "GatedDeltaNet" and "gated-delta" all hit
# the same entry. Names match the AutoFixer corpus so an external kernel dir laid
# out as <name>/ resolves directly.
PRIMITIVE_TO_KERNEL: dict[str, str] = {
    # linear attention / gated delta rule (Qwen3.5 / Qwen3-Next GDN mixer)
    "linearattention": "DeltaNet",
    "lineattn": "DeltaNet",
    "linearattn": "DeltaNet",
    "gateddeltanet": "DeltaNet",
    "gateddelta": "DeltaNet",
    "gateddeltarule": "DeltaNet",
    "deltarule": "DeltaNet",
    "deltanet": "DeltaNet",
    "hybridssmattn": "DeltaNet",   # hybrid linear+full → the linear path is DeltaNet
    # state-space / Mamba
    "mamba": "Mamba2",
    "mamba1": "Mamba2",
    "mamba2": "Mamba2",
    "ssm": "Mamba2",
    "selectivescan": "Mamba2",
    "mamba2ssd": "Mamba2",         # mamba2_ssd (harvested SSD kernel, dir Mamba2)
    "ssmscan": "Mamba2",           # ssm_scan
    # other linear-recurrent families the corpus covers
    "rwkv6": "RWKV6",
    "rwkv7": "RWKV7",
    "lightningattention": "LightningAttn",
    "lightningattn": "LightningAttn",
    "mlstm": "mLSTM",
    "slstm": "sLSTM",
    "rglru": "RGLRU",
    "kda": "KDA",           # Kimi Delta Attention
    "kimideltaattention": "KDA",           # kimi_delta_attention
    "gateddeltalinearattention": "KDA",    # gated_delta_linear_attention (KDA path)
    "powerretention": "PowerRetention",
    # dense-attention variants that also need authored kernels
    "mla": "MLA",
    "attentionsink": "AttentionSink",
    # long-context flash attention — the streaming online-softmax kernel that
    # never materializes [S,S] and is the ONLY path that runs S=8192 attention
    # (the compiler OOMs on the dense form). On-device validated (rank 4).
    "flashattention": "FlashAttention",
    "flashattn": "FlashAttention",
    "flash": "FlashAttention",
    "longcontextattention": "FlashAttention",
    "attentionlongcontext": "FlashAttention",
    # --- harvested from AWShtokoyo/vllm-neuron (docs/vllm-neuron-harvest.md) ---
    # Routing INTENT for the newly-catalogued arches: these descriptors route to
    # canonical kernel names whose kernels are not authored yet, so for_primitive
    # returns None until one exists (the loop then authors). Recorded so a model
    # exposing the descriptor is a named work-item, not a silent skip.
    "glmmoedsa": "GlmMoeDsa",              # GLM-5.2 MLA + 256-expert sigmoid MoE (DSA-omitted)
    "glm5": "GlmMoeDsa",
    "mlaabsorbed": "MLA",                  # MLA weight-absorbed decode/prefill -> MLA corpus
    "heteroattention": "FlashAttention",   # Gemma-4 sliding-window + global, head_dim<=512
    "slidingwindowattention": "FlashAttention",
    "gemma4attention": "FlashAttention",
}


def _norm(s: str) -> str:
    """Lower-case + strip non-alphanumerics, so descriptor spellings collapse."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def kernel_for_primitive(primitive: str) -> str | None:
    """Canonical kernel name for a primitive descriptor, or None if unmapped."""
    return PRIMITIVE_TO_KERNEL.get(_norm(primitive))


# Outcome ladder — higher = closer to trustworthy-on-hardware. Only >= PASSED is
# reusable at all; >= PASSED_ON_DEVICE is reusable as HW-ready (see module doc).
STATUS_RANK: dict[str, int] = {
    "analysis-only": 0, "written-not-compiled": 0, "algorithm-documented": 0,
    # a candidate rejected by the adversarial anti-cheat (reward-hack / not a real
    # kernel) is rank 0 — there is nothing to reuse or repair, author for real.
    "failed-adversarial": 0,
    "failed-compile": 1,
    "compiled": 2, "failed-numerical": 2,
    "passed": 3,                       # numerics match via nki.simulate
    "passed-on-device": 4, "on-device-passed": 4, "hardware-validated": 4,
}

# Minimum status a kernel must have before the pipeline will try to USE it.
MIN_USABLE_RANK = STATUS_RANK["passed"]           # simulate-correct
MIN_HW_READY_RANK = STATUS_RANK["passed-on-device"]  # re-usable without re-verify


@dataclass
class KernelSpec:
    """What the registry knows about one authored kernel — from its manifest,
    never its source. `entry`/`path` locate the (proprietary, external) kernel;
    they are opaque strings here."""

    name: str
    status: str = "analysis-only"
    entry: str = ""                    # e.g. "gdn_chunked_prefill_nki:gdn_chunked_prefill_kernel"
    path: str = ""                     # dir/file, resolved against the external kernel dir
    variants: list[str] = field(default_factory=list)   # e.g. ["gated_deltanet"]
    tolerances: dict[str, float] = field(default_factory=dict)  # {"bf16": 8.3e-4, ...}
    backend: str = ""                  # "native-pytorch-beta3" | "vllm-serve" | ...
    notes: str = ""

    @property
    def rank(self) -> int:
        return STATUS_RANK.get(self.status, 0)

    @property
    def usable(self) -> bool:
        """Numerically correct at least in simulation — worth trying."""
        return self.rank >= MIN_USABLE_RANK

    @property
    def hw_ready(self) -> bool:
        """Validated on a real NeuronCore — reusable without re-verification."""
        return self.rank >= MIN_HW_READY_RANK


class KernelRegistry:
    """Pluggable lookup of authored kernels available on THIS install.

    Reads per-kernel ``<kernel_dir>/<KernelName>/kernel.json`` manifests only —
    the proprietary NKI source next to the manifest is never read here. With no
    kernel dir configured the registry is simply EMPTY (every lookup -> None),
    so the public framework runs unchanged and a kernel-harvest route degrades
    to "no kernel available yet" rather than erroring.
    """

    def __init__(self, kernel_dir: str | os.PathLike | None = None,
                 library: Any = None) -> None:
        d = kernel_dir or os.environ.get("TRN_OPT_KERNEL_DIR") or ""
        self.kernel_dir: Path | None = Path(d) if d else None
        # Optional IN-REPO validated kernel library (kernel_library.KernelLibrary).
        # Consulted FIRST in for_primitive so a kernel WE wrote+validated is
        # reused before an external proprietary one and before authoring. Default
        # None => byte-for-byte the previous (external-only) behaviour.
        self.library = library
        self._cache: dict[str, KernelSpec | None] = {}

    def _manifest_path(self, kernel_name: str) -> Path | None:
        if self.kernel_dir is None:
            return None
        return self.kernel_dir / kernel_name / "kernel.json"

    def lookup(self, kernel_name: str) -> KernelSpec | None:
        """The KernelSpec for a kernel by canonical name, or None if this install
        has no (readable) manifest for it. Never raises: a malformed/absent
        manifest is treated as 'not available'."""
        if kernel_name in self._cache:
            return self._cache[kernel_name]
        spec: KernelSpec | None = None
        mp = self._manifest_path(kernel_name)
        if mp is not None and mp.is_file():
            try:
                data = json.loads(mp.read_text())
                spec = KernelSpec(
                    name=data.get("name", kernel_name),
                    status=str(data.get("status", "analysis-only")),
                    entry=str(data.get("entry", "")),
                    path=str(data.get("path", "")),
                    variants=list(data.get("variants", []) or []),
                    tolerances={k: float(v) for k, v in
                                (data.get("tolerances", {}) or {}).items()},
                    backend=str(data.get("backend", "")),
                    notes=str(data.get("notes", "")),
                )
            except Exception:  # noqa: BLE001 — a bad manifest is "not available"
                spec = None
        self._cache[kernel_name] = spec
        return spec

    def for_primitive(self, primitive: str) -> KernelSpec | None:
        """Resolve a primitive descriptor to an available KernelSpec, or None.
        Back-compat shim; delegates to :meth:`for_signature` (which adds a safe
        near-miss fallback but resolves an exact hit identically)."""
        return self.for_signature(primitive)

    def for_signature(self, primitive: str, op_name: str = "") -> KernelSpec | None:
        """Resolve a primitive descriptor (optionally aided by the op NAME) to an
        available KernelSpec, or None.

        Consult order: the in-repo validated kernel library FIRST (a kernel we
        wrote + validated, source in this repo), then the external proprietary
        registry. So "harvest before invent" reuses our own kernels first.

        Name resolution uses ``op_signature.resolve_kernel_name`` — an EXACT
        ``kernel_for_primitive`` hit first (byte-identical to before), then a
        longest-embedded-descriptor fallback on the primitive and, last, on the
        op name. This makes cross-model reuse robust to a near-miss spelling
        (e.g. ``"qwen3next_gated_delta"``) that the exact map would miss, without
        ever changing an existing exact hit."""
        if self.library is not None:
            try:
                lk = self.library.lookup(primitive)
                if lk is not None and lk.usable:
                    spec = self.library.to_kernel_spec(lk)
                    if spec is not None:
                        return spec
            except Exception:  # noqa: BLE001 — a library miss must never break routing
                pass
        try:
            from op_signature import resolve_kernel_name
            name = resolve_kernel_name(primitive, op_name)
        except Exception:  # noqa: BLE001 — fall back to the exact map if unavailable
            name = kernel_for_primitive(primitive)
        return self.lookup(name) if name else None

    def available(self, kernel_name: str) -> bool:
        spec = self.lookup(kernel_name)
        return spec is not None and spec.usable
