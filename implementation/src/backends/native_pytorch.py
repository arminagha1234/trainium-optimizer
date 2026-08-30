"""
Native PyTorch (TorchNeuron / Beta 3) backend — REAL implementation.

Runs HuggingFace models on AWS Trainium via native PyTorch (NOT XLA). Each
measurement shells out to `neuron_worker.py` under `torchrun`, because real
tensor parallelism needs one rank per NeuronCore + init_process_group(
backend="neuron"). The optimizer core stays single-process; the heavy,
multi-rank work happens in the worker subprocess, which writes a JSON result.

Confirmed on-device (2026-08-18, trn2.48xlarge, Beta 3 DLC concourse-release):
  - torch.device("neuron") works (device verify -> 16.0)
  - dist.init_process_group(backend="neuron") + cross-chip all_reduce at TP=8 OK
    (the barrier documented FAILING on Trn1 — it WORKS on Trn2)
  - full TP=8 forward OK (Qwen3-0.6B, logits (1,128,151936), 3.1s)
Toolchain: torch 2.12.1, torch_neuronx 2.12.3.0.0, neuronx_cc 2.27.2878.0, nki 0.6.0

Config axes (Stage 1): tp_degree, weights_dtype, attn_implementation, compile_mode.
These are real, measurable knobs on native PyTorch eager/compile.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from backends.base import (
    Artifact,
    Measurements,
    Neff,
    OpSite,
    Profile,
    placement_axes,
)
from transformers import AutoConfig

from backends.device_reap import reap as _reap

_WORKER = Path(__file__).resolve().parent / "neuron_worker.py"
# 30-min guardrail, enforced on the subprocess itself. Overridable because a
# very large or dequantized checkpoint can spend longer than this just
# loading: DeepSeek-V4-Flash (159.6 GB fp8 -> ~319 GB bf16) hit exactly this
# wall with no compile error, which reads as a failure but is only a budget.
_COMPILE_TIMEOUT_S = int(os.environ.get("TRN_OPT_COMPILE_TIMEOUT_S", "1800"))
# Per-CANDIDATE wall. The baseline is allowed the full _COMPILE_TIMEOUT_S because
# there is nothing to fall back to if it fails, but a single search candidate must
# not be able to spend the whole run: one config burned 10800s and then, via
# stranded ranks, took the ~20 candidates after it down with it. Defaults to a
# quarter of the baseline budget, floored so a slow-but-real compile still fits.
_CONFIG_TIMEOUT_S = int(os.environ.get(
    "TRN_OPT_CONFIG_TIMEOUT_S", str(max(900, _COMPILE_TIMEOUT_S // 4))))
# How long to wait for a killed worker's ranks to release the devices.
_REAP_TIMEOUT_S = int(os.environ.get("TRN_OPT_REAP_TIMEOUT_S", "180"))


def _shape_input_len(shape: str) -> int:
    """'chat 1k/512' -> 1024 input tokens (prefill length is what we bench)."""
    import re
    m = re.search(r"(\d+)\s*k?\s*/", shape)
    if not m:
        return 1024
    tok = m.group(1)
    # crude: the shapes use '1k','10k','512','64k' style before the slash
    raw = shape.split("/")[0].strip().split()[-1].lower()
    if raw.endswith("k"):
        return int(float(raw[:-1]) * 1024)
    try:
        return int(raw)
    except ValueError:
        return int(tok)


_ERR_TAIL_CHARS = 4000

# torchrun --tee prefixes each rank's own stream, e.g. "[rank1]: ...".
_RANK_PREFIX = re.compile(r"^\s*\[rank\d+\]:")

# Ordered, most-specific-first. Recognising the common shapes turns an opaque
# metric=0.0 into an actionable line without needing the full log.
_ERR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("out of memory", "OOM: device ran out of HBM"),
    # Device ACQUISITION failures, before the generic HBM match. A core held by an
    # orphaned rank from a previous candidate is not memory pressure, and calling
    # it OOM sends the next person to shrink the model instead of reaping the box.
    ("NERR_RESOURCE", "device busy: cores held by another process "
                      "(orphaned ranks from an earlier candidate?)"),
    ("nrt_tensor_allocate", "runtime allocation failed (cores held by another process?)"),
    ("EXEC_BAD_INPUT", "runtime rejected the executable"),
    ("Unable to acquire", "device busy: could not acquire a NeuronCore"),
    ("failed to open device", "device busy: could not open a NeuronCore"),
    ("REAP INCOMPLETE", "device busy: a previous candidate's ranks were never "
                        "reaped, so this result is unreliable"),
    ("out of range for device memory", "OOM: device ran out of HBM"),
    # Generic, LAST among the memory patterns: matches any remaining mention of
    # HBM once the env dump and the specific causes above are excluded.
    ("HBM", "OOM / HBM pressure"),
    ("nrt_init", "Neuron runtime failed to initialise"),
    ("No module named", "missing Python dependency in the worker env"),
    ("KeyError: 'architectures'", "model config has no architectures field"),
    ("Dynamic shape is not supported", "dynamic shape reached the compiler "
                                       "(set dynamic=False / pin the batch)"),
    # neuronx-cc crashed inside itself. Not a model or config problem -- the same
    # graph on a fixed compiler would compile. Distinguishing this from
    # "unsupported op" matters: one is escalate, the other is rewrite.
    ("INTERNAL_ERROR", "neuronx-cc INTERNAL ERROR (compiler bug -- escalate, "
                       "do not tune around it)"),
    ("BIRCodeGenLoop", "neuronx-cc INTERNAL ERROR in BIR codegen (compiler bug)"),
    ("not supported", "unsupported architecture/op for this backend"),
    ("CUDA", "worker tried a CUDA path on a Neuron device"),
    # NOT a bare "torch.distributed" match: torchrun's own ChildFailedError
    # wrapper names torch.distributed on EVERY worker failure, so that
    # signature mislabelled unrelated crashes as collective failures (a real
    # AttributeError on rank1 was reported as "collective/TP initialisation
    # failed"). Match only markers that genuinely mean the collective failed.
    ("init_process_group", "collective/TP initialisation failed"),
    ("device barrier", "collective/TP initialisation failed"),
    ("ProcessGroup", "collective/TP initialisation failed"),
    ("aws-ofi-nccl", "collective transport init failed"),
    ("Killed", "worker was killed (OOM-killer / host memory)"),
)


# Files transformers needs to build a model. Deliberately not "*" -- a repo can
# carry GGUF/ONNX/consolidated variants that would never be read but would make a
# completeness check fail and defeat the prewarm.
_HF_WEIGHT_PATTERNS = ("*.safetensors", "*.json", "*.txt", "*.model", "*.py")


def prewarm_hf_cache(model_id: str, log=None) -> bool:
    """Resolve the checkpoint ONCE here, so every rank can then read it locally.

    Returns True when the cache is complete and the workers can be put in
    HF_HUB_OFFLINE mode.

    Why: measurement runs one worker process per core, and each one resolves the
    checkpoint independently. At tp=16 that is 16 concurrent unauthenticated Hub
    lookups; a rate-limited rank falls back to a local lookup that reports an
    already-cached shard as absent::

        [rank6] OSError: Qwen/Qwen3.5-35B-A3B does not appear to have a file named
                model.safetensors-00001-of-00014.safetensors

    A different rank and a different shard each run, which reads exactly like a
    corrupt cache. It is not: the cache was verified complete -- 14/14 shards,
    correct sizes, a real read at the tail of each -- by the parent process
    seconds before a run that then failed this way. One resolution here plus
    offline workers removes the contention entirely, and also stops `tp` ranks
    downloading the same file on a cold cache.

    Never raises. If prewarming fails (offline box, gated repo, no hub package)
    the caller leaves the workers online and they behave as before.
    """
    if not model_id or os.sep in model_id and Path(model_id).exists():
        return False                      # explicit local path: nothing to resolve
    try:
        from huggingface_hub import snapshot_download
    except Exception:  # noqa: BLE001 - advisory only
        return False
    try:
        snapshot_download(model_id, allow_patterns=_HF_WEIGHT_PATTERNS,
                          local_files_only=True)
        return True                       # already complete; no network at all
    except Exception:  # noqa: BLE001 - means "not fully cached", not an error
        pass
    try:
        snapshot_download(model_id, allow_patterns=_HF_WEIGHT_PATTERNS, max_workers=8)
        if log:
            log(f"hf prewarm: cached {model_id} once; workers will read offline")
        return True
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"hf prewarm failed ({e!r}); each rank will resolve on its own")
        return False


def tp_cap_for(archs: str, heads: int | None, linear_value_heads: int | None,
               core_count: int) -> int:
    """Largest TENSOR-PARALLEL degree this architecture can express.

    Single definition, called by both the baseline chooser and the search axis.
    They disagreed before: #121 raised the Qwen3.5 cap from 4 to the head count in
    `_fit_baseline_tp` and missed `config_axes`, so the baseline could run at tp=8
    or tp=32 while the search would never propose above 4 for the very models that
    needed it.

    Bounds, in order of how hard they are:

    * **Gemma4: a hard 4.** Its Global layers use head_dim 512 with only 4 KV
      heads, so tp>4 shards a KV head below one head_dim and crashes.
    * **Query heads.** TP splits attention by head, so nothing above the head count
      is expressible at all.
    * **GatedDeltaNet value heads.** These cannot be replicated the way KV heads
      can (#135): out_proj is row-sharded with an all-reduce, so two ranks holding
      the same value head would have its contribution summed twice. Bounding here
      means such a candidate is never proposed, rather than proposed and then
      raising inside the shard.
    * **Physical cores.**
    """
    cap = 4 if "Gemma4" in archs else (heads or 64)
    if linear_value_heads:
        cap = min(cap, linear_value_heads)
    return max(1, min(int(cap), int(core_count)))


def tp_candidates(heads: int | None, cap: int) -> list[int]:
    """Every TP degree that divides the head count, up to ``cap``.

    Deliberately NOT restricted to powers of two, which is what the axis used to
    offer. A 24-head model then capped at tp=8 and left 40 of a 64-core box idle,
    when tp=12 and tp=24 are both perfectly valid shardings:

        Qwen3.8-27B (24 heads)   powers of two: 1,2,4,8
                                 divisors:      1,2,3,4,6,8,12,24

    Falls back to the power-of-two ladder only when the head count is unknown,
    since without it there is nothing to divide.
    """
    if not heads:
        return [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= cap]
    return [t for t in range(1, cap + 1) if heads % t == 0]


def _is_noise(line: str) -> bool:
    """Separator banners, rule lines and runtime env dumps carry no diagnosis.

    ``nrt_infodump`` is the big one: on ANY runtime error the Neuron runtime dumps
    its configuration, including lines like ``NEURON_RT_MAP_HBM=1``. A bare "HBM"
    signature matches that, so a whole run's worth of device-acquisition failures
    got labelled "OOM / HBM pressure" and pointed at memory, which was fine. The
    dump is never the diagnosis, so it must never be eligible to be picked as one.
    """
    t = line.strip()
    if not t:
        return True
    if "nrt_infodump" in t:
        return True
    # A line made only of separator punctuation ('====', '----', '****', ...).
    return len(set(t)) <= 2 and t[0] in "=-*_#~+ "


def _child_traceback(log_dir: Path) -> str:
    """Best child-rank stderr from a torchrun --log-dir tree, or ''.

    torchrun reports only ChildFailedError on the parent stream; the rank's real
    exception is written per-rank under --log-dir. Prefer whichever rank log
    actually contains a traceback.
    """
    try:
        logs = sorted(log_dir.rglob("stderr.log")) + sorted(log_dir.rglob("*stderr*"))
    except OSError:
        return ""
    best = ""
    for f in logs:
        try:
            txt = f.read_text("utf-8", "replace")
        except OSError:
            continue
        if "Traceback" in txt or "Error" in txt:
            best = txt[-_ERR_TAIL_CHARS:]
    return best


def _worker_failure_reason(rc: int | None, err_tail: str) -> str:
    """Turn a dead worker into one actionable line, keeping the raw tail.

    Picking the *last* non-empty line is wrong in practice: worker tracebacks are
    wrapped in '=' banners, so the last line is a rule and the real exception is
    silently dropped. Prefer, in order: the line that matched a known signature,
    then the last exception-shaped line, then the last line that is not a banner.
    """
    tail = (err_tail or "").strip()
    if not tail:
        return f"worker exited rc={rc} with no stderr"
    lines = [ln for ln in tail.splitlines() if not _is_noise(ln)]
    # Under --tee each rank's own output is prefixed "[rankN]:". Those lines are
    # the CHILD speaking; everything else is torchrun framing. When any exist,
    # restrict the search to them so the launcher's boilerplate cannot outrank
    # the actual exception.
    rank_lines = [ln for ln in lines if _RANK_PREFIX.search(ln)]
    if rank_lines:
        lines = rank_lines
    label, matched = "", ""
    for pat, lbl in _ERR_SIGNATURES:
        hit = next((ln for ln in reversed(lines) if pat.lower() in ln.lower()), "")
        if hit:
            label, matched = lbl, hit
            break
    if not matched:
        matched = next(
            (ln for ln in reversed(lines)
             if any(k in ln for k in ("Error", "Exception", "error:", "Failed", "failed"))),
            lines[-1] if lines else tail[:200],
        )
    head = f"{label}: " if label else ""
    return f"{head}rc={rc}: {matched.strip()[:400]}"


class NativePyTorchBackend:
    """Native PyTorch (TorchNeuron, Beta 3) backend — drives real hardware."""

    name = "native-pytorch-beta3"

    def __init__(self, instance_type: str = "trn2.48xlarge",
                 core_count: int = 64, venv_python: str | None = None) -> None:
        self.instance_type = instance_type
        self.core_count = core_count
        # Python from the Beta 3 venv (has torch_neuronx). Default: current.
        self.python = venv_python or sys.executable
        self._cfg_cache: dict[str, Any] = {}

    # -- helpers -------------------------------------------------------------

    def _hf_config(self, model_id: str):
        if model_id not in self._cfg_cache:
            self._cfg_cache[model_id] = AutoConfig.from_pretrained(
                model_id, trust_remote_code=True)
        return self._cfg_cache[model_id]

    def _raw_config_dict(self, model_id: str) -> dict[str, Any]:
        """The HF config as a plain dict (what capability.estimate_params takes)."""
        cfg = self._hf_config(model_id)
        try:
            d = cfg.to_dict()
        except Exception:  # noqa: BLE001
            return {}
        tc = getattr(cfg, "text_config", None)
        if tc is not None and not isinstance(d.get("text_config"), dict):
            try:
                d["text_config"] = tc.to_dict()
            except Exception:  # noqa: BLE001
                pass
        return d

    def _fit_baseline_tp(self, model_id: str) -> int:
        """Smallest tp whose bf16 weights fit ~30GB/core, dividing both the
        attention- and KV-head counts (so the DTensor plan is valid)."""
        cfg = self._hf_config(model_id)
        cfg = getattr(cfg, "text_config", None) or cfg
        try:
            setattr(cfg, "allow_global_per_layer_attribute_access", True)
        except Exception:  # noqa: BLE001
            pass

        def _int(name, default):
            try:
                v = getattr(cfg, name, default)
                return v if isinstance(v, int) else default
            except Exception:  # noqa: BLE001
                return default

        h = _int("hidden_size", 4096)
        L = _int("num_hidden_layers", 32)
        inter = _int("intermediate_size", 4 * h)
        vocab = _int("vocab_size", 32000)
        # MoE-AWARE. The dense formula below drops the expert dimension, which
        # undercounts a 256-expert model by ~10x -- measured against on-disk
        # weights: Qwen3.5-35B-A3B 7.4 GB dense vs 71.9 GB actual. That made the
        # "< 10 GB/rank" test below pass at tp=1 and put a 72 GB model on one
        # 24 GB core, so it OOM'd before the run started. capability.estimate_params
        # counts (n_experts + shared) x 3 x h x moe_intermediate.
        params = 0
        try:
            from capability import estimate_params as _est
            params, _bd = _est(self._raw_config_dict(model_id))
        except Exception:  # noqa: BLE001 - fall back to the dense estimate
            params = 0
        if not params:
            params = (4 * h * h + 3 * h * inter) * L + 2 * vocab * h
        weight_gb = params * 2 / 1e9  # bf16
        heads = _int("num_attention_heads", 32)
        # Some architectures bind the max clean TP below head-count. Gemma4's
        # Global layers use head_dim=512 with only 4 KV heads, so tp>4 shards a
        # KV head below one head_dim and crashes — cap at 4.
        archs = " ".join(getattr(self._hf_config(model_id), "architectures", []) or [])
        # Gemma4 Global layers cap at tp4 (head_dim 512, 4 kv). Qwen3.5/3.8
        # DeltaNet is validated at tp4 (manual head-parallel adapter).
        # Gemma4 stays at 4: head_dim 512 with 4 KV heads is a HARD limit -- tp>4
        # shards a KV head below one head_dim and crashes.
        #
        # Qwen3_5's 4 was a VALIDATION limit, not an arithmetic one, and it is what
        # made the large MoEs unreachable. The real bound is the head count, which
        # the `heads % tp` test below already enforces, so raise the cap to it:
        #     35B-A3B  (68 GB)  tp=8  ->  8.5 GB/rank
        #     122B-A10B (239 GB) tp=16 -> 14.9 GB/rank
        # both inside the 24 GB/core budget, where a cap of 4 left them at 17 and
        # 60 GB/rank respectively. Nothing above the head count is expressible
        # anyway, so this is the widest correct value rather than a new guess.
        # Same cap the search axis uses, from one definition -- these two drifted
        # apart once already (#121 fixed this site and missed config_axes).
        _lvh = _int("linear_num_value_heads", 0) or None
        max_tp = tp_cap_for(archs, heads, _lvh, self.core_count)
        best = None
        # 24GB per NeuronCore -> keep weights under ~10GB/rank so there is room
        # for activations + compiler scratch. tp only needs to divide the query
        # heads; the worker's GQA->MHA adapter expands K/V when kv<tp.
        for tp in (1, 2, 4, 8, 16, 32, 64):
            if tp <= max_tp and heads % tp == 0:
                best = tp
                if weight_gb / tp < 10:
                    return tp
        return best or 1

    # -- Stage 0 -------------------------------------------------------------

    def build_baseline(self, model_id: str) -> Artifact:
        """Cheap: reads HF *config* only (no weights), picks a fitting baseline
        config. The naive baseline (bf16 / eager attn / no compile) is what
        Stage 1 improves on."""
        tp = self._fit_baseline_tp(model_id)
        self._model_id = model_id   # cached so config_axes() can filter TP by heads
        config: dict[str, Any] = {
            "tp_degree": tp,
            "weights_dtype": "bf16",
            "attn_implementation": "eager",
            "compile_mode": "eager",
            "batch": 1,
        }
        # Seed the baseline with the known-safe placement for any separable
        # component this model exposes (CPU scheduler / device text-encoder —
        # the Wan 2.2 evidence). The placement axis then searches the
        # alternative, gated by measure() + equivalence. Causal LMs expose no
        # such components, so this adds nothing for the LLM seeds.
        for comp in self._placeable_components():
            config[f"place:{comp}"] = self._DEFAULT_PLACEMENT.get(comp, "cpu")
        # Remembered so measure() can tell the baseline from a search candidate and
        # give them different compile budgets. Compared by value, not identity, so
        # a copied/round-tripped config still matches.
        self._baseline_config = dict(config)
        return Artifact(model_id=model_id, backend=self.name, config=config)

    def _config_timeout_s(self, cfg: dict[str, Any]) -> int:
        """Compile/run wall for THIS candidate.

        The baseline gets the full budget because there is no fallback if it fails
        -- losing it means the model is skipped entirely (FAIL_NO_BASELINE). A
        search candidate gets a quarter of it: one candidate spending the whole run
        is how ~20 later candidates were lost.

        When the baseline config is unknown (no build_baseline in this process, e.g.
        a resumed run) the generous budget is used. Mistakenly short-changing the
        baseline costs the whole model; mistakenly being generous to one candidate
        now only costs that candidate, because it gets reaped either way.
        """
        base = getattr(self, "_baseline_config", None)
        if base is None or dict(cfg) == base:
            return _COMPILE_TIMEOUT_S
        return min(_CONFIG_TIMEOUT_S, _COMPILE_TIMEOUT_S)

    # Known-safe default placement per separable component (see base.placement_axes
    # and the Wan 2.2 evidence): the scheduler's bf16 solver drifts on device
    # over many steps, so it defaults to CPU; a one-shot text-encode is a big
    # device win with no drift, so it defaults to device.
    _DEFAULT_PLACEMENT = {"scheduler": "cpu", "text_encoder": "device"}

    # -- Stage 3 (BORROW): fused MoE megakernel for the MoE family -----------

    def _is_moe_model(self, model_id: str | None = None) -> bool:
        """True if the HF config describes a (sparse) MoE causal LM. Best-effort
        off the architecture names / expert-count attributes, mirroring
        _placeable_components()'s config-only detection. Unknown/unloadable
        configs are treated as non-MoE (the borrow is simply not offered)."""
        mid = model_id or getattr(self, "_model_id", None)
        if not mid:
            return False
        try:
            from kernels.moe_fused import is_moe_arch
            cfg = self._hf_config(mid)
            cfg = getattr(cfg, "text_config", None) or cfg
            return is_moe_arch(cfg)
        except Exception:  # noqa: BLE001
            return False

    def moe_kernel_candidates(self, artifact: Artifact) -> list[tuple[str, dict[str, Any]]]:
        """Stage-3 BORROW candidates for this model: swap the HF MoE layer's
        forward with the vendored fused NKI megakernel.

        Returns a list of (ledger-label, config-patch) pairs. EMPTY for a
        non-MoE model (a dense causal LM), so the borrow degrades to a graceful
        no-op there — the same contract as placement_axes([]) for models with
        no separable components. The orchestrator evaluates each patch as a
        normal BORROW candidate, gated by the existing equivalence check; a
        drifting or non-faster kernel is discarded, never forced."""
        if not self._is_moe_model(artifact.model_id):
            return []
        from kernels.moe_fused import FUSED_NKI, MOE_KERNEL_KEY
        return [("moe:fused-nki-megakernel", {MOE_KERNEL_KEY: FUSED_NKI})]

    def _placeable_components(self) -> list[str]:
        """Separable components whose device-vs-CPU placement the search may
        flip. Dense/hybrid causal LMs run entirely on-device and expose NONE,
        so the placement axis is a graceful no-op for them. A diffusion pipeline
        exposes its scheduler and text-encoder — where the Wan 2.2 evidence
        (see base.placement_axes) applies. Detection is best-effort off the HF
        config's architecture names; unknown/unloadable configs expose nothing."""
        mid = getattr(self, "_model_id", None)
        if not mid:
            return []
        try:
            archs = " ".join(
                getattr(self._hf_config(mid), "architectures", []) or [])
        except Exception:  # noqa: BLE001
            return []
        diffusion_markers = ("Diffusion", "UNet", "Transformer2D", "DiT", "Pipeline")
        if any(m in archs for m in diffusion_markers):
            return ["scheduler", "text_encoder"]
        return []

    # -- Stage 1 -------------------------------------------------------------

    def config_axes(self) -> dict[str, list[Any]]:
        # #2 Model-aware TP: only offer tp that divides the query-head count (and
        # respects the gemma4 cap), so the search never wastes a candidate
        # loading a 30-60GB model just to reject an impossible shard.
        tps = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= self.core_count]
        mid = getattr(self, "_model_id", None)
        if mid:
            try:
                cfg = self._hf_config(mid)
                cfg = getattr(cfg, "text_config", None) or cfg
                try:
                    setattr(cfg, "allow_global_per_layer_attribute_access", True)
                except Exception:  # noqa: BLE001
                    pass
                heads = getattr(cfg, "num_attention_heads", None)
                lvh = getattr(cfg, "linear_num_value_heads", None)
                archs = " ".join(getattr(self._hf_config(mid), "architectures", []) or [])
                cap = tp_cap_for(archs, heads if isinstance(heads, int) else None,
                                 lvh if isinstance(lvh, int) else None,
                                 self.core_count)
                tps = tp_candidates(heads if isinstance(heads, int) else None, cap)
            except Exception as e:  # noqa: BLE001
                # Without the config there is no head count to divide, so the full
                # ladder is proposed and the worker's invalid_tp check rejects the
                # impossible ones. That check runs BEFORE any weights load, so the
                # cost is a few process launches -- but say so, because silently
                # sweeping tp the model cannot express looks like a search bug.
                print(f"[axes] could not read {mid} config ({e!r}); proposing the "
                      f"full tp ladder and letting the worker reject invalid ones",
                      file=sys.stderr)
        axes = {
            "tp_degree": tps or [1],
            "weights_dtype": ["bf16", "fp32"],
            "attn_implementation": ["eager", "sdpa"],
            "compile_mode": ["eager", "compile-default"],
            # #1 Batch sweep: batch is the biggest untapped throughput lever
            # (batch-1 leaves the box idle). The search finds the best batch.
            "batch": [1, 8, 32],
        }
        # Placement axis (device vs CPU per separable component). Emitted ONLY
        # for components the model actually exposes — none for causal LMs, so
        # this is a no-op there. A placement that is faster but drifts is caught
        # by the equivalence gate in the tournament, not assumed here.
        axes.update(placement_axes(self._placeable_components()))
        return axes

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        return Artifact(model_id=artifact.model_id, backend=self.name,
                        config={**artifact.config, **config})

    def compile(self, artifact: Artifact) -> Neff:
        """No separate compile step — native PyTorch compiles lazily on the
        first forward inside the worker. Returns a Neff carrying the config;
        the real run (incl. compile time) happens in measure()."""
        return Neff(artifact=artifact, path="", compile_seconds=0.0)

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        """Launch torchrun -> neuron_worker.py for this config, parse the JSON.

        A subprocess timeout of 30 min enforces the compile-kill guardrail.
        A failed/invalid run returns metric=0 so the orchestrator discards it.
        """
        cfg = neff.artifact.config
        tp = int(cfg.get("tp_degree", 1))
        batch = int(cfg.get("batch", batch))   # #1 batch is a searched config axis
        input_len = _shape_input_len(shape)
        out_f = Path(tempfile.gettempdir()) / f"neuron_meas_{os.getpid()}_{time.time_ns()}.json"
        # Per-rank stdout/stderr land here so a child traceback survives even if
        # the captured tail is truncated. Cheap: a few small text files per run.
        _rank_log_dir = Path(tempfile.mkdtemp(prefix="neuron_ranklogs_"))

        cmd = [
            "torchrun", "--nnodes", "1", "--nproc_per_node", str(tp),
            "--rdzv_backend", "c10d", "--rdzv_endpoint", "localhost:0",
            # Without these, torchrun DISCARDS the child traceback: a dead rank
            # surfaces only as ChildFailedError with "error_file: <N/A>" and
            # "traceback : To enable traceback see: ...", so the actual exception
            # (OOM, unsupported arch, collective failure) is never recorded
            # anywhere. --tee=3 mirrors both rank streams onto the parent's
            # stderr, which measure() already captures, and --log-dir keeps a
            # per-rank copy for anything the tail truncates. This is what makes
            # large-model bring-up debuggable at all.
            "--tee", "3", "--redirects", "3", "--log-dir", str(_rank_log_dir),
            str(_WORKER),
            "--model", neff.artifact.model_id,
            "--tp", str(tp),
            "--dtype", cfg.get("weights_dtype", "bf16"),
            "--attn", cfg.get("attn_implementation", "eager"),
            "--compile", "1" if cfg.get("compile_mode") == "compile-default" else "0",
            "--input-len", str(input_len),
            "--batch", str(batch),
            "--cc-flags", str(cfg.get("cc_flags", "")),   # Stage 2-5 compiler rewrites
            "--moe-kernel", str(cfg.get("moe_kernel", "")),  # Stage 3 MoE borrow
            # Generic kernel injection: a JSON descriptor (target/entry/path) is
            # threaded through to the worker's inject_kernel hook, exactly the way
            # moe_kernel is threaded. Empty string => no injection (eager).
            "--kernel", str(cfg.get("kernel", "")),
            "--out", str(out_f),
        ]
        env = {**os.environ,
               "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        # One Hub resolution here beats `tp` concurrent ones in the workers. Only
        # flip the workers offline once the cache is provably complete, so a cold
        # cache still downloads normally.
        if prewarm_hf_cache(neff.artifact.model_id, log=None):
            env["HF_HUB_OFFLINE"] = "1"
        # Capture the worker's stderr instead of discarding it. Every silent
        # metric=0.0 below used to be indistinguishable -- an OOM, an unsupported
        # architecture, a missing dependency and a plain import error all looked
        # identical -- which made bringing up a new/large model a guessing game.
        # Bounded tail only, so a chatty compiler cannot balloon memory.
        _err_tail = ""
        # start_new_session so the worker gets its OWN process group. Without it,
        # a timeout kills torchrun and leaves its rank children holding
        # /dev/neuron*, which wedges every later candidate in the run -- and
        # killpg would hit our own group. See backends/device_reap.py.
        _budget = self._config_timeout_s(cfg)
        _proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, start_new_session=True)
        try:
            _, _err = _proc.communicate(timeout=_budget)
            _err_tail = (_err or b"").decode("utf-8", "replace")[-_ERR_TAIL_CHARS:]
            _rc = _proc.returncode
        except subprocess.TimeoutExpired:
            # Compile/run blew past the wall. Record it as a compile that hit the
            # ceiling (compile_seconds = the budget) rather than a silent 0.0, so
            # the ledger shows WHY this candidate was discarded -- and reap the
            # rank processes, because a timed-out candidate that keeps the cores
            # turns one dead end into a dead run.
            _note = _reap(_proc, log=None, timeout_s=_REAP_TIMEOUT_S)
            return Measurements(metric=0.0, shape=shape, batch=batch,
                                hbm_peak_gb=999, hbm_available_gb=48,
                                compile_seconds=float(_budget),
                                failure_reason=(
                                    f"timeout: exceeded the {_budget}s "
                                    f"compile/run wall; {_note}"))

        # A crashed or signalled torchrun can strand ranks too, not only a timed-out
        # one. Reap before the next candidate is launched, and record it when the box
        # was NOT left clean -- every later candidate is then suspect, which has to be
        # visible in the ledger rather than inferred afterwards.
        _reap_note = ""
        if _rc != 0:
            _n = _reap(None, log=None, timeout_s=_REAP_TIMEOUT_S)
            if _n.startswith("REAP INCOMPLETE"):
                _reap_note = f" [{_n}]"

        if not out_f.exists():
            # Worker never wrote its JSON: it died before reporting. This is the
            # case that matters most for new-model bring-up.
            reason = _worker_failure_reason(_rc, _err_tail)
            # torchrun's own message is a wrapper (ChildFailedError). If a rank
            # log exists, the child's real traceback is in there -- prefer it.
            _child = _child_traceback(_rank_log_dir)
            if _child:
                reason = _worker_failure_reason(_rc, _child)
            reason += _reap_note
            print(f"[measure] worker produced no result (rc={_rc}): {reason}",
                  file=sys.stderr, flush=True)
            return Measurements(metric=0.0, shape=shape, batch=batch,
                                failure_reason=reason)
        data = json.loads(out_f.read_text())
        out_f.unlink(missing_ok=True)
        if not data.get("ok"):
            reason = str(data.get("error") or "").strip() or _worker_failure_reason(_rc, _err_tail)
            reason += _reap_note
            print(f"[measure] worker reported not-ok: {reason}", file=sys.stderr, flush=True)
            return Measurements(metric=0.0, shape=shape, batch=batch,
                                failure_reason=reason)

        return Measurements(
            metric=data["tok_s"],
            metric_p50=data["tok_s"],
            metric_p99=data.get("p99_ms", 0.0) and
            (data["batch"] * data["shape_input_len"] / (data["p99_ms"] / 1000)),
            ttft_ms_p50=data.get("p50_ms", 0.0),
            ttft_ms_p99=data.get("p99_ms", 0.0),
            hbm_peak_gb=data.get("hbm_peak_gb", 0.0),
            hbm_available_gb=data.get("hbm_available_gb", 48.0),
            mfu_percent=data.get("mfu_percent", -1.0),
            shape=shape, batch=batch,
            warmup_iters=3, measured_iters=10,
            top1_tokens=data.get("top1_tokens", []),
            top_logprobs=data.get("top_logprobs", []),   # task_eval distribution gate
            # Real compile time from the worker (first-forward JIT under
            # torch.compile). 0.0 for eager runs — the worker only sets compile_s
            # when --compile is on. This is what makes the orchestrator's
            # compile-timeout guardrail live and the ledger's compile_s honest.
            compile_seconds=float(data.get("compile_s", 0.0)),
            # Stage-3 MoE borrow status from the worker: "swapped: ..." (kernel
            # ran), "eager-fallback: ..." (precondition unmet), or "not-requested".
            # Surfaced in the ledger so the borrow row is honest about whether the
            # fused NKI megakernel actually executed.
            moe_kernel_swap=data.get("moe_kernel_swap", ""),
        )

    # -- Stages 2-5 ----------------------------------------------------------

    def profile(self, neff: Neff, shape: str) -> Profile:
        """Minimal profile: classify bottleneck from the measured config.
        A full NEFF/NTFF profile via neuron-profile is the next increment;
        this gives the symptom key the bank queries on."""
        cfg = neff.artifact.config
        # Prefill at long context on a dense model is matmul-heavy -> compute.
        return Profile(
            op_sites=[
                OpSite(op_name="attention_prefill", cost_share=0.45,
                       current_kernel=cfg.get("attn_implementation", "eager")),
                OpSite(op_name="mlp", cost_share=0.35, current_kernel="dense"),
                OpSite(op_name="rmsnorm", cost_share=0.08, current_kernel="eager"),
            ],
            bottleneck="compute_bound",
            engine_utilization={"PE": 0.6, "DMA": 0.25, "CC": 0.15},
        )

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        return [
            OpSite(op_name="attention_prefill", cost_share=0.45),
            OpSite(op_name="rmsnorm", cost_share=0.08),
        ]

    # -- reproducibility -----------------------------------------------------

    def toolchain_stamp(self) -> dict[str, str]:
        stamp = {
            "backend": self.name,
            "stack": "native-pytorch-beta3",
            "device_string": "neuron",
            "instance_type": self.instance_type,
        }
        try:
            import torch, torch_neuronx  # noqa
            stamp["torch"] = torch.__version__
            stamp["torch_neuronx"] = getattr(torch_neuronx, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
        for pkg, key in (("neuronx-cc", "neuronx_cc"), ("nki", "nki")):
            try:
                from importlib.metadata import version
                stamp[key] = version(pkg)
            except Exception:  # noqa: BLE001
                pass
        return stamp
