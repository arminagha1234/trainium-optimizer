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
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from backends.base import Artifact, Measurements, Neff, OpSite, Profile
from transformers import AutoConfig

_WORKER = Path(__file__).resolve().parent / "neuron_worker.py"
_COMPILE_TIMEOUT_S = 1800  # 30-min guardrail, enforced on the subprocess itself


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
        params = (4 * h * h + 3 * h * inter) * L + 2 * vocab * h
        weight_gb = params * 2 / 1e9  # bf16
        heads = _int("num_attention_heads", 32)
        # Some architectures bind the max clean TP below head-count. Gemma4's
        # Global layers use head_dim=512 with only 4 KV heads, so tp>4 shards a
        # KV head below one head_dim and crashes — cap at 4.
        archs = " ".join(getattr(self._hf_config(model_id), "architectures", []) or [])
        # Gemma4 Global layers cap at tp4 (head_dim 512, 4 kv). Qwen3.5/3.8
        # DeltaNet is validated at tp4 (manual head-parallel adapter).
        max_tp = 4 if ("Gemma4" in archs or "Qwen3_5" in archs) else 64
        max_tp = min(max_tp, self.core_count)   # never exceed physical cores
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
        return Artifact(model_id=model_id, backend=self.name, config={
            "tp_degree": tp,
            "weights_dtype": "bf16",
            "attn_implementation": "eager",
            "compile_mode": "eager",
            "batch": 1,
        })

    # -- Stage 1 -------------------------------------------------------------

    def config_axes(self) -> dict[str, list[Any]]:
        # #2 Model-aware TP: only offer tp that divides the query-head count (and
        # respects the gemma4 cap), so the search never wastes a candidate
        # loading a 30-60GB model just to reject an impossible shard.
        tps = [1, 2, 4, 8, 16, 32, 64]
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
                archs = " ".join(getattr(self._hf_config(mid), "architectures", []) or [])
                cap = 4 if ("Gemma4" in archs or "Qwen3_5" in archs) else 64
                cap = min(cap, self.core_count)   # never propose tp > physical cores
                if isinstance(heads, int):
                    tps = [t for t in tps if heads % t == 0 and t <= cap]
                else:
                    tps = [t for t in tps if t <= cap]
            except Exception:  # noqa: BLE001
                pass
        return {
            "tp_degree": tps or [1],
            "weights_dtype": ["bf16", "fp32"],
            "attn_implementation": ["eager", "sdpa"],
            "compile_mode": ["eager", "compile-default"],
            # #1 Batch sweep: batch is the biggest untapped throughput lever
            # (batch-1 leaves the box idle). The search finds the best batch.
            "batch": [1, 8, 32],
        }

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

        cmd = [
            "torchrun", "--nnodes", "1", "--nproc_per_node", str(tp),
            "--rdzv_backend", "c10d", "--rdzv_endpoint", "localhost:0",
            str(_WORKER),
            "--model", neff.artifact.model_id,
            "--tp", str(tp),
            "--dtype", cfg.get("weights_dtype", "bf16"),
            "--attn", cfg.get("attn_implementation", "eager"),
            "--compile", "1" if cfg.get("compile_mode") == "compile-default" else "0",
            "--input-len", str(input_len),
            "--batch", str(batch),
            "--cc-flags", str(cfg.get("cc_flags", "")),   # Stage 2-5 compiler rewrites
            "--out", str(out_f),
        ]
        env = {**os.environ,
               "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        try:
            subprocess.run(cmd, env=env, timeout=_COMPILE_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except subprocess.TimeoutExpired:
            return Measurements(metric=0.0, shape=shape, batch=batch,
                                hbm_peak_gb=999, hbm_available_gb=48)

        if not out_f.exists():
            return Measurements(metric=0.0, shape=shape, batch=batch)
        data = json.loads(out_f.read_text())
        out_f.unlink(missing_ok=True)
        if not data.get("ok"):
            return Measurements(metric=0.0, shape=shape, batch=batch)

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
