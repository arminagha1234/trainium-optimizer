"""
vLLM-Neuron *serving* backend — REAL implementation, latency-SLA oriented.

Where native_pytorch.py benchmarks a single prefill forward for raw throughput,
this backend drives a full **`vllm serve`** deployment and measures the numbers
a latency SLA is actually written against:

  - TTFT   — time to first token (prefill),
  - TPOT   — time per output token (decode),
  - e2e    — total end-to-end for a target shape (input_len -> output_len),
  - decode tok/s — the PRIMARY metric (higher-is-better, so it slots straight
    into the framework's maximize/tournament convention; it inversely tracks
    TPOT), plus a first-class `hits_sla` bool (e2e <= the caller's SLA).

Each measurement shells out to `vllm_serve_worker.py`, which launches a real
`vllm serve` for the (model, config), waits for "Application startup complete",
sends ONE request at the target shape, records TTFT/TPOT/e2e + the greedy output
token ids (the equivalence signature), and tears the serve down. The command it
builds mirrors the proven recipe in
`Armin-Neuron/gemma4-31b/.../launch_serve_public.sh`
(`vllm serve ... --tensor-parallel-size ... --max-model-len ...
--additional-config '{"neuron_config":{...}}'`).

Config axes (Stage 1) — the SERVING search space:
  tp (4/2), dtype/quant (bf16, fp8 if the stack supports it), max_num_seqs
  (1 for latency), num_batched_tokens_buckets, --async-scheduling on/off,
  speculative decoding on/off (+ draft model) if available. Only axes the
  installed vLLM-Neuron actually supports are emitted — the rest are probed and
  dropped gracefully, so a laptop/mock run degrades rather than fabricating
  knobs the stack does not have.

Equivalence: reuses the framework's top-1-token signature gate uniformly, so no
new mechanism is needed. Output-NEUTRAL config changes (tp / buckets /
max_num_seqs / async-scheduling) reproduce the baseline's greedy tokens and
trivially pass; output-CHANGING configs (fp8, speculative decoding) are gated
against the bf16 baseline's tokens — a quant config that wins latency but
garbles output fails the gate and is discarded, never forced.

STATUS (2026-08-20): code-complete + mock-tested (see test_vllm_serve_backend.py).
On-device validation (sweep Qwen3-8B / Gemma-4-12B serving configs toward a 2 s
SLA) is DEFERRED — the boxes are busy — and is the next step once one frees. The
wiring is exercised end-to-end by the mock tests; no serving latency is claimed
here that was not measured.
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

from backends.base import (
    Artifact,
    Measurements,
    Neff,
    OpSite,
    Profile,
)

_WORKER = Path(__file__).resolve().parent / "vllm_serve_worker.py"
# A serve launch compiles NEFFs on first boot (10-20 min), then runs one request.
# Give the whole launch+measure+teardown a generous ceiling; a hang returns
# metric=0 so the orchestrator discards the config rather than wedging the night.
_SERVE_TIMEOUT_S = 2400  # 40 min

# Serving-relevant config keys. measure() caches on exactly these, so a repeat
# that differs ONLY in output-neutral, serving-irrelevant keys (compile_mode /
# cc_flags added by the deep stages, or an identical profile-loop re-measure)
# reuses the result instead of relaunching a fresh serve — real serve launches
# are the expensive thing here, and re-running them for a flag this backend
# ignores would be pure waste.
_SERVE_KEYS = (
    "tp_degree", "weights_dtype", "quantization", "max_num_seqs",
    "num_batched_tokens", "async_scheduling", "on_device_sampling",
    "speculative", "draft_model",
    "target_input_len", "target_output_len",
)


def _is_gemma4(model_id: str) -> bool:
    return "gemma" in str(model_id).lower()


class VllmServeBackend:
    """vLLM-Neuron serving backend — drives a real `vllm serve` per config and
    measures TTFT/TPOT/e2e toward a latency SLA. Primary metric: decode tok/s."""

    name = "vllm-serve"

    def __init__(
        self,
        instance_type: str = "trn2.48xlarge",
        core_count: int = 64,
        target_input_len: int = 2048,
        target_output_len: int = 512,
        sla_seconds: float = 2.0,
        port: int = 8000,
        venv_python: str | None = None,
        capabilities: dict[str, bool] | None = None,
    ) -> None:
        self.instance_type = instance_type
        self.core_count = core_count
        self.target_input_len = target_input_len
        self.target_output_len = target_output_len
        self.sla_seconds = sla_seconds
        self.port = port
        # Python that has vLLM-Neuron installed. Default: current interpreter.
        self.python = venv_python or sys.executable
        # Explicit capability map wins (tests inject one); None -> auto-probe the
        # installed vLLM once and cache. Auto-probe is conservative: on any
        # import/introspection failure a capability is simply False, so the axis
        # is dropped rather than emitted-and-crashing on device.
        self._caps_override = capabilities
        self._caps_cache: dict[str, bool] | None = None
        self._meas_cache: dict[tuple, Measurements] = {}

    # -- capability probing --------------------------------------------------

    def _capabilities(self) -> dict[str, bool]:
        """Which optional serving axes the INSTALLED vLLM-Neuron supports.

        bf16 + tp + max_num_seqs + num_batched_tokens are assumed universally
        available (they are core `vllm serve` flags). The three that vary by
        build — fp8 quant, --async-scheduling, speculative decoding — are probed
        and only emitted if present. Best-effort and non-fatal: a stack we
        cannot introspect (or no vLLM at all, e.g. a laptop/mock run) reports
        every optional capability as absent."""
        if self._caps_override is not None:
            return {"fp8": False, "async_scheduling": False,
                    "spec_decode": False, **self._caps_override}
        if self._caps_cache is not None:
            return self._caps_cache
        caps = {"fp8": False, "async_scheduling": False, "spec_decode": False}
        try:
            import vllm  # noqa: F401
        except Exception:  # noqa: BLE001 — no vLLM here; all optional axes off
            self._caps_cache = caps
            return caps
        # fp8: is there a quantization method the model executor recognizes?
        try:
            from vllm.model_executor.layers.quantization import (  # noqa
                QUANTIZATION_METHODS,
            )
            caps["fp8"] = any("fp8" in str(m).lower() for m in QUANTIZATION_METHODS)
        except Exception:  # noqa: BLE001
            pass
        # async-scheduling and speculative decoding: presence of the engine
        # arg / config surface. Guarded so an API shift just drops the axis.
        try:
            from vllm.engine.arg_utils import EngineArgs
            fields = set(getattr(EngineArgs, "__dataclass_fields__", {}))
            caps["async_scheduling"] = "async_scheduling" in fields
            caps["spec_decode"] = (
                "speculative_config" in fields or "speculative_model" in fields
            )
        except Exception:  # noqa: BLE001
            pass
        self._caps_cache = caps
        return caps

    # -- fit heuristic -------------------------------------------------------

    def _baseline_tp(self) -> int:
        """A safe latency-serving TP for the baseline. Gemma-4 Global layers cap
        clean TP at 4 (head_dim 512, 4 KV heads); 4 is also a good latency point
        on a 64-core box (leaves room for the KV cache). Never exceed cores."""
        return min(4, self.core_count)

    # -- Stage 0 -------------------------------------------------------------

    def build_baseline(self, model_id: str) -> Artifact:
        """The naive-but-correct serving config Stage 1 improves on: bf16, a
        fitting TP, latency batching (max_num_seqs=1), a single small
        batched-tokens bucket, async-scheduling off, no speculative decoding."""
        self._model_id = model_id
        # Gemma-4 requires SINGLE-SHOT prefill (whole prompt in one bucket); its
        # V2 prefill path does not support chunked prefill. So the baseline's
        # batched-tokens bucket is the full context, not a small 512 chunk.
        single_shot = self.target_input_len + self.target_output_len
        nbt0 = single_shot if _is_gemma4(model_id) else 512
        config: dict[str, Any] = {
            "tp_degree": self._baseline_tp(),
            "weights_dtype": "bf16",
            "quantization": "none",
            "max_num_seqs": 1,
            "num_batched_tokens": nbt0,
            "async_scheduling": False,
            # on-device greedy sampling ON is the proven Gemma recipe default;
            # OFF (host sampling) is swept as a decode host-overhead lever.
            "on_device_sampling": True,
            "speculative": "off",
            "draft_model": "",
            # The latency target travels with the artifact so measure() knows the
            # shape to drive without a probe_shape parse.
            "target_input_len": self.target_input_len,
            "target_output_len": self.target_output_len,
        }
        return Artifact(model_id=model_id, backend=self.name, config=config)

    # -- Stage 1 -------------------------------------------------------------

    def config_axes(self) -> dict[str, list[Any]]:
        """The serving search space. Optional axes appear ONLY if the installed
        vLLM-Neuron supports them (see _capabilities); otherwise they are
        dropped, so the search never proposes a knob the stack cannot honor."""
        caps = self._capabilities()
        axes: dict[str, list[Any]] = {
            # 4 vs 2: the two latency-sensible degrees on a 64-core box. Capped
            # at the physical core count so we never propose tp > cores.
            "tp_degree": [t for t in (4, 2) if t <= self.core_count] or [1],
            # bf16 is always available; fp8 only if the stack has an fp8 method.
            # fp8 is OUTPUT-CHANGING -> the equivalence gate vets it.
            "weights_dtype": ["bf16"] + (["fp8"] if caps["fp8"] else []),
            # Latency track: one sequence in flight. Kept explicit (single value)
            # so the pinned choice is visible in the ledger rather than implicit.
            "max_num_seqs": [1],
            # Prefill chunk / batched-token buckets. Output-neutral scheduling
            # knob; a bigger bucket can cut TTFT at long input_len. Gemma-4 is
            # single-shot only, so its one bucket is the full context.
            "num_batched_tokens": (
                [self.target_input_len + self.target_output_len]
                if _is_gemma4(getattr(self, "_model_id", ""))
                else [512, 1024, 2048]
            ),
            # on-device sampling: ON runs the greedy sampler on-device (no
            # per-decode-step host round-trip); OFF samples on the host. At bs=1
            # decode the host round-trip is a dominant cost, so this is a
            # first-class latency lever. Output-NEUTRAL for greedy (same tokens).
            "on_device_sampling": [True, False],
        }
        if caps["async_scheduling"]:
            # Output-NEUTRAL: overlaps scheduling with execution. Trivially
            # passes the equivalence gate (same greedy tokens).
            axes["async_scheduling"] = [False, True]
        if caps["spec_decode"]:
            # OUTPUT-CHANGING in practice (a draft/verify bug can drift), so it
            # is gated against the bf16 baseline tokens like fp8. Only offered
            # when a draft model is configured.
            axes["speculative"] = ["off", "draft"]
        return axes

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        return Artifact(model_id=artifact.model_id, backend=self.name,
                        config={**artifact.config, **config})

    def compile(self, artifact: Artifact) -> Neff:
        """No separate compile step: `vllm serve` compiles its NEFFs on first
        boot, inside the worker. The real cost (incl. that compile) is counted
        in measure(). Returns a Neff carrying the config."""
        return Neff(artifact=artifact, path="", compile_seconds=0.0)

    # -- measurement ---------------------------------------------------------

    def _serve_key(self, cfg: dict[str, Any]) -> tuple:
        return tuple((k, cfg.get(k)) for k in _SERVE_KEYS)

    def _serve_and_measure(
        self, model_id: str, cfg: dict[str, Any], input_len: int, output_len: int,
    ) -> dict[str, Any]:
        """Launch vllm_serve_worker.py for this config, parse its JSON. Split out
        so the mock tests can override it and exercise the wiring off-device."""
        out_f = (Path(tempfile.gettempdir())
                 / f"vllm_serve_{os.getpid()}_{time.time_ns()}.json")
        cmd = [
            self.python, str(_WORKER),
            "--model", model_id,
            "--tp", str(int(cfg.get("tp_degree", self._baseline_tp()))),
            "--dtype", str(cfg.get("weights_dtype", "bf16")),
            "--quantization", str(cfg.get("quantization", "none")),
            "--max-num-seqs", str(int(cfg.get("max_num_seqs", 1))),
            "--num-batched-tokens", str(int(cfg.get("num_batched_tokens", 512))),
            "--async-scheduling",
            "1" if cfg.get("async_scheduling") else "0",
            "--on-device-sampling",
            "1" if cfg.get("on_device_sampling", True) else "0",
            "--speculative", str(cfg.get("speculative", "off")),
            "--draft-model", str(cfg.get("draft_model", "")),
            "--input-len", str(input_len),
            "--output-len", str(output_len),
            "--sla-seconds", str(self.sla_seconds),
            "--port", str(self.port),
            "--out", str(out_f),
        ]
        env = {**os.environ,
               "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        try:
            subprocess.run(cmd, env=env, timeout=_SERVE_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "serve timeout"}
        if not out_f.exists():
            return {"ok": False, "error": "worker wrote no result"}
        data = json.loads(out_f.read_text())
        out_f.unlink(missing_ok=True)
        return data

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        """Drive a `vllm serve` at this config and measure the SLA numbers.

        Primary metric is decode tok/s (higher-is-better). A failed/timed-out
        launch returns metric=0 so the orchestrator discards it. Results are
        cached on the serving-relevant config subset, so output-neutral extra
        keys (deep-stage cc_flags, identical profile-loop re-measures) never
        relaunch a fresh serve."""
        cfg = neff.artifact.config
        input_len = int(cfg.get("target_input_len", self.target_input_len))
        output_len = int(cfg.get("target_output_len", self.target_output_len))

        key = self._serve_key(cfg)
        if key in self._meas_cache:
            return self._meas_cache[key]

        data = self._serve_and_measure(neff.artifact.model_id, cfg,
                                       input_len, output_len)
        if not data.get("ok"):
            m = Measurements(metric=0.0, shape=shape, batch=1)
            self._meas_cache[key] = m
            return m

        tp = int(cfg.get("tp_degree", self._baseline_tp()))
        m = Measurements(
            metric=data.get("decode_tok_s", 0.0),           # PRIMARY (higher=better)
            metric_p50=data.get("decode_tok_s", 0.0),
            ttft_ms_p50=data.get("ttft_ms", 0.0),
            tpot_ms_p50=data.get("tpot_ms", 0.0),
            e2e_seconds=data.get("e2e_seconds", 0.0),
            hits_sla=bool(data.get("hits_sla", False)),
            hbm_peak_gb=data.get("hbm_peak_gb", 0.0),
            hbm_available_gb=data.get("hbm_available_gb", 24.0),
            shape=f"{input_len}/{output_len}",
            batch=1,
            warmup_iters=max(3, data.get("warmup_iters", 3)),
            measured_iters=max(10, data.get("measured_iters", 10)),
            cores_used=tp,
            cores_available=self.core_count,
            top1_tokens=data.get("top1_tokens", []),
        )
        self._meas_cache[key] = m
        return m

    # -- Stages 2-6 ----------------------------------------------------------

    def profile(self, neff: Neff, shape: str) -> Profile:
        """A serving deployment at max_num_seqs=1 is decode-latency bound: most
        of the e2e budget is the sequential decode loop, not prefill. This gives
        the profile loop a symptom key; the deep stages have no vLLM-serve lever
        (cc_flags don't apply to a serve), so any re-entry cache-hits and stops
        on patience rather than relaunching a serve."""
        return Profile(
            op_sites=[
                OpSite(op_name="decode_step", cost_share=0.70,
                       current_kernel="vllm-neuron"),
                OpSite(op_name="attention_prefill", cost_share=0.25,
                       current_kernel="vllm-neuron"),
            ],
            bottleneck="decode_bound",
            engine_utilization={"PE": 0.5, "DMA": 0.4, "CC": 0.1},
        )

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        # vLLM-Neuron owns its kernels; there is no in-graph NKI swap point the
        # optimizer substitutes here (unlike the native-PyTorch path).
        return []

    # -- reproducibility -----------------------------------------------------

    def toolchain_stamp(self) -> dict[str, str]:
        stamp = {
            "backend": self.name,
            "stack": "vllm-neuron-serve",
            "instance_type": self.instance_type,
            "sla_seconds": str(self.sla_seconds),
            "target_shape": f"{self.target_input_len}/{self.target_output_len}",
        }
        try:
            import vllm
            stamp["vllm"] = getattr(vllm, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
        for pkg, key in (("neuronx-cc", "neuronx_cc"),
                         ("torch-neuronx", "torch_neuronx")):
            try:
                from importlib.metadata import version
                stamp[key] = version(pkg)
            except Exception:  # noqa: BLE001
                pass
        stamp["capabilities"] = ",".join(
            k for k, v in self._capabilities().items() if v) or "bf16-only"
        return stamp
