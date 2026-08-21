"""
Diffusion backend (native PyTorch / TorchNeuron, Beta 3) — the text-to-image
sibling of native_pytorch.NativePyTorchBackend.

Same Backend protocol (backends/base.py), same launch-a-worker-subprocess-and-
parse-JSON pattern, same equivalence-signature-in-`top1_tokens` contract — so
the optimizer core, beam search, guardrails, ledger and reporting all work
UNCHANGED. Only the measurement leaf differs: it shells out to
`diffusion_worker.py` (a real UNet+VAE denoise on Neuron) instead of
`neuron_worker.py` (causal-LM prefill), and the metric is IMAGES/SEC (with
step-latency), not tok/s.

Config axes (Stage 1): weights_dtype, attn_implementation, compile_mode, steps,
plus the correctness-gated component-placement axis introduced in PR #5. A
diffusion pipeline is exactly the separable-component case that axis was built
for: the scheduler and the text-encoder each get a `place:<component>` axis
(cpu vs device). The known-safe default (validated on SD-Turbo) keeps both on
CPU — the scheduler because its bf16 solver drifts over sequential steps (the
Wan 2.2 evidence in base.placement_axes) and the CLIP text-encode because a
one-shot host encode avoids the LNC flag conflict and costs no HBM. The device
placements are proposed as normal search candidates and kept only if they are
faster AND still pass equivalence — never assumed. These are real, measurable
knobs on a diffusion denoise loop.

Install: drop this file (and diffusion_worker.py) next to native_pytorch.py in
implementation/src/backends/, and register it in overnight._make_backend as
backend name "diffusion-native".
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
    Artifact, Measurements, Neff, OpSite, Profile,
    placement_axes, placement_axis_key,
)

_WORKER = Path(__file__).resolve().parent / "diffusion_worker.py"
_COMPILE_TIMEOUT_S = 1800  # 30-min guardrail, enforced on the subprocess itself


def _parse_shape(shape: str) -> tuple[int, int, int]:
    """'512x512 x1step' -> (height, width, steps). Tolerant of missing parts."""
    import re
    h = w = 512
    steps = 1
    m = re.search(r"(\d+)\s*[xX]\s*(\d+)", shape)
    if m:
        h, w = int(m.group(1)), int(m.group(2))
    m2 = re.search(r"x?\s*(\d+)\s*step", shape)
    if m2:
        steps = int(m2.group(1))
    return h, w, steps


class DiffusionBackend:
    """Native-PyTorch text-to-image backend — drives real Trainium hardware."""

    name = "diffusion-native"

    def __init__(self, instance_type: str = "trn2.3xlarge",
                 core_count: int = 4, venv_python: str | None = None) -> None:
        self.instance_type = instance_type
        self.core_count = core_count
        self.python = venv_python or sys.executable
        self._model_id: str | None = None

    # -- Stage 0 -------------------------------------------------------------

    # Separable components a diffusion pipeline exposes to the PR #5 placement
    # axis, with their known-safe (validated) default placement. Both default to
    # CPU: the scheduler's bf16 solver drifts over sequential steps (Wan 2.2
    # evidence, base.placement_axes) and the one-shot CLIP text-encode is kept on
    # the host to dodge the LNC flag conflict and use no HBM. The `device`
    # placements are searched and equivalence-gated, never assumed.
    _PLACEABLE_COMPONENTS = ["scheduler", "text_encoder"]
    _DEFAULT_PLACEMENT = {"scheduler": "cpu", "text_encoder": "cpu"}

    def _placeable_components(self) -> list[str]:
        """A text-to-image diffusion pipeline always exposes a scheduler and a
        text-encoder (this backend only ever drives diffusion models), so — unlike
        the causal-LM backend, which sniffs the HF config and usually returns [] —
        this is unconditionally the two-component list. The placement axis is thus
        always active for the diffusion family."""
        return list(self._PLACEABLE_COMPONENTS)

    def build_baseline(self, model_id: str) -> Artifact:
        """Naive baseline: bf16 / eager attn / no compile / minimal steps.
        Stage 1 improves on it. Diffusion is tp=1 on this box (single UNet)."""
        self._model_id = model_id
        config = {
            "tp_degree": 1,
            "weights_dtype": "bf16",
            "attn_implementation": "eager",
            "compile_mode": "eager",
            "steps": 1,
            "batch": 1,
        }
        # Seed the baseline with the known-safe placement for each separable
        # component (PR #5). The placement axis then searches the device
        # alternative, gated by measure() + equivalence.
        for comp in self._placeable_components():
            config[placement_axis_key(comp)] = self._DEFAULT_PLACEMENT.get(comp, "cpu")
        return Artifact(model_id=model_id, backend=self.name, config=config)

    # -- Stage 1 -------------------------------------------------------------

    def config_axes(self) -> dict[str, list[Any]]:
        axes: dict[str, list[Any]] = {
            "tp_degree": [1],
            "weights_dtype": ["bf16", "fp32"],
            "attn_implementation": ["eager", "sdpa"],
            "compile_mode": ["eager", "compile-default"],
            # SD-Turbo is a few-step distilled model: 1-4 steps is the useful band.
            "steps": [1, 2, 4],
            "batch": [1],
        }
        # PR #5 placement axis (device vs CPU per separable component). For a
        # diffusion pipeline this is always the scheduler + text-encoder pair;
        # a device placement that is faster but drifts is caught by the
        # equivalence gate in the tournament, not assumed here.
        axes.update(placement_axes(self._placeable_components()))
        return axes

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        return Artifact(model_id=artifact.model_id, backend=self.name,
                        config={**artifact.config, **config})

    def compile(self, artifact: Artifact) -> Neff:
        """Lazy compile on the first forward inside the worker (like native)."""
        return Neff(artifact=artifact, path="", compile_seconds=0.0)

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        """Launch diffusion_worker.py for this config, parse the JSON.
        A failed/invalid/timed-out run returns metric=0 so it is discarded.
        Only the Stage-0 baseline runs the (slow) CPU-fp32 parity gate; later
        configs rely on the cross-config latent-fingerprint signature."""
        cfg = neff.artifact.config
        tp = int(cfg.get("tp_degree", 1))
        batch = int(cfg.get("batch", batch))
        h, w, shape_steps = _parse_shape(shape)
        steps = int(cfg.get("steps", shape_steps))
        run_parity = 1 if cfg.get("run_parity", False) else 0
        # PR #5 placement: pass the searched device-vs-CPU placement of each
        # separable component through to the worker. Default to the known-safe
        # CPU placement (the validated SD-Turbo path) when the axis is absent.
        place_sched = cfg.get(placement_axis_key("scheduler"), "cpu")
        place_txt = cfg.get(placement_axis_key("text_encoder"), "cpu")
        out_f = Path(tempfile.gettempdir()) / f"diff_meas_{os.getpid()}_{time.time_ns()}.json"

        base = [
            str(_WORKER),
            "--model", neff.artifact.model_id,
            "--tp", str(tp),
            "--dtype", cfg.get("weights_dtype", "bf16"),
            "--attn", cfg.get("attn_implementation", "eager"),
            "--compile", "1" if cfg.get("compile_mode") == "compile-default" else "0",
            "--steps", str(steps),
            "--height", str(h), "--width", str(w),
            "--batch", str(batch),
            "--parity", str(run_parity),
            "--place-scheduler", str(place_sched),
            "--place-text-encoder", str(place_txt),
            "--cc-flags", str(cfg.get("cc_flags", "")),
            "--out", str(out_f),
        ]
        if cfg.get("image_out"):
            base += ["--image-out", str(cfg["image_out"])]
        # tp=1 runs standalone (no torchrun/dist needed); tp>1 uses torchrun.
        if tp > 1:
            cmd = ["torchrun", "--nnodes", "1", "--nproc_per_node", str(tp),
                   "--rdzv_backend", "c10d", "--rdzv_endpoint", "localhost:0"] + base
        else:
            cmd = [self.python, "-u"] + base

        env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TOKENIZERS_PARALLELISM": "false"}
        try:
            subprocess.run(cmd, env=env, timeout=_COMPILE_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except subprocess.TimeoutExpired:
            return Measurements(metric=0.0, shape=shape, batch=batch,
                                hbm_peak_gb=999, hbm_available_gb=24)

        if not out_f.exists():
            return Measurements(metric=0.0, shape=shape, batch=batch)
        data = json.loads(out_f.read_text())
        out_f.unlink(missing_ok=True)
        if not data.get("ok"):
            return Measurements(metric=0.0, shape=shape, batch=batch)

        # A run whose self-contained decode-parity gate REVIEWed is not a valid
        # measurement (device output diverged from the fp32 reference): metric=0
        # so the orchestrator discards it — even before the cross-config gate.
        if data.get("parity_run") and not data.get("parity_ok", True):
            return Measurements(metric=0.0, shape=shape, batch=batch,
                                top1_tokens=data.get("top1_tokens", []))

        return Measurements(
            metric=data["images_s"],                 # IMAGES/SEC (not tok/s)
            metric_p50=data["images_s"],
            metric_p99=data.get("images_s", 0.0),
            ttft_ms_p50=data.get("step_latency_ms", 0.0),
            ttft_ms_p99=data.get("step_latency_p99_ms", 0.0),
            hbm_peak_gb=data.get("hbm_peak_gb", 0.0),
            hbm_available_gb=data.get("hbm_available_gb", 24.0),
            mfu_percent=-1.0,
            shape=shape, batch=batch,
            cores_used=int(cfg.get("tp_degree", 1)),
            cores_available=self.core_count,
            warmup_iters=data.get("warmup_iters", 3),
            measured_iters=data.get("measured_iters", 10),
            top1_tokens=data.get("top1_tokens", []),   # latent-fingerprint signature
        )

    # -- Stages 2-5 ----------------------------------------------------------

    def profile(self, neff: Neff, shape: str) -> Profile:
        """A diffusion denoise step is UNet-dominated (conv+attention), with the
        VAE decode a one-shot tail. Gives the symptom key the bank queries on."""
        cfg = neff.artifact.config
        return Profile(
            op_sites=[
                OpSite(op_name="unet_attention", cost_share=0.40,
                       current_kernel=cfg.get("attn_implementation", "eager")),
                OpSite(op_name="unet_conv", cost_share=0.40, current_kernel="dense"),
                OpSite(op_name="vae_decode", cost_share=0.15, current_kernel="dense"),
            ],
            bottleneck="compute_bound",
            engine_utilization={"PE": 0.55, "DMA": 0.30, "CC": 0.0},
        )

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        return [
            OpSite(op_name="unet_attention", cost_share=0.40),
            OpSite(op_name="unet_conv", cost_share=0.40),
        ]

    # -- reproducibility -----------------------------------------------------

    def toolchain_stamp(self) -> dict[str, str]:
        stamp = {"backend": self.name, "stack": "native-pytorch-beta3",
                 "device_string": "neuron", "instance_type": self.instance_type,
                 "task": "text-to-image-diffusion"}
        try:
            import torch, torch_neuronx  # noqa
            stamp["torch"] = torch.__version__
            stamp["torch_neuronx"] = getattr(torch_neuronx, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
        try:
            import diffusers
            stamp["diffusers"] = diffusers.__version__
        except Exception:  # noqa: BLE001
            pass
        for pkg, key in (("neuronx-cc", "neuronx_cc"), ("nki", "nki")):
            try:
                from importlib.metadata import version
                stamp[key] = version(pkg)
            except Exception:  # noqa: BLE001
                pass
        return stamp
