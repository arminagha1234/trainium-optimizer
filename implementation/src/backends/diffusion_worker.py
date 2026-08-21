"""
Diffusion benchmark worker — the measurement leaf that runs a TEXT-TO-IMAGE
diffusion model on Trainium under one config and writes a single measurement to
JSON. It is the diffusion analogue of neuron_worker.py (which does causal-LM
prefill tok/s); the optimizer core, search engine, guardrails and ledger are
unchanged — only this measurement leaf is new.

Launched by DiffusionBackend.measure() via either:
    python diffusion_worker.py --model <path> --out <json>          (tp=1)
    torchrun --nproc_per_node=<tp> diffusion_worker.py ...           (tp>1)

Design mirrors the Wan e2e port (kernel_research/e2e_video.py):
  * text-encoder (CLIP) encodes the prompt on CPU  -> cheap, no HBM, avoids the
    LNC flag conflict and keeps the device graph a pure UNet+VAE graph.
  * UNet denoise on neuron, torch.compile(backend="neuron", dynamic=False).
  * VAE decode on neuron in the shipped dtype (bf16).
  * scheduler kept on CPU (host-side numerics); latents re-hosted to neuron for
    the UNet forward, model_output re-hosted to CPU for the scheduler step.
  * everything glued through the diffusers pipeline __call__ so the measured
    wall-clock is a genuine end-to-end denoise, not a sum-of-stages estimate.

METRIC (diffusion, NOT tok/s):
  * step_latency_ms  = median UNet forward latency (the compile-friendly,
    low-variance core number, analogous to the LLM worker's p50 prefill).
  * images_s         = batch / steady-state denoise wall-clock  (the headline).

EQUIVALENCE — two complementary, tensor-based gates (NO token match):
  1. Wan decode-parity gate (self-contained, intra-run): capture the exact
     latent fed to the VAE, decode it on-device in bf16 AND on CPU in fp32
     (the reference), compare PSNR / SSIM. -> parity_ok. This reuses the Wan
     gate verbatim (metrics ported from e2e_video.py).
  2. Cross-config signature: a deterministic quantised fingerprint of the final
     latent (fixed seed) is emitted as `top1_tokens` — an integer list — so the
     orchestrator's EXISTING equivalence mechanism (fraction-of-signature match
     vs the Stage-0 baseline, >=0.75) works unchanged. A config that changes the
     produced image changes the fingerprint and is correctly gated as a bug.

Run standalone (venv activated):
  NEURON_CC_FLAGS="--model-type=unet -O2" \
    python -u diffusion_worker.py --model /home/ubuntu/sd-turbo \
      --dtype bf16 --steps 1 --height 512 --width 512 --compile 1 \
      --parity 1 --image-out /home/ubuntu/kernel_research/sdturbo_out.png \
      --out /tmp/meas.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch


# --- surgical fix (ported from the Wan port): some schedulers' correctors build
# tensors of mixed dtype then call torch.linalg.solve, which errors "A and B must
# have the same dtype". Promote to a common dtype only when they differ, so the
# scheduler runs unchanged otherwise. Harmless for single-step SD-Turbo. ---
_ORIG_SOLVE = torch.linalg.solve


def _solve_dtype_safe(A, B, *a, **k):
    if torch.is_tensor(A) and torch.is_tensor(B) and A.dtype != B.dtype:
        c = torch.promote_types(A.dtype, B.dtype)
        A, B = A.to(c), B.to(c)
    return _ORIG_SOLVE(A, B, *a, **k)


torch.linalg.solve = _solve_dtype_safe


def _r0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _log(msg: str) -> None:
    if _r0():
        print(f"[diffusion-worker] {msg}", flush=True)


def _sync(o):
    """Force the async Neuron queue to finish by pulling a scalar to host."""
    o = getattr(o, "sample", o)
    if isinstance(o, (tuple, list)):
        o = o[0]
    if torch.is_tensor(o):
        float(o.detach().float().flatten()[:1].cpu())
    return o


# ----------------------------------------------------------------------------
# correctness metrics (numpy-only, ported verbatim from Wan e2e_video.py)
# ----------------------------------------------------------------------------
def _gaussian_kernel1d(sigma=1.5, radius=5):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _blur(img, k):
    r = len(k) // 2
    p = np.pad(img, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i, w in enumerate(k):
        out += w * p[i:i + img.shape[0], :]
    p = np.pad(out, ((0, 0), (r, r)), mode="reflect")
    out2 = np.zeros_like(img)
    for i, w in enumerate(k):
        out2 += w * p[:, i:i + img.shape[1]]
    return out2


def ssim_frame(a, b, data_range=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = _gaussian_kernel1d()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = _blur(a, k), _blur(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = _blur(a * a, k) - mu_a2
    sb = _blur(b * b, k) - mu_b2
    sab = _blur(a * b, k) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return float(np.mean(num / den))


try:
    from skimage.metrics import structural_similarity as _sk_ssim

    def ssim_gray(a, b):
        return float(_sk_ssim(a, b, data_range=255.0))
    SSIM_IMPL = "skimage"
except Exception:  # noqa: BLE001
    def ssim_gray(a, b):
        return ssim_frame(a, b)
    SSIM_IMPL = "numpy-manual"


def psnr(a, b, data_range=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20 * math.log10(data_range / math.sqrt(mse)))


def _img_to_uint8(img_bchw):
    """(B,C,H,W) in ~[-1,1] -> uint8 (H,W,C) for the first image in the batch."""
    v = img_bchw
    if torch.is_tensor(v):
        v = v.detach().float().cpu().numpy()
    v = v[0]                              # (C,H,W)
    v = np.transpose(v, (1, 2, 0))        # (H,W,C)
    v = (v / 2.0 + 0.5).clip(0, 1)
    return (v * 255.0).round().astype(np.uint8)


def _gray(img_uint8_hwc):
    return img_uint8_hwc.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def _latent_fingerprint(latent, nbins=64, ntokens=64):
    """Deterministic quantised fingerprint of the final latent -> integer list.
    Used as the cross-config equivalence signature so the orchestrator's existing
    top1-token match gate works unchanged. Robust to tiny bf16 numerical jitter
    (coarse bins) but sensitive to any real change in the produced image."""
    x = latent.detach().float().cpu().flatten()
    n = x.numel()
    if n == 0:
        return []
    # pool into ntokens contiguous chunks -> mean -> quantise to nbins levels
    step = max(1, n // ntokens)
    means = [float(x[i:i + step].mean()) for i in range(0, step * ntokens, step)]
    means = np.array(means[:ntokens], dtype=np.float64)
    lo, hi = float(means.min()), float(means.max())
    if hi - lo < 1e-9:
        return [0] * len(means)
    q = np.clip(((means - lo) / (hi - lo) * (nbins - 1)).round(), 0, nbins - 1)
    return [int(v) for v in q]


# ----------------------------------------------------------------------------
# pipeline build (mirrors Wan build_pipeline, adapted UNet+CLIP instead of DiT+T5)
# ----------------------------------------------------------------------------
def build_pipeline(args, dev):
    from diffusers import AutoPipelineForText2Image

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    _log(f"loading pipeline (dtype={args.dtype}) from {args.model} ...")
    t0 = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False)
    load_s = time.time() - t0
    _log(f"  loaded in {load_s:.1f}s  (unet={type(pipe.unet).__name__})")

    # UNet attention implementation lever (eager vs sdpa), when supported.
    if hasattr(pipe.unet, "set_attn_processor") and args.attn == "sdpa":
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            _log("  UNet attn processor -> AttnProcessor2_0 (sdpa)")
        except Exception as e:  # noqa: BLE001
            _log(f"  sdpa attn processor unavailable ({e}); keeping default")

    # --- CLIP text-encode: placement axis (PR #5). Default 'cpu' is the
    # known-safe, validated path (robust: no LNC conflict, no HBM); 'device'
    # runs the one-shot encode on Neuron, a searched candidate kept only if it
    # is faster AND still equivalent. SD-Turbo is distilled: guidance_scale=0
    # (no CFG), so encode the positive prompt only. ---
    place_txt = getattr(args, "place_text_encoder", "cpu")
    if place_txt == "device":
        _log("encoding prompt with CLIP on device (placement=device)...")
        pipe.text_encoder.to(device=dev, dtype=dtype)
        enc_dev = dev
    else:
        _log("encoding prompt with CLIP on CPU (fp32 compute)...")
        pipe.text_encoder.to(device="cpu", dtype=torch.float32)
        enc_dev = torch.device("cpu")
    t0 = time.time()
    with torch.no_grad():
        pe, npe = pipe.encode_prompt(
            prompt=args.prompt, device=enc_dev,
            num_images_per_prompt=args.batch,
            do_classifier_free_guidance=False, negative_prompt=None)
    t_txt = time.time() - t0
    pe = pe.to(dtype)
    _log(f"  CLIP encode ({place_txt}): {t_txt:.2f}s  embeds={tuple(pe.shape)}")

    # free the text encoder, place UNet + VAE on neuron
    pipe.text_encoder = None
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2 = None
    gc.collect()
    pipe.unet.to(dev)
    pipe.vae.to(dev)
    # force the pipeline's execution device to neuron (text_encoder is gone)
    type(pipe)._execution_device = property(lambda self: dev)
    pe = pe.to(dev)

    T = {}
    CAP = {}

    # compile the UNet (our win) + instrument the denoise loop
    if args.compile:
        pipe.unet.forward = torch.compile(
            pipe.unet.forward, backend="neuron", dynamic=False)
    _unet = pipe.unet.forward

    def _to_dev(x):
        return x.to(dev) if torch.is_tensor(x) else x

    def unet_fwd(*a, **k):
        a = tuple(_to_dev(x) for x in a)
        k = {kk: _to_dev(vv) for kk, vv in k.items()}
        t = time.time()
        r = _unet(*a, **k)
        _sync(r)
        T["unet"] = T.get("unet", 0.0) + time.time() - t
        T["unet_n"] = T.get("unet_n", 0) + 1
        return r
    pipe.unet.forward = unet_fwd

    # Scheduler placement axis (PR #5). Default 'cpu' keeps the scheduler
    # entirely on the host (host-side numerics), like the Wan port — the
    # known-safe placement, because a bf16 solver drifts over sequential steps.
    # 'device' leaves the scheduler running on device tensors (no re-hosting): a
    # searched candidate kept only if faster AND still equivalence-parity-clean.
    place_sched = getattr(args, "place_scheduler", "cpu")
    if place_sched == "cpu":
        _set_ts = pipe.scheduler.set_timesteps

        def set_ts_cpu(num_inference_steps=None, device=None, **kw):
            return _set_ts(num_inference_steps=num_inference_steps,
                           device=torch.device("cpu"), **kw)
        pipe.scheduler.set_timesteps = set_ts_cpu

        _sched_step = pipe.scheduler.step

        def sched_step_cpu(model_output, timestep, sample, *a, **k):
            mo = model_output.detach().to("cpu")
            smp = sample.detach().to("cpu")
            ts = timestep.to("cpu") if torch.is_tensor(timestep) else timestep
            out = _sched_step(mo, ts, smp, *a, **k)
            if isinstance(out, tuple):
                return (out[0].to(dev),) + tuple(out[1:])
            if hasattr(out, "prev_sample"):
                out.prev_sample = out.prev_sample.to(dev)
            return out
        pipe.scheduler.step = sched_step_cpu
    else:
        _log("scheduler placement=device (no host re-hosting)")

    # instrument + capture the VAE decode (drives the parity gate)
    _vd = pipe.vae.decode

    def vd(latents, *a, **k):
        CAP["latent"] = latents.detach().float().cpu().clone()
        t = time.time()
        r = _vd(latents, *a, **k)
        o = _sync(r)
        T["vae"] = T.get("vae", 0.0) + time.time() - t
        CAP["neuron_decode"] = o.detach().float().cpu().clone()
        return r
    pipe.vae.decode = vd

    return pipe, pe, T, CAP, load_s, t_txt


def _generate(pipe, pe, args, dev):
    """One full denoise. Fixed seed -> deterministic latent for the fingerprint."""
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    kw = dict(
        prompt_embeds=pe,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height, width=args.width,
        num_images_per_prompt=args.batch,
        generator=gen,
        output_type="np",
    )
    t0 = time.time()
    result = pipe(**kw)
    return result, time.time() - t0


# ----------------------------------------------------------------------------
# parity gate — decode the SAME captured latent on CPU fp32 (reference), compare
# ----------------------------------------------------------------------------
def cpu_reference_decode(args, CAP):
    from diffusers import AutoencoderKL
    _log("[gate] loading CPU fp32 VAE for reference decode...")
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32).to("cpu").eval()
    lat = CAP["latent"].to(torch.float32)
    t0 = time.time()
    with torch.no_grad():
        out = vae.decode(lat, return_dict=False)[0]
    _log(f"[gate] CPU fp32 decode: {time.time()-t0:.1f}s")
    del vae
    gc.collect()
    return out.detach().float().cpu()


def parity_gate(args, CAP):
    """Wan decode-parity gate: Neuron-bf16 vs CPU-fp32 of the same latent."""
    neuron_dec = CAP.get("neuron_decode")
    if neuron_dec is None:
        return {"parity_run": False, "parity_ok": False,
                "note": "no neuron decode captured"}
    cpu_dec = cpu_reference_decode(args, CAP)
    nf = _img_to_uint8(neuron_dec)   # (H,W,C)
    cf = _img_to_uint8(cpu_dec)
    p = psnr(nf, cf)
    s = ssim_gray(_gray(nf), _gray(cf))
    ma = int(np.abs(nf.astype(int) - cf.astype(int)).max())
    ok = (p > 30.0) and (s > 0.95)   # image-VAE gate (bf16 vs fp32, 512x512 RGB)
    _log(f"[decode parity] Neuron-bf16 vs CPU-fp32 (SSIM impl: {SSIM_IMPL})")
    _log(f"    PSNR={p:.1f} dB (gate >30)  SSIM={s:.4f} (gate >0.95)  "
         f"max-abs pixel diff={ma}")
    _log(f"    -> decode parity: {'PASS' if ok else 'REVIEW'}")
    return {"parity_run": True, "parity_ok": bool(ok),
            "psnr_db": p, "ssim": s, "max_abs_pixel_diff": ma}


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1)          # SD-Turbo: 1-4
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)          # images per prompt
    ap.add_argument("--guidance", type=float, default=0.0)   # SD-Turbo: no CFG
    ap.add_argument("--seed", type=int, default=0)
    # PR #5 component placement (device vs CPU). Defaults are the known-safe,
    # SD-Turbo-validated placement: scheduler on CPU (bf16 solver drift over
    # sequential steps) and CLIP text-encode on CPU (no LNC conflict, no HBM).
    # The 'device' placements are searched candidates, equivalence-gated upstream.
    ap.add_argument("--place-scheduler", default="cpu", choices=["cpu", "device"])
    ap.add_argument("--place-text-encoder", default="cpu",
                    choices=["cpu", "device"])
    ap.add_argument("--prompt",
                    default="a photograph of a red fox in a snowy forest, "
                            "sharp focus, high detail")
    # >=3 warmup / >=10 measured to satisfy the framework's measurement-trust
    # guardrail; diffusion steady-state iters are cheap (~0.1-0.26s each).
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--parity", type=int, default=1,
                    help="run the CPU-fp32 decode-parity correctness gate")
    ap.add_argument("--image-out", default="",
                    help="optional: write the generated PNG here")
    ap.add_argument("--cc-flags", default="",
                    help="extra NEURON_CC_FLAGS (Stage 2-5 compiler rewrites)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cc_flags:
        os.environ["NEURON_CC_FLAGS"] = (
            os.environ.get("NEURON_CC_FLAGS", "") + " " + a.cc_flags).strip()

    result = {"ok": False, "model": a.model, "tp": a.tp, "dtype": a.dtype,
              "attn": a.attn, "compile": a.compile, "steps": a.steps,
              "height": a.height, "width": a.width, "batch": a.batch,
              "shape": f"{a.height}x{a.width} x{a.steps}step"}

    def dump(extra: dict) -> None:
        result.update(extra)
        if _r0():
            with open(a.out, "w") as f:
                json.dump(result, f, indent=2)

    # tp>1 would init a process group here; SD-Turbo runs tp=1 (standalone).
    if a.tp > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="neuron")

    dev = torch.device("neuron")

    try:
        import torch_neuronx as _tnx
    except Exception:  # noqa: BLE001
        _tnx = None

    try:
        pipe, pe, T, CAP, load_s, txt_s = build_pipeline(a, dev)

        if _tnx is not None:
            try:
                _tnx.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass

        # WARMUP generation (includes UNet+VAE compile in compile-mode).
        _log("=== WARMUP generation (includes compile) ===")
        t0 = time.time()
        _res, _e = _generate(pipe, pe, a, dev)
        warm_s = time.time() - t0
        compile_s = warm_s if a.compile else 0.0
        _log(f"  warmup e2e {warm_s:.1f}s (compile_s~{compile_s:.1f}), "
             f"unet forwards={T.get('unet_n',0)}")

        # capture the equivalence signature from the warmup's final latent.
        eq_sig = _latent_fingerprint(CAP.get("latent")) if "latent" in CAP else []

        # extra warmups (no timing kept)
        for _ in range(max(0, a.warmup - 1)):
            _generate(pipe, pe, a, dev)

        # TIMED steady-state iterations.
        step_times = []   # per-UNet-forward latency
        e2e_times = []     # per-image denoise wall-clock
        last_res = None
        for _ in range(a.iters):
            T["unet"] = 0.0
            T["unet_n"] = 0
            res, e2e = _generate(pipe, pe, a, dev)
            last_res = res
            n = max(1, T.get("unet_n", 1))
            step_times.append(T["unet"] / n)   # mean forward latency this run
            e2e_times.append(e2e)

        step_times.sort()
        e2e_times.sort()
        step_p50 = statistics.median(step_times)
        step_p99 = step_times[min(len(step_times) - 1, int(0.99 * len(step_times)))]
        e2e_p50 = statistics.median(e2e_times)
        images_s = (a.batch / e2e_p50) if e2e_p50 > 0 else 0.0

        # HBM: real peak from the Neuron runtime, like the LLM worker.
        hbm_peak = 0.0
        hbm_estimated = True
        try:
            if _tnx is not None:
                hbm_peak = float(_tnx.max_memory_allocated()) / 1e9
                if hbm_peak > 0:
                    hbm_estimated = False
        except Exception:  # noqa: BLE001
            hbm_peak = 0.0

        _log(f"images/s={images_s:.3f}  step_latency p50={step_p50*1000:.1f}ms "
             f"p99={step_p99*1000:.1f}ms  e2e_p50={e2e_p50*1000:.0f}ms  "
             f"hbm~{hbm_peak:.1f}GB")

        # optional: write the generated PNG (proof an image came out).
        img_written = ""
        if a.image_out and last_res is not None:
            try:
                from PIL import Image
                arr = last_res.images[0]
                arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(a.image_out)
                img_written = a.image_out
                _log(f"wrote image -> {a.image_out}")
            except Exception as e:  # noqa: BLE001
                _log(f"image write failed (non-fatal): {e}")

        # PARITY gate (Wan decode-parity, self-contained).
        parity = {"parity_run": False, "parity_ok": True}
        if a.parity:
            parity = parity_gate(a, CAP)

        dump({
            "ok": True,
            # diffusion metric (NOT tok/s)
            "images_s": images_s,
            "step_latency_ms": step_p50 * 1000,
            "step_latency_p99_ms": step_p99 * 1000,
            "e2e_ms": e2e_p50 * 1000,
            "unet_forwards_per_image": T.get("unet_n", a.steps),
            "vae_decode_ms": T.get("vae", 0.0) * 1000,
            # generic metric fields the backend/orchestrator read
            "metric": images_s,
            "warmup_s": warm_s,
            "compile_s": compile_s,
            "load_s": load_s,
            "text_encode_s": txt_s,
            "hbm_peak_gb": hbm_peak,
            "hbm_available_gb": 24.0,
            "hbm_estimated": hbm_estimated,
            "warmup_iters": a.warmup,
            "measured_iters": a.iters,
            # equivalence: cross-config latent fingerprint (integer signature)
            "top1_tokens": eq_sig,
            # equivalence: Wan decode-parity gate result
            **parity,
            "image_out": img_written,
        })
    except Exception as e:  # noqa: BLE001
        import traceback
        _log(f"FAILED: {e}\n{traceback.format_exc()}")
        dump({"ok": False, "error": str(e)})

    sys.stdout.flush()
    os._exit(0)   # teardown can SIGSEGV; hard-exit after writing the result.


if __name__ == "__main__":
    main()
