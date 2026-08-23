"""
Neuron benchmark worker — the multi-rank process that actually runs a model on
Trainium under a config, and writes one measurement to JSON.

Launched by native_pytorch.NativePyTorchBackend via:
    torchrun --nproc_per_node=<tp> neuron_worker.py --config <json> --out <json>

Why a separate process: the optimizer core is single-process Python, but real
tensor parallelism on Neuron needs `torchrun` with one rank per NeuronCore and
init_process_group(backend="neuron"). So each measurement is an independent
torchrun invocation; rank 0 writes the result file the backend parses.

Metric: PREFILL throughput (tokens/sec) = batch * input_len / median_forward_s.
This matches the reference implementation's avg_prefill_tok_per_s — a stable,
compile-friendly throughput number that does not depend on generate() behaving
under DTensor.

Beta 3 patterns (confirmed on-device, torch_api_compatibility.md):
  - torch.device("neuron")
  - dist.init_process_group(backend="neuron")  -> ProcessGroupNeuron
  - torch.compile(model, backend="neuron", dynamic=False)
  - DTensor TP via parallelize_module (Colwise q/k/v/gate/up, Rowwise o/down)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import AutoConfig, AutoModelForCausalLM

# Generic kernel-injection hook. Kept in a torch-free sibling module so the
# inject/resolve logic is unit-testable on a CPU box (this worker itself is not
# importable without torch). Fallback import mirrors the moe_fused pattern below,
# covering both the package-relative (`backends.`) and src-on-path layouts.
try:
    from backends.kernel_inject import inject_kernel
except Exception:  # noqa: BLE001
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from backends.kernel_inject import inject_kernel

PEAK_TFLOPS_BF16 = 380.0  # per NeuronCore, Trn2 (trajectory-reporting.md)
HBM_GB_PER_LOGICAL_CORE = 48.0  # 96 GB/device / 4 phys cores * LNC2 (2 phys/logical)


def _r0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _log(msg: str) -> None:
    if _r0():
        print(f"[worker] {msg}", flush=True)


def _sync(t: torch.Tensor) -> float:
    """Force the async Neuron queue to finish by pulling a scalar to host.
    Tolerates DTensor outputs (from vocab-parallel lm_head)."""
    if hasattr(t, "to_local"):
        try:
            t = t.to_local()
        except Exception:  # noqa: BLE001
            pass
    return float(t.detach().float().flatten()[:1].cpu().item())


def _dense_tp_plan() -> dict:
    """Standard dense-transformer TP plan. Requires num_heads % tp == 0 and
    (for GQA) num_kv_heads % tp == 0. Works for Llama/Qwen/Gemma-family."""
    return {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }


def _rhas(module, dotted: str) -> bool:
    cur = module
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return False
        cur = getattr(cur, part)
    return True


def _filtered_plan(layer, plan: dict) -> dict:
    """Only keep plan entries whose module path exists in THIS layer. Lets a
    dense plan apply cleanly to hybrid stacks (e.g. Gated DeltaNet layers that
    lack self_attn.q_proj are simply left un-sharded)."""
    return {k: v for k, v in plan.items() if _rhas(layer, k)}


def _expand_gqa_to_mha(model, tp, log):
    """Adapter: repeat K/V projection weights so num_kv_heads == num_heads,
    letting a GQA model shard uniformly at any tp dividing num_heads (past its
    KV-head count). K/V weights are tiny vs Q+MLP, so the overhead is small.
    After expansion HF's repeat_kv becomes identity (n_rep=1).

    This is what unlocks qwen3.8-27b (24 q heads / 4 kv heads) at tp8: without
    it, tp is capped at 4 and the model OOMs a 24GB core."""
    layers = _get_decoder_layers(model)
    n = 0
    from collections import Counter
    dbg = Counter()
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            dbg["no_self_attn"] += 1
            continue
        k = getattr(attn, "k_proj", None)
        v = getattr(attn, "v_proj", None)
        q = getattr(attn, "q_proj", None)
        if k is None or v is None or q is None:
            dbg["no_qkv"] += 1
            continue
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            dbg["no_head_dim"] += 1
            continue
        nkv = k.out_features // head_dim
        nh = q.out_features // head_dim
        dbg[f"hd{head_dim}_nkv{nkv}_nh{nh}"] += 1
        # Expand ONLY layers that cannot shard cleanly at this tp. Layers whose
        # kv already divides tp are left alone (efficient). This is what makes
        # gemma4 work: its 10 Global layers (kv=4) expand, its 50 SWA layers
        # (kv=16, divides tp8) do not.
        if nkv >= nh or nh % tp != 0 or nkv % tp == 0:
            continue
        rep = nh // nkv
        if rep * nkv != nh:
            continue
        for name, lin in (("k_proj", k), ("v_proj", v)):
            W = lin.weight.data.view(nkv, head_dim, -1)
            W = W.repeat_interleave(rep, dim=0).reshape(nh * head_dim, -1).contiguous()
            new = torch.nn.Linear(lin.in_features, nh * head_dim,
                                  bias=lin.bias is not None, dtype=lin.weight.dtype)
            new.weight.data = W
            if lin.bias is not None:
                b = lin.bias.data.view(nkv, head_dim).repeat_interleave(rep, dim=0)
                new.bias.data = b.reshape(nh * head_dim).contiguous()
            setattr(attn, name, new)
        # Make HF treat it as MHA: n_rep = 1.
        for a in ("num_key_value_heads", "num_kv_heads"):
            if hasattr(attn, a):
                setattr(attn, a, nh)
        if hasattr(attn, "num_key_value_groups"):
            attn.num_key_value_groups = 1
        cfg = getattr(attn, "config", None)
        if cfg is not None and hasattr(cfg, "num_key_value_heads"):
            try:
                cfg.num_key_value_heads = nh
            except Exception:  # noqa: BLE001
                pass
        n += 1
    log(f"gqa->mha: expanded k/v to full MHA on {n} layers (rep enables tp={tp}) | seen={dict(dbg)}")
    return n > 0


def _get_decoder_layers(model):
    """Find the decoder layer list across flat / multimodal / nested layouts."""
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "model.text_model.layers"):
        cur = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise RuntimeError("could not locate decoder layers on model")


def _find_attr(root, names):
    """Return (parent_module, attr_name) for the first existing dotted path."""
    for dotted in names:
        cur = root; ok = True; parts = dotted.split(".")
        for p in parts[:-1]:
            if not hasattr(cur, p):
                ok = False; break
            cur = getattr(cur, p)
        if ok and hasattr(cur, parts[-1]):
            return cur, parts[-1]
    return None, None


def _shard_vocab(model, mesh, log):
    """Vocab-parallel the token embedding + lm_head to cut per-rank memory on
    big-vocab models (e.g. qwen3.8-27b: 248k vocab -> ~3.8GB/rank saved). Only
    called when per-rank weights are tight, so the common models are untouched.
    Skips tied embeddings (sharding row+col differently would break the tie)."""
    emb_parent, emb_name = _find_attr(model, ["model.embed_tokens",
        "model.language_model.embed_tokens", "language_model.model.embed_tokens"])
    lm_parent, lm_name = _find_attr(model, ["lm_head"])
    if emb_parent is None or lm_parent is None:
        log("vocab-shard: embed/lm_head not found; skipping"); return False
    emb_w = getattr(emb_parent, emb_name).weight
    lm_w = getattr(lm_parent, lm_name).weight
    if emb_w.data_ptr() == lm_w.data_ptr():
        log("vocab-shard: tied embeddings; skipping"); return False
    # NOTE: RowwiseParallel on the embedding triggers a reduce_scatter that the
    # Neuron compiler rejects ("Invalid NEFF"). lm_head (ColwiseParallel ->
    # all-gather) is fine and is the 2.5GB alloc that was OOMing, so shard only
    # lm_head. Frees ~1.9GB/rank, enough to clear the fragmentation OOM.
    parallelize_module(lm_parent, mesh, {lm_name: ColwiseParallel(
        output_layouts=Replicate())})
    log("vocab-shard: sharded lm_head only (embed reduce_scatter unsupported)")
    return True


def _has_deltanet(model):
    """True if any decoder layer is a Gated-DeltaNet (linear_attn) layer —
    i.e. this is qwen3.8/Qwen3.5-Next and needs the manual head-parallel TP
    path (qwen38_tp.shard_model), not the DTensor dense plan."""
    try:
        return any(hasattr(L, "linear_attn") for L in _get_decoder_layers(model))
    except Exception:  # noqa: BLE001
        return False


def _text_cfg(cfg):
    """Return the text sub-config for multimodal models, else the config.
    Also permit reading global values from heterogeneous per-layer configs
    (e.g. Gemma 4, where KV-head count varies across sliding/full layers)."""
    c = getattr(cfg, "text_config", None) or cfg
    try:
        setattr(c, "allow_global_per_layer_attribute_access", True)
    except Exception:  # noqa: BLE001
        pass
    return c


def _cfg_int(cfg, name, default=None):
    """getattr that tolerates heterogeneous-per-layer configs raising."""
    try:
        v = getattr(cfg, name, default)
        return v if isinstance(v, int) else default
    except Exception:  # noqa: BLE001
        return default


def _tp_divides(cfg, tp: int) -> tuple[bool, str]:
    cfg = _text_cfg(cfg)
    heads = _cfg_int(cfg, "num_attention_heads")
    kv = _cfg_int(cfg, "num_key_value_heads", heads)
    if heads is None:
        return True, ""  # unknown; let it try
    if heads % tp != 0:
        return False, f"num_attention_heads={heads} not divisible by tp={tp}"
    if kv is not None and kv % tp != 0:
        return False, f"num_key_value_heads={kv} not divisible by tp={tp}"
    return True, ""


def _param_count(cfg) -> float:
    """Rough total params from config, for MFU + HBM estimate."""
    cfg = _text_cfg(cfg)
    h = getattr(cfg, "hidden_size", 4096)
    L = getattr(cfg, "num_hidden_layers", 32)
    inter = getattr(cfg, "intermediate_size", 4 * h)
    vocab = getattr(cfg, "vocab_size", 32000)
    # attn (qkvo ~4*h*h) + mlp (gate+up+down ~3*h*inter) per layer + embeddings
    per_layer = 4 * h * h + 3 * h * inter
    return per_layer * L + 2 * vocab * h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--cc-flags", default="",
                    help="extra NEURON_CC_FLAGS for Stage 2-5 compiler rewrites")
    ap.add_argument("--moe-kernel", default="",
                    help="Stage-3 MoE borrow: 'fused_nki' swaps the HF MoE "
                         "layer forward with the vendored fused NKI megakernel")
    ap.add_argument("--kernel", default="",
                    help="Generic kernel injection: a JSON descriptor "
                         "{'target','entry','path'} pointing at an external "
                         "(proprietary) kernel file whose 'entry' forward-factory "
                         "is monkeypatched onto every module matching 'target'. "
                         "See backends.kernel_inject.inject_kernel.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # Stage 2-5 lever: neuronx-cc compiler flags (graph rewrites / kernel
    # selection). Applied to the torch.compile(backend='neuron') path via the
    # NEURON_CC_FLAGS env var, which neuronx-cc reads at compile time.
    if a.cc_flags:
        os.environ["NEURON_CC_FLAGS"] = (
            os.environ.get("NEURON_CC_FLAGS", "") + " " + a.cc_flags).strip()

    result = {"ok": False, "model": a.model, "tp": a.tp, "dtype": a.dtype,
              "attn": a.attn, "compile": a.compile, "shape_input_len": a.input_len,
              "batch": a.batch}

    def dump(extra: dict) -> None:
        result.update(extra)
        if _r0():
            with open(a.out, "w") as f:
                json.dump(result, f, indent=2)

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    dist.init_process_group(backend="neuron")
    world = dist.get_world_size()
    dev = torch.device("neuron")

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    tcfg = _text_cfg(cfg)
    heads = _cfg_int(tcfg, "num_attention_heads")
    kv = _cfg_int(tcfg, "num_key_value_heads", heads)
    # Valid as long as tp divides the query-head count. If tp does not divide
    # the KV-head count, the GQA->MHA adapter expands K/V so it still shards.
    if heads is not None and heads % a.tp != 0:
        why = f"num_attention_heads={heads} not divisible by tp={a.tp}"
        _log(f"INVALID_TP: {why}")
        dump({"ok": False, "error": f"invalid_tp: {why}"})
        sys.stdout.flush(); os._exit(0)
    need_expand = bool(heads and kv and a.tp > 1 and kv % a.tp != 0 and heads % a.tp == 0)

    params = _param_count(cfg)

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, attn_implementation=a.attn, trust_remote_code=True
    )
    # Multimodal text-only: drop the vision tower so it doesn't eat HBM.
    mm = getattr(model, "model", model)
    for vattr in ("vision_tower", "embed_vision", "vision_model", "multi_modal_projector"):
        if hasattr(mm, vattr) and getattr(mm, vattr) is not None:
            try:
                setattr(mm, vattr, torch.nn.Identity())
                _log(f"dropped {vattr} (text-only)")
            except Exception:  # noqa: BLE001
                pass

    if a.tp > 1 and _has_deltanet(model):
        # qwen3.8 hybrid (Gated DeltaNet + attention): manual head-parallel TP,
        # numerically validated exact (16/16 top-1 tokens vs CPU oracle at tp4).
        # No DTensor — the delta rule/attention are per-head so whole-head slices
        # keep the recurrence local; only rowwise outputs all-reduce.
        try:
            from backends.qwen38_tp import shard_model as _dn_shard
        except Exception:  # noqa: BLE001
            from qwen38_tp import shard_model as _dn_shard
        na, nd, nm = _dn_shard(model, dist.get_rank(), a.tp)
        _log(f"qwen3.8 head-parallel TP: attn={na} deltanet={nd} mlp={nm}")
    elif a.tp > 1:
        # Always attempt GQA->MHA expansion; it self-skips layers that already
        # shard cleanly, and expands only the ones that don't (handles gemma4's
        # per-layer heterogeneous kv-head counts and qwen3.8's GQA-4).
        _expand_gqa_to_mha(model, a.tp, _log)
        mesh = init_device_mesh("neuron", (world,))
        plan = _dense_tp_plan()
        layers = _get_decoder_layers(model)
        sharded = 0
        for layer in layers:
            fp = _filtered_plan(layer, plan)
            if fp:
                parallelize_module(layer, mesh, fp)
                sharded += 1
        _log(f"sharded {sharded}/{len(layers)} decoder layers (dense-plan match)")
        # Vocab-parallel embed+lm_head when per-rank weights are tight (>10GB),
        # so big-vocab models like qwen3.8-27b fit the 24GB core.
        dtb = 2 if a.dtype == "bf16" else 4
        per_rank_gb = (params * dtb / a.tp) / 1e9
        if per_rank_gb > 10.0:
            _shard_vocab(model, mesh, _log)
    # BASELINE UNBLOCK (MoE family): HF's generic MoE routing groups routed
    # token/expert pairs with `torch.sort(expert_ids)` (int64) — which lowers to
    # the AwsNeuronTopK custom-op and CRASHES ("does not support 32/64-bit
    # integer types", moe.py:393). Install a dtype-safe torch.{topk,sort,argsort}
    # so integer inputs route through float32 (order-preserving for expert ids)
    # and never reach AwsNeuronTopK as integers. Gated to MoE arches so dense
    # models are untouched; runs for the BASELINE too (independent of the
    # Stage-3 --moe-kernel borrow). See backends/moe_router_patch.py.
    try:
        try:
            from kernels.moe_fused import is_moe_arch
            from backends.moe_router_patch import install_neuron_safe_moe_topk
        except Exception:  # noqa: BLE001
            import os as _os, sys as _sys
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from kernels.moe_fused import is_moe_arch
            from backends.moe_router_patch import install_neuron_safe_moe_topk
        if is_moe_arch(tcfg):
            install_neuron_safe_moe_topk(_log)
    except Exception as e:  # noqa: BLE001 — never let the unblock crash a run
        _log(f"moe-router-patch: skipped ({e!r})")

    # Stage-3 BORROW: optionally swap the HF MoE block forward with the vendored
    # fused NKI megakernel. The adapter runs a full precondition gauntlet (arch,
    # exact A3B/TP4 dims, nkilib availability) and NEVER raises — on any unmet
    # precondition it leaves the eager model untouched and returns a reason, so
    # this measurement stays a correct, unchanged (non-faster) candidate that the
    # orchestrator's equivalence gate handles normally. See kernels/moe_fused.
    moe_swap = "not-requested"
    if a.moe_kernel:
        try:
            try:
                from kernels.moe_fused import swap_moe_forward
            except Exception:  # noqa: BLE001
                import os as _os, sys as _sys
                _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
                from kernels.moe_fused import swap_moe_forward
            swapped, reason = swap_moe_forward(model, a.tp, _log)
            moe_swap = f"{'swapped' if swapped else 'eager-fallback'}: {reason}"
        except Exception as e:  # noqa: BLE001 — never let the borrow crash a run
            moe_swap = f"eager-fallback: adapter error {e!r}"
            _log(f"moe-borrow adapter failed, running eager: {e!r}")

    # GENERIC KERNEL INJECTION: the generalization of the hardcoded MoE swap
    # above. Given a --kernel JSON descriptor (target/entry/path), import the
    # kernel from its EXTERNAL on-disk file and monkeypatch it onto every module
    # matching the target. Like the MoE borrow, inject_kernel NEVER raises — an
    # unloadable kernel or no-match leaves the model eager and reports why, so
    # this stays a correct (possibly unchanged) candidate for the equivalence
    # gate. See backends/kernel_inject.py (torch-free so it is CPU-mock-testable).
    kernel_inject = "not-requested"
    if a.kernel:
        try:
            injected, reason = inject_kernel(model, a.kernel, _log)
            kernel_inject = f"{'injected' if injected else 'eager-fallback'}: {reason}"
        except Exception as e:  # noqa: BLE001 — never let injection crash a run
            kernel_inject = f"eager-fallback: inject error {e!r}"
            _log(f"kernel-inject failed, running eager: {e!r}")

    model = model.to(dev)
    model.eval()
    load_s = time.time() - t_load
    _log(f"loaded+sharded tp={a.tp} dtype={a.dtype} attn={a.attn} in {load_s:.1f}s")

    if a.compile:
        model.forward = torch.compile(model.forward, backend="neuron", dynamic=False)

    # DETERMINISTIC prompt (arange, not random) so the equivalence gate can
    # compare top-1 tokens of this config against the baseline byte-for-byte.
    vocab_cap = min(1000, getattr(tcfg, "vocab_size", 32000))
    ids = (torch.arange(a.batch * a.input_len, device=dev) % vocab_cap
           ).reshape(a.batch, a.input_len)

    # Reset the Neuron memory counters so max_memory_allocated reflects THIS
    # run's peak (real HBM, #4 — replaces the earlier estimate).
    try:
        import torch_neuronx as _tnx
        _tnx.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        _tnx = None

    # First forward: triggers NEFF compile in compile-mode (and lazy graph
    # build in eager). Time it separately -> that is our compile_s proxy.
    t0 = time.time()
    with torch.no_grad():
        out = model(ids)
    _sync(out.logits)
    first_s = time.time() - t0
    compile_s = first_s if a.compile else 0.0
    _log(f"first forward {first_s:.1f}s (compile_s={compile_s:.1f})")

    # EQUIVALENCE SIGNATURE: top-1 predicted token id at the last K positions.
    # The orchestrator compares this against the Stage-0 baseline's signature;
    # a config that changes the output tokens is a correctness failure, not a
    # win (see CLAUDE.md rule 1). This makes the gate REAL, not a stub.
    EQ_K = 16
    lg = out.logits
    if hasattr(lg, "to_local"):
        try:
            lg = lg.to_local()
        except Exception:  # noqa: BLE001
            pass
    eq_tokens = lg[0, -EQ_K:, :].argmax(-1).detach().cpu().tolist()

    # Remaining warmup.
    with torch.no_grad():
        for _ in range(max(0, a.warmup - 1)):
            o = model(ids); _sync(o.logits)

    # Timed iterations, each synced to device completion.
    times = []
    with torch.no_grad():
        for _ in range(a.iters):
            t = time.time()
            o = model(ids)
            _sync(o.logits)
            times.append(time.time() - t)

    times.sort()
    p50 = statistics.median(times)
    p99 = times[min(len(times) - 1, int(0.99 * len(times)))]
    tokens = a.batch * a.input_len
    tok_s = tokens / p50 if p50 > 0 else 0.0

    # MFU: 2 * params * tok/s / (peak_per_core * tp_cores)
    mfu = 100.0 * (2 * params * tok_s) / (PEAK_TFLOPS_BF16 * 1e12 * a.tp)

    # HBM: REAL peak from the Neuron runtime (#4). Falls back to an estimate
    # only if the API is unavailable.
    dt_bytes = 2 if a.dtype == "bf16" else 4
    hbm_estimated = True
    hbm_peak = 0.0
    try:
        if _tnx is not None:
            hbm_peak = float(_tnx.max_memory_allocated()) / 1e9
            if hbm_peak > 0:
                hbm_estimated = False
    except Exception:  # noqa: BLE001
        hbm_peak = 0.0
    if hbm_peak <= 0:
        weight_gb = (params * dt_bytes / a.tp) / 1e9
        act_gb = (a.batch * a.input_len * getattr(tcfg, "hidden_size", 4096)
                  * dt_bytes * 4) / 1e9
        hbm_peak = weight_gb + act_gb
    hbm_avail = 24.0  # measured per-rank HBM on this trn2 (one physical core)

    _log(f"tok/s={tok_s:.1f} p50={p50*1000:.1f}ms p99={p99*1000:.1f}ms "
         f"mfu={mfu:.3f}% hbm~{hbm_peak:.1f}/{hbm_avail:.0f}GB")

    dump({
        "ok": True,
        "tok_s": tok_s,
        "p50_ms": p50 * 1000,
        "p99_ms": p99 * 1000,
        "first_forward_s": first_s,
        "compile_s": compile_s,
        "load_s": load_s,
        "mfu_percent": mfu,
        "hbm_peak_gb": hbm_peak,
        "hbm_available_gb": hbm_avail,
        "hbm_estimated": hbm_estimated,
        "params_est": params,
        "world_size": world,
        "top1_tokens": eq_tokens,   # equivalence signature (last-K argmax)
        "moe_kernel_swap": moe_swap,  # Stage-3 borrow status (audit trail)
        "kernel_inject": kernel_inject,  # generic-injection status (audit trail)
    })

    sys.stdout.flush()
    os._exit(0)  # Beta 3: teardown can SIGSEGV; hard-exit after writing result.


if __name__ == "__main__":
    main()
