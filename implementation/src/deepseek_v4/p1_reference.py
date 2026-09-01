#!/usr/bin/env python
"""P1 - Shrunk DeepSeek-V4-Flash config + CPU FP32 reference oracle.

Runbook: neuron/deepseek-v4-flash-native-pytorch-48xl-runbook.md, phase P1.

Uses the transformers-native ``DeepseekV4ForCausalLM`` (the authoritative
implementation, transformers >= 5.15) as the FP32 reference. Shrinks depth and
expert count while keeping every real tensor-shape dimension (hidden 4096,
head_dim 512, 64 heads, q_lora/o_lora 1024, o_groups 8, hc_mult 4,
moe_intermediate 2048). CSA is OFF: all layer_types = ``sliding_attention``
(the compress_ratio-0 dense path), so the model is architecturally valid and
exercises MLA + grouped-O + HyperConnections + MoE router + experts without the
compressor/indexer.

The config is built from the vendored public ``v4_flash_config.json`` (the real
checkpoint's config, ~1.7 KB), so P1-P6 need **no 160 GB checkpoint** and run in
CI. Emits ``ref.pt``: per-component (input, output) boundaries + final logits for
the P2 device ladder. Byte-reproducible for a fixed seed (P1 gate).

All ``transformers`` imports are lazy so this module imports without transformers.
"""
import os
import sys
import json
import argparse

import torch

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
_CONFIG_JSON = os.path.join(os.path.dirname(__file__), "v4_flash_config.json")

# nn.Module classes whose (input, output) boundaries form the correctness ladder.
CAPTURE_CLASSES = {
    "DeepseekV4RMSNorm", "DeepseekV4UnweightedRMSNorm", "DeepseekV4RotaryEmbedding",
    "DeepseekV4Attention", "DeepseekV4GroupedLinear",
    "DeepseekV4HyperConnection", "DeepseekV4HyperHead",
    "DeepseekV4TopKRouter", "DeepseekV4HashRouter",
    "DeepseekV4Experts", "DeepseekV4MLP", "DeepseekV4SparseMoeBlock",
}


def build_shrunk_config(num_layers=4, n_experts=8, experts_per_tok=2, n_hash=1,
                        vocab=4096, max_pos=8192, config_json=None):
    """Shrunk V4-Flash config from the vendored public config.json (dense, CSA off).

    Keeps real hidden/head/lora/hc/moe dims; shrinks depth, experts, vocab. Sets
    ``compress_ratios`` all-zero so every layer folds to ``sliding_attention``
    (dense) and drops the FP8/FP4 quantization_config (random-weight reference).
    """
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    with open(config_json or _CONFIG_JSON) as f:
        raw = json.load(f)
    raw.pop("quantization_config", None)       # random-weight ref: storage-quant N/A
    raw.pop("layer_types", None)               # recomputed from compress_ratios
    raw.pop("mlp_layer_types", None)           # recomputed from num_hash_layers
    n_hash = min(n_hash, num_layers)
    raw.update(dict(
        num_hidden_layers=num_layers,
        compress_ratios=[0] * num_layers,      # legacy -> all sliding_attention (CSA off)
        num_hash_layers=n_hash,
        n_routed_experts=n_experts,
        num_experts_per_tok=experts_per_tok,
        n_shared_experts=1,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        num_nextn_predict_layers=0,
        use_cache=False,
    ))
    return DeepseekV4Config(**raw)


def _to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().to(torch.float32).cpu()
    if isinstance(x, (tuple, list)):
        return type(x)(_to_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


def build_and_run(cfg, seed, ids, capture=None, device="cpu", dtype=torch.float32):
    """Seed, instantiate, optionally hook component boundaries, run one forward."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    torch.manual_seed(seed)
    model = M.DeepseekV4ForCausalLM(cfg).to(dtype).eval()
    if device != "cpu":
        model = model.to(device)
        ids = ids.to(device)
    handles = []
    if capture is not None:
        def mk(name):
            def hook(mod, inp, out):
                capture[name] = {"cls": type(mod).__name__, "in": _to_cpu(inp), "out": _to_cpu(out)}
            return hook
        for name, mod in model.named_modules():
            if type(mod).__name__ in CAPTURE_CLASSES:
                handles.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        out = model(ids)
    for h in handles:
        h.remove()
    return model, out.logits.detach().to(torch.float32).cpu()


def _shape(o):
    if torch.is_tensor(o):
        return tuple(o.shape)
    if isinstance(o, (tuple, list)) and o and torch.is_tensor(o[0]):
        return f"tuple[{len(o)}] first={tuple(o[0].shape)}"
    return type(o).__name__


def main():
    ap = argparse.ArgumentParser(description="Generate the P1 CPU FP32 reference oracle.")
    ap.add_argument("--out", default="/tmp/v4ref")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = build_shrunk_config(num_layers=args.layers, n_experts=args.experts)
    ids = (torch.arange(args.seq).unsqueeze(0)) % cfg.vocab_size

    cap = {}
    model, logits1 = build_and_run(cfg, args.seed, ids, capture=cap)
    _, logits2 = build_and_run(cfg, args.seed, ids, capture=None)      # reproducibility

    reproducible = torch.equal(logits1, logits2)
    finite = bool(torch.isfinite(logits1).all())
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[P1] params={nparams/1e6:.1f}M  layer_types={cfg.layer_types}")
    print(f"[P1] mlp_layer_types={cfg.mlp_layer_types}")
    print(f"[P1] logits={tuple(logits1.shape)} finite={finite} reproducible(byte-eq)={reproducible}")
    print(f"[P1] captured {len(cap)} component boundaries")

    seen = set()
    for name, rec in cap.items():
        if rec["cls"] in seen:
            continue
        seen.add(rec["cls"])
        print(f"[P1]   {rec['cls']:30s} eg '{name}'  out~{_shape(rec['out'])}")

    payload = {"input_ids": ids, "logits": logits1, "components": cap,
               "config": cfg.to_dict(), "seed": args.seed, "seq": args.seq}
    torch.save(payload, os.path.join(args.out, "ref.pt"))
    print(f"[P1] saved {args.out}/ref.pt ({os.path.getsize(os.path.join(args.out,'ref.pt'))/1e6:.1f} MB)")

    ok = finite and reproducible
    print(f"[P1] GATE {'PASS' if ok else 'FAIL'}")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
