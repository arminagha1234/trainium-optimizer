#!/usr/bin/env python
"""P2 - Single-core eager device equivalence vs the P1 CPU FP32 oracle.

Runbook phase P2. Rebuilds the identical shrunk model (same seed => same weights
as the P1 oracle) on torch.device("neuron") in BF16 (the only device compute
dtype: the MoE grouped_mm kernel requires BF16 and a token dim divisible by 128),
runs one eager forward, and compares device-vs-oracle:

  * per-component cumulative drift, bottom-up, to localize any first divergence;
  * final-logits cosine.

Two configurations, because learned top-k routing is data-dependent:
  * ALL-HASH control (deterministic token->expert routing): isolates the model
    MATH. Device BF16 must match the FP32 oracle at cos >= 0.99. THIS is the P2
    correctness gate -- it exercises MLA, HyperConnections (Sinkhorn), RoPE,
    RMSNorm, grouped experts, the grouped low-rank O projection, the shared
    expert, and the hyper-head with an identical expert selection on both sides.
  * MIXED (learned TopKRouter): expected to diverge on RANDOM weights, because
    top-k over a near-uniform (untrained) score distribution is ill-conditioned
    -- any BF16 perturbation of the router input flips a selection. This is NOT a
    bug (the router SCORES match, and FP32 routing does not recover it); trained
    weights have separated winners, so this is validated on real weights at P8.

Device-only (needs torch_neuronx + a Trainium core). Not a CI test.
"""
import os
import sys
import argparse

import torch

from .p1_reference import build_shrunk_config, build_and_run, CAPTURE_CLASSES  # noqa: F401


def _cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def _first_tensor(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, (tuple, list)):
        for v in o:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _run_device(cfg, seed, ids, capture=None):
    """Build the shrunk model on the neuron device in BF16 and run one forward."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    torch.manual_seed(seed)
    model = M.DeepseekV4ForCausalLM(cfg).to(torch.bfloat16).eval().to("neuron")
    handles = []
    if capture is not None:
        def mk(name):
            def hook(mod, inp, out):
                t = _first_tensor(out)
                capture[name] = {"cls": type(mod).__name__,
                                 "out": (t.detach().float().cpu() if t is not None else None)}
            return hook
        for name, mod in model.named_modules():
            if type(mod).__name__ in CAPTURE_CLASSES:
                handles.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        out = model(ids.to("neuron"))
    for h in handles:
        h.remove()
    return out.logits.detach().float().cpu()


def compare_mixed(ref_path, seed=0):
    """MIXED config vs the saved oracle: final cos + bottom-up component drift."""
    ref = torch.load(ref_path, weights_only=False)
    ids, logits_ref, ref_cap = ref["input_ids"], ref["logits"].float(), ref["components"]
    cfg = build_shrunk_config()                       # default: 1 hash + rest moe
    dev_cap = {}
    logits_dev = _run_device(cfg, seed, ids, capture=dev_cap)
    print(f"[P2] MIXED final logits cos={_cos(logits_dev, logits_ref):.6f}")
    for name in ref_cap:
        if name not in dev_cap or dev_cap[name]["out"] is None:
            continue
        ot = _first_tensor(ref_cap[name]["out"])
        dt = dev_cap[name]["out"]
        if ot is None or ot.shape != dt.shape:
            continue
        c = _cos(dt, ot)
        if name.count(".") <= 3 or c < 0.999:
            flag = "  <-- diverge" if c < 0.99 else ""
            print(f"[P2]   {c:.5f} cos  {dev_cap[name]['cls']:26s} {name}{flag}")


def control_all_hash(seed=0, seq=128, layers=4, experts=8):
    """ALL-HASH control: deterministic routing isolates the math. Gate: cos>=0.99."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    cfg = build_shrunk_config(num_layers=layers, n_experts=experts, n_hash=layers)
    ids = (torch.arange(seq).unsqueeze(0)) % cfg.vocab_size
    torch.manual_seed(seed)
    mc = M.DeepseekV4ForCausalLM(cfg).float().eval()
    with torch.no_grad():
        lref = mc(ids).logits.float().cpu()
    del mc
    ldev = _run_device(cfg, seed, ids)
    return _cos(ldev, lref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/v4ref/ref.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-mixed", action="store_true")
    args = ap.parse_args()

    if not args.skip_mixed and os.path.exists(args.ref):
        compare_mixed(args.ref, args.seed)

    c = control_all_hash(args.seed)
    ok = c >= 0.99
    print(f"[P2] ALL-HASH (deterministic routing) final cos={c:.6f}")
    print(f"[P2] GATE {'PASS' if ok else 'FAIL'} (device math correct: all-hash cos {c:.5f} vs >=0.99)")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
