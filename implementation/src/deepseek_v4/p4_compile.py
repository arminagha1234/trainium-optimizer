#!/usr/bin/env python
"""P4 - torch.compile(backend="neuron", dynamic=False) on the shrunk config.

Applies the compile patches (iterative-argmax router + Stage A dense experts),
compiles the shrunk model, and validates it compiles at two batch sizes and
matches eager (cos >= 0.99). Device-only (needs torch_neuronx + a Trainium core).

  python -m deepseek_v4.p4_compile [--mixed] [--seq 128]
"""
import os
import sys
import time
import argparse
import traceback

import torch

from .p1_reference import build_shrunk_config
from .compile_patches import apply_compile_patches


def _cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--mixed", action="store_true", help="learned router (default: all-hash)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    n_hash = 1 if args.mixed else args.layers
    cfg = build_shrunk_config(num_layers=args.layers, n_experts=args.experts, n_hash=n_hash)
    apply_compile_patches(M)

    torch.manual_seed(args.seed)
    model = M.DeepseekV4ForCausalLM(cfg).to(torch.bfloat16).eval().to("neuron")
    seq = args.seq
    ids1 = (torch.arange(seq).unsqueeze(0)) % cfg.vocab_size
    ids2 = torch.stack([(torch.arange(seq)) % cfg.vocab_size, (torch.arange(seq) + 7) % cfg.vocab_size])

    with torch.inference_mode():
        e1 = model(ids1.to("neuron")).logits.float().cpu()
        e2 = model(ids2.to("neuron")).logits.float().cpu()
    comp = torch.compile(model, backend="neuron", dynamic=False)
    ok = True
    for tag, ids, e in [("b1", ids1, e1), ("b2", ids2, e2)]:
        try:
            t0 = time.time()
            with torch.inference_mode():
                c = comp(ids.to("neuron")).logits.float().cpu()
            cs = _cos(c, e)
            ok = ok and (cs >= 0.99)
            print(f"[P4] COMPILE {tag} OK in {time.time()-t0:.0f}s cos_vs_eager={cs:.6f}", flush=True)
        except Exception:
            ok = False
            print(f"[P4] COMPILE {tag} FAILED:", flush=True)
            traceback.print_exc()
    print(f"[P4] GATE {'PASS' if ok else 'FAIL'} ({'mixed' if args.mixed else 'all-hash'})")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
