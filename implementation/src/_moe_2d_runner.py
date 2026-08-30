"""Real-file driver for the 2-D mesh forward proof (imported by test_moe_2d_forward).

Why a separate file instead of calling mp.spawn from the test: torch's mp.spawn uses
the 'spawn' start method, and a spawned child reconstructs the parent's __main__ by
PATH. Under pytest the "main" is the pytest entry, so the child re-exec fails with
FileNotFoundError and every rank dies -- which looks exactly like "gloo unavailable"
and silently downgrades the correctness gate to a skip. A real file with an
`if __name__ == "__main__"` guard is re-imported by spawn as `__mp_main__` (guard does
NOT re-run, so no fork bomb), which is the pattern that actually works.

Modes (argv):
    probe                      exit 0 if gloo multiprocess works, 42 if not (env skip)
    mesh <tp> <ep> <out_path>  shard the tiny MoE across a tp*ep=4 mesh; rank 0 writes
                               its logits to <out_path>. exit 0 ok, 1 = mesh error
                               (traceback to stderr -- a REAL failure, never a skip).
"""
from __future__ import annotations

import os
import socket
import sys
import traceback

import torch
import torch.multiprocessing as mp

CFG = dict(
    hidden_size=256, num_hidden_layers=2,
    num_attention_heads=16, num_key_value_heads=2, head_dim=16,
    intermediate_size=128, moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=32, num_experts_per_tok=8, vocab_size=256,
    layer_types=["linear_attention", "full_attention"],
    linear_num_key_heads=16, linear_num_value_heads=32,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
)
IDS = [[3, 9, 27, 81, 5, 25, 1, 7]]
_SRC = os.path.dirname(os.path.abspath(__file__))


def _build():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )
    cfg = Qwen3_5MoeTextConfig(**CFG)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3_5MoeForCausalLM(cfg).eval()


def reference_logits():
    m = _build()
    with torch.inference_mode():
        return m(torch.tensor(IDS)).logits


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _probe_worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world)
    t = torch.ones(1) * (rank + 1)
    dist.all_reduce(t)
    dist.destroy_process_group()
    assert t.item() == world * (world + 1) / 2


def _mesh_worker(rank, world, tp, ep, port, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import torch.distributed as dist
    from backends.moe_mesh import build_2d_groups
    from backends.moe_ep import shard_moe_experts
    from backends.qwen38_tp import shard_model
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        plan, tp_pg, ep_pg = build_2d_groups(world, tp, ep, rank)
        model = _build()
        # experts across the EP column, attention across the TP row -- the two axes.
        shard_moe_experts(model, rank, world, ep_plan=(ep, plan.ep_rank, ep_pg))
        shard_model(model, plan.tp_rank, tp, group=tp_pg)
        with torch.inference_mode():
            out = model(torch.tensor(IDS)).logits
        if rank == 0:
            torch.save(out.detach().clone(), out_path)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _main(argv) -> int:
    mode = argv[1] if len(argv) > 1 else "probe"
    if mode == "probe":
        try:
            mp.spawn(_probe_worker, args=(2, _free_port()), nprocs=2, join=True)
            return 0
        except Exception:  # noqa: BLE001 - env cannot run gloo multiprocess
            sys.stderr.write("PROBE FAILED:\n" + traceback.format_exc())
            return 42
    if mode == "mesh":
        tp, ep, out_path = int(argv[2]), int(argv[3]), argv[4]
        world = tp * ep
        try:
            mp.spawn(_mesh_worker, args=(world, tp, ep, _free_port(), out_path),
                     nprocs=world, join=True)
            return 0
        except Exception:  # noqa: BLE001 - a REAL mesh failure, surface it
            sys.stderr.write("MESH FAILED:\n" + traceback.format_exc())
            return 1
    sys.stderr.write(f"unknown mode {mode!r}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
