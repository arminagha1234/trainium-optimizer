"""Shard-on-read: never let a full model exist in host DRAM.

The problem
-----------
``from_pretrained`` materialises the WHOLE model in the calling process, and tensor
parallelism runs one process per core, so the transient host-DRAM peak is
``ranks x model_size``. On a trn2.48xlarge (2147 GB) that is the wall the big models
hit long before HBM:

    Qwen3.5-122B-A10B  250 GB x 32 ranks =  8.0 TB   OOMKilled
    DeepSeek-V4-Flash  319 GB x 64 ranks = 20.4 TB   OOMKilled (137)

``load_stagger`` (#129) made those loadable by letting only N ranks be in the load
window at once, but the load then serialises into ``ceil(ranks/N)`` waves -- the 122B
ran 8 waves and one rank waited 204 s for a slot. This module removes the multiplier
instead of scheduling around it: each rank reads the checkpoint file by file and
materialises ONLY its own slice, so the peak is

    resident shard  +  one safetensors file in flight   (~= W/ranks + 5 GB)

How the slicing is decided, and why it is trustworthy
----------------------------------------------------
The obvious implementation -- reimplement every shard rule against parameter names --
is also the dangerous one: a wrong offset does not crash, it silently returns a model
that produces plausible garbage. So the rules here are DECLARATIVE (``slice_for``),
and ``test_shard_stream.py`` proves them equivalent to the existing, battle-tested
imperative sharding (``qwen38_tp`` + ``moe_ep``) by building a real tiny
Qwen3_5MoeForCausalLM, sharding it both ways, and comparing every parameter
bit-for-bit on every rank. If the two ever disagree, the test fails rather than a
benchmark quietly reporting nonsense.

Anything this module does not recognise is REPLICATED, which is the safe default: an
unsharded parameter costs memory but is always correct, whereas a wrongly sharded one
is wrong. ``unrecognised()`` reports what fell through so the omission is visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Slice", "slice_for", "expert_range", "kv_range", "unrecognised"]


@dataclass(frozen=True)
class Slice:
    """Take ``[lo:hi]`` along ``dim`` of the checkpoint tensor. None means replicate."""
    dim: int
    lo: int
    hi: int


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def expert_range(num_experts: int, rank: int, world: int) -> tuple[int, int]:
    """This rank's half-open expert range -- identical to moe_ep.expert_shard_plan.

    Imported rather than duplicated where possible; re-derived here only so this
    module stays importable without torch.
    """
    if world <= 1:
        return 0, num_experts
    base, extra = divmod(num_experts, world)
    lo = rank * base + min(rank, extra)
    return lo, lo + base + (1 if rank < extra else 0)


def kv_range(heads: int, kv_heads: int, rank: int, world: int) -> tuple[int, int]:
    """KV heads this rank needs, mirroring qwen38_tp.shard_attention.

    Replicates when ``world > kv_heads`` instead of slicing to zero width -- the bug
    #127 fixed. Reduces to ``kv_heads // world`` whenever world divides kv_heads.
    """
    qpr = max(1, heads // world)
    count = max(1, (qpr * kv_heads) // heads)
    start = (rank * qpr * kv_heads) // heads
    return start, start + count


def slice_for(name: str, cfg: dict, rank: int, world: int,
              shape: tuple[int, ...] | None = None) -> Slice | None:
    """How to slice checkpoint tensor ``name`` for ``rank``, or None to replicate.

    ``cfg`` is the plain text-config dict. ``shape`` is the tensor's GLOBAL shape,
    which safetensors exposes in its header without reading any data.

    The shape is what makes the attention rules layout-agnostic, and it is not
    optional in practice. Rows-per-head is derived from the checkpoint instead of
    assumed: Qwen3.5 interleaves q|gate so a q_proj head block is ``head_dim * 2``
    rows wide, while a Llama-style q_proj uses ``head_dim``. Assuming either one
    silently mis-slices the other -- the first version of this function hardcoded a
    head-index slice and the equivalence test caught it as [1, 256] against a real
    [32, 256].
    """
    if world <= 1:
        return None

    def _i(key, default=0):
        v = cfg.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) else default

    heads = _i("num_attention_heads")
    kv = _i("num_key_value_heads", heads)
    hd = _i("head_dim") or (_i("hidden_size") // heads if heads else 0)
    n_exp = _i("num_experts") or _i("n_routed_experts")

    # --- MoE experts: fused 3-D tensors, sliced on the leading expert dim -------
    if ".mlp.experts." in name and name.rsplit(".", 1)[-1] in (
            "gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"):
        if not n_exp:
            return None
        lo, hi = expert_range(n_exp, rank, world)
        return Slice(0, lo, hi)

    # The router must stay GLOBAL: routing is computed over all experts on every
    # rank, and only the expert WEIGHTS are local (see moe_ep.ExpertParallelExperts).
    if ".mlp.gate." in name:
        return None

    # --- attention: whole-head slices ------------------------------------------
    if ".self_attn." in name and heads and shape:
        qpr = max(1, heads // world)
        if ".q_proj." in name:
            rph = shape[0] // heads          # head_dim, or head_dim*2 when gated
            if rph * heads != shape[0]:
                return None                  # unexpected layout: replicate, stay correct
            return Slice(0, rank * qpr * rph, (rank * qpr + qpr) * rph)
        if ".k_proj." in name or ".v_proj." in name:
            if not kv:
                return None
            rph = shape[0] // kv
            if rph * kv != shape[0]:
                return None
            lo, hi = kv_range(heads, kv, rank, world)
            return Slice(0, lo * rph, hi * rph)
        if ".o_proj.weight" in name and len(shape) > 1:
            cph = shape[1] // heads          # o_proj is sliced along its INPUT dim
            if cph * heads != shape[1]:
                return None
            return Slice(1, rank * qpr * cph, (rank * qpr + qpr) * cph)
    return None


def unrecognised(shapes: dict, cfg: dict, rank: int, world: int) -> list[str]:
    """Parameter names that will be REPLICATED, for logging.

    Replication is always correct, so this is a memory report rather than an error --
    but a large tensor showing up here means a shard rule is missing. Takes
    ``{name: shape}`` because the attention rules need the shape to decide.
    """
    return [n for n, sh in shapes.items()
            if slice_for(n, cfg, rank, world, tuple(sh)) is None]


def stream_shard(files, cfg: dict, rank: int, world: int, *, log=None):
    """Yield ``(name, tensor)`` holding only THIS rank's slice of each weight.

    ``files`` are the checkpoint's ``.safetensors`` paths. Reads one file at a time
    and uses safetensors' slice API, which seeks to the requested region rather than
    materialising the tensor first -- that is what makes the peak
    ``resident shard + one region in flight`` instead of a full model per rank.

    Coverage, computed from the published configs: the rules above cover **96.9%** of
    Qwen3.5-122B-A10B's weight bytes and **94.8%** of Qwen3.5-35B-A3B's, because
    experts dominate those models (231.9 GB of 241.4 GB, and 64.4 of 68.5). The
    GatedDeltaNet projections are replicated and cost only 4.5 GB of the 122B.
    Dense hybrids like Qwen3.8-27B come out at ~20% covered, but at 16.5 GB total
    they never had a host-DRAM problem to solve.
    """
    from safetensors import safe_open

    n_shard = n_repl = 0
    for path in files:
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():
                sl = slice_for(name, cfg, rank, world, tuple(f.get_slice(name).get_shape()))
                if sl is None:
                    yield name, f.get_tensor(name)
                    n_repl += 1
                    continue
                sliced = f.get_slice(name)
                # Build an index tuple that is `:` everywhere except the shard dim,
                # so safetensors reads only the bytes this rank needs.
                idx = [slice(None)] * len(sliced.get_shape())
                idx[sl.dim] = slice(sl.lo, sl.hi)
                yield name, sliced[tuple(idx)]
                n_shard += 1
    if log:
        log(f"shard-on-read: {n_shard} tensor(s) sliced, {n_repl} replicated "
            f"(rank {rank}/{world})")
