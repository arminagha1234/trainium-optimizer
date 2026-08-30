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
              shape: tuple[int, ...] | None = None,
              *, experts_only: bool = False) -> Slice | None:
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
    # experts_only: shard ONLY the fused-expert tensors, replicate everything else.
    # This is the loader mode -- sharding an attention WEIGHT without installing the
    # matching sharded-attention MODULE (head-count reshape + o_proj all-reduce) would
    # be silently wrong, so shard-on-read replicates attention and lets expert
    # parallelism carry the memory win (experts are 96.9% of the bytes that OOM). Not
    # the default, because the search's TP path DOES install sharded attention modules
    # and wants the attention slices too.
    if experts_only and ".mlp.experts." not in name:
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


def stream_shard(files, cfg: dict, rank: int, world: int, *, experts_only: bool = False, log=None):
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
                sl = slice_for(name, cfg, rank, world,
                               tuple(f.get_slice(name).get_shape()),
                               experts_only=experts_only)
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


def load_ep_sharded(model_id, files, cfg, rank, world, *, dtype=None, log=None):
    """Assemble an expert-parallel model WITHOUT materialising the full model.

    This is the wiring that turns the proven slice_for / stream_shard plan into a live
    model. The steps, and why each is safe:

    1. Instantiate on the ``meta`` device from config. ``from_config`` builds the
       module tree with zero real memory. This is what makes the host-DRAM peak
       ``resident shard + one file in flight`` instead of ``world x model``.

    2. Replace each fused-expert module with a rank-local one. Experts are 96.9% of the
       bytes for the models that OOM (122B, 235B, MiniMax) and are the ONLY thing
       slice_for shards for them -- attention and the rest are replicated. So only the
       expert modules change shape; everything else keeps its full (meta) shape and
       receives a full replicated tensor that matches.

    3. Stream the checkpoint and fill parameters in place with
       ``load_state_dict(assign=True)``: experts get their [lo:hi] slice (built off
       disk), the rest get replicated full copies. ``assign=True`` swaps the meta
       placeholders for real tensors rather than copying into them, so no full-width
       buffer is ever allocated for a sharded parameter.

    Correctness is not asserted by inspection: test_shard_stream.py proves the assembled
    per-rank parameters are bit-for-bit identical to the imperative shard_moe_experts
    path, itself numerically validated in test_moe_ep.py. Any parameter left on meta
    after loading is a missing rule and is raised, not silently shipped.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        from .moe_ep import ExpertParallelExperts
    except ImportError:
        from moe_ep import ExpertParallelExperts

    hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)
    if dtype is not None:
        model = model.to(dtype)

    n_exp = 0
    for key in ("num_experts", "n_routed_experts"):
        if isinstance(cfg.get(key), int):
            n_exp = cfg[key]
            break

    streamed = dict(stream_shard(files, cfg, rank, world,
                                 experts_only=True, log=log))
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        lm = getattr(root, "language_model", None)
        layers = getattr(lm, "layers", []) if lm is not None else []
    swapped = 0
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None or not hasattr(experts, "gate_up_proj") or not n_exp:
            continue
        prefix = None
        for nm in streamed:
            if nm.endswith("mlp.experts.gate_up_proj") and f".layers.{li}." in nm:
                prefix = nm[: -len("gate_up_proj")]
                break
        if prefix is None:
            continue
        mlp.experts = ExpertParallelExperts.from_sliced(
            num_experts=n_exp, rank=rank, world=world, act_fn=experts.act_fn,
            gate_up_proj=streamed[prefix + "gate_up_proj"],
            down_proj=streamed[prefix + "down_proj"])
        swapped += 1

    expert_names = {n for n in streamed if ".mlp.experts." in n}
    load_sd = {n: t for n, t in streamed.items() if n not in expert_names}
    _missing, unexpected = model.load_state_dict(load_sd, strict=False, assign=True)

    still_meta = [n for n, pr in model.named_parameters() if pr.is_meta]
    if still_meta:
        raise RuntimeError(
            f"shard-on-read left {len(still_meta)} parameter(s) on meta, e.g. "
            f"{still_meta[:5]} -- a shard rule is missing for them")
    if log:
        log(f"shard-on-read assembled: {swapped} expert layer(s) rank-local, "
            f"{len(load_sd)} tensor(s) loaded, {len(unexpected)} unexpected")
    return model
