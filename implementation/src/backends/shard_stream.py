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

__all__ = ["Slice", "slice_for", "expert_range", "kv_range", "unrecognised", "remap_vl_text_keys"]


@dataclass(frozen=True)
class Slice:
    """Take ``[lo:hi]`` along ``dim`` of the checkpoint tensor. None means replicate."""
    dim: int
    lo: int
    hi: int


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def remap_vl_text_keys(streamed: dict) -> dict:
    """Remap a VL-wrapped checkpoint's tensors onto the text CausalLM key space.
    Vision-language MoE checkpoints (e.g. Qwen3.5-MoE, which ships as a
    ``*ForConditionalGeneration``) namespace the text tower under
    ``model.language_model.`` and carry a ``model.visual.`` vision tower. The
    meta CausalLM that shard-on-read builds expects ``model.<...>`` and has no
    vision tower, so ``model.language_model.<x>`` -> ``model.<x>`` and vision keys
    are dropped. No-op (returns the input) for an already text-native checkpoint,
    so text models are untouched. Pure/torch-free, hence unit-testable; a wrong
    result is still caught downstream by the no-meta-tensor guard in
    ``load_ep_sharded`` (a mismapped key leaves a param on meta and RAISES)."""
    if not any(".language_model." in n for n in streamed):
        return streamed
    out = {}
    for n, t in streamed.items():
        if ".visual." in n or "vision_model" in n or n.startswith("visual."):
            continue
        out[n.replace("model.language_model.", "model.", 1)] = t
    return out


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


def load_ep_sharded(model_id, files, cfg, rank, world, *, dtype=None,
                    attn_implementation=None, log=None):
    """Assemble an expert-parallel model WITHOUT ever materialising it whole.

    ``from_pretrained`` builds the full model in every rank before sharding, so the
    host-DRAM peak is ``world x model_size`` -- the wall Qwen3-235B (470 GB) and the
    MiniMax models hit at load. This builds the model on ``meta`` (zero memory), gives
    each rank only its expert slice, and fills the rest replicated.

    The sequence, and the trap each step avoids:

    1. **Meta-instantiate from config.** CPU init would allocate the full model per rank
       at construction -- OOM before a single weight is read -- so meta is mandatory,
       not an optimisation.

    2. **(world>1) swap each fused-expert module for a rank-local one** sized to this
       rank's ``[lo:hi]`` experts, holding weights read pre-sliced off disk. At world=1
       nothing is sharded, so the experts load full into the original module and this
       path is a pure meta-load -- which is what the forward-equivalence test checks.

    3. **``load_state_dict(assign=True)``** swaps meta placeholders for the streamed
       tensors rather than copying into them, so no full-width buffer is allocated.

    4. **``tie_weights()``.** ``assign=True`` replaces the embedding parameter object,
       which breaks an lm_head that was tied to it; re-tying restores it (no-op when
       the model does not tie).

    5. **Re-materialise computed buffers left on meta** -- ``inv_freq`` and friends are
       not in the checkpoint, so ``load_state_dict`` never fills them and they would
       reach the forward as meta tensors and crash. They are recomputed by
       re-instantiating the owning module on CPU, which reuses the module's OWN init
       (version-agnostic -- the private rope API's key moved between versions).

    6. **Refuse to return a model with ANY meta tensor left** -- parameter OR buffer.
       A wrong loader does not crash, it benchmarks garbage, so an unfilled tensor is
       raised here, loudly, rather than shipped.

    Proven in test_shard_stream.py: per-rank parameters bit-for-bit vs the imperative
    ``shard_moe_experts`` path (world 2/4/8), and world=1 forward logits equal to
    ``from_pretrained``.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        from .moe_ep import ExpertParallelExperts
    except ImportError:
        from moe_ep import ExpertParallelExperts

    hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if attn_implementation is not None:
        # match the full-load path's attention kernel choice
        try:
            hf_cfg._attn_implementation = attn_implementation
        except Exception:  # noqa: BLE001
            pass
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
    # VL-wrapped checkpoints (Qwen3.5-MoE ships as *ForConditionalGeneration)
    # namespace the text tower under `model.language_model.`; remap it onto the
    # text CausalLM key space and drop the vision tower. A wrong remap leaves
    # params on meta -> the guard below RAISES, never silently ships mismapped weights.
    _pre = len(streamed)
    streamed = remap_vl_text_keys(streamed)
    if log and len(streamed) != _pre:
        log(f"shard-on-read: VL-wrapped -> remapped {len(streamed)}/{_pre} text keys, dropped vision")

    swapped = 0
    if world > 1 and n_exp:
        root = getattr(model, "model", model)
        layers = getattr(root, "layers", None)
        if layers is None:
            lm = getattr(root, "language_model", None)
            layers = getattr(lm, "layers", []) if lm is not None else []
        for li, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is None or not hasattr(experts, "gate_up_proj"):
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
        # experts now live inside the swapped modules, not the load set
        expert_names = {n for n in streamed if ".mlp.experts." in n}
        load_sd = {n: t for n, t in streamed.items() if n not in expert_names}
    else:
        load_sd = streamed          # world==1: everything loads full, incl. experts

    _missing, unexpected = model.load_state_dict(load_sd, strict=False, assign=True)
    model.tie_weights()

    # 5. recompute non-checkpoint buffers still on meta (inv_freq, ...).
    for _mn, mod in model.named_modules():
        meta_bufs = [bn for bn, b in mod.named_buffers(recurse=False) if b.is_meta]
        if not meta_bufs:
            continue
        rebuilt = None
        if hasattr(mod, "config"):
            try:
                with torch.device("cpu"):
                    rebuilt = type(mod)(mod.config)
            except Exception:  # noqa: BLE001
                rebuilt = None
        for bn in meta_bufs:
            src = getattr(rebuilt, bn, None) if rebuilt is not None else None
            if src is None or src.is_meta:
                raise RuntimeError(
                    f"shard-on-read could not recompute buffer {_mn}.{bn}; "
                    f"it would reach the forward as a meta tensor")
            # assign into the buffer slot directly, preserving persistent-ness
            mod._buffers[bn] = src.detach().clone()

    still_meta = ([n for n, pr in model.named_parameters() if pr.is_meta]
                  + [n for n, b in model.named_buffers() if b.is_meta])
    if still_meta:
        raise RuntimeError(
            f"shard-on-read left {len(still_meta)} tensor(s) on meta, e.g. "
            f"{still_meta[:5]} -- a shard rule or buffer recompute is missing")
    if log:
        log(f"shard-on-read assembled: {swapped} expert layer(s) rank-local, "
            f"{len(load_sd)} tensor(s) loaded, {len(unexpected)} unexpected")
    return model
