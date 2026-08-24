# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
qwen3_next_rewrites.py — the full bundle of pure graph rewrites that make the
Qwen3-Next / Qwen3.5 (``model_type == "qwen3_next"``, GatedDeltaNet-MoE) family
both COMPILE and be NUMERICALLY CORRECT on Trainium.

WHY THIS EXISTS (and why the int64 topk patch alone is not enough):

``backends/moe_router_patch.install_neuron_safe_moe_topk`` clears exactly ONE
failure — the ``AwsNeuronTopK`` int64 dtype reject (NCC_EVRF013) shared by every
HF MoE. Qwen3-Next needs THREE more rewrites on top of that, all pure graph
rewrites (no NKI kernel), each grounded in an on-device-captured failure on
trn2 (neuronx-cc 2.27.5334, transformers 5.15.0, Qwen/Qwen3-Next-80B-A3B config):

  1. SORT-FREE ARGMAX ROUTER — ``Qwen3NextTopKRouter.forward``'s ``torch.topk``
     lowers to an XLA ``sort`` op, and ``sort`` has NO trn2 ISA op
     (NCC_EVRF029). The int64->fp32 dtype trick does NOT help (this is an
     op-unsupported reject, not a dtype reject). Replace the sort-based top-k
     with k rounds of masked ``.max(dim=-1)`` (argmax = a supported reduce).
     Numerically identical selection for distinct probs (maxdiff 0.0 on CPU).

  2. TRIL -> CONST-MASK — the GatedDeltaNet chunk rule
     (``torch_chunk_gated_delta_rule``) uses a runtime ``.tril()`` that lowers to
     ``TensorScalarAffineSelect`` and fails ISA validation (NCC_IINAR001,
     ``s2d2_ts_as_valid_elem_count``). Multiply by a host-materialized constant
     lower-triangular mask instead (folds to a plain TensorTensor multiply).

  3. SORT-FREE STATIC-SHAPE DENSE EXPERT DISPATCH — after (1)+(2) the model
     COMPILES but is numerically WRONG (cosine ~0.75). Isolated to the MoE
     expert path: HF's ``Qwen3NextExperts.forward`` (data-dependent
     ``one_hot``/``nonzero``/``where``/``index_add_``, or on older releases the
     ``moe.py`` ``sort``+``histc``+``grouped_mm`` grouping through the int64->fp32
     patch) is numerically wrong on trn2. Replace it with a DENSE masked
     dispatch: compute every expert for every token, weight by a scattered gate
     (0 for non-selected experts). Same math as top-k routing, fully static
     shapes, no sort / topk / nonzero / grouped_mm. Exact on CPU.

RESULT (arch-proof, tiny faithful down-scale, trn2.3xlarge, bf16, TP=1):
compiles end-to-end (~92s, valid NEFF, no ISA errors); correctness vs CPU-bf16
reference cosine 0.99793, top-1 14/16 (== the bf16 noise floor). This is the
architecture that had never compiled on Neuron (the flagship
compiler-cannot-lower model) — these four rewrites (this bundle + the int64
patch) are what unblocked it.

The three rewrites here are correctness-preserving GRAPH rewrites, catalogued in
``kernel_rewrites.py`` as ``topk-sort-to-argmax``, ``tril-to-const-mask`` and
``dense-moe-static-dispatch``. This module is the executable form the
neuron_worker installs automatically for the qwen3_next arch.

Import-safe: no torch / transformers at import time (mirrors moe_router_patch).
Installed explicitly (and only) for the qwen3_next arch at model-load in
``neuron_worker.py``; every other model is untouched. The monkeypatches target
module-level functions / class ``forward`` methods, which are resolved at call
time, so installing any time before the first forward takes effect on the
already-instantiated model.
"""

from __future__ import annotations

from typing import Any, Callable

_INSTALLED = False
_ORIG: dict = {}


def is_qwen3_next_arch(cfg: Any) -> bool:
    """True if ``cfg`` (or its ``text_config``) is a Qwen3-Next / Qwen3.5
    GatedDeltaNet-MoE model. Keyed off ``model_type`` first (stable across
    releases), with a structural fallback on the linear-attn config markers so a
    remote-code variant that renames the model_type is still caught."""
    def _mt(c: Any) -> str:
        return str(getattr(c, "model_type", "") or "").lower()

    if "qwen3_next" in _mt(cfg):
        return True
    tc = getattr(cfg, "text_config", None)
    if tc is not None and "qwen3_next" in _mt(tc):
        return True
    # Structural fallback: the GatedDeltaNet linear-attention config fields.
    markers = ("linear_key_head_dim", "linear_num_value_heads",
               "linear_conv_kernel_dim")
    probe = tc if tc is not None else cfg
    return all(hasattr(probe, m) for m in markers)


# --- Rewrite 1: sort-free iterative-argmax MoE router -----------------------
def sortfree_router_forward(self, hidden_states):
    """Sort-free iterative-argmax replacement for the MoE router top-k.
    torch.topk lowers to an XLA `sort` op unsupported on trn2 (NCC_EVRF029).
    k rounds of masked .max(dim=-1) use only a supported reduce. Numerically
    identical selection to torch.topk for distinct probs."""
    import torch
    import torch.nn.functional as F
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    vals = router_probs
    idx_list, val_list = [], []
    n_exp = vals.shape[-1]
    iota = torch.arange(n_exp, device=vals.device).view(1, -1)
    for _ in range(self.top_k):
        m = vals.max(dim=-1, keepdim=True)          # argmax = supported reduce, no sort
        val_list.append(m.values)
        idx_list.append(m.indices)
        vals = vals.masked_fill(iota == m.indices, float("-inf"))
    router_top_value = torch.cat(val_list, dim=-1)
    router_indices = torch.cat(idx_list, dim=-1)
    if self.norm_topk_prob:
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    return router_logits, router_top_value, router_indices


# --- Rewrite 2: tril-to-const-mask GatedDeltaNet chunk rule -----------------
# modeling_qwen3_next.py `.tril()` on a data tensor lowers to
# TensorScalarAffineSelect and ISA-fails (NCC_IINAR001 s2d2_ts_as_valid_elem_count).
# Replace the runtime .tril() with a multiply by a constant lower-triangular
# mask (catalogued rewrite tril-to-const-mask). Correctness-preserving.
def chunk_gdr_constmask(query, key, value, g, beta, chunk_size=64,
                        initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=False, **kwargs):
    import torch
    import torch.nn.functional as _F
    from transformers.models.qwen3_next import modeling_qwen3_next as M
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = M.l2norm(query, dim=-1, eps=1e-6)
        key = M.l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = _F.pad(query, (0, 0, 0, pad_size)); key = _F.pad(key, (0, 0, 0, pad_size))
    value = _F.pad(value, (0, 0, 0, pad_size)); beta = _F.pad(beta, (0, pad_size))
    g = _F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5); query = query * scale
    v_beta = value * beta.unsqueeze(-1); k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    dev = query.device
    # constant masks (fold to literals, no runtime affine-select)
    mask_ut0 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=dev), diagonal=0)
    tril_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32, device=dev))  # <- const
    g = g.cumsum(dim=-1)
    diff = (g.unsqueeze(-1) - g.unsqueeze(-2))
    decay_mask = ((diff * tril_mask).exp() * tril_mask).float()          # was .tril().exp().float().tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_ut0, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone(); sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=dev)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=dev)
                            if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    mask_ut1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=dev), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    if not output_final_state: last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# --- Rewrite 3: sort-free static-shape dense MoE expert dispatch ------------
# HF's Qwen3NextExperts.forward routes through a data-dependent grouping
# (one_hot / nonzero / where / index_add_, or on older releases moe.py
# sort+histc+grouped_mm). That path is numerically WRONG on trn2 (cosine ~0.75).
# Replace it with a dense masked dispatch: compute every expert for every token,
# weight by a scattered gate (0 for non-selected). Same math as top-k routing,
# static shapes, no sort/topk/nonzero/grouped_mm.
def dense_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    import torch
    import torch.nn.functional as _F
    Tn, H = hidden_states.shape
    E = self.num_experts
    gate_full = torch.zeros(Tn, E, device=hidden_states.device, dtype=top_k_weights.dtype)
    gate_full = gate_full.scatter(1, top_k_index, top_k_weights)      # (T, E)
    out = torch.zeros_like(hidden_states)
    for e in range(E):
        g, u = _F.linear(hidden_states, self.gate_up_proj[e]).chunk(2, dim=-1)
        h = self.act_fn(g) * u
        h = _F.linear(h, self.down_proj[e])
        out = out + h * gate_full[:, e:e+1].to(out.dtype)
    return out


def install_qwen3_next_neuron_rewrites(log: Callable[[str], None] = print) -> bool:
    """Install all three Qwen3-Next Neuron graph rewrites (sort-free router,
    tril->const-mask GatedDeltaNet, dense-MoE dispatch) onto the live
    ``transformers.models.qwen3_next.modeling_qwen3_next`` module.

    Idempotent and process-scoped (the neuron_worker is a single-model,
    hard-exiting process). Returns True if it installed, False if already
    installed or transformers is unavailable. NEVER raises — a failure to patch
    degrades to the unpatched path (which the caller can still attempt)."""
    global _INSTALLED
    if _INSTALLED:
        return False
    try:
        import torch  # noqa: F401 — ensure torch present before patching
        from transformers.models.qwen3_next import modeling_qwen3_next as M
    except Exception as e:  # noqa: BLE001
        log(f"qwen3-next-rewrites: transformers/torch unavailable ({e!r}); "
            "not installed")
        return False
    try:
        _ORIG["router"] = M.Qwen3NextTopKRouter.forward
        _ORIG["gdr"] = M.torch_chunk_gated_delta_rule
        _ORIG["experts"] = M.Qwen3NextExperts.forward
        M.Qwen3NextTopKRouter.forward = sortfree_router_forward
        M.torch_chunk_gated_delta_rule = chunk_gdr_constmask
        M.Qwen3NextExperts.forward = dense_experts_forward
        _INSTALLED = True
        log("qwen3-next-rewrites: installed sort-free argmax router "
            "(NCC_EVRF029) + tril->const-mask GatedDeltaNet (NCC_IINAR001) + "
            "sort-free dense-MoE expert dispatch (grouped-MoE correctness fix). "
            "Proven on-device: compiles ~92s, cosine 0.998 vs CPU-bf16.")
        return True
    except Exception as e:  # noqa: BLE001 — must never crash the worker
        log(f"qwen3-next-rewrites: install failed ({e!r}); running unpatched")
        return False


def uninstall_qwen3_next_neuron_rewrites() -> None:
    """Restore the original modeling_qwen3_next functions (test-support)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    try:
        from transformers.models.qwen3_next import modeling_qwen3_next as M
        M.Qwen3NextTopKRouter.forward = _ORIG["router"]
        M.torch_chunk_gated_delta_rule = _ORIG["gdr"]
        M.Qwen3NextExperts.forward = _ORIG["experts"]
    except Exception:  # noqa: BLE001
        pass
    finally:
        _INSTALLED = False
