#!/usr/bin/env python
"""MLA tensor parallelism for DeepSeek-V4 (runbook phase P5).

MLA has ``num_key_value_heads = 1`` (a single shared low-rank latent KV), so the
framework's GQA head-sharding (``qwen38_tp``) slices the KV head to zero width --
the exact "size of tensor a (N) must match tensor b (0)" crash seen on
DeepSeek-V2-Lite. The correct MLA TP:

  * shard the QUERY heads: ``q_b_proj`` rows + the per-head ``sinks``;
  * REPLICATE the shared latent KV (``kv_proj``/``kv_norm``), ``q_a_proj``/norm,
    and the grouped-O projection (it is small: o_lora_rank);
  * ALL-GATHER the per-rank attention output across ranks before the O projection,
    so O runs on the full head set and matches tp=1 exactly.

Validated on the shrunk config at tp=16/32/64: rank-0 logits cos 0.999962 vs tp=1.
(tp=2 and tp=8 are degenerate world sizes on this trn2.48xl at LNC=2; use >=16.)

Scope: this forward covers the dense ``sliding_attention`` path (P7 runs with
CSA off, so it applies). The experts are left REPLICATED -- fine when they fit
per rank (shrunk config). The real 256-expert model additionally needs expert
parallelism (EP; see moe_ep / moe_mesh) so each rank holds a subset of experts.
"""
import torch
import torch.nn as nn
import torch.distributed as dist


def _slice_rows(lin, r0, r1):
    W = lin.weight.data[r0:r1, :]
    n = nn.Linear(W.shape[1], W.shape[0], bias=lin.bias is not None, dtype=W.dtype)
    n.weight.data = W.contiguous()
    if lin.bias is not None:
        n.bias.data = lin.bias.data[r0:r1].contiguous()
    return n


def shard_mla(a, r, tp):
    """Shard one DeepseekV4Attention onto rank r of tp: query heads + sinks."""
    nh, hd = a.num_heads, a.head_dim
    if nh % tp:
        raise ValueError(f"num_attention_heads={nh} not divisible by tp={tp}")
    hpr = nh // tp
    a.q_b_proj = _slice_rows(a.q_b_proj, r * hpr * hd, (r + 1) * hpr * hd)
    a.sinks = nn.Parameter(a.sinks.data[r * hpr:(r + 1) * hpr].clone())
    a.num_heads = hpr
    a.num_key_value_groups = hpr
    # q_a_proj/q_a_norm/kv_proj/kv_norm/q_b_norm/o_a_proj/o_b_proj stay replicated.


def mla_tp_attn_forward(self, hidden_states, position_embeddings, position_ids,
                        attention_mask, past_key_values=None, **kwargs):
    """DeepseekV4Attention.forward for the dense sliding path + all-gather over heads."""
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as M
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings[self.rope_layer_type]
    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
    q = self.q_b_norm(q)
    q = M.apply_rotary_pos_emb(q, cos, sin)
    kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
    kv = M.apply_rotary_pos_emb(kv, cos, sin)
    attn_interface = M.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, M.eager_attention_forward)
    attn_output, attn_weights = attn_interface(
        self, q, kv, kv, attention_mask, dropout=0.0, scaling=self.scaling,
        sliding_window=self.sliding_window, s_aux=self.sinks, **kwargs)
    attn_output = M.apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
    if dist.is_initialized() and dist.get_world_size() > 1:
        g = [torch.empty_like(attn_output) for _ in range(dist.get_world_size())]
        dist.all_gather(g, attn_output.contiguous())
        attn_output = torch.cat(g, dim=2)                       # concat heads -> full set
    grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    return self.o_b_proj(grouped), attn_weights


def apply_mla_tp(model, rank, tp, modeling_module=None):
    """Monkeypatch the MLA forward and shard every attention onto (rank, tp)."""
    if modeling_module is None:
        import transformers.models.deepseek_v4.modeling_deepseek_v4 as modeling_module
    modeling_module.DeepseekV4Attention.forward = mla_tp_attn_forward
    if tp > 1:
        for _, mod in model.named_modules():
            if type(mod).__name__ == "DeepseekV4Attention":
                shard_mla(mod, rank, tp)
    return model
