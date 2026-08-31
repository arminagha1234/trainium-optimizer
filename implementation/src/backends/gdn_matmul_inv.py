"""backends/gdn_matmul_inv.py -- matmul-only GatedDeltaNet chunk inverse.

Fixes the tp>1 compile wall (NCC_IBCG901 "Too many strides") in the HuggingFace
`torch_chunk_gated_delta_rule` prefill path. The stock function inverts the
per-chunk (I - A) with a sequential forward-substitution loop over
variable-length strided slices (``attn[..., i, :i]``). At tp>1 the head-shard
adds another stride level and neuronx-cc cannot codegen that select
(NCC_IBCG901). This module replaces ONLY that loop with a matmul-only inverse
(banded Neumann series + residual refinement) -- fixed shapes, dense matmuls, no
data-dependent slicing -- which is numerically identical to the stock function
(verified cos 1.0 on out + final state vs HF on CPU) and compiles cleanly when
sharded.

Enable in the worker with ``TRN_OPT_GDN_MATMUL_INV=1``. ``install_gdn_matmul_inverse``
monkeypatches the module-global ``torch_chunk_gated_delta_rule`` in whichever HF
qwen3_5 / qwen3_5_moe / qwen3_next modeling module is present. The GDN layer's
forward resolves that global by name at call time (the real forward, inside the
``@force_accelerate_hooks`` wrapper closure, has ``__globals__`` == the modeling
module), so the swap takes effect without editing model code. The decode path
(``torch_recurrent_gated_delta_rule``, seq_len==1) is left untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def matmul_only_inverse(A: torch.Tensor, N: int = 3, Sres: int = 8) -> torch.Tensor:
    """Return (I - A)^-1 for a batched strictly-lower-triangular ``A`` of shape
    ``[..., C, C]`` using ONLY dense matmuls + a static band mask (no
    variable-stride select, so it compiles under tp-sharding).

    T0 = banded partial Neumann sum_{k=0..N} A^k; residual refine
    E = I - (I - A) @ T0; R = sum_{k=0..Sres} E^k; return R @ T0.
    """
    C = A.shape[-1]
    ident = torch.eye(C, dtype=A.dtype, device=A.device)
    P = ident.expand_as(A).clone()
    T0 = ident.expand_as(A).clone()
    for _ in range(N):
        P = P @ A
        T0 = T0 + P
    idx = torch.arange(C, device=A.device)
    band = ((idx[:, None] - idx[None, :]).abs() <= N).to(A.dtype)
    T0 = T0 * band
    E = ident - (ident - A) @ T0
    Pe = ident.expand_as(A).clone()
    R = ident.expand_as(A).clone()
    for _ in range(Sres):
        Pe = Pe @ E
        R = R + Pe
    return R @ T0


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_matmul_inv(query, key, value, g, beta, chunk_size=64, initial_state=None,
                     output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    """Drop-in for HF ``torch_chunk_gated_delta_rule``. Identical math, except the
    per-chunk (I - A) inverse uses :func:`matmul_only_inverse` instead of the
    sequential forward-substitution loop. Verified cos 1.0 vs stock on both the
    output and the final recurrent state."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # === matmul-only inverse replaces the strided forward-substitution loop ===
    # stock: for i in range(1, chunk_size): attn[..., i, :i] = ... ; attn += eye
    attn = matmul_only_inverse(attn, N=3, Sres=8)
    # =========================================================================
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


_CANDIDATE_MODULES = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    "transformers.models.qwen3_5.modeling_qwen3_5",
    "transformers.models.qwen3_next.modeling_qwen3_next",
)
_ORIGINALS: dict = {}


def install_gdn_matmul_inverse(log=None):
    """Monkeypatch ``torch_chunk_gated_delta_rule`` -> :func:`chunk_matmul_inv` in
    every present qwen3_5 / qwen3_5_moe / qwen3_next modeling module. Idempotent.
    Returns the list of patched module names."""
    import importlib
    patched = []
    for modname in _CANDIDATE_MODULES:
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if not hasattr(m, "torch_chunk_gated_delta_rule"):
            continue
        if modname not in _ORIGINALS:
            _ORIGINALS[modname] = m.torch_chunk_gated_delta_rule
        m.torch_chunk_gated_delta_rule = chunk_matmul_inv
        patched.append(modname)
    msg = ("GDN matmul-inverse installed on: "
           + (", ".join(patched) if patched else "NONE (no qwen3_5/_moe/next modeling module present)"))
    (log or (lambda s: print(f"[gdn_matmul_inv] {s}", flush=True)))(msg)
    return patched


def uninstall_gdn_matmul_inverse():
    """Restore the original ``torch_chunk_gated_delta_rule`` (used by tests)."""
    import importlib
    for modname, orig in list(_ORIGINALS.items()):
        try:
            importlib.import_module(modname).torch_chunk_gated_delta_rule = orig
        except Exception:
            pass
    _ORIGINALS.clear()
