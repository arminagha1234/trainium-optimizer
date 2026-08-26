"""gdn_reference.py — pure-numpy ground truth for the GatedDeltaNet (GDN) chunked
gated delta rule, ported 1:1 from transformers Qwen3_5 `torch_chunk_gated_delta_rule`,
and cross-checked against the naive per-token recurrent form.

This is pipeline step (2): a numpy implementation that validates the math BEFORE
any NKI is written. Every NKI kernel we author is checked against
`chunk_gated_delta_rule` here (fp32) on the same random inputs.

Shapes (single batch, single head — the kernel tiles over (batch, head)):
  q, k : [T, dk]      (dk = linear_key_head_dim, 128 on Qwen3.5)
  v    : [T, dv]      (dv = linear_value_head_dim)
  g    : [T]          per-token log-gate (a_t = exp(g_t) is the decay)
  beta : [T]          delta write strength
Returns o : [T, dv].
"""

from __future__ import annotations

import numpy as np


def _l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def recurrent_gated_delta_rule(q, k, v, g, beta, *, scale=None,
                               use_qk_l2norm=False):
    """The naive O(T) per-token reference — the simplest correct statement of the
    op, used to cross-check the chunked form. Matches
    `torch_recurrent_gated_delta_rule`."""
    q, k, v, g, beta = (np.asarray(x, np.float64) for x in (q, k, v, g, beta))
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    T, dk = k.shape
    dv = v.shape[-1]
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    q = q * scale
    S = np.zeros((dk, dv), np.float64)          # recurrent state [dk, dv]
    out = np.zeros((T, dv), np.float64)
    for t in range(T):
        a = np.exp(g[t])                        # scalar decay
        S = S * a
        kv_mem = (S * k[t][:, None]).sum(0)     # [dv]  = k_t^T S
        delta = (v[t] - kv_mem) * beta[t]       # [dv]
        S = S + k[t][:, None] * delta[None, :]  # rank-1 update
        out[t] = (S * q[t][:, None]).sum(0)     # [dv]  = q_t^T S
    return out


def chunk_gated_delta_rule(q, k, v, g, beta, *, chunk_size=64, scale=None,
                           use_qk_l2norm=False):
    """Chunk-parallel form — the 1:1 numpy port of transformers
    `torch_chunk_gated_delta_rule`. This is the structure the NKI kernel mirrors:
    a per-chunk triangular solve (the T matrix / WY representation) + a cross-chunk
    gated recurrence."""
    q, k, v, g, beta = (np.asarray(x, np.float64) for x in (q, k, v, g, beta))
    if use_qk_l2norm:
        q, k = _l2norm(q), _l2norm(k)
    T, dk = k.shape
    dv = v.shape[-1]
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    q = q * scale
    C = chunk_size
    pad = (C - T % C) % C
    if pad:
        q = np.pad(q, ((0, pad), (0, 0)))
        k = np.pad(k, ((0, pad), (0, 0)))
        v = np.pad(v, ((0, pad), (0, 0)))
        g = np.pad(g, (0, pad))
        beta = np.pad(beta, (0, pad))
    Tp = T + pad
    n = Tp // C

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]
    # reshape into chunks: [n, C, d]
    qc = q.reshape(n, C, dk)
    kc = k.reshape(n, C, dk)
    vc = v.reshape(n, C, dv)
    kb = k_beta.reshape(n, C, dk)
    vb = v_beta.reshape(n, C, dv)
    gc = g.reshape(n, C).cumsum(-1)             # within-chunk cumulative gate [n,C]

    # decay_mask[n,i,j] = exp(g_i - g_j) for i>=j else 0   (tril, incl diag)
    diff = gc[:, :, None] - gc[:, None, :]      # [n,C,C]
    tril_incl = np.tril(np.ones((C, C)))        # i>=j
    decay_mask = np.exp(np.where(tril_incl.astype(bool), diff, 0.0)) * tril_incl

    # T matrix: attn = -(k_beta @ k^T * decay_mask) with STRICT-lower kept
    strict_lower = np.tril(np.ones((C, C)), -1)  # i>j
    attn = -(np.einsum("ncd,nmd->ncm", kb, kc) * decay_mask) * strict_lower
    # forward substitution: invert (I - strict_lower(attn))
    for i in range(1, C):
        row = attn[:, i, :i].copy()                       # [n, i]
        sub = attn[:, :i, :i]                             # [n, i, i]
        attn[:, i, :i] = row + np.einsum("ni,nij->nj", row, sub)
    attn = attn + np.eye(C)[None]                         # add I -> the T matrix

    v_corr = np.einsum("ncm,nmd->ncd", attn, vb)          # attn @ v_beta  [n,C,dv]
    k_cumdecay = np.einsum("ncm,nmd->ncd", attn,
                           kb * np.exp(gc)[:, :, None])    # [n,C,dk]

    S = np.zeros((dk, dv), np.float64)                    # cross-chunk state
    strict_lower_excl = np.tril(np.ones((C, C)), -1)      # causal, excl diag
    out = np.zeros((n, C, dv), np.float64)
    for i in range(n):
        q_i, k_i = qc[i], kc[i]
        # decay_mask[i] is already tril-inclusive, so this is the causal intra-
        # chunk score (matches the transformers ref exactly).
        attn_intra = (q_i @ k_i.T) * decay_mask[i]
        v_prime = k_cumdecay[i] @ S                       # [C,dv]
        v_new = v_corr[i] - v_prime
        attn_inter = (q_i * np.exp(gc[i])[:, None]) @ S   # [C,dv]
        out[i] = attn_inter + attn_intra @ v_new
        g_last = gc[i, -1]
        S = (S * np.exp(g_last)
             + (k_i * np.exp(g_last - gc[i])[:, None]).T @ v_new)
    out = out.reshape(Tp, dv)[:T]
    return out


def _demo():
    rng = np.random.default_rng(0)
    T, dk, dv = 128, 128, 128
    q = rng.standard_normal((T, dk)); k = rng.standard_normal((T, dk))
    v = rng.standard_normal((T, dv))
    g = -np.abs(rng.standard_normal(T)) * 0.1      # small negative log-gates (decay<1)
    beta = rng.uniform(0, 1, T)
    o_rec = recurrent_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm=True)
    o_chk = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, use_qk_l2norm=True)
    err = np.abs(o_rec - o_chk).max()
    rel = err / (np.abs(o_rec).max() + 1e-9)
    print(f"recurrent vs chunk: max_abs_err={err:.3e} rel={rel:.3e} "
          f"(shape {o_chk.shape})")
    assert rel < 1e-6, "chunk form must match the recurrent reference"
    print("OK: chunk form matches the per-token recurrence")


if __name__ == "__main__":
    _demo()
