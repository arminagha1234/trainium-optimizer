"""gdn_bwd_oracle.py — differentiable torch chunk GatedDeltaNet forward, used as
the AUTOGRAD ORACLE for the NKI backward (gradcheck-grade). Mirrors transformers
torch_chunk_gated_delta_rule (single head). q,k,v,g,beta -> o; autograd gives
dq,dk,dv,dg,dbeta. Inputs are the PREPROCESSED tensors the kernel sees (q already
l2-normed+scaled, k l2-normed) so the backward matches the kernel's boundary.
"""
from __future__ import annotations
import torch

def chunk_gdr_torch(q, k, v, g, beta, chunk_size=64):
    # all [T, d] float64; g,beta [T]; returns o [T, d_v]. Differentiable.
    T, dk = k.shape; dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    if pad:
        q = torch.cat([q, q.new_zeros(pad, dk)]); k = torch.cat([k, k.new_zeros(pad, dk)])
        v = torch.cat([v, v.new_zeros(pad, dv)]); g = torch.cat([g, g.new_zeros(pad)])
        beta = torch.cat([beta, beta.new_zeros(pad)])
    Tp = T + pad; n = Tp // C
    v_beta = v * beta[:, None]; k_beta = k * beta[:, None]
    qc = q.reshape(n, C, dk); kc = k.reshape(n, C, dk); vc = v.reshape(n, C, dv)
    kb = k_beta.reshape(n, C, dk); vb = v_beta.reshape(n, C, dv)
    gc = g.reshape(n, C).cumsum(-1)
    tril_incl = torch.tril(torch.ones(C, C, dtype=q.dtype, device=q.device))
    diff = gc[:, :, None] - gc[:, None, :]
    decay = torch.where(tril_incl.bool(), diff, torch.zeros_like(diff)).exp() * tril_incl
    strict = torch.tril(torch.ones(C, C, dtype=q.dtype, device=q.device), -1)
    A = -(torch.einsum("ncd,nmd->ncm", kb, kc) * decay) * strict    # strictly-lower [n,C,C]
    # T = (I-A)^-1 via recursive doubling: (I+A)(I+A^2)(I+A^4)...(I+A^{C/2}).
    # Pure matmuls -> cleanly differentiable AND matches the NKI kernel's algorithm.
    import math
    eye = torch.eye(C, dtype=q.dtype, device=q.device).expand(n, C, C)
    Lp = A; attn = eye + A
    for _ in range(int(math.log2(C)) - 1):                         # 5 squarings for C=64
        Lp = torch.bmm(Lp, Lp)
        attn = torch.bmm(attn, eye + Lp)
    v_corr = torch.einsum("ncm,nmd->ncd", attn, vb)
    k_cd = torch.einsum("ncm,nmd->ncd", attn, kb * gc[..., None].exp())
    S = torch.zeros(dk, dv, dtype=q.dtype, device=q.device)
    outs = []
    for i in range(n):
        q_i, k_i = qc[i], kc[i]
        ai = (q_i @ k_i.transpose(-1, -2)) * decay[i]
        vp = k_cd[i] @ S
        vn = v_corr[i] - vp
        inter = (q_i * gc[i, :, None].exp()) @ S
        outs.append(inter + ai @ vn)
        gl = gc[i, -1]
        S = S * gl.exp() + (k_i * (gl - gc[i]).exp()[:, None]).transpose(-1, -2) @ vn
    o = torch.stack(outs, dim=0).reshape(Tp, dv)[:T]
    return o

def grads(q, k, v, g, beta, dO, chunk_size=64):
    q, k, v, g, beta = (x.clone().double().requires_grad_(True) for x in (q, k, v, g, beta))
    o = chunk_gdr_torch(q, k, v, g, beta, chunk_size)
    o.backward(dO.double())
    return {"dq": q.grad, "dk": k.grad, "dv": v.grad, "dg": g.grad, "dbeta": beta.grad, "o": o}

def _demo():
    import numpy as np
    sys_p = os.path.dirname(os.path.abspath(__file__))
    torch.manual_seed(0)
    T, dk, dv = 128, 128, 128
    # the oracle expects PREPROCESSED inputs (q l2-normed+scaled, k l2-normed) —
    # exactly the kernel boundary; without this, un-normalized k makes the chunk
    # matrix large and the doubling inverse overflows.
    def l2(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = l2(torch.randn(T, dk)) * (1.0 / dk ** 0.5); k = l2(torch.randn(T, dk))
    v = torch.randn(T, dv)
    g = -torch.rand(T) * 0.3; beta = torch.rand(T); dO = torch.randn(T, dv)
    gr = grads(q, k, v, g, beta, dO)
    # cross-check the forward matches the numpy recurrence oracle
    import importlib.util
    ref_path = os.path.join(os.path.dirname(sys_p), "gdn_reference.py")
    spec = importlib.util.spec_from_file_location("gdnref", ref_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    o_rec = m.recurrent_gated_delta_rule(q.numpy(), k.numpy(), v.numpy(), g.numpy(), beta.numpy(),
                                         use_qk_l2norm=False, scale=1.0)
    err = np.abs(gr["o"].detach().numpy() - o_rec).max()
    print(f"oracle forward vs recurrent-ref: max_abs_err={err:.2e}")
    for kk in ("dq","dk","dv","dg","dbeta"):
        print(f"  {kk}: shape {tuple(gr[kk].shape)} norm {gr[kk].norm().item():.4f}")

if __name__ == "__main__":
    import os
    _demo()
