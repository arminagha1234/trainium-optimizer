"""gdn_chunk_bwd_explicit.py — hand-derived per-chunk VJP (the NKI backward blueprint).

chunk_step forward: (q,k,v)[C,d], g,beta[C], S[dk,dv] -> (o[C,dv], S_next[dk,dv]).
This computes the LOCAL VJP explicitly (no autograd): given (do, dS_next) return
(dq,dk,dv,dg,dbeta,dS_in). The intricate parts are the doubling-inverse backward
(VJP of T=(I+A)(I+A^2)... — a reverse loop of matmul-VJPs) and the gate
reverse-cumsum for dg. Validated here against torch.autograd on chunk_step; once
exact, these are the exact ops the NKI backward runs per chunk.
"""
from __future__ import annotations
import math, torch


def _doubling_fwd(A):
    """T=(I+A)(I+A^2)...(I+A^{C/2}); returns T and the saved (T_prev, Lp_prev) per
    level needed by the VJP."""
    C = A.shape[-1]
    eye = torch.eye(C, dtype=A.dtype, device=A.device)
    levels = int(math.log2(C)) - 1
    Lp = A; T = eye + A
    saved = []                      # (T_prev, L_prev) before each level's update
    for _ in range(levels):
        L_prev = Lp
        Lp = Lp @ Lp
        saved.append((T, L_prev, Lp))    # T=T_{j-1}, L_prev=L_{j-1}, Lp=L_j
        T = T @ (eye + Lp)
    return T, saved, eye


def _doubling_vjp(dT, A, saved, eye):
    """VJP of the doubling product wrt A, given dT (grad wrt final T)."""
    # dA from the T_0 = I+A term is added at the end. Walk levels in reverse.
    dL_accum = torch.zeros_like(A)
    for (T_prev, L_prev, L_j) in reversed(saved):
        # T_j = T_prev @ (I + L_j)
        dT_prev = dT @ (eye + L_j).transpose(-1, -2)
        dL_j = T_prev.transpose(-1, -2) @ dT + dL_accum      # (I+L_j) deriv + carry
        # L_j = L_prev @ L_prev  (squaring)
        dL_prev = dL_j @ L_prev.transpose(-1, -2) + L_prev.transpose(-1, -2) @ dL_j
        dT = dT_prev
        dL_accum = dL_prev
    # remaining: T_0 = I + A  -> dA += dT (the final dT is dT_0); plus dL_accum is dA from L_0=A
    return dT + dL_accum


def chunk_step_bwd(q, k, v, g, beta, S, do, dSn):
    C, dk = k.shape; dv = v.shape[-1]
    q, k, v, g, beta, S, do, dSn = (x.double() for x in (q, k, v, g, beta, S, do, dSn))
    # ---- recompute forward (needed intermediates) ----
    gc = g.cumsum(0); egc = gc.exp()
    tril = torch.tril(torch.ones(C, C, dtype=q.dtype), 0)
    strict = torch.tril(torch.ones(C, C, dtype=q.dtype), -1)
    diff = gc[:, None] - gc[None, :]
    decay = torch.where(tril.bool(), diff, torch.zeros_like(diff)).exp() * tril
    kb = k * beta[:, None]; vb = v * beta[:, None]
    KKt = kb @ k.transpose(-1, -2)
    A = -(KKt * decay) * strict
    T, saved, eye = _doubling_fwd(A)
    u = T @ vb
    kge = kb * egc[:, None]; w = T @ kge
    QKt = q @ k.transpose(-1, -2); ai = QKt * decay
    vp = w @ S; vn = u - vp
    qeg = q * egc[:, None]
    gl = gc[-1]; egl = gl.exp(); gmg = (gl - gc).exp(); kdec = k * gmg[:, None]

    # ---- reverse ----
    dq = torch.zeros_like(q); dkk = torch.zeros_like(k); dvv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta); dgc = torch.zeros_like(gc); d_egc = torch.zeros_like(egc)
    dS_in = torch.zeros_like(S); dT = torch.zeros_like(T); d_decay = torch.zeros_like(decay)

    # S_next = S*egl + kdec^T @ vn
    dS_in += dSn * egl
    d_egl = (dSn * S).sum()
    dkdec = vn @ dSn.transpose(-1, -2)          # [C,dk]
    dvn = kdec @ dSn                             # [C,dv]
    # o = inter + ai@vn
    d_inter = do
    dai = do @ vn.transpose(-1, -2)             # [C,C]
    dvn = dvn + ai.transpose(-1, -2) @ do
    # inter = qeg @ S
    dqeg = d_inter @ S.transpose(-1, -2)
    dS_in += qeg.transpose(-1, -2) @ d_inter
    # vn = u - vp
    du = dvn; dvp = -dvn
    # vp = w @ S
    dw = dvp @ S.transpose(-1, -2)
    dS_in += w.transpose(-1, -2) @ dvp
    # w = T @ kge
    dT += dw @ kge.transpose(-1, -2)
    dkge = T.transpose(-1, -2) @ dw
    # u = T @ vb
    dT += du @ vb.transpose(-1, -2)
    dvb = T.transpose(-1, -2) @ du
    # ai = QKt * decay
    dQKt = dai * decay; d_decay += dai * QKt
    # QKt = q @ k^T
    dq += dQKt @ k; dkk += dQKt.transpose(-1, -2) @ q
    # qeg = q * egc
    dq += dqeg * egc[:, None]; d_egc += (dqeg * q).sum(-1)
    # kge = kb * egc
    dkb = dkge * egc[:, None]; d_egc += (dkge * kb).sum(-1)
    # vb = v * beta
    dvv += dvb * beta[:, None]; dbeta += (dvb * v).sum(-1)
    # T = doubling(A)
    dA = _doubling_vjp(dT, A, saved, eye)
    # A = -(KKt*decay)*strict
    dKKtdecay = -(dA * strict)
    dKKt = dKKtdecay * decay; d_decay += dKKtdecay * KKt
    # KKt = kb @ k^T
    dkb += dKKt @ k; dkk += dKKt.transpose(-1, -2) @ kb
    # kb = k * beta
    dkk += dkb * beta[:, None]; dbeta += (dkb * k).sum(-1)
    # kdec = k * gmg ; gmg = exp(gl - gc)
    dkk += dkdec * gmg[:, None]
    d_gmg = (dkdec * k).sum(-1)                  # [C]
    d_glmgc = d_gmg * gmg                        # d(gl-gc)
    d_egl += 0.0
    dgc += -d_glmgc
    dgl = d_glmgc.sum() + d_egl * egl            # gl feeds egl and (gl-gc)
    # decay[a,b] = exp(gc_a-gc_b)*mask -> d(gc_a-gc_b) = d_decay*decay
    dgc_diff = d_decay * decay                   # [C,C], already masked by decay
    dgc += dgc_diff.sum(1)                        # +gc_a
    dgc += -dgc_diff.sum(0)                       # -gc_b
    # egc = exp(gc)
    dgc += d_egc * egc
    # gl = gc[-1]
    dgc[-1] += dgl
    # gc = cumsum(g) -> dg = reverse cumsum of dgc
    dg = torch.flip(torch.cumsum(torch.flip(dgc, [0]), 0), [0])
    return {"dq": dq, "dk": dkk, "dv": dvv, "dg": dg, "dbeta": dbeta, "dS_in": dS_in}


def _demo():
    torch.manual_seed(0)
    from gdn_bwd_explicit import chunk_step
    C, dk, dv = 64, 128, 128
    def l2(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = l2(torch.randn(C, dk)) * (1/dk**0.5); k = l2(torch.randn(C, dk)); v = torch.randn(C, dv)
    g = -torch.rand(C)*0.3; beta = torch.rand(C); S = torch.randn(dk, dv)*0.1
    do = torch.randn(C, dv); dSn = torch.randn(dk, dv)
    # autograd reference
    ins = [x.clone().double().requires_grad_(True) for x in (q, k, v, g, beta, S)]
    o, Sn = chunk_step(*ins)
    ref = torch.autograd.grad([o, Sn], ins, grad_outputs=[do.double(), dSn.double()])
    ex = chunk_step_bwd(q, k, v, g, beta, S, do, dSn)
    names = ["dq", "dk", "dv", "dg", "dbeta", "dS_in"]
    def cos(a, b):
        a, b = a.reshape(-1).double(), b.reshape(-1).double(); return (a@b/(a.norm()*b.norm()+1e-12)).item()
    print("hand per-chunk VJP vs autograd:")
    for nm, r in zip(names, ref):
        print(f"  {nm}: cos={cos(ex[nm], r):.8f}  max_abs_err={(ex[nm]-r).abs().max().item():.2e}")


if __name__ == "__main__":
    _demo()
