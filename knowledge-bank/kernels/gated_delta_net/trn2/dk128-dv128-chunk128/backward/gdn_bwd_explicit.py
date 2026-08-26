"""gdn_bwd_explicit.py — the EXPLICIT reverse-loop GDN backward (the NKI blueprint).

Factors the chunk forward into a per-chunk step(q,k,v,g,beta,S)->(o,S_next), then:
  forward re-pass stores S_in per chunk; reverse loop carries dS and, per chunk,
  gets the local VJP via autograd.grad on step(...) with grad_outputs=(dO_i, dS).
This is EXACTLY the reverse-chunk-loop structure the NKI backward will run (dS
recurrence + per-chunk local grads) — verified here against the full-autograd
oracle so the NKI port has a gradcheck reference for every gradient.
"""
from __future__ import annotations
import math, torch


def chunk_step(q_i, k_i, v_i, g_i, beta_i, S):
    """One chunk: (q,k,v)[C,d], g,beta[C], S[dk,dv] -> (o[C,dv], S_next[dk,dv])."""
    C, dk = k_i.shape
    gc = g_i.cumsum(0)                                           # [C]
    tril_incl = torch.tril(torch.ones(C, C, dtype=q_i.dtype, device=q_i.device))
    strict = torch.tril(torch.ones(C, C, dtype=q_i.dtype, device=q_i.device), -1)
    diff = gc[:, None] - gc[None, :]
    decay = torch.where(tril_incl.bool(), diff, torch.zeros_like(diff)).exp() * tril_incl
    kb = k_i * beta_i[:, None]; vb = v_i * beta_i[:, None]
    A = -((kb @ k_i.transpose(-1, -2)) * decay) * strict         # strictly-lower
    eye = torch.eye(C, dtype=q_i.dtype, device=q_i.device)
    Lp = A; T = eye + A
    for _ in range(int(math.log2(C)) - 1):                       # doubling inverse
        Lp = Lp @ Lp; T = T @ (eye + Lp)
    u = T @ vb
    w = T @ (kb * gc.exp()[:, None])
    ai = (q_i @ k_i.transpose(-1, -2)) * decay
    vp = w @ S
    vn = u - vp
    inter = (q_i * gc.exp()[:, None]) @ S
    o = inter + ai @ vn
    gl = gc[-1]
    S_next = S * gl.exp() + (k_i * (gl - gc).exp()[:, None]).transpose(-1, -2) @ vn
    return o, S_next


def explicit_backward(q, k, v, g, beta, dO, C=64):
    q, k, v, g, beta, dO = (x.double() for x in (q, k, v, g, beta, dO))
    T, dk = k.shape; dv = v.shape[-1]; n = T // C
    qc = q.reshape(n, C, dk); kc = k.reshape(n, C, dk); vc = v.reshape(n, C, dv)
    gck = g.reshape(n, C); bc = beta.reshape(n, C); dOc = dO.reshape(n, C, dv)
    # forward re-pass: store S entering each chunk
    S = torch.zeros(dk, dv, dtype=q.dtype); S_in = [S]
    for i in range(n):
        _, S = chunk_step(qc[i], kc[i], vc[i], gck[i], bc[i], S)
        S_in.append(S)
    # reverse loop: carry dS; per chunk get the local VJP via autograd on chunk_step
    dS = torch.zeros(dk, dv, dtype=q.dtype)
    dq = torch.zeros_like(q); dk_ = torch.zeros_like(k); dv_ = torch.zeros_like(v)
    dg = torch.zeros_like(g); dbeta = torch.zeros_like(beta)
    for i in reversed(range(n)):
        ins = [x.clone().requires_grad_(True) for x in (qc[i], kc[i], vc[i], gck[i], bc[i])]
        Sin = S_in[i].clone().requires_grad_(True)
        o_i, S_out = chunk_step(*ins, Sin)
        gq, gk, gv, gg, gb, gS = torch.autograd.grad(
            [o_i, S_out], ins + [Sin], grad_outputs=[dOc[i], dS], retain_graph=False)
        dq[i*C:(i+1)*C] = gq; dk_[i*C:(i+1)*C] = gk; dv_[i*C:(i+1)*C] = gv
        dg[i*C:(i+1)*C] = gg; dbeta[i*C:(i+1)*C] = gb
        dS = gS
    return {"dq": dq, "dk": dk_, "dv": dv_, "dg": dg, "dbeta": dbeta}


def _demo():
    import os, importlib.util
    torch.manual_seed(0)
    T, dk, dv = 128, 128, 128
    def l2(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = l2(torch.randn(T, dk)) * (1/dk**0.5); k = l2(torch.randn(T, dk)); v = torch.randn(T, dv)
    g = -torch.rand(T)*0.3; beta = torch.rand(T); dO = torch.randn(T, dv)
    spec = importlib.util.spec_from_file_location(
        "orc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_bwd_oracle.py"))
    orc = importlib.util.module_from_spec(spec); spec.loader.exec_module(orc)
    ref = orc.grads(q, k, v, g, beta, dO, chunk_size=64)
    ex = explicit_backward(q, k, v, g, beta, dO, C=64)
    def cos(a, b):
        a, b = a.reshape(-1).double(), b.reshape(-1).double()
        return (a @ b / (a.norm()*b.norm()+1e-12)).item()
    print("explicit reverse-loop backward vs full-autograd oracle:")
    for kk in ("dq", "dk", "dv", "dg", "dbeta"):
        print(f"  {kk}: cos={cos(ex[kk], ref[kk]):.8f}  max_abs_err={(ex[kk]-ref[kk]).abs().max().item():.2e}")


if __name__ == "__main__":
    _demo()
