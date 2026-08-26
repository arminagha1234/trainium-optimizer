"""mamba2_ssd_autograd.py — Mamba-2 SSD as a trainable torch op on Trainium.

Wraps the validated NKI fwd (mamba2_ssd_fwd, cos 1.0) + bwd (mamba2_ssd_bwd, all-6
grads cos 1.0) in torch.autograd.Function, looping over (batch·heads). Mirrors the
GDN gdn() wrapper. Public: mamba2_ssd(x, dt, A, B, C, state0=None) -> y.
  x:[Bb,H,L,p]  dt:[Bb,H,L]  A:[H]  B,C:[Bb,H,L,n]  state0:[Bb,H,n,p] or None
  y:[Bb,H,L,p]
L%128==0, p<=128, n<=128. All internal compute fp32.
"""
from __future__ import annotations
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mamba2_ssd_fwd import mamba2_ssd_fwd
from mamba2_ssd_bwd import mamba2_ssd_bwd

Q = 128
_causal = _tri_ge = None


def _consts(dev):
    global _causal, _tri_ge
    if _causal is None or _causal.device != dev:
        _causal = torch.from_numpy(np.tril(np.ones((Q, Q), np.float32))).to(dev)
        _tri_ge = torch.from_numpy(np.triu(np.ones((Q, Q), np.float32))).to(dev)
    return _causal, _tri_ge


class Mamba2SSD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, state0):
        Bb, H, L, p = x.shape
        n = B.shape[-1]
        dev = x.device
        causal, _ = _consts(dev)
        ys = []
        for b in range(Bb):
            for h in range(H):
                A_bh = A[h].reshape(1, 1).float()
                st = state0[b, h] if state0 is not None else torch.zeros(n, p, device=dev)
                y, _sf = mamba2_ssd_fwd(x[b, h].float().contiguous(), dt[b, h].reshape(L, 1).float().contiguous(),
                                        A_bh, B[b, h].float().contiguous(), C[b, h].float().contiguous(),
                                        st.float().contiguous(), causal)
                ys.append(y)
        ctx.save_for_backward(x, dt, A, B, C)
        ctx.state0 = state0
        ctx.shape = (Bb, H, L, p, n)
        return torch.stack(ys).reshape(Bb, H, L, p)

    @staticmethod
    def backward(ctx, dy):
        x, dt, A, B, C = ctx.saved_tensors
        Bb, H, L, p, n = ctx.shape
        dev = x.device
        causal, tri_ge = _consts(dev)
        state0 = ctx.state0
        dx = torch.empty_like(x); ddt = torch.empty_like(dt)
        dB = torch.empty_like(B); dC = torch.empty_like(C)
        dA = torch.zeros_like(A); dstate0 = torch.zeros(Bb, H, n, p, device=dev)
        zero_np = torch.zeros(n, p, device=dev)
        for b in range(Bb):
            for h in range(H):
                A_bh = A[h].reshape(1, 1).float()
                st = state0[b, h] if state0 is not None else zero_np
                dSf = torch.zeros(n, p, device=dev)   # no grad from final state (y-only loss path)
                gx, gdt, gA, gB, gC, gs0 = mamba2_ssd_bwd(
                    x[b, h].float().contiguous(), dt[b, h].reshape(L, 1).float().contiguous(), A_bh,
                    B[b, h].float().contiguous(), C[b, h].float().contiguous(), st.float().contiguous(),
                    dy[b, h].float().contiguous(), dSf, causal, tri_ge)
                dx[b, h] = gx; ddt[b, h] = gdt.reshape(L); dB[b, h] = gB; dC[b, h] = gC
                dA[h] = dA[h] + gA.reshape(-1)[0]
                if state0 is not None:
                    dstate0[b, h] = gs0
        d_state0 = dstate0 if state0 is not None else None
        return dx, ddt, dA, dB, dC, d_state0


def mamba2_ssd(x, dt, A, B, C, state0=None):
    return Mamba2SSD.apply(x, dt, A, B, C, state0)
