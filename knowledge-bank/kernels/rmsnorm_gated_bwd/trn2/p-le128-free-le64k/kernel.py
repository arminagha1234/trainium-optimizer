"""rmsnorm_gated_bwd.py — backward of Qwen3.5 RMSNormGated, NKI 0.6.0.
Forward: ms=mean(x^2); r=rsqrt(ms+eps); xn=x*r; wn=weight*xn; s=silu(gate); out=wn*s.
Given dout -> dx, dweight [1,F], dgate. dweight sums over the partition (token) axis
via ones@(.) matmul. Validated vs torch autograd. x,gate:[P<=128,F]; weight:[1,F]."""
from __future__ import annotations
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def rmsnorm_gated_bwd(x, gate, weight, dout, ones_1p):
    P, F = x.shape
    dx_out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    dw_out = nl.ndarray((1, F), dtype=nl.float32, buffer=nl.shared_hbm)
    dgate_out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.load(x); gt = nl.load(gate); w = nl.load(weight); do = nl.load(dout)
    onesP = nl.load(ones_1p)                                   # [1,P] for partition-sum
    # forward recompute
    sq = nisa.activation(nl.square, xt)
    ss = nl.sum(sq, axis=1, keepdims=True)                    # [P,1]
    ms = nisa.tensor_scalar(ss, nl.multiply, 1.0 / F)
    r = nl.rsqrt(nisa.tensor_scalar(ms, nl.add, 1e-6))        # [P,1]
    xn = nisa.tensor_scalar(xt, nl.multiply, r)               # x*r
    wn = nisa.tensor_tensor(xn, nl.broadcast_to(w, shape=(P, F)), nl.multiply)  # weight*xn
    sig = nl.sigmoid(gt); s = nisa.tensor_tensor(gt, sig, nl.multiply)          # silu(gate)
    # backward
    ds = nisa.tensor_tensor(do, wn, nl.multiply)              # dout*wn
    dwn = nisa.tensor_tensor(do, s, nl.multiply)              # dout*s
    # dgate = ds * silu'(gate),  silu'(g)=sig*(1+g*(1-sig))
    one_m = nisa.tensor_scalar(nisa.tensor_scalar(sig, nl.multiply, -1.0), nl.add, 1.0)  # 1-sig
    silup = nisa.tensor_tensor(sig, nisa.tensor_scalar(nisa.tensor_tensor(gt, one_m, nl.multiply), nl.add, 1.0), nl.multiply)
    dgate = nisa.tensor_tensor(ds, silup, nl.multiply)
    # dxn = dwn*weight ; dweight = sum_P(dwn*xn)
    dxn = nisa.tensor_tensor(dwn, nl.broadcast_to(w, shape=(P, F)), nl.multiply)
    dw = nisa.tensor_copy(nl.matmul(onesP, nisa.tensor_tensor(dwn, xn, nl.multiply)))  # [1,F]
    # dr = sum_F(dxn*x) [P,1]; dms = dr*(-0.5 r^3); dx = dxn*r + dms*2x/F
    dr = nl.sum(nisa.tensor_tensor(dxn, xt, nl.multiply), axis=1, keepdims=True)        # [P,1]
    r3 = nisa.tensor_tensor(nisa.tensor_tensor(r, r, nl.multiply), r, nl.multiply)
    dms = nisa.tensor_scalar(nisa.tensor_tensor(dr, r3, nl.multiply), nl.multiply, -0.5)  # [P,1]
    dx = nisa.tensor_tensor(nisa.tensor_scalar(dxn, nl.multiply, r),
                            nisa.tensor_scalar(nisa.tensor_scalar(xt, nl.multiply, dms), nl.multiply, 2.0 / F), nl.add)
    nisa.dma_copy(dst=dx_out[:, :], src=nisa.tensor_copy(dx))
    nisa.dma_copy(dst=dw_out[:, :], src=nisa.tensor_copy(dw))
    nisa.dma_copy(dst=dgate_out[:, :], src=nisa.tensor_copy(dgate))
    return dx_out, dw_out, dgate_out
