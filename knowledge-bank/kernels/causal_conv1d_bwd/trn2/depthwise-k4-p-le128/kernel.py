"""conv1d_bwd.py — backward of the depthwise causal conv1d (k=4) + SiLU, NKI 0.6.0.

Forward: pre[c,t] = sum_i w[c,i]*xpad[c,t+i];  out = silu(pre) = pre*sigmoid(pre).
Given dout, returns dx [C,L] and dweight [C,K]. Recomputes `pre` from x,w (cheap;
strategy-B "recompute in backward"). No matmul — K-tap shift/reduce.
  d_pre  = dout * silu'(pre),  silu'(p) = sig*(1 + p*(1-sig))
  dw[:,i]= sum_t d_pre[:,t] * xpad[:, t+i]                 (free-axis reduce, K taps)
  dx[:,n]= sum_u w[:, K-1-u] * dpre_rpad[:, n+u]           (flipped taps, right-pad)
x, dout: [C<=128, L]; weight: [C, K]. First NKI conv1d backward in the tree.
"""
from __future__ import annotations
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

K = 4

@nki.jit
def causal_conv1d_bwd(x, weight, dout):
    C, L = x.shape
    dx_out = nl.ndarray((C, L), dtype=nl.float32, buffer=nl.shared_hbm)
    dw_out = nl.ndarray((C, K), dtype=nl.float32, buffer=nl.shared_hbm)
    w  = nl.load(weight)                                       # [C,K]
    do = nl.load(dout)                                         # [C,L]

    # recompute pre = causal depthwise conv (no activation)
    xpad = nisa.memset((C, L + K - 1), 0.0, dtype=nl.float32)
    nisa.dma_copy(dst=xpad[:, K - 1:L + K - 1], src=x)
    pre = None
    for i in nl.static_range(K):
        c = nisa.tensor_scalar(xpad[:, i:i + L], nl.multiply, w[:, i:i + 1])
        pre = c if pre is None else nisa.tensor_tensor(pre, c, nl.add)

    # d_pre = dout * silu'(pre),  silu'(p) = sig*(1 + p*(1-sig))
    sig = nl.sigmoid(pre)
    one_m = nisa.tensor_scalar(sig, nl.multiply, -1.0)         # -sig
    one_m = nisa.tensor_scalar(one_m, nl.add, 1.0)            # 1-sig
    pm = nisa.tensor_tensor(pre, one_m, nl.multiply)          # p*(1-sig)
    pm = nisa.tensor_scalar(pm, nl.add, 1.0)                  # 1 + p*(1-sig)
    dsilu = nisa.tensor_tensor(sig, pm, nl.multiply)          # silu'(pre)
    d_pre = nisa.tensor_tensor(do, dsilu, nl.multiply)        # [C,L]

    # dw[:,i] = sum_t d_pre[:,t] * xpad[:, t+i]   (reduce over free) -> write column
    for i in nl.static_range(K):
        prod = nisa.tensor_tensor(d_pre, xpad[:, i:i + L], nl.multiply)
        col = nl.sum(prod, axis=1, keepdims=True)             # [C,1]
        nisa.dma_copy(dst=dw_out[:, i:i + 1], src=col)

    # dx[:,n] = sum_u w[:, K-1-u] * dpre_rpad[:, n+u]  (right-pad d_pre, flipped taps)
    dprp = nisa.memset((C, L + K - 1), 0.0, dtype=nl.float32)
    nisa.dma_copy(dst=dprp[:, 0:L], src=d_pre)
    dx = None
    for u in nl.static_range(K):
        c = nisa.tensor_scalar(dprp[:, u:u + L], nl.multiply, w[:, K - 1 - u:K - u])
        dx = c if dx is None else nisa.tensor_tensor(dx, c, nl.add)
    nisa.dma_copy(dst=dx_out[:, :], src=dx)
    return dx_out, dw_out
