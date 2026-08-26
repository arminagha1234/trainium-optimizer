"""rmsnorm_gated.py — Qwen3.5 RMSNormGated as an NKI kernel (neuronxcc.nki 0.6.0).

    out = weight * (x * rsqrt(mean(x^2,-1) + eps)) * silu(gate)     ("norm before gate")

No NKI version of this fused gated-RMSNorm existed anywhere (survey gap); it is
reused by GDN / KDA / Mamba2 / Nemotron-H / Zamba2 gated-linear-attention blocks.
Elementwise + one free-axis reduction — no matmul. Gate path kept fp32.
x, gate: [P, F] (tokens on partition, normed dim F on free); weight: [1, F].
"""
from __future__ import annotations
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

def make_weight(F):  # convenience for callers/tests
    return np.ones((1, F), np.float32)

@nki.jit
def rmsnorm_gated(x, gate, weight):
    P, F = x.shape
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.load(x)                                   # [P,F]
    gt = nl.load(gate)                                # [P,F]
    w  = nl.load(weight)                              # [1,F]
    # mean(x^2) over the free axis in ONE Scalar-engine pass (square + reduce fused)
    sq = nisa.activation(nl.square, xt)                          # [P,F] x^2 (Scalar engine)
    ss = nl.sum(sq, axis=1, keepdims=True)                       # [P,1] sum of squares
    ms = nisa.tensor_scalar(ss, nl.multiply, 1.0 / F)            # [P,1] mean-square
    inv = nl.rsqrt(nisa.tensor_scalar(ms, nl.add, 1e-6))          # [P,1] 1/sqrt(ms+eps)
    xn = nisa.tensor_scalar(xt, nl.multiply, inv)                 # x * inv  (per-row scalar)
    xn = nisa.tensor_tensor(xn, nl.broadcast_to(w, shape=(P, F)), nl.multiply)  # * weight
    sg = nisa.tensor_tensor(gt, nl.sigmoid(gt), nl.multiply)     # silu(gate) = g*sigmoid(g)
    o  = nisa.tensor_tensor(xn, sg, nl.multiply)
    nisa.dma_copy(dst=out[:, :], src=o)
    return out
