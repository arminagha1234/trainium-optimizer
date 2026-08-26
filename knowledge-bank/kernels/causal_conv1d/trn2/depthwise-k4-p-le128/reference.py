"""Numpy ground truth for the Qwen3.5 GDN short-conv: depthwise causal conv1d,
kernel_size K (=4), NO bias, left-pad K-1, truncate to L, then activation (silu).
    out[c,t] = act( sum_{i=0..K-1} w[c,i] * xpad[c, t+i] ),  xpad = [0]*(K-1) ++ x
x: [C, L], weight: [C, K].  (act='silu' by default; None for linear.)"""
from __future__ import annotations
import numpy as np

def causal_conv1d(x, weight, activation="silu"):
    x = np.asarray(x, np.float64); w = np.asarray(weight, np.float64)
    C, L = x.shape; K = w.shape[1]
    xp = np.concatenate([np.zeros((C, K - 1)), x], axis=1)      # left causal pad
    out = np.zeros((C, L))
    for i in range(K):
        out += w[:, i:i+1] * xp[:, i:i+L]
    if activation == "silu":
        out = out * (1.0 / (1.0 + np.exp(-out)))
    return out
