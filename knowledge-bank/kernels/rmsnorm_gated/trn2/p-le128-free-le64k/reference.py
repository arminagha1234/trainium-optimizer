"""Numpy ground truth for Qwen3.5 Qwen3_5RMSNormGated (norm-before-gate):
    out = weight * (x * rsqrt(mean(x^2,-1) + eps)) * silu(gate),  silu(z)=z*sigmoid(z)
Used by kernels/normalization/... to validate the NKI kernel on-device."""
from __future__ import annotations
import numpy as np

def rmsnorm_gated(x, gate, weight, eps=1e-6):
    x = np.asarray(x, np.float64); gate = np.asarray(gate, np.float64); weight = np.asarray(weight, np.float64)
    var = (x * x).mean(-1, keepdims=True)
    xn = x * (1.0 / np.sqrt(var + eps))
    xn = weight * xn
    silu = gate * (1.0 / (1.0 + np.exp(-gate)))
    return xn * silu
