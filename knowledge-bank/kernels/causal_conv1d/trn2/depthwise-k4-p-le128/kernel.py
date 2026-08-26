"""conv1d_causal.py — Qwen3.5 GDN depthwise causal conv1d (k=4) + SiLU, NKI 0.6.0.

Depthwise (groups=channels), kernel_size K, no bias, causal (left-pad K-1),
output truncated to L, then SiLU (hidden_act). Reused by the GDN/KDA/Mamba mixer
short-conv. K taps of shift-multiply-accumulate — no matmul.
x: [C<=128, L] (channels on partition, length on free); weight: [C, K].

Padding trick (avoids an output slice-write, which return-form ops can't do):
build a zero-initialised padded buffer xpad[C, L+K-1] and dma_copy x into its
[:, K-1:] slice; then tap i reads the clean window xpad[:, i:i+L] and the causal
zeros fall out for free. out[:,t] = sum_i w[:,i] * xpad[:, t+i].
"""
from __future__ import annotations
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

K = 4  # linear_conv_kernel_dim on Qwen3.5

@nki.jit
def causal_conv1d(x, weight):
    C, L = x.shape
    out = nl.ndarray((C, L), dtype=nl.float32, buffer=nl.shared_hbm)
    w = nl.load(weight)                                         # [C,K]
    xpad = nisa.memset((C, L + K - 1), 0.0, dtype=nl.float32)   # zeros [C, L+K-1] (SBUF)
    nisa.dma_copy(dst=xpad[:, K - 1:L + K - 1], src=x)          # place x after the causal pad
    acc = None
    for i in nl.static_range(K):
        contrib = nisa.tensor_scalar(xpad[:, i:i + L], nl.multiply, w[:, i:i + 1])
        acc = contrib if acc is None else nisa.tensor_tensor(acc, contrib, nl.add)
    o = nisa.tensor_tensor(acc, nl.sigmoid(acc), nl.multiply)   # SiLU = a * sigmoid(a)
    nisa.dma_copy(dst=out[:, :], src=o)
    return out
