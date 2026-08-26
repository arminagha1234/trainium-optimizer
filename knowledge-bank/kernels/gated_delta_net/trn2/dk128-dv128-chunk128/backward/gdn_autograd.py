"""gdn_autograd.py — drop-in torch.autograd.Function for the GDN NKI fwd+bwd pair.

Wraps the on-device-validated forward (gdn_chunked_fwd, chunk=128) and backward
(gdn_bwd_batched, chunk=64) as a differentiable op. Same math (only internal
blocking differs). Inputs pre-normalized (kernel boundary):
   q [BH,S,128] l2-normed AND scaled by 1/sqrt(dk)
   k [BH,S,128] l2-normed
   v [BH,S,128]
   g [BH,S,1]   raw per-token log-decay
   beta [BH,S,1] write gate
S must be a multiple of 128 for fwd and of 64 for bwd (=> S%128==0 satisfies both).
"""
from __future__ import annotations
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_bwd_batched import gdn_bwd_batched
# forward lives under the library once merged; local import for now
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gdn_chunked_fwd import gdn_chunked_fwd, make_masks, CHUNK_SIZE as FWD_C

BWD_C = 64
DK = 128


def _bwd_masks():
    tril = np.tril(np.ones((BWD_C, BWD_C), np.float32), 0)
    strict = np.tril(np.ones((BWD_C, BWD_C), np.float32), -1)
    eye = np.eye(BWD_C, dtype=np.float32)
    triu = np.triu(np.ones((BWD_C, BWD_C), np.float32), 0)
    last = np.zeros((BWD_C, 1), np.float32); last[BWD_C - 1, 0] = 1.0
    return tril, strict, eye, triu, last


class GDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta):
        BH, S, D = q.shape
        assert D == DK and S % max(FWD_C, BWD_C) == 0, (S, D)
        dev = q.device
        lower, ident, lower_diag = make_masks()
        S0 = torch.zeros(BH, DK, DK, device=dev, dtype=torch.float32)
        t = lambda a: torch.as_tensor(a, device=dev)
        out, S_final = gdn_chunked_fwd(q, k, v, g, beta, S0,
                                       t(lower), t(ident), t(lower_diag))
        ctx.save_for_backward(q, k, v, g, beta)
        return out

    @staticmethod
    def backward(ctx, dO):
        q, k, v, g, beta = ctx.saved_tensors
        BH = q.shape[0]; dev = q.device
        tril, strict, eye, triu, last = _bwd_masks()
        S0 = torch.zeros(BH, DK, DK, device=dev, dtype=torch.float32)
        t = lambda a: torch.as_tensor(a, device=dev)
        dq, dk, dv, dg, dbeta, dS0 = gdn_bwd_batched(
            q, k, v, g, beta, S0, dO.contiguous(),
            t(tril), t(strict), t(eye), t(triu), t(last))
        return dq, dk, dv, dg, dbeta


def gdn(q, k, v, g, beta):
    return GDN.apply(q, k, v, g, beta)
