"""mamba2_ssd_fwd.py — Mamba-2 SSD (selective-scan) chunked forward, NKI 0.6.0.

The Lumos-34M backbone scan (real, SISO). Chunk form (from nki-library ssd_torch /
mamba3_ssd_torch_ref.ssd_mamba2_ref) — no triangular solve, so simpler than GDN:
  per chunk (Q tokens): cs=cumsum(dt*A); CB=(C@B^T)*causal; Xs=dt*x*exp(-cs);
  Y_intra=exp(cs)*(CB@Xs); Y_off=exp(cs)*(C@state); y=Y_intra+Y_off(+D*x);
  state = exp(cs_last)*state + B^T@(dt*x*exp(cs_last-cs)).
Single (b,h): x[L,p], dt[L,1], A[1,1], B[L,n], C[L,n], state0[n,p]; L%Q==0, Q,n,p<=128.
Reuses the GDN kernel idioms (scan cumsum, _mm/_tmm/_mmt, chunk loop, state in SBUF).
"""
from __future__ import annotations
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

Q = 128   # chunk size


def _mm(a, b):
    return nisa.tensor_copy(nl.matmul(a, b))


def _tmm(a, b):                       # a^T @ b
    return nisa.tensor_copy(nl.matmul(a, b, transpose_x=True))


def _mmt(a, b):                       # a @ b^T
    return nisa.tensor_copy(nl.matmul(a, nisa.tensor_copy(nisa.nc_transpose(b))))


@nki.jit
def mamba2_ssd_fwd(x, dt, A, B, C, state0, causal):
    L, p = x.shape
    n = B.shape[1]
    nchunks = L // Q
    y_out = nl.ndarray((L, p), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out = nl.ndarray((n, p), dtype=nl.float32, buffer=nl.shared_hbm)
    Ah = nl.broadcast_to(nl.load(A), shape=(Q, 1))    # [Q,1] per-head scalar, part-broadcast
    cmask = nl.load(causal)                           # [Q,Q] lower-tri incl
    ones_1q = nisa.memset((1, Q), 1.0, dtype=nl.float32)
    zero_11 = nisa.memset((1, 1), 0.0, dtype=nl.float32)
    state = nl.load(state0)                           # [n,p], carried in SBUF

    for i in nl.static_range(nchunks):
        cs0 = i * Q
        x_c = nl.load(x[cs0:cs0 + Q, :])              # [Q,p]
        dt_c = nl.load(dt[cs0:cs0 + Q, :])            # [Q,1]
        B_c = nl.load(B[cs0:cs0 + Q, :])              # [Q,n]
        C_c = nl.load(C[cs0:cs0 + Q, :])              # [Q,n]
        # cs = cumsum(dt*A) along the chunk (scan on free axis)
        dtA = nisa.tensor_scalar(dt_c, nl.multiply, Ah)              # [Q,1] (Ah broadcast scalar)
        dtA_row = nisa.tensor_copy(nisa.nc_transpose(dtA))          # [1,Q]
        cs_row = nisa.tensor_tensor_scan(ones_1q, dtA_row, zero_11, nl.multiply, nl.add)  # [1,Q]
        cs = nisa.tensor_copy(nisa.nc_transpose(cs_row))           # [Q,1]
        exp_cs = nisa.activation(nl.exp, cs)                        # [Q,1]
        exp_neg = nisa.activation(nl.exp, nisa.tensor_scalar(cs, nl.multiply, -1.0))  # [Q,1]
        dtx = nisa.tensor_scalar(x_c, nl.multiply, dt_c)            # [Q,p]
        # CB = (C@B^T) * causal
        CB = nisa.tensor_tensor(_mmt(C_c, B_c), cmask, nl.multiply)  # [Q,Q]
        Xs = nisa.tensor_scalar(dtx, nl.multiply, exp_neg)          # [Q,p]
        Y_intra = nisa.tensor_scalar(_mm(CB, Xs), nl.multiply, exp_cs)   # exp_cs*(CB@Xs)
        Y_off = nisa.tensor_scalar(_mm(C_c, state), nl.multiply, exp_cs)  # exp_cs*(C@state)
        y_c = nisa.tensor_tensor(Y_intra, Y_off, nl.add)
        nisa.dma_copy(dst=y_out[cs0:cs0 + Q, :], src=nisa.tensor_copy(y_c))
        # state = exp(cs_last)*state + B^T @ (dtx*exp(cs_last-cs))
        cs_last = nisa.tensor_copy(cs_row[0:1, Q - 1:Q])           # [1,1]
        exp_last = nisa.activation(nl.exp, cs_last)                 # [1,1]
        dec = nisa.activation(nl.exp, nisa.tensor_tensor(nl.broadcast_to(cs_last, shape=(Q, 1)), cs, nl.subtract))  # [Q,1]
        chunk_state = _tmm(B_c, nisa.tensor_scalar(dtx, nl.multiply, dec))   # B_c^T @ (dtx*dec) [n,p]
        state = nisa.tensor_tensor(nisa.tensor_scalar(state, nl.multiply, nl.broadcast_to(exp_last, shape=(n, 1))),
                                   chunk_state, nl.add)
    nisa.dma_copy(dst=state_out[:, :], src=nisa.tensor_copy(state))
    return y_out, state_out
