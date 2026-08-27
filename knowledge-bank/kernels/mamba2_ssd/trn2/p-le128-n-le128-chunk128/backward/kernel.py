"""mamba2_ssd_bwd.py — Mamba-2 SSD chunked backward, NKI 0.6.0.

Ports the hand VJP (mamba2_ssd_bwd_ref, cos 1.0 vs autograd) to device. Two passes
in one NEFF: (1) recompute forward, holding each chunk's entry state in a Python list
of SBUF tiles (static_range unrolls, so this is legal); (2) reverse loop applying the
VJP. Every step is a matmul, tensor_scalar, or the reverse-cumsum d_dtA = tri_ge@d_cs
plus the +d_csl-to-every-row trick (tri_ge@e_last == ones). No triangular solve.
Grads: dx[L,p], ddt[L,1], dA[1,1], dB[L,n], dC[L,n], dstate0[n,p].
"""
from __future__ import annotations
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

Q = 128


def _mm(a, b):                        # a @ b
    return nisa.tensor_copy(nl.matmul(a, b))


def _tmm(a, b):                       # a^T @ b
    return nisa.tensor_copy(nl.matmul(a, b, transpose_x=True))


def _mmt(a, b):                       # a @ b^T
    return nisa.tensor_copy(nl.matmul(a, nisa.tensor_copy(nisa.nc_transpose(b))))


def _colsum(t):                       # sum over free axis -> [P,1]
    return nl.sum(t, axis=1, keepdims=True)


@nki.jit
def mamba2_ssd_bwd(x, dt, A, B, C, state0, dy, dS_final, causal, tri_ge):
    L, p = x.shape
    n = B.shape[1]
    nchunks = L // Q
    dx = nl.ndarray((L, p), dtype=nl.float32, buffer=nl.shared_hbm)
    ddt = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dA_out = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dB = nl.ndarray((L, n), dtype=nl.float32, buffer=nl.shared_hbm)
    dC = nl.ndarray((L, n), dtype=nl.float32, buffer=nl.shared_hbm)
    dstate0 = nl.ndarray((n, p), dtype=nl.float32, buffer=nl.shared_hbm)

    Ah = nl.broadcast_to(nl.load(A), shape=(Q, 1))
    cmask = nl.load(causal)                 # [Q,Q] lower-tri incl
    trig = nl.load(tri_ge)                  # [Q,Q] upper-tri incl (reverse-cumsum)
    ones_1q = nisa.memset((1, Q), 1.0, dtype=nl.float32)
    zero_11 = nisa.memset((1, 1), 0.0, dtype=nl.float32)

    # ---- pass 1: forward recompute, cache entry state per chunk -----------------
    st = nl.load(state0)
    states = [st]
    for i in nl.static_range(nchunks):
        s0 = i * Q
        x_c = nl.load(x[s0:s0 + Q, :]); dt_c = nl.load(dt[s0:s0 + Q, :])
        B_c = nl.load(B[s0:s0 + Q, :])
        dtA = nisa.tensor_scalar(dt_c, nl.multiply, Ah)
        dtA_row = nisa.tensor_copy(nisa.nc_transpose(dtA))
        cs_row = nisa.tensor_tensor_scan(ones_1q, dtA_row, zero_11, nl.multiply, nl.add)
        cs = nisa.tensor_copy(nisa.nc_transpose(cs_row))
        csl = nisa.tensor_copy(cs_row[0:1, Q - 1:Q])
        dec = nisa.activation(nl.exp, nisa.tensor_tensor(nl.broadcast_to(csl, shape=(Q, 1)), cs, nl.subtract))
        dtx = nisa.tensor_scalar(x_c, nl.multiply, dt_c)
        chunk_state = _tmm(B_c, nisa.tensor_scalar(dtx, nl.multiply, dec))
        st = nisa.tensor_tensor(
            nisa.tensor_scalar(st, nl.multiply, nl.broadcast_to(nisa.activation(nl.exp, csl), shape=(n, 1))),
            chunk_state, nl.add)
        states.append(st)

    # ---- pass 2: reverse VJP ----------------------------------------------------
    dstate = nl.load(dS_final)              # dS_out for the last chunk
    dA_acc = nisa.memset((1, 1), 0.0, dtype=nl.float32)
    for j in nl.static_range(nchunks):
        i = nchunks - 1 - j
        s0 = i * Q
        x_c = nl.load(x[s0:s0 + Q, :]); dt_c = nl.load(dt[s0:s0 + Q, :])
        B_c = nl.load(B[s0:s0 + Q, :]); C_c = nl.load(C[s0:s0 + Q, :])
        dy_c = nl.load(dy[s0:s0 + Q, :])
        st_in = states[i]                   # entry state of chunk i
        # recompute intermediates
        dtA = nisa.tensor_scalar(dt_c, nl.multiply, Ah)
        dtA_row = nisa.tensor_copy(nisa.nc_transpose(dtA))
        cs_row = nisa.tensor_tensor_scan(ones_1q, dtA_row, zero_11, nl.multiply, nl.add)
        cs = nisa.tensor_copy(nisa.nc_transpose(cs_row))
        csl = nisa.tensor_copy(cs_row[0:1, Q - 1:Q])            # [1,1]
        ecs = nisa.activation(nl.exp, cs)
        encs = nisa.activation(nl.exp, nisa.tensor_scalar(cs, nl.multiply, -1.0))
        dec = nisa.activation(nl.exp, nisa.tensor_tensor(nl.broadcast_to(csl, shape=(Q, 1)), cs, nl.subtract))
        dtx = nisa.tensor_scalar(x_c, nl.multiply, dt_c)
        CB = nisa.tensor_tensor(_mmt(C_c, B_c), cmask, nl.multiply)
        Xs = nisa.tensor_scalar(dtx, nl.multiply, encs)
        P = _mm(CB, Xs)
        G = _mm(C_c, st_in)
        Sd = nisa.tensor_scalar(dtx, nl.multiply, dec)
        dS_out = dstate                     # grad of chunk-i output state
        # y = ecs*P + ecs*G
        d_ecs = nisa.tensor_tensor(_colsum(nisa.tensor_tensor(dy_c, P, nl.multiply)),
                                   _colsum(nisa.tensor_tensor(dy_c, G, nl.multiply)), nl.add)
        dP = nisa.tensor_scalar(dy_c, nl.multiply, ecs)
        dG = nisa.tensor_scalar(dy_c, nl.multiply, ecs)
        # P = CB@Xs ; G = C_c@st_in
        dCB = _mmt(dP, Xs)                                     # dP @ Xs^T
        dXs = _tmm(CB, dP)                                     # CB^T @ dP
        dC_c = _mmt(dG, st_in)                                 # dG @ st_in^T  -> [Q,n]
        dstate_in = _tmm(C_c, dG)                             # C_c^T @ dG   (Y_off term)
        # CB = (C@B^T)*causal
        dM = nisa.tensor_tensor(dCB, cmask, nl.multiply)
        dC_c = nisa.tensor_tensor(dC_c, _mm(dM, B_c), nl.add)  # + dM @ B_c
        dB_c = _tmm(dM, C_c)                                   # dM^T @ C_c
        # Xs = dtx*encs
        d_dtx = nisa.tensor_scalar(dXs, nl.multiply, encs)
        d_encs = _colsum(nisa.tensor_tensor(dXs, dtx, nl.multiply))
        # state_out = exp(csl)*st_in + B^T@Sd
        E = nisa.tensor_tensor(dS_out, st_in, nl.multiply)     # [n,p]
        d_exp_csl = _colsum(nisa.tensor_copy(nisa.nc_transpose(_colsum(E))))  # sum-all -> [1,1]
        exp_csl = nisa.activation(nl.exp, csl)
        dstate_in = nisa.tensor_tensor(
            dstate_in, nisa.tensor_scalar(dS_out, nl.multiply, nl.broadcast_to(exp_csl, shape=(n, 1))), nl.add)
        dB_c = nisa.tensor_tensor(dB_c, _mmt(Sd, dS_out), nl.add)  # + Sd @ dS_out^T
        dSd = _mm(B_c, dS_out)                                 # B_c @ dS_out
        d_dtx = nisa.tensor_tensor(d_dtx, nisa.tensor_scalar(dSd, nl.multiply, dec), nl.add)
        d_dec = _colsum(nisa.tensor_tensor(dSd, dtx, nl.multiply))
        # exp chain -> cs
        d_cs = nisa.tensor_tensor(
            nisa.tensor_tensor(nisa.tensor_tensor(d_ecs, ecs, nl.multiply),
                               nisa.tensor_tensor(d_encs, encs, nl.multiply), nl.subtract),
            nisa.tensor_tensor(d_dec, dec, nl.multiply), nl.subtract)
        d_csl = nisa.tensor_tensor(_colsum(nisa.tensor_copy(nisa.nc_transpose(_colsum(
                    nisa.tensor_tensor(d_dec, dec, nl.multiply))))),
                    nisa.tensor_tensor(d_exp_csl, exp_csl, nl.multiply), nl.add)   # [1,1]
        # cs = cumsum(dtA) -> reverse cumsum; +d_csl to every row (tri_ge@e_last == ones)
        d_dtA = nisa.tensor_scalar(_mm(trig, d_cs), nl.add, nl.broadcast_to(d_csl, shape=(Q, 1)))
        # dtA = dt*A ; dtx = dt*x
        ddt_c = nisa.tensor_tensor(nisa.tensor_scalar(d_dtA, nl.multiply, Ah),
                                   _colsum(nisa.tensor_tensor(d_dtx, x_c, nl.multiply)), nl.add)
        dx_c = nisa.tensor_scalar(d_dtx, nl.multiply, dt_c)
        dA_acc = nisa.tensor_tensor(
            dA_acc, _colsum(nisa.tensor_copy(nisa.nc_transpose(
                nisa.tensor_tensor(d_dtA, dt_c, nl.multiply)))), nl.add)
        # store
        nisa.dma_copy(dst=dx[s0:s0 + Q, :], src=nisa.tensor_copy(dx_c))
        nisa.dma_copy(dst=ddt[s0:s0 + Q, :], src=nisa.tensor_copy(ddt_c))
        nisa.dma_copy(dst=dB[s0:s0 + Q, :], src=nisa.tensor_copy(dB_c))
        nisa.dma_copy(dst=dC[s0:s0 + Q, :], src=nisa.tensor_copy(dC_c))
        dstate = dstate_in                  # -> previous chunk's dS_out
    nisa.dma_copy(dst=dstate0[:, :], src=nisa.tensor_copy(dstate))
    nisa.dma_copy(dst=dA_out[:, :], src=nisa.tensor_copy(dA_acc))
    return dx, ddt, dA_out, dB, dC, dstate0
