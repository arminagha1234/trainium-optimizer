"""gdn_chunk_bwd_nki.py — the per-chunk GDN backward VJP as an NKI kernel (0.6.0).

Mechanical translation of the VERIFIED gdn_chunk_bwd_explicit.chunk_step_bwd
(matched autograd to ~1e-15). Given one chunk's (q,k,v,g,beta,S) and upstream
(do, dSn) -> (dq,dk,dv,dg,dbeta,dS_in). Includes the doubling-inverse backward
and the gate reverse-cumsum for dg. Validated on-device vs the torch reference.

matmul helpers (module-level — NKI forbids inner defs):
  _mm(a,b)   = a @ b        (nl.matmul fuses the stationary transpose)
  _tmm(a,b)  = a^T @ b      (transpose_x=True)
  _mmt(a,b)  = a @ b^T      (transpose b first, then a@bT)
All return SBUF tiles (PSUM results copied out) so they can feed further matmuls.
"""
from __future__ import annotations
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

C = 64
DK = 128
DV = 128
LEVELS = int(math.log2(C)) - 1     # doubling levels (5 for C=64)


def _mm(a, b):
    return nisa.tensor_copy(nl.matmul(a, b))


def _tmm(a, b):
    return nisa.tensor_copy(nl.matmul(a, b, transpose_x=True))


def _mmt(a, b):
    bt = nisa.tensor_copy(nisa.nc_transpose(b))
    return nisa.tensor_copy(nl.matmul(a, bt))


@nki.jit
def gdn_chunk_bwd(q, k, v, g, beta, S, do, dSn,
                  tril_incl, strict, eye, revJ, last_mask):
    # tril_incl/strict/eye [C,C]; revJ [C,C] anti-identity (free-axis flip). g,beta [C,1].
    dq_out = nl.ndarray((C, DK), dtype=nl.float32, buffer=nl.shared_hbm)
    dk_out = nl.ndarray((C, DK), dtype=nl.float32, buffer=nl.shared_hbm)
    dv_out = nl.ndarray((C, DV), dtype=nl.float32, buffer=nl.shared_hbm)
    dg_out = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbeta_out = nl.ndarray((C, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dSin_out = nl.ndarray((DK, DV), dtype=nl.float32, buffer=nl.shared_hbm)

    qt = nl.load(q); kt = nl.load(k); vt = nl.load(v)
    gcol = nl.load(g); bcol = nl.load(beta)
    St = nl.load(S); dot = nl.load(do); dSnt = nl.load(dSn)
    trilm = nl.load(tril_incl); strictm = nl.load(strict); eyem = nl.load(eye); J = nl.load(revJ)

    # ---- forward recompute ----
    # gc = cumsum(g): flip col->row, scan, unflip. g is [C,1]; scan runs on free axis.
    grow = nisa.tensor_copy(nisa.nc_transpose(gcol))              # [1,C]
    ones_1c = nisa.memset((1, C), 1.0, dtype=nl.float32)
    zero_11 = nisa.memset((1, 1), 0.0, dtype=nl.float32)
    gcrow = nisa.tensor_tensor_scan(ones_1c, grow, zero_11, nl.multiply, nl.add)  # [1,C]
    gccol = nisa.tensor_copy(nisa.nc_transpose(gcrow))            # [C,1]
    egc = nisa.activation(nl.exp, gccol)                          # [C,1]
    # decay[a,b] = exp(gc_a - gc_b) * tril_incl
    gc_rows = nl.broadcast_to(gccol, shape=(C, C))                # [a,b]=gc_a
    gc_cols = nl.broadcast_to(gcrow, shape=(C, C))                # [a,b]=gc_b
    gdiff = nisa.tensor_tensor(gc_rows, gc_cols, nl.subtract)
    gdiff = nisa.tensor_scalar(gdiff, nl.minimum, 0.0)           # clamp <=0
    decay = nisa.tensor_tensor(nisa.activation(nl.exp, gdiff), trilm, nl.multiply)
    kb = nisa.tensor_scalar(kt, nl.multiply, bcol)               # k*beta [C,DK]
    vb = nisa.tensor_scalar(vt, nl.multiply, bcol)               # v*beta [C,DV]
    KKt = _mmt(kb, kt)                                           # kb@k^T [C,C]
    A = nisa.tensor_tensor(nisa.tensor_scalar(
        nisa.tensor_tensor(KKt, decay, nl.multiply), nl.multiply, -1.0), strictm, nl.multiply)
    # doubling fwd, saving (T_prev, L_prev, L_j) per level
    Lp = nisa.tensor_copy(A); T = nisa.tensor_tensor(eyem, A, nl.add)
    Tprev_s = []; Lprev_s = []; Lj_s = []
    for _ in nl.static_range(LEVELS):
        Tprev_s.append(nisa.tensor_copy(T)); Lprev_s.append(nisa.tensor_copy(Lp))
        Lp = _mm(Lp, Lp)
        Lj_s.append(nisa.tensor_copy(Lp))
        T = _mm(T, nisa.tensor_tensor(eyem, Lp, nl.add))
    egc_col = egc
    kge = nisa.tensor_scalar(kb, nl.multiply, egc_col)          # kb*exp(gc) [C,DK]
    u = _mm(T, vb)                                               # [C,DV]
    w = _mm(T, kge)                                             # [C,DK]
    QKt = _mmt(qt, kt)                                          # q@k^T [C,C]
    ai = nisa.tensor_tensor(QKt, decay, nl.multiply)
    vp = _mm(w, St)                                             # w@S [C,DV]
    vn = nisa.tensor_tensor(u, vp, nl.subtract)
    qeg = nisa.tensor_scalar(qt, nl.multiply, egc_col)         # q*exp(gc)
    # gl = gc[-1], egl, gmg=exp(gl-gc)
    gl_11 = nisa.tensor_copy(gcrow[0:1, C - 1:C])              # [1,1]
    egl = nisa.activation(nl.exp, gl_11)                       # [1,1]
    gl_col = nl.broadcast_to(gl_11, shape=(C, 1))
    gmg = nisa.activation(nl.exp, nisa.tensor_tensor(gl_col, gccol, nl.subtract))  # [C,1]
    kdec = nisa.tensor_scalar(kt, nl.multiply, gmg)            # [C,DK]

    # ---- reverse VJP ----
    egl_col = nl.broadcast_to(egl, shape=(DK, 1))              # broadcast to state rows
    dS_in = nisa.tensor_scalar(dSnt, nl.multiply, egl_col)     # dSn*egl [DK,DV]
    dkdec = _mmt(vn, dSnt)                                     # vn@dSn^T [C,DK]
    dvn = _mm(kdec, dSnt)                                      # kdec@dSn [C,DV]
    dai = _mmt(dot, vn)                                        # do@vn^T [C,C]
    dvn = nisa.tensor_tensor(dvn, _tmm(ai, dot), nl.add)      # + ai^T@do
    dqeg = _mmt(dot, St)                                       # do@S^T [C,DK]
    dS_in = nisa.tensor_tensor(dS_in, _tmm(qeg, dot), nl.add) # + qeg^T@do
    du = dvn; dvp = nisa.tensor_scalar(dvn, nl.multiply, -1.0)
    dw = _mmt(dvp, St)                                        # dvp@S^T [C,DK]
    dS_in = nisa.tensor_tensor(dS_in, _tmm(w, dvp), nl.add)   # + w^T@dvp
    dT = nisa.tensor_tensor(_mmt(dw, kge), _mmt(du, vb), nl.add)  # dw@kge^T + du@vb^T
    dkge = _tmm(T, dw)                                        # T^T@dw [C,DK]
    dvb = _tmm(T, du)                                         # T^T@du [C,DV]
    dQKt = nisa.tensor_tensor(dai, decay, nl.multiply)
    d_decay = nisa.tensor_tensor(dai, QKt, nl.multiply)
    dq = _mm(dQKt, kt)                                        # dQKt@k [C,DK]
    dk = _tmm(dQKt, qt)                                       # dQKt^T@q [C,DK]
    dq = nisa.tensor_tensor(dq, nisa.tensor_scalar(dqeg, nl.multiply, egc_col), nl.add)
    # d_egc from qeg: sum_dk(dqeg*q); from kge: sum_dk(dkge*kb) (added after dkge below)
    d_egc = nl.sum(nisa.tensor_tensor(dqeg, qt, nl.multiply), axis=1, keepdims=True)  # [C,1]
    dkb = nisa.tensor_scalar(dkge, nl.multiply, egc_col)
    d_egc = nisa.tensor_tensor(d_egc, nl.sum(nisa.tensor_tensor(dkge, kb, nl.multiply), axis=1, keepdims=True), nl.add)
    dv = nisa.tensor_scalar(dvb, nl.multiply, bcol)          # dvb*beta
    dbeta = nl.sum(nisa.tensor_tensor(dvb, vt, nl.multiply), axis=1, keepdims=True)  # from v side
    # doubling VJP
    dL_acc = nisa.memset((C, C), 0.0, dtype=nl.float32)
    for j in nl.static_range(LEVELS):
        idx = LEVELS - 1 - j
        Tp = Tprev_s[idx]; Lprev = Lprev_s[idx]; Lj = Lj_s[idx]
        dT_prev = _mmt(dT, nisa.tensor_tensor(eyem, Lj, nl.add))     # dT@(I+Lj)^T
        dLj = nisa.tensor_tensor(_tmm(Tp, dT), dL_acc, nl.add)       # Tp^T@dT + carry
        dL_prev = nisa.tensor_tensor(_mmt(dLj, Lprev), _tmm(Lprev, dLj), nl.add)
        dT = dT_prev; dL_acc = dL_prev
    dA = nisa.tensor_tensor(dT, dL_acc, nl.add)             # I+A term + L0 term
    dKKtdecay = nisa.tensor_scalar(nisa.tensor_tensor(dA, strictm, nl.multiply), nl.multiply, -1.0)
    dKKt = nisa.tensor_tensor(dKKtdecay, decay, nl.multiply)
    d_decay = nisa.tensor_tensor(d_decay, nisa.tensor_tensor(dKKtdecay, KKt, nl.multiply), nl.add)
    dkb = nisa.tensor_tensor(dkb, _mm(dKKt, kt), nl.add)    # + dKKt@k
    dk = nisa.tensor_tensor(dk, _tmm(dKKt, kb), nl.add)     # + dKKt^T@kb
    dk = nisa.tensor_tensor(dk, nisa.tensor_scalar(dkb, nl.multiply, bcol), nl.add)  # kb=k*beta
    dbeta = nisa.tensor_tensor(dbeta, nl.sum(nisa.tensor_tensor(dkb, kt, nl.multiply), axis=1, keepdims=True), nl.add)
    # kdec = k*gmg
    dk = nisa.tensor_tensor(dk, nisa.tensor_scalar(dkdec, nl.multiply, gmg), nl.add)
    d_gmg = nl.sum(nisa.tensor_tensor(dkdec, kt, nl.multiply), axis=1, keepdims=True)  # [C,1]
    d_glmgc = nisa.tensor_tensor(d_gmg, gmg, nl.multiply)   # d(gl-gc)
    dgc = nisa.tensor_scalar(d_glmgc, nl.multiply, -1.0)   # -d(gl-gc) into gc
    # decay: d(gc_a-gc_b) = d_decay*decay ; dgc += rowsum - colsum
    dgc_diff = nisa.tensor_tensor(d_decay, decay, nl.multiply)
    dgc = nisa.tensor_tensor(dgc, nl.sum(dgc_diff, axis=1, keepdims=True), nl.add)  # +gc_a (rows)
    dgc_diff_T = nisa.tensor_copy(nisa.nc_transpose(dgc_diff))
    dgc = nisa.tensor_tensor(dgc, nisa.tensor_scalar(nl.sum(dgc_diff_T, axis=1, keepdims=True), nl.multiply, -1.0), nl.add)  # -gc_b (cols)
    # egc = exp(gc): dgc += d_egc*egc
    dgc = nisa.tensor_tensor(dgc, nisa.tensor_tensor(d_egc, egc_col, nl.multiply), nl.add)
    # gl feeds gc[-1] and egl. dgl = sum(d_glmgc) + d_egl*egl,  d_egl = sum_all(dSn*S).
    dglmgc_row = nisa.tensor_copy(nisa.nc_transpose(d_glmgc))              # [1,C]
    sum_dglmgc = nl.sum(dglmgc_row, axis=1, keepdims=True)                 # [1,1]
    dS_S = nl.sum(nisa.tensor_tensor(dSnt, St, nl.multiply), axis=1, keepdims=True)  # [DK,1]
    dS_S_row = nisa.tensor_copy(nisa.nc_transpose(dS_S))                   # [1,DK]
    d_egl = nl.sum(dS_S_row, axis=1, keepdims=True)                       # [1,1]
    dgl = nisa.tensor_tensor(sum_dglmgc, nisa.tensor_tensor(d_egl, egl, nl.multiply), nl.add)  # [1,1]
    # add dgl to dgc[-1] via the last-position one-hot mask
    dgl_col = nl.broadcast_to(dgl, shape=(C, 1))
    dgc = nisa.tensor_tensor(dgc, nisa.tensor_tensor(dgl_col, nl.load(last_mask), nl.multiply), nl.add)
    # gc = cumsum(g) -> dg = reverse-cumsum(dgc): flip(row) -> scan -> flip
    dgc_row = nisa.tensor_copy(nisa.nc_transpose(dgc))     # [1,C]
    dgc_flip = _mm(dgc_row, J)                             # reverse along free
    dg_scan = nisa.tensor_tensor_scan(ones_1c, dgc_flip, zero_11, nl.multiply, nl.add)
    dg_row = _mm(dg_scan, J)                               # flip back [1,C]
    dg = nisa.tensor_copy(nisa.nc_transpose(dg_row))       # [C,1]

    # force SBUF for all outputs (dma_copy cannot read PSUM)
    nisa.dma_copy(dst=dq_out[:, :], src=nisa.tensor_copy(dq))
    nisa.dma_copy(dst=dk_out[:, :], src=nisa.tensor_copy(dk))
    nisa.dma_copy(dst=dv_out[:, :], src=nisa.tensor_copy(dv))
    nisa.dma_copy(dst=dbeta_out[:, :], src=nisa.tensor_copy(dbeta))
    nisa.dma_copy(dst=dg_out[:, :], src=nisa.tensor_copy(dg))
    nisa.dma_copy(dst=dSin_out[:, :], src=nisa.tensor_copy(dS_in))
    return dq_out, dk_out, dv_out, dg_out, dbeta_out, dSin_out
