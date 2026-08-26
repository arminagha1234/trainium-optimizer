"""gdn_bwd_full_nki.py — FULL multi-chunk GatedDeltaNet backward, NKI 0.6.0.

Everything inlined into two static_range loops (NKI can't return multi-tile state
across helper scopes): a forward re-pass storing the state entering each chunk,
then a REVERSE chunk loop carrying dS and running the validated per-chunk VJP.
Validated against gdn_bwd_explicit (verified exact). q,k,v [T,128], g,beta [T,1],
dO [T,128], S0 [128,128]; T%C==0. Only _mm/_tmm/_mmt (single-tile) helpers.
"""
from __future__ import annotations
import math
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

C = 64
DK = 128
DV = 128
LEVELS = int(math.log2(C)) - 1


def _mm(a, b):
    return nisa.tensor_copy(nl.matmul(a, b))


def _tmm(a, b):
    return nisa.tensor_copy(nl.matmul(a, b, transpose_x=True))


def _mmt(a, b):
    return nisa.tensor_copy(nl.matmul(a, nisa.tensor_copy(nisa.nc_transpose(b))))


@nki.jit
def gdn_bwd_full(q, k, v, g, beta, S0, dO, tril_incl, strict, eye, triu_incl, last_mask):
    T = q.shape[0]; n = T // C
    dq_out = nl.ndarray((T, DK), dtype=nl.float32, buffer=nl.shared_hbm)
    dk_out = nl.ndarray((T, DK), dtype=nl.float32, buffer=nl.shared_hbm)
    dv_out = nl.ndarray((T, DV), dtype=nl.float32, buffer=nl.shared_hbm)
    dg_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbeta_out = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dS0_out = nl.ndarray((DK, DV), dtype=nl.float32, buffer=nl.shared_hbm)
    trilm = nl.load(tril_incl); strictm = nl.load(strict); eyem = nl.load(eye)
    triu = nl.load(triu_incl); lastm = nl.load(last_mask)
    ones_1c = nisa.memset((1, C), 1.0, dtype=nl.float32)
    zero_11 = nisa.memset((1, 1), 0.0, dtype=nl.float32)

    # ---- forward re-pass: store S entering each chunk ----
    S_list = [nl.load(S0)]
    for i in nl.static_range(n):
        cs = i * C
        kt = nl.load(k[cs:cs + C, :]); vt = nl.load(v[cs:cs + C, :])
        gcol = nl.load(g[cs:cs + C, :]); bcol = nl.load(beta[cs:cs + C, :])
        grow = nisa.tensor_copy(nisa.nc_transpose(gcol))
        gcrow = nisa.tensor_tensor_scan(ones_1c, grow, zero_11, nl.multiply, nl.add)
        gccol = nisa.tensor_copy(nisa.nc_transpose(gcrow))
        egc = nisa.activation(nl.exp, gccol)
        gdiff = nisa.tensor_scalar(nisa.tensor_tensor(nl.broadcast_to(gccol, shape=(C, C)),
                                    nl.broadcast_to(gcrow, shape=(C, C)), nl.subtract), nl.minimum, 0.0)
        decay = nisa.tensor_tensor(nisa.activation(nl.exp, gdiff), trilm, nl.multiply)
        kb = nisa.tensor_scalar(kt, nl.multiply, bcol); vb = nisa.tensor_scalar(vt, nl.multiply, bcol)
        A = nisa.tensor_tensor(nisa.tensor_scalar(nisa.tensor_tensor(
            _mmt(kb, kt), decay, nl.multiply), nl.multiply, -1.0), strictm, nl.multiply)
        Lp = nisa.tensor_copy(A); Tm = nisa.tensor_tensor(eyem, A, nl.add)
        for _ in nl.static_range(LEVELS):
            Lp = _mm(Lp, Lp); Tm = _mm(Tm, nisa.tensor_tensor(eyem, Lp, nl.add))
        u = _mm(Tm, vb); w = _mm(Tm, nisa.tensor_scalar(kb, nl.multiply, egc))
        vn = nisa.tensor_tensor(u, _mm(w, S_list[i]), nl.subtract)
        gl_11 = nisa.tensor_copy(gcrow[0:1, C - 1:C]); egl = nisa.activation(nl.exp, gl_11)
        gmg = nisa.activation(nl.exp, nisa.tensor_tensor(nl.broadcast_to(gl_11, shape=(C, 1)), gccol, nl.subtract))
        S_dec = nisa.tensor_scalar(S_list[i], nl.multiply, nl.broadcast_to(egl, shape=(DK, 1)))
        S_list.append(nisa.tensor_tensor(S_dec, _tmm(nisa.tensor_scalar(kt, nl.multiply, gmg), vn), nl.add))

    # ---- reverse chunk loop ----
    dS = nisa.memset((DK, DV), 0.0, dtype=nl.float32)
    for jj in nl.static_range(n):
        i = n - 1 - jj
        cs = i * C
        qt = nl.load(q[cs:cs + C, :]); kt = nl.load(k[cs:cs + C, :]); vt = nl.load(v[cs:cs + C, :])
        gcol = nl.load(g[cs:cs + C, :]); bcol = nl.load(beta[cs:cs + C, :]); dot = nl.load(dO[cs:cs + C, :])
        St = S_list[i]
        # forward recompute
        grow = nisa.tensor_copy(nisa.nc_transpose(gcol))
        gcrow = nisa.tensor_tensor_scan(ones_1c, grow, zero_11, nl.multiply, nl.add)
        gccol = nisa.tensor_copy(nisa.nc_transpose(gcrow)); egc = nisa.activation(nl.exp, gccol)
        gdiff = nisa.tensor_scalar(nisa.tensor_tensor(nl.broadcast_to(gccol, shape=(C, C)),
                                    nl.broadcast_to(gcrow, shape=(C, C)), nl.subtract), nl.minimum, 0.0)
        decay = nisa.tensor_tensor(nisa.activation(nl.exp, gdiff), trilm, nl.multiply)
        kb = nisa.tensor_scalar(kt, nl.multiply, bcol); vb = nisa.tensor_scalar(vt, nl.multiply, bcol)
        KKt = _mmt(kb, kt)
        A = nisa.tensor_tensor(nisa.tensor_scalar(nisa.tensor_tensor(KKt, decay, nl.multiply), nl.multiply, -1.0), strictm, nl.multiply)
        Lp = nisa.tensor_copy(A); Tm = nisa.tensor_tensor(eyem, A, nl.add)
        Tprev_s = []; Lprev_s = []; Lj_s = []
        for _ in nl.static_range(LEVELS):
            Tprev_s.append(nisa.tensor_copy(Tm)); Lprev_s.append(nisa.tensor_copy(Lp))
            Lp = _mm(Lp, Lp); Lj_s.append(nisa.tensor_copy(Lp))
            Tm = _mm(Tm, nisa.tensor_tensor(eyem, Lp, nl.add))
        kge = nisa.tensor_scalar(kb, nl.multiply, egc)
        u = _mm(Tm, vb); w = _mm(Tm, kge)
        QKt = _mmt(qt, kt); ai = nisa.tensor_tensor(QKt, decay, nl.multiply)
        vn = nisa.tensor_tensor(u, _mm(w, St), nl.subtract)
        qeg = nisa.tensor_scalar(qt, nl.multiply, egc)
        gl_11 = nisa.tensor_copy(gcrow[0:1, C - 1:C]); egl = nisa.activation(nl.exp, gl_11)
        gmg = nisa.activation(nl.exp, nisa.tensor_tensor(nl.broadcast_to(gl_11, shape=(C, 1)), gccol, nl.subtract))
        kdec = nisa.tensor_scalar(kt, nl.multiply, gmg)
        # reverse
        dS_in = nisa.tensor_scalar(dS, nl.multiply, nl.broadcast_to(egl, shape=(DK, 1)))
        dkdec = _mmt(vn, dS); dvn = _mm(kdec, dS)
        dai = _mmt(dot, vn); dvn = nisa.tensor_tensor(dvn, _tmm(ai, dot), nl.add)
        dqeg = _mmt(dot, St); dS_in = nisa.tensor_tensor(dS_in, _tmm(qeg, dot), nl.add)
        du = dvn; dvp = nisa.tensor_scalar(dvn, nl.multiply, -1.0)
        dw = _mmt(dvp, St); dS_in = nisa.tensor_tensor(dS_in, _tmm(w, dvp), nl.add)
        dT = nisa.tensor_tensor(_mmt(dw, kge), _mmt(du, vb), nl.add)
        dkge = _tmm(Tm, dw); dvb = _tmm(Tm, du)
        dQKt = nisa.tensor_tensor(dai, decay, nl.multiply)
        d_decay = nisa.tensor_tensor(dai, QKt, nl.multiply)
        dq = _mm(dQKt, kt); dk = _tmm(dQKt, qt)
        dq = nisa.tensor_tensor(dq, nisa.tensor_scalar(dqeg, nl.multiply, egc), nl.add)
        d_egc = nl.sum(nisa.tensor_tensor(dqeg, qt, nl.multiply), axis=1, keepdims=True)
        dkb = nisa.tensor_scalar(dkge, nl.multiply, egc)
        d_egc = nisa.tensor_tensor(d_egc, nl.sum(nisa.tensor_tensor(dkge, kb, nl.multiply), axis=1, keepdims=True), nl.add)
        dv = nisa.tensor_scalar(dvb, nl.multiply, bcol)
        dbeta = nl.sum(nisa.tensor_tensor(dvb, vt, nl.multiply), axis=1, keepdims=True)
        dL_acc = nisa.memset((C, C), 0.0, dtype=nl.float32)
        for jl in nl.static_range(LEVELS):
            idx = LEVELS - 1 - jl
            Tp = Tprev_s[idx]; Lprev = Lprev_s[idx]; Lj = Lj_s[idx]
            dT_prev = _mmt(dT, nisa.tensor_tensor(eyem, Lj, nl.add))
            dLj = nisa.tensor_tensor(_tmm(Tp, dT), dL_acc, nl.add)
            dL_prev = nisa.tensor_tensor(_mmt(dLj, Lprev), _tmm(Lprev, dLj), nl.add)
            dT = dT_prev; dL_acc = dL_prev
        dA = nisa.tensor_tensor(dT, dL_acc, nl.add)
        dKKtdecay = nisa.tensor_scalar(nisa.tensor_tensor(dA, strictm, nl.multiply), nl.multiply, -1.0)
        dKKt = nisa.tensor_tensor(dKKtdecay, decay, nl.multiply)
        d_decay = nisa.tensor_tensor(d_decay, nisa.tensor_tensor(dKKtdecay, KKt, nl.multiply), nl.add)
        dkb = nisa.tensor_tensor(dkb, _mm(dKKt, kt), nl.add)
        dk = nisa.tensor_tensor(dk, _tmm(dKKt, kb), nl.add)
        dk = nisa.tensor_tensor(dk, nisa.tensor_scalar(dkb, nl.multiply, bcol), nl.add)
        dbeta = nisa.tensor_tensor(dbeta, nl.sum(nisa.tensor_tensor(dkb, kt, nl.multiply), axis=1, keepdims=True), nl.add)
        dk = nisa.tensor_tensor(dk, nisa.tensor_scalar(dkdec, nl.multiply, gmg), nl.add)
        d_gmg = nl.sum(nisa.tensor_tensor(dkdec, kt, nl.multiply), axis=1, keepdims=True)
        d_glmgc = nisa.tensor_tensor(d_gmg, gmg, nl.multiply)
        dgc = nisa.tensor_scalar(d_glmgc, nl.multiply, -1.0)
        dgc_diff = nisa.tensor_tensor(d_decay, decay, nl.multiply)
        dgc = nisa.tensor_tensor(dgc, nl.sum(dgc_diff, axis=1, keepdims=True), nl.add)
        dgc_diff_T = nisa.tensor_copy(nisa.nc_transpose(dgc_diff))
        dgc = nisa.tensor_tensor(dgc, nisa.tensor_scalar(nl.sum(dgc_diff_T, axis=1, keepdims=True), nl.multiply, -1.0), nl.add)
        dgc = nisa.tensor_tensor(dgc, nisa.tensor_tensor(d_egc, egc, nl.multiply), nl.add)
        dglmgc_row = nisa.tensor_copy(nisa.nc_transpose(d_glmgc))
        sum_dglmgc = nl.sum(dglmgc_row, axis=1, keepdims=True)
        dS_S = nl.sum(nisa.tensor_tensor(dS, St, nl.multiply), axis=1, keepdims=True)
        d_egl = nl.sum(nisa.tensor_copy(nisa.nc_transpose(dS_S)), axis=1, keepdims=True)
        dgl = nisa.tensor_tensor(sum_dglmgc, nisa.tensor_tensor(d_egl, egl, nl.multiply), nl.add)
        dgc = nisa.tensor_tensor(dgc, nisa.tensor_tensor(nl.broadcast_to(dgl, shape=(C, 1)), lastm, nl.multiply), nl.add)
        dg = _mm(triu, dgc)
        nisa.dma_copy(dst=dq_out[cs:cs + C, :], src=nisa.tensor_copy(dq))
        nisa.dma_copy(dst=dk_out[cs:cs + C, :], src=nisa.tensor_copy(dk))
        nisa.dma_copy(dst=dv_out[cs:cs + C, :], src=nisa.tensor_copy(dv))
        nisa.dma_copy(dst=dg_out[cs:cs + C, :], src=nisa.tensor_copy(dg))
        nisa.dma_copy(dst=dbeta_out[cs:cs + C, :], src=nisa.tensor_copy(dbeta))
        dS = dS_in
    nisa.dma_copy(dst=dS0_out[:, :], src=nisa.tensor_copy(dS))
    return dq_out, dk_out, dv_out, dg_out, dbeta_out, dS0_out
