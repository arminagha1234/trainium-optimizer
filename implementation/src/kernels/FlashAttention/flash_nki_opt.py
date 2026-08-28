"""OPTIMIZED streaming / online-softmax FLASH-ATTENTION NKI kernel (long context).

Same math as flash_nki.py (online softmax, never materializes [S,S]); adds:
  L1 CAUSAL BLOCK-SKIP: static-unrolled q-loop so the KV loop only visits blocks
     jb where jb*BK <= last row of the q-tile; affine_select mask applied ONLY to
     the diagonal (partial) block, not every block.
  L4 FUSED VECTOR OPS: (a) track the *negated* running max so exp bias needs no
     separate negate (tensor_reduce(negate=True) + tensor_tensor min); (b) fuse
     l = l*alpha + l_blk into one two-op nisa.tensor_scalar.
  L5 EXP-FROM-PSUM: reduce/exp read the QK result straight out of its PSUM bank,
     eliminating the per-block 128xBK nl.copy(qk_psum -> sbuf).

Layout: q,k,v are (d_head, seqlen); out is (seqlen, d_head). Unscaled scores.
"""
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

NEG_INF = -30000.0
POS_BIG = 30000.0     # init for negated running max (= -(-inf))


def _online_update(p_ps, mask_pred, nm, l, acc, v_t, kv_base, sub, d_head, P, dt):
    """Emit one online-softmax + PV block update, in place on nm/l/acc.
    p_ps      : (P, BK) QK scores in a PSUM bank
    mask_pred : an affine predicate tile for causal masking, or None
    nm        : (P,1) running NEGATED max ( = -max )
    l         : (P,1) running sum ; acc : (P,d_head) running PV accumulator
    """
    BK = p_ps.shape[1]

    if mask_pred is not None:
        qk = nl.ndarray((P, BK), dtype=nl.float32, buffer=nl.sbuf)
        qk[:, :] = nisa.affine_select(pred=mask_pred, on_true_tile=p_ps,
                                      on_false_value=NEG_INF)
        src = qk
    else:
        src = p_ps                                   # read straight from PSUM (L5)

    # negated block max, running negated max (L4a)
    nm_blk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nm_blk[:, :] = nisa.tensor_reduce(nl.maximum, src, axis=(1,), negate=True)
    nm_new = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nm_new[:, :] = nisa.tensor_tensor(nm, nm_blk, nl.minimum)   # -max_new = min(-a,-b)

    # p = exp(qk - max_new) = exp(qk + nm_new), rowsum -> l_blk (fused on ScalarE)
    p = nl.ndarray((P, BK), dtype=dt, buffer=nl.sbuf)
    l_blk = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    p[:, :] = nisa.activation(nl.exp, src, bias=nm_new,
                              reduce_op=nl.add, reduce_res=l_blk)

    # alpha = exp(m_old - m_new) = exp(nm_new - nm_old)
    m_diff = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    m_diff[:, :] = nisa.tensor_tensor(nm_new, nm, nl.subtract)
    alpha = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    alpha[:, :] = nisa.activation(nl.exp, m_diff)

    # l = l*alpha + l_blk   (single fused two-op tensor_scalar, L4b)
    l[:, :] = nisa.tensor_scalar(l, nl.multiply, alpha, op1=nl.add, operand1=l_blk)

    # PV: pv_block(P,d) = p(P,BK) @ v_block(BK,d) via 128-wide transpose+matmul
    pv_ps = nl.zeros((P, d_head), dtype=nl.float32, buffer=nl.psum)
    for s in nl.affine_range(sub):
        pt = nisa.nc_transpose(data=p[:, nl.ds(s * P, P)])
        pv_ps[:, :] += nisa.nc_matmul(pt, v_t[:, kv_base + s, :])

    # acc = acc*alpha + pv_block
    acc_s = nl.ndarray((P, d_head), dtype=nl.float32, buffer=nl.sbuf)
    acc_s[:, :] = nisa.tensor_scalar(acc, nl.multiply, alpha)
    acc[:, :] = nisa.tensor_tensor(acc_s, pv_ps, nl.add)

    # advance running (negated) max
    nm[:, :] = nl.copy(nm_new)


def _finalize(out, iq_off, l, acc, P, d_head):
    inv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    inv[:, :] = nisa.reciprocal(l)
    ao = nl.ndarray((P, d_head), dtype=nl.float32, buffer=nl.sbuf)
    ao[:, :] = nisa.tensor_scalar(acc, nl.multiply, inv)
    nl.store(out[nl.ds(iq_off, P), :], value=ao)


def _load_and_vt(q, k, v, dt, P):
    d_head, seqlen = q.shape
    q_sb = nl.load(q); k_sb = nl.load(k); v_sb = nl.load(v)
    n_k128 = seqlen // P
    v_t = nl.ndarray((P, n_k128, P), dtype=dt, buffer=nl.sbuf)
    for j in nl.affine_range(n_k128):
        vp = nisa.nc_transpose(data=v_sb[:, nl.ds(j * P, P)])
        v_t[:, j, :] = nl.copy(vp)
    return q_sb, k_sb, v_t


def _flash_nc(q, k, v, out, BK, downcast):
    """Non-causal: affine q-loop, sequential KV loop."""
    d_head, seqlen = q.shape
    P = nl.tile_size.pmax
    n_q = seqlen // P
    n_blk = seqlen // BK
    sub = BK // P
    dt = nl.bfloat16 if downcast else nl.float32
    q_sb, k_sb, v_t = _load_and_vt(q, k, v, dt, P)

    for iq in nl.affine_range(n_q):
        qtile = q_sb[:, nl.ds(iq * P, P)]
        nm = nl.full((P, 1), POS_BIG, dtype=nl.float32, buffer=nl.sbuf)
        l = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.zeros((P, d_head), dtype=nl.float32, buffer=nl.sbuf)
        for jb in nl.sequential_range(n_blk):
            p_ps = nisa.nc_matmul(qtile, k_sb[:, nl.ds(jb * BK, BK)])
            _online_update(p_ps, None, nm, l, acc, v_t, jb * sub, sub, d_head, P, dt)
        _finalize(out, iq * P, l, acc, P, d_head)


def _flash_causal(q, k, v, out, BK, downcast):
    """Causal: static-unrolled q-loop with block-skip; mask only diagonal block."""
    d_head, seqlen = q.shape
    P = nl.tile_size.pmax
    n_q = seqlen // P
    sub = BK // P
    dt = nl.bfloat16 if downcast else nl.float32
    q_sb, k_sb, v_t = _load_and_vt(q, k, v, dt, P)

    for iq in nl.static_range(n_q):
        qtile = q_sb[:, nl.ds(iq * P, P)]
        nm = nl.full((P, 1), POS_BIG, dtype=nl.float32, buffer=nl.sbuf)
        l = nl.zeros((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        acc = nl.zeros((P, d_head), dtype=nl.float32, buffer=nl.sbuf)
        last_row = iq * P + P - 1
        nblk = last_row // BK + 1                       # L1: skip above-diagonal
        for jb in nl.static_range(nblk):
            p_ps = nisa.nc_matmul(qtile, k_sb[:, nl.ds(jb * BK, BK)])
            need_mask = (jb * BK + BK - 1) > iq * P     # partial (diagonal) block
            if need_mask:
                i_p = nl.arange(P)[:, None]
                i_f = nl.arange(BK)[None, :]
                pred = (iq * P + i_p >= jb * BK + i_f)
                _online_update(p_ps, pred, nm, l, acc, v_t, jb * sub, sub, d_head, P, dt)
            else:
                _online_update(p_ps, None, nm, l, acc, v_t, jb * sub, sub, d_head, P, dt)
        _finalize(out, iq * P, l, acc, P, d_head)


# ---- entry points (OUTPUT as trailing tensor arg) ----
def flash_fwd(q, k, v, out):
    _flash_nc(q, k, v, out, BK=512, downcast=True)

def flash_fwd_fp32(q, k, v, out):
    _flash_nc(q, k, v, out, BK=512, downcast=False)

def flash_fwd_causal(q, k, v, out):
    _flash_causal(q, k, v, out, BK=512, downcast=True)


# ---- BATCHED multi-head (single dispatch, all B*H heads on-device) ----------
# The regime where flash BEATS the compiler: at batched-multi-head long context
# the dense [B,H,S,S] score matrix exceeds HBM and the compiler OOMs
# (NCC_EOOM001) — this streaming kernel never materializes [S,S], so it is the
# ONLY path. q,k,v are [BH, d_head, seqlen]; out is [BH, seqlen, d_head]; one
# nki_jit dispatch loops the heads (no per-head host round-trip — the fix that
# made flash competitive vs the ~0.7ms/head dispatch of looping single-head).
# Validated on-device 2026-08-28: cos=1.0 at BH=4/S=512; ran at BH=256/S=4096
# (~tie w/ dense) and at BH=128/S=8192 where dense OOM'd. NOTE the ``affine_range``
# batch loop UNROLLS — compile time grows with BH; a tiled/looped batch dim is a
# follow-up for very large BH.
def flash_fwd_batched(q, k, v, out):
    BH = q.shape[0]
    for bh in nl.affine_range(BH):
        _flash_nc(q[bh], k[bh], v[bh], out[bh], BK=512, downcast=True)
