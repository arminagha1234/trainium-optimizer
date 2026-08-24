# DEVICE-VERIFIED attn_decode kernel — self-improve loop, trn2, 2026-08-24.
# Raced CORRECT at 0.438x vs torch-eager SDPA (single-query decode, S=512, hd=128).
# Harvested worked-example referenced by nki_knowledge.py _EX_ATTENTION note.

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def attn_decode_kernel(q, k, v):
    """Single-query (decode) attention: out = softmax(q @ k^T / sqrt(d)) @ v.

    Fully fused: one load per operand tile, softmax entirely in SBUF/PSUM,
    scores laid out as [128, S/128] so every reduction is a cheap free-axis
    reduce (cross-partition combines ride on the otherwise-idle PE).
    """
    D = 128
    scale = 1.0 / (D ** 0.5)

    # ---------------- q -> stationary [D, 1] (bf16, SBUF) ----------------
    qshape = q.shape
    qnd = len(qshape)
    q_is_row = (qshape[-2] == 1)
    if qnd == 2:
        qtile = nl.load(q[:, :])
    elif qnd == 3:
        qtile = nl.load(q[0, :, :])
    else:
        qtile = nl.load(q[0, 0, :, :])
    if q_is_row:
        qs = nl.copy(nisa.nc_transpose(data=qtile), dtype=nl.bfloat16)   # [D,1]
    else:
        qs = nl.copy(qtile, dtype=nl.bfloat16)                           # [D,1]

    # ---------------- layout / sizes ----------------
    knd = len(k.shape)
    if k.shape[-1] == 128:
        k_seq_first = True
        S = k.shape[-2]
    else:
        k_seq_first = False
        S = k.shape[-1]

    vnd = len(v.shape)
    if v.shape[-1] == 128 and v.shape[-2] == S:
        v_seq_first = True
    elif v.shape[-1] == S:
        v_seq_first = False
    else:
        v_seq_first = True

    nchunks = S // 128

    # loop-invariant constants, hoisted OUT of the tile loops
    ones_row = nl.full((1, 128), 1.0, dtype=nl.bfloat16)     # [K=1, M=128]
    ones_col = nl.full((128, 1), 1.0, dtype=nl.bfloat16)     # [K=128, M=1]
    negs = nl.full((128, 1), -scale, dtype=nl.float32)

    # ---------------- pass 1: scores[s] = q . k[s]  (S on partition) ------
    scores = nl.zeros((128, nchunks), dtype=nl.float32, buffer=nl.sbuf)
    for j in nl.affine_range(nchunks):
        if k_seq_first:
            if knd == 2:
                kt = nl.load(k[j * 128:(j + 1) * 128, :])
            elif knd == 3:
                kt = nl.load(k[0, j * 128:(j + 1) * 128, :])
            else:
                kt = nl.load(k[0, 0, j * 128:(j + 1) * 128, :])
            kst = nl.copy(nisa.nc_transpose(data=kt), dtype=nl.bfloat16)   # [D,128]
        else:
            if knd == 2:
                kst = nl.load(k[:, j * 128:(j + 1) * 128])
            elif knd == 3:
                kst = nl.load(k[0, :, j * 128:(j + 1) * 128])
            else:
                kst = nl.load(k[0, 0, :, j * 128:(j + 1) * 128])
        ps = nisa.nc_matmul(kst, qs)                                        # [128,1]
        scores[:, j:j + 1] = nl.copy(ps, dtype=nl.float32)

    # ---------------- softmax (global max + delayed division) -------------
    rmax = nl.max(scores, axis=1, keepdims=True)                            # [128,1]
    rmax_b = nl.copy(rmax, dtype=nl.bfloat16)
    rrow = nl.copy(nisa.nc_transpose(data=rmax_b), dtype=nl.bfloat16)       # [1,128]
    gmax = nl.max(rrow, axis=1, keepdims=True)                              # [1,1]
    bmax = nisa.nc_matmul(ones_row, gmax)                                   # [128,1] bcast
    bias = nl.multiply(bmax, negs)                                          # -scale*gmax

    # exp(scale*scores - scale*gmax) in ONE Scalar-engine instruction
    e = nisa.activation(nl.exp, scores, bias=bias, scale=scale,
                        dtype=nl.bfloat16)                                  # [128,nchunks]

    pcol = nl.sum(e, axis=1, dtype=nl.float32, keepdims=True)               # [128,1]
    pcol_b = nl.copy(pcol, dtype=nl.bfloat16)
    den = nisa.nc_matmul(ones_col, pcol_b)                                  # [1,1]
    inv = nl.reciprocal(nl.copy(den, dtype=nl.float32))                     # [1,1]

    # ---------------- pass 2: PV (contract S on partition) ---------------
    acc = nl.zeros((128, nchunks), dtype=nl.float32, buffer=nl.sbuf)
    for j in nl.affine_range(nchunks):
        if v_seq_first:
            if vnd == 2:
                vt = nl.load(v[j * 128:(j + 1) * 128, :])
            elif vnd == 3:
                vt = nl.load(v[0, j * 128:(j + 1) * 128, :])
            else:
                vt = nl.load(v[0, 0, j * 128:(j + 1) * 128, :])
            vst = nl.copy(vt, dtype=nl.bfloat16)                            # [S128,D]
        else:
            if vnd == 2:
                vtmp = nl.load(v[:, j * 128:(j + 1) * 128])
            elif vnd == 3:
                vtmp = nl.load(v[0, :, j * 128:(j + 1) * 128])
            else:
                vtmp = nl.load(v[0, 0, :, j * 128:(j + 1) * 128])
            vst = nl.copy(nisa.nc_transpose(data=vtmp), dtype=nl.bfloat16)  # [S128,D]
        pv = nisa.nc_matmul(vst, e[:, j:j + 1])                             # [D,1]
        acc[:, j:j + 1] = nl.copy(pv, dtype=nl.float32)

    out_t = nl.sum(acc, axis=1, keepdims=True, dtype=nl.float32)            # [D,1]

    out = nl.ndarray(qshape, dtype=q.dtype, buffer=nl.shared_hbm)
    if q_is_row:
        ot_b = nl.copy(out_t, dtype=nl.bfloat16)
        orow = nl.copy(nisa.nc_transpose(data=ot_b), dtype=nl.float32)      # [1,D]
        res = nl.multiply(orow, nl.broadcast_to(inv, shape=(1, 128)))
    else:
        inv_b = nl.copy(inv, dtype=nl.bfloat16)
        binv = nisa.nc_matmul(ones_row, inv_b)                              # [D,1]
        res = nl.multiply(out_t, binv)

    if qnd == 2:
        nl.store(out[:, :], value=res)
    elif qnd == 3:
        nl.store(out[0, :, :], value=res)
    else:
        nl.store(out[0, 0, :, :], value=res)
    return out