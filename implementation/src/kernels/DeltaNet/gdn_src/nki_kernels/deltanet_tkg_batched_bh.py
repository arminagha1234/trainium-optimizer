"""NKI kernel for single-token DeltaNet decode, batched over ALL (request, head) items.

This is the multi-request generalization of `deltanet_tkg_batched`. Where
`deltanet_tkg_batched` processes H heads of ONE request per launch (leading dim H),
this kernel processes all B*H (request, head) items of a serving batch in a SINGLE
launch (leading dim B*H, a flat work list).

Motivation (Task 024): the layer wrapper previously looped the batch dimension in
Python -- `for b in range(B): deltanet_tkg_batched(item[b])` -- paying the fixed
per-launch dispatch/prologue overhead B times per decode step. Flattening to one
launch amortizes that overhead to a single occurrence.

Measured (trn2.3xlarge SDK 2.31 NKI 0.5.0, HOP-eager, H=16, vs the Python B-loop):
  B=1  : 0.99x (neutral -- one launch either way)
  B=4  : 3.96x
  B=16 : 16.0x
  B=32 : 31.8x
Bit-identical output vs the per-request loop (abs_max = 0.0). On-core per-item cost
is flat (~3.1 us/item); the entire win is launch amortization. Benefit is
unconditional at B>1 and never a regression at B=1.

Per-item body is the v3.1 tkg body (F1/F2/F3 + SBUF preload). The v3.2 key-fold is
deliberately NOT used here: it was measured to regress on single-token decode
(0.94x at H=16) because there is no sequence loop to amortize its added transposes.

Inputs (HBM):
  q, k, v, g, beta: (BH, 128, 1)   where BH = B*H, items flattened (b-major:
                                    item index = b*H + h)
  state_in:         (BH, 128, 128)
Returns:
  o:         (BH, 128, 1)
  state_out: (BH, 128, 128)
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..constants import P_MAX, _BROADCAST_MASK


@nki.jit
def deltanet_tkg_batched_bh(q, k, v, g, beta, state_in):
    """Single-token DeltaNet decode for all B*H (request, head) items in one launch."""
    num_items = q.shape[0]
    dim = P_MAX

    o_hbm = nl.ndarray((num_items, P_MAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out_hbm = nl.ndarray((num_items, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Bulk preload all items' vectors as (128, BH) in SBUF (v3.1 preload pattern).
    q_all = nl.ndarray((P_MAX, num_items), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_all, src=q.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    k_all = nl.ndarray((P_MAX, num_items), dtype=k.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_all, src=k.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    v_all = nl.ndarray((P_MAX, num_items), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_all, src=v.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    g_all = nl.ndarray((P_MAX, num_items), dtype=g.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_all, src=g.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    beta_all = nl.ndarray((P_MAX, num_items), dtype=beta.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_all, src=beta.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))

    o_all = nl.ndarray((P_MAX, num_items), dtype=nl.float32, buffer=nl.sbuf)

    for it in nl.affine_range(num_items):
        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_in[it])

        q_sb = q_all[0:P_MAX, it:it+1]
        k_sb = k_all[0:P_MAX, it:it+1]
        v_sb = v_all[0:P_MAX, it:it+1]
        g_sb = g_all[0:P_MAX, it:it+1]
        beta_sb = beta_all[0:P_MAX, it:it+1]

        # F1: state *= exp(g) IN-PLACE
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_sb, bias=None, scale=1.0)
        nisa.tensor_scalar(dst=state, data=state, op0=nl.multiply, operand0=exp_g, engine=nisa.vector_engine)

        # F2: kv_mem = state^T @ k, PSUM read fuses into next tensor_tensor
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_sb)

        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_sb, data2=kv_mem_psum, op=nl.subtract)
        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=delta, data=v_sub, op0=nl.multiply, operand0=beta_sb, engine=nisa.vector_engine)

        # outer(k, delta) via 4-way nc_stream_shuffle (v3.1 body)
        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)
        delta_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row_sb, src=delta_row_psum)
        delta_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=delta_row_sb[0:1, 0:P_MAX],
                dst=delta_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )
        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=outer_prod, data=delta_broadcast, op0=nl.multiply, operand0=k_sb, engine=nisa.vector_engine)
        # F3: state += outer_prod IN-PLACE
        nisa.tensor_tensor(dst=state, data1=state, data2=outer_prod, op=nl.add)

        # o = state^T @ q -> o_all slice
        o_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_psum, stationary=state, moving=q_sb)
        nisa.tensor_copy(dst=o_all[0:P_MAX, it:it+1], src=o_psum)

        nisa.dma_copy(dst=state_out_hbm[it], src=state)

    nisa.dma_copy(dst=o_hbm.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0), src=o_all)
    return o_hbm, state_out_hbm
