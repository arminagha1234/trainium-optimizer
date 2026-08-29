"""NKI kernel for single-token DeltaNet update, batched over heads.

Processes all H heads in a single kernel launch via an inner nl.affine_range
loop. This is the decode/TKG counterpart to `deltanet_recurrent_fwd_state`:
one token per call, state read from and written back to HBM.

Batching all heads inside one kernel invocation (rather than a Python
`for bh in range(B*H)` loop calling a per-head kernel) reduces the number of
launches from ~288 per token (BS=1, H=16, layers=18) to 18 per token, one per
DeltaNet layer.

Optimizations:
  F1/F2/F3 (Task 007, v2.0-task007): in-place state ops + PSUM operand fusion,
    applied inside the affine_range loop.
  Opt #1 (Task 022, v3.1): SBUF preload of all H heads' input vectors as a
    single (128, H) SBUF buffer per operand, plus single strided DMA for output
    vector flush at exit. State is loaded/stored per-head (state is 128*128*4 =
    64 KB per head -- not descriptor-issue-bound; already efficient DMA).

    Effect at H=16 (measured on trn2.3xlarge SDK 2.31 NKI 0.5):
      - DMA count: 128 -> 38 (-70%)
      - Wall-clock: 115.5 us -> 49.8 us (**2.32x speedup**)
      - Bit-identical output (abs_max = 0.0 vs v3.0)

    Smaller (but still real) wins at lower H: 1.28x at H=4, 1.00x at H=1.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..constants import P_MAX, _BROADCAST_MASK


@nki.jit
def deltanet_tkg_batched(q, k, v, g, beta, state_in):
    """Single-token DeltaNet update for all H heads.

    Args (HBM):
      q, k, v, g, beta: (H, 128, 1)
      state_in:         (H, 128, 128)

    Returns:
      o:         (H, 128, 1)
      state_out: (H, 128, 128)
    """
    num_heads = q.shape[0]
    dim = P_MAX

    o_hbm = nl.ndarray((num_heads, P_MAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out_hbm = nl.ndarray((num_heads, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Opt #1: bulk preload of all H heads' vectors as (128, H) in SBUF.
    q_all = nl.ndarray((P_MAX, num_heads), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_all, src=q.ap(pattern=[[1, P_MAX], [P_MAX, num_heads]], offset=0))

    k_all = nl.ndarray((P_MAX, num_heads), dtype=k.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_all, src=k.ap(pattern=[[1, P_MAX], [P_MAX, num_heads]], offset=0))

    v_all = nl.ndarray((P_MAX, num_heads), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_all, src=v.ap(pattern=[[1, P_MAX], [P_MAX, num_heads]], offset=0))

    g_all = nl.ndarray((P_MAX, num_heads), dtype=g.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_all, src=g.ap(pattern=[[1, P_MAX], [P_MAX, num_heads]], offset=0))

    beta_all = nl.ndarray((P_MAX, num_heads), dtype=beta.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_all, src=beta.ap(pattern=[[1, P_MAX], [P_MAX, num_heads]], offset=0))

    o_all = nl.ndarray((P_MAX, num_heads), dtype=nl.float32, buffer=nl.sbuf)

    for h in nl.affine_range(num_heads):
        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_in[h])

        q_sb = q_all[0:P_MAX, h:h+1]
        k_sb = k_all[0:P_MAX, h:h+1]
        v_sb = v_all[0:P_MAX, h:h+1]
        g_sb = g_all[0:P_MAX, h:h+1]
        beta_sb = beta_all[0:P_MAX, h:h+1]

        # F1: state *= exp(g) IN-PLACE
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_sb, bias=None, scale=1.0)
        nisa.tensor_scalar(dst=state, data=state, op0=nl.multiply, operand0=exp_g, engine=nisa.vector_engine)

        # F2
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_sb)

        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_sb, data2=kv_mem_psum, op=nl.subtract)
        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=delta, data=v_sub, op0=nl.multiply, operand0=beta_sb, engine=nisa.vector_engine)

        # outer(k, delta) via 4-way nc_stream_shuffle
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
        # F3
        nisa.tensor_tensor(dst=state, data1=state, data2=outer_prod, op=nl.add)

        # o -> o_all slice
        o_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_psum, stationary=state, moving=q_sb)
        nisa.tensor_copy(dst=o_all[0:P_MAX, h:h+1], src=o_psum)

        # State store: per-head (already efficient at 64 KB per DMA).
        nisa.dma_copy(dst=state_out_hbm[h], src=state)

    # Single strided DMA flush of all H output vectors.
    nisa.dma_copy(dst=o_hbm.ap(pattern=[[1, P_MAX], [P_MAX, num_heads]], offset=0), src=o_all)

    return o_hbm, state_out_hbm
