"""NKI kernel for DeltaNet recurrence with state I/O.

Same recurrence as `deltanet_recurrent_fwd` but reads the initial state from HBM
and writes the final state back to HBM. Used for cached-chunk decode
(prefill_seq_len > 1 with previous cache_params).

Same optimizations as `deltanet_recurrent_fwd`:
  F1/F2/F3 (Task 007), Opt #1 SBUF preload (Task 022, v3.1),
  key-fold body (Task 023, v3.2 -- moves the delta update off VE onto the idle PE).

Effect (measured, vs v3.1): 1.32x @ S=51, 1.41x @ S=256. Bit-identical output
AND final state (cos_sim = 1.0). See consolidation/results/task023_key_fold.md.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..constants import P_MAX


@nki.jit
def deltanet_recurrent_fwd_state(query, key, value, g_in, beta_in, state_in):
    """DeltaNet recurrence, state read from HBM and written back.

    Args (HBM):
      query, key, value, g_in, beta_in: (S, 128)
      state_in: (128, 128)

    Returns:
      output:    (S, 128)
      state_out: (128, 128)
    """
    seq_len, dim = query.shape
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Load initial state.
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=state, src=state_in)

    # Opt #1 (v3.1): bulk preload of all input vectors.
    q_all = nl.ndarray((P_MAX, seq_len), dtype=query.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_all, src=query.ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
    k_all = nl.ndarray((P_MAX, seq_len), dtype=key.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_all, src=key.ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
    v_all = nl.ndarray((P_MAX, seq_len), dtype=value.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_all, src=value.ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
    g_all = nl.ndarray((P_MAX, seq_len), dtype=g_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_all, src=g_in.ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
    beta_all = nl.ndarray((P_MAX, seq_len), dtype=beta_in.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_all, src=beta_in.ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))

    o_all = nl.ndarray((P_MAX, seq_len), dtype=nl.float32, buffer=nl.sbuf)

    for t in nl.sequential_range(seq_len):
        q_t = q_all[0:P_MAX, t:t+1]
        k_t = k_all[0:P_MAX, t:t+1]
        v_t = v_all[0:P_MAX, t:t+1]
        g_t = g_all[0:P_MAX, t:t+1]
        beta_t = beta_all[0:P_MAX, t:t+1]

        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

        # k_dec = k * exp_g
        k_dec = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_dec, data1=k_t, data2=exp_g, op=nl.multiply)

        # kv_mem[1,v] = k_dec^T @ S (undecayed)
        kv_mem_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=k_dec, moving=state)
        kv_mem = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        # diff_beta[1,v] = beta * (v^T - kv_mem)
        v_row_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=v_row_psum, data=v_t)
        v_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_row, src=v_row_psum)
        diff = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_row, data2=kv_mem, op=nl.subtract)
        diff_beta = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=diff_beta, data=diff, op0=nl.multiply,
                           operand0=beta_t[0:1, 0:1], engine=nisa.vector_engine)

        # k row (unscaled)
        k_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=k_row_psum, data=k_t)
        k_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_row, src=k_row_psum)

        # outer product on the PE
        outer_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_psum, stationary=k_row, moving=diff_beta)

        # fused decay + accumulate
        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=state_new, data=state, op0=nl.multiply, operand0=exp_g,
            op1=nl.add, operand1=outer_psum,
        )
        nisa.tensor_copy(dst=state, src=state_new)

        # o -> o_all slice
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        nisa.tensor_copy(dst=o_all[0:P_MAX, t:t+1], src=o_t_psum)

    nisa.dma_copy(dst=output.ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0), src=o_all)
    nisa.dma_copy(dst=state_out, src=state)
    return output, state_out
