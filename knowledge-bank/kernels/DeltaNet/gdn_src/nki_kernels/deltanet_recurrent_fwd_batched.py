"""NKI kernel: BATCHED DeltaNet gated-delta-rule prefill (all B*H groups in one launch).

Batched counterpart of `deltanet_recurrent_fwd`: instead of grid-launching the
single-(batch, head) prefill kernel B*H times, this processes ALL G = B*H
independent (sequence, state) groups inside ONE kernel launch, looping groups on
`affine_range` and running the SAME optimized per-token key-fold body
(F1/F2/F3 + v3.2 key-fold) for each group with its own SBUF state.

WHY (and the honest performance caveat):
  DeltaNet prefill is a SEQUENTIAL recurrent scan (`sequential_range` over S). The
  scan is latency-bound, not throughput-bound: the per-token compute is a chain of
  small matmuls that cannot overlap across timesteps, and on a single core the
  outer groups also serialize on the Tensor/Vector engines. Batching G groups into
  one launch therefore amortizes only the PER-LAUNCH overhead (kernel dispatch,
  the 5 input DMAs' fixed cost) -- it does NOT reduce the O(G * S) serial compute.
  Measured benefit is a few percent, NOT the multiplicative speedup seen for the
  batched DECODE kernel (`deltanet_tkg_batched_bh`, 3.96x@B=4 ... 31.8x@B=32).
  Decode batches a single-token [B*H, ...] matmul (throughput-bound); prefill does
  not. Use this kernel to cut launch count / simplify the graph across a serving
  batch, not to speed up the per-token recurrence.

Output/state bit-identical to `deltanet_recurrent_fwd` run once per group
(verified via nki.simulate: out_maxabs = state_maxabs = 0.0).

Layout matches deltanet_recurrent_fwd, with a leading group axis:
  query, key, value, g_in, beta_in: (G, S, 128)  where G = B*H
  output: (G, S, 128)
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ..constants import P_MAX


@nki.jit
def deltanet_recurrent_fwd_batched(query, key, value, g_in, beta_in):
    """Batched DeltaNet recurrence over G=B*H groups, fresh (zero) initial state.

    Args (HBM):
      query, key, value: (G, S, 128)
      g_in, beta_in:     (G, S, 128)  - per-head scalar gate/beta broadcast to dim=128
    Returns:
      output: (G, S, 128)
    """
    G, seq_len, dim = query.shape
    output = nl.ndarray((G, seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)

    # Independent per (batch, head) group. affine_range: groups are disjoint
    # (each has its own state + SBUF tiles); the INNER token loop stays sequential.
    for gi in nl.affine_range(G):
        # Opt #1 (v3.1): bulk preload of all input vectors (feature on partition dim).
        q_all = nl.ndarray((P_MAX, seq_len), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_all, src=query[gi].ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
        k_all = nl.ndarray((P_MAX, seq_len), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_all, src=key[gi].ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
        v_all = nl.ndarray((P_MAX, seq_len), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_all, src=value[gi].ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
        g_all = nl.ndarray((P_MAX, seq_len), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_all, src=g_in[gi].ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))
        beta_all = nl.ndarray((P_MAX, seq_len), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=beta_all, src=beta_in[gi].ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0))

        o_all = nl.ndarray((P_MAX, seq_len), dtype=nl.float32, buffer=nl.sbuf)

        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=state, value=0.0)

        for t in nl.sequential_range(seq_len):
            q_t = q_all[0:P_MAX, t:t+1]
            k_t = k_all[0:P_MAX, t:t+1]
            v_t = v_all[0:P_MAX, t:t+1]
            g_t = g_all[0:P_MAX, t:t+1]
            beta_t = beta_all[0:P_MAX, t:t+1]

            exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_g, op=nl.exp, data=g_t, bias=None, scale=1.0)

            # k_dec = k * exp_g -- fold decay into the key.
            k_dec = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=k_dec, data1=k_t, data2=exp_g, op=nl.multiply)

            # kv_mem[1,v] = k_dec^T @ S (contract partition dim; state undecayed).
            kv_mem_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_mem_psum, stationary=k_dec, moving=state)
            kv_mem = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

            # diff_beta[1,v] = beta * (v^T - kv_mem); beta per-head scalar.
            v_row_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=v_row_psum, data=v_t)
            v_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_row, src=v_row_psum)
            diff = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=diff, data1=v_row, data2=kv_mem, op=nl.subtract)
            diff_beta = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=diff_beta, data=diff, op0=nl.multiply,
                               operand0=beta_t[0:1, 0:1], engine=nisa.vector_engine)

            # k as a [1,dk] row (stationary for rank-1 outer product), UNSCALED.
            k_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=k_row_psum, data=k_t)
            k_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_row, src=k_row_psum)

            # outer[k,v] = k[k] * diff_beta[v] via contraction-1 nc_matmul.
            outer_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=outer_psum, stationary=k_row, moving=diff_beta)

            # Fused decay + accumulate: S_new = exp_g * S + outer.
            state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(
                dst=state_new, data=state, op0=nl.multiply, operand0=exp_g,
                op1=nl.add, operand1=outer_psum,
            )
            nisa.tensor_copy(dst=state, src=state_new)

            # o = state^T @ q -> o_all slice.
            o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
            nisa.tensor_copy(dst=o_all[0:P_MAX, t:t+1], src=o_t_psum)

        nisa.dma_copy(dst=output[gi].ap(pattern=[[1, P_MAX], [P_MAX, seq_len]], offset=0), src=o_all)

    return output
