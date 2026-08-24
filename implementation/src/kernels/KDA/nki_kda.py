"""NKI kernels for KDA (Kernel-based Decomposed Attention) recurrent forward.

Model-agnostic NKI implementation of KDA linear attention. Targets the KDA
layer described in the flash-linear-attention (fla-core) library and used by
any KDA-based HuggingFace model.

References:
  - Algorithm: https://github.com/fla-org/flash-linear-attention (KDA)

KEY ALGORITHMIC DETAIL (vs. Gated DeltaNet):
  Gated DeltaNet: g is per-head SCALAR -> state decay is uniform across all state rows
                  g_t shape: (P_MAX, 1) broadcast to (P_MAX, 128) via tensor_scalar

  KDA: g is per-K-dimension VECTOR -> each ROW (per-K) of state decays independently
                  g_t shape: (P_MAX, 1) with per-K values
                  state[k, v] *= exp(g_t[k]) (fla-canonical per-K decay)

  This is the ONLY algorithmic change in the kernel body vs. Gated DeltaNet.
  Everything else (delta rule, outer product, matmul for output) is identical.
  The state is decayed per-K row (state[k, v] *= exp(g[k])), matching the
  fla-core canonical KDA convention.

Input layout: All inputs are 2D contiguous tensors of shape ``(S, HEAD_DIM)`` where
``HEAD_DIM == P_MAX == 128``. Each call processes one ``(batch, head)`` element's
full sequence.

For KDA: g is (S, 128) with per-channel (per-K) decay values.
         beta is (S, 128) with the SAME value per row (per-token scalar, broadcast).

Requires NKI >= 0.3.0.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Partition dimension max (NeuronCore SBUF tile width)
P_MAX = 128


@nki.jit
def kda_recurrent_fwd(
    query: nl.ndarray, # (S, 128) float32 -- L2-normed, scaled
    key: nl.ndarray, # (S, 128) float32 -- L2-normed
    value: nl.ndarray, # (S, 128) float32
    g_in: nl.ndarray, # (S, 128) float32 -- PER-DIMENSION log-decay (KDA)
    beta_in: nl.ndarray, # (S, 128) float32 -- write gate (same value per row)
) -> nl.ndarray:
    """NKI kernel for KDA recurrent forward -- single (batch, head).

    Iterates over sequence tokens with sequential_range.
    State matrix (128 x 128) lives in SBUF.

    KDA difference: g_in is (S, 128) with per-dimension values.
    State decay is element-wise: state[i,j] *= exp(g[j]) for each column j.

    Args:
        query: (S, 128) float32 -- already L2-normed and scaled by 1/sqrt(dk)
        key: (S, 128) float32 -- already L2-normed
        value: (S, 128) float32
        g_in: (S, 128) float32 -- per-dimension gate (log-space)
        beta_in: (S, 128) float32 -- write gate broadcast across dim

    Returns:
        output: (S, 128) float32
    """
    seq_len, dim = query.shape

    # Output tensor in HBM
    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)

    seq_stride = dim

    # Initialize recurrent state in SBUF: (128, 128)
    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    # Sequential loop over tokens
    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        # ---- Load inputs for token t ----
        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=q_t,
            src=query.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_t,
            src=key.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_t,
            src=value.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        g_col = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=g_col,
            src=g_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_t,
            src=beta_in.ap(pattern=[[1, P_MAX]], offset=tok_offset),
        )

        # ---- key-fold decode body (fused decode body) ----
        # Prior body decayed the FULL [128,128] state in place, then read kv_mem
        # off the decayed state, then built the outer product via a broadcast
        # (nc_transpose + nc_matmul(ones) + tensor_scalar) and a separate accumulate.
        #
        # Reformulation (see the design notes; kda_decode_nki.py lines 40-43): a diagonal (per-channel) gate does NOT
        # commute out of the contraction -- GDN's "decay the tiny result" trick is
        # invalid. But it DOES fold into the [dk,1] key:
        # kv_mem[v] = sum_k k[k]*exp_g[k]*S[k,v] = (k * exp_g) @ S_undecayed
        # so the kv_mem read no longer depends on a full-width state decay pass.
        # The persisted state still decays, but that fuses with the additive update
        # into ONE scalar_tensor_tensor Vector pass:
        # S_new[k,v] = exp_g[k]*S[k,v] + outer(k*beta, diff)[k,v]
        # and the outer product is a contraction-1 nc_matmul on the otherwise-idle PE.
        exp_g_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g_col, op=nl.exp, data=g_col, bias=None, scale=1.0)

        # k_dec = k * exp_g ([dk,1]); fold the per-channel decay into the key.
        k_dec = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_dec, data1=k_t, data2=exp_g_col, op=nl.multiply)

        # kv_mem[1,v] = k_dec.T @ S (contract partition dim k; state stays undecayed)
        kv_mem_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=k_dec, moving=state)
        kv_mem = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        # diff[1,v] = v_t^T - kv_mem. v_t is [dk,1]; we need v as a [1,dv] row.
        # (dk == dv == 128 here.) Transpose v_t to a row once.
        v_row_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=v_row_psum, data=v_t)
        v_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_row, src=v_row_psum)
        diff = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_row, data2=kv_mem, op=nl.subtract)

        # k*beta as a [1,dk] row (stationary for the rank-1 outer product).
        kbeta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=kbeta, data1=k_t, data2=beta_t, op=nl.multiply)
        kbeta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kbeta_row_psum, data=kbeta)
        kbeta_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kbeta_row, src=kbeta_row_psum)

        # outer[k,v] = kbeta[k] * diff[v] = nc_matmul(stationary=kbeta_row(1,dk),
        # moving=diff(1,dv)) = kbeta_row.T @ diff = (dk, dv), on the idle PE.
        outer_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_psum, stationary=kbeta_row, moving=diff)

        # Fused decay + accumulate into a FRESH buffer (never in-place on the
        # recurrence): S_new = exp_g * S + outer. outer_psum read straight from PSUM
        # as operand1 (legal: `data` S is in SBUF).
        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=state_new,
            data=state,
            op0=nl.multiply,
            operand0=exp_g_col,
            op1=nl.add,
            operand1=outer_psum,
        )
        nisa.tensor_copy(dst=state, src=state_new)

        # ---- o_t = state^T @ q_t ----
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        # ---- Store output for token t ----
        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=o_t,
        )

    return output


@nki.jit
def kda_recurrent_fwd_state(
    query: nl.ndarray, # (S, 128) float32
    key: nl.ndarray, # (S, 128) float32
    value: nl.ndarray, # (S, 128) float32
    g_in: nl.ndarray, # (S, 128) float32 -- per-dimension gate
    beta_in: nl.ndarray, # (S, 128) float32 -- write gate
):
    """KDA recurrent forward with final state output for CTE->TKG carry-over.

    Same as kda_recurrent_fwd but also returns the final recurrent state (128, 128).

    Returns:
        output: (S, 128) float32 -- per-token output
        final_state: (128, 128) float32 -- recurrent state after last token
    """
    seq_len, dim = query.shape

    output = nl.ndarray((seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    seq_stride = dim

    state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=state, value=0.0)

    for t in nl.sequential_range(seq_len):
        tok_offset = t * seq_stride

        q_t = nl.ndarray((P_MAX, 1), dtype=query.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=query.ap(pattern=[[1, P_MAX]], offset=tok_offset))

        k_t = nl.ndarray((P_MAX, 1), dtype=key.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=key.ap(pattern=[[1, P_MAX]], offset=tok_offset))

        v_t = nl.ndarray((P_MAX, 1), dtype=value.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=value.ap(pattern=[[1, P_MAX]], offset=tok_offset))

        g_col = nl.ndarray((P_MAX, 1), dtype=g_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_col, src=g_in.ap(pattern=[[1, P_MAX]], offset=tok_offset))

        beta_t = nl.ndarray((P_MAX, 1), dtype=beta_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=beta_t, src=beta_in.ap(pattern=[[1, P_MAX]], offset=tok_offset)
        )

        # ---- key-fold decode body (see kda_recurrent_fwd) ----
        exp_g_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g_col, op=nl.exp, data=g_col, bias=None, scale=1.0)

        k_dec = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_dec, data1=k_t, data2=exp_g_col, op=nl.multiply)

        kv_mem_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=k_dec, moving=state)
        kv_mem = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kv_mem, src=kv_mem_psum)

        v_row_psum = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=v_row_psum, data=v_t)
        v_row = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=v_row, src=v_row_psum)
        diff = nl.ndarray((1, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=diff, data1=v_row, data2=kv_mem, op=nl.subtract)

        kbeta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=kbeta, data1=k_t, data2=beta_t, op=nl.multiply)
        kbeta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=kbeta_row_psum, data=kbeta)
        kbeta_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kbeta_row, src=kbeta_row_psum)

        outer_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=outer_psum, stationary=kbeta_row, moving=diff)

        state_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=state_new,
            data=state,
            op0=nl.multiply,
            operand0=exp_g_col,
            op1=nl.add,
            operand1=outer_psum,
        )
        nisa.tensor_copy(dst=state, src=state_new)

        # ---- o_t = state^T @ q_t ----
        o_t_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_t_psum, stationary=state, moving=q_t)
        o_t = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_t, src=o_t_psum)

        # ---- Store output ----
        nisa.dma_copy(
            dst=output.ap(pattern=[[1, dim]], offset=tok_offset),
            src=o_t,
        )

    # Write final state to HBM
    nisa.dma_copy(dst=final_state, src=state)

    return output, final_state
