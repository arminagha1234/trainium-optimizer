# =============================================================================
# gdn-backward project -- backward kernel (Task 009 in progress)
# =============================================================================
# Status: Task 008 skeleton + Task 009 intra-chunk gradients (dQ, dK, dV, dbeta partials)
# and full dS reverse recurrence.
#
# Task 010 still to do: dg log-decay reverse cumsum + wy_repr backward
#                      (Neumann-inverse gradient formula).
#
# Kernel API (matches design/context_tuple_api.md):
#   deltanet_fused_chunked_bwd(
#       query, key, value, g_in, beta_in,
#       initial_state, N_stack, w_stack,
#       dO, dS_final,
#       lower_mask, identity, lower_mask_diag,
#   ) -> (dq, dk, dv, dg_stub, dbeta, dinitial_state)
#
# The `dg_stub` output is still zero. Task 010 implements dg.
# `dbeta` gets contributions from dv_beta (implemented in Task 009 partially);
# Task 010 adds contributions from wy_repr backward.
# `dk` gets partial contributions from intra-QK and dS-state paths;
# Task 010 adds the wy_repr contribution.
#
# Per-chunk backward math (with all Task 009 pieces landed):
#   Reverse chunk loop c = NC-1 .. 0:
#     load q_c, k_c, v_c, g_c, beta_c, dO_c
#     compute gc, exp_gc, g_last, exp_g_last, exp_gl_minus_gc
#     compute k_beta = k * beta[:, None], v_beta = v * beta[:, None]
#     load N[c] and w[c] from HBM stacks
#     recompute u = N @ v_beta, v_prime = w @ S_c, v_new = u - v_prime
#       (S_c here is the FORWARD state at chunk c's entry; we recompute
#        this via a separate PASS -- see below)
#     compute dv_new = attn^T @ dO_c + k_last_decay @ dS      [Task 009]
#     compute d_attn = dO_c @ v_new^T                          [Task 009]
#     compute dQK = d_attn * exp_gc_diff * m_diag              [Task 009]
#     compute dq_intra = dQK @ k_c                             [Task 009]
#     compute dq_state = (dO_c @ S_c^T) * exp(gc)              [Task 009]
#     write dq = dq_intra + dq_state to HBM                    [Task 009]
#     compute dk_intra = dQK^T @ q_c                           [Task 009]
#     compute dk_state = (dS @ v_new^T) * exp(g_last - gc)     [Task 009]
#     write dk_partial = dk_intra + dk_state to HBM            [Task 009]
#     compute dv_beta = N[c]^T @ dv_new                        [Task 009]
#     dv = dv_beta * beta[:, None]                             [Task 009]
#     dbeta_partial (from v_beta side): sum(dv_beta * v, -1)   [Task 009]
#     Update dS in place:
#       dS = exp(g_last) * dS + (q*exp(gc))^T @ dO_c - w[c]^T @ dv_new  [Task 009]
#
# S_c (forward state at chunk c entry): we recompute the forward state trajectory
# in a SEPARATE forward pass at the beginning of the backward kernel, then walk it
# in reverse chunk order. For Task 009 we implement this as a "pre-pass" that
# builds a per-chunk saved S_stack in HBM. Task 007 chose to recompute rather
# than save state; this is where that recompute happens.
# =============================================================================

"""Fused chunked DeltaNet backward kernel.

Task 009 status:
  - Full dS reverse recurrence including intra-chunk contributions.
  - dQ = dQ_intra + dQ_state.
  - dK partial (intra + state paths; wy_repr contribution deferred to Task 010).
  - dV = dv_beta * beta.
  - dbeta partial (from v_beta side).
  - dg still zero (Task 010).

The kernel performs an in-kernel forward re-pass to reconstruct the state
trajectory S[c] at each chunk boundary, avoiding the need to save it from
the forward.
"""

import numpy as np

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
CHUNK_SIZE = 128
_BROADCAST_MASK = [0] * 32


def _make_lower_mask():
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=-1)


def _make_lower_mask_diag():
    return np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=np.float32), k=0)


def _make_identity():
    return np.eye(CHUNK_SIZE, dtype=np.float32)


@nki.jit
def deltanet_fused_chunked_bwd_batched(
    query: nl.ndarray,          # (S, 128) fp32 -- l2-normed + scaled
    key: nl.ndarray,            # (S, 128) fp32 -- l2-normed
    value: nl.ndarray,          # (S, 128) fp32
    g_in: nl.ndarray,           # (S, 1) fp32 -- raw log-decay
    beta_in: nl.ndarray,        # (S, 1) fp32 -- post-sigmoid write gate
    initial_state: nl.ndarray,  # (128, 128) fp32
    N_stack: nl.ndarray,        # (num_chunks, 128, 128) fp32 -- saved Neumann
    w_stack: nl.ndarray,        # (num_chunks, 128, 128) fp32 -- saved k_cumdecay
    dO: nl.ndarray,             # (S, 128) fp32
    dS_final: nl.ndarray,       # (128, 128) fp32
    lower_mask: nl.ndarray,     # (128, 128) fp32
    identity: nl.ndarray,       # (128, 128) fp32
    lower_mask_diag: nl.ndarray, # (128, 128) fp32
):
    """Backward kernel -- Tasks 008 + 009. dg is still zero (Task 010)."""

    BH = query.shape[0]
    seq_len = query.shape[1]
    dim = query.shape[2]  # 128
    num_chunks = seq_len // CHUNK_SIZE

    # -----------------------------------------------------------------
    # HBM outputs
    # -----------------------------------------------------------------
    dq_out = nl.ndarray((BH, seq_len, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    dk_out = nl.ndarray((BH, seq_len, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    dv_out = nl.ndarray((BH, seq_len, dim), dtype=nl.float32, buffer=nl.shared_hbm)
    dg_out = nl.ndarray((BH, seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbeta_out = nl.ndarray((BH, seq_len, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dinitial_state_out = nl.ndarray((BH, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # A saved forward state trajectory. State[c] = the state at chunk c's entry.
    # State[0] = initial_state; State[NC] = final_state (not stored).
    # We only need State[0..NC-1] for the backward.
    S_stack = nl.ndarray(
        (BH, num_chunks, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # -----------------------------------------------------------------
    # Zero-init dg (still a stub in Task 009).
    # -----------------------------------------------------------------
    # =================================================================
    # Outer batched loop over (batch, head) slices.
    # =================================================================
    for sample_idx in nl.sequential_range(BH):
        zero_chunk_s1 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero_chunk_s1, value=0.0)
        for i_chunk in nl.affine_range(num_chunks):
            chunk_start = i_chunk * CHUNK_SIZE
            nisa.dma_copy(
                dst=dg_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
                src=zero_chunk_s1,
            )

        # Constants loaded once
        Lmask = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=Lmask, src=lower_mask)
        Lmask_d = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=Lmask_d, src=lower_mask_diag)
        eye = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=eye, src=identity)

        # Ones vector for cumsum
        ones_1xC = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_1xC, value=1.0)
        zero_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero_11, value=0.0)

        # =================================================================
        # PASS 1 (forward re-pass): build S_stack by iterating chunks in order.
        # This is a full forward but ONLY computing v_new + state update (skipping
        # the intra-chunk output part). We use the saved N and w to skip the
        # Neumann and matmul recomputation for u and w.
        # =================================================================
        state_fwd = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state_fwd, src=initial_state[sample_idx, 0:P_MAX, 0:dim])

        for i_chunk in nl.sequential_range(num_chunks):
            chunk_start = i_chunk * CHUNK_SIZE

            # Save current state into S_stack[i_chunk] BEFORE updating
            nisa.dma_copy(
                dst=S_stack[sample_idx, i_chunk, 0:P_MAX, 0:dim], src=state_fwd
            )

            # Load k, v, g, beta for this chunk
            k_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_c, src=key[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim])
            v_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_c, src=value[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim])
            g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
                src=g_in[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
            )
            beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=beta_p[0:CHUNK_SIZE, 0:1],
                src=beta_in[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
            )

            # Compute gc = cumsum(g_chunk) via row-scan
            gc_row_psum = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=gc_row_psum, data=g_chunk_p[0:CHUNK_SIZE, 0:1])
            g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g_row, src=gc_row_psum)

            gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=gc_row, data0=ones_1xC, data1=g_row, initial=zero_11,
                op0=nl.multiply, op1=nl.add,
            )
            # Transpose gc back to (P_MAX, 1) column form
            gc_col_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=gc_col_psum, data=gc_row[0:1, 0:CHUNK_SIZE])
            gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gc_p, src=gc_col_psum)

            # g_last = gc_row[0, C-1]
            gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gl_11, src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE])

            # exp(g_last - gc)
            gl_broadcast = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=gl_11[0:1, 0:1],
                    dst=gl_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )
            gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gl_minus_gc, data1=gl_broadcast, data2=gc_p, op=nl.subtract
            )
            exp_gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_gl_minus_gc, op=nl.exp, data=gl_minus_gc, bias=None, scale=1.0
            )

            # exp(g_last)
            exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_gl_11, op=nl.exp, data=gl_11, bias=None, scale=1.0)
            exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=exp_gl_11[0:1, 0:1],
                    dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )

            # v_beta = v * beta
            v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=v_beta, data=v_c, op0=nl.multiply, operand0=beta_p,
                engine=nisa.vector_engine,
            )

            # Load saved N and w for this chunk
            N_c = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=N_c, src=N_stack[sample_idx, i_chunk, 0:P_MAX, 0:P_MAX])
            w_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_c, src=w_stack[sample_idx, i_chunk, 0:P_MAX, 0:dim])

            # u = N @ v_beta (N stored as SBUF, but nc_matmul needs stationary transposed).
            # In forward we did: N_T = transpose(N); u = N_T @ v_beta.  Actually the forward
            # kernel does nc_matmul(stationary=N_T, moving=v_beta) which computes N_T^T @ v_beta = N @ v_beta.
            # Wait -- let's confirm: nc_matmul(stationary=A, moving=B) computes A^T @ B.
            # In forward line 443: nc_matmul(dst=vc_psum, stationary=N_T, moving=v_beta) -> N_T^T @ v_beta = N @ v_beta.
            # So to reproduce here: transpose N to get N_T, then matmul.
            N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=N_T_psum, data=N_c)
            N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=N_T, src=N_T_psum)

            u_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=u_psum, stationary=N_T, moving=v_beta)
            u = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=u, src=u_psum)

            # v_prime = w @ state_fwd
            w_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=w_T_psum, data=w_c)
            w_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_T, src=w_T_psum)

            vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=vp_psum, stationary=w_T, moving=state_fwd)
            v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_prime, src=vp_psum)

            v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=v_new, data1=u, data2=v_prime, op=nl.subtract)

            # k_state_decay = k * exp(g_last - gc)
            k_state_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_state_decay, data=k_c, op0=nl.multiply, operand0=exp_gl_minus_gc,
                engine=nisa.vector_engine,
            )
            # kv_outer = k_state_decay^T @ v_new
            kv_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=kv_psum, stationary=k_state_decay, moving=v_new)
            # state_decayed = exp(g_last) * state
            state_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=state_decayed, data=state_fwd, op0=nl.multiply, operand0=exp_gl_p,
                engine=nisa.vector_engine,
            )
            nisa.tensor_tensor(
                dst=state_fwd, data1=state_decayed, data2=kv_psum, op=nl.add
            )

        # =================================================================
        # PASS 2: reverse loop with intra-chunk gradients and dS recurrence.
        # =================================================================
        dS = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dS, src=dS_final[sample_idx, 0:P_MAX, 0:dim])

        zero_chunk_sd = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero_chunk_sd, value=0.0)

        for i_chunk_rev in nl.sequential_range(num_chunks - 1, -1, -1):
            chunk_start = i_chunk_rev * CHUNK_SIZE

            # ---- Reload chunk inputs ----
            q_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_c, src=query[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim])
            k_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_c, src=key[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim])
            v_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_c, src=value[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim])
            g_chunk_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=g_chunk_p[0:CHUNK_SIZE, 0:1],
                src=g_in[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
            )
            beta_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=beta_p[0:CHUNK_SIZE, 0:1],
                src=beta_in[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
            )
            dO_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=dO_c, src=dO[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim])

            # ---- Recompute gc, exp_gc, g_last, exp_g_last, exp_gl_minus_gc ----
            gc_row_psum = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=gc_row_psum, data=g_chunk_p[0:CHUNK_SIZE, 0:1])
            g_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=g_row, src=gc_row_psum)

            gc_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=gc_row, data0=ones_1xC, data1=g_row, initial=zero_11,
                op0=nl.multiply, op1=nl.add,
            )
            gc_col_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=gc_col_psum, data=gc_row[0:1, 0:CHUNK_SIZE])
            gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gc_p, src=gc_col_psum)

            exp_gc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_gc_p, op=nl.exp, data=gc_p, bias=None, scale=1.0)

            gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gl_11, src=gc_row[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE])

            gl_broadcast = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=gl_11[0:1, 0:1],
                    dst=gl_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )
            gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gl_minus_gc, data1=gl_broadcast, data2=gc_p, op=nl.subtract
            )
            exp_gl_minus_gc = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_gl_minus_gc, op=nl.exp, data=gl_minus_gc, bias=None, scale=1.0
            )

            exp_gl_11 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_gl_11, op=nl.exp, data=gl_11, bias=None, scale=1.0)
            exp_gl_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=exp_gl_11[0:1, 0:1],
                    dst=exp_gl_p[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )

            # Build exp_gc_diff mask (same as forward)
            ones_PP = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_PP, value=1.0)
            gc_mat_rows = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=gc_mat_rows, data=ones_PP, op0=nl.multiply, operand0=gc_p,
                engine=nisa.vector_engine,
            )
            gc_mat_cols = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=gc_row[0:1, 0:CHUNK_SIZE],
                    dst=gc_mat_cols[i_shuf * 32 : i_shuf * 32 + 32, 0:CHUNK_SIZE],
                    shuffle_mask=_BROADCAST_MASK,
                )
            gc_diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=gc_diff, data1=gc_mat_rows, data2=gc_mat_cols, op=nl.subtract
            )
            gc_diff_clamped = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=gc_diff_clamped, data=gc_diff, op0=nl.minimum, operand0=0.0,
                engine=nisa.vector_engine,
            )
            exp_gc_diff = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=exp_gc_diff, op=nl.exp, data=gc_diff_clamped, bias=None, scale=1.0
            )

            # ---- Load saved N[c], w[c] and forward state S[c] ----
            N_c = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=N_c, src=N_stack[sample_idx, i_chunk_rev, 0:P_MAX, 0:P_MAX])
            w_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_c, src=w_stack[sample_idx, i_chunk_rev, 0:P_MAX, 0:dim])
            S_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=S_c, src=S_stack[sample_idx, i_chunk_rev, 0:P_MAX, 0:dim])

            # ---- Recompute v_beta, u = N @ v_beta, v_prime = w @ S_c, v_new = u - v_prime ----
            v_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=v_beta, data=v_c, op0=nl.multiply, operand0=beta_p,
                engine=nisa.vector_engine,
            )

            N_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=N_T_psum, data=N_c)
            N_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=N_T, src=N_T_psum)

            u_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=u_psum, stationary=N_T, moving=v_beta)
            u = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=u, src=u_psum)

            w_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=w_T_psum, data=w_c)
            w_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=w_T, src=w_T_psum)

            vp_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=vp_psum, stationary=w_T, moving=S_c)
            v_prime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_prime, src=vp_psum)

            v_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=v_new, data1=u, data2=v_prime, op=nl.subtract)

            # k_last_decay = k_c * exp(g_last - gc)
            k_last_decay = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_last_decay, data=k_c, op0=nl.multiply, operand0=exp_gl_minus_gc,
                engine=nisa.vector_engine,
            )

            # Recompute attn = (q_c @ k_c^T) * exp_gc_diff * m_diag
            q_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=q_T_psum, data=q_c)
            q_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=q_T, src=q_T_psum)
            k_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=k_T_psum, data=k_c)
            k_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_T, src=k_T_psum)

            qk_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=qk_psum, stationary=q_T, moving=k_T)
            # Task 016 opt: cache raw q @ k^T in SBUF for reuse by the dgc-intra-Lmask path
            # (originally recomputed at line 901). This eliminates one nc_matmul per chunk.
            qkT = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qkT, src=qk_psum)
            qk_decay = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=qk_decay, data1=qkT, data2=exp_gc_diff, op=nl.multiply)
            attn = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=attn, data1=qk_decay, data2=Lmask_d, op=nl.multiply)

            # =================================================================
            # BEGIN Task 009 GRADIENT COMPUTATIONS
            # =================================================================

            # ---- dv_new = attn^T @ dO_c + k_last_decay @ dS ----
            # Part A: attn^T @ dO_c
            # Trick: nc_matmul(stationary=attn, moving=dO_c) computes attn^T @ dO_c directly.
            dv_new_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dv_new_psum, stationary=attn, moving=dO_c)
            dv_new_from_intra = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dv_new_from_intra, src=dv_new_psum)

            # Part B: k_last_decay @ dS
            # nc_matmul(stationary=k_last_decay^T, moving=dS) = k_last_decay @ dS.
            # k_last_decay is (P_MAX, dim); need k_last_decay^T = (dim, P_MAX) as stationary.
            kld_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=kld_T_psum, data=k_last_decay)
            kld_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kld_T, src=kld_T_psum)

            dv_new_state_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dv_new_state_psum, stationary=kld_T, moving=dS)
            dv_new_from_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dv_new_from_state, src=dv_new_state_psum)

            dv_new = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dv_new, data1=dv_new_from_intra, data2=dv_new_from_state, op=nl.add
            )

            # ---- d_attn = dO_c @ v_new^T. Then dQK = d_attn * exp_gc_diff (masked below). ----
            # nc_matmul(stationary=dO_c^T, moving=v_new^T)? Simpler:
            # Compute via nc_matmul(stationary=v_new, moving=dO_c) = v_new^T @ dO_c = d_attn^T,
            # then transpose. Actually let's do d_attn directly:
            # d_attn[i,j] = dO[i] . v_new[j]. So d_attn = dO_c @ v_new^T.
            # nc_matmul(stationary=A, moving=B) = A^T @ B, so we want A^T = dO_c, B = v_new^T.
            # That means stationary = dO_c^T (dim, P_MAX), moving = v_new^T (dim, P_MAX).
            # But both dO_c and v_new are (P_MAX, dim). We can compute v_new^T @ dO_c = d_attn^T
            # via stationary=v_new (P_MAX, dim), moving=dO_c (P_MAX, dim), and matmul contracts
            # over partition -> result (dim, dim). Wait that's not right shape either.
            # Let me think again. We have d_attn shape (P_MAX, P_MAX). It's an outer product
            # over the D contraction dim: d_attn = dO_c @ v_new^T, so:
            #   d_attn[i,j] = sum_d dO_c[i,d] * v_new[j,d]
            # nc_matmul(stationary=A, moving=B) with A shape (K, M) and B shape (K, N) gives (M, N):
            #   result[m,n] = sum_k A[k,m] * B[k,n]
            # We want: sum_d dO_c[i,d] * v_new[j,d] with output [i,j].
            # If we transpose dO_c to (D, P_MAX): dO_c^T[d,i] * v_new^T[d,j] via stationary=dO_c^T,
            # moving=v_new^T -> result[i,j] = sum_d dO_c[i,d] * v_new[j,d].  YES.
            dO_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dO_T_psum, data=dO_c)
            dO_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dO_T, src=dO_T_psum)

            v_new_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=v_new_T_psum, data=v_new)
            v_new_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_new_T, src=v_new_T_psum)

            # Task 016 opt: fuse d_attn = dO_T^T @ v_new_T and dq_state_pre = dO_T^T @ S_c_T
            # into ONE nc_matmul with stationary=dO_T, moving=[v_new_T | S_c_T] (128x256).
            # We need S_c_T ready first, so move its transpose ahead.
            S_c_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=S_c_T_psum, data=S_c)
            S_c_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=S_c_T, src=S_c_T_psum)

            # Build the concatenated moving tile in SBUF: (P_MAX, 2*P_MAX)
            movings_dO_2x = nl.ndarray((P_MAX, 2 * P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=movings_dO_2x[0:P_MAX, 0:P_MAX], src=v_new_T)
            nisa.tensor_copy(dst=movings_dO_2x[0:P_MAX, P_MAX:2 * P_MAX], src=S_c_T)

            # Single fused matmul: produces [d_attn | dq_state_pre] in one PSUM tile.
            dattn_dqstate_psum = nl.ndarray((P_MAX, 2 * P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dattn_dqstate_psum, stationary=dO_T, moving=movings_dO_2x)

            # Split into d_attn (first half) and dq_state_pre (second half).
            d_attn = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=d_attn, src=dattn_dqstate_psum[0:P_MAX, 0:P_MAX])
            dq_state_pre = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dq_state_pre, src=dattn_dqstate_psum[0:P_MAX, P_MAX:2 * P_MAX])

            # dQK = d_attn * exp_gc_diff * m_diag (mask upper triangle)
            dQK_scaled = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dQK_scaled, data1=d_attn, data2=exp_gc_diff, op=nl.multiply
            )
            dQK = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dQK, data1=dQK_scaled, data2=Lmask_d, op=nl.multiply)

            # ---- dq_intra = dQK @ k_c ----
            # nc_matmul(stationary=dQK^T, moving=k_c) = dQK @ k_c.
            dQK_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dQK_T_psum, data=dQK)
            dQK_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dQK_T, src=dQK_T_psum)

            dq_intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dq_intra_psum, stationary=dQK_T, moving=k_c)
            dq_intra = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dq_intra, src=dq_intra_psum)

            # ---- dq_state = dq_state_pre * exp(gc) ----
            # (dq_state_pre was already computed above via the fused matmul)
            dq_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dq_state, data=dq_state_pre, op0=nl.multiply, operand0=exp_gc_p,
                engine=nisa.vector_engine,
            )

            # dq = dq_intra + dq_state
            dq_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dq_c, data1=dq_intra, data2=dq_state, op=nl.add)
            nisa.dma_copy(dst=dq_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim], src=dq_c)

            # ---- dk_intra = dQK^T @ q_c ----
            dk_intra_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dk_intra_psum, stationary=dQK, moving=q_c)
            dk_intra = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dk_intra, src=dk_intra_psum)

            # ---- dk_state = (v_new @ dS^T) * exp(g_last - gc) ----
            # dk_state[t, d_k] = sum_{d_v} v_new[t, d_v] * dS[d_k, d_v]
            # nc_matmul(stationary=A, moving=B) = A^T @ B, so with A^T = v_new, B = dS^T:
            # -> stationary = v_new^T (d_v, t), moving = dS^T (d_v, d_k). Result = (t, d_k).
            # (We already have v_new_T from d_attn computation.)
            dS_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dS_T_psum, data=dS)
            dS_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dS_T, src=dS_T_psum)

            dk_state_pre_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dk_state_pre_psum, stationary=v_new_T, moving=dS_T)
            dk_state_pre = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dk_state_pre, src=dk_state_pre_psum)

            # Scale by exp(g_last - gc)
            dk_state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dk_state, data=dk_state_pre, op0=nl.multiply, operand0=exp_gl_minus_gc,
                engine=nisa.vector_engine,
            )
            # Wait -- dk_state_pre was declared (P_MAX, P_MAX) above. Its actual content is (P_MAX, dim).
            # Since P_MAX == dim == 128 in this kernel, the shapes match, but let's be consistent
            # in future. For now the numeric result is correct because both dims are 128.

            # dk_partial = dk_intra + dk_state
            dk_partial = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dk_partial, data1=dk_intra, data2=dk_state, op=nl.add
            )
            nisa.dma_copy(dst=dk_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim], src=dk_partial)

            # ---- Task 016 opt: precompute dv_new_T, v_beta_T, and fused dw+dA_from_u
            # so we can fuse dv_beta+dkbs_from_w via stationary=N_c. ----
            dv_new_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dv_new_T_psum, data=dv_new)
            dv_new_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dv_new_T, src=dv_new_T_psum)

            # Compute v_beta_T early too so we can fuse dw_pos and dA_from_u
            # (both use stationary=dv_new_T).
            v_beta_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=v_beta_T_psum, data=v_beta)
            v_beta_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=v_beta_T, src=v_beta_T_psum)

            # Fused matmul: [dw_pos | dA_from_u] = dv_new_T^T @ [S_c_T | v_beta_T]
            # (S_c_T was computed earlier for the dq/d_attn fused matmul.)
            movings_dvnew_2x = nl.ndarray((P_MAX, 2 * P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=movings_dvnew_2x[0:P_MAX, 0:P_MAX], src=S_c_T)
            nisa.tensor_copy(dst=movings_dvnew_2x[0:P_MAX, P_MAX:2 * P_MAX], src=v_beta_T)

            dw_dAu_psum = nl.ndarray((P_MAX, 2 * P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dw_dAu_psum, stationary=dv_new_T, moving=movings_dvnew_2x)

            # dw_pos = first half (negated below); dA_from_u = second half.
            dw = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            # Apply negation via tensor_scalar with -1 on the PSUM slice.
            nisa.tensor_scalar(
                dst=dw, data=dw_dAu_psum[0:P_MAX, 0:dim], op0=nl.multiply, operand0=-1.0,
                engine=nisa.vector_engine,
            )
            dA_from_u = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dA_from_u, src=dw_dAu_psum[0:P_MAX, P_MAX:2 * P_MAX])

            # Fused matmul: [dv_beta | dkbs_from_w] = N_c^T @ [dv_new | dw]
            # Build concatenated moving in SBUF (P_MAX, 2*dim).
            movings_N_2x = nl.ndarray((P_MAX, 2 * dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=movings_N_2x[0:P_MAX, 0:dim], src=dv_new)
            nisa.tensor_copy(dst=movings_N_2x[0:P_MAX, dim:2 * dim], src=dw)

            dvb_dkbs_psum = nl.ndarray((P_MAX, 2 * dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dvb_dkbs_psum, stationary=N_c, moving=movings_N_2x)

            dv_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dv_beta, src=dvb_dkbs_psum[0:P_MAX, 0:dim])
            dkbs_from_w = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dkbs_from_w, src=dvb_dkbs_psum[0:P_MAX, dim:2 * dim])

            # ---- dv = dv_beta * beta (row-scale) ----
            dv_c = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dv_c, data=dv_beta, op0=nl.multiply, operand0=beta_p,
                engine=nisa.vector_engine,
            )
            nisa.dma_copy(dst=dv_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim], src=dv_c)

            # =================================================================
            # Task 010: Neumann-inverse gradient (wy_repr backward)
            # =================================================================
            # Forward flow (per Task 006 backward_math.md):
            #     v_new = u - w @ S,  u = N @ v_beta,  w = N @ (k_beta * exp(gc))
            # Backward:
            #     du = dv_new
            #     dw = -dv_new @ S_c^T
            #     dv_beta = N^T @ du  (already computed above)
            #     d(k_beta * exp(gc)) = N^T @ dw
            # And the Neumann-inverse gradient (FLA wy_fast.py:189-200):
            #     dA_tmp = du @ v_beta^T + dw @ (k_beta_scaled)^T   (both terms)
            #     dA_tmp masked to strict lower (i > j)
            #     dA = -N^T @ (dA_tmp @ N^T)                         (matrix-inverse chain rule)
            #     apply L_mask factor: dA *= exp_gc_diff  (which for i>j is exp(gc[i]-gc[j]))
            #     dA masked to strict lower with sign flip
            # Then A_raw = -tril(k_beta @ k^T), so:
            #     dB = -dA * (masked)  where B = k_beta @ k^T -- wait, A_raw = -B * L_mask,
            #     and dA above already has the L_mask factor absorbed. Actually in FLA the
            #     mask absorbtion + sign is handled by their -b_dA masking at line 200.
            # After sign flip, dA now equals what FLA calls "b_dA" for k-side accumulation.
            # dk_beta from A = dA @ k
            # dk from A = dA^T @ k_beta
            # dbeta from k_beta = sum(dk_beta * k) + sum(dv_beta * v)
            # =================================================================

            # -- dv_new_T and dw were computed earlier for the fused N_c matmul. --

            # -- Compute k_beta_scaled = k_beta * exp(gc) --
            k_beta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_beta, data=k_c, op0=nl.multiply, operand0=beta_p,
                engine=nisa.vector_engine,
            )
            k_beta_scaled = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=k_beta_scaled, data=k_beta, op0=nl.multiply, operand0=exp_gc_p,
                engine=nisa.vector_engine,
            )

            # -- Compute dA_tmp = du @ v_beta^T + dw @ k_beta_scaled^T --
            # du @ v_beta^T shape (P_MAX, P_MAX) where element [i,j] = sum_d(du[i,d] * v_beta[j,d]).
            # Task 016 opt: v_beta_T and dA_from_u were computed earlier via the fused
            # dv_new_T matmul (with dw_pos). Reuse them here.

            # dw @ k_beta_scaled^T shape (P_MAX, P_MAX): elem[i,j] = sum_d dw[i,d] * k_beta_scaled[j,d]
            dw_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dw_T_psum, data=dw)
            dw_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dw_T, src=dw_T_psum)

            kbs_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=kbs_T_psum, data=k_beta_scaled)
            kbs_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kbs_T, src=kbs_T_psum)

            dA_from_w_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dA_from_w_psum, stationary=dw_T, moving=kbs_T)
            dA_from_w = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dA_from_w, src=dA_from_w_psum)

            dA_tmp = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dA_tmp, data1=dA_from_u, data2=dA_from_w, op=nl.add
            )

            # Mask to strict lower (i > j)
            dA_tmp_masked = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dA_tmp_masked, data1=dA_tmp, data2=Lmask, op=nl.multiply
            )

            # -- Apply matrix-inverse gradient: dA = N^T @ (dA_tmp_masked @ N^T) (up to signs) --
            # In FLA (with b_A = N^T): dot(dA, b_A) = dA @ N^T, then dot(b_A, that) = N^T @ dA @ N^T.
            # We have N_c stored plain (N). To get N^T we already have N_T.
            # Step 1: dA_tmp_masked @ N^T. nc_matmul(stationary=dA_tmp_masked^T, moving=N^T).
            # But we're contracting on partition. Let's compute (dA_tmp_masked @ N^T) directly.
            # nc_matmul(stationary=A, moving=B) = A^T @ B. We want C = X @ Y = X @ N^T where X = dA_tmp_masked.
            # So A^T = X -> A = X^T = dA_tmp_masked^T; B = N^T. Result C shape (P_MAX, P_MAX).
            dA_masked_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dA_masked_T_psum, data=dA_tmp_masked)
            dA_masked_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dA_masked_T, src=dA_masked_T_psum)

            dA_step1_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dA_step1_psum, stationary=dA_masked_T, moving=N_T)
            dA_step1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dA_step1, src=dA_step1_psum)

            # Step 2: N^T @ dA_step1. Same pattern.
            N_T_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=N_T_T_psum, data=N_T)
            N_TT = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=N_TT, src=N_T_T_psum)
            # Note: N_TT = N (transposing N_T back). But keep it separate for clarity.

            dA_step2_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            # We want N^T @ dA_step1, so stationary = N^T^T = N, moving = dA_step1.
            # nc_matmul(stationary=N, moving=dA_step1) = N^T @ dA_step1. Wait but we already have N_c.
            # Actually nc_matmul(stationary=A, moving=B) = A^T @ B. So we want A^T = N^T, meaning A = N.
            # Just use N_c directly.
            nisa.nc_matmul(dst=dA_step2_psum, stationary=N_c, moving=dA_step1)
            dA_step2 = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dA_step2, src=dA_step2_psum)

            # Apply L_mask factor: dA_step2 *= exp_gc_diff (which for i>j is exp(gc[i] - gc[j]))
            dA_scaled = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dA_scaled, data1=dA_step2, data2=exp_gc_diff, op=nl.multiply
            )

            # Sign flip and strict-lower mask
            dA_scaled_neg = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dA_scaled_neg, data=dA_scaled, op0=nl.multiply, operand0=-1.0,
                engine=nisa.vector_engine,
            )
            dA_final = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dA_final, data1=dA_scaled_neg, data2=Lmask, op=nl.multiply
            )

            # -- dk_beta_from_A = dA_final @ k_c --
            # nc_matmul(stationary=dA_final^T, moving=k_c) = dA_final @ k_c.
            dA_final_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dA_final_T_psum, data=dA_final)
            dA_final_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dA_final_T, src=dA_final_T_psum)

            dk_beta_from_A_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dk_beta_from_A_psum, stationary=dA_final_T, moving=k_c)
            dk_beta_from_A = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dk_beta_from_A, src=dk_beta_from_A_psum)

            # -- dk_from_A = dA_final^T @ k_beta --
            # nc_matmul(stationary=dA_final, moving=k_beta) = dA_final^T @ k_beta.
            dk_from_A_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dk_from_A_psum, stationary=dA_final, moving=k_beta)
            dk_from_A = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dk_from_A, src=dk_from_A_psum)

            # -- dk_beta_from_w = (N^T @ dw) * exp(gc) --
            # dkbs_from_w = N^T @ dw was computed earlier via the fused N_c matmul.
            # Now multiply by exp(gc) to get dk_beta_from_w.
            dk_beta_from_w = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dk_beta_from_w, data=dkbs_from_w, op0=nl.multiply, operand0=exp_gc_p,
                engine=nisa.vector_engine,
            )

            # -- Total dk_beta = dk_beta_from_A + dk_beta_from_w --
            dk_beta_total = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dk_beta_total, data1=dk_beta_from_A, data2=dk_beta_from_w, op=nl.add
            )

            # -- Update dk: dk = dk_intra + dk_state + dk_from_A + dk_beta_total * beta --
            # dk from k_beta path = dk_beta * beta (since k_beta = k * beta).
            dk_from_kbeta = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dk_from_kbeta, data=dk_beta_total, op0=nl.multiply, operand0=beta_p,
                engine=nisa.vector_engine,
            )
            dk_add_A = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dk_add_A, data1=dk_partial, data2=dk_from_A, op=nl.add)
            dk_full = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dk_full, data1=dk_add_A, data2=dk_from_kbeta, op=nl.add)

            # OVERWRITE the dk_out slice (Task 009 wrote dk_partial; now write full dk)
            nisa.dma_copy(dst=dk_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:dim], src=dk_full)

            # -- Total dbeta = sum(dv_beta * v_c, axis=-1) + sum(dk_beta_total * k_c, axis=-1) --
            dv_beta_times_v = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dv_beta_times_v, data1=dv_beta, data2=v_c, op=nl.multiply
            )
            dbeta_from_v = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dbeta_from_v, data=dv_beta_times_v, op=nl.add, axis=(1,),
            )

            dk_beta_times_k = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dk_beta_times_k, data1=dk_beta_total, data2=k_c, op=nl.multiply
            )
            dbeta_from_k = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dbeta_from_k, data=dk_beta_times_k, op=nl.add, axis=(1,),
            )
            dbeta_full = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dbeta_full, data1=dbeta_from_v, data2=dbeta_from_k, op=nl.add
            )
            # OVERWRITE dbeta_out (Task 009 wrote partial; now write full)
            nisa.dma_copy(
                dst=dbeta_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1],
                src=dbeta_full,
            )

            # =================================================================
            # Task 010: dg log-decay gradient via reverse cumsum
            # =================================================================
            # dgc[t] = sum of contributions from 5 per-token paths:
            #   1. o_state: sum_d(dq_state[t,d] * q_c[t,d])
            #   2. intra-QK L_mask: sum(X_intra, axis=1)_t - sum(X_intra, axis=0)_t where
            #      X_intra = d_attn * (q @ k^T) * m_diag * L_mask
            #   3. A L_mask: sum(X_A, axis=1)_t - sum(X_A, axis=0)_t where
            #      X_A = dA_final * (k_beta @ k^T)
            #   4. k_beta_scaled (w path): sum_d(dkbs_from_w[t,d] * k_beta_scaled[t,d])
            #   5. k_last_decay: -sum_d(dk_state_pre[t,d] * k_last_decay[t,d])
            # dg_last (scalar, folded into dgc[C-1]) from:
            #   6a. k_last_decay: sum_{t,d}(dk_state_pre[t,d] * k_last_decay[t,d])
            #   6b. exp(g_last) on state update: sum_{i,j}(dS[i,j] * exp(g_last) * S_c[i,j])
            # Then dg[t] = reverse_cumsum(dgc)[t] via flip-scan-flip pattern.
            # =================================================================

            # ---- Path 1: dgc_from_o_state = sum(dq_state * q_c, axis=1) ----
            dq_state_x_q = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dq_state_x_q, data1=dq_state, data2=q_c, op=nl.multiply
            )
            dgc_from_o_state = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dgc_from_o_state, data=dq_state_x_q, op=nl.add, axis=(1,),
            )

            # ---- Path 2: dgc_from_intra_Lmask ----
            # X_intra = d_attn * (q @ k^T) * m_diag * L_mask
            # Task 016 opt: reuse qkT computed earlier (line 470-ish) instead of recomputing.
            # (Previously we re-ran nc_matmul(stationary=q_T, moving=k_T) here.)

            # X_intra step by step
            d_attn_x_qkT = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=d_attn_x_qkT, data1=d_attn, data2=qkT, op=nl.multiply
            )
            d_attn_x_qkT_mdiag = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=d_attn_x_qkT_mdiag, data1=d_attn_x_qkT, data2=Lmask_d, op=nl.multiply
            )
            X_intra = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=X_intra, data1=d_attn_x_qkT_mdiag, data2=exp_gc_diff, op=nl.multiply
            )
            # NOTE: exp_gc_diff is only correct for i>=j (upper is bogus but we've masked with m_diag).
            # Since X_intra is masked by m_diag (i>=j), upper triangle is zero -- safe.

            row_sum_Xi = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=row_sum_Xi, data=X_intra, op=nl.add, axis=(1,),
            )

            # For column sum: transpose X_intra then row-sum.
            X_intra_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=X_intra_T_psum, data=X_intra)
            X_intra_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=X_intra_T, src=X_intra_T_psum)
            col_sum_Xi = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=col_sum_Xi, data=X_intra_T, op=nl.add, axis=(1,),
            )

            dgc_from_intra_Lmask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dgc_from_intra_Lmask, data1=row_sum_Xi, data2=col_sum_Xi, op=nl.subtract
            )

            # ---- Path 3: dgc_from_A_Lmask ----
            # X_A = dA_final * (k_beta @ k^T)
            k_beta_kT_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            # nc_matmul(stationary=k_beta^T, moving=k^T) = k_beta @ k^T
            # k_beta shape (P_MAX, dim); need k_beta^T (dim, P_MAX). k already has k_T from earlier.
            k_beta_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=k_beta_T_psum, data=k_beta)
            k_beta_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_beta_T, src=k_beta_T_psum)
            nisa.nc_matmul(dst=k_beta_kT_psum, stationary=k_beta_T, moving=k_T)
            k_beta_kT = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=k_beta_kT, src=k_beta_kT_psum)

            X_A = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=X_A, data1=dA_final, data2=k_beta_kT, op=nl.multiply
            )
            row_sum_XA = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=row_sum_XA, data=X_A, op=nl.add, axis=(1,),
            )
            X_A_T_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=X_A_T_psum, data=X_A)
            X_A_T = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=X_A_T, src=X_A_T_psum)
            col_sum_XA = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=col_sum_XA, data=X_A_T, op=nl.add, axis=(1,),
            )
            dgc_from_A_Lmask = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dgc_from_A_Lmask, data1=row_sum_XA, data2=col_sum_XA, op=nl.subtract
            )

            # ---- Path 4: dgc_from_kbs = sum(dkbs_from_w * k_beta_scaled, axis=1) ----
            dkbs_x_kbs = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dkbs_x_kbs, data1=dkbs_from_w, data2=k_beta_scaled, op=nl.multiply
            )
            dgc_from_kbs = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dgc_from_kbs, data=dkbs_x_kbs, op=nl.add, axis=(1,),
            )

            # ---- Path 5: dgc_from_k_last_decay = -sum(dk_state_pre * k_last_decay, axis=1) ----
            dkstate_x_klast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dkstate_x_klast, data1=dk_state_pre, data2=k_last_decay, op=nl.multiply
            )
            dgc_from_k_last_pos = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dgc_from_k_last_pos, data=dkstate_x_klast, op=nl.add, axis=(1,),
            )
            dgc_from_k_last_decay = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dgc_from_k_last_decay, data=dgc_from_k_last_pos,
                op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine,
            )

            # ---- Sum all per-token dgc contributions ----
            dgc_1 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dgc_1, data1=dgc_from_o_state, data2=dgc_from_intra_Lmask, op=nl.add)
            dgc_2 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dgc_2, data1=dgc_1, data2=dgc_from_A_Lmask, op=nl.add)
            dgc_3 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dgc_3, data1=dgc_2, data2=dgc_from_kbs, op=nl.add)
            dgc_p = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dgc_p, data1=dgc_3, data2=dgc_from_k_last_decay, op=nl.add)

            # ---- dg_last (scalar) from k_last_decay and state update ----
            # dg_last_from_klast = sum over all t,d of dk_state_pre * k_last_decay = sum(dgc_from_k_last_pos)
            # This is a (P_MAX, 1) reduced over axis=0 to scalar. Use another tensor_reduce.
            dg_last_from_klast_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            # tensor_reduce over axis=0 for (P_MAX, 1) -> (1, 1). But NKI reduce is on free axis.
            # Alternative: compute via transpose + reduce axis=1.
            dgc_pos_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dgc_pos_row_psum, data=dgc_from_k_last_pos)
            dgc_pos_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dgc_pos_row, src=dgc_pos_row_psum)
            nisa.tensor_reduce(
                dst=dg_last_from_klast_p, data=dgc_pos_row, op=nl.add, axis=(1,),
            )

            # dg_last_from_state = sum(dS * exp(g_last) * S_c)
            # First compute dS * S_c elementwise, then reduce all.
            dS_x_Sc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dS_x_Sc, data1=dS, data2=S_c, op=nl.multiply
            )
            # sum axis=1 first -> (P_MAX, 1)
            dS_x_Sc_row = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dS_x_Sc_row, data=dS_x_Sc, op=nl.add, axis=(1,),
            )
            # then transpose to (1, P_MAX) and sum axis=1 -> (1, 1)
            dS_x_Sc_row_T_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dS_x_Sc_row_T_psum, data=dS_x_Sc_row)
            dS_x_Sc_row_T = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dS_x_Sc_row_T, src=dS_x_Sc_row_T_psum)
            dS_x_Sc_sum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=dS_x_Sc_sum, data=dS_x_Sc_row_T, op=nl.add, axis=(1,),
            )
            # multiply by exp(g_last)
            dg_last_from_state_p = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dg_last_from_state_p, data1=dS_x_Sc_sum, data2=exp_gl_11, op=nl.multiply
            )

            # dg_last_total = dg_last_from_klast + dg_last_from_state
            dg_last_total = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dg_last_total, data1=dg_last_from_klast_p, data2=dg_last_from_state_p, op=nl.add
            )

            # ---- Convert dgc_p (P_MAX, 1) to row form (1, C) BEFORE modifying dgc[C-1] ----
            # Device compiler forbids partition-dim slicing starting at non-zero offsets,
            # so we work entirely in row form (free-dim slicing is unrestricted).
            dgc_row_psum = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dgc_row_psum, data=dgc_p)
            dgc_row_orig = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dgc_row_orig, src=dgc_row_psum)

            # Add dg_last_total to dgc_row[0, C-1] using free-dim slice (allowed)
            # First read the (1,1) slice at the end
            dgc_last_slice = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dgc_last_slice, src=dgc_row_orig[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE])
            dgc_last_new = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dgc_last_new, data1=dgc_last_slice, data2=dg_last_total, op=nl.add
            )
            # Write back
            nisa.tensor_copy(
                dst=dgc_row_orig[0:1, CHUNK_SIZE - 1 : CHUNK_SIZE], src=dgc_last_new
            )

            # ---- Reverse cumsum in row form via flip -> forward-scan -> flip ----
            # Step 1: flip dgc_row_orig along free axis (free-dim slicing is allowed)
            dgc_flipped_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            for i_flip in nl.static_range(CHUNK_SIZE):
                nisa.tensor_copy(
                    dst=dgc_flipped_row[0:1, i_flip : i_flip + 1],
                    src=dgc_row_orig[0:1, (CHUNK_SIZE - 1 - i_flip) : (CHUNK_SIZE - i_flip)],
                )
            # Step 2: forward cumsum
            dgc_scan_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(
                dst=dgc_scan_row, data0=ones_1xC, data1=dgc_flipped_row,
                initial=zero_11, op0=nl.multiply, op1=nl.add,
            )
            # Step 3: flip again in row form to get final dg (row layout)
            dg_row = nl.ndarray((1, CHUNK_SIZE), dtype=nl.float32, buffer=nl.sbuf)
            for i_flip in nl.static_range(CHUNK_SIZE):
                nisa.tensor_copy(
                    dst=dg_row[0:1, i_flip : i_flip + 1],
                    src=dgc_scan_row[0:1, (CHUNK_SIZE - 1 - i_flip) : (CHUNK_SIZE - i_flip)],
                )
            # Step 4: transpose back to column form (P_MAX, 1)
            dg_col_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=dg_col_psum, data=dg_row)
            dg_c = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dg_c, src=dg_col_psum)

            # Write dg to HBM (OVERWRITE the earlier zero stub)
            nisa.dma_copy(
                dst=dg_out[sample_idx, chunk_start : chunk_start + CHUNK_SIZE, 0:1], src=dg_c
            )

            # =================================================================
            # dS UPDATE (full recurrence per Task 006):
            #   dS = exp(g_last) * dS + (q_c * exp(gc))^T @ dO_c - w^T @ dv_new
            # =================================================================

            # Term 1: exp(g_last) * dS
            dS_decayed = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=dS_decayed, data=dS, op0=nl.multiply, operand0=exp_gl_p,
                engine=nisa.vector_engine,
            )

            # Term 2: (q_c * exp(gc))^T @ dO_c
            q_exp_gc = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=q_exp_gc, data=q_c, op0=nl.multiply, operand0=exp_gc_p,
                engine=nisa.vector_engine,
            )
            # nc_matmul(stationary=q_exp_gc, moving=dO_c) = q_exp_gc^T @ dO_c
            dS_ostate_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dS_ostate_psum, stationary=q_exp_gc, moving=dO_c)
            dS_ostate = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dS_ostate, src=dS_ostate_psum)

            # Term 3: - w^T @ dv_new
            # nc_matmul(stationary=w, moving=dv_new) = w^T @ dv_new
            dS_vprime_psum = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dS_vprime_psum, stationary=w_c, moving=dv_new)
            dS_vprime = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=dS_vprime, src=dS_vprime_psum)

            # Combine: dS = dS_decayed + dS_ostate - dS_vprime
            dS_new_part = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=dS_new_part, data1=dS_decayed, data2=dS_ostate, op=nl.add
            )
            nisa.tensor_tensor(
                dst=dS, data1=dS_new_part, data2=dS_vprime, op=nl.subtract
            )

        # -----------------------------------------------------------------
        # Write dS to dinitial_state HBM
        # -----------------------------------------------------------------
        nisa.dma_copy(dst=dinitial_state_out[sample_idx, 0:P_MAX, 0:dim], src=dS)

    return dq_out, dk_out, dv_out, dg_out, dbeta_out, dinitial_state_out
