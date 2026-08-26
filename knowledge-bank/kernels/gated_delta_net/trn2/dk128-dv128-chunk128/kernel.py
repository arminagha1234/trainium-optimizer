"""gdn_chunked_fwd.py — GatedDeltaNet (GDN) chunked forward, ported to the
neuronxcc.nki 0.6.0 stack (trn2).

Ported from `deltanet_fused_chunked_fwd_batched.py` (NKI 0.3, DeltaNet scalar
decay ≡ GDN). Two changes from the base:
  1. imports `nki` -> `neuronxcc.nki` (on this stack `nki`'s nc_matmul does NOT
     lower; neuronxcc.nki is the compiling package).
  2. dst-first ISA calls -> 0.6.0 RETURN-form (assign the result; no dst=), per
     the on-device signature map. Only dma_copy stays dst=/src=. Partition
     broadcasts use nl.broadcast_to instead of nc_stream_shuffle.

Math is UNCHANGED from the base (it is already GDN): per-chunk Neumann
power-doubling for the intra-chunk WY correction + cross-chunk gated state carry.
Validated against gdn_reference.py (numpy) at the bf16 floor.

Layout: token-major [BH, S, D], D = dk = dv = 128 = P_MAX = CHUNK_SIZE. S % 128 == 0.
Inputs: query (l2-normed AND scaled by 1/sqrt(dk)), key (l2-normed), value,
g_in (RAW per-token log-decay, cumsum in-kernel), beta_in (write gate),
initial_state (pass zeros if unused). The gate path is kept fp32 throughout
(bf16 exp = ~28% error).
"""

from __future__ import annotations

import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

P_MAX = 128
CHUNK_SIZE = 128
N_DOUBLING = 6            # log2(128); (I+A)(I+A^2)...(I+A^64) resolves rank up to 128


def make_masks():
    """Host constants: strict-lower, lower+diag, identity (128x128)."""
    lower = np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), np.float32), k=-1)
    lower_diag = np.tril(np.ones((CHUNK_SIZE, CHUNK_SIZE), np.float32), k=0)
    identity = np.eye(CHUNK_SIZE, dtype=np.float32)
    return lower, identity, lower_diag


@nki.jit
def gdn_chunked_fwd(query, key, value, g_in, beta_in, initial_state,
                    lower_mask, identity, lower_mask_diag):
    BH = query.shape[0]
    seq_len = query.shape[1]
    dim = query.shape[2]
    num_chunks = seq_len // CHUNK_SIZE

    output = nl.ndarray((BH, seq_len, dim), dtype=query.dtype, buffer=nl.shared_hbm)
    final_state_out = nl.ndarray((BH, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # constant masks -> SBUF once (nl.load = HBM -> SBUF; tensor_copy is SBUF/PSUM only)
    eye = nl.load(identity)                         # [P,P]
    Lmask = nl.load(lower_mask)
    Lmask_d = nl.load(lower_mask_diag)
    ones_1xC = nisa.memset((1, CHUNK_SIZE), 1.0, dtype=nl.float32)
    zero_11 = nisa.memset((1, 1), 0.0, dtype=nl.float32)

    for sample_idx in nl.sequential_range(BH):
        # recurrent state [dk, dv] in SBUF, carried across chunks
        state = nl.load(initial_state[sample_idx, 0:P_MAX, 0:dim])

        for i_chunk in nl.static_range(num_chunks):
            cs = i_chunk * CHUNK_SIZE

            # ---- load chunk (token-major, tokens on partition axis) ----
            q_c = nl.load(query[sample_idx, cs:cs + CHUNK_SIZE, 0:dim])
            k_c = nl.load(key[sample_idx, cs:cs + CHUNK_SIZE, 0:dim])
            v_c = nl.load(value[sample_idx, cs:cs + CHUNK_SIZE, 0:dim])
            g_col = nl.load(g_in[sample_idx, cs:cs + CHUNK_SIZE, 0:1])     # [C,1]
            beta_p = nl.load(beta_in[sample_idx, cs:cs + CHUNK_SIZE, 0:1])  # [C,1]

            # ---- cumsum(g) along the sequence, in-kernel ----
            # transpose [C,1] -> [1,C] (scan runs on the free axis), scan, back.
            g_row = nisa.nc_transpose(g_col)                                   # [1,C] (PSUM)
            g_row = nisa.tensor_copy(g_row)                                    # -> SBUF
            gc_row = nisa.tensor_tensor_scan(ones_1xC, g_row, zero_11,
                                             nl.multiply, nl.add)             # [1,C] cumsum
            gc_col = nisa.nc_transpose(gc_row)                                 # [C,1] (PSUM)
            gc_col = nisa.tensor_copy(gc_col)                                  # -> SBUF [C,1]
            gl_11 = nisa.tensor_copy(gc_row[0:1, CHUNK_SIZE - 1:CHUNK_SIZE])   # [1,1] g_last

            # ---- exp factors (all args <= 0 -> in (0,1], no overflow) ----
            exp_gc = nisa.activation(nl.exp, gc_col)                           # [C,1] exp(gc)
            gl_bc = nl.broadcast_to(gl_11, shape=(P_MAX, 1))                   # [C,1] g_last
            gl_minus_gc = nisa.tensor_tensor(gl_bc, gc_col, nl.subtract)       # [C,1]
            exp_gl_minus_gc = nisa.activation(nl.exp, gl_minus_gc)             # [C,1]
            exp_gl_11 = nisa.activation(nl.exp, gl_11)                         # [1,1]
            exp_gl = nl.broadcast_to(exp_gl_11, shape=(P_MAX, 1))              # [C,1]

            # ---- k_beta, v_beta (per-token scalar broadcast across free dim) ----
            k_beta = nisa.tensor_scalar(k_c, nl.multiply, beta_p, engine=nisa.vector_engine)
            v_beta = nisa.tensor_scalar(v_c, nl.multiply, beta_p, engine=nisa.vector_engine)

            # v2 OPT: use nl.matmul(x, y) = x @ y (the stationary transpose is FUSED
            # into the TensorE load — no explicit nc_transpose + PSUM->SBUF copy).
            # Only k_T (= k_c^T) is still materialized, because it is the shared RHS
            # of the two QK-style products (k_beta @ k^T and q @ k^T). All ~13 other
            # per-chunk transposes the base kernel did are eliminated.
            k_T = nisa.tensor_copy(nisa.nc_transpose(k_c))                    # [dk,C]

            # ---- stable decay mask exp(gc[i]-gc[j]) ----
            gc_mat_rows = nl.broadcast_to(gc_col, shape=(P_MAX, CHUNK_SIZE))   # [i,j]=gc[i]
            gc_mat_cols = nl.broadcast_to(gc_row, shape=(P_MAX, CHUNK_SIZE))   # [i,j]=gc[j]
            gc_diff = nisa.tensor_tensor(gc_mat_rows, gc_mat_cols, nl.subtract)
            gc_diff_c = nisa.tensor_scalar(gc_diff, nl.minimum, 0.0, engine=nisa.vector_engine)
            exp_gc_diff = nisa.activation(nl.exp, gc_diff_c)                   # [C,C] in (0,1]

            # ---- A = -(k_beta @ k^T) * decay * strict_lower ----
            QK = nl.matmul(k_beta, k_T)                                       # k_beta @ k^T [C,C]
            QK_decay = nisa.tensor_tensor(QK, exp_gc_diff, nl.multiply)
            neg = nisa.tensor_scalar(QK_decay, nl.multiply, -1.0, engine=nisa.vector_engine)
            A_mat = nisa.tensor_tensor(neg, Lmask, nl.multiply)               # [C,C]

            # ---- Neumann doubling: N = (I+A)(I+A^2)...(I+A^64) — transpose-free ----
            P_acc = nisa.tensor_tensor(eye, A_mat, nl.add)
            A_pow = nisa.tensor_copy(A_mat)
            for _r in nl.static_range(N_DOUBLING):
                A_pow = nisa.tensor_copy(nl.matmul(A_pow, A_pow))            # A_pow^2
                IpA = nisa.tensor_tensor(eye, A_pow, nl.add)
                P_acc = nisa.tensor_copy(nl.matmul(IpA, P_acc))             # (I+A_pow)@P_acc

            # ---- value_corr = N @ v_beta ; k_cumdecay = N @ (k_beta*exp(gc)) ----
            value_corr = nisa.tensor_copy(nl.matmul(P_acc, v_beta))          # [C,dv]
            kb_exp = nisa.tensor_scalar(k_beta, nl.multiply, exp_gc, engine=nisa.vector_engine)
            k_cumdecay = nisa.tensor_copy(nl.matmul(P_acc, kb_exp))          # [C,dk]

            # ---- attn_intra = (q@k^T)*decay*lower_diag ----
            qk = nl.matmul(q_c, k_T)                                          # q @ k^T [C,C]
            qk_decay = nisa.tensor_tensor(qk, exp_gc_diff, nl.multiply)
            attn_intra = nisa.tensor_tensor(qk_decay, Lmask_d, nl.multiply)

            # ---- v_prime = k_cumdecay @ state ; v_new = value_corr - v_prime ----
            vp = nl.matmul(k_cumdecay, state)                                # [C,dv]
            v_new = nisa.tensor_tensor(value_corr, vp, nl.subtract)

            # ---- attn_inter = (q*exp(gc)) @ state ----
            q_exp = nisa.tensor_scalar(q_c, nl.multiply, exp_gc, engine=nisa.vector_engine)
            attn_inter = nisa.tensor_copy(nl.matmul(q_exp, state))           # [C,dv]

            # ---- out = attn_inter + attn_intra @ v_new ----
            intra = nl.matmul(attn_intra, v_new)                             # [C,dv]
            chunk_out = nisa.tensor_tensor(attn_inter, intra, nl.add)
            nisa.dma_copy(dst=output[sample_idx, cs:cs + CHUNK_SIZE, 0:dim], src=chunk_out)

            # ---- state = exp(g_last)*state + (k*exp(g_last-gc))^T @ v_new ----
            k_state_decay = nisa.tensor_scalar(k_c, nl.multiply, exp_gl_minus_gc,
                                               engine=nisa.vector_engine)
            kv = nl.matmul(k_state_decay, v_new, transpose_x=True)           # k_sd^T @ v_new [dk,dv]
            state_dec = nisa.tensor_scalar(state, nl.multiply, exp_gl, engine=nisa.vector_engine)
            state = nisa.tensor_tensor(state_dec, kv, nl.add)                 # carry

        nisa.dma_copy(dst=final_state_out[sample_idx, 0:P_MAX, 0:dim], src=state)

    return output, final_state_out
