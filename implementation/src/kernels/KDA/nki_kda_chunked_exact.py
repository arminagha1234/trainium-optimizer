"""Exact per-channel KDA prefill chunk -- single 128-token chunk.

Numerically-exact per-channel decay, ported from a reference prefill kernel into a
single-function kernel matching this package's kda_chunk_step contract. Unlike the
scalar-mean approximation in nki_kda_chunked.py (which degrades to cos_sim ~0.22-0.49
once the gate decay is non-trivial), this reaches the fp32 floor in every regime.

WHY SUB=16: a per-channel gate does NOT factor the pairwise token decay into a
[C,C] mask. The only separable form is k_inv = k*exp(-gcum), whose range grows as
exp(|lower_bound|*SUB). At lower_bound=-5, exp(80)~5.5e34 is safe in fp32 at SUB=16
but exp(160) overflows at SUB=32. So the sub-chunk size is forced by NUMERICS.

Contract (matches kda_chunk_step; inputs PRE-PROCESSED):
  query : (C, dk) L2-normed, scaled by 1/sqrt(dk)
  key : (C, dk) L2-normed
  value : (C, dv)
  beta : (C, 1) per-token scalar write gate
  g : (C, dk) per-channel ACTIVATED log-decay (<= 0), NOT cumulative
  state_in: (dk, dv) recurrent state

Per SUB=16 sub-chunk (threaded across the 8 sub-chunks of the 128-tile):
  gcum = cumsum(g)
  k_dec = k*exp(gcum); q_dec = q*exp(gcum); k_inv = k*exp(-gcum)
  k_restr = k*exp(g_total - gcum) # stable, exponent <= 0
  L = strict_lower(k_dec @ k_inv.T) * beta
  Mqk = lower_incl(q_dec @ k_inv.T)
  INV = (I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8) # exact, L^16=0
  v' = (v - k_dec @ S) * beta
  U = INV @ v'
  out = q_dec @ S + Mqk @ U
  S = S*exp(g_total)[:,None] + k_restr.T @ U

DTYPE: gate path (gcum, exp_gcum, exp_neg_gcum, k_restr) stays fp32 -- these are
exponents; a bf16 rounding of an exponent makes exp wrong by ~28%. This kernel
keeps everything fp32 for simplicity and exactness (bf16 matmul optimization is a
later step; correctness first).

Written with plain for-loops only (public-NKI compatible; no comprehensions,
no tuple-unpacking targets).

nc_matmul(dst, stationary, moving) = stationary.T @ moving, contract partition dim.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
SUB = 16
NSUB = P_MAX // SUB # 8
NEUMANN_FACTORS = 3 # squarings: (I-L)(I+L^2)(I+L^4)(I+L^8); L^16=0 -> exact
_L2_EPS = 1.0e-6


@nki.jit
def kda_chunk_step_exact(
    query: nl.ndarray, # (C, dk) pre-normed + scaled
    key: nl.ndarray, # (C, dk) pre-normed
    value: nl.ndarray, # (C, dv)
    beta: nl.ndarray, # (C, 1) per-token scalar write gate
    g: nl.ndarray, # (C, dk) activated per-channel log-decay (<= 0)
    state_in: nl.ndarray, # (dk, dv)
):
    """Exact per-channel KDA prefill for one 128-token chunk.

    Returns:
        chunk_out : (C, dv)
        state_out : (dk, dv)
    """
    C, dk = query.shape
    dv = value.shape[-1]

    chunk_out = nl.ndarray((C, dv), dtype=query.dtype, buffer=nl.shared_hbm)
    state_out = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.shared_hbm)

    # State in SBUF, seeded from HBM.
    S = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=S, src=state_in[0:dk, 0:dv])

    # Constant SUBxSUB masks: strict lower and lower-inclusive, plus eye.
    # Built with the proven iota+relu+sign pattern (matches nki_kda_chunked.py),
    # avoiding the iota `base=` kwarg (not in this NKI version).
    strict = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
    incl = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
    eye = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)

    # row_minus_col[i,j] = i - j
    row_minus_col = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=row_minus_col, pattern=[[-1, SUB]], offset=0, channel_multiplier=1)
    # incl[i,j] = 1 if i >= j : sign(relu(i - j + 0.5))
    rmc_d = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rmc_d, data=row_minus_col, op0=nl.add, operand0=0.5, engine=nisa.vector_engine)
    nisa.activation(dst=rmc_d, op=nl.relu, data=rmc_d, bias=None, scale=1.0)
    nisa.activation(dst=incl, op=nl.sign, data=rmc_d, bias=None, scale=1.0)
    # strict[i,j] = 1 if i > j : sign(relu(i - j - 0.5))
    rmc_s = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rmc_s, data=row_minus_col, op0=nl.add, operand0=-0.5, engine=nisa.vector_engine)
    nisa.activation(dst=rmc_s, op=nl.relu, data=rmc_s, bias=None, scale=1.0)
    nisa.activation(dst=strict, op=nl.sign, data=rmc_s, bias=None, scale=1.0)
    # eye = incl - strict
    nisa.tensor_tensor(dst=eye, data1=incl, data2=strict, op=nl.subtract)

    # (scan constants no longer needed -- cumsum is via incl matmul)

    for s in nl.sequential_range(NSUB):
        o = s * SUB
        # ---- Load sub-chunk tiles (token-major [SUB, d]) ----
        q_s = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_s, src=query[o : o + SUB, 0:dk])
        k_s = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_s, src=key[o : o + SUB, 0:dk])
        v_s = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_s, src=value[o : o + SUB, 0:dv])
        g_s = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_s, src=g[o : o + SUB, 0:dk])
        b_s = nl.ndarray((SUB, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_s, src=beta[o : o + SUB, 0:1])

        # ---- gcum = cumsum(g) over tokens (partition axis here is tokens) ----
        # tokens on partition axis; cumsum is along partition, but tensor_tensor_scan
        # scans the FREE axis. So compute cumsum via a lower-triangular matmul:
        # gcum[i,d] = sum_{j<=i} g[j,d] = incl @ g (incl is [SUB,SUB] lower-incl)
        # nc_matmul(stationary=incl_T, moving=g_s) = incl_T.T @ g_s = incl @ g_s.
        # incl is symmetric only in structure; we need incl (i>=j). incl.T has j>=i,
        # so use strict/incl transposed appropriately: we want row i = sum over j<=i.
        # nc_matmul contracts partition dim: (incl_stat).T @ g_s, so stationary must
        # be incl.T so that (incl.T).T = incl.
        incl_T = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=incl_T, data=incl)
        gcum_p = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=gcum_p, stationary=incl_T, moving=g_s)
        gcum = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gcum, src=gcum_p)

        exp_gcum = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_gcum, op=nl.exp, data=gcum)
        exp_neg_gcum = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_neg_gcum, op=nl.exp, data=gcum, scale=-1.0)

        k_dec = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_dec, data1=k_s, data2=exp_gcum, op=nl.multiply)
        q_dec = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_dec, data1=q_s, data2=exp_gcum, op=nl.multiply)
        k_inv = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_inv, data1=k_s, data2=exp_neg_gcum, op=nl.multiply)

        # g_total = gcum[-1] (last token row). k_restr = k * exp(g_total - gcum).
        # g_total is a [1,dk] row; broadcast-subtract from gcum along partition.
        g_total = nl.ndarray((1, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_total, src=gcum[SUB - 1 : SUB, 0:dk])
        gdiff = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        # gdiff = gcum - g_total (broadcast g_total across partition via tensor_tensor
        # with a partition-broadcast; use nc_matmul(ones[SUB,1], g_total)).
        ones_col = nl.ndarray((1, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        gtot_bc_p = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=gtot_bc_p, stationary=ones_col, moving=g_total)
        gtot_bc = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gtot_bc, src=gtot_bc_p)
        nisa.tensor_tensor(dst=gdiff, data1=gcum, data2=gtot_bc, op=nl.subtract)
        decay = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=decay, op=nl.exp, data=gdiff, scale=-1.0) # exp(g_total - gcum)
        k_restr = nl.ndarray((SUB, dk), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_restr, data1=k_s, data2=decay, op=nl.multiply)

        # ---- L = strict_lower(k_dec @ k_inv.T) * beta ----
        # k_dec @ k_inv.T : need k_inv transposed. nc_matmul(k_dec_stat, moving) does
        # k_dec.T @ moving. We want [SUB,SUB] = k_dec @ k_inv.T, contract dk.
        # nc_matmul contracts PARTITION dim (tokens here). We need contract dk, so
        # transpose both to [dk, SUB] first (channel-major), then
        # nc_matmul(stationary=k_dec_T, moving=k_inv_T) = k_dec_T.T @ k_inv_T
        # = k_dec @ k_inv.T ([SUB,SUB]). Good.
        # Tensor-engine transpose writes PSUM; drain to SBUF for use as matmul operands.
        k_dec_Tp = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(engine=nisa.tensor_engine, dst=k_dec_Tp, data=k_dec)
        k_dec_T = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_dec_T, src=k_dec_Tp)
        k_inv_Tp = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(engine=nisa.tensor_engine, dst=k_inv_Tp, data=k_inv)
        k_inv_T = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_inv_T, src=k_inv_Tp)
        q_dec_Tp = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(engine=nisa.tensor_engine, dst=q_dec_Tp, data=q_dec)
        q_dec_T = nl.ndarray((dk, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q_dec_T, src=q_dec_Tp)

        L_p = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=L_p, stationary=k_dec_T, moving=k_inv_T)
        L = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=L, src=L_p)
        nisa.tensor_tensor(dst=L, data1=L, data2=strict, op=nl.multiply)
        # * beta (per-token, row i). beta is [SUB,1]; tensor_scalar broadcasts free.
        nisa.tensor_scalar(dst=L, data=L, op0=nl.multiply, operand0=b_s)

        # Mqk = lower_incl(q_dec @ k_inv.T)
        Mqk_p = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Mqk_p, stationary=q_dec_T, moving=k_inv_T)
        Mqk = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Mqk, src=Mqk_p)
        nisa.tensor_tensor(dst=Mqk, data1=Mqk, data2=incl, op=nl.multiply)

        # ---- INV = (I+L)^-1 = (I-L)(I+L^2)(I+L^4)(I+L^8), exact (L^16=0) ----
        INV = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=INV, data1=eye, data2=L, op=nl.subtract) # I - L
        Lp = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Lp, src=L)
        for _f in nl.static_range(NEUMANN_FACTORS):
            # Lp = Lp @ Lp
            Lp_T = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_transpose(dst=Lp_T, data=Lp)
            Lp2_p = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=Lp2_p, stationary=Lp_T, moving=Lp) # Lp.T.T? -> Lp@Lp
            Lp2 = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=Lp2, src=Lp2_p)
            nisa.tensor_copy(dst=Lp, src=Lp2)
            # INV = INV + INV @ Lp = INV @ (I + Lp)
            IpLp = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=IpLp, data1=eye, data2=Lp, op=nl.add)
            INV_T = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
            nisa.nc_transpose(dst=INV_T, data=INV)
            newINV_p = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=newINV_p, stationary=INV_T, moving=IpLp) # INV @ IpLp
            nisa.tensor_copy(dst=INV, src=newINV_p)

        # ---- v' = (v - k_dec @ S) * beta ; U = INV @ v' ----
        # k_dec @ S : contract dk. k_dec is [SUB,dk], S is [dk,dv].
        # nc_matmul(stationary=k_dec_T[dk,SUB], moving=S[dk,dv]) = k_dec @ S = [SUB,dv].
        kS_p = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kS_p, stationary=k_dec_T, moving=S)
        vprime = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=vprime, data1=v_s, data2=kS_p, op=nl.subtract)
        nisa.tensor_scalar(dst=vprime, data=vprime, op0=nl.multiply, operand0=b_s)
        # U = INV @ v' : contract SUB. nc_matmul(stationary=INV_T2, moving=vprime).
        INV_T2 = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=INV_T2, data=INV)
        U_p = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=U_p, stationary=INV_T2, moving=vprime)
        U = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=U, src=U_p)

        # ---- out = q_dec @ S + Mqk @ U ----
        qS_p = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=qS_p, stationary=q_dec_T, moving=S) # q_dec @ S
        qS = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qS, src=qS_p)
        Mqk_T = nl.ndarray((SUB, SUB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.nc_transpose(dst=Mqk_T, data=Mqk)
        MqkU_p = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=MqkU_p, stationary=Mqk_T, moving=U) # Mqk @ U
        out_s = nl.ndarray((SUB, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out_s, data1=qS, data2=MqkU_p, op=nl.add)
        nisa.dma_copy(dst=chunk_out[o : o + SUB, 0:dv], src=out_s)

        # ---- S = S * exp(g_total) + k_restr.T @ U ----
        # exp(g_total) is [1,dk]; per-channel decay along partition (dk) of S.
        # Need it as [dk,1] column. Transpose g_total row -> col, exp.
        gtot_col_p = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(engine=nisa.tensor_engine, dst=gtot_col_p, data=g_total)
        exp_gtot_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_gtot_col, op=nl.exp, data=gtot_col_p)
        # k_restr.T @ U : contract SUB (tokens). k_restr is [SUB,dk], U is [SUB,dv].
        # nc_matmul(stationary=k_restr[SUB,dk], moving=U[SUB,dv]) = k_restr.T @ U = [dk,dv].
        dS_p = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=dS_p, stationary=k_restr, moving=U)
        S_new = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=S_new,
            data=S,
            op0=nl.multiply,
            operand0=exp_gtot_col,
            op1=nl.add,
            operand1=dS_p,
        )
        nisa.tensor_copy(dst=S, src=S_new)

    nisa.dma_copy(dst=state_out[0:dk, 0:dv], src=S)
    return chunk_out, state_out
