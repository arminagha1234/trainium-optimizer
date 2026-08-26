"""reference.py — Mamba-2 SSD (selective-scan) chunk-form ground truth (numpy).

Oracle for the mamba2_ssd forward kernel. Single (b,h); the layer loops b,h.
Cross-checked against a per-token recurrence and torch autograd (see backward/reference.py).
"""
import numpy as np

Q = 128


def ssd_mamba2_ref(x, dt, A, B, C, chunk_size=Q, initial_state=None):
    """x:[L,p] dt:[L] A:scalar B:[L,n] C:[L,n] -> y:[L,p], final_state:[n,p]."""
    L, p = x.shape; n = B.shape[1]; QQ = chunk_size; nch = L // QQ
    x, dt, B, C = (np.asarray(t, np.float64) for t in (x, dt, B, C)); A = float(A)
    y = np.zeros((L, p), np.float64)
    causal = np.tril(np.ones((QQ, QQ)))
    state = np.zeros((n, p)) if initial_state is None else np.asarray(initial_state, np.float64)
    for ic in range(nch):
        s, e = ic * QQ, ic * QQ + QQ
        x_c, dt_c, B_c, C_c = x[s:e], dt[s:e], B[s:e], C[s:e]
        cs = np.cumsum(dt_c * A, 0)                       # (Q,)
        exp_cs, exp_neg = np.exp(cs), np.exp(-cs)
        dtx = dt_c[:, None] * x_c
        CB = (C_c @ B_c.T) * causal
        Xs = dtx * exp_neg[:, None]
        Y_intra = exp_cs[:, None] * (CB @ Xs)
        Y_off = exp_cs[:, None] * (C_c @ state)
        y[s:e] = Y_intra + Y_off
        csl = cs[-1]; dec = np.exp(csl - cs)
        state = np.exp(csl) * state + B_c.T @ (dtx * dec[:, None])
    return y, state
