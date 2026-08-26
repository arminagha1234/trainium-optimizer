"""mamba2_ssd_bwd_ref.py — Mamba-2 SSD backward: hand VJP vs autograd oracle (CPU).

Phase-2 of the kernel pipeline (oracle -> explicit VJP -> [next: NKI]). SSD has NO
triangular solve, so the per-chunk VJP is a clean set of matmuls + a reverse-cumsum
(mirrors the GDN dg reverse-cumsum, minus the doubling). Validates the hand grads
against torch autograd through a differentiable chunk forward. All-fp64, cos ~1.0.
"""
import numpy as np, torch

Q = 128


# ---- differentiable torch chunk forward (autograd oracle) -----------------------
def ssd_fwd_torch(x, dt, A, B, C, state0):
    L, p = x.shape; n = B.shape[1]; nch = L // Q
    causal = torch.tril(torch.ones(Q, Q, dtype=x.dtype))
    state = state0; ys = []
    for ic in range(nch):
        s, e = ic * Q, ic * Q + Q
        x_c, dt_c, B_c, C_c = x[s:e], dt[s:e], B[s:e], C[s:e]
        cs = torch.cumsum(dt_c * A, 0)                          # [Q,1]
        ecs, encs = cs.exp(), (-cs).exp()
        dtx = dt_c * x_c
        CB = (C_c @ B_c.T) * causal
        Xs = dtx * encs
        Y_intra = ecs * (CB @ Xs)
        Y_off = ecs * (C_c @ state)
        ys.append(Y_intra + Y_off)
        csl = cs[-1]; dec = (csl - cs).exp()
        state = csl.exp() * state + B_c.T @ (dtx * dec)
    return torch.cat(ys, 0), state


# ---- hand VJP (explicit, numpy) -------------------------------------------------
def ssd_bwd_hand(x, dt, A, B, C, state0, dy, dS_final):
    L, p = x.shape; n = B.shape[1]; nch = L // Q
    causal = np.tril(np.ones((Q, Q)))
    tri_ge = np.triu(np.ones((Q, Q)))          # reverse-cumsum: d_dtA = tri_ge @ d_cs
    # forward pass, caching per-chunk intermediates + entry states
    states = [state0.copy()]; cache = []
    st = state0.copy()
    for ic in range(nch):
        s, e = ic * Q, ic * Q + Q
        x_c, dt_c, B_c, C_c = x[s:e], dt[s:e], B[s:e], C[s:e]
        cs = np.cumsum(dt_c * A, 0)
        ecs, encs = np.exp(cs), np.exp(-cs)
        dtx = dt_c * x_c
        CB = (C_c @ B_c.T) * causal
        Xs = dtx * encs
        P = CB @ Xs
        G = C_c @ st
        csl = cs[-1:]; dec = np.exp(csl - cs)
        Sd = dtx * dec
        cache.append(dict(x_c=x_c, dt_c=dt_c, B_c=B_c, C_c=C_c, cs=cs, ecs=ecs, encs=encs,
                          dtx=dtx, CB=CB, Xs=Xs, P=P, G=G, csl=csl, dec=dec, Sd=Sd, st=st))
        st = np.exp(csl) * st + B_c.T @ Sd
        states.append(st.copy())
    dx = np.zeros_like(x); ddt = np.zeros_like(dt); dB = np.zeros_like(B); dC = np.zeros_like(C)
    dA = 0.0
    dstate = dS_final.copy()                    # grad flowing back through the state chain
    for ic in reversed(range(nch)):
        s, e = ic * Q, ic * Q + Q
        c = cache[ic]; dy_c = dy[s:e]
        ecs, encs, dec = c['ecs'], c['encs'], c['dec']
        # incoming dstate is dS_out (grad of THIS chunk's OUTPUT state) — capture before
        # adding the Y_off (state_in) term, since Y_off uses state_in not state_out.
        dS_out = dstate
        # y = ecs*P + ecs*G
        d_ecs = np.sum(dy_c * c['P'], 1, keepdims=True) + np.sum(dy_c * c['G'], 1, keepdims=True)
        dP = dy_c * ecs
        dG = dy_c * ecs
        # P = CB@Xs
        dCB = dP @ c['Xs'].T
        dXs = c['CB'].T @ dP
        # G = C_c@state_in  -> contributes to dC and to state_in grad
        dC[s:e] += dG @ c['st'].T
        dstate_in = c['C_c'].T @ dG                    # Y_off term of d(state_in)
        # CB = (C@B^T)*causal
        dM = dCB * causal
        dC[s:e] += dM @ c['B_c']
        dB[s:e] += dM.T @ c['C_c']
        # Xs = dtx*encs
        d_dtx = dXs * encs
        d_encs = np.sum(dXs * c['dtx'], 1, keepdims=True)
        # state_out = exp(csl)*state_in + B^T@Sd
        d_exp_csl = np.sum(dS_out * c['st'])
        dstate_in = dstate_in + np.exp(c['csl']) * dS_out   # carry term of d(state_in)
        dB[s:e] += c['Sd'] @ dS_out.T
        dSd = c['B_c'] @ dS_out
        d_dtx = d_dtx + dSd * dec
        d_dec = np.sum(dSd * c['dtx'], 1, keepdims=True)
        # exp chain -> cs
        d_cs = d_ecs * ecs + d_encs * (-encs) + d_dec * (-dec)
        d_csl = float(np.sum(d_dec * dec)) + d_exp_csl * float(np.exp(c['csl']).item())
        d_cs[-1] += d_csl                              # csl = cs[-1]
        # cs = cumsum(dtA) -> reverse cumsum
        d_dtA = tri_ge @ d_cs
        # dtA = dt*A
        ddt[s:e] += d_dtA * A
        dA += float(np.sum(d_dtA * c['dt_c']))
        # dtx = dt*x
        ddt[s:e] += np.sum(d_dtx * c['x_c'], 1, keepdims=True)
        dx[s:e] += d_dtx * c['dt_c']
        dstate = dstate_in                              # d(state_in) -> previous chunk's dS_out
    return dict(dx=dx, ddt=ddt, dA=dA, dB=dB, dC=dC, dstate0=dstate)


def cos(a, b):
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def run(L, n, p, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((L, p)); dt = np.abs(rng.standard_normal((L, 1))) * 0.1
    A = -abs(float(rng.standard_normal())); B = rng.standard_normal((L, n)); C = rng.standard_normal((L, n))
    st0 = rng.standard_normal((n, p)) * 0.1
    tt = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)
    xt, dtt, At, Bt, Ct, st0t = tt(x), tt(dt), torch.tensor(A, dtype=torch.float64, requires_grad=True), tt(B), tt(C), tt(st0)
    y, sf = ssd_fwd_torch(xt, dtt, At, Bt, Ct, st0t)
    dy = rng.standard_normal((L, p)); dS = rng.standard_normal((n, p))
    (torch.tensor(dy) * y).sum().backward(retain_graph=True)
    g_auto = dict(dx=xt.grad.clone(), ddt=dtt.grad.clone(), dA=At.grad.clone(), dB=Bt.grad.clone(),
                  dC=Ct.grad.clone(), dstate0=st0t.grad.clone())
    for t in (xt, dtt, At, Bt, Ct, st0t): t.grad = None
    y2, sf2 = ssd_fwd_torch(xt, dtt, At, Bt, Ct, st0t)
    ((torch.tensor(dy) * y2).sum() + (torch.tensor(dS) * sf2).sum()).backward()  # include dS_final
    g_full = dict(dx=xt.grad, ddt=dtt.grad, dA=At.grad, dB=Bt.grad, dC=Ct.grad, dstate0=st0t.grad)
    g_hand = ssd_bwd_hand(x, dt, A, B, C, st0, dy, dS)
    print(f"[L={L} n={n} p={p}]")
    for k in ('dx', 'ddt', 'dA', 'dB', 'dC', 'dstate0'):
        print(f"   {k:8s} cos={cos(g_hand[k], g_full[k].numpy()):.8f}  "
              f"maxerr={np.abs(np.asarray(g_hand[k]) - g_full[k].numpy()).max():.2e}")


if __name__ == "__main__":
    print("=== Mamba-2 SSD backward: hand VJP vs autograd (fp64) ===")
    run(128, 64, 128, 0)
    run(256, 64, 128, 1)
    run(256, 128, 64, 2)
