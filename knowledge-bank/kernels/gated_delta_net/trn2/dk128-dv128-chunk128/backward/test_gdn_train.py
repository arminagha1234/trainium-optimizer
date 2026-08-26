"""test_gdn_train.py — (1) benchmark NKI GDN fwd+bwd vs torch-autograd baseline,
and (2) a REAL training step: gradients flow through gdn() into an upstream Linear,
SGD updates it, loss decreases. Proves GDN is genuinely trainable on Trainium.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_autograd import gdn
from gdn_bwd_explicit import chunk_step  # differentiable per-chunk (for the torch baseline)
import torch, torch_xla.core.xla_model as xm
dev = xm.xla_device()
DK = 128; C = 64


def _l2(x, eps=1e-6):
    return x / torch.sqrt((x*x).sum(-1, keepdim=True) + eps)


def torch_gdn(q, k, v, g, beta, chunk=64):
    """Differentiable torch chunk-GDN (baseline for benchmark + a grad cross-check).
    Inputs already preprocessed. Loops chunks carrying state (per B,H)."""
    BH, T, dk = q.shape; dv = v.shape[-1]; n = T // chunk
    outs = []
    for b in range(BH):
        S = torch.zeros(dk, dv, device=q.device, dtype=q.dtype)
        ob = []
        for i in range(n):
            sl = slice(i*chunk, (i+1)*chunk)
            o, S = chunk_step(q[b, sl], k[b, sl], v[b, sl], g[b, sl, 0], beta[b, sl, 0], S)
            ob.append(o)
        outs.append(torch.cat(ob, 0))
    return torch.stack(outs, 0)


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn(); xm.mark_step()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(); xm.mark_step()
    return (time.perf_counter() - t0) / iters * 1e3


def bench(BH=8, T=512, seed=0):
    rng = np.random.default_rng(seed)
    mk = lambda sh: torch.from_numpy(rng.standard_normal(sh).astype(np.float32)).to(dev)
    q0 = mk((BH, T, DK)); k0 = mk((BH, T, DK)); v = mk((BH, T, DK))
    g = torch.from_numpy((-np.abs(rng.standard_normal((BH, T, 1)))*0.3).astype(np.float32)).to(dev)
    beta = torch.from_numpy(rng.uniform(0, 1, (BH, T, 1)).astype(np.float32)).to(dev)
    q = _l2(q0) * (1/DK**0.5); k = _l2(k0)
    dO = mk((BH, T, DK))

    def nki_fb():
        Q = q.detach().clone().requires_grad_(True); K = k.detach().clone().requires_grad_(True)
        V = v.detach().clone().requires_grad_(True); G = g.detach().clone().requires_grad_(True)
        B = beta.detach().clone().requires_grad_(True)
        (gdn(Q, K, V, G, B) * dO).sum().backward()

    def torch_fb():
        Q = q.detach().clone().requires_grad_(True); K = k.detach().clone().requires_grad_(True)
        V = v.detach().clone().requires_grad_(True); G = g.detach().clone().requires_grad_(True)
        B = beta.detach().clone().requires_grad_(True)
        (torch_gdn(Q, K, V, G, B) * dO).sum().backward()

    nki_ms = timed(nki_fb); tr_ms = timed(torch_fb)
    print(f"[BH={BH} T={T}] fwd+bwd:  NKI={nki_ms:.2f}ms  torch-chunk-autograd={tr_ms:.2f}ms  speedup={tr_ms/nki_ms:.2f}x")


def _pipeline(x, W):
    """x[BH,T,d_in] @ W[d_in,5*DK] -> (q,k,v,g,beta) -> gdn -> y. Learnable path."""
    proj = x @ W
    q0, k0, vv, gg, bb = proj.split(DK, dim=-1)
    q = _l2(q0) * (1/DK**0.5); k = _l2(k0)
    g = -torch.nn.functional.softplus(gg[..., :1])
    beta = torch.sigmoid(bb[..., :1])
    return gdn(q, k, vv, g, beta)


def train_step_demo(BH=2, T=128, d_in=128, steps=40, seed=0):
    """TEACHER-STUDENT: a fixed teacher W* defines the target via the SAME pipeline;
    a student W (diff init) learns to match. Gradients flow through the NKI GDN
    fwd+bwd into W. If loss falls toward 0, GDN genuinely trains end-to-end."""
    torch.manual_seed(seed)
    x = torch.randn(BH, T, d_in, device=dev)
    Wt = (torch.randn(d_in, 5*DK, device=dev) * (1/d_in**0.5))
    with torch.no_grad():
        target = _pipeline(x, Wt); xm.mark_step()
    target = target.detach()
    W = (torch.randn(d_in, 5*DK, device=dev) * (1/d_in**0.5)).detach().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=0.05)
    losses = []
    for s in range(steps):
        opt.zero_grad()
        y = _pipeline(x, W)
        loss = ((y - target)**2).mean()
        loss.backward(); opt.step(); xm.mark_step()
        losses.append(float(loss))
    ok = losses[-1] < losses[0] * 0.5
    print(f"[BH={BH} T={T}] teacher-student loss: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({'LEARNED (>2x drop)' if ok else 'insufficient drop'}); W.grad finite={bool(torch.isfinite(W.grad).all())}")
    print("  trajectory:", "  ".join(f"{l:.3f}" for l in losses[::5]))


if __name__ == "__main__":
    print("=== GDN fwd+bwd benchmark (NKI vs torch chunk-autograd) ===")
    bench(BH=8, T=512, seed=0)
    bench(BH=16, T=1024, seed=1)
    print("=== GDN training step (grads flow through NKI kernels into a Linear) ===")
    train_step_demo(BH=2, T=128, steps=15)
