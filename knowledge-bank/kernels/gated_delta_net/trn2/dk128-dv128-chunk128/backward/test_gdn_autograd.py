import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_bwd_batched import gdn_bwd_batched, C, DK, DV
from gdn_bwd_explicit import explicit_backward
from gdn_autograd import gdn
import torch, torch_xla.core.xla_model as xm
dev = xm.xla_device()


def cos(a, b):
    a, b = np.asarray(a).reshape(-1).astype(np.float64), np.asarray(b).reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def _prep(BH, T, seed=0):
    rng = np.random.default_rng(seed)
    def l2(x): return x / np.sqrt((x*x).sum(-1, keepdims=True)+1e-6)
    q = (l2(rng.standard_normal((BH, T, DK))) * (1/DK**0.5)).astype(np.float32)
    k = l2(rng.standard_normal((BH, T, DK))).astype(np.float32)
    v = rng.standard_normal((BH, T, DV)).astype(np.float32)
    g = (-np.abs(rng.standard_normal((BH, T, 1)))*0.3).astype(np.float32)
    beta = rng.uniform(0, 1, (BH, T, 1)).astype(np.float32)
    dO = rng.standard_normal((BH, T, DV)).astype(np.float32)
    return q, k, v, g, beta, dO


def test_batched_backward(BH, T, seed=0):
    q, k, v, g, beta, dO = _prep(BH, T, seed)
    # per-slice reference
    refs = [explicit_backward(torch.tensor(q[b]), torch.tensor(k[b]), torch.tensor(v[b]),
                              torch.tensor(g[b, :, 0]), torch.tensor(beta[b, :, 0]),
                              torch.tensor(dO[b]), C=C) for b in range(BH)]
    S0 = np.zeros((BH, DK, DV), np.float32)
    tril = np.tril(np.ones((C, C), np.float32), 0); strict = np.tril(np.ones((C, C), np.float32), -1)
    eye = np.eye(C, dtype=np.float32); triu = np.triu(np.ones((C, C), np.float32), 0)
    last = np.zeros((C, 1), np.float32); last[C-1, 0] = 1.0
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    dq, dk, dv, dg, dbeta, dS0 = gdn_bwd_batched(
        t(q), t(k), t(v), t(g), t(beta), t(S0), t(dO),
        t(tril), t(strict), t(eye), t(triu), t(last))
    xm.mark_step()
    dq = dq.cpu().numpy(); dk = dk.cpu().numpy(); dv = dv.cpu().numpy()
    dg = dg.cpu().numpy(); dbeta = dbeta.cpu().numpy()
    print(f"[BH={BH} T={T}] batched backward vs verified reference:")
    for name, arr in (("dq", dq), ("dk", dk), ("dv", dv), ("dg", dg[..., 0]), ("dbeta", dbeta[..., 0])):
        cs = [cos(arr[b], refs[b][name].detach().numpy()) for b in range(BH)]
        errs = [np.abs(arr[b] - refs[b][name].detach().numpy()).max() for b in range(BH)]
        print(f"  {name}: cos={min(cs):.6f}..{max(cs):.6f}  max_err={max(errs):.2e}")


def test_autograd_function(BH=2, T=128, seed=0):
    """The whole autograd.Function loop: forward -> scalar loss -> backward."""
    q, k, v, g, beta, dO = _prep(BH, T, seed)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).requires_grad_(True)
    Q, K, V, G, B = t(q), t(k), t(v), t(g), t(beta)
    out = gdn(Q, K, V, G, B)
    loss = (out * torch.from_numpy(dO).to(dev)).sum()
    loss.backward()
    xm.mark_step()
    # reference (per-slice)
    refs = [explicit_backward(torch.tensor(q[b]), torch.tensor(k[b]), torch.tensor(v[b]),
                              torch.tensor(g[b, :, 0]), torch.tensor(beta[b, :, 0]),
                              torch.tensor(dO[b]), C=C) for b in range(BH)]
    print(f"[BH={BH} T={T}] autograd.Function fwd->loss->backward vs verified reference:")
    for name, grad in (("dq", Q.grad), ("dk", K.grad), ("dv", V.grad),
                       ("dg", G.grad[..., 0]), ("dbeta", B.grad[..., 0])):
        arr = grad.cpu().numpy()
        cs = [cos(arr[b], refs[b][name].detach().numpy()) for b in range(BH)]
        errs = [np.abs(arr[b] - refs[b][name].detach().numpy()).max() for b in range(BH)]
        print(f"  {name}: cos={min(cs):.6f}..{max(cs):.6f}  max_err={max(errs):.2e}")


if __name__ == "__main__":
    print("=== BATCHED GDN backward ===")
    test_batched_backward(BH=2, T=128, seed=0)
    test_batched_backward(BH=4, T=256, seed=1)
    print("=== autograd.Function end-to-end ===")
    test_autograd_function(BH=2, T=128, seed=2)
