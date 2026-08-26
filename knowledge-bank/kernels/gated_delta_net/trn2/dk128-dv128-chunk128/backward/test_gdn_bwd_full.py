import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_bwd_full_nki import gdn_bwd_full, C, DK, DV
from gdn_bwd_explicit import explicit_backward
import torch, torch_xla.core.xla_model as xm
dev = xm.xla_device()

def cos(a, b):
    a, b = np.asarray(a).reshape(-1).astype(np.float64), np.asarray(b).reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def run(T, seed=0):
    rng = np.random.default_rng(seed)
    def l2(x): return x / np.sqrt((x*x).sum(-1, keepdims=True)+1e-6)
    q = (l2(rng.standard_normal((T, DK))) * (1/DK**0.5)).astype(np.float32)
    k = l2(rng.standard_normal((T, DK))).astype(np.float32)
    v = rng.standard_normal((T, DV)).astype(np.float32)
    g = (-np.abs(rng.standard_normal((T, 1)))*0.3).astype(np.float32)
    beta = rng.uniform(0, 1, (T, 1)).astype(np.float32)
    dO = rng.standard_normal((T, DV)).astype(np.float32)
    S0 = np.zeros((DK, DV), np.float32)
    # verified reference (full-sequence grads)
    ref = explicit_backward(torch.tensor(q), torch.tensor(k), torch.tensor(v),
                            torch.tensor(g[:, 0]), torch.tensor(beta[:, 0]), torch.tensor(dO), C=C)
    ref = {kk: vv.detach().numpy() for kk, vv in ref.items()}
    tril = np.tril(np.ones((C, C), np.float32), 0); strict = np.tril(np.ones((C, C), np.float32), -1)
    eye = np.eye(C, dtype=np.float32); triu = np.triu(np.ones((C, C), np.float32), 0)
    last = np.zeros((C, 1), np.float32); last[C-1, 0] = 1.0
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    outs = gdn_bwd_full(t(q), t(k), t(v), t(g), t(beta), t(S0), t(dO),
                        t(tril), t(strict), t(eye), t(triu), t(last))
    xm.mark_step()
    dq, dk, dv, dg, dbeta, dS0 = (o.cpu().numpy() for o in outs)
    got = {"dq": dq, "dk": dk, "dv": dv, "dg": dg[:, 0], "dbeta": dbeta[:, 0]}
    print(f"[T={T} n={T//C}] full multi-chunk backward vs verified reference:")
    for kk in ("dq", "dk", "dv", "dg", "dbeta"):
        r = ref[kk]; gk = got[kk].reshape(r.shape)
        print(f"  {kk}: cos={cos(gk, r):.6f}  max_abs_err={np.abs(gk-r).max():.2e}")

if __name__ == "__main__":
    print("=== FULL multi-chunk GDN backward validation ===")
    run(128, 0)   # n=2
    run(256, 1)   # n=4
