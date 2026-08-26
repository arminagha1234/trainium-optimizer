import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_chunk_bwd_nki import gdn_chunk_bwd, C, DK, DV
from gdn_chunk_bwd_explicit import chunk_step_bwd
import torch, torch_xla.core.xla_model as xm
dev = xm.xla_device()

def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def run(seed=0):
    rng = np.random.default_rng(seed)
    def l2(x): return x / np.sqrt((x*x).sum(-1, keepdims=True)+1e-6)
    q = (l2(rng.standard_normal((C, DK))) * (1/DK**0.5)).astype(np.float32)
    k = l2(rng.standard_normal((C, DK))).astype(np.float32)
    v = rng.standard_normal((C, DV)).astype(np.float32)
    g = (-np.abs(rng.standard_normal((C, 1)))*0.3).astype(np.float32)
    beta = rng.uniform(0, 1, (C, 1)).astype(np.float32)
    S = (rng.standard_normal((DK, DV))*0.1).astype(np.float32)
    do = rng.standard_normal((C, DV)).astype(np.float32)
    dSn = rng.standard_normal((DK, DV)).astype(np.float32)
    # verified torch reference (per-chunk VJP)
    ref = chunk_step_bwd(torch.tensor(q), torch.tensor(k), torch.tensor(v),
                         torch.tensor(g[:, 0]), torch.tensor(beta[:, 0]), torch.tensor(S),
                         torch.tensor(do), torch.tensor(dSn))
    ref = {kk: (vv.detach().numpy() if hasattr(vv, "detach") else vv) for kk, vv in ref.items()}
    # constants
    tril = np.tril(np.ones((C, C), np.float32), 0); strict = np.tril(np.ones((C, C), np.float32), -1)
    eye = np.eye(C, dtype=np.float32); J = np.eye(C, dtype=np.float32)[::-1].copy()
    last = np.zeros((C, 1), np.float32); last[C-1, 0] = 1.0
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    outs = gdn_chunk_bwd(t(q), t(k), t(v), t(g), t(beta), t(S), t(do), t(dSn),
                         t(tril), t(strict), t(eye), t(J), t(last))
    xm.mark_step()
    dq, dk, dv, dg, dbeta, dS_in = (o.cpu().numpy() for o in outs)
    got = {"dq": dq, "dk": dk, "dv": dv, "dg": dg[:, 0], "dbeta": dbeta[:, 0], "dS_in": dS_in}
    print("NKI per-chunk backward vs verified torch VJP:")
    for kk in ("dq", "dk", "dv", "dg", "dbeta", "dS_in"):
        r = ref[kk]; gk = got[kk].reshape(r.shape)
        print(f"  {kk}: cos={cos(gk, r):.6f}  max_abs_err={np.abs(gk-r).max():.2e}")

if __name__ == "__main__":
    print("=== NKI GDN per-chunk backward validation ===")
    run(0)
