import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conv1d_bwd import causal_conv1d_bwd as kern, K
import torch, torch_xla.core.xla_model as xm
dev = xm.xla_device()
def cos(a,b):
    a,b=a.ravel().astype(np.float64),b.ravel().astype(np.float64); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def torch_bwd(x,w,dout):
    # oracle: depthwise causal conv1d + silu, autograd. x:[C,L], w:[C,K]
    C,L=x.shape
    xt=torch.tensor(x,requires_grad=True); wt=torch.tensor(w,requires_grad=True)
    xin=xt.unsqueeze(0)                                    # [1,C,L]
    wt3=wt.unsqueeze(1)                                    # [C,1,K]
    pre=torch.nn.functional.conv1d(xin, wt3, padding=K-1, groups=C)[:,:, :L]
    out=torch.nn.functional.silu(pre)
    out.backward(torch.tensor(dout).unsqueeze(0))
    return xt.grad.numpy(), wt.grad.numpy()
def run(C,L,seed=0):
    rng=np.random.default_rng(seed)
    x=rng.standard_normal((C,L)).astype(np.float32); w=rng.standard_normal((C,K)).astype(np.float32)
    dout=rng.standard_normal((C,L)).astype(np.float32)
    dx_ref,dw_ref=torch_bwd(x,w,dout)
    t=lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    dx,dw=kern(t(x),t(w),t(dout)); xm.mark_step(); dx=dx.cpu().numpy(); dw=dw.cpu().numpy()
    print(f"[C={C} L={L}] dx: cos={cos(dx,dx_ref):.6f} err={np.abs(dx-dx_ref).max():.2e} | dw: cos={cos(dw,dw_ref):.6f} err={np.abs(dw-dw_ref).max():.2e}")
print("=== conv1d backward vs torch autograd ===")
run(128,64,0); run(96,512,1); run(64,256,2)
