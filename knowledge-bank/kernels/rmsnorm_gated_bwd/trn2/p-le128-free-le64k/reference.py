import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmsnorm_gated_bwd import rmsnorm_gated_bwd
import torch, torch_xla.core.xla_model as xm
dev = xm.xla_device()
def cos(a,b):
    a,b=np.asarray(a).reshape(-1).astype(np.float64),np.asarray(b).reshape(-1).astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def run(P,F,seed=0):
    rng=np.random.default_rng(seed)
    x=rng.standard_normal((P,F)).astype(np.float32); g=rng.standard_normal((P,F)).astype(np.float32)
    w=(1+0.1*rng.standard_normal((1,F))).astype(np.float32); do=rng.standard_normal((P,F)).astype(np.float32)
    # torch autograd oracle
    xt=torch.tensor(x,requires_grad=True); gt=torch.tensor(g,requires_grad=True); wt=torch.tensor(w,requires_grad=True)
    var=xt.pow(2).mean(-1,keepdim=True); xn=xt*torch.rsqrt(var+1e-6); out=wt*xn*torch.nn.functional.silu(gt)
    out.backward(torch.tensor(do))
    onesP=np.ones((1,P),np.float32)
    t=lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    dx,dw,dgate=rmsnorm_gated_bwd(t(x),t(g),t(w),t(do),t(onesP)); xm.mark_step()
    dx=dx.cpu().numpy(); dw=dw.cpu().numpy(); dgate=dgate.cpu().numpy()
    print(f"[P={P} F={F}] dx:cos={cos(dx,xt.grad.numpy()):.6f} dw:cos={cos(dw,wt.grad.numpy()):.6f} dgate:cos={cos(dgate,gt.grad.numpy()):.6f} "
          f"| err dx={np.abs(dx-xt.grad.numpy()).max():.1e} dw={np.abs(dw-wt.grad.numpy()).max():.1e} dgate={np.abs(dgate-gt.grad.numpy()).max():.1e}")
print("=== RMSNormGated backward vs torch autograd ===")
run(128,128,0); run(128,512,1); run(64,256,2)
