"""gdn_block_train.py — (b) train a REAL Qwen3.5-style GatedDeltaNet mixer block
end-to-end using our NKI gdn() op, on Trainium. Teacher-student so loss is
meaningful (O(1) magnitudes). Proves the kernel trains inside a realistic block:
gradients flow through gdn() into in_proj / out_proj / norm weights and the loss
drops toward 0.

Block: x -> in_proj -> (q,k,v,g,beta) -> depthwise causal conv1d(qkv) -> preprocess
(l2norm q,k; scale q; softplus g; sigmoid beta) -> gdn() [NKI] -> RMSNormGated(gate)
-> out_proj -> y.  conv1d/norm are differentiable torch here (their NKI kernels
exist in the library for the all-kernel path); gdn() is the NKI fwd+bwd op.
"""
import sys, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_autograd import gdn
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
DK = 128; KCONV = 4


def _l2(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def causal_conv1d_torch(x, w):  # x[BH,T,C], w[C,K] depthwise causal + silu
    BH, T, Cc = x.shape
    xt = x.transpose(1, 2)                                   # [BH,C,T]
    xt = F.conv1d(xt, w.unsqueeze(1), padding=KCONV - 1, groups=Cc)[:, :, :T]
    return F.silu(xt.transpose(1, 2))


class GDNMixer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 5 * DK, bias=False)
        self.conv_w = nn.Parameter(torch.randn(3 * DK, KCONV) * 0.1)   # conv on q,k,v
        self.norm_w = nn.Parameter(torch.ones(DK))
        self.out_proj = nn.Linear(DK, d_model, bias=False)

    def forward(self, x):
        BH, T, _ = x.shape
        proj = self.in_proj(x)
        q0, k0, v0, gg, bb = proj.split(DK, -1)
        qkv = causal_conv1d_torch(torch.cat([q0, k0, v0], -1), self.conv_w)
        q0, k0, v0 = qkv.split(DK, -1)
        q = _l2(q0) * (1 / DK ** 0.5); k = _l2(k0)
        g = -F.softplus(gg[..., :1]); beta = torch.sigmoid(bb[..., :1])
        core = gdn(q, k, v0, g, beta)                        # NKI fwd+bwd
        # RMSNormGated (norm before gate), differentiable torch
        var = core.pow(2).mean(-1, keepdim=True)
        core = core * torch.rsqrt(var + 1e-6) * self.norm_w
        core = core * F.silu(gg)                             # gate by the g-branch proj
        return self.out_proj(core)


def train(BH=2, T=128, d=128, steps=60, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(BH, T, d, device=dev)
    teacher = GDNMixer(d).to(dev)
    with torch.no_grad():
        target = teacher(x); xm.mark_step()
    target = target.detach()
    student = GDNMixer(d).to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=0.03)
    losses = []
    for s in range(steps):
        opt.zero_grad()
        y = student(x)
        loss = ((y - target) ** 2).mean() / (target.pow(2).mean() + 1e-8)  # relative MSE
        loss.backward(); opt.step(); xm.mark_step()
        losses.append(float(loss))
    grads_ok = all(torch.isfinite(p.grad).all().item() for p in student.parameters() if p.grad is not None)
    drop = losses[0] / max(losses[-1], 1e-9)
    print(f"[BH={BH} T={T}] GDN-mixer teacher-student rel-MSE: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({drop:.1f}x drop, {'LEARNED' if drop > 3 else 'weak'}); all grads finite={grads_ok}")
    print("  trajectory:", "  ".join(f"{l:.3f}" for l in losses[::8]))


if __name__ == "__main__":
    print("=== (b) train a real GDN mixer block end-to-end via NKI gdn() ===")
    train(BH=2, T=128, steps=60)
