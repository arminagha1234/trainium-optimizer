#!/usr/bin/env python
"""FP8/FP4 dequantization for the DeepSeek-V4-Flash checkpoint (runbook phase P6).

The checkpoint stores two quantized formats (config quantization_config +
expert_dtype):
  * attention + shared expert + embeddings: FP8 e4m3 with 128x128 block scales in
    the ue8m0 (power-of-two, exponent-only) scale format.
  * routed experts (~97.5% of params): FP4 e2m1 with block scales.

Trainium2 native PyTorch (Beta 3) has no float8/float4 compute dtype (runbook
1.2), so quantization is a STORAGE format: dequantize each shard to BF16 at load
time, after slicing (never a full host materialization). These helpers model the
representable grids and block-scaled dequant so the P7 loader's numerics can be
validated by round-trip before touching the 160 GB checkpoint.
"""
import torch

# e4m3: 1 sign, 4 exp (bias 7), 3 mantissa. Max normal 448. (No inf; 0x7f = NaN.)
# e2m1 (fp4): 1 sign, 2 exp (bias 1), 1 mantissa. Representable |x|: {0,.5,1,1.5,2,3,4,6}.
_FP4_E2M1_ABS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _e4m3_abs_grid():
    """All positive representable magnitudes of e4m3 (normals + subnormals)."""
    vals = {0.0}
    for e in range(0, 16):            # stored exponent 0..15
        for m in range(0, 8):
            if e == 0:                # subnormals: 2^(1-7) * (m/8)
                v = (2.0 ** (1 - 7)) * (m / 8.0)
            else:                     # normals: 2^(e-7) * (1 + m/8)
                v = (2.0 ** (e - 7)) * (1 + m / 8.0)
            if not (e == 15 and m == 7):   # 0x7f reserved (NaN)
                vals.add(v)
    return torch.tensor(sorted(vals))


_E4M3_ABS = _e4m3_abs_grid()
_FMT = {"e4m3": (_E4M3_ABS, 448.0), "fp4": (_FP4_E2M1_ABS, 6.0)}


def _quantize_to_grid(x, grid):
    """Round |x| to the nearest representable magnitude on `grid`, keep sign."""
    s = torch.sign(x)
    a = x.abs().unsqueeze(-1)
    idx = (a - grid).abs().argmin(dim=-1)
    return s * grid[idx]


def quantize_blockwise(w, fmt, block=128, ue8m0=True):
    """Quantize a 2-D weight to `fmt` with per-[block,block] scales.

    Returns (codes_bf16, scales): `codes_bf16` are the on-grid values in [-fmax,
    fmax] (a stand-in for the packed byte payload), `scales` the per-block scale.
    dequantize(codes, scales) reconstructs the BF16 weight. For ue8m0 the scale is
    rounded up to a power of two (exponent-only), matching the checkpoint.
    """
    grid, fmax = _FMT[fmt]
    grid = grid.to(w.dtype)
    R, C = w.shape
    nR, nC = (R + block - 1) // block, (C + block - 1) // block
    codes = torch.zeros_like(w)
    scales = torch.zeros(nR, nC, dtype=torch.float32)
    for i in range(nR):
        for j in range(nC):
            blk = w[i * block:(i + 1) * block, j * block:(j + 1) * block].float()
            amax = blk.abs().max()
            if amax == 0:
                scales[i, j] = 1.0
                continue
            scale = amax / fmax
            if ue8m0:
                scale = 2.0 ** torch.ceil(torch.log2(scale))
            scales[i, j] = scale
            q = _quantize_to_grid((blk / scale).to(w.dtype), grid) * scale
            codes[i * block:(i + 1) * block, j * block:(j + 1) * block] = q.to(w.dtype)
    return codes, scales


def dequantize_blockwise(codes, scales, block=128):
    """Inverse of quantize_blockwise (codes already carry the scale here)."""
    return codes  # codes are stored pre-scaled in this reference model


def roundtrip_metrics(w, fmt, block=128, ue8m0=True):
    """Quantize->dequantize round-trip quality. Returns {cos, rel_rms}.

    Block quantization flushes sub-grid-minimum values, so per-element MAX
    relative error is meaningless (a tiny value in a high-dynamic-range block
    gives ~1.0); cosine similarity and relative RMS are the right measures.
    """
    codes, scales = quantize_blockwise(w, fmt, block=block, ue8m0=ue8m0)
    deq = dequantize_blockwise(codes, scales, block=block).float()
    wf = w.float()
    cos = float((deq.flatten() @ wf.flatten()) / (deq.norm() * wf.norm() + 1e-12))
    rel_rms = float((deq - wf).norm() / (wf.norm() + 1e-12))
    return {"cos": cos, "rel_rms": rel_rms}


def expert_shard_range(num_experts, rank, tp):
    """Expert-parallel loader sharding: the [start, end) experts owned by `rank`.

    Each rank holds num_experts/tp experts (e.g. 256 experts at tp=64 -> 4/rank,
    ~8.9 GB/rank of routed-expert weight, matching the P0 budget). This is the
    loader side of EP; the compute side routes tokens to the owning rank.
    """
    if num_experts % tp:
        raise ValueError(f"num_experts={num_experts} not divisible by tp={tp}")
    per = num_experts // tp
    return rank * per, (rank + 1) * per


def slice_experts(stacked, rank, tp):
    """Slice a stacked expert weight [E, ...] to this rank's experts (EP)."""
    a, b = expert_shard_range(stacked.shape[0], rank, tp)
    return stacked[a:b]
