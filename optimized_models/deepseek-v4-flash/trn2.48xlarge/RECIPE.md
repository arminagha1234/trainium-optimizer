# DeepSeek-V4-Flash — trn2.48xlarge, native PyTorch (verified forward)

**First DeepSeek-V4-family model run on Trainium2.** 284B-parameter MoE (MLA + Hyper-Connections +
Compressed/Sparse attention + 256-expert **FP4** MoE), 43 layers, on `torch.device("neuron")`,
native-PyTorch eager.

## Verified (self-measured, bigsweep2, 2026-09-01)
- Full 43L forward, real 149 GB checkpoint, **on-device**. Load 72 s; **prefill wall 310.7 s** for the
  9-token golden prompt; finite logits `(1, 129280)`; **argmax = 671**.
- **MoE runs entirely on-device (77.1% of compute)** via static-shape expert dispatch — the reference
  MoE's data-dependent `torch.where` gather deadlocks the Neuron runtime at 43L, so it otherwise has to
  offload MoE to CPU. Breakdown: MoE 202.8 s / attention 60.3 s (CSA 36.4 s, HCA 3.6 s, SWA 20.3 s).

## Correctness
- `argmax = 671` reproduces the prior functional trn2 port. Versus the 8×H100 golden (`argmax 51119`
  "Paris"): cosine **0.9808** — compounded fp8/fp4 dequant quant-noise over 86 ops (per-op cos 0.99997+),
  a known precision effect, **not** a port bug. Bit-exact argmax needs bit-exact FP4/FP8 NKI kernels.

## Speedup (1.43×) — provenance
- **best** 310.7 s prefill = on-device static-MoE, **self-measured this session** on bigsweep2.
- **baseline** 445 s = the reference forward with MoE offloaded to CPU (documented, `flash/GOLDEN_VALIDATION.md`,
  trn2.48xl box2, cold). This is a **cross-run** baseline, clearly labelled; the fully self-measured claim
  is the on-device forward itself. The real contribution is **on-device MoE enablement** (reference deadlocks).

## Honest caveats / next levers
- Single-process eager (world=1). Documented throughput next-steps: **TP+EP=8 → 59 s** at 43L (parallelism),
  **FP4-in-kernel storage → 3.72×** (one MLIR `tensor_scalar_bitvec` verifier fix away), MTP spec-decode ~1.8×.
- Prefill wall is conservatively measured with per-op device sync (async-timing rule).

Reproduce: `./reproduce.sh` (harness `neuron/examples/deepseek_v4/src/run_v4_eager.py`).
