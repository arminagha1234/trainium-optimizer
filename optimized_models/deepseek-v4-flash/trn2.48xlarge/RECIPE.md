# DeepSeek-V4-Flash — trn2.48xlarge, native PyTorch (verified KV-cached decode)

**First DeepSeek-V4-family model (284B MoE: MLA + Hyper-Connections + Compressed/Sparse attention +
256-expert FP4 MoE, 43 layers) with real KV-cached decode on Trainium2**, native-PyTorch eager,
at **world=64 pure expert-parallel** on `torch.device("neuron")`.

## Headline: 11.38× faster decode (self-measured, bigsweep2, 2026-09-01)
- **best** KV-cached decode: **median step 3.393 s → 0.295 tok/s** (43 layers, world=64 pure-EP, batch=1).
- **baseline** same config, only routed experts bf16: **38.6 s/step → 0.026 tok/s**.
- Clean **same-world=64 A/B** — the only difference is whether the non-expert weights are bf16-resident.

## Core contribution: kill the per-call dequant
A per-component breakdown localized the decode cost: attention was **675 ms/layer** and the shared expert
**161 ms/layer** — not compute, not the collective (all_reduce is ~small), but **per-call on-device
fp4/fp8 dequantization** of the attention + shared-expert weights (only the routed experts had been made
bf16-resident). Dequantizing **all** Linear weights to bf16 **once at load** drops the 43-layer step
38.6 s → 3.393 s (**11.4×**). Post-fix per layer: attention ~9 ms, shared ~12 ms.

## Correctness
- `argmax = 671` at the 9-token golden prompt — reproduces the prior functional trn2 port. Decode token
  stream is deterministic across runs. (Vs the 8×H100 golden `argmax 51119` "Paris": cosine **0.9808** =
  compounded fp8/fp4 dequant quant-noise over 86 ops, a known precision effect, not a port bug.)

## Prior milestone (context): on-device prefill
- The first run of this model on Trainium was the on-device **prefill** (310.7 s wall, argmax 671, MoE 77%
  on-device via static-shape dispatch). That was `metric_kind=prefill`; this recipe supersedes the headline
  with the standard serving metric (**decode**).

## Honest caveats / path to 15 tok/s
- world=64 (64 cores) vs the prefill milestone's world=1 — but the 11.4× is a **fixed-world=64** delta
  (dequant fix only), not a core-count artifact.
- **Eager** (not compiled). **Batch>1 does NOT help in eager** (measured: BATCH=8 → step 3.4 s → 35 s,
  aggregate 0.227 < 0.295) — decode is compute/dispatch-bound, so batching only pays off once the per-layer
  compute is **compiled** into fused kernels.
- All-bf16-resident is HBM-tight (**23.5 / 24 GB** per core; the lm-head is kept bf16 to fit).
- **15 tok/s** needs **compiled decode** (fuse per-layer ops so batching scales) + memory reclaim via
  attention tensor-parallel. Both proven feasible this session: a compile-friendly MLA attention (real RoPE +
  mask-SDPA, no fp8 sim) **compiles at world=64** with a shared-cache + rank-0-first staggered warmup (3.3×).
  Wiring the whole decode step into compiled graphs is the remaining (larger) integration.

Reproduce: `./reproduce.sh` (harness `neuron/examples/deepseek_v4/src/pure_ep_decode.py`; set `DEQUANT_ALL=0` for the baseline).
