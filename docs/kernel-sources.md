# Neuron / NKI Kernel Sources — harvest reference

Where to find Neuron/NKI kernels + examples to **harvest** into the optimizer (the "harvest before invent" path: register in `kernel_registry` / `$TRN_OPT_KERNEL_DIR`, learn the technique, validate on-device against a torch reference). The go-to list when a model needs a kernel the compiler can't lower.

## 1. jburtoft — HF community Neuron kernels for novel architectures (differentiator goldmine)
Browse: <https://huggingface.co/jburtoft> · kernels view: <https://huggingface.co/kernels?search=jburtoft>

| Repo | Architecture |
|---|---|
| `jburtoft/kda-neuron-kernels` | **KDA (Kimi Delta Attention)** — Kimi-K3; fwd/bwd, chunked-exact, recurrent, decode-batch, autograd |
| `jburtoft/mamba2-ssd-neuron-kernels` | Mamba-2 SSD |
| `jburtoft/mamba3-neuron-kernels` | Mamba-3 |
| `jburtoft/minimax-m3-msa-neuron-kernels` | MiniMax M3 (lightning / linear attention) |
| `jburtoft/qwen35-deltanet-neuron-kernels` | **Qwen3.5 GatedDeltaNet** (the perf-path kernel the Qwen3.5 arch-proof flagged as missing) |
| `jburtoft/fnet-neuron-kernels`, `jburtoft/gelu-erf-neuron` | FNet FFT, gelu-erf |

> Note: HF "kernels"-type repos 404 on plain `git clone https://huggingface.co/jburtoft/<r>`; model-type repos (kda, mamba2-ssd, qwen35-deltanet, fnet) clone with `GIT_LFS_SKIP_SMUDGE=1 git clone`. For kernels-hub repos use the HF kernels API / `kernels` pip package.

## 2. aws-neuron/nki-library — official AWS reference kernels
<https://github.com/aws-neuron/nki-library> (`src/nkilib_src/nkilib/`)

- **Scan / SSM:** SSD (Mamba-2, `experimental/scan/ssd.py` + `ssd_torch.py` ref), Selective-Scan (Mamba-1), Linear-Scan
- **Attention/MoE:** Attention CTE/TKG, MoE CTE/TKG, **Router Top-K** (the sort→argmax we hand-rolled for Qwen3.5 — swap in the official one), QKV, Output-Projection
- **Building blocks:** Cumsum, Depthwise-Conv1D (Mamba conv), Conv1D/3D, RoPE, RMSNorm-Quant, MLP, Cross-Entropy

## 3. aws-neuron/nki-samples — tutorials + optimization ladder
<https://github.com/aws-neuron/nki-samples>

- `tutorials/attention_fwd_performance/attention_kernels.py` — the v1→v8a attention **optimization ladder** (ISA, tiling, loop-fusion, delayed-softmax-division, ScalarE/VectorE placement, downcast-before-transpose, direct-alloc pipelining)
- matmul tutorial (PSUM-native tiling), `contributed/pipelined_attention.py`, mxfp8 matmul, fused_mamba

> Samples use the `nki` package (dst-first ISA); our stack compiles with `neuronxcc.nki` (return-form) — translate signatures, verify by on-device compile.

## 4. wafer-ai/gpu-perf-engineering-resources — perf learning path
<https://github.com/wafer-ai/gpu-perf-engineering-resources> — roofline, FlashAttention 1–4, online-softmax, speed-of-light benchmarks (KernelBench/SOL-ExecBench). Rigor standard: every perf claim needs hardware + workload + precision + baseline + correctness-method.

## 5. HF kernels hub (general) · Neuron docs
- <https://huggingface.co/kernels?sort=trending> — community kernels (Triton/CUDA/Neuron)
- <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/> — NKI programming model, perf guide, `nki.language`/`nki.isa` API, Trainium2/3 arch

## Harvest priority (differentiator = models the compiler can't lower)
1. `qwen35-deltanet` + nki-library **Router-Top-K** → complete Qwen3.5
2. `mamba2-ssd` / nki-library **SSD** → Mamba-2
3. `kda` → Kimi-K3 · 4. `mamba3`, `minimax` → the frontier
