#!/usr/bin/env bash
# Reproduce Qwen/Qwen3.5-4B peak (trn2.48xlarge).
# Peak: 20,470 tok/s (TP=16, torch.compile(neuron), bf16, batch=1); 21.8x vs eager.
set -euo pipefail
NEURON_RT_NUM_CORES=16 TRN_OPT_GDN_MATMUL_INV=1 TORCH_NEURONX_ENABLE_HOST_CC=1 \
  torchrun --nproc_per_node=16 backends/neuron_worker.py \
  --model Qwen/Qwen3.5-4B --tp 16 --dtype bf16 --compile 1 --batch 1 --input-len 1024 --out out.json
