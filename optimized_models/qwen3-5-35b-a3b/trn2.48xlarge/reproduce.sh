#!/bin/bash
# Reproduce Qwen3.5-35B-A3B verified throughput win on trn2.48xlarge (native-pytorch-beta3).
# Requires PR#180 (num_local_experts sizing + shard-on-read VL-tower remap + TRN_OPT_SKIP_TP).
set -euo pipefail
export HF_HOME=/ustore/fsx/team_shared_rw/hf_cache_shared HF_HUB_OFFLINE=1
export TRN_OPT_SKIP_TP=8            # tp8 is a degenerate 2-device collective on trn2.48xl; tp16 = valid 4-device ring
export TRN_OPT_SHARD_ON_READ=1      # expert-parallel + VL-tower remap loader
export TRN_OPT_GDN_MATMUL_INV=1 TORCH_NEURONX_ENABLE_HOST_CC=1
cd implementation
python -u run_overnight.py --backend native-pytorch-beta3 \
  --model Qwen/Qwen3.5-35B-A3B --family hybrid_attention_causal_lm \
  --no-preflight --in 512 --out 128 --cycles 1 --max-configs 12 \
  --out-root /tmp/art_35b --publish-repo-dir "$PWD/.."
# baseline eager tp16 batch=1 = 453 tok/s ; winner eager tp16 batch=8 = 2694 tok/s (5.95x)
