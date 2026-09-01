#!/bin/bash
# Reproduce DeepSeek-V4-Flash 43L KV-cached DECODE on trn2.48xlarge (native PyTorch,
# world=64 pure expert-parallel, all-bf16-resident). Result: median decode step 3.393s
# => 0.295 tok/s, argmax=671. 11.4x over the same config with per-call fp4/fp8 dequant.
# Harness: neuron/examples/deepseek_v4/src/pure_ep_decode.py
set -euo pipefail
SNAP=/ustore/fsx/team_shared_rw/hf_cache_shared/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
printf '%s' '{"prompt_token_ids":[0,128803,671,6102,294,8760,344,128804,128822]}' > /tmp/prompt.json
cd neuron/examples/deepseek_v4/src
# Rule #4 cleanup (orphaned elastic_agent/multiprocessing wedge the runtime -> barrier fail):
pkill -9 -f torchrun 2>/dev/null || true; pkill -9 -f elastic_agent 2>/dev/null || true
pkill -9 -f multiprocessing 2>/dev/null || true; pkill -9 -f spawn_main 2>/dev/null || true; sleep 15
# world=64 pure-EP, all Linears dequant fp4/fp8 -> bf16 resident (DEQUANT_ALL=1).
NL=43 NDEC=4 EP_STATIC=1 DEQUANT_ALL=1 V4_CKPT="$SNAP" HF_HUB_OFFLINE=1 \
NEURON_RT_NUM_CORES=64 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
  torchrun --nnodes 1 --nproc_per_node=64 --rdzv_backend c10d --rdzv_endpoint localhost:29583 \
  pure_ep_decode.py
# Expect: "prefill ... argmax0=671" ; "L2_RESULT ... median_step~=3.39s AGG_TOK_S~=0.29".
# A/B baseline: set DEQUANT_ALL=0 (only routed experts bf16; attention+shared fp4/fp8 dequant
#   per-call) => median_step ~= 38.6s, AGG_TOK_S ~= 0.026. The 11.4x is this fixed-world=64 delta.
# NOTE: NEVER set FI_PROVIDER=shm (breaks the neuron collective bootstrap). Launch detached
#   (setsid nohup ... </dev/null &) if running over a transient session.
