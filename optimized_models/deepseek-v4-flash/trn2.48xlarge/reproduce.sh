#!/bin/bash
# Reproduce the DeepSeek-V4-Flash 43L on-device forward (trn2.48xlarge, native PyTorch).
# Harness: neuron/examples/deepseek_v4/src/run_v4_eager.py (static-shape on-device MoE).
set -euo pipefail
SNAP=/ustore/fsx/team_shared_rw/hf_cache_shared/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1
printf '%s' '{"prompt_token_ids":[0,128803,671,6102,294,8760,344,128804,128822]}' > /tmp/prompt.json
export V4_CKPT="$SNAP"
cd neuron/examples/deepseek_v4/src
NEURON_RT_NUM_CORES=1 python3 run_v4_eager.py --layers 43 --measure \
  --prompt_json /tmp/prompt.json --device neuron
# Expect: [prefill] finite=True argmax=671 wall~=311s ; MoE ~77% on-device ; no deadlock.
