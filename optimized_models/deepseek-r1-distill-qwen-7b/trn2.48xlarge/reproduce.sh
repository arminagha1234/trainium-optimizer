#!/usr/bin/env bash
# Reproduce the optimized deepseek-ai/DeepSeek-R1-Distill-Qwen-7B recipe.
# Backend: native-pytorch-beta3
# Expected: 19505 tok/s (6.318x baseline)
# Toolchain at publish time: {"backend": "native-pytorch-beta3", "stack": "native-pytorch-beta3", "device_string": "neuron", "instance_type": "trn2.48xlarge", "torch": "2.12.1+cu130", "torch_neuronx": "2.12.3.0.1636+5c472775", "neuronx_cc": "2.27.2878.0+8220f7ac", "nki": "0.6.0+30289107548.gd2d9cc57"}
set -euo pipefail

# See ENVIRONMENT.md for backend setup (DLC pull, driver, venv).
python -m optimizer.apply \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --backend native-pytorch-beta3 \
    --set tp_degree=4 --set weights_dtype=bf16 --set attn_implementation=eager --set compile_mode=compile-default --set batch=8 --set cp_degree=1 --set dp_degree=16 --set kv_replication=1 --set cores_used=64 --set cores_available=64

# Then measure to confirm you land within tolerance of 19505 tok/s.
python -m optimizer.measure --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --backend native-pytorch-beta3 --all-shapes
