#!/usr/bin/env bash
# Reproduce the optimized Qwen/Qwen3.5-4B recipe.
# Backend: native-pytorch-beta3
# Expected: 812 tok/s (1.37x baseline)
# Toolchain at publish time: {"backend": "native-pytorch-beta3", "stack": "native-pytorch-beta3", "device_string": "neuron", "instance_type": "trn2.48xlarge", "torch": "2.12.1+cu130", "torch_neuronx": "2.12.3.0.1636+5c472775", "neuronx_cc": "2.27.2878.0+8220f7ac", "nki": "0.6.0+30289107548.gd2d9cc57"}
set -euo pipefail

# See ENVIRONMENT.md for backend setup (DLC pull, driver, venv).
python -m optimizer.apply \
    --model Qwen/Qwen3.5-4B \
    --backend native-pytorch-beta3 \
    --set tp_degree=8 --set weights_dtype=bf16 --set attn_implementation=eager --set compile_mode=eager --set batch=1 --set cp_degree=1 --set dp_degree=1 --set kv_replication=1 --set cores_used=8 --set cores_available=64

# Then measure to confirm you land within tolerance of 812 tok/s.
python -m optimizer.measure --model Qwen/Qwen3.5-4B --backend native-pytorch-beta3 --all-shapes
