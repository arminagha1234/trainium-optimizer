#!/usr/bin/env bash
# Reproduce the optimized Qwen/Qwen3-0.6B recipe.
# Backend: vllm-serve
# Expected: 164 tok/s (4.41x baseline)
# Toolchain at publish time: {"neuronxcc": "2.27.5334", "vllm": "0.24.0", "instance": "trn2.3xlarge"}
set -euo pipefail

# See ENVIRONMENT.md for backend setup (DLC pull, driver, venv).
python -m optimizer.apply \
    --model Qwen/Qwen3-0.6B \
    --backend vllm-serve \
    --set tp_degree=2 --set weights_dtype=bf16 --set backend=vllm-serve

# Then measure to confirm you land within tolerance of 164 tok/s.
python -m optimizer.measure --model Qwen/Qwen3-0.6B --backend vllm-serve --all-shapes
