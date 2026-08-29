#!/usr/bin/env bash
# Reproduce the optimized HuggingFaceTB/SmolLM2-1.7B-Instruct recipe.
# Backend: native-pytorch-beta3
# Expected: 50650 tok/s (12.95x baseline)
# Toolchain at publish time: {"backend": "native-pytorch-beta3", "stack": "native-pytorch-beta3", "device_string": "neuron", "instance_type": "trn2.3xlarge", "neuronx_cc_recorded_sdk": "2.28.0", "_provenance": "reconstructed from committed HISTORY.tsv + LEADERBOARD.md on 2026-08-29; the full per-stage config-search trace and exact per-run toolchain live in the source-box bundle. Numbers are the genuine recorded results."}
set -euo pipefail

# See ENVIRONMENT.md for backend setup (DLC pull, driver, venv).
python -m optimizer.apply \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --backend native-pytorch-beta3 \
    --set tp_degree=4 --set batch=8 --set weights_dtype=bf16 --set compile_mode=compile-default

# Then measure to confirm you land within tolerance of 50650 tok/s.
python -m optimizer.measure --model HuggingFaceTB/SmolLM2-1.7B-Instruct --backend native-pytorch-beta3 --all-shapes
