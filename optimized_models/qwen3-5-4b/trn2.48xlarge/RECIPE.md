# Optimized Recipe: Qwen/Qwen3.5-4B

**812 tok/s** — 1.37x over baseline
(593 tok/s).

Backend: `native-pytorch-beta3`  ·  Correctness: **verified** (trusted-grader re-measure)
Generated: 2026-08-30T16:04:17Z

## Winning config

```json
{
  "tp_degree": 8,
  "weights_dtype": "bf16",
  "attn_implementation": "eager",
  "compile_mode": "eager",
  "batch": 1,
  "cp_degree": 1,
  "dp_degree": 1,
  "kv_replication": 1,
  "cores_used": 8,
  "cores_available": 64
}
```

## Kernels

- (none — config-only recipe)

## Toolchain (reproducibility)

```json
{
  "backend": "native-pytorch-beta3",
  "stack": "native-pytorch-beta3",
  "device_string": "neuron",
  "instance_type": "trn2.48xlarge",
  "torch": "2.12.1+cu130",
  "torch_neuronx": "2.12.3.0.1636+5c472775",
  "neuronx_cc": "2.27.2878.0+8220f7ac",
  "nki": "0.6.0+30289107548.gd2d9cc57"
}
```

## Reproduce

```bash
./reproduce.sh
```

See `results.tsv` for the full search trace and the trajectory chart for how
this recipe was reached.
