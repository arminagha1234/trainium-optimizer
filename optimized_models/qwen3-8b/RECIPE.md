# Optimized Recipe: Qwen/Qwen3-8B

**16,876 tok/s** — 8.87x over baseline
(1,903 tok/s).

Backend: `native-pytorch-beta3`  ·  Correctness: **verified** (trusted-grader re-measure)
Generated: 2026-08-22T07:56:30Z

## Winning config

```json
{
  "tp_degree": 4,
  "weights_dtype": "bf16",
  "attn_implementation": "eager",
  "compile_mode": "compile-default",
  "batch": 8,
  "track": "latency",
  "dp_degree": 1,
  "cp_degree": 2,
  "batching": "static",
  "kv_replication": 1,
  "cores_used": 8,
  "cores_available": 4
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
  "instance_type": "trn2.3xlarge",
  "torch": "2.12.1+cpu",
  "torch_neuronx": "2.12.3.0.0+unknown.dev",
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
