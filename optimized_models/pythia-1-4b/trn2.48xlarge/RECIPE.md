# Optimized Recipe: EleutherAI/pythia-1.4b

**13,430 tok/s** — 2.396x over baseline
(5,606 tok/s).

Backend: `native-pytorch-beta3`  ·  Correctness: **verified** (trusted-grader re-measure)
Generated: 2026-09-12T17:43:31Z

## Winning config

```json
{
  "tp_degree": 1,
  "weights_dtype": "bf16",
  "attn_implementation": "sdpa",
  "compile_mode": "eager",
  "batch": 8,
  "cp_degree": 1,
  "dp_degree": 64,
  "kv_replication": 1,
  "cores_used": 64,
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
