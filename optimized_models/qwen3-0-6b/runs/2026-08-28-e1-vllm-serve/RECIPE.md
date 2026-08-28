# Optimized Recipe: Qwen/Qwen3-0.6B

**164 tok/s** — 4.41x over baseline
(37 tok/s).

Backend: `vllm-serve`  ·  Correctness: **ungraded** (trusted-grader re-measure)
Generated: 2026-08-28T00:29:35Z

## Winning config

```json
{
  "tp_degree": 2,
  "weights_dtype": "bf16",
  "backend": "vllm-serve"
}
```

## Kernels

- (none — config-only recipe)

## Toolchain (reproducibility)

```json
{
  "neuronxcc": "2.27.5334",
  "vllm": "0.24.0",
  "instance": "trn2.3xlarge"
}
```

## Reproduce

```bash
./reproduce.sh
```

See `results.tsv` for the full search trace and the trajectory chart for how
this recipe was reached.
