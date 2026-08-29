# Optimized Recipe: Qwen/Qwen3-32B

**3,698 tok/s** — 3.794x over baseline
(975 tok/s).

Backend: `native-pytorch-beta3`  ·  Correctness: **verified** (trusted-grader re-measure)
Generated: 2026-08-23T04:29:26Z

## Winning config

```json
{
  "tp_degree": 4,
  "cp_degree": 2,
  "batch": 1,
  "weights_dtype": "bf16",
  "compile_mode": "compile-default"
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
  "neuronx_cc_recorded_sdk": "2.28.0",
  "_provenance": "reconstructed from committed HISTORY.tsv + LEADERBOARD.md on 2026-08-29; the full per-stage config-search trace and exact per-run toolchain live in the source-box bundle. Numbers are the genuine recorded results."
}
```

## Reproduce

```bash
./reproduce.sh
```

See `results.tsv` for the full search trace and the trajectory chart for how
this recipe was reached.

---

_Provenance: reconstructed from committed HISTORY.tsv + LEADERBOARD.md on 2026-08-29; the full per-stage config-search trace and exact per-run toolchain live in the source-box bundle. Numbers are the genuine recorded results._
