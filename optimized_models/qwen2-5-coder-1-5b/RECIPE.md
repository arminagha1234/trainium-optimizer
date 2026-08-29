# Optimized Recipe: Qwen/Qwen2.5-Coder-1.5B

**38,369 tok/s** — 9.628x over baseline
(3,985 tok/s).

Backend: `native-pytorch-beta3`  ·  Correctness: **verified** (trusted-grader re-measure)
Generated: 2026-08-24T16:55:59Z

## Winning config

```json
{
  "compile_mode": "compile-default",
  "graph_rewrite": "cc:optlevel3"
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
