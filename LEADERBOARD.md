# Trainium Optimizer — Leaderboard

Current standings, one row per model (latest cycle). See [`HISTORY.tsv`](./HISTORY.tsv) for the full append-only record, and [`optimized_models/<model>/`](./optimized_models/) for the per-model recipe bundle (recipe.json + RECIPE.md + reproduce.sh + results.tsv).

| Model | Cycle | Baseline (tok/s) | Best (tok/s) | Speedup | Correctness | Verified | Win stage |
|-------|------:|-----------------:|-------------:|--------:|------------:|:---------|:----------|
| qwen3-0-6b | 1 | 3,085 | **83,450** | **27.05×** | 93.8% | ✅ verified | config |

Backend: `native-pytorch-beta3` · Neuron SDK **2.28.0** · verified on-device.

**How to reproduce a row:** the recipe bundle in `optimized_models/<model>/` contains a self-contained `reproduce.sh` — set your `HF_TOKEN`, run it on a matching Trainium instance, and it will re-measure both baseline and best config with the same probe shape and correctness gate.

