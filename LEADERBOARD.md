# Trainium Optimizer — Leaderboard

Current standings, one row per model (latest cycle). See [`HISTORY.tsv`](./HISTORY.tsv) for the full append-only record. Recipes and charts live under [`optimized_models/<family>/<model>/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, `optimization_timeline.png`, and `optimization_highlights.png`.

| Model | Family | Cycle | Baseline (tok/s) | Best (tok/s) | Speedup | Correctness | Verified | Win stage | Recipe |
|-------|--------|------:|-----------------:|-------------:|--------:|------------:|:---------|:----------|:-------|
| qwen3-0-6b | qwen3 | 1 | 3,085 | **83,450** | **27.05×** | 93.8% | ✅ verified | config | [recipe](./optimized_models/qwen3/qwen3-0-6b/) |

Backend: `native-pytorch-beta3` · Neuron SDK **2.28.0** · verified on-device.

**Trajectory charts** (per model): `optimization_timeline.png` shows every attempt with stage colors and provenance markers; `optimization_highlights.png` shows the kept-path staircase with stage dividers and a final Nx callout.

**How to reproduce a row:** the recipe bundle contains a self-contained `reproduce.sh` — set your `HF_TOKEN`, run it on a matching Trainium instance, and it will re-measure both baseline and best config with the same probe shape and correctness gate.

