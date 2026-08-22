# Trainium Optimizer — Leaderboard

Verified optimized models, one row per model, sorted by speedup over the eager baseline on real Trainium hardware (`native-pytorch-beta3`). Auto-published by the optimizer loop — do not edit by hand.

Recipes and trajectory charts live under [`optimized_models/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and `optimization_timeline.png`.

| Rank | Model | Family | Params | Baseline (tok/s) | Best (tok/s) | Speedup | Best config | Hardware | Verified | Recipe |
|-----:|:------|:-------|-------:|-----------------:|-------------:|--------:|:------------|:-------------|:---------|:-------|
| 🥇 | Qwen3-0.6B | qwen3 | 0.6B | 3,333 | **85,937** | **25.788×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-0-6b/) |
| 🥈 | Qwen3-1.7B | qwen3 | 1.7B | 2,975 | **51,278** | **17.239×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/) |
| 🥉 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/) |
| 4 | Qwen3-4B | qwen3 | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/) |
| 5 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | 2,856 | **35,343** | **12.375×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/) |
| 6 | Mistral-7B-Instruct-v0.3 | mistral | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/mistral-7b-instruct-v0-3/) |
| 7 | Qwen3-8B | qwen3 | 8B | 1,903 | **16,876** | **8.87×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/) |
| 8 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | 2,916 | **9,870** | **3.384×** | TP=4, bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/) |

8 verified model(s). Speedup is measured against the eager baseline on the same instance and probe shape. See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.
