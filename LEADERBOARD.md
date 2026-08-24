# Trainium Optimizer — Leaderboard

Verified optimized models, one row per model, sorted by speedup over the eager baseline on real Trainium hardware (`native-pytorch-beta3`). Auto-published by the optimizer loop — do not edit by hand.

Recipes and trajectory charts live under [`optimized_models/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and `optimization_timeline.png`.

| Rank | Model | Family | Params | Baseline (tok/s) | Best (tok/s) | Speedup | Best config | Hardware | Verified | Recipe |
|-----:|:------|:-------|-------:|-----------------:|-------------:|--------:|:------------|:-------------|:---------|:-------|
| 🥇 | Qwen3-0.6B | qwen3 | 0.6B | 3,179 | **87,017** | **27.374×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-0-6b/) |
| 🥈 | Qwen3-1.7B | qwen3 | 1.7B | 2,770 | **51,151** | **18.469×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/) |
| 🥉 | Gemma-2-2B | gemma-2 | 2B | 1,996 | **34,051** | **17.061×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gemma-2-2b/) |
| 4 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | 3,797 | **59,241** | **15.601×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-1-5b-instruct/) |
| 5 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/) |
| 6 | Qwen3-4B | qwen3 | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/) |
| 7 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | 3,911 | **50,650** | **12.950×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b-instruct/) |
| 8 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | 2,754 | **35,056** | **12.729×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/) |
| 9 | Mistral-7B-Instruct-v0.3 | mistral | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/mistral-7b-instruct-v0-3/) |
| 10 | Qwen3-8B | qwen3 | 8B | 1,903 | **16,876** | **8.870×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/) |
| 11 | Qwen3-14B | qwen3 | 14B | 1,699 | **10,343** | **6.087×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-14b/) |
| 12 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | 1,819 | **10,256** | **5.637×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-14b-instruct/) |
| 13 | Qwen3-32B | qwen3 | 32B | 975 | **3,698** | **3.794×** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-32b/) |
| 14 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | 2,920 | **9,891** | **3.387×** | TP=4, bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/) |

14 verified model(s). Speedup is measured against the eager baseline on the same instance and probe shape. See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.
