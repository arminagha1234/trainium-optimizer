# Trainium Optimizer — Leaderboard

Verified optimized models, **one row per model per hardware target**, sorted by speedup over the eager baseline on real Trainium hardware (`native-pytorch-beta3`). Auto-published by the optimizer loop — do not edit by hand.

Speedup is relative to the eager baseline **on the same instance**, so rows for different hardware are each internally consistent but are **not comparable to one another** — a bigger box can score a lower multiple.

Recipes and trajectory charts live under [`optimized_models/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and `optimization_timeline.png`.

Every row is generated from a `recipe.json` in an existing bundle: no bundle, no row, and no number that is not read straight out of that file. Rows are never hand-added — a hand-added row is removed on the next publish.

| Rank | Model | Family | Params | Baseline (tok/s) | Best (tok/s) | Speedup | Best config | Hardware | Verified | Recipe |
|-----:|:------|:-------|-------:|-----------------:|-------------:|--------:|:------------|:-------------|:---------|:-------|
| 🥇 | Qwen3-0.6B | qwen3 | 0.6B | 3,085 | **83,450** | **27.05×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3/qwen3-0-6b/) |
| 🥈 | deepseek-coder-1.3b-instruct | deepseek | 1.3B | 4,329 | **81,574** | **18.843×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-coder-1-3b-instruct/) |
| 🥉 | granite-3.1-2b-instruct | granite | 2B | 2,097 | **38,708** | **18.456×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/granite-3-1-2b-instruct/) |
| 4 | Qwen3-1.7B | qwen3 | 1.7B | 2,975 | **51,278** | **17.239×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/) |
| 5 | gemma-2-2b | gemma | 2B | 1,996 | **34,051** | **17.061×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gemma-2-2b/) |
| 6 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | 3,797 | **59,241** | **15.601×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-1-5b-instruct/) |
| 7 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/) |
| 8 | OLMo-1B-0724-hf | olmo | 1B | 6,164 | **87,175** | **14.142×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/olmo-1b-0724-hf/) |
| 9 | Qwen3-4B | qwen3 | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/) |
| 10 | gpt2 | gpt2 | — | 11,729 | **156,974** | **13.383×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2/) |
| 11 | SmolLM2-360M-Instruct | smollm2 | — | 3,661 | **48,064** | **13.13×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m-instruct/) |
| 12 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | 3,911 | **50,650** | **12.95×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b-instruct/) |
| 13 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | 2,856 | **35,343** | **12.375×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/) |
| 14 | TinyLlama-1.1B-Chat-v1.0 | tinyllama | 1.1B | 4,355 | **48,108** | **11.048×** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/tinyllama-1-1b-chat-v1-0/) |
| 15 | gpt2-medium | gpt2 | — | 6,083 | **61,158** | **10.055×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-medium/) |
| 16 | bloom-1b7 | bloom | — | 2,556 | **24,852** | **9.723×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-1b7/) |
| 17 | Qwen2.5-Coder-1.5B | qwen2.5 | 1.5B | 3,985 | **38,369** | **9.628×** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-1-5b/) |
| 18 | Mistral-7B-Instruct-v0.3 | mistral | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/mistral-7b-instruct-v0-3/) |
| 19 | Qwen3-8B | qwen3 | 8B | 1,903 | **16,876** | **8.87×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/) |
| 20 | deepseek-llm-7b-base | deepseek | 7B | 2,781 | **20,702** | **7.443×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-llm-7b-base/) |
| 21 | gpt2-large | gpt2 | — | 4,170 | **30,620** | **7.343×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-large/) |
| 22 | Qwen2.5-Math-7B | qwen2.5 | 7B | 2,930 | **19,826** | **6.765×** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-math-7b/) |
| 23 | Qwen2.5-Coder-7B | qwen2.5 | 7B | 2,949 | **19,866** | **6.736×** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-7b/) |
| 24 | Qwen3-14B | qwen3 | 14B | 1,699 | **10,343** | **6.087×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-14b/) |
| 25 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | 1,819 | **10,256** | **5.637×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-14b-instruct/) |
| 26 | Qwen3-32B | qwen3 | 32B | 975 | **3,698** | **3.794×** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-32b/) |
| 27 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | 2,916 | **9,870** | **3.384×** | TP=4, bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/) |
| 28 | opt-1.3b | opt | 1.3B | 6,859 | **21,077** | **3.073×** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-1-3b/) |
| 29 | stablelm-2-1_6b | stablelm | 6B | 4,828 | **14,183** | **2.938×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-2-1-6b/) |
| 30 | RedPajama-INCITE-Instruct-3B-v1 | redpajama | 3B | 3,345 | **9,824** | **2.937×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/redpajama-incite-instruct-3b-v1/) |
| 31 | SmolLM2-360M | smollm2 | — | 10,836 | **31,764** | **2.931×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m/) |
| 32 | bloom-560m | bloom | — | 17,095 | **48,320** | **2.826×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-560m/) |
| 33 | pythia-1.4b | pythia | 1.4B | 5,454 | **13,294** | **2.438×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-1-4b/) |
| 34 | opt-2.7b | opt | 2.7B | 4,810 | **11,649** | **2.422×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-2-7b/) |
| 35 | phi-2 | phi | — | 4,176 | **9,841** | **2.357×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/phi-2/) |
| 36 | stablelm-3b-4e1t | stablelm | 3B | 3,393 | **7,721** | **2.275×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-3b-4e1t/) |
| 37 | pythia-2.8b | pythia | 2.8B | 3,108 | **6,450** | **2.076×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-2-8b/) |
| 38 | SmolLM2-1.7B | smollm2 | 1.7B | 10,573 | **14,460** | **1.368×** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b/) |
| 39 | Qwen3.5-2B | qwen3.5 | 2B | 1,071 | **1,129** | **1.054×** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-2b/trn2.48xlarge/) |
| 40 | Qwen3.5-0.8B | qwen3.5 | 0.8B | 1,093 | **1,143** | **1.045×** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-0-8b/trn2.48xlarge/) |

40 verified result(s) across 40 model(s) and 2 hardware target(s). Speedup is measured against the eager baseline on the same instance and probe shape. See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.
