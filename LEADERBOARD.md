# Trainium Optimizer — Leaderboard

**Peak measured throughput per model on real Trainium hardware (`native-pytorch-beta3`), and the config that achieved it.** One row per model per hardware target, ranked by throughput (tok/s). Auto-published by the optimizer loop — do not edit by hand.

Every row is generated from a `recipe.json` in an existing bundle: no bundle, no row, and no number that is not read straight out of that file. Rows are never hand-added — a hand-added row is removed on the next publish. Recipes and trajectory charts live under [`optimized_models/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and `optimization_timeline.png`.

## Peak throughput

| Rank | Model | Family | Params | Peak (tok/s) | Config | Hardware | Verified | Recipe |
|-----:|:------|:-------|-------:|-------------:|:-------|:-------------|:---------|:-------|
| 🥇 | gpt2 | gpt2 | — | **156,974** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2/) |
| 🥈 | OLMo-1B-0724-hf | olmo | 1B | **87,175** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/olmo-1b-0724-hf/) |
| 🥉 | Qwen3-0.6B | qwen3 | 0.6B | **85,937** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-0-6b/) |
| 4 | deepseek-coder-1.3b-instruct | deepseek | 1.3B | **81,574** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-coder-1-3b-instruct/) |
| 5 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | **74,269** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/) |
| 6 | gpt2-medium | gpt2 | — | **61,158** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-medium/) |
| 7 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | **59,241** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-1-5b-instruct/) |
| 8 | Qwen3-1.7B | qwen3 | 1.7B | **51,278** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/) |
| 9 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | **50,650** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b-instruct/) |
| 10 | bloom-560m | bloom | — | **48,320** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-560m/) |
| 11 | TinyLlama-1.1B-Chat-v1.0 | tinyllama | 1.1B | **48,108** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/tinyllama-1-1b-chat-v1-0/) |
| 12 | SmolLM2-360M-Instruct | smollm2 | — | **48,064** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m-instruct/) |
| 13 | granite-3.1-2b-instruct | granite | 2B | **38,708** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/granite-3-1-2b-instruct/) |
| 14 | Qwen2.5-Coder-1.5B | qwen2.5 | 1.5B | **38,369** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-1-5b/) |
| 15 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | **35,343** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/) |
| 16 | gemma-2-2b | gemma | 2B | **34,051** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gemma-2-2b/) |
| 17 | SmolLM2-360M | smollm2 | — | **31,764** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m/) |
| 18 | gpt2-large | gpt2 | — | **30,620** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-large/) |
| 19 | Qwen3-4B | qwen3 | 4B | **26,548** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/) |
| 20 | bloom-1b7 | bloom | — | **24,852** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-1b7/) |
| 21 | Mistral-7B-Instruct-v0.3 | mistral | 7B | **23,270** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/mistral-7b-instruct-v0-3/) |
| 22 | opt-1.3b | opt | 1.3B | **21,077** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-1-3b/) |
| 23 | deepseek-llm-7b-base | deepseek | 7B | **20,702** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-llm-7b-base/) |
| 24 | Qwen3.5-4B | qwen3.5 | 4B | **20,470** | TP=16, torch.compile(neuron), bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-4b/trn2.48xlarge/) |
| 25 | Qwen2.5-Coder-7B | qwen2.5 | 7B | **19,866** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-7b/) |
| 26 | Qwen2.5-Math-7B | qwen2.5 | 7B | **19,826** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-math-7b/) |
| 27 | Qwen3-8B | qwen3 | 8B | **16,876** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/) |
| 28 | SmolLM2-1.7B | smollm2 | 1.7B | **14,460** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b/) |
| 29 | stablelm-2-1_6b | stablelm | 6B | **14,183** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-2-1-6b/) |
| 30 | pythia-1.4b | pythia | 1.4B | **13,294** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-1-4b/) |
| 31 | opt-2.7b | opt | 2.7B | **11,649** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-2-7b/) |
| 32 | Qwen3-14B | qwen3 | 14B | **10,343** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-14b/) |
| 33 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | **10,256** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-14b-instruct/) |
| 34 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | **9,870** | TP=4, bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/) |
| 35 | phi-2 | phi | — | **9,841** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/phi-2/) |
| 36 | RedPajama-INCITE-Instruct-3B-v1 | redpajama | 3B | **9,824** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/redpajama-incite-instruct-3b-v1/) |
| 37 | stablelm-3b-4e1t | stablelm | 3B | **7,721** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-3b-4e1t/) |
| 38 | pythia-2.8b | pythia | 2.8B | **6,450** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-2-8b/) |
| 39 | Qwen3-32B | qwen3 | 32B | **3,698** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-32b/) |
| 40 | Qwen3.5-35B-A3B | qwen3.5 | 35B | **2,695** | TP=16, bf16, batch=8 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-35b-a3b/trn2.48xlarge/) |
| 41 | Qwen3.5-0.8B | qwen3.5 | 0.8B | **1,143** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-0-8b/trn2.48xlarge/) |
| 42 | Qwen3.5-2B | qwen3.5 | 2B | **1,129** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-2b/trn2.48xlarge/) |
| 43 | Qwen3.8-27B | qwen3.8 | 27B | **343** | TP=8, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8-27b/trn2.48xlarge/) |
| 44 | DeepSeek-V4-Flash | deepseek | 284B | **0.29** | TP=1, EP=64, bf16-resident (all Linears dequant-once), batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/deepseek-v4-flash/trn2.48xlarge/) |

44 verified result(s) across 44 model(s) and 2 hardware target(s). Throughput is the prefill tok/s measured on real hardware at the recipe's probe shape. Absolute throughput is comparable across rows on the same hardware target.

## Improvement over eager baseline

The same verified results, ranked by speedup over the **eager** baseline on the same instance and probe shape. A speedup is internally consistent per row but **not comparable across hardware** — a bigger box can score a lower multiple. This shows how far the optimizer moved each model; the peak-throughput table above is the headline.

| Model | Params | Baseline (tok/s) | Best (tok/s) | Speedup | Config | Hardware |
|:------|-------:|-----------------:|-------------:|--------:|:-------|:-------------|
| Qwen3-0.6B | 0.6B | 3,333 | **85,937** | **25.788×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen3.5-4B | 4B | 939 | **20,470** | **21.795×** | TP=16, torch.compile(neuron), bf16, batch=1 | trn2.48xlarge |
| deepseek-coder-1.3b-instruct | 1.3B | 4,329 | **81,574** | **18.843×** | (config-only) | trn2.3xlarge |
| granite-3.1-2b-instruct | 2B | 2,097 | **38,708** | **18.456×** | (config-only) | trn2.3xlarge |
| Qwen3-1.7B | 1.7B | 2,975 | **51,278** | **17.239×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| gemma-2-2b | 2B | 1,996 | **34,051** | **17.061×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| Qwen2.5-1.5B-Instruct | 1.5B | 3,797 | **59,241** | **15.601×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen2.5-0.5B-Instruct | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge |
| OLMo-1B-0724-hf | 1B | 6,164 | **87,175** | **14.142×** | (config-only) | trn2.3xlarge |
| Qwen3-4B | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| gpt2 | — | 11,729 | **156,974** | **13.383×** | (config-only) | trn2.3xlarge |
| SmolLM2-360M-Instruct | — | 3,661 | **48,064** | **13.13×** | (config-only) | trn2.3xlarge |
| SmolLM2-1.7B-Instruct | 1.7B | 3,911 | **50,650** | **12.95×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen2.5-3B-Instruct | 3B | 2,856 | **35,343** | **12.375×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| DeepSeek-V4-Flash | 284B | 0.03 | **0.29** | **11.38×** | TP=1, EP=64, bf16-resident (all Linears dequant-once), batch=1 | trn2.48xlarge |
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | 4,355 | **48,108** | **11.048×** | torch.compile(neuron) | trn2.3xlarge |
| gpt2-medium | — | 6,083 | **61,158** | **10.055×** | (config-only) | trn2.3xlarge |
| bloom-1b7 | — | 2,556 | **24,852** | **9.723×** | (config-only) | trn2.3xlarge |
| Qwen2.5-Coder-1.5B | 1.5B | 3,985 | **38,369** | **9.628×** | torch.compile(neuron) | trn2.3xlarge |
| Mistral-7B-Instruct-v0.3 | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge |
| Qwen3-8B | 8B | 1,903 | **16,876** | **8.87×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| deepseek-llm-7b-base | 7B | 2,781 | **20,702** | **7.443×** | (config-only) | trn2.3xlarge |
| gpt2-large | — | 4,170 | **30,620** | **7.343×** | (config-only) | trn2.3xlarge |
| Qwen2.5-Math-7B | 7B | 2,930 | **19,826** | **6.765×** | torch.compile(neuron) | trn2.3xlarge |
| Qwen2.5-Coder-7B | 7B | 2,949 | **19,866** | **6.736×** | torch.compile(neuron) | trn2.3xlarge |
| Qwen3-14B | 14B | 1,699 | **10,343** | **6.087×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen3.5-35B-A3B | 35B | 453 | **2,695** | **5.946×** | TP=16, bf16, batch=8 | trn2.48xlarge |
| Qwen2.5-14B-Instruct | 14B | 1,819 | **10,256** | **5.637×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| Qwen3-32B | 32B | 975 | **3,698** | **3.794×** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge |
| Qwen2.5-7B-Instruct | 7B | 2,916 | **9,870** | **3.384×** | TP=4, bf16, batch=8 | trn2.3xlarge |
| opt-1.3b | 1.3B | 6,859 | **21,077** | **3.073×** | torch.compile(neuron) | trn2.3xlarge |
| stablelm-2-1_6b | 6B | 4,828 | **14,183** | **2.938×** | (config-only) | trn2.3xlarge |
| RedPajama-INCITE-Instruct-3B-v1 | 3B | 3,345 | **9,824** | **2.937×** | (config-only) | trn2.3xlarge |
| SmolLM2-360M | — | 10,836 | **31,764** | **2.931×** | (config-only) | trn2.3xlarge |
| bloom-560m | — | 17,095 | **48,320** | **2.826×** | (config-only) | trn2.3xlarge |
| pythia-1.4b | 1.4B | 5,454 | **13,294** | **2.438×** | (config-only) | trn2.3xlarge |
| opt-2.7b | 2.7B | 4,810 | **11,649** | **2.422×** | (config-only) | trn2.3xlarge |
| phi-2 | — | 4,176 | **9,841** | **2.357×** | (config-only) | trn2.3xlarge |
| stablelm-3b-4e1t | 3B | 3,393 | **7,721** | **2.275×** | (config-only) | trn2.3xlarge |
| pythia-2.8b | 2.8B | 3,108 | **6,450** | **2.076×** | (config-only) | trn2.3xlarge |
| SmolLM2-1.7B | 1.7B | 10,573 | **14,460** | **1.368×** | (config-only) | trn2.3xlarge |
| Qwen3.5-2B | 2B | 1,071 | **1,129** | **1.054×** | TP=4, bf16, batch=1 | trn2.48xlarge |
| Qwen3.5-0.8B | 0.8B | 1,093 | **1,143** | **1.045×** | TP=4, bf16, batch=1 | trn2.48xlarge |
| Qwen3.8-27B | 27B | 332 | **343** | **1.034×** | TP=8, bf16, batch=1 | trn2.48xlarge |

See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.
