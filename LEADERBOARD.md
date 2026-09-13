# Trainium Optimizer — Leaderboard

**Peak measured throughput per model on real Trainium hardware (`native-pytorch-beta3`), and the config that achieved it.** One row per model per hardware target, ranked by throughput (tok/s). Auto-published by the optimizer loop — do not edit by hand.

Every row is generated from a `recipe.json` in an existing bundle: no bundle, no row, and no number that is not read straight out of that file. Rows are never hand-added — a hand-added row is removed on the next publish. Recipes and trajectory charts live under [`optimized_models/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and `optimization_timeline.png`.

## Peak throughput

| Rank | Model | Family | Params | Peak (tok/s) | Config | Hardware | Verified | Recipe |
|-----:|:------|:-------|-------:|-------------:|:-------|:-------------|:---------|:-------|
| 🥇 | gpt2 | gpt2 | — | **156,974** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2/) |
| 🥈 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | **105,160** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b-instruct/trn2.48xlarge/) |
| 🥉 | Qwen3-0.6B | qwen3 | 0.6B | **99,743** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-0-6b/trn2.48xlarge/) |
| 4 | SmolLM2-135M-Instruct | smollm2 | — | **99,468** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/smollm2-135m-instruct/trn2.48xlarge/) |
| 5 | OLMo-1B-0724-hf | olmo | 1B | **87,175** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/olmo-1b-0724-hf/) |
| 6 | Qwen3-0.6B | qwen3 | 0.6B | **85,937** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-0-6b/) |
| 7 | deepseek-coder-1.3b-instruct | deepseek | 1.3B | **81,574** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-coder-1-3b-instruct/) |
| 8 | Qwen2-0.5B-Instruct | qwen2 | 0.5B | **74,507** | TP=2, torch.compile(neuron), bf16, batch=8, DP=32 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-0-5b-instruct/trn2.48xlarge/) |
| 9 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | **74,449** | TP=2, torch.compile(neuron), bf16, batch=8, DP=32 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/trn2.48xlarge/) |
| 10 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | **74,269** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/) |
| 11 | Qwen3-1.7B | qwen3 | 1.7B | **62,363** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/trn2.48xlarge/) |
| 12 | gpt2-medium | gpt2 | — | **61,158** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-medium/) |
| 13 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | **59,241** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-1-5b-instruct/) |
| 14 | granite-3.1-2b-instruct | granite | 2B | **56,698** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/granite-3-1-2b-instruct/trn2.48xlarge/) |
| 15 | Qwen3-1.7B | qwen3 | 1.7B | **51,278** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/) |
| 16 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | **50,804** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-1-5b-instruct/trn2.48xlarge/) |
| 17 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | **50,650** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b-instruct/) |
| 18 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | **49,880** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/trn2.48xlarge/) |
| 19 | bloom-560m | bloom | — | **48,320** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-560m/) |
| 20 | SmolLM2-360M-Instruct | smollm2 | — | **48,203** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m-instruct/trn2.48xlarge/) |
| 21 | TinyLlama-1.1B-Chat-v1.0 | tinyllama | 1.1B | **48,108** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/tinyllama-1-1b-chat-v1-0/) |
| 22 | SmolLM2-360M-Instruct | smollm2 | — | **48,064** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m-instruct/) |
| 23 | TinyLlama-1.1B-Chat-v1.0 | tinyllama | 1.1B | **46,899** | TP=2, torch.compile(neuron), bf16, batch=32, DP=32 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/tinyllama-1-1b-chat-v1-0/trn2.48xlarge/) |
| 24 | Yi-1.5-6B-Chat | yi | 6B | **45,042** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/yi-1-5-6b-chat/trn2.48xlarge/) |
| 25 | granite-3.1-2b-instruct | granite | 2B | **38,708** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/granite-3-1-2b-instruct/) |
| 26 | Qwen2.5-Coder-1.5B | qwen2.5 | 1.5B | **38,369** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-1-5b/) |
| 27 | Qwen3-4B | qwen3 | 4B | **36,938** | TP=32, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/trn2.48xlarge/) |
| 28 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | **35,343** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/) |
| 29 | gemma-2-2b | gemma | 2B | **34,051** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gemma-2-2b/) |
| 30 | Yi-1.5-9B-Chat | yi | 9B | **32,482** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/yi-1-5-9b-chat/trn2.48xlarge/) |
| 31 | bloomz-1b1 | bloomz | — | **32,110** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/bloomz-1b1/trn2.48xlarge/) |
| 32 | granite-3.1-8b-instruct | granite | 8B | **31,787** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/granite-3-1-8b-instruct/trn2.48xlarge/) |
| 33 | SmolLM2-360M | smollm2 | — | **31,764** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m/) |
| 34 | gpt2-large | gpt2 | — | **30,620** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-large/) |
| 35 | Qwen3-4B | qwen3 | 4B | **26,548** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/) |
| 36 | Qwen3-8B | qwen3 | 8B | **26,166** | TP=32, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/trn2.48xlarge/) |
| 37 | bloom-1b7 | bloom | — | **24,852** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-1b7/) |
| 38 | Falcon3-7B-Instruct | falcon3 | 7B | **23,471** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/falcon3-7b-instruct/trn2.48xlarge/) |
| 39 | Mistral-7B-Instruct-v0.3 | mistral | 7B | **23,270** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/mistral-7b-instruct-v0-3/) |
| 40 | opt-1.3b | opt | 1.3B | **21,077** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-1-3b/) |
| 41 | deepseek-llm-7b-base | deepseek | 7B | **20,702** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-llm-7b-base/) |
| 42 | Qwen3.5-4B | qwen3.5 | 4B | **20,470** | TP=16, torch.compile(neuron), bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-4b/trn2.48xlarge/) |
| 43 | Qwen2.5-Coder-7B | qwen2.5 | 7B | **19,866** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-7b/) |
| 44 | Qwen2.5-Math-7B | qwen2.5 | 7B | **19,826** | torch.compile(neuron) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-math-7b/) |
| 45 | Qwen2.5-Coder-7B-Instruct | qwen2.5 | 7B | **19,532** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-7b-instruct/trn2.48xlarge/) |
| 46 | DeepSeek-R1-Distill-Qwen-7B | deepseek | 7B | **19,505** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/deepseek-r1-distill-qwen-7b/trn2.48xlarge/) |
| 47 | Qwen2-7B-Instruct | qwen2 | 7B | **19,503** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-7b-instruct/trn2.48xlarge/) |
| 48 | Falcon3-10B-Instruct | falcon3 | 10B | **17,331** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/falcon3-10b-instruct/trn2.48xlarge/) |
| 49 | Qwen3-8B | qwen3 | 8B | **16,876** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/) |
| 50 | SmolLM2-1.7B | smollm2 | 1.7B | **14,460** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b/) |
| 51 | stablelm-2-1_6b-chat | stablelm | 6B | **14,390** | TP=1, bf16, batch=8, DP=64 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/stablelm-2-1-6b-chat/trn2.48xlarge/) |
| 52 | stablelm-2-1_6b | stablelm | 6B | **14,183** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-2-1-6b/) |
| 53 | pythia-1.4b | pythia | 1.4B | **13,430** | TP=1, bf16, batch=8, DP=64 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/pythia-1-4b/trn2.48xlarge/) |
| 54 | pythia-1.4b | pythia | 1.4B | **13,294** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-1-4b/) |
| 55 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | **13,269** | TP=8, torch.compile(neuron), bf16, batch=8, DP=8 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-14b-instruct/trn2.48xlarge/) |
| 56 | opt-2.7b | opt | 2.7B | **11,649** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-2-7b/) |
| 57 | Qwen3-14B | qwen3 | 14B | **10,343** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-14b/) |
| 58 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | **10,256** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-14b-instruct/) |
| 59 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | **9,976** | TP=4, bf16, batch=8, DP=16 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/trn2.48xlarge/) |
| 60 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | **9,870** | TP=4, bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/) |
| 61 | phi-2 | phi | — | **9,841** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/phi-2/) |
| 62 | RedPajama-INCITE-Instruct-3B-v1 | redpajama | 3B | **9,824** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/redpajama-incite-instruct-3b-v1/) |
| 63 | QwQ-32B | qwq | 32B | **8,078** | TP=8, torch.compile(neuron), bf16, batch=8, DP=8 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwq-32b/trn2.48xlarge/) |
| 64 | Qwen2.5-32B-Instruct | qwen2.5 | 32B | **8,049** | TP=8, torch.compile(neuron), bf16, batch=8, DP=8 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-32b-instruct/trn2.48xlarge/) |
| 65 | stablelm-zephyr-3b | stablelm | 3B | **7,743** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/stablelm-zephyr-3b/trn2.48xlarge/) |
| 66 | stablelm-3b-4e1t | stablelm | 3B | **7,721** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-3b-4e1t/) |
| 67 | gemma-4-E4B-it | gemma | 4B | **7,451** | TP=2, torch.compile(neuron), bf16, batch=1, DP=32 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/gemma-4-e4b-it/trn2.48xlarge/) |
| 68 | pythia-2.8b | pythia | 2.8B | **6,450** | (config-only) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-2-8b/) |
| 69 | Qwen3-32B | qwen3 | 32B | **3,698** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-32b/) |
| 70 | Qwen3.5-35B-A3B | qwen3.5 | 35B | **2,695** | TP=16, bf16, batch=8 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-35b-a3b/trn2.48xlarge/) |
| 71 | Qwen3.5-0.8B | qwen3.5 | 0.8B | **1,143** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-0-8b/trn2.48xlarge/) |
| 72 | Qwen3.5-2B | qwen3.5 | 2B | **1,129** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-5-2b/trn2.48xlarge/) |
| 73 | Qwen3.8-27B | qwen3.8 | 27B | **343** | TP=8, bf16, batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8-27b/trn2.48xlarge/) |
| 74 | DeepSeek-V4-Flash | deepseek | 284B | **0.29** | TP=1, EP=64, bf16-resident (all Linears dequant-once), batch=1 | trn2.48xlarge | ✅ verified | [recipe](./optimized_models/deepseek-v4-flash/trn2.48xlarge/) |

74 verified result(s) across 60 model(s) and 2 hardware target(s). Throughput is the prefill tok/s measured on real hardware at the recipe's probe shape. Absolute throughput is comparable across rows on the same hardware target.

## Improvement over eager baseline

The same verified results, ranked by speedup over the **eager** baseline on the same instance and probe shape. A speedup is internally consistent per row but **not comparable across hardware** — a bigger box can score a lower multiple. This shows how far the optimizer moved each model; the peak-throughput table above is the headline.

| Model | Params | Baseline (tok/s) | Best (tok/s) | Speedup | Config | Hardware |
|:------|-------:|-----------------:|-------------:|--------:|:-------|:-------------|
| Qwen3-0.6B | 0.6B | 3,449 | **99,743** | **28.916×** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge |
| bloomz-1b1 | — | 1,237 | **32,110** | **25.95×** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge |
| Qwen3-0.6B | 0.6B | 3,333 | **85,937** | **25.788×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| SmolLM2-1.7B-Instruct | 1.7B | 4,133 | **105,160** | **25.446×** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge |
| granite-3.1-2b-instruct | 2B | 2,346 | **56,698** | **24.171×** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge |
| Qwen3.5-4B | 4B | 939 | **20,470** | **21.795×** | TP=16, torch.compile(neuron), bf16, batch=1 | trn2.48xlarge |
| SmolLM2-135M-Instruct | — | 4,707 | **99,468** | **21.133×** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge |
| Qwen3-1.7B | 1.7B | 3,172 | **62,363** | **19.659×** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge |
| deepseek-coder-1.3b-instruct | 1.3B | 4,329 | **81,574** | **18.843×** | (config-only) | trn2.3xlarge |
| granite-3.1-2b-instruct | 2B | 2,097 | **38,708** | **18.456×** | (config-only) | trn2.3xlarge |
| Qwen3-4B | 4B | 2,089 | **36,938** | **17.683×** | TP=32, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.48xlarge |
| Qwen3-1.7B | 1.7B | 2,975 | **51,278** | **17.239×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| gemma-2-2b | 2B | 1,996 | **34,051** | **17.061×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| Qwen2.5-3B-Instruct | 3B | 2,974 | **49,880** | **16.771×** | TP=16, torch.compile(neuron), bf16, batch=8, DP=4 | trn2.48xlarge |
| granite-3.1-8b-instruct | 8B | 2,015 | **31,787** | **15.778×** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge |
| Qwen2.5-1.5B-Instruct | 1.5B | 3,797 | **59,241** | **15.601×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Yi-1.5-6B-Chat | 6B | 2,901 | **45,042** | **15.529×** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge |
| Qwen2.5-0.5B-Instruct | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge |
| Qwen2.5-0.5B-Instruct | 0.5B | 5,220 | **74,449** | **14.263×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=32 | trn2.48xlarge |
| OLMo-1B-0724-hf | 1B | 6,164 | **87,175** | **14.142×** | (config-only) | trn2.3xlarge |
| Yi-1.5-9B-Chat | 9B | 2,297 | **32,482** | **14.14×** | TP=32, torch.compile(neuron), bf16, batch=32, DP=2 | trn2.48xlarge |
| Qwen3-4B | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| Qwen2-0.5B-Instruct | 0.5B | 5,401 | **74,507** | **13.795×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=32 | trn2.48xlarge |
| gpt2 | — | 11,729 | **156,974** | **13.383×** | (config-only) | trn2.3xlarge |
| SmolLM2-360M-Instruct | — | 3,661 | **48,064** | **13.13×** | (config-only) | trn2.3xlarge |
| SmolLM2-1.7B-Instruct | 1.7B | 3,911 | **50,650** | **12.95×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen3-8B | 8B | 2,045 | **26,166** | **12.795×** | TP=32, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.48xlarge |
| Qwen2.5-3B-Instruct | 3B | 2,856 | **35,343** | **12.375×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen2.5-1.5B-Instruct | 1.5B | 4,201 | **50,804** | **12.092×** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge |
| SmolLM2-360M-Instruct | — | 4,160 | **48,203** | **11.586×** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge |
| DeepSeek-V4-Flash | 284B | 0.03 | **0.29** | **11.38×** | TP=1, EP=64, bf16-resident (all Linears dequant-once), batch=1 | trn2.48xlarge |
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | 4,355 | **48,108** | **11.048×** | torch.compile(neuron) | trn2.3xlarge |
| gpt2-medium | — | 6,083 | **61,158** | **10.055×** | (config-only) | trn2.3xlarge |
| bloom-1b7 | — | 2,556 | **24,852** | **9.723×** | (config-only) | trn2.3xlarge |
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | 4,842 | **46,899** | **9.685×** | TP=2, torch.compile(neuron), bf16, batch=32, DP=32 | trn2.48xlarge |
| Qwen2.5-Coder-1.5B | 1.5B | 3,985 | **38,369** | **9.628×** | torch.compile(neuron) | trn2.3xlarge |
| Mistral-7B-Instruct-v0.3 | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge |
| Qwen3-8B | 8B | 1,903 | **16,876** | **8.87×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| deepseek-llm-7b-base | 7B | 2,781 | **20,702** | **7.443×** | (config-only) | trn2.3xlarge |
| gpt2-large | — | 4,170 | **30,620** | **7.343×** | (config-only) | trn2.3xlarge |
| Falcon3-7B-Instruct | 7B | 3,225 | **23,471** | **7.278×** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge |
| Qwen2.5-14B-Instruct | 14B | 1,889 | **13,269** | **7.024×** | TP=8, torch.compile(neuron), bf16, batch=8, DP=8 | trn2.48xlarge |
| Qwen2.5-Math-7B | 7B | 2,930 | **19,826** | **6.765×** | torch.compile(neuron) | trn2.3xlarge |
| Falcon3-10B-Instruct | 10B | 2,569 | **17,331** | **6.748×** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge |
| Qwen2.5-Coder-7B | 7B | 2,949 | **19,866** | **6.736×** | torch.compile(neuron) | trn2.3xlarge |
| Qwen2-7B-Instruct | 7B | 3,075 | **19,503** | **6.343×** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge |
| Qwen2.5-Coder-7B-Instruct | 7B | 3,085 | **19,532** | **6.332×** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 3,087 | **19,505** | **6.318×** | TP=4, torch.compile(neuron), bf16, batch=8, DP=16 | trn2.48xlarge |
| Qwen3-14B | 14B | 1,699 | **10,343** | **6.087×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge |
| Qwen3.5-35B-A3B | 35B | 453 | **2,695** | **5.946×** | TP=16, bf16, batch=8 | trn2.48xlarge |
| Qwen2.5-14B-Instruct | 14B | 1,819 | **10,256** | **5.637×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge |
| QwQ-32B | 32B | 1,494 | **8,078** | **5.407×** | TP=8, torch.compile(neuron), bf16, batch=8, DP=8 | trn2.48xlarge |
| Qwen2.5-32B-Instruct | 32B | 1,490 | **8,049** | **5.4×** | TP=8, torch.compile(neuron), bf16, batch=8, DP=8 | trn2.48xlarge |
| gemma-4-E4B-it | 4B | 1,407 | **7,451** | **5.296×** | TP=2, torch.compile(neuron), bf16, batch=1, DP=32 | trn2.48xlarge |
| Qwen3-32B | 32B | 975 | **3,698** | **3.794×** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge |
| Qwen2.5-7B-Instruct | 7B | 2,916 | **9,870** | **3.384×** | TP=4, bf16, batch=8 | trn2.3xlarge |
| Qwen2.5-7B-Instruct | 7B | 3,079 | **9,976** | **3.241×** | TP=4, bf16, batch=8, DP=16 | trn2.48xlarge |
| opt-1.3b | 1.3B | 6,859 | **21,077** | **3.073×** | torch.compile(neuron) | trn2.3xlarge |
| stablelm-2-1_6b | 6B | 4,828 | **14,183** | **2.938×** | (config-only) | trn2.3xlarge |
| RedPajama-INCITE-Instruct-3B-v1 | 3B | 3,345 | **9,824** | **2.937×** | (config-only) | trn2.3xlarge |
| SmolLM2-360M | — | 10,836 | **31,764** | **2.931×** | (config-only) | trn2.3xlarge |
| bloom-560m | — | 17,095 | **48,320** | **2.826×** | (config-only) | trn2.3xlarge |
| stablelm-2-1_6b-chat | 6B | 5,114 | **14,390** | **2.814×** | TP=1, bf16, batch=8, DP=64 | trn2.48xlarge |
| pythia-1.4b | 1.4B | 5,454 | **13,294** | **2.438×** | (config-only) | trn2.3xlarge |
| opt-2.7b | 2.7B | 4,810 | **11,649** | **2.422×** | (config-only) | trn2.3xlarge |
| pythia-1.4b | 1.4B | 5,606 | **13,430** | **2.396×** | TP=1, bf16, batch=8, DP=64 | trn2.48xlarge |
| phi-2 | — | 4,176 | **9,841** | **2.357×** | (config-only) | trn2.3xlarge |
| stablelm-3b-4e1t | 3B | 3,393 | **7,721** | **2.275×** | (config-only) | trn2.3xlarge |
| stablelm-zephyr-3b | 3B | 3,574 | **7,743** | **2.167×** | TP=1, torch.compile(neuron), bf16, batch=8, DP=64 | trn2.48xlarge |
| pythia-2.8b | 2.8B | 3,108 | **6,450** | **2.076×** | (config-only) | trn2.3xlarge |
| SmolLM2-1.7B | 1.7B | 10,573 | **14,460** | **1.368×** | (config-only) | trn2.3xlarge |
| Qwen3.5-2B | 2B | 1,071 | **1,129** | **1.054×** | TP=4, bf16, batch=1 | trn2.48xlarge |
| Qwen3.5-0.8B | 0.8B | 1,093 | **1,143** | **1.045×** | TP=4, bf16, batch=1 | trn2.48xlarge |
| Qwen3.8-27B | 27B | 332 | **343** | **1.034×** | TP=8, bf16, batch=1 | trn2.48xlarge |

See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.
