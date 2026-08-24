# Trainium Optimizer — Leaderboard

Verified optimized models, one row per model, sorted by speedup over the eager baseline on real Trainium hardware (`native-pytorch-beta3`). Auto-published by the optimizer loop — do not edit by hand.

Recipes and trajectory charts live under [`optimized_models/`](./optimized_models/) — each folder holds `recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and `optimization_timeline.png`.

| Rank | Model | Family | Params | Baseline (tok/s) | Best (tok/s) | Speedup | Best config | Hardware | Verified | Recipe |
|-----:|:------|:-------|-------:|-----------------:|-------------:|--------:|:------------|:-------------|:---------|:-------|
| 🥇 | Qwen3-0.6B | qwen3 | 0.6B | 3,022 | **84,938** | **28.109×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-0-6b/) |
| 🥈 | Qwen3-1.7B | qwen3 | 1.7B | 2,770 | **51,151** | **18.469×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-1-7b/) |
| 🥉 | Gemma-2-2B | gemma-2 | 2B | 1,996 | **34,051** | **17.061×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gemma-2-2b/) |
| 4 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | 3,797 | **59,241** | **15.601×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-1-5b-instruct/) |
| 5 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-0-5b-instruct/) |
| 6 | Qwen3-4B | qwen3 | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-4b/) |
| 7 | gpt2 | gpt2 | 124M | 11,764 | **156,833** | **13.332×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2/) |
| 8 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | 3,911 | **50,650** | **12.950×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-1-7b-instruct/) |
| 9 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | 2,754 | **35,056** | **12.729×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-3b-instruct/) |
| 10 | SmolLM2-360M-Instruct | smollm2 | 360M | 3,954 | **48,130** | **12.171×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m-instruct/) |
| 11 | TinyLlama-1.1B-Chat-v1.0 | tinyllama | 1.1B | 4,355 | **48,108** | **11.048×** | config + compiler graph-rewrite | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/tinyllama-1-1b-chat-v1-0/) |
| 12 | gpt2-medium | gpt2 | 355M | 6,083 | **61,158** | **10.055×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-medium/) |
| 13 | Qwen2.5-Coder-1.5B | qwen2.5 | 1.5B | 4,078 | **39,159** | **9.603×** | config + compiler graph-rewrite | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-1-5b/) |
| 14 | Mistral-7B-Instruct-v0.3 | mistral | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/mistral-7b-instruct-v0-3/) |
| 15 | Qwen3-8B | qwen3 | 8B | 1,903 | **16,876** | **8.870×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-8b/) |
| 16 | bloom-1b7 | bloom | 1.7B | 2,969 | **24,926** | **8.395×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-1b7/) |
| 17 | deepseek-llm-7b-base | deepseek | 7B | 2,781 | **20,702** | **7.443×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/deepseek-llm-7b-base/) |
| 18 | gpt2-large | gpt2 | 774M | 4,170 | **30,620** | **7.343×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/gpt2-large/) |
| 19 | Qwen2.5-Math-7B | qwen2.5 | 7B | 2,930 | **19,826** | **6.765×** | config + compiler graph-rewrite | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-math-7b/) |
| 20 | Qwen2.5-Coder-7B | qwen2.5 | 7B | 2,949 | **19,866** | **6.736×** | config + compiler graph-rewrite | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-coder-7b/) |
| 21 | Qwen3-14B | qwen3 | 14B | 1,699 | **10,343** | **6.087×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-14b/) |
| 22 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | 1,819 | **10,256** | **5.637×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-14b-instruct/) |
| 23 | Qwen3-32B | qwen3 | 32B | 975 | **3,698** | **3.794×** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen3-32b/) |
| 24 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | 2,920 | **9,891** | **3.387×** | TP=4, bf16, batch=8, CP=2 | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/qwen2-5-7b-instruct/) |
| 25 | SmolLM2-360M | smollm2 | 360M | 10,836 | **31,764** | **2.931×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/smollm2-360m/) |
| 26 | opt-1.3b | opt | 1.3B | 7,396 | **21,452** | **2.900×** | config + compiler graph-rewrite | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-1-3b/) |
| 27 | stablelm-2-1_6b | stablelm | 1.6B | 4,933 | **14,141** | **2.866×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-2-1-6b/) |
| 28 | bloom-560m | bloom | 560M | 17,095 | **48,320** | **2.826×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/bloom-560m/) |
| 29 | pythia-1.4b | pythia | 1.3B | 5,454 | **13,294** | **2.438×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-1-4b/) |
| 30 | opt-2.7b | opt | 2.7B | 4,810 | **11,649** | **2.422×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/opt-2-7b/) |
| 31 | stablelm-3b-4e1t | stablelm | 3B | 3,393 | **7,721** | **2.275×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/stablelm-3b-4e1t/) |
| 32 | pythia-2.8b | pythia | 2.7B | 3,108 | **6,450** | **2.076×** | config search (native-pytorch-beta3) | trn2.3xlarge | ✅ verified | [recipe](./optimized_models/pythia-2-8b/) |

32 verified model(s). Speedup is measured against the eager baseline on the same instance and probe shape. See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.
