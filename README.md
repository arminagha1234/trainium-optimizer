# Autonomous Trainium Model Optimizer

Point it at any HuggingFace model → it optimizes for Trainium2, verifies
correctness, publishes a reproducible recipe, and banks every trick so the
next model is cheaper to optimize. Repeat across the top open models and
publish a Neuron-optimized leaderboard that refreshes on each SDK release.

![Qwen3-1.7B optimization trajectory](optimized_models/qwen3-1-7b/optimization_timeline.png)

*Qwen3-1.7B, optimized end-to-end on one trn2.3xlarge → 51,278 tok/s = 17.2×
the eager baseline, correctness-verified. Recipes + charts: [`optimized_models/`](./optimized_models/).*

## Quickstart

```bash
git clone https://github.com/arminagha1234/trainium-optimizer
cd trainium-optimizer && pip install -e .
python -m optimizer.run --backend mock      # hardware-free end-to-end demo (~90s)
```

Full guide: [QUICKSTART.md](./QUICKSTART.md). Real Neuron backends need the
on-device toolchain — see [implementation/ENVIRONMENT.md](./implementation/ENVIRONMENT.md).

## How it works

An outer loop optimizes each model through a cheapest-first pipeline, gated by
correctness + a %SOL check at every step:

1. **Config search** — TP / dtype / batch / `torch.compile(backend="neuron")`. Most of the speedup lands here.
2. **Harvest** — reuse a banked, on-device-validated kernel when the model exposes a known primitive (FlashAttention, GatedDeltaNet, KDA, Mamba2, …).
3. **Author** — write a new NKI kernel *only where the compiler is weak* (long-context/sparse attention, linear-attention scans). On standard ops the compiler is already ~80% of speed-of-light, so we don't.
4. **Verify & publish** — a logprob/KL correctness gate, then a reproducible recipe.
5. **Bank & compound** — every win becomes a lesson, so model N+1 is cheaper than model N.

## Leaderboard

Verified results from the autonomous loop on real Trainium (`native-pytorch-beta3`).
Full standings: [LEADERBOARD.md](./LEADERBOARD.md) · per-model recipes: [`optimized_models/`](./optimized_models/).

<!-- LEADERBOARD:START -->
| Rank | Model | Family | Params | Baseline (tok/s) | Optimized (tok/s) | Speedup | Best config | Hardware | Status |
|-----:|:------|:-------|-------:|-----------------:|------------------:|--------:|:------------|:-------------|:-------|
| 🥇 | Qwen3-0.6B | qwen3 | 0.6B | 3,085 | **83,450** | **27.05×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 🥈 | deepseek-coder-1.3b-instruct | deepseek | 1.3B | 4,329 | **81,574** | **18.843×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 🥉 | granite-3.1-2b-instruct | granite | 2B | 2,097 | **38,708** | **18.456×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 4 | Qwen3-1.7B | qwen3 | 1.7B | 2,975 | **51,278** | **17.239×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 5 | gemma-2-2b | gemma | 2B | 1,996 | **34,051** | **17.061×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ Verified |
| 6 | Qwen2.5-1.5B-Instruct | qwen2.5 | 1.5B | 3,797 | **59,241** | **15.601×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 7 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ Verified |
| 8 | OLMo-1B-0724-hf | olmo | 1B | 6,164 | **87,175** | **14.142×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 9 | Qwen3-4B | qwen3 | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ Verified |
| 10 | gpt2 | gpt2 | — | 11,729 | **156,974** | **13.383×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 11 | SmolLM2-360M-Instruct | smollm2 | — | 3,661 | **48,064** | **13.13×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 12 | SmolLM2-1.7B-Instruct | smollm2 | 1.7B | 3,911 | **50,650** | **12.95×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 13 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | 2,856 | **35,343** | **12.375×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 14 | TinyLlama-1.1B-Chat-v1.0 | tinyllama | 1.1B | 4,355 | **48,108** | **11.048×** | torch.compile(neuron) | trn2.3xlarge | ✅ Verified |
| 15 | gpt2-medium | gpt2 | — | 6,083 | **61,158** | **10.055×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 16 | bloom-1b7 | bloom | — | 2,556 | **24,852** | **9.723×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 17 | Qwen2.5-Coder-1.5B | qwen2.5 | 1.5B | 3,985 | **38,369** | **9.628×** | torch.compile(neuron) | trn2.3xlarge | ✅ Verified |
| 18 | Mistral-7B-Instruct-v0.3 | mistral | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ Verified |
| 19 | Qwen3-8B | qwen3 | 8B | 1,903 | **16,876** | **8.87×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ Verified |
| 20 | deepseek-llm-7b-base | deepseek | 7B | 2,781 | **20,702** | **7.443×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 21 | gpt2-large | gpt2 | — | 4,170 | **30,620** | **7.343×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 22 | Qwen2.5-Math-7B | qwen2.5 | 7B | 2,930 | **19,826** | **6.765×** | torch.compile(neuron) | trn2.3xlarge | ✅ Verified |
| 23 | Qwen2.5-Coder-7B | qwen2.5 | 7B | 2,949 | **19,866** | **6.736×** | torch.compile(neuron) | trn2.3xlarge | ✅ Verified |
| 24 | Qwen3-14B | qwen3 | 14B | 1,699 | **10,343** | **6.087×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 25 | Qwen2.5-14B-Instruct | qwen2.5 | 14B | 1,819 | **10,256** | **5.637×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ Verified |
| 26 | Qwen3-32B | qwen3 | 32B | 975 | **3,698** | **3.794×** | TP=4, torch.compile(neuron), bf16, batch=1, CP=2 | trn2.3xlarge | ✅ Verified |
| 27 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | 2,916 | **9,870** | **3.384×** | TP=4, bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 28 | opt-1.3b | opt | 1.3B | 6,859 | **21,077** | **3.073×** | torch.compile(neuron) | trn2.3xlarge | ✅ Verified |
| 29 | stablelm-2-1_6b | stablelm | 6B | 4,828 | **14,183** | **2.938×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 30 | RedPajama-INCITE-Instruct-3B-v1 | redpajama | 3B | 3,345 | **9,824** | **2.937×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 31 | SmolLM2-360M | smollm2 | — | 10,836 | **31,764** | **2.931×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 32 | bloom-560m | bloom | — | 17,095 | **48,320** | **2.826×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 33 | pythia-1.4b | pythia | 1.4B | 5,454 | **13,294** | **2.438×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 34 | opt-2.7b | opt | 2.7B | 4,810 | **11,649** | **2.422×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 35 | phi-2 | phi | — | 4,176 | **9,841** | **2.357×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 36 | stablelm-3b-4e1t | stablelm | 3B | 3,393 | **7,721** | **2.275×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 37 | pythia-2.8b | pythia | 2.8B | 3,108 | **6,450** | **2.076×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 38 | SmolLM2-1.7B | smollm2 | 1.7B | 10,573 | **14,460** | **1.368×** | (config-only) | trn2.3xlarge | ✅ Verified |
| 39 | Qwen3.5-0.8B | qwen3.5 | 0.8B | 1,093 | **1,143** | **1.045×** | TP=4, bf16, batch=1 | trn2.48xlarge | ✅ Verified |
<!-- LEADERBOARD:END -->

*Speedup is vs the eager baseline on the same instance + probe shape. ✅ Verified
= correctness-gated + reproducible via each model's `reproduce.sh`.*

## Repo map

| Path | What |
|------|------|
| `optimizer/` | CLI — `python -m optimizer.{run,apply,measure}` |
| `implementation/src/` | the framework — orchestrator, backends, kernel authoring, opportunity sweep |
| `knowledge-bank/` | the compounding lesson + kernel store |
| `optimized_models/` | published recipe bundles |
| [`plan.md`](./plan.md) | roadmap + strategy — **read first** |
| [`architecture.md`](./architecture.md) · [`optimization-stages.md`](./optimization-stages.md) · [`docs/`](./docs) | design deep-dives |

## Design principles

- **Optimize only where the compiler is weak.** Measured on-device: the compiler is ~80% of speed-of-light on standard ops; NKI wins only in the compiler-weak regime (long-context/sparse attention, scans). Target selection is driven by measured %SOL.
- **Borrow before invent.** Reuse a proven kernel first; a novel one must beat it by ≥5%.
- **Honest measurement.** Speedup vs `torch.compile`-fused, reported as % of speed-of-light, device-timed (never host wallclock). Losses are banked as anti-patterns.
- **Compounding bank.** Lessons make each successive model cheaper — the flywheel.

Prior art: [references-analysis.md](./references-analysis.md) · shapes/ceilings: [guardrails.md](./guardrails.md).
