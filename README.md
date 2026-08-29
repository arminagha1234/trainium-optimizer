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
| Rank | Model | Params | Optimized (tok/s) | Speedup | Hardware | Status |
|-----:|:------|-------:|------------------:|--------:|:---------|:-------|
| 🥇 | Qwen3-0.6B | 0.6B | **85,937** | **25.8×** | trn2.3xlarge | ✅ Verified |
| 🥈 | Qwen3-1.7B | 1.7B | **51,278** | **17.2×** | trn2.3xlarge | ✅ Verified |
| 🥉 | Qwen2.5-0.5B | 0.5B | **74,269** | **15.4×** | trn2.3xlarge | ✅ Verified |
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
