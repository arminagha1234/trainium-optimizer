# plan.md sweep — what fits on ONE trn2.48xlarge, and why

Every target named in `plan.md`, triaged through the capability gate before any device
time was spent. Config metadata only: one Hub call per model, no weights downloaded.
That is the point of the gate — "can this run here" costs milliseconds, so there is no
excuse for discovering the answer from an OOM an hour in.

Box: 64 NeuronCores × 24 GB = **1536 GB HBM**, **2147 GB host DRAM** (measured from
`/proc/meminfo`). Weight budget 14.4 GB/rank (0.60 of a core).

## Runnable on one node

`conc` is `TRN_OPT_LOAD_CONCURRENCY` — how many ranks may hold a full model copy at
once. It is per-model because the host peak is `ranks × model_size`, so the setting
that fits a 72 GB model is not the one that fits a 915 GB one.

| Model | On-disk | tp | conc | GB/rank | Notes |
|:--|--:|--:|--:|--:|:--|
| Qwen3.5-0.8B | 1.7 | 8 | 4 | 0.2 | |
| Qwen3.5-2B | 4.5 | 8 | 4 | 0.6 | |
| Qwen3.5-4B | 9.3 | 16 | 4 | 0.6 | |
| Qwen3.5-9B | 19.3 | 16 | 4 | 1.2 | |
| gemma-4-12b-it | 23.9 | 4 | 4 | 6.0 | Gemma4 hard cap tp=4 |
| Qwen3.6-27B | 55.6 | 24 | 4 | 2.3 | tp 8→24 via divisor sweep (#140) |
| Qwen3.8-27B | 55.6 | 24 | 4 | 2.3 | tp 8→24 via divisor sweep (#140) |
| Qwen3-30B-A3B | 61.1 | 32 | 4 | 1.9 | |
| Qwen3.5-35B-A3B | 71.9 | 16 | 4 | 4.5 | |
| Kimi-Linear-48B-A3B | 98.2 | 32 | 4 | 3.1 | near-pure linear attention |
| DeepSeek-V4-Flash | 159.6 | 64 | 4 | 5.0 | fp8 → 319 GB bf16 at load |
| Qwen3.5-122B-A10B | 250.2 | 32 | 4 | 7.8 | |
| **Qwen3-235B-A22B** | 470.2 | 64 | **3** | 7.3 | host peak 1859 GB of 2147 |
| **MiniMax-M2** | 230.1 | **48** | 3 | 9.6 | fp8 → 460 GB; tp=48 needs #142 |
| **MiniMax-Text-01** | 914.7 | 64 | **1** | 14.3 | TIGHT; peak 1815 GB |

15 models. Three of them are only reachable because of fixes made while triaging:

* **MiniMax-M2** has 48 attention heads. The gate predicted tp=16 (largest power of
  two dividing 48) → 28.8 GB/rank → rejected. Every divisor is now considered, so
  tp=48 → 9.6 GB/rank (#142).
* **Qwen3-235B** and **MiniMax-Text-01** fit only at a lower load concurrency. At the
  default of 4 they need 2322 GB and 4516 GB of host DRAM respectively.

## Blocked, with the reason

| Model | On-disk | Blocker |
|:--|--:|:--|
| Kimi-K3 | 1560.9 | `NEEDS_MULTINODE` — exceeds all 1536 GB of HBM |
| Kimi-K2-Instruct | 1029.2 | `NEEDS_MULTINODE` — 2058 GB after fp8→bf16 |
| GLM-5.2 | 1506.7 | host DRAM: 2990 GB even at conc=1 |
| DeepSeek-V3.1 | 688.6 | host DRAM: 2733 GB even at conc=1 (1377 GB dequantized) |
| GLM-4.6 | 713.6 | HBM: 96 heads → tp=48 → 14.9 GB/rank, just over the 14.4 budget |
| gemma-4-31b-it | 62.5 | HBM: Gemma4's tp=4 cap → 16 GB/rank |
| Llama-4 Scout / Maverick | 217 / 803 | `config.json` unreadable — gated repo |
| Kimi-Linear (bare id) | — | 404; the real id is `Kimi-Linear-48B-A3B-Instruct` |
| GLM-5.2 @ THUDM, DeepSeek-V4 | — | 404; `zai-org/GLM-5.2` and `…-V4-Flash` are the live ids |

Three of those blockers are addressable, in increasing order of work:

1. **gemma-4-31b-it** is capped at tp=4 by a limit whose stated reason no longer
   holds. The comment says tp>4 "shards a KV head below one head_dim" — true before
   #127, which made KV heads REPLICATE instead of being sliced. At tp=8 each rank
   would hold one whole KV head. Worth an experiment, not a blind cap lift: Gemma4's
   attention is per-layer heterogeneous, so it needs device evidence.
2. **GLM-4.6** misses by 0.5 GB/rank. Shard-on-read (#141) does not change HBM, but
   dropping `weight_budget_frac` is not the answer either — the honest fix is a real
   measurement of how much headroom that model actually needs.
3. **GLM-5.2 / DeepSeek-V3.1 / Kimi-K2 / Kimi-K3** need either multinode or weights
   held quantized ON DEVICE. On trn2 an fp8 checkpoint dequantizes to bf16 at load, so
   the on-disk size is not what has to fit — DeepSeek-V3.1's 688 GB becomes 1377 GB.

## How the sweep runs

Four workloads, one per size band, models sequential within a band. Each model gets
its own `TRN_OPT_LOAD_CONCURRENCY`, and every cycle **auto-publishes** (#138) — the
leaderboard and the per-model route READMEs are written by the run, not by hand, and
still only for results that are `verified` with `speedup > 1.0` on real Neuron
hardware (#139).

Regenerate this triage with `.tmp/kaizen-setup/triage.py`-equivalent logic:
`capability.assess(config, TRN2_48XLARGE, weight_gb=<Hub metadata>, load_concurrency=N)`.
