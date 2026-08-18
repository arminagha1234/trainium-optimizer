# Overnight run — 2026-08-18 15:13 UTC (cycle 1)

**One snapshot** from the continuous autonomous run of this framework on real
Trainium hardware — native PyTorch on a `trn2.48xlarge`. Not a curated demo:
these are the numbers the loop wrote when it finished cycle 1 at 15:13 UTC.

The run is a **long-running process** (`overnight.py --forever --auto-promote`)
that cycles the seed models continuously, so newer cycles will produce
different numbers as auto-promotion compounds learned priors. This folder is
the first published snapshot; future runs land in sibling `runs/YYYY-MM-DD…/`
folders.

## Results — 5 of 7 seeds

| Model      | Baseline (tok/s) | Best (tok/s) |   Speedup |
|------------|-----------------:|-------------:|----------:|
| Qwen3-0.6B |            3,196 |   **44,100** | **13.80×** |
| Qwen3-1.7B |            2,779 |   **30,704** | **11.05×** |
| Qwen3-4B   |            1,893 |   **20,423** | **10.79×** |
| Qwen3-8B   |            1,806 |   **15,454** |  **8.56×** |
| Qwen3-32B  |            1,004 |    **7,909** |  **7.88×** |

Toolchain (stamped in every recipe): torch 2.12.1 · torch\_neuronx 2.12.3
· neuronx\_cc 2.27 · nki 0.6.0 · driver 2.30.2.0.

The dominant Stage-1 lever in every case is `torch.compile(backend="neuron")`.
The search reliably reaches it because the proposer tries `compile_mode`
before every other axis and no soft stopping condition can fire until each
axis has been explored at least once. Both fixes were merged in the framework
before this run began; see [`optimization-stages.md`](../../optimization-stages.md).

## Two seeds that need adapters (still 0×)

Both errored during the run and are recorded in the leaderboard as attempted
but not resolved — the loop retries them every cycle and moves on.

| Model | Blocker |
|-------|---------|
| Gemma-4-31B  | heterogeneous per-layer head layout; `k_proj.view(...,-1,512)` fails under uniform sharding. Needs a Gemma-4 family adapter. |
| Qwen3.8-27B  | 4 KV heads cap the simple GQA plan at TP=4, where 27B weights don't fit a 24 GB core. Needs KV-head replication (or vocab-parallel embed/lm\_head) to reach TP=8. A partial GQA→MHA expansion adapter has already landed on the backend and helped a subset of cases. |

## Charts

Per model, the framework writes two charts:

- `charts/<model>-highlights.png` — kept-path staircase, prose step labels
  ("torch.compile", "TP=4", "NKI flash-attention"), stage dividers, and a
  giant final Nx callout. Written for the models that completed under the
  current code. **Hero: [`charts/qwen3-0-6b-highlights.png`](./charts/qwen3-0-6b-highlights.png)**.
- `charts/<model>-timeline.png` — every attempt (incl. discards), stage
  colors, provenance markers, MFU on a secondary axis. Engineer view.

## Technical notes

Full narrative in [`NIGHT_LEARNINGS.md`](./NIGHT_LEARNINGS.md). Short version:

1. **The search must try `compile_mode` early.** Old ordering + a 5-round
   no-improvement stop was terminating small models at 1.06× because they
   plateaued before the compile axis was tried. Fixed: 1.06× → 13.80×.
2. **Cross-chip TP works on Trn2** — the Trn1 device-barrier failure does
   not reproduce.
3. **DP replicas fill the rest of the instance** once TP is capped by KV
   heads — the fill planner derives `dp = cores // tp`.
4. **The bank compounds across cycles.** Auto-promotion moves a proven
   provisional lesson to `verified/` under explicit criteria, so the next
   model starts from what the last one proved.

## Caveats — read before quoting these numbers

- Cycle 1 only. Later cycles may go higher or lower as the compounding kicks
  in and lessons transfer across the Qwen3 family.
- Correctness is currently gated by top-1 token match against the eager
  baseline (≥75%) — real, not the stub. Full logit-level tolerance is future
  work.
- 2 of 7 seeds are unresolved (see above). Publishing the leaderboard with
  those rows present is intentional: the framework's design records what
  it *tried* and *couldn't do*, not just what it succeeded at.
