# Night Learnings — Autonomous Native-PyTorch Optimization (Trn2)

**Instance:** trn2.48xlarge `<INSTANCE_ID>` · **Backend:** native-pytorch-beta3 (real HW)
**Toolchain:** torch 2.12.1 · torch_neuronx 2.12.3 · neuronx_cc 2.27.2878.0 · nki 0.6.0 · driver 2.30.2.0

## Is it running? (check any time)
```
ssh -i .../arminagha.pem ubuntu@<EC2_HOST>...  'pgrep -af run_overnight'          # process
  '                                                 tail -f ~/optimizer/run.log'       # milestones
  '                                                 cat ~/optimizer/artifacts/LEADERBOARD.md'
touch ~/optimizer/artifacts/STOP    # stop cleanly after current model (leaves instance up)
```
The **forever loop is live** (`--forever --auto-promote --cycle-pause 30`), detached (setsid), so it
survives disconnects and cycles the model list continuously, compounding learned priors.

## The three text-to-text seeds — status

| Seed | Works? | TP | Notes |
|------|--------|----|-------|
| **Qwen3-32B** (replaces muse-glimmer) | ✅ YES | tp8 | 8.4 GB/rank, verified real forward. muse-glimmer-30b is **not open-source**, so I substituted this Apache-2.0 dense 32B — the same "large dense text LLM" role. |
| **gemma-4-31b** | ❌ not yet | — | Needs a **Gemma4 adapter**: heterogeneous per-layer head layout — `k_proj.view(...,-1,512)` fails under uniform sharding. Also multimodal (we run text-only). |
| **qwen3.8-27b** | ❌ not yet | — | Needs a **GQA-4 / tp8 adapter**: only 4 KV heads caps the simple plan at tp4, where 26.9B weights don't fit a 24 GB core (fragmentation at ~18 GB). tp8 requires KV-head replication. |

**Bonus (DeepSeek-V4-Flash):** exists (`deepseek-ai/DeepSeek-V4-Flash`) but is a ~284B **MoE** — needs
an expert-parallel adapter + ~500 GB download. Out of scope for the dense worker tonight; noted for later.

## Models that WORK — real wins (the compile lever is the story)

`torch.compile(backend="neuron")` is the dominant Stage-1 win, 3–11× over eager. Real measurements:

| Model | Baseline tok/s (eager) | Best (compile) | Speedup |
|-------|------------------------|----------------|---------|
| Qwen3-0.6B | 3,196 | 35,904 | **11.2×** |
| Qwen3-1.7B | ~2,800 | (compile now reached each cycle) | ~+ |
| Qwen3-4B | ~1,900 | (compile) | ~+ |
| Qwen3-8B | 1,633 | 5,064 (tp1) / 11,332 (tp4) | **3.1–6.0×** |
| Qwen3-32B | 537 (eager,256) | (compile in loop) | large |

The forever loop refines these every cycle and auto-promotes the winners into the knowledge bank.

## Key hardware/software learnings (verified on-device)

1. **TP=8 was NOT the hardware max — the box does TP=64.** My earlier `[1,2,4,8]` was an artificial
   cap; raised to `[1,2,4,8,16,32,64]`. The real limit is per-*model*: the simple GQA plan needs
   `kv_heads % tp == 0`, so a model's KV-head count bounds clean TP (qwen3.8 → tp4).
2. **Each torchrun rank = one physical NeuronCore = 24 GB HBM** (probed: `total_hbm=25.77 GB`, OOM at
   23.5 GB). `NEURON_RT_VIRTUAL_CORE_SIZE=2` did not hand a rank a 48 GB logical core in these runs. So
   a 30B model must reach tp≈8 to fit comfortably (~8 GB/rank).
3. **Cross-chip TP works on Trn2** (the Trn1 `device barrier 1` failure does NOT reproduce) — TP=8
   `all_reduce` + full forward verified. This was the project's blocking unknown; it's cleared.
4. **Beta 3 uses `init_process_group(backend="neuron")`** (confirmed in `torch_api_compatibility.md`),
   not a Beta-2 anti-pattern. HF `from_pretrained(tp_plan="auto")` is **not** supported here
   (`unexpected kwarg`), and transformers 5.15 has no `tensor_parallel()` method — so manual DTensor
   (Colwise q/k/v/gate/up, Rowwise o/down) is the path.
5. **The greedy beam must try `compile_mode` early** — with the old ordering + 5-round no-improve stop,
   small models never reached compile (1.06× instead of 11×). The synced proposer fixes this: compile
   is tried first and every axis is explored. Confirmed: qwen3-0.6B went 1.06× → 11.2×.
6. **Fragmentation matters**: qwen3.8 at tp4 had 9.99 GB free but the largest contiguous chunk was
   1.9 GB, so a 2.54 GB (full embed/lm_head) alloc failed. Vocab-parallel sharding of embed+lm_head is
   the right lever (added, auto-triggers >10 GB/rank); tp8-via-KV-replication is the fuller fix.
7. **Multimodal text models load text-only fine** via `AutoModelForCausalLM` when no pixels are passed
   (qwen3.8 loads as pure text: `model` 25.6B + `lm_head` 1.27B, no vision tower on device).

## What I changed on the box tonight
- Extended TP axis to 64; baseline-fit picks the smallest valid tp under ~12 GB/rank.
- Added vocab-parallel embed+lm_head sharding (auto, >10 GB/rank) — DTensor-safe sync.
- Added Qwen3-32B as the large dense seed (muse-glimmer replacement).
- Backend (`native_pytorch.py` / `neuron_worker.py`) kept as the real implementation throughout.

## Adapter work done tonight (getting hands dirty)

Wrote a real **GQA→MHA expansion adapter** (`_expand_gqa_to_mha` in
neuron_worker.py): repeats a model's K/V projection weights so `num_kv_heads ==
num_heads`, letting a GQA model shard uniformly at any tp dividing the query
heads (past its KV-head count). K/V weights are tiny vs Q+MLP so overhead is
small. Baseline-TP + config axes now use query-head divisibility (KV handled by
the adapter) and tp goes up to 64. **Safe for the working models** — expansion
only triggers when `kv % tp != 0` (never for the clean Qwen3 dense models).

**Discovery via the adapter:** qwen3.8-27b expanded K/V on **only 16 of 64
layers** → it is a **hybrid** model: ~16 full-attention layers + ~48 Gated
DeltaNet (linear-attention) layers. So:
- The GQA→MHA adapter correctly handles the 16 attention layers.
- Remaining blockers for qwen3.8 end-to-end at tp8: (a) `repeat_kv` still uses
  the config n_rep (=6) at forward — need to null it to 1 after expansion (the
  `3 vs 6` shape error), and (b) the 48 Gated DeltaNet layers need their own TP
  sharding (conv/gate/state projections) — a genuine hybrid adapter. This is
  the `hybrid_attention_causal_lm` adapter the design docs anticipated.

## Next steps to make ALL three seeds work
1. **qwen3.8-27b → tp8 GQA adapter**: replicate the 4 KV heads across TP groups (shard Q/O by 8, keep
   K/V replicated) and patch `repeat_kv` n_rep to local head counts. Gets it to ~7 GB/rank → fits.
2. **gemma-4-31b → Gemma4 adapter**: per-layer head-count-aware sharding (its sliding/full layers
   differ), text-only via `model.language_model`.
3. Wire the real equivalence checker before publishing any recipe as correctness-verified.
4. Optionally add DeepSeek-V4-Flash behind an MoE/expert-parallel adapter.
