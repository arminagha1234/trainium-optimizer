# DeepSeek-V4-Flash Native-PyTorch Bring-Up — Run Log

Execution log for `neuron/deepseek-v4-flash-native-pytorch-48xl-runbook.md`
(staged SOP P0-P9, hard gate per phase). Each phase records config, result, and
gate pass/fail. The runbook supersedes `docs/deepseek-v4-flash-plan.md` where
they disagree.

Box: `resource-pod-e8e4cc43-b425-4138-80cb-0afc1af18fa5-desktop` (Kaizen), a
trn2.48xlarge `i-07bcc40b8a7cbbe11`. Toolchain: torch 2.12.1+cu130,
torch_neuronx 2.12.3.0.1636+5c472775, transformers 5.15.0.

## P0 — Environment + topology + per-rank HBM budget — GATE: PASS

- **Device sanity**: `(ones(8)+ones(8)).sum()` → `16.0` on `torch.device("neuron")` (native path OK).
- **Topology (`neuron-ls`)**: `logical-neuroncore-config = 2` (LNC=2). 16 NeuronDevices ×
  4 cores = **64 cores**; **96 GB HBM per device** (1536 GB device total); 1999 GB host RAM.
  This resolves the repo-wide "32 vs 64 cores" ambiguity (runbook §3.2 / open-question #1)
  for this box: at LNC=2 the runtime exposes 64 cores, 4 sharing each 96 GB device.
- **Per-rank HBM budget** (LNC=2, one lone core, 1 GiB bf16 alloc-until-OOM):
  **22 GiB usable** (last success 22, OOM at 23). Under tp=64, 4 ranks share one 96 GB
  device → ~24 GB nominal/rank, ~22 usable — matching `why-no-48xl-wins-yet.md`'s "24 GB core".
  LNC=1 not measured: the desktop is fixed at LNC=2 (the operative mode for every run here);
  a reconfig was avoided to preserve a clean runtime.
- **Memory vs the corrected ~568 GB BF16 model** (runbook §0.2 Correction 2):
  tp=64 → 8.9 GB/rank weights vs 22 GiB budget = comfortable (~13 GB headroom for
  activations + KV); tp=32 → 17.8 GB/rank = tight (~4 GB left). **Confirms runbook
  Correction 3: target tp=64 first, tp=32 fallback.**
- **`torch.compile(backend="neuron", dynamic=False)`** 2-layer MLP smoke: compiled and ran at
  batch 1 and batch 8 (rule #1 healthy at the toolchain level).
- **TP collective gate** (all-reduce of ones, after a runtime cleanup):
  - world=8 → **fails** (`device barrier`) — the known tp=8 degeneracy on this box; use
    `TRN_OPT_SKIP_TP=8` / avoid tp=8 for DeepSeek runs.
  - world=16 → OK (16.0); **world=64 → OK (64.0)** — the target tp=64 collective forms cleanly.
  - Rule #4 confirmed: the first barrier failure came from a **dirty runtime** left by the
    per-rank-HBM OOM crash; `pkill torchrun` + sleep cleaned it and 16/64 then passed.
- **Disk**: `/` = 3.5 TB local NVMe, 1.8 TB free (fine for the P7 ~160 GB download; four
  spare 1.7 TB NVMe available). Shared HF cache at `/ustore/fsx/team_shared_rw/hf_cache_shared`.

## P1 — Shrunk config + CPU FP32 reference oracle — GATE: PASS

Key enabler: **transformers 5.15.0 ships a complete native `modeling_deepseek_v4.py`**
(1517 lines) implementing every hard component — `DeepseekV4HyperConnection` (full Sinkhorn
loop with column normalization, i.e. *not* the runbook's Critical Bug #1),
`DeepseekV4GroupedLinear` (o_groups grouped low-rank O), `DeepseekV4Indexer` (lightning-indexer
top-k), `DeepseekV4{TopK,Hash}Router`, `DeepseekV4Experts` (swiglu_limit clamp),
`DeepseekV4CSACompressor`, MLA attention, etc. So the P1 FP32 reference is the
transformers-native model itself, not a hand-authored re-implementation.

Config normalization: the checkpoint's legacy `compress_ratios`
`[0,0,4,128,...,4,0]` (43 entries) folds into per-layer `layer_types`
(`{0: sliding_attention, 4: compressed_sparse_attention, 128: heavily_compressed_attention}`),
and `num_hash_layers=3` folds into `mlp_layer_types` (first 3 = `hash_moe`). Setting all
`layer_types = sliding_attention` gives the dense, CSA-off baseline the runbook wants for P2-P6.

Reference (`implementation/src/deepseek_v4/p1_reference.py`): shrunk to 4 layers / 8 experts /
top-2 / 1 hash layer / vocab 4096, keeping real hidden 4096, head_dim 512, 64 heads,
q_lora/o_lora 1024, o_groups 8, hc_mult 4, moe_intermediate 2048. Seed 0, seq 32, CPU FP32.

- Result: 1.37B params; logits `(1, 32, 4096)`; **finite = True**; **byte-reproducible = True**
  across two fresh instantiations (so P2 can re-seed to reproduce weights on device instead of
  shipping a state_dict).
- **64 component boundaries captured** to `ref.pt` (input+output per module) as the P2 oracle:
  `layers.N.attn_hc` (HyperConnection, `(B,S,hc_mult=4)`), `self_attn.o_a_proj`
  (GroupedLinear `(B,S,8,1024)`), `self_attn` (MLA out), `mlp.gate` (HashRouter on layer 0,
  TopKRouter on 1-3, `(T,8)`), `mlp.experts` (`(T,4096)`), `mlp.shared_experts`, `mlp`
  (SparseMoeBlock), `rotary_emb` (cos/sin), `hc_head`, and the RMSNorm/UnweightedRMSNorm nodes.

Next: P2 — move each captured component to the neuron device (eager), compare against this
oracle bottom-up (RMSNorm → RoPE → HC Sinkhorn → HC pre/post → router → expert SwiGLU →
attention → full layer → full model), MoE Stage A (dense all-experts). Gate: final logits
cos ≥ 0.99; per-component tolerances per runbook §8.

## P2 — Single-core eager device equivalence — GATE: PASS (device math), routing-conditioning characterized

Rebuilt the identical shrunk model on `torch.device("neuron")` in BF16 and compared to the
P1 FP32 oracle. Harness: `implementation/src/deepseek_v4/p2_device.py`.

Device compute facts (both are hard constraints, discovered here):
- The MoE expert path lowers to torch_neuronx's `grouped_mm`, which requires (a) the token
  dimension **divisible by 128** (seq=32 → `ValueError: t must be divisible by 128`; seq=128
  works) and (b) **BF16** inputs (`grouped_mm` has no FP32 path). So the device forward is BF16
  end-to-end — consistent with runbook §1.2 (compute is BF16; quantization is storage-only).
  The oracle was regenerated at seq=128.

Results:
- **ALL-HASH control (deterministic routing): final logits cos = 0.999660** → **all device math
  is correct**. Bottom-up, every non-router component matches the oracle at cos ≥ 0.999 through
  the early layers: RoPE (cos/sin), RMSNorm / UnweightedRMSNorm, HyperConnection (Sinkhorn),
  MLA attention, the grouped low-rank O projection (`o_a_proj`/`o_b_proj`), grouped experts, the
  shared expert, and the hyper-head. The 0.9997 residual is pure BF16 quantization.
- **MIXED (learned TopKRouter): final logits cos = 0.758** — diverges **solely** from learned
  top-k **selection flips under BF16 on random weights**. This is a random-weight *conditioning*
  artifact, not a bug, proven three ways: (1) the hash-routed layer-0 experts match at cos ≥
  0.999; (2) the router *scores* match at cos ≥ 0.999 (only the argmax selection flips); (3)
  upcasting the entire routing decision to FP32 barely moved it (0.758 → 0.760), i.e. the flips
  are driven by the BF16-drifted *input* into an ill-conditioned near-uniform router, not by
  scoring precision. Trained weights have well-separated winners, so this does not occur on the
  real checkpoint; it is validated there by the P8 top-1 / KL gate, not by a random-weight cos.

Router internals (transformers `DeepseekV4TopKRouter`): `logits = F.linear(flat, weight)` →
`sqrtsoftplus` → `torch.topk(scores + e_score_correction_bias, top_k)`. Confirms open-question
#3: `noaux_tc` == sqrtsoftplus scoring + bias-corrected top-k (bias = 0 in the untrained shrunk
config). Two items this hands to P4: (i) `torch.topk` compiles in eager but the runbook flags it
as failing under `torch.compile` on this model → swap to **iterative argmax**; (ii) the native
router does **not** upcast to FP32 (runbook §4.2 pseudocode does) and the eager `DeepseekV4Experts`
loop accumulates in BF16 (runbook §4.6 wants FP32 MoE accumulation) — both matter for the P8
accuracy gate on real weights.

Next: P4 — static shapes + `torch.compile(backend="neuron", dynamic=False)` on the shrunk config,
with the router as iterative-argmax + fixed-capacity (Stage B) dispatch.
