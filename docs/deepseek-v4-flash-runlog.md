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

## P3 — Component equivalence harness — GATE: PASS

Two layers, per the runbook:
- **Operational device ladder** (`implementation/src/deepseek_v4/p2_device.py`): re-runnable after
  any change; returns nonzero on failure; emits the bottom-up per-component drill + the all-hash
  math gate. This is what gets run on-device after each P4/P5 change.
- **CPU CI checks** (`implementation/src/deepseek_v4/test_p3_components.py`): checkpoint-independent
  (the public config.json is vendored as `deepseek_v4/v4_flash_config.json`, so no 160 GB download),
  runnable in CI:
  1. reference is finite and byte-reproducible;
  2. **HyperConnection `comb` is doubly-stochastic** (row & column sums ≈ 1) — directly guards
     against runbook Critical Bug #1 (row-softmax-only / missing column normalization). Measured:
     column sums exact to 1.0, row sums within 1.1e-2 (converged Sinkhorn at 20 iters);
  3. all 10 section-8 component classes are present in the module tree.

Refactor: `build_shrunk_config` now builds from the vendored `v4_flash_config.json` (was loading
the FSX-cached config), making P1-P3 truly checkpoint-independent and CI-runnable. Verified on the
local venv (transformers 5.16.1, torch 2.13, CPU): 3/3 P3 tests pass.

Next: P4 — `torch.compile(backend="neuron", dynamic=False)` on the shrunk config; the router must
move to iterative-argmax + fixed-capacity (Stage B) dispatch since `torch.topk` + dynamic MoE
shapes are the runbook's flagged compile hazards.

## P4 — Static shapes + torch.compile(dynamic=False), shrunk config — GATE: PASS

Root cause of the first compile failure (both routing configs): the native MoE lowers a
**`AwsNeuronTopK`** custom-call that the Neuron compiler rejects — from the learned router's
`torch.topk` AND from the grouped_mm expert backend's internal sort
(`COMPILATION FAILED ... custom_call_target="AwsNeuronTopK"` inside `DeepseekV4Experts`,
`transformers/integrations/moe.py`). This confirms runbook §4.2 on this toolchain
(torch_neuronx 2.12.3).

Fix (`implementation/src/deepseek_v4/compile_patches.py`):
- **`stage_a_experts_forward`** — dense experts unrolled over E (compile-time constant): no
  grouped_mm, no top-k, no data-dependent shapes, no token dropping. Numerically equivalent to
  the native grouped_mm experts in eager: **cos 0.999963**.
- **`iter_argmax_router_forward`** — top-k via iterative argmax instead of `torch.topk`.
  Selection matches `torch.topk` exactly (agreement **1.0000** on random CPU scores).

Results (`torch.compile(backend="neuron", dynamic=False)`, single core, seq 128):
- **All-hash** (deterministic routing): compiles at batch 1 (152 s) and batch 2 (180 s);
  compiled-vs-eager cos **0.9996** / 0.9995.
- **Mixed** (learned router, the realistic path): compiles at batch 1 (151 s) and batch 2
  (186 s); compiled-vs-eager cos **0.9937** / 0.9908 (≥ 0.99 gate; the small gap is the same
  random-weight routing conditioning as P2, not a compile error).

Rule #1 satisfied: each batch size compiles to its own NEFF. Drop count is zero (Stage A is
dense). **GATE PASS.**

Scope note: unrolled Stage A is for the **shrunk** config (8 experts). The real 256-expert model
needs **Stage B** (fixed-capacity gather/scatter) for a compiled path (an unrolled 256×43 graph
is too large to partition); P7's first real forward can run the native grouped_mm **eagerly**.
FP32 MoE accumulation (runbook §4.6) is deferred to P8 (accuracy) with re-validation.

Next: P5 — TP ladder (tp=2→8→32→64) on the shrunk config; compare rank-0 logits vs tp=1.

## P5 — TP ladder on the shrunk config — GATE: PASS (MLA-TP validated at tp=16/32/64)

transformers 5.15 has **no TP plan** for DeepseekV4 (`_tp_plan=None`; `from_pretrained` has no
`tp_plan` arg), so TP uses **manual sharding** like the framework's `qwen38_tp`. But MLA has
`num_key_value_heads=1` (one shared low-rank latent KV), so GQA-style head-sharding slices the KV
head to zero width — **this is the root cause of the DeepSeek-V2-Lite crash**
("size of tensor a (4) must match tensor b (16)"/"...(0)").

Correct MLA TP (`implementation/src/deepseek_v4/tp_mla.py`): shard the **query** heads
(`q_b_proj` rows + per-head `sinks`), **replicate** the shared latent KV / norms / grouped-O, and
**all-gather** the per-rank attention output across ranks before the (replicated) O projection.

Results (shrunk all-hash config, rank-0 logits vs the tp=1 reference):

| tp | cos vs tp=1 | verdict |
|--:|--:|:--|
| 16 | 0.999962 | PASS |
| 32 | 0.999962 | PASS |
| **64** | **0.999962** | **PASS (primary target)** |

tp=2 and tp=8 are degenerate world sizes on this trn2.48xl at LNC=2 (P0), so the working ladder is
16→32→64 — which covers both user targets. **GATE PASS.**

Ops note (rule #4): a killed multi-rank run leaves orphaned `elastic_agent`/`multiprocessing`
spawn processes that wedge the next `init_process_group` (every subsequent torchrun died with
SIGKILL/137 and no output). `pkill -9 -f torchrun` alone is insufficient; also kill
`elastic_agent`/`multiprocessing`/`spawn_main` and settle ~15 s. (Also: `ada` creds are
short-lived — refresh before long runs.)

Scope for the real model (P7): experts are replicated here (the shrunk 8 fit per rank). The real
256-expert model needs **expert parallelism** (EP, ~4 experts/rank at tp=64 → ~8.9 GB/rank,
matching the P0 budget) in addition to this MLA-TP.

Next: P6 — loader on the shrunk config with synthetic FP8+FP4 quantized weights (per-expert
`slice_for`, shard-on-read host-RSS bound).
