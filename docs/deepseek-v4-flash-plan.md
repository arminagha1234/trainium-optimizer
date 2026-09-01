# DeepSeek-V4-Flash on Trainium (native PyTorch) — in-depth bring-up plan

**Goal:** run inference for `deepseek-ai/DeepSeek-V4-Flash` (`DeepseekV4ForCausalLM`,
`model_type=deepseek_v4`) on a single trn2.48xlarge via the native-PyTorch backend, and
land verified leaderboard numbers. This is the hardest model we have targeted; the plan
is deliberately staged so each hard component is de-risked on a cheaper model first.

---

## 1. What the model actually is (read off the cached checkpoint)

43 layers, hidden 4096, vocab 129280, `torch_dtype=bfloat16`, YaRN to **1,048,576** ctx
(`rope_scaling` factor 16 over 65536), `rope_theta=10000`, `sliding_window=128`.

Six hard, independent pieces — each is a bring-up risk:

1. **MLA (Multi-head Latent Attention), V4 variant.** 64 query heads, `num_key_value_heads=1`
   (single latent KV), `q_lora_rank=1024`, **`o_lora_rank=1024` + `o_groups=8`** (the *output*
   projection is also low-rank — new vs V3), `qk_rope_head_dim=64`, `head_dim=512`, plus
   **compressed-KV** (`compress_rope_theta=160000`) and an **attention sink**
   (`attn.attn_sink`). Checkpoint tensor names are DeepSeek-native:
   `attn.wq_a/wq_b`, `attn.wkv`, `attn.q_norm/kv_norm`, `attn.wo_a/wo_b`, `attn.attn_sink`.
2. **DeepSeek Sparse Attention (DSA) "lightning indexer".** `index_topk=512`,
   `index_n_heads=64`, `index_head_dim=128`. Tensors: `attn.indexer.compressor.{wkv,wgate,norm,ape}`,
   `attn.indexer.weights_proj`, `attn.indexer.wq_b`. It scores all past tokens and attention
   only runs over the **top-512** (+ the sliding-window-128 local band). This is the piece
   that does *not* exist anywhere in our kernel corpus.
3. **MoE.** 256 routed experts, **top-6**, `+1 shared`, `moe_intermediate_size=2048`,
   `scoring_func=sqrtsoftplus`, `topk_method=noaux_tc` (bias-corrected grouped top-k, gate has
   a `.bias`), `routed_scaling_factor=1.5`, `norm_topk_prob=True`. Per-expert tensors
   `ffn.experts.N.{w1,w2,w3}` (**not** a fused `gate_up_proj`).
4. **FP8 + FP4 quantization.** `quantization_config`: fp8 e4m3, block `[128,128]`, `ue8m0`
   scales; **`expert_dtype=fp4`** (experts are FP4). Every weight carries a `.scale`. Trainium
   has no fp8/fp4 *compute* path, so transformers dequantizes to bf16 at load: **159.6 GB on
   disk → ~319 GB bf16 in HBM** (`capability.dequant_factor` already models this).
5. **MTP (multi-token prediction).** `num_nextn_predict_layers=1` — a `mtp.0.*` block mirroring
   a decoder layer + `e_proj/h_proj/enorm/hnorm`. For speculative decoding; defer.
6. **Hierarchical-clustering bits (V4-new, undocumented publicly).** `num_hash_layers=3`,
   `hc_eps/hc_mult/hc_sinkhorn_iters`, per-block `hc_attn_* / hc_ffn_* / hc_head_*`. **Not**
   present in GLM-5.2 or DeepSeek-V3.2 — a genuine unknown; treat as research (see §10).

**Memory / feasibility.** 319 GB bf16 ÷ **tp32** (a valid 8-device NeuronLink ring, verified
working) ≈ 10 GB/rank — **single-node trn2.48xl is feasible** once it loads. tp16 (20 GB/rank)
is too tight with activations. So the target serving config is **tp32, bf16**.

---

## 2. Strategy — native-PyTorch-first, escalation ladder (don't write a kernel unless forced)

Mirror what worked for GDN/Qwen3.5: bring the model up **eager on the `neuron` device**, let
`neuronx-cc` compile the graph, and only escalate when the compiler can't lower something:

> **graph/torch rewrite → reuse an authored kernel → parameterize an existing kernel →
> compiler flag → author a new NKI kernel** (last resort).

Two facts force this order:
- The framework's **only** hook that puts a custom kernel into a *served* model today is the
  MoE megakernel swap + the generic `--kernel` inject seam (`neuron_worker.py`). Authoring a
  kernel that can't be injected only banks a lesson; it can't prove a throughput win. So
  **eager + rewrites is the primary path**; authored kernels are reserved for primitives with
  no faithful in-graph expression.
- Qwen3.5 proved rewrites beat kernels: `.tril()`→const-mask and sort-free argmax router were
  **graph rewrites, no kernel**. DeepSeek's `noaux_tc` router and `sqrtsoftplus` are the same
  shape of problem.

**The one primitive that will almost certainly need an authored kernel is the DSA
lightning-indexer** (top-512 select + gather + sparse attention): the compiler OOMs on dense
[S,S] and there is no in-graph top-k-gather-attention it can lower at 1M context.

---

## 3. What to STEAL (borrow-before-invent), ranked by leverage

1. **GLM-5.2-FP8 (`GlmMoeDsaForCausalLM`) — the single biggest steal.** Already harvested
   (`docs/vllm-neuron-harvest.md`) from **AWShtokoyo/vllm-neuron@add-glm-5-2**, Apache-2.0,
   "Copyright Amazon.com" — **vendorable with attribution**. It *is* the DeepSeek-V3 family:
   MLA + 256-expert MoE + shared expert + fp8, **DSA omitted (full attn)**. Steal:
   - **MLA weight-absorption** (q_nope absorbed into `kv_b[:nope]`, output into `kv_b[nope:]`,
     **v-cache aliases k-cache** → a single **576-dim latent KV / token**),
   - **in-kernel FP8-ROW dequant**, interleaved RoPE (θ=8e6),
   - the **model + dispatcher structure** and the **graph-rewrite recipes** (8 are already
     live in `kernel_rewrites.py` tagged "HARVESTED from AWShtokoyo/vllm-neuron").
   Registry slots already reserved: `glmmoedsa`→`glm_moe_dsa`, `mlaabsorbed`→`mla_absorbed_*`.
   GLM-5.2 = **DeepSeek-V4 minus DSA minus the V4-exotica** → the perfect scale rehearsal.

2. **The private `deepseek_v32` vLLM example** (aws-neuron/private-vllm-neuron @feat/dhwanw-gemma4).
   Same family + **the DSA lightning indexer** (the delta from GLM-5.2 → V3.2/V4) + the V3.2 fp8
   path. This is the authoritative Neuron reference for the indexer. **Access blocker:** the
   dev box has no `aws-neuron` GitHub creds (`git clone` → "could not read Username"). Need a
   token on the box, or a mirror of `vllm_neuron/model/deepseek_v32/` into the workspace/FSX.
   Per the harvest license note, the heavy kernels live in proprietary `nkilib.*` (**learn-from
   only, re-author**); the model/dispatcher code + inline `@nki.jit` kernels are vendorable.

3. **`aws-neuron/nki-library`** — official reference kernels: attention CTE/TKG, **MoE CTE/TKG**,
   **router-topk**, rope, rmsnorm-quant, output-proj. Direct source for the MoE + router + rope
   primitives (don't re-derive).

4. **DeepSeek public repo + DeepSeek-V3.2 paper/code** — authoritative *math* for MLA and the
   lightning-indexer scoring/top-k selection (public). Use to validate our re-authored indexer.

5. **NVIDIA ModelOpt** — MTP / EAGLE draft-head *architecture* (portable), for the MTP
   spec-decode axis in M4+. (GPU quant menu is not trn2-relevant.)

6. **`nki-samples`** (attention opt ladder v1–v8a, flash, mxfp8 matmul) — the perf ladder for
   the indexer/flash kernel; translate `nki` (dst-first) → `neuronxcc.nki` (return-form).

Already in our corpus and reusable now (passed-on-device): **FlashAttention, moe_fused,
rmsnorm_gated** (+ KDA/DeltaNet/Mamba2). MLA and GlmMoeDsa are **registered but unauthored**.

---

## 4. Staged milestones — small first, NOT DeepSeek-V3 first

**Why not V3 first:** 671 B (a monster just to load) *and* it has **no DSA**, so it doesn't
de-risk V4's hardest piece. GLM-5.2 dominates it (harvested, vendorable, cleanly DSA-omitted).

- **M0 — prerequisites (cheap, do first).**
  - Confirm `transformers==5.15` actually has `deepseek_v4` modeling (like we checked
    `qwen3_5_moe`); if not, use `trust_remote_code` / the DeepSeek original modeling, and plan a
    **key remap** (native `wq_a/wkv/wo_a/ffn.experts.N.w1` → whatever the modeling expects), the
    same shape of fix as the Qwen3.5 VL-tower remap (PR#180).
  - Run `capability.assess()` for an honest verdict (expect: fits tp32 bf16, host-DRAM at load
    is the risk → needs the streaming loader, §5).
- **M1 — DeepSeek-V2-Lite (16 B, MLA + MoE, bf16, NO DSA/fp8).** Smallest MLA+MoE; downloads
  in minutes (~31 GB), runs at tp4/tp16. **De-risks:** MLA weight-loading (latent q/kv/o
  projections), per-expert MoE loading, and **eager-MLA compile** (does `neuronx-cc` lower MLA
  as-is? if not, which rewrite?). *Deliverable: first MLA model on the board; per-expert loader
  landed (also unblocks Qwen3-30B/235B).*
- **M2 — GLM-5.2 (DeepSeek-V3 family, MLA + 256-expert MoE + fp8, DSA-omitted).** Vendor the
  harvested model/dispatcher code. **De-risks at V4 scale, minus DSA:** fp8→bf16 (and fp4)
  dequant load, weight-absorbed MLA (576-dim latent, v aliases k), noaux_tc-style grouped router
  as a rewrite, big-MoE EP + streaming load. *Deliverable: a DeepSeek-family MLA+MoE+fp8 model
  running at tp32 — everything except the indexer and the V4 exotica.*
- **M3 — DSA lightning-indexer kernel (the long pole).** Re-author from `deepseek_v32` + the
  public V3.2 algorithm: compressor (`wkv/wgate/norm/ape`) → per-token index scores over
  `index_n_heads=64 × index_head_dim=128` → **top-512 select + gather → sparse attention**,
  fused with `sliding_window=128` local + `attn_sink`. Borrow the FlashAttention corpus +
  `nki-samples` online-softmax ladder. Validate numerics vs the DeepSeek reference, then wire
  it through the generic `--kernel` inject seam (extend it if needed — this is the
  measurement-path work `kernel-stage-deepdive.md` calls priority #1).
- **M4 — DeepSeek-V4-Flash full integration (tp32).** MLA(+`o_lora`/`o_groups`, compressed-KV,
  attn_sink) + DSA + fp4 experts + noaux_tc + `sliding_window` + YaRN + the **hc / hash-layer**
  bits (§10). MTP deferred to a later spec-decode axis.

---

## 5. Loader plan (per-expert + fp8/fp4 + MLA latent + native names)

Current gaps and the fix for each:
- **Native DeepSeek tensor names** (`wq_a/wq_b/wkv/wo_a/wo_b`, `ffn.experts.N.w{1,2,3}`,
  `attn.indexer.*`). Confirm the transformers `deepseek_v4` mapping; if the checkpoint keys
  don't match the modeling, add a **remap** (proven pattern: `shard_stream.remap_vl_text_keys`).
- **Per-expert experts** (`experts.N.w1/w2/w3`, not fused). `shard-on-read`'s `slice_for`
  currently recognizes **fused** `experts.gate_up_proj/down_proj` only, so it no-ops on
  per-expert layouts (this is exactly why Qwen3-30B fell back to full-load + OOM). **Extend
  `slice_for` to the per-expert `.experts.N.w{1,2,3}` layout** → reusable win for Qwen3-30B/235B
  *and* DeepSeek. This is the single highest-leverage loader change.
- **fp8-block (128×128) + fp4 experts.** Dequant-to-bf16 happens in `from_pretrained`, but the
  transient is `ranks × model` host DRAM → the streaming **shard-on-read** loader (meta-init +
  per-rank slice, `capability.host_load_peak_gb` lean path) is *mandatory* at 319 GB. Verify
  the dequant survives the streaming path (read fp8/fp4 block + scale, dequant per shard).
- **MLA latent projections** (`wq_a`[down]→`wq_b`[up], `wkv`, `wo_a`→`wo_b`): shard by head on
  the up-projections; the latent (`kv=1`) is small and **replicated**. Follow the GLM-5.2
  weight-absorption so KV cache is a single 576-dim latent/token (huge KV-cache saving).

---

## 6. Attention plan (MLA + DSA + sliding-window + sink)

- **MLA:** eager first. If the compiler struggles, apply the **weight-absorption** rewrite
  (absorb `q_nope` into `kv_b[:nope]`, output into `kv_b[nope:]`; v-cache aliases k-cache).
  Interleaved RoPE on the `qk_rope_head_dim=64` slice; `compress_rope_theta=160000` for the
  compressed path. `o_lora`/`o_groups=8` is a grouped low-rank output proj — new; validate
  against the reference (GLM-5.2 has only `wo`, so this is a small V4 delta to add).
- **DSA (M3):** the authored kernel. Pipeline: indexer.compressor → index logits (64 heads ×
  128 dim) → **top-512** per query → gather selected K/V → online-softmax attention over 512,
  unioned with the `sliding_window=128` local band and the `attn_sink`. Roofline it (bs=1
  decode is host-bound — the win is graph-reuse/async, not raw FLOPs).

---

## 7. MoE plan

- 256 routed (top-6) + 1 shared, `noaux_tc` (grouped, aux-loss-free, bias-corrected top-k),
  `sqrtsoftplus` scoring, `routed_scaling_factor=1.5`, `norm_topk_prob`. Implement the router as
  a **graph rewrite** (bias-add → grouped top-k → renorm → scale) — the same class as the
  Qwen3.5 sort-free router; no kernel. Expert compute reuses **`moe_fused`** + shard-on-read EP
  once the per-expert loader (§5) lands. fp4 experts dequant to bf16.

---

## 8. Numerics, quantization, equivalence

- **Compute in bf16** (no fp8/fp4 path on trn2; mxfp4 is trn3-only). The "win" is running the
  model at all + prefill/batch throughput, not a quant speedup.
- **Equivalence gate:** top-1 **+ KL** over the baseline top-k distribution (not exact-token —
  the Qwen3.5 batch-variance lesson). Reference = the model on CPU/GPU bf16. Use a modest
  input-len (512–2048) for bring-up; validate long-context (YaRN) separately.

---

## 9. Measurement-path work (the real priority-#1 from kernel-stage-deepdive.md)

To turn an authored DSA kernel into a *served* win, the generic `--kernel` inject seam must
place it on the `attn.indexer` module and survive compile. Budget explicit time to extend the
injection hook + an on-device equivalence check for the indexer, before/parallel to authoring
it. Without this, M3 can only bank a lesson.

---

## 10. Risks / unknowns (ranked)

1. **DSA indexer kernel** — net-new, no corpus entry, needs the private example for the Neuron
   idioms. Long pole. *Mitigation:* get repo access; stage on GLM-5.2 (no DSA) so everything
   else is already green when we add it.
2. **V4 hierarchical-clustering / hash layers** (`num_hash_layers=3`, `hc_*`) — **not** in
   GLM-5.2 or V3.2; the `deepseek_v32` example won't cover it. *Mitigation:* read the DeepSeek-V4
   modeling (trust_remote_code) to learn what hc does; it may be a routing/attention refinement
   that eager-compiles fine, or a new rewrite.
3. **fp4 experts** — dequant path at 319 GB; streaming loader must handle fp4 blocks + scales.
4. **`o_lora`/`o_groups`, compressed-KV, attn_sink** — small V4 deltas over GLM-5.2/V3.2; add
   incrementally at M4.
5. **Host-DRAM at load** — `ranks × 319 GB` without the lean loader = OOMKill (we hit exactly
   this class on the 35B). shard-on-read streaming is mandatory.

---

## 11. First actions

- **M0 now:** verify `deepseek_v4` modeling in transformers 5.15 (else trust_remote_code +
  remap); run `capability.assess()`; get `aws-neuron` repo access (or a mirror of `deepseek_v32`).
- **M1 now:** download DeepSeek-V2-Lite, feed it through `run_overnight.py`
  (`--no-preflight`, TRN_OPT_SKIP_TP if needed) and iterate on the loader/MLA errors — the same
  "feed it, watch the errors, fix" loop that just landed Qwen3.5-35B.

*Sources to vendor/learn-from are catalogued in `kernel_sources.yaml` / `docs/kernel-sources.md`
and `docs/vllm-neuron-harvest.md`. Heavy `nkilib.*` kernels are learn-from-only; model/dispatcher
code + graph rewrites + inline `@nki.jit` kernels from AWShtokoyo/vllm-neuron are Apache-2.0
vendorable with attribution.*
