# Harvest: AWShtokoyo/vllm-neuron contributed models → knowledge bank

**Harvested 2026-08-28** from the "Contributed Models" of `htokoyo/vllm-neuron@add-glm-5-2`,
which link to feature branches on the fork **`AWShtokoyo/vllm-neuron`** (GLM-5.2 is on
`htokoyo@add-glm-5-2`; the rest on `AWShtokoyo` branches).

**License / provenance.** All branches are **Apache-2.0, "Copyright Amazon.com, Inc."**
(every source file carries `SPDX-License-Identifier: Apache-2.0`). **Safe to vendor with
attribution (retain NOTICE).** BUT the heavy kernels are imported from the proprietary
`nkilib.core.*` / `nkilib.experimental.*` (NOT in these repos) — those are **learn-from
only**. Vendorable here: the model/dispatcher code, the graph-rewrite recipes, the idioms,
and the **inline `@nki.jit` kernels** (`_argsort_unstable_nki`, `gdn_*`, `paged_kv_*`,
Gemma-4 `attention_decode_kernel`). Note: several load-bearing comments in the Ministral3
files are in Japanese. `add-qwen36-27b` 404s; `add-llama-embed-nemotron` = pooling Llama,
low kernel novelty (skipped).

This doc is the durable capture. The catalog-fitting **rewrites are already LIVE** in
`kernel_rewrites.py` (8 entries tagged "HARVESTED from AWShtokoyo/vllm-neuron"). The
idioms and primitives below are the follow-up work-list for `nki_knowledge.py` and
`kernel_registry.py` + kernel authoring.

---

## Model inventory

| Model | HF id | Arch | Neuron-non-trivial core |
|---|---|---|---|
| **GLM-5.2-FP8** | zai-org/GLM-5.2-FP8 | `GlmMoeDsaForCausalLM` (DeepSeek-V3 family): MLA + 256-expert top-8 **sigmoid** MoE + shared expert, 3 dense layers, 78L | MLA weight-absorption, single-latent KV (576 dims/tok, v aliases k), interleaved RoPE θ=8e6, in-kernel FP8-ROW dequant, DSA indexer omitted (full attn) |
| **Qwen3.6-35B-A3B** ★ | Qwen/Qwen3.6-35B-A3B | Hybrid: 30 **GatedDeltaNet** + 10 full-attn (3:1), 256-expert top-8 **softmax** MoE + sigmoid-gated shared expert | Sequential NKI GDN scan, slot-indexed recurrent+conv state in unified paged pool, head_dim=256 partial RoPE, QK-norm |
| **Devstral-2-123B** (Ministral3) | mistralai/Devstral-2-123B-Instruct-2512 | Dense GQA, 88L, 96Q/8KV | YaRN interleaved RoPE, per-tensor static **FP8 native** (240-vs-448 e4m3), packed FP8 KV + segmented prefill |
| **Gemma-4 31B IT** | google/gemma-4-31B-it | `Gemma4ForCausalLM`, 60L | **Heterogeneous attn**: sliding-window head_dim=256 ⟷ global head_dim=512 (both >128), QK/V-norm, logit softcap tanh, GeGLU |
| llama-embed-nemotron-8b | nvidia/llama-embed-nemotron-8b | Llama embed/pool | (skipped — low novelty) |

---

## PRIMITIVE_TO_KERNEL additions (proposed for kernel_registry.py)

**Net-new (not covered today):**
- `glmmoedsa` / `moe_mla_dsa_sigmoid` → `glm_moe_dsa` (GLM-5.2 arch marker; DSA-omitted variant)
- `mlaabsorbed` → `mla_absorbed_decode`/`_prefill` (q_nope absorbed into `kv_b[:nope]`, out into `kv_b[nope:]`, v-cache aliases k-cache)
- `gdnseqprefill` → `gdn_seq_prefill` (Qwen3.6 — sequential bounded-graph scan via `nl.sequential_range`, NOT chunked) ★
- `gdnstateupdate` → `gdn_state_update` (rank-1 recurrent, contract=k·d) ; `gdnconvupdate` → `gdn_conv_update` (depthwise causal-conv1d + silu, slot-indexed)
- `pagedkvgather`/`pagedkvwrite`/`pagedstategather`/`pagedstatescatter` → `paged_kv_*` (unified slab, 32-bit-safe addressing)
- `heteroattn` (sliding-window + global, head_dim≤512) → `attention_cte` with `_MAX_HEAD_DIM` 128→512
- `attndecode_hd_gt128` → `gemma4_attention_decode_dtile` (d-tiled, negated-max online softmax, GQA head-packing)
- `ropeyarninterleaved` → distinguish from split-half RoPE (Ministral3); `ropepartialproportional` → `rope_proportional_partial` (factor 0.25, zero-freq pass-through)
- `logitsoftcaptanh`/`attnsoftcaptanh` → `logit_softcap_tanh` (Gemma-4, cap=30)

**Redundant primitives** (already routed: GatedDeltaNet, DeltaNet, Mamba2, FlashAttention, KDA, MLA, RWKV) — but the concrete on-device kernels + rewrites are still net-new.

---

## nki_knowledge.py additions (idioms / laws / landmines)

**Authoring laws (cross-model, highest value):**
- **Packed-axis DMA discipline (Gemma-4):** many single-slice `nisa.dma_copy` keyed by a scalar index COLLAPSE to the first index under torch-xla → aliases every head to head 0. Fix: one multi-partition DMA (axis on partition) + on-chip `nc_transpose`. *(also live as the lint-per-index-dma rewrite)*
- **`nl.sequential_range` vs `affine_range`:** loop-carried RMW (flash running buffers, GDN scan) MUST use `sequential_range`; `affine_range` lets the compiler reorder → "valid-ranged but WRONG."
- **Sequential > chunked GDN on Neuron (Qwen3.6):** chunked `(I−M)⁻¹` doubling yields bf16 values that differ from the recurrence → **perturbs MoE top-8 routing** + over-runs dispatch; deep Python unrolls hang neuronx-cc. Match the sequential recurrence by construction.
- **32-bit-safe paged addressing:** never form flat `block*page_stride` int32 (overflows ≥2³¹, silently caps ~16383 blocks); pass block_id + pos separately, `.ap offset=0`, engine scales in >32-bit space.
- **Per-partition broadcast:** `tensor_scalar` won't broadcast [1,1] across partitions; build [D,1] via `ones[1,D]ᵀ @ g[1,T]` (K=1, no DGE). `dma_transpose` src innermost ≤128.
- **Flash negated-max online softmax (Gemma-4):** store running_max as −max, apply via activation `bias`; fuse exp+reduce in one `nisa.activation(reduce_op=add, reduce_cmd=reset_reduce)`; NaN-guard via `1e30` sentinel + min-clamp.

**Numerics / layout:**
- **In-kernel FP8-ROW dequant (GLM):** store e4m3 + per-out-channel scale `[1,out]`, expand `[1,D]→[128,D]` at forward for TKG; CTE prefill dequants to BF16 transiently. Scale buffers shaped `[128,·]` (P_MAX) serve both CTE and TKG.
- **fp32-internal RMSNorm (Gemma-4):** normalize AND apply weight in fp32, downcast once. **RMSNormGated uses plain `weight` (ones-init), NOT `(1+weight)`** (Qwen) — else 2× GDN output.
- **e4m3 240 vs 448:** byte-saturate OCP codes >240 onto the ±240 grid (`(b&0x7F)>=0x78 → (b&0x80)|0x77`); do NOT ×240/448 rescale. *(also live as fp8-e4m3-240-saturate rewrite)*
- **Softcap:** fp32 `cap·tanh(x/cap)`; **Partial RoPE:** inv_freq denom = full head_dim, non-rotary dims → zero-freq pass-through (SWA+global share one path).
- **Sliding-window segmented prefill:** never skip prior-KV when tokens≥window (invalid past chunk 0); feed gathered prior KV + `prior_used_len` for dynamic masking. Remap `PAD_SLOT_ID=-1`→block 0 (`null_block`) to avoid last-block wrap corruption.

---

## kernel_rewrites.py — LIVE (8 harvested entries)
`repeat-interleave-to-broadcast`, `gelu-tanh-inline`, `argsort-to-argmax-mask`,
`moe-dge-16-alignment`, `moe-padding-token-dispatch-oob`, `fp8-promotion-to-bf16-cast`,
`inplace-scatter-stale-read`, `fp8-e4m3-240-saturate` — plus the FX/trace traps recorded
inline in those entries (fp8 `.view`→`.contiguous`, enum `==` not `.is_logical_row()`,
compile-folded slot indices → `idx + 0*anchor`, bind state via in-place `copy_`, wrap all
NKI at import not in forward, multimodal→text config rewrite).

---

## Priority ranking (for authoring the actual kernels)

- **P0** — Qwen3.6 GatedDeltaNet hybrid (`add-qwen36-moe`): `gdn_seq_prefill` + `gdn_state_update` + `gdn_conv_update` + `paged_kv_gather` (32-bit-safe) + ~15 rewrites. **Single highest-value branch** (validated sequential kernels + unified-paged-state plumbing we don't have).
- **P0** — GLM-5.2 GlmMoeDsa (`add-glm-5-2`): new arch; DGE-16 pad + MTP einsum fallback + in-kernel FP8-ROW dequant, reusable across any MLA + sigmoid-MoE model.
- **P1** — Gemma-4: hetero SWA/global attn head_dim>128 (raise `_MAX_HEAD_DIM`→512), d-tiled decode, softcap, partial RoPE, DMA-aliasing law.
- **P1** — Ministral3: FP8 e4m3 240/448 saturation, YaRN interleaved RoPE, packed FP8 KV, FX/dtype trace-trap catalog.
- **P2 / redundant** — llama-embed-nemotron (pooling); generic MLA/FlashAttention/Mamba2 primitives already routed (take the rewrites/kernels, skip the primitive entries).
