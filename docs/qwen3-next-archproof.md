# Qwen3-Next (GatedDeltaNet-MoE) — Trainium architecture proof

HONEST SCOPE: this is a tiny, random-init faithful down-scale of the Qwen3-Next /
Qwen3.5 architecture (model_type qwen3_next, transformers 5.15.0, config keyed to
Qwen/Qwen3-Next-80B-A3B-Instruct). It is NOT a pretrained-checkpoint benchmark — the
released Qwen3-Next is 80B-A3B, far too large for a trn2.3xlarge single 24GB core. The
goal (and result) is to prove the ARCHITECTURE compiles + runs + is numerically correct
end-to-end on trn2, which it had never done (it was the flagship compiler-cannot-lower
model). Do NOT merge into the verified-pretrained LEADERBOARD.md as a real checkpoint.

## Tiny config (faithful to the real arch)
- hidden=256, 4 layers, layer_types = [linear_attention x3, full_attention x1]
  (real pattern: full_attention every 4th layer, 3:1 linear:full — both mixers present)
- full attention: 4 q heads / 2 kv heads, head_dim 64
- GatedDeltaNet (linear attn): 2 key / 4 value heads, k/v head_dim 32, conv kernel 4
- MoE: 8 experts, top-2, moe_intermediate 64, shared expert, norm_topk_prob
- vocab 512, seq 128, bf16, TP=1, ~2.71M params

## Blockers found + fixes (all pure graph rewrites, no kernel)
1. MoE router torch.topk -> XLA sort -> NCC_EVRF029 (sort unsupported on trn2).
   FIX: sort-free iterative-argmax router (k rounds of masked .max). Catalog entry
   topk-sort-to-argmax. CONFIRMED: this cleared EVRF029. Exact on CPU (maxdiff 0).
2. HF MoE expert-grouping torch.sort(expert_ids) int64 -> NCC_EVRF013 (AwsNeuronTopK
   rejects int64). FIX: install_neuron_safe_moe_topk (int->fp32 view). Cleared the crash.
3. GatedDeltaNet torch_chunk_gated_delta_rule .tril() (modeling_qwen3_next.py:418) ->
   TensorScalarAffineSelect NCC_IINAR001 (s2d2_ts_as_valid_elem_count). FIX:
   tril-to-const-mask (multiply by constant lower-tri mask). Exact on CPU. Cleared it.
   NOTE: at THIS scale .tril IS a blocker — contra the memory note "tril compiles fine".
4. After 1-3 the model COMPILED but was numerically WRONG (cosine 0.75). Isolated to the
   MoE expert path: HF grouped_mm_experts_forward (sort+histc+grouped_mm, through the
   int64->fp32 sort patch) is numerically WRONG on trn2. Full and linear attention were
   both correct in isolation (cosine 0.99998 / 0.99999). FIX: sort-free static-shape
   dense expert dispatch (compute all experts, weight by scattered gate). Exact on CPU.

## Result (trn2.3xlarge, neuronx-cc 2.27.5334, native_venv torch-neuronx 2.9, bf16, TP=1)
- Compiles end-to-end: YES (~92s), valid NEFF, no ISA errors.
- Correctness vs CPU-bf16 reference: cosine 0.99793, top-1 14/16, argmax-agree 96.1%.
  (bf16 noise floor: CPU-bf16 vs CPU-fp32 is itself 14/16 — correct to bf16 precision.)
- Baseline eager bf16:              743.4 tok/s  (172 ms / 128-tok prefill)
- Optimized torch.compile(neuron):  10,721 tok/s (11.94 ms)
- SPEEDUP 14.42x, correctness-gated.

## Linear-attention (GatedDeltaNet) resolution
- DeltaNet NKI kernel: NOT available on this install (TRN_OPT_KERNEL_DIR empty;
  kernel_registry has no DeltaNet manifest). flash_attn kernel is softmax attn, not a
  drop-in for gated-delta recurrence.
- HOWEVER the reference torch torch_chunk_gated_delta_rule + tril-const-mask rewrite
  COMPILES and is NUMERICALLY CORRECT on-device (cosine 0.99999 in isolation). So at this
  scale the linear-attention path does NOT require the DeltaNet kernel — the compiler
  lowers the reference recurrence correctly once .tril is de-affine-selected. The DeltaNet
  kernel remains the expected PERFORMANCE path at scale; the chunk recurrence 64-iter
  in-place-scatter loop is a scaling risk.

## HISTORY.tsv-format row (arch-proof, NOT a pretrained-checkpoint entry)
2026-08-24T00:00:00Z	1	qwen3-next-tiny-archproof	tiny:qwen3_next(random-init)	743.4	10721.3	14.42	0	arch-proof-bf16-correct	graph_rewrite	14/16(bf16-floor);cosine0.998	baseline+sortfree_router+int64_topk+tril_constmask+dense_moe	2.27.5334
