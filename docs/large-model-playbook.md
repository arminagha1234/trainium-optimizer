# Large-Model Playbook — memory placement, parallelism, and how to problem-solve

Permanent guidance for any agent bringing a LARGE model up on Trainium
(native-PyTorch or vLLM-Neuron). Small models "just work" because everything fits
on a rank; large models fail because **something isn't sharded**. So the method
flips: **solve placement (memory) FIRST, tune for speed second.**

Written after the Qwen3.5-30B MoE OOM (2026-08-29): experts were replicated on
every rank because the TP path only sharded attention + the *dense* MLP. No TP
degree or shape could fix it — the wrong lever can't shard the wrong component.

## The mental model

Per-rank HBM = **weights + KV cache + activations**. Before compiling anything,
decompose it and find the un-sharded bulk. Each component has ONE lever that
shards it:

| Component | Grows with | Lever that shards it | In repo? |
|---|---|---|---|
| Attention (Q/K/V/O) | heads × head_dim | **TP** (by heads; capped at #KV-heads for GQA) | ✅ `backends/qwen38_tp.py` |
| Dense MLP | hidden × intermediate | **TP** (row/col + all-reduce) | ✅ `shard_mlp` |
| **MoE experts** (usually the bulk) | #experts × intermediate | **expert-TP** or **EP** | build (see below) |
| KV cache | layers × kv_heads × seq × batch | **CP** (shard sequence) / smaller batch | partial (CP in configs) |
| Activations | seq × hidden × batch | **CP** / smaller batch | partial |

**No amount of the wrong lever fixes the wrong component.** "No TP fixes it" is the
correct read when the bulk (experts) isn't reachable by TP.

## The parallelism toolkit — which lever, when

- **TP (tensor)** — shards weights *within* a layer, all-reduce per layer. Cuts
  weight memory by `tp`. **Cap:** #attention/KV-heads. Cheap, low-risk, already here.
- **expert-TP** — TP applied to each expert's FFN (shard the intermediate dim).
  Cuts expert memory by `tp`, **no all-to-all**. Cap: expert-intermediate
  divisibility (large → can go high). Reuses the proven `AllReduceLinear` path.
- **EP (expert)** — each rank owns `E/W` experts; all-to-all dispatch→compute→
  combine. Cuts expert memory *and* compute by `W`. **Cap:** #experts. **Risk:**
  cross-core collectives (wedge). Build only when expert-TP's redundant compute hurts.
- **CP (context/sequence)** — shard the sequence. Cuts KV+activation memory for
  long context; does *not* touch weights.
- **PP (pipeline)** — split layers across ranks. Cuts memory by depth; adds a
  pipeline bubble + microbatching. No example in repo → a real build. Last resort
  when TP+expert-TP still don't fit on one node.
- **DP (data)** — replicate the model, split the batch. **Does NOT reduce per-rank
  model memory** — throughput only, when the model already fits.
- **Multinode** — when one node's cores can't supply enough shard factor (e.g.
  122B needs W≈16+). DeepSeek-V4 / Kimi-K3 are multinode in the plan.

## The diagnostic procedure (run for EVERY large model)

1. **Compute the memory budget on paper BEFORE compiling.**
   `weights_bytes ≈ n_params × 2` (bf16). Split into attention / dense / experts /
   embeddings. Add `KV_cache ≈ 2 × layers × kv_heads × head_dim × seq × batch × 2`.
   Two minutes; gives the answer without burning a compile.
2. **Find the un-sharded bulk** — which component × current shard factor still
   exceeds per-core HBM (~24 GB usable/core on trn2; leave headroom for
   activations + fragmentation).
3. **Pick the lever from the table** that shards *that* component.
4. **Recompute GB/rank with the lever applied** → confirm it fits.
5. **Only then compile and measure.** This is the opposite of "try TP degrees until
   one works" — which wastes compiles and can never succeed if the bulk isn't
   reachable by that lever.
6. **Write the budget + the lever into the run log / bank** so the next large model
   reuses the reasoning (compounding).

**Fit math worked example (MoE, expert memory `E` GB):** expert-TP/EP both give
`E / W` per rank. 30B experts 64.4 GB → tp=4 = 16.1 GB/rank ✓ (tp=8 = 8 GB). 122B
231.9 GB → needs W≈16 (14.5 GB/rank) → full core set / multinode.

## The MoE placement fix (the canonical example)

The bug: `qwen38_tp.py`'s `shard_model` gated on `hasattr(L.mlp, "gate_proj")`,
which is False for a sparse MoE block (`mlp.experts` + `mlp.gate`, no top-level
`gate_proj`) → experts left whole on every rank. The DTensor `_DENSE_PLAN` in
`neuron_worker.py` has the same gap (matches `mlp.gate/up/down_proj`, never
`mlp.experts[i].*`).

**The fix — expert-TP, reusing the existing `AllReduceLinear` / `_slice_linear`
pattern:**
```python
def shard_moe(mlp, r, tp):
    for e in mlp.experts:                       # every rank keeps ALL experts,
        ipr = e.gate_proj.out_features // tp     # but only 1/tp of each FFN's intermediate
        e.gate_proj = _slice_linear(e.gate_proj, rows=(r*ipr, (r+1)*ipr))
        e.up_proj   = _slice_linear(e.up_proj,   rows=(r*ipr, (r+1)*ipr))
        d           = _slice_linear(e.down_proj, cols=(r*ipr, (r+1)*ipr))
        e.down_proj = AllReduceLinear(d)
```
In `shard_model`: `if hasattr(L.mlp, "experts"): shard_moe(L.mlp, r, tp)` else the
dense `shard_mlp`. Keep the qwen3_next router rewrites (correctness, orthogonal to
memory). Prefer expert-TP over true EP: it dodges the all-to-all NeuronLink wedge risk.

## How to problem-solve / learn what you don't have (borrow before invent)

- **Copy the mechanical pattern already in the repo** — `qwen38_tp.py:8-30`
  (`AllReduceLinear`, `_slice_linear`) is the template for any weight-sharding lever.
- **Harvest from GPU frameworks** (`kernel_sources.yaml`) — vLLM / SGLang have
  mature, readable expert-parallel code (`FusedMoE`, `all_to_all` dispatch/combine).
  The *algorithm* ports even where the CUDA doesn't (how they compute
  `n_local_experts`, the dispatch mask, the two all-to-alls).
- **Read the model's own HF `modeling_*.py`** — exact module tree, expert count,
  top_k, GQA head counts. Don't guess the structure; the fit math needs it.
- **`docs/vllm-neuron-harvest.md`** — Neuron MoE dispatch patterns + paged/int32
  addressing landmines.
- **Measure, don't guess** — `neuron-explorer` per-engine + the roofline
  (`PEAK_TFLOPS_BF16_PER_CORE=79e12` dense; see `docs/nki-perf-guide.md`).
- **Remember bs=1 decode is HOST-bound** (device >99% idle). For a model that
  *fits*, the win is batching / graph reuse / async scheduling, not a faster
  kernel. Don't over-optimize kernels for a host-bound path.
- **Compound into the bank** — every "model X OOM'd on component Y → lever Z fixed
  it, GB/rank before/after" is a banked lesson for model N+1.
- **Escalate kernel work** — placement/backends is the e2e agent's lane; a
  compiler-weak kernel (GDN/KDA scan, sparse attention, gather-as-matmul) goes to
  the kernel-R&D agent.

## Honest caveats

- **Wedge risk** — any all-to-all / multi-rank collective can HBM-OOM + orphan
  ranks and wedge the box. expert-TP avoids it; EP and multinode don't. Have the
  reaping/kill procedure ready before launching collectives.
- **mxfp4 is trn3-only** — on trn2, mxfp4 experts dequant→bf16 in-kernel, which
  *increases* resident memory vs the mxfp4 footprint. Factor it into the budget.
- **Quantization is a weak trn2 lever** — fp8 is the only sub-bf16 path and is
  unproven vs bf16+compile. Shard MoE memory; don't quantize to solve it.
