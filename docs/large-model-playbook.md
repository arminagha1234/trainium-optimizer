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
| **Host DRAM during load** | model size × **ranks** | **shard-on-read loader** | ❌ build (see below) |

**No amount of the wrong lever fixes the wrong component.** "No TP fixes it" is the
correct read when the bulk (experts) isn't reachable by TP.

Note the last row: it is not HBM, and on a trn2.48xlarge it is the wall you hit
**first**. See "The wall before HBM" below.

## The parallelism toolkit — which lever, when

- **TP (tensor)** — shards weights *within* a layer, all-reduce per layer. Cuts
  weight memory by `tp`. **Cap:** #attention/KV-heads. Cheap, low-risk, already here.
- **expert-TP** — TP applied to each expert's FFN (shard the intermediate dim).
  Cuts expert memory by `tp`, **no all-to-all**. Cap: expert-intermediate
  divisibility (large → can go high). Reuses the proven `AllReduceLinear` path.
- **EP (expert)** — each rank owns `E/W` experts. Cuts expert memory *and* compute
  by `W`. **Cap:** #experts. ✅ **Built: `backends/moe_ep.py` (#125)**, and it needs
  **no all-to-all**: a rank computes only its own experts, leaves zeros elsewhere,
  and ONE `all_reduce` sums the partial mixtures — exact, because every expert
  contributes on exactly one rank. That is the same collective expert-TP would use,
  so EP now has expert-TP's risk profile with `1/W` the expert compute. Prefer it.
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
memory).

**What actually shipped is real EP, not expert-TP.** The all-to-all that made true
EP look risky is avoidable: keep the router global, give each rank a disjoint expert
range, and `all_reduce` the partial mixtures once per layer. `backends/moe_ep.py`
does that in ~130 lines and is numerically exact (CPU equivalence at atol 1e-6 in
`test_moe_ep.py`). Qwen3.5-35B-A3B experts went 64.4 GB/rank → 4.0 GB/rank at tp=16
and the load stopped OOMing. Fused 3-D expert tensors
(`experts.gate_up_proj`, `experts.down_proj`) are the modern HF layout — there is no
`mlp.experts[i].gate_proj` to loop over, so slice the leading expert dimension.

**Second bug, uncovered by fixing the first: GQA KV heads sliced to zero.**
Qwen3.5-35B-A3B has 16 query heads and **2** KV heads. Expert memory forces tp=16,
and `shard_attention` computed `nkv // tp` = 0, so `k_proj`/`v_proj` became
zero-width. Nothing raised at shard time; the model died six minutes later inside
attention with `size of tensor a (2) must match the size of tensor b (0)`. Fix:
derive the KV heads a rank needs from the query heads it owns and **replicate** when
`tp > nkv` (`kpr = max(1, (qpr*nkv)//nh)`, `kv_start = (r*qpr*nkv)//nh`), then set
`num_key_value_groups` from the LOCAL shapes. `_slice_linear` now also raises on an
empty slice, so this class of bug fails at the seam instead of six layers later.
**Consequence for the table above: attention TP is NOT capped at #KV-heads.** It is
capped at #query-heads; below `nkv` you shard KV, above it you replicate KV.

## The wall before HBM: host DRAM × ranks

Placement reasoning that stops at HBM misses the constraint that actually killed
the two biggest models. `AutoModelForCausalLM.from_pretrained` materialises the
**whole model in the calling process**, and tensor parallelism runs **one process
per core**, with sharding applied only after the model object exists. So the
transient peak is `ranks × model_size` — every rank holds a full private copy at
the same moment.

trn2.48xlarge: 64 cores × 24 GB = 1536 GB HBM, **2147 GB host DRAM**
(`/proc/meminfo`, measured 2026-08-29).

| ranks | HBM allows (14.4 GB/core budget) | host DRAM allows | binding |
|---|---|---|---|
| 4  | 58 GB  | 537 GB | HBM |
| 8  | 115 GB | 268 GB | HBM |
| **16** | 230 GB | **134 GB** | **crossover** |
| 32 | 461 GB | 67 GB  | host |
| 64 | 922 GB | 34 GB  | host |

**The two budgets pull in opposite directions.** HBM per rank wants more ranks;
host DRAM wants fewer. That is why "try a bigger tp" could never resolve these
models — past 16 ranks it makes things strictly worse. It also explains why the
observed failures were indistinguishable: they were not device errors at all.

| model | weights | ranks | host peak | HBM/rank | observed |
|---|---|---|---|---|---|
| Qwen3.8-27B | 54 GB | 4 | 216 GB ✓ | 13.5 GB ✓ | ran, 344 tok/s |
| Qwen3.5-35B-A3B | 71.9 GB | 16 | 1150 GB ✓ | 4.5 GB ✓ | loads (after EP) |
| Qwen3.5-122B-A10B | 250 GB | 32 | **8006 GB ✗** | 7.8 GB ✓ | OOM |
| DeepSeek-V4-Flash | 319 GB (fp8→bf16) | 64 | **20429 GB ✗** | 5.0 GB ✓ | OOMKilled-137 |

HBM per rank was **fine in every case**. Expert parallelism alone therefore does
NOT unblock 122B or DeepSeek — they need the loader fixed.

`capability.py` models this: `host_load_peak_gb(weight_gb, ranks, lean_loader=)`
and a `HOST_LIMITED` verdict that says so instead of reporting TOO_LARGE (which
would send the next person off to raise tp again). `host_ram_gb` is populated only
for boxes where it was actually read off `/proc/meminfo`; elsewhere it is 0.0 and
the check is skipped, because an invented DRAM size would reject models on no
evidence.

### The ceiling on one trn2.48xlarge

`capability.ceiling()` sweeps the rank counts and names the binding constraint:

| loader | best ranks | weights | ≈ params (bf16) | binding |
|---|---|---|---|---|
| today (eager `from_pretrained`) | 16 | 134 GB | **~67B** | host DRAM |
| shard-on-read | 64 | 922 GB | **~460B** | HBM per rank |
| shard-on-read, 2 nodes | 128 | 1843 GB | ~920B | HBM per rank |

So **fixing the loader is worth ~7× in reachable model size** and is the single
highest-leverage change for large models. A 1T model in bf16 is 2000 GB of weights
against 1536 GB of HBM — it does not fit one node at any tp, with any loader. 400B
–500B is the honest single-node target; 1T needs 2–3 nodes or weights held
quantized *on device* (and on trn2, fp8 checkpoints currently dequantize to bf16 at
load, which removes the saving).

### Building the shard-on-read loader

Peak host DRAM becomes `model + ranks × one_file` instead of `ranks × model`:

1. Build the model on `meta` (`init_empty_weights` / `torch_dtype` + `low_cpu_mem_usage`).
2. Iterate the safetensors shards, one file open at a time.
3. For each tensor, slice THIS rank's portion before materialising it — reuse
   `moe_ep.expert_shard_plan` for `experts.*` and the `qwen38_tp` head math for
   attention/MLP — then copy straight to the device.
4. Release the file before opening the next.

This is what vLLM does, and it is the prerequisite for everything past ~67B here.

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

- **Wedge risk** — any multi-rank collective can HBM-OOM + orphan ranks and wedge
  the box. `moe_ep.py` uses a plain `all_reduce` (no all-to-all), so it is no worse
  than the attention/MLP path already in use; multinode still carries the risk.
  Have the reaping/kill procedure ready before launching collectives.
- **A zero-width shard fails silently and late.** `nkv // tp == 0`, an expert range
  narrower than one expert, an intermediate dim not divisible by tp — all succeed at
  shard time and surface as an unrelatable shape error deep in a forward, often on
  one rank only. Guard at the slice, not at the symptom.
- **Do not verify a shard against your own reimplementation of the module.**
  `test_moe_ep.py` compared sharded vs unsharded against a local `RefExperts`; it
  was faithful, so the tests were sound — and they still could not catch the KV bug,
  because that lived in a different module at a head geometry the reimplementation
  did not model. `test_qwen38_tp_geometry.py` builds a real (tiny)
  `Qwen3_5MoeForCausalLM` at the REAL head counts and shards it rank by rank. Shrink
  depth and width; never shrink the thing under test.
- **mxfp4 is trn3-only** — on trn2, mxfp4 experts dequant→bf16 in-kernel, which
  *increases* resident memory vs the mxfp4 footprint. Factor it into the budget.
- **Quantization is a weak trn2 lever** — fp8 is the only sub-bf16 path and is
  unproven vs bf16+compile. Shard MoE memory; don't quantize to solve it.
