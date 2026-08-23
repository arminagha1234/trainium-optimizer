# NKI kernel-optimization playbook — Trainium2 / NeuronCore-v3 (LNC2)

The deep technical reference for writing/optimizing NKI kernels on trn2. Assembled
from the AWS Neuron NKI docs + `aws-neuron/nki-samples` + internal perf guides, and
cross-checked against ~35 real authored kernels and their lesson notes. Every
technique is stated with the **hardware reason**, the **NKI API**, and **when it
applies**. This doc is meant to be *retrieved* by the kernel author (the bank's
`nki_kernel` / `anti_pattern` lessons point back here).

Honesty flags are preserved: **[UNSURE]** = not verified against a doc; **[CONFLICT]**
= public vs internal sources disagree — verify against your SDK.

---

## 0. The hardware you're optimizing against (NeuronCore-v3 / trn2)

3-tier memory (HBM → SBUF → PSUM) feeding 4 async engines + DMA. Every rule below
is a consequence of one of these numbers.

| Resource | NC-v3 (trn2) value |
|---|---|
| PE (Tensor) systolic array | **128×128** @2.4 GHz; FP8 presents **256×128**. **79 BF16 / 158 FP8 / 20 FP32 TFLOPS**. Accumulate **always fp32**. |
| PSUM | **2 MiB = 8 banks × (128 part × 512 fp32)**. Accumulator-only; TensorE write-only (Vec/Scalar may read/write). ≤**8** live accumulation groups. Cannot DMA PSUM→HBM. |
| SBUF | **28 MiB = 128 part × 224 KiB** (v3; was 24 MiB on v2). Software-managed, **no HW cache**. Free-dim ≤64K elems (≤4K in PSUM). |
| Vector (DVE) | 128 lanes; free-axis reductions only (partition axis cannot be reduced). |
| Scalar (ACT) | 128 lanes; computes in fp32; **free dtype cast on in/out**; fuses `op(data*scale+bias)` + a free-axis reduce. |
| GpSimd (Pool) | 8 procs × 128 lanes; iota/affine_select/memset; integrated DMA. |
| DMA | **16 engines/core → 368 GB/s**; ~**1300 ns** cross-engine descriptor cost, **~600 ns** hwdge (v3+). Device: 96 GiB HBM @ 3 TB/s. |
| LNC2 (trn2 default) | fuses **2 physical NC-v3 → 1 logical (`NC_V3d`)**, shared HBM, **SPMD** (`program_id` 0/1). Does **NOT** widen partitions to 256 — each core keeps its own 128. |

---

## 1. Tensor Engine — feed the 128×128 array correctly (the master invariant)

**Rule.** All matmuls go through `nisa.nc_matmul(dst, stationary=[K,M], moving=[K,N])` →
`dst[M,N] = stationaryᵀ @ moving`, fp32 in PSUM. **Both operands put the contraction
dim K on the partition axis** (≤128). `M`≤128 (becomes output partitions), `N`≤512
(one PSUM bank).

**Why.** The array is 128 rows (contraction) × 128 cols. K<128 under-fills the rows
(a 96-wide op uses 96/128 cols). N≤512 = one bank of 512 fp32.

**Two traps:**
- **Wrong-axis / transpose trap** → produces confident *garbage* (the "rel_fro≈1.0"
  symptom [UNSURE as exact doc quote, real in practice]). `nl.matmul`'s arg is
  **`transpose_x` (NOT `lhsT`)** and it's *experimental*; deliver the LHS already
  transposed `[K,M]` or call `nisa.nc_matmul` with the transposed tile as stationary.
- **K>128** → tile K and **accumulate in PSUM** (repeated `nc_matmul` into the same
  tile sums). K<128 → under-filled; pack work to reach 128 (§5 partition vectorization).

**Load-stationary is up to 4× cheaper than moving.** Put the **reused (more-tiles)**
operand *stationary*, the thin/short one *moving*. For matrix-vector / decode (M or
N ≪ 128) map the short vector to *moving*. `A@B = (Bᵀ@Aᵀ)ᵀ` — swapping operands
transposes the output; account for it (and exploit it, §7).

**Win.** 25% → ~99% PE utilization; the wrong-axis case is a correctness bug.

---

## 2. PSUM — accumulate in place, evict once

Keep one PSUM tile as the accumulator across the K-loop; move to SBUF exactly once
(`nisa.tensor_copy`, which also downcasts), then DMA to HBM. **Each live matmul
accumulator = one whole bank; ≤8 live.** Over-allocating spills. This 8-bank limit
*dictates your tiling* (don't keep >8 accumulators live).

---

## 3. SBUF — tile to fit, keep reused operands resident

Tiles `[part ≤128, free]`; working set < **28 MiB (224 KiB/part)**. Keep inner-loop-
reused operands resident instead of re-DMA'ing (there is **no HW cache** — anything
not deliberately kept resident is re-fetched from HBM). **Declare buffers inside the
inner loop, not hoisted** — hoisting can force a spill. Blocking M/N to sit under
budget took the tutorial matmul 87% → 99.85% PE-busy.

---

## 4. DMA — coalesce, go wide, become compute-bound

**Rule.** Few large multi-partition transfers, not many small ones. Target **≥4 KiB
per partition, P=128** (all 16 engines busy). Free-dim to hit 2 KiB/part: fp32=512,
bf16=1024, fp8=2048.

**Why.** Each DMA pays ~1300 ns descriptor/sync (600 ns hwdge). Below ~256 B/part
you're **packet-rate bound**, not bandwidth-bound.

**The packed-axis rule (do NOT DMA a packed axis one slice at a time):**
- **Partition axis must be contiguous** (leading-dim step = full free-dim element
  count); strided partition access is illegal.
- **On-chip transpose beats DMA transpose**: `nl.load()` (wide, contiguous) +
  `nisa.nc_transpose()` instead of `nl.load_transpose2d` (low DMA BW).
- Coalesce adjacent sub-tiles into one DMA.

**DGE modes** (`nisa.dma_copy(..., dge_mode=, engine=)`): `none` (host pre-builds
descriptors, lowest latency, permanent HBM cost), `swdge` (GpSimd builds at runtime —
only mode for dynamic gather/scatter, burns GpSimd), `hwdge` (v3+, ~600 ns, and
**overlaps with prior compute when triggered from the Scalar engine** — hides the cost).

---

## 5. Vector / Scalar / GpSimd — use all 128 lanes, run parallel to the PE

The 4 engines run **independent instruction streams in parallel** — do elementwise/
reductions on Vec/Scalar *while the PE does matmul* (free parallelism). Route ops:
- **Reduce** (sum/max) → Vector `nisa.tensor_reduce(..., axis, negate=)` — **free-axis
  only** (partition can't be reduced → use a ones-matmul, below). `negate=True` fuses ×(−1).
- **Activation + fused affine + reduce** → Scalar `nisa.activation(dst, op, data,
  bias=, scale=, reduce_op=nl.add, ...)` = `op(data*scale+bias)` **plus a free-axis
  reduce for free** (arg is `op`, NOT `func`).
- **Elementwise binary** → Vector `nisa.tensor_tensor(...)` (both inputs can't be in PSUM).
- **Affine/clamp** → `nisa.tensor_scalar(...)` (operands must be fp32; on trn2 Scalar
  only does mul/add — route others to Vector/GpSimd).
- **Index/mask** → GpSimd `nisa.iota`, `nisa.affine_select` (no generic `select`), `nisa.memset`.

**Partition vectorization (Opt #5b, 2×):** each of the 128 partitions = one lane, so a
64-partition op costs the *same wall-clock* as 128 — always use all 128; give each
engine **≥128 elems/partition** (tiny free dims expose ~100-cycle per-instruction
overhead). Fusing multiply+add+exp into one `activation` = **3× lower latency** (Opt #6).

**Ones-matmul for partition-axis reductions:** to sum over the partition axis (no
vector op can), multiply by `ones[K,1]` through the PE — the softmax denominator,
linear-attention normalizers, Sinkhorn colsums.

---

## 6. Pipelining / double-buffering — overlap DMA with compute

Structure loops so tile *n+1*'s DMA overlaps tile *n*'s compute. **Diagnostic:** an
engine waiting on a semaphore whose name **starts with `q`** (`qSyncIO0`,
`qSyncSpillReload0`) is DMA-blocked → prefetch earlier. Express via loop structure +
buffer rotation (no documented `num_buffers` knob [UNSURE]); `hwdge` from Scalar
auto-overlaps descriptor gen. Goal: latency → `max(compute, dma)` not `compute + dma`.

---

## 7. Layout & transpose — pay for a transpose only to buy a single matmul

Choose layouts so the contraction axis is already on partition for *both* GEMMs, so
QKᵀ and P@V are each one `nc_matmul` with no inline transpose. Unavoidable transpose →
on-chip `nisa.nc_transpose` (identity-matmul on PE, ≤128×128; or Vector ≤32×32).
**v3-specific, load-bearing:** a PSUM tile **cannot feed the stationary port on gen3+**,
so `nl.matmul(transpose_x=False)` (which lands the transpose in PSUM) forces you to
**pre-transpose into SBUF**. From v3 all TensorE transposes are bit-accurate.
**Opt #7/#8:** because swapping stationary/moving transposes the output, make the
matmul emit the layout the *next* op wants (map the weight to *moving* so linear→norm
output lands `[hidden, seq]` — no transpose).

---

## 8. Numerics for speed + correctness

- **bf16-in / fp32-accumulate** (~4× vs fp32; fp8 ~2× more, 256-wide contraction).
  Cast to bf16 only at the final store — the Scalar engine's embedded cast pipelines it free.
- **Max-subtract before exp**: `tensor_reduce(op=max, negate=True)` → pass as the
  `bias` of `activation(op=exp, reduce_op=add)` = `exp(x−max)` + running-sum in one
  Scalar instruction, overflow-safe.
- **Clamp/mask BEFORE exp** to avoid `exp(+big)=inf` then `inf·0(mask)=NaN`
  (`exp(nl.minimum(diff,0))`, or add −1e30 in the masked region before the row-max).
- **Delayed division**: apply `1/sum` to the final PV output (`nisa.reciprocal`), not
  to every score.
- **Stable softplus**: `relu(x)+log1p(exp(−|x|))` (naive overflows ~x>88).
- **[scope]** the public nki-samples attention uses **two-pass** (full-row) softmax,
  NOT online/flash streaming — write the running-max rescale yourself if you need it.

---

## 9. Loop structure — affine indexing, the right range

- Tile indices must be **affine** (`offset + channel*mult + Σ idx*step`) — never
  `floordiv`/`mod` a loop index (forces gather/scatter). For GQA/GVA nest `(h_kv, g)`
  and compute `h = h_kv*G + g`. Use `nl.mgrid[0:128, 0:512]` and `nl.ds(start, size)`.
- `nl.affine_range` = no loop-carried dep (compiler may reorder/pipeline);
  `nl.sequential_range` = enforce order (reductions/state). **[CONFLICT]** the
  `/latest/` API now documents both as `range()` aliases marked "deprecated, prefer
  `range()`" — but nki-samples still use them for dependency intent; check your SDK.
- **`nl.arange` only over fixed constants** (`nl.arange(128)`); a shape-derived size
  errors ("'Index' object cannot be interpreted as an integer").

---

## 10. Roofline / profiling — diagnose before optimizing

Use **Neuron Explorer** (`neuron-profile`): `NEURON_FRAMEWORK_DEBUG=1 XLA_IR_DEBUG=1`,
compile to `graph.neff`, `neuron-explorer capture -n graph.neff -s p.ntff
--profile-nth-exec=2`, `view`. Classify:
- **Compute-bound** (an engine ~100% active; matmul target ~100% MFU) → §1/§5/§7. To
  saturate bf16 TensorE you need ~**222 Flops/Byte** arithmetic intensity — below
  that you're memory-bound (blocking took the tutorial 102 → 683 Flops/Byte).
- **Memory-bound** (device MBU ~100%) → §3/§4/§11.
- **DMA-blocked/serialized** (waiting on a **`q`-prefixed** semaphore) → §6 prefetch.
  Spill bytes >30% of SBUF traffic → fuse.
- **CC/collective-bound** → no documented profiler class or fix [UNSURE].

Bench framework-free with `nki.benchmark(warmup=10, iters=100)` → P50/P99 (µs; no
mean); it does **not** use real inputs (don't check accuracy from it).

---

## 11. Fusion — one kernel, fewer HBM round-trips

Fuse producer→consumer so intermediates never touch HBM (attention `matmul→softmax→
matmul`; norm+quant; linear→norm→linear). Diagnose via `spill_*_bytes` > 30%. Declare
the fused intermediate **inside** the inner loop or it spills. **RMSNorm pattern:**
tile tokens→partition, hidden→free; `activation_reduce` (square+sum free); `activation`
does eps+scale+rsqrt; **broadcast gamma via a TensorE matmul-against-ones to keep the
idle PE busy**; 3-engine pipeline (RMS on Scalar, gamma-broadcast on Tensor, apply on
Vector) — no transpose.

---

## 12. The structural win for linear-attention/SSM — the chunked "dual"

Turn the O(T) recurrent scan into intra-chunk dense matmuls + a compact inter-chunk
state carry. Shared skeleton across DeltaNet/GDN, Mamba2-SSD, LightningAttn, RWKV6/7,
mLSTM, PowerRetention, RG-LRU, KDA.
- Split into chunks of length C; process C tokens as a `[C×C]` decay-masked attention-
  like Gram matmul; carry only the `[K,V]`/`[N,P]` state across ⌈T/C⌉ **sequential**
  steps. Turns T matrix-vector ops (~1/128 PE at bs=1) into PE-saturating block matmuls.
- **Cumulative decay = triangular matmul**: `cum = nc_matmul(tri_le, log_decay)`.
- **Delta-rule solve = log-depth nilpotent inverse**: `(I+L)⁻¹ = Σ(−L)^p` (L strictly-
  lower/nilpotent → exact + finite). Do NOT python-unroll (C−1≈127 matmuls → ~35-min
  compile); use **repeated doubling** (`U_{k+1}=U_k+N^{2^k}U_k`, ~`3·⌈log₂C⌉≈21` matmuls).
- **Chunk size C** is a numerics-vs-parallelism-vs-compile lever (C≤128, T%C==0). Too
  large → fp32 overflow → all-NaN. Safe span depends on decay strength (KDA: **C≲176/
  |log a_min|**; practical defaults GDN/Lightning ≤128, mLSTM 64, RWKV 16, PowerRet 8).
- **Why it's mandatory, not just faster**: neuronx-cc **unrolls `sequential_range`**,
  so a per-token prefill grows the NEFF ~linearly in T (~780–880 B/token) and blows the
  **5M-instruction budget → NCC_EBVF030 "graph too big"** over many layers at ctx≥2048.
  The chunked form is ~9.7× flatter and compiles in ~200s. Auto-dispatch to the chunked
  kernel at chunk-count ≥4.

Decode (T=1) stays on the recurrent kernel with state `S[K,V]` in SBUF (no HBM round-
trip/token); a 1-step chunk *is* the recurrence.

---

## 13. Anti-patterns / gotchas (trap → symptom → fix)

The hard-won bugs. Many are **fake-GREEN** — they compile and run on Neuron but are
wrong; several are **simulate-invisible** (numpy `nki.simulate` passes, real silicon
fails) — this is *the* reason on-device is the only authoritative gate.

1. **Immutable output param.** `nl.store` into a passed `out` → prod neuronxcc "Cannot
   update immutable parameter". Fix: `out is None` → allocate `nl.shared_hbm` + RETURN;
   `out` passed → write in place (dual-SDK convention).
2. **Wrong matmul contraction axis → rel_fro≈1.0** (§1). K on partition for both operands.
3. **Tile REBIND vs in-place write.** `acc = nl.add(...)` rebinds a fresh tile → post-
   loop store reads original zeros / drops high-order Neumann powers (GDN out_rel→0.98).
   Fix: always `acc[...] = ...` in place. *Latent* where L entries are tiny → a
   production-chunk-only bug.
4. **Bare partition-0 broadcast reads STALE SBUF on real HW (simulator-invisible).**
   `multiply(S[K,V], scalar[1,1])`: numpy-sim replicates & passes, silicon reads stale
   partitions 1..K-1 → coherent gibberish. Fix: lift via `nc_matmul(ones[1,K], scalar)`
   first (per-partition free broadcast is HW-safe). **This is the load-bearing reason
   rank-3(sim) ≠ rank-4(device).**
5. **0·inf → NaN in decay/score exp** → clamp/mask before exp (§8).
6. **Chunk-too-large fp32 overflow → all-NaN** (not graceful) → the chunk-envelope law
   (§12); the clamp lever is dead; only nested sub-chunking grows C safely.
7. **Floordiv on a loop index** → tracer rejection → affine nesting (§9).
8. **Single-partition slice at offset≠0** (`cum[C-1:C]`) → BIR StreamShuffle rejection →
   get last-token via a partition-reduction matmul or one-hot; slice free-axis only.
9. **Load-bearing default scale (fake-GREEN).** `scale=1.0` default → √K×-too-large
   finite output. Fix: `scale=None → K**-0.5`. (But check the *model-correct* value —
   for some linear-attn 1.0 IS correct.)
10. **`nl.arange` over a runtime size** → fixed-128 arange + `affine_range` for counts.
11. **`nki.baremetal` can't prove rank-4 on prod Trn2** (immutable-param or all-zero
    buffer) → the real device gate is `@nki.jit` called directly on `xm.xla_device()`
    tensors with output omitted (allocate-and-return); ~1e-6 diffs prove the store landed.
12. **rank-3(sim) vs rank-4(device), and rank-3.5** (sim+neff but `libnrt` absent so
    never executed) — only rank-4 closes the simulate-invisible class (#4, #11).
13. **Toolchain PATH shadowing.** A stale `neuronx-cc 2.9` first on PATH rejects
    `--internal-tensorizer-opt-level=nki` → a CORRECT kernel mis-ranked failed. Self-heal
    PATH (version-matched bin first); use a clean artifacts dir.
14. **Silent remainder-tile drop** on unaligned dims (`M//BLK`, M=130 → rel 0.18) →
    assert 128-alignment at trace time; pad on host.
15. **Wrong scale-grid / quant packing (fake-GREEN)** → guard exact scale-grid + packing;
    the HF `[out,in]`→contract-K-on-partition transpose is load-bearing.
16. **MoE per-expert custom-call explosion (NCC_EMOD018)** → put the expert axis E
    INSIDE one kernel as a static `affine_range(E)`.
17. **State-handoff contract (fake-GREEN).** Dropping/transposing recurrent/conv state
    between prefill↔decode → gibberish once state≠0. Gate: `chunked-prefill[:k] →
    decode[k:] == full-run`; pad the chunk tail with state-identity tokens. SWA decode:
    mask evicted (older than pos−W+1) and stale-padding keys.
18. **Non-vacuous test defeaters.** An online-softmax that overwrites (not accumulates)
    per key-block passes single-block tests → every test must perturb an EARLY block and
    require the output to change; drop the mask and require parity to FAIL.
19. **`torch.topk` → XLA `sort` unsupported on trn2 (NCC_EVRF029)** → sort-free top-k
    (iterative argmax + iota-mask). This unblocks Qwen3.5 GatedDeltaNet-MoE — pure graph
    rewrite, no kernel. (In our `kernel_rewrites` catalog as `topk-sort-to-argmax`.)
20. **`.tril()`/`.triu()` runtime affine-select** can trip `TensorScalarAffineSelect`
    ISA validation at scale → host-materialized constant mask × elementwise multiply.
    (Catalogued as `tril-to-const-mask`; note: NOT the full-model Qwen3.5 blocker on
    neuronx-cc 2.27.5334 — #19 is.)
21. **SDK misc:** `where()` operands must be tiles not python floats; no
    `right_shift`/`select`/`iota`-LUT (nibble/E2M1 unpack done arithmetically); `range()`
    inside a traced body symbolizes into an affine loop (build fixed layouts in a
    module-level helper returning python ints); `nki.simulate_kernel` caches by
    (module,fn,shapes) — import under a uuid module name for a real regression gate;
    fp8 loads must declare `dtype=nl.float8_e4m3` explicitly.

---

## 14. Per-primitive "the one thing that made it work"

- **DeltaNet / GatedDeltaNet** — the UT/WY chunked dual with the **log-depth nilpotent
  inverse** (§12) + in-place tile writes (#3). 60/60 on NeuronCore-v3, bf16 max|Δ|≤8.3e-4.
- **Mamba2-SSD** — the **state-space/attention duality**: scalar-per-head decay makes
  the recurrence a causal-masked `[C×C]` attention matrix; only `[N,P]` state carries.
- **Flash/SWA attention** — **scores kept transposed** (both GEMMs single matmuls) +
  max-subtract-as-bias stable softmax + **key-tiling under the 512 PSUM cap** (long
  context needs it on BOTH prefill and decode).
- **Attention-sink** = fold `exp(sink_h)` into the denominator once (row-constant, no
  value). **ALiBi** = softmax-invariant abs offset → decode needs only `slope·j`.
  **Blockwise-FP8** = factor per-token/per-block scales OUT of the 128-contraction inner
  sum → raw fp8×fp8 at native rate, then apply scales. **MXFP4** = in-kernel arithmetic
  E2M1 decode + K-half packing + E8M0 stride-0 scale broadcast. **MoE** = static
  `affine_range(E)` + fp32 cross-expert accumulate. **KDA** = accept per-channel
  data-dependent decay (per-head scalar collapse is a 0.14/layer fake-GREEN); CHUNK=16.

---

## Top 10 highest-impact optimizations

1. Right stationary/moving + full-128 contraction; never trip the `transpose_x` trap (§1).
2. **Profile first** — compute vs memory vs `q`-semaphore DMA-blocked (§10).
3. **Fuse** to kill HBM round-trips, esp. attention scores + norm chains (§11).
4. **Wide coalesced DMAs**: ≥4 KiB/part, P=128, contiguous partition axis (§4).
5. **bf16-in / fp32-accumulate** (and fp8 where safe) (§8).
6. **Transpose-free attention layout**; pre-transpose scores once on-chip (§7).
7. **Engine balancing / instruction fusion** — one `activation` for scale+bias+exp+sum (§5).
8. **Partition vectorization** — all 128 lanes, ≥128 elems/part (§5).
9. **PSUM discipline** — accumulate in place, ≤8 banks, evict once (§2).
10. **Chunked dual for linear-attention** + double-buffer DMA behind compute (§12, §6).

---

## API corrections worth flagging to the author

- The matmul arg is **`transpose_x`**, not `lhsT`; `nl.matmul` is *experimental* —
  prefer `nisa.nc_matmul` with an already-transposed stationary tile.
- `nisa.activation`'s function arg is **`op`**, not `func`.
- `affine_range`/`sequential_range` are being **deprecated toward `range()`** in
  `/latest/` — still express dependency intent, but check your SDK version.
- Public nki-samples softmax is **two-pass**, not online/flash — don't assume streaming.
- **LNC2 does NOT give 256 partitions** — each physical core keeps 128; LNC changes
  sharding/shapes + adds compiler-managed inter-core comms (and LNC1 is *sometimes*
  faster — benchmark both).

---

## Sources
AWS Neuron NKI docs at `awsdocs-neuron.readthedocs-hosted.com/.../nki/` (arch guides,
`nki_perf_guide`, `nki-dma-bandwidth-guide`, `nki-dge`, `nki-aps`, `use-neuron-profile`,
tutorials matmul/attention/transpose), the `nki.isa`/`nki.language` API reference,
`github.com/aws-neuron/nki-samples`, internal NKI Performance Guide + Waimea/CoreIdSpaces
wikis, and ~35 authored kernels + lesson notes in the Neuron AutoFixer corpus (techniques/
math/lessons only — no proprietary source reproduced).
