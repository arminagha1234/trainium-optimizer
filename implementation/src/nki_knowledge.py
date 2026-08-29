"""nki_knowledge.py — an OP-INDEXED knowledge & worked-example base for the
kernel author (Pillar 1 of making the author *skilled*).

The author preamble in ``kernel_author`` (``_NKI_PREAMBLE`` / ``_PERF_PREAMBLE``)
is the STANDING contract: it applies to every op equally. That is necessary but
undifferentiated — an attention kernel and an elementwise softcap get the exact
same prose. This module adds the missing half: **op-aware retrieval**. Given the
op being authored, it returns the *relevant* verified techniques, the *correct*
NKI-0.6.0 return-form signatures the op actually needs, the *specific* landmines
that bite that op family, and 1-2 short **worked-example** snippets that are
on-device-idiom-correct.

Provenance (why these are trustworthy, not invented):
  * The elementwise / reduction / normalization / matmul / broadcast snippets are
    distilled from ``docs/verified_nki_idioms.py`` — kernels that COMPILED and
    matched a numpy reference on a real NeuronCore (nki 0.6.0 / neuronx-cc 2.27 /
    trn2). Each idiom's on-device verdict is cited in the example ``note``.
  * The softmax / attention / scan / MoE-router snippets are distilled from the
    ALGORITHM STRUCTURE of the AWS expert kernels (nki-samples
    ``attention_fwd_performance``, nki-library ``experimental/scan/ssd.py`` +
    ``core/router_topk``), then **translated into the verified 0.6.0 return-form**.
    IMPORTANT: the public nki-samples use the OLD ``dst=`` out-parameter form
    (``nisa.nc_matmul(dst=..., stationary=..., moving=...)``); on THIS stack that
    is WRONG — 0.6.0 nc_matmul / nc_transpose / activation RETURN a tile and take
    NO ``dst=``. The examples here use the return-form the framework verified.

Design: a PURE-PYTHON data module. No heavy imports (no numpy/torch/nki), no I/O,
no side effects — so it is trivially unit-testable and cheap to import inside the
prompt builder. Everything is plain dataclasses / dicts / strings.

Public surface:
  * ``classify_op(name, family=None, notes=None) -> str`` — op -> op-family key.
  * ``retrieve(spec_or_name, family=None, notes=None) -> KnowledgeEntry`` — the
    matched knowledge entry (techniques + signatures + landmines + examples).
  * ``render_knowledge_section(entry) -> str`` — the prompt block the author sees.
  * ``TECHNIQUES`` / ``LANDMINES`` / ``SIGNATURES`` / ``KNOWLEDGE`` — the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# op-family keys
# ---------------------------------------------------------------------------
ELEMENTWISE = "elementwise"
REDUCTION = "reduction"
NORMALIZATION = "normalization"
MATMUL = "matmul"
SOFTMAX = "softmax"
ATTENTION = "attention"
SCAN = "scan"          # scan / SSM / linear-attention (Mamba-2, GatedDeltaNet, KDA)
MOE_ROUTER = "moe_router"
INDIRECT_GATHER = "indirect_gather"   # gather/interpolate/resample/scatter w/ a STATIC
                                      # access pattern — win = gather-as-matmul on TensorE

OP_FAMILIES = (
    ELEMENTWISE, REDUCTION, NORMALIZATION, MATMUL,
    SOFTMAX, ATTENTION, SCAN, MOE_ROUTER, INDIRECT_GATHER,
)


# ---------------------------------------------------------------------------
# technique registry — the verified levers, keyed so entries reference by name
# (cross-references the 9 optimization techniques in kernel_author._PERF_PREAMBLE
# and the on-device idioms in docs/verified_nki_idioms.py).
# ---------------------------------------------------------------------------
TECHNIQUES: dict[str, str] = {
    "isa-return-form":
        "nc_matmul / nc_transpose / activation RETURN a tile in NKI 0.6.0 — there "
        "is NO dst=/out= param. ASSIGN the return value (`psum = nisa.nc_matmul("
        "stat, mov)`). A dst= 3rd positional errors 'too many positional args'.",
    "tile-to-hw-limits":
        "Respect the systolic-array limits: contraction K (partition) <= 128, "
        "stationary free M <= 128, moving free N <= 512 (one PSUM bank). Tile any "
        "larger free dim into <=512 chunks and loop, accumulating in PSUM.",
    "loop-fusion":
        "Fuse the whole op into ONE kernel: one load per input, intermediates stay "
        "in SBUF, one store of the output — never round-trip a temporary through "
        "HBM (there is no HW cache; a spilled intermediate is pure loss).",
    "activation-reduce-fusion":
        "nisa.activation_reduce(op, data, *, reduce_op, reduce_res, bias, scale) "
        "computes op(scale*data+bias) AND a fused free-axis reduce in ONE Scalar-"
        "engine instruction, WRITING the [P,1] reduction into the ``reduce_res=`` "
        "out-param (it has NO return form — allocate reduce_res first). "
        "rmsnorm: `ms = nl.ndarray((P,1), dtype=nl.float32, buffer=nl.sbuf); "
        "nisa.activation_reduce(op=nl.square, data=x, reduce_op=nl.add, "
        "reduce_res=ms[...])` gets the sum-of-squares in one pass; softmax: "
        "`nisa.activation_reduce(op=nl.exp, data=x, bias=neg_rowmax, "
        "reduce_op=nl.add, reduce_res=den[...])` gives the denominator at once. "
        "Don't materialize a squared/exp tile then sum it separately, and do NOT "
        "use `nisa.activation(..., reduce_op=)` (return-form) for the reduce — on "
        "trn2/neuronx-cc 2.27 that returns the FULL activation, not the reduction "
        "(NCC_INIC902 / store-shape error); the reduce MUST go through "
        "``activation_reduce`` with ``reduce_res=`` (on-device validated 2026-08-27, "
        "idiom per internal NKI-Autotune kernel_library/rmsnorm_linear.py).",
    "delayed-softmax-division":
        "Reduce to a [P,1] denominator, take its reciprocal ONCE (nisa.reciprocal "
        "or nl.reciprocal), and multiply the result at the END — never divide "
        "every element mid-stream. Same shape rule for the numerator's row-max.",
    "keepdims-2d":
        "Reductions MUST keep the tile >= 2-D: `nl.sum(x, axis=1, keepdims=True)` "
        "-> [P,1], never a 1-D collapse. Applies to nl.max / nl.sum / tensor_reduce.",
    "explicit-broadcast":
        "NKI forbids an IMPLICIT partition broadcast (`t[P,F] + iota[1,F]` or "
        "`t[P,F] * col[P,1]`). Broadcast the [1,F]/[P,1] operand to the FULL [P,F] "
        "shape first: `nl.broadcast_to(t, shape=(P,F))` — shape= is KEYWORD-only.",
    "hoist-invariant":
        "HOIST loop-invariant loads (gamma / beta / cap / row-max operands) OUT of "
        "the tile loop — there is no HW cache, so a per-tile re-DMA of an invariant "
        "is wasted bandwidth. Load once, keep resident in SBUF, reuse every tile.",
    "bf16-in-fp32-accumulate":
        "Read bf16, accumulate in fp32 (PSUM / Scalar are fp32), cast back to bf16 "
        "only at the FINAL store. Do the output dtype cast on the HOST after .cpu() "
        "— an in-graph device cast can trip the neuronx-cc Simplifier (NCC_ISMP902).",
    "psum-native-accumulation":
        "Accumulate matmul partials natively in PSUM across the K/N tiling loop "
        "(the systolic array writes PSUM) rather than summing SBUF copies — evict "
        "to SBUF only once the accumulation for that output tile is complete.",
    "downcast-before-transpose":
        "When a transpose feeds a matmul, downcast to bf16 BEFORE nc_transpose so "
        "the transpose moves half the bytes and the matmul reads its native dtype.",
    "engine-overlap":
        "Keep the PE busy: route the reduce to the Scalar engine and the "
        "elementwise apply to the Vector engine so they run concurrently, and "
        "broadcast an invariant via a TensorE matmul-against-ones — a 3-engine "
        "pipeline, not one serial engine doing everything.",
    "double-buffer":
        "Structure the tile loop so tile n+1's DMA overlaps tile n's compute "
        "(buffer rotation), driving latency toward max(compute, dma), not "
        "compute + dma.",
    "wide-aligned-tiles":
        "Partition dim = 128; free dim >= 512 (bf16 >= 1024) so each DMA moves "
        ">= 2 KiB/partition and all 16 DMA engines stay busy — small tiles are "
        "packet-rate bound, not bandwidth bound.",
    "chunked-scan":
        "A sequential scan (SSM / linear-attention) is parallelized by CHUNKING: "
        "chunk_size <= 128 lives on the partition axis, the intra-chunk term is a "
        "matmul, and a small recurrent state (dstate <= 128) is carried in SBUF "
        "across chunks. Decay is exp(dt*A) (A<0 for stability).",
    "sort-free-topk":
        "Top-K (K <= 8) needs NO sort: do K sequential passes of "
        "`nl.max(logits, axis=1, keepdims=True)` + argmax-via-iota, masking the "
        "winner to -inf before the next pass. Cheaper than a full sort and keeps "
        "every tile 2-D.",
    # --- NKI Performance Guide Opt #5a/#5b/#7/#8 (docs/nki-perf-guide.md) --------
    # These are the levers that BEAT the compiler where it is weak: the compiler
    # already does coarse fusion/tiling/DMA-sizing (Opt #1/#2/#9), so the win is
    # the low-level nki.isa trick it does not find.
    "fast-weight-load":
        "Opt #7 (up to 4x on a SHORT-dim matmul, e.g. bs=1 decode matrix-vector). "
        "When one matmul dim << 128, map the SHORT tensor to the MOVING position "
        "and the LARGE tensor to STATIONARY: fast LoadStationary is up to 4x faster "
        "than MultiplyMoving of the same free size (Short-Moving: LS_II~=32 cyc, MM "
        "every ~64 cyc; Short-Stationary: MM every ~128 cyc). Swap the nc_matmul "
        "args and use A@B = (B^T@A^T)^T (swapping operands transposes the output) so "
        "the result layout still matches. THE lever for autoregressive decode.",
    "partition-vectorize":
        "Opt #5b (2x). Two ops each spanning < 128 partitions run SERIALLY (half the "
        "lanes idle). Write the two nc_matmul outputs to DISJOINT partitions of ONE "
        "128-partition PSUM tile (0:63 and 64:127), then do a SINGLE full-width "
        "nl.max/nl.sum(axis=1, keepdims=True) over the [128,F] tile — 2x vs two "
        "[64,F] reduces. Always fill all 128 partitions (a [128,96] operand uses only "
        "96 PE columns; widening to 128 is FREE) — never leave lanes idle.",
    "tensor-tensor-scan":
        "Opt #5a. A scan written as seq_len back-to-back single-element "
        "nisa.tensor_scalar ops pays the ~100-cycle STATIC instruction overhead every "
        "step (measured 189 ns / 264 cyc overhead vs 1 cyc useful work) — overhead, "
        "not math, dominates. Replace the per-element recurrence with the fused "
        "VectorE primitive `nisa.tensor_tensor_scan` (ONE instruction over the free "
        "axis). Generally: make each instruction touch a free dim >= 128 so the "
        "static overhead amortizes; read-after-write chains make tiny tiles worse.",
    "transpose-swap-for-layout":
        "Opt #8. Do not pay for an intermediate transpose the matmul can absorb. "
        "SWAP stationary/moving in nc_matmul so the OUTPUT layout already matches the "
        "NEXT op (e.g. feeding layernorm bn_stats which wants the feature dim on the "
        "FREE axis: map the WEIGHT to moving -> output comes out pre-transposed, no "
        "nc_transpose). Or move the reduce to the engine whose native layout you have "
        "(TensorE partition-reduce via matmul-against-ones vs VectorE free-axis "
        "tensor_reduce). If a transpose is unavoidable AND memory-bound, use nl.load()"
        " + nisa.nc_transpose() tiled, NOT nl.load_transpose2d (lower DMA bandwidth).",
    # --- static-addressing / gather-as-matmul (UniVR shutter-unroll win) --------
    # An indirect/dynamic gather (data-dependent DMA addresses) runs on GpSimd's
    # programmable procs + slow integrated DMA (~153 GB/s/dir) and serializes while
    # TensorE sits idle. Two levers turn it into compiler-strong work — proven on
    # the UniVR model: F.interpolate 758.9->106 ms (7x) by static addressing, then
    # ->55.6 ms (1.92x) by the gather-as-matmul, TensorE 2.4%->67%, 13x cumulative.
    "host-static-addressing":
        "If sampling indices / DMA offsets are DATA-INDEPENDENT at trace time, "
        "compute them on the HOST and bake them in as COMPILE-TIME CONSTANTS so "
        "every load is STATIC-addressed. An indirect/dynamic gather runs on GpSimd's "
        "~153 GB/s/dir integrated DMA and serializes on a programmable engine; static "
        "addressing frees GpSimd and lets the load stream at HBM peak. Diagnostic: "
        "GpSimd busiest while TensorE idle (gpsimd_bound). Applies ONLY to static "
        "access patterns — NOT data-dependent gathers (KV-paging, MoE dispatch).",
    "gather-as-matmul":
        "A gather / interpolate / resample / scatter with a FIXED (data-independent) "
        "access pattern is a LINEAR op — express it as `out = W @ x` with a "
        "precomputed, mostly-ZERO weight matrix (e.g. bilinear = 4 non-zeros per "
        "output row at the corner taps; upsample / one-hot dispatch similarly). "
        "Trades the indirect-DMA/GpSimd bottleneck for a single STATIC-addressed "
        "matmul on TensorE (the strongest engine). Build W on the HOST once. This is "
        "the framework's whole thesis on a new op class: move a compiler-weak "
        "indirect op onto the compiler-strong systolic array. STATIC patterns only.",
    # --- harvested from AWShtokoyo/vllm-neuron (see docs/vllm-neuron-harvest.md) ---
    "sequential-range-for-scan":
        "A loop-carried read-modify-write (a scan's recurrent state, a flash "
        "running max/denominator) MUST use `nl.sequential_range`, NOT "
        "`nl.affine_range`: affine_range lets the compiler REORDER iterations, "
        "which is 'valid-ranged but WRONG' for a dependency chain. sequential_range "
        "for the carry loop; affine_range only for truly independent tiles.",
    "sequential-gdn-not-chunked":
        "On this neuronx-cc, a GatedDeltaNet/linear-attention scan is best done as "
        "the SEQUENTIAL recurrence (nl.sequential_range over T), NOT the chunked "
        "(I-M)^-1 Neumann-doubling form: the chunked bf16 values DIFFER from the "
        "recurrence enough to perturb a downstream MoE top-8 routing (and over-run "
        "its dispatch buffer), and a Python T-deep unroll hangs the compiler. Keep "
        "head_dim on the partition axis, T on the free axis (<=32767).",
    "paged-32bit-safe-addressing":
        "For a paged KV / recurrent-state slab, NEVER form a flat int32 offset "
        "`block * page_stride` (it overflows at >=2^31 and silently caps the cache "
        "~16383 blocks). Pass block_id and in-block position SEPARATELY, use `.ap "
        "offset=0`, and let the DMA engine scale in >32-bit space.",
    "negated-max-online-softmax":
        "Flash/online softmax: store the running max as its NEGATIVE (-max) and "
        "apply it via the activation `bias=` arg; fuse exp+row-reduce in ONE "
        "`nisa.activation(op=exp, bias=neg_max, reduce_op=add, "
        "reduce_cmd=reset_reduce)`; NaN-guard the running buffers with a 1e30 "
        "sentinel + a min-clamp so an all-masked row does not poison the max.",
    "fp8-row-dequant-in-kernel":
        "Store weights as e4m3 + a per-output-channel scale `[1,out]`; dequant IN "
        "the kernel by expanding the scale `[1,D] -> [P_MAX=128, D]` at forward "
        "(token-generation), or dequant transiently to BF16 for a compute-bound "
        "prefill (no resident second copy). Shape FP8 scale buffers `[128, .]` "
        "(P_MAX) so ONE buffer serves both the [1,.] prefill and [128,.] decode.",
}


# ---------------------------------------------------------------------------
# landmine registry — the op-family-specific traps (subset mirrors repair_hints
# and the [NOTE] contradictions in docs/verified_nki_idioms.py).
# ---------------------------------------------------------------------------
LANDMINES: dict[str, str] = {
    "no-dst-param":
        "Do NOT pass dst=/out= to nc_matmul / nc_transpose / activation — they "
        "RETURN the tile in 0.6.0. (The public nki-samples show the OLD dst= form; "
        "it errors on this stack.)",
    "no-1d-collapse":
        "A reduction that collapses a tile to 1-D fails — keep it [P,1] with "
        "keepdims=True.",
    "implicit-partition-broadcast":
        "`t[P,F] + x[1,F]` / `t[P,F] * col[P,1]` raises 'Unexpected partition "
        "broadcast!' — broadcast_to the full [P,F] shape first.",
    "broadcast-freefn":
        "The tensor-method `tile.broadcast_to(...)` is unreliable; use the free "
        "function `nl.broadcast_to(tile, shape=(P,F))` with shape= keyword-only.",
    "no-bare-float-mul":
        "Multiplying a tile by a bare python float can be an object*float type "
        "error — use `x * (1.0/n)` inline or nl.multiply with a matching-dtype tile.",
    "host-side-cast":
        "Cast the output dtype on the HOST after .cpu(), never on the device tile "
        "(`out.to(fp32).cpu()` trips NCC_ISMP902; `out.cpu().to(fp32)` is fine).",
    "moving-free-512":
        "nc_matmul moving free dim > 512 errors — tile the free dim into <=512 "
        "chunks and accumulate in PSUM.",
    "partition-le-128":
        "Every SBUF/PSUM tile's FIRST dim is the partition axis and must be <= 128 "
        "— put a large feature dim (H=4096) on the FREE axis and tile it.",
    "no-python-controlflow":
        "No try/except, nested def, or tuple-unpacking in the NKI body; prefer "
        "affine_range/static_range over a bare range() for the tile loop.",
    "size-1-partition":
        "A size-1-partition tile can trip an internal vectorizer crash "
        "(NCC_ISFV902/SFKVectorizer) — carry a genuine partition dim (P>=2).",
    "attn-scores-on-partition":
        "DEVICE-VERIFIED (self-improve, trn2): for single-query attention lay the "
        "scores as [128, S/128] with the KV sequence on the PARTITION axis (never "
        "the [1,S] row — that transposes a size-1 partition and trips NCC_IPMN902). "
        "Reduce max/sum on the free axis; take the global max via a transpose + "
        "nc_matmul(ones_row, gmax) broadcast; use nisa.nc_transpose(data=), not "
        "nl.transpose. This is the structure that raced correct at 0.438x.",
    "chunk-partition-limit":
        "Scan chunk_size AND state dstate must each be <= 128 (they sit on the "
        "partition axis). seqlen must be divisible by chunk_size.",
    "topk-k-limit":
        "Router top-K is designed for K <= 8, experts E <= 512, hidden H a "
        "multiple of 128 — outside that the sort-free K-pass approach degrades.",
    # --- harvested from AWShtokoyo/vllm-neuron (see docs/vllm-neuron-harvest.md) ---
    "rmsnormgated-plain-weight":
        "A GATED RMSNorm (GatedDeltaNet) applies the PLAIN learnable weight "
        "(ones-initialized), NOT `(1 + weight)` like Gemma-style norms — using "
        "(1+weight) DOUBLES the GDN output. Know which norm convention the arch uses.",
    "fp8-e4m3-240-not-448":
        "trn2's legacy e4m3 max is 240, but OCP e4m3 max is 448 — out-of-range OCP "
        "codes read back inf -> NaN. BYTE-SATURATE oob codes onto the +-240 grid "
        "(`if (b&0x7F) >= 0x78: b = (b&0x80)|0x77`), exact for in-range codes. Do "
        "NOT rescale by 240/448 (that shifts EVERY element off the fp8 grid).",
    "packed-axis-dma-alias":
        "Many single-slice `nisa.dma_copy` calls keyed by a SCALAR loop index "
        "collapse to the FIRST index under torch-xla — aliasing every head/expert to "
        "index 0 (silent wrong output, compiles clean). Do ONE multi-partition DMA "
        "(the varying axis on the partition dim) + an on-chip nc_transpose instead.",
    "static-pattern-only":
        "The gather-as-matmul / host-static-addressing win applies ONLY when the "
        "access pattern is DATA-INDEPENDENT (known at trace time): image "
        "interpolate/upsample, fixed resample grids, one-hot dispatch. Do NOT apply "
        "it to DATA-DEPENDENT gathers whose indices come from runtime values "
        "(KV-cache paging, MoE token dispatch, argmax-driven selection) — there the "
        "indices cannot be baked as constants and the weight matrix is not fixed.",
    "partial-rope-zero-freq":
        "Partial RoPE (only the first r dims rotate): the inv_freq denominator is "
        "still the FULL head_dim; the non-rotary tail gets zero-frequency "
        "pass-through (cos=1, sin=0) so a sliding-window and a global head can share "
        "ONE rope code path. head_dim here can exceed 128 (256/512) — tile it.",
}


# ---------------------------------------------------------------------------
# signature registry — the exact 0.6.0 return-form signatures, referenced by op.
# ---------------------------------------------------------------------------
SIGNATURES: dict[str, str] = {
    "nc_matmul":
        "nisa.nc_matmul(stationary, moving, *, is_transpose=False, "
        "tile_position=(), tile_size=(), mask=None) -> tile   # stationary [K,M], "
        "moving [K,N] -> [M,N]; K,M<=128, N<=512; RETURNS the PSUM tile (no dst=).",
    "nc_transpose":
        "nisa.nc_transpose(data, *, mask=None, dtype=None) -> tile   # [P,F]->[F,P], "
        "P,F<=128; RETURNS the tile (no dst=). nl.transpose(x) is a simpler alias.",
    "activation":
        "nisa.activation(op, data, *, bias=None, scale=1.0, dtype=None) -> tile   "
        "# returns op(scale*data+bias); op FIRST, data SECOND, rest keyword-only. "
        "For a FUSED free-axis reduce use nisa.activation_reduce(op, data, *, "
        "reduce_op, reduce_res, bias=, scale=) which writes the [P,1] reduction "
        "into the reduce_res OUT-PARAM (no return form) — the activation(reduce_op=) "
        "return-form does NOT return the reduction on trn2.",
    "reduce":
        "nl.sum(x, axis=1, keepdims=True) / nl.max(x, axis=1, keepdims=True) -> "
        "[P,1]  (keep 2-D). nisa.tensor_reduce(op=, data=, axis=) is the ISA form.",
    "reciprocal":
        "nl.reciprocal(x) / nisa.reciprocal(x) -> tile   # for delayed division.",
    "broadcast":
        "nl.broadcast_to(tile, shape=(P,F)) -> tile   # shape= is KEYWORD-only.",
    "iota":
        "nisa.iota(nl.arange(F)[None,:], dtype=nl.int32) then nl.broadcast_to; or "
        "nl.mgrid[0:P,0:F] for full [P,F] index grids (no partition broadcast).",
    "rsqrt":
        "nl.rsqrt(x + eps) -> tile   # normalization inverse-norm.",
}


# ---------------------------------------------------------------------------
# worked examples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkedExample:
    """One short, on-device-idiom-correct snippet the author can pattern-match on.

    ``title``  — what the snippet demonstrates.
    ``code``   — the snippet (kept short; return-form; 2-D tiles; keepdims).
    ``note``   — provenance + on-device verdict / the key idea being taught.
    """

    title: str
    code: str
    note: str


# -- distilled from docs/verified_nki_idioms.py (on-device VERIFIED) --------
_EX_RMSNORM = WorkedExample(
    "Fused RMSNorm (load -> square -> sum(keepdims) -> rsqrt -> broadcast -> mul)",
    """@nki.jit
def rmsnorm_kernel(a, gamma):
    P, F = a.shape
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    t   = nl.load(a[:, :])                                   # [P,F]
    g   = nl.load(gamma[:, :])                               # [1,F] invariant (hoisted)
    ms  = nl.sum(t * t, axis=1, keepdims=True) * (1.0 / F)   # [P,1] mean-square, 2-D
    inv = nl.rsqrt(ms + 1e-6)                                # [P,1]
    o   = t * nl.broadcast_to(inv, shape=(P, F))             # explicit broadcast
    nl.store(out[:, :], value=o * nl.broadcast_to(g, shape=(P, F)))
    return out""",
    "VERIFIED on trn2 (idiom 7, max_err 5.48e-05). Perf lever (on-device "
    "validated, ~11% faster): replace `nl.sum(t*t,...)` with a fused reduce — "
    "`ms = nl.ndarray((P,1), dtype=nl.float32, buffer=nl.sbuf); "
    "nisa.activation_reduce(op=nl.square, data=t, reduce_op=nl.add, "
    "reduce_res=ms[...])` — ONE Scalar-engine pass. Use activation_REDUCE with a "
    "reduce_res out-param, NOT `nisa.activation(..., reduce_op=)` (that return-form "
    "does not return the reduction on trn2).",
)

_EX_TILED_MATMUL = WorkedExample(
    "Tiled matmul, moving free-dim > 512 (PSUM-native accumulation)",
    """@nki.jit
def matmul_kernel(lhsT, rhs):
    K, M = lhsT.shape            # lhsT is [K,M]: contraction K on the PARTITION axis
    _, N = rhs.shape             # rhs is [K,N]
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    lt  = nl.load(lhsT[:, :])                          # [K,M], K<=128, M<=128
    n0 = 0
    while n0 < N:                                      # tile the free dim by 512
        n1 = min(n0 + 512, N)
        rt   = nl.load(rhs[:, n0:n1])                  # [K,<=512]
        psum = nisa.nc_matmul(stationary=lt, moving=rt)  # RETURN-form -> [M,n1-n0]
        nl.store(out[:, n0:n1], value=psum)
        n0 = n1
    return out""",
    "VERIFIED on trn2 (idiom 1, max_err 2.29e-05). nc_matmul contracts on the "
    "PARTITION dim, RETURNS the PSUM tile (no dst=), moving free <= 512.",
)

# -- distilled from expert kernels, TRANSLATED to the verified return-form --
_EX_SOFTMAX = WorkedExample(
    "Numerically-stable row softmax (max-subtract -> exp -> sum -> delayed div)",
    """@nki.jit
def softmax_kernel(x):
    P, F = x.shape
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    t   = nl.load(x[:, :])                                   # [P,F], one load
    mx  = nl.max(t, axis=1, keepdims=True)                   # [P,1] row-max, 2-D
    xmax = t - nl.broadcast_to(mx, shape=(P, F))             # stabilize (explicit bcast)
    e   = nisa.activation(nl.exp, xmax)                      # exp(t-mx), return-form
    den = nl.sum(e, axis=1, keepdims=True)                   # [P,1] denominator, 2-D
    inv = nl.reciprocal(den)                                 # delayed division: 1/sum
    o   = e * nl.broadcast_to(inv, shape=(P, F))             # multiply ONCE at the end
    nl.store(out[:, :], value=o)
    return out""",
    "Structure from nki-samples attn_fwd_v1 softmax (row_max -> exp -> sum -> "
    "reciprocal -> mul), translated to 0.6.0 return-form. PERF lever: fuse the "
    "max-subtract into the exp via `nisa.activation(nl.exp, t, bias=neg_rowmax)` "
    "(bias is the [P,1] -row-max). To also fuse the denominator, use a fused "
    "reduce with an out-param — `den = nl.ndarray((P,1), dtype=nl.float32, "
    "buffer=nl.sbuf); nisa.activation_reduce(op=nl.exp, data=t, bias=neg_rowmax, "
    "reduce_op=nl.add, reduce_res=den[...])` (activation-reduce-fusion) — then "
    "reciprocal once and multiply (delayed-softmax-division). Do NOT expect "
    "`nisa.activation(..., reduce_op=)` to return the denominator; the reduce goes "
    "through activation_reduce's reduce_res.",
)

_EX_ATTENTION = WorkedExample(
    "Single-query attention decode block: QK^T -> softmax -> PV (all return-form)",
    """@nki.jit
def attn_decode_kernel(q, k, v):
    # q [d,1] (stationary), k [d,S] (moving) -> qk [1,S]; d<=128 on partition axis.
    qs = nl.load(q[:, :]); ks = nl.load(k[:, :]); vs = nl.load(v[:, :])
    qk = nisa.nc_matmul(stationary=qs, moving=ks)            # [1,S] RETURN-form, no dst
    mx = nl.max(qk, axis=1, keepdims=True)                   # [1,1] keepdims
    e  = nisa.activation(nl.exp, qk, bias=nl.multiply(mx, -1.0))  # exp(qk-mx), return-form
    sm = nl.sum(e, axis=1, keepdims=True)                    # denominator [1,1]
    p  = e * nl.broadcast_to(nl.reciprocal(sm), shape=e.shape)   # delayed division
    # PV: contract over S -> transpose p to [S,1] so S is the partition axis.
    out = nisa.nc_matmul(stationary=nisa.nc_transpose(data=p), moving=vs)  # [1,d_v]
    o = nl.ndarray(out.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    nl.store(o[:, :], value=out); return o""",
    "Algorithm from nki-samples attention_fwd_performance (v1->v8a ladder), "
    "return-form-translated. For long S tile K/V by 512 and run the online-softmax "
    "(running max + rescale) so the denominator accumulates without materializing "
    "the full score row (flash attention). nc_matmul contracts on the partition "
    "axis — keep head-dim d<=128 there. "
    "DEVICE-VERIFIED (self-improve loop, trn2, 2026-08-24): a full decode kernel "
    "built on this structure raced CORRECT at 0.438x vs torch-eager SDPA. The two "
    "fixes the loop found and that this example now reflects: (1) transpose with "
    "`nisa.nc_transpose(data=...)`, NEVER `nl.transpose` (the alias trips "
    "NCC_IPMN902/vectorizer on the [1,S] score row); (2) lay scores as "
    "[128, S/128] with S on the PARTITION axis (P>=2) so max/sum are cheap "
    "free-axis reduces and you never transpose a size-1-partition tile — compute "
    "the global max via a transpose + `nc_matmul(ones_row, gmax)` broadcast. The "
    "complete verified template is banked at "
    "knowledge-bank/harvested/attn_decode_verified.py.",
)

_EX_ELEMENTWISE = WorkedExample(
    "Fused elementwise activation (softcap: tanh via activation scale, one mul)",
    """@nki.jit
def softcap_kernel(x):
    P, F = x.shape
    out = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.load(x[:, :])                                     # one load
    # tanh(x/cap)*cap: do the /cap inside activation's scale, one mul by cap after.
    capped = nl.multiply(nisa.activation(nl.tanh, t, scale=1.0 / 30.0), 30.0)
    nl.store(out[:, :], value=capped)                        # one store
    return out""",
    "Elementwise ops fuse into ONE straight-line kernel (loop-fusion): one load, "
    "compute in SBUF, one store. Use activation's scale=/bias= to fold constants "
    "in; multiply by a matching-dtype scalar (nl.multiply), never a bare python "
    "float on the raw tile.",
)

_EX_SCAN = WorkedExample(
    "Chunked SSM/linear-attention scan (chunk on partition, state carried in SBUF)",
    """# Per chunk c (chunk_size Q<=128 on the partition axis, dstate<=128):
#   decay = exp(dt * A)                       # A<0 for stable dynamics
#   intra = (C_c @ B_c^T * causal_mask) @ x_c # intra-chunk contribution (a matmul)
#   y_c   = intra + (C_c @ state) * decay_pref # + carried inter-chunk state
#   state = decay_chunk * state + B_c^T @ (x_c * decay_suf)   # update, stays in SBUF
# Every @ is nisa.nc_matmul(stationary=.., moving=..) RETURN-form; keep all tiles
# 2-D; seqlen must be divisible by chunk_size.""",
    "Structure from nki-library experimental/scan/ssd.py (Mamba-2 SSD) and the "
    "KDA chunked kernels. The sequential recurrence is parallelized by chunking: "
    "the intra-chunk term is a matmul over the chunk (partition<=128) and only a "
    "small [dstate<=128, headdim] state crosses chunk boundaries in SBUF "
    "(chunked-scan). Head-outer keeps state in SBUF; chunk-outer shares B/C.",
)

_EX_MOE_ROUTER = WorkedExample(
    "MoE router top-K (logits = x@w, sort-free K-pass argmax with iota + masking)",
    """# logits = x @ w  -> [T, E]  (nisa.nc_matmul; T tokens on partition, E<=512 free)
# probs  = softmax(logits) (or activation)         # router gate
# top-K (K<=8) WITHOUT a sort — K sequential passes:
#   for _ in range(K):                              # static_range, K<=8
#       m   = nl.max(probs, axis=1, keepdims=True)  # [T,1] current best (2-D)
#       idx = argmax_via_iota(probs, m)             # index of the winner
#       # record (m, idx) into the k-th output column; then mask the winner out:
#       probs = mask_to_neg_inf(probs, idx)         # so the next pass finds the next
# norm_topk_prob: divide the K kept affinities by their L1 sum (delayed division).""",
    "Structure from nki-library core/router_topk.py. K<=8 needs NO sort: repeated "
    "max + argmax-via-iota + mask-winner (sort-free-topk). Keep logits/probs 2-D "
    "([T,E], keepdims on every reduce); E<=512, H a multiple of 128.",
)


_EX_GATHER_MATMUL = WorkedExample(
    "Bilinear interpolate as a static matmul (gather-as-matmul; UniVR shutter-unroll)",
    """# INDIRECT gather (SLOW): out[i] = bilinear(in, sample_xy[i]) via 4 data-
# dependent DMA reads per output -> runs on GpSimd's ~153 GB/s/dir integrated
# DMA, serializes while TensorE idles (gpsimd_bound).
#
# STATIC-MATMUL form (FAST) — the sample grid is data-INDEPENDENT, so:
#   1. HOST: build W [n_out, n_in], ZERO except the 4 bilinear corner taps per
#      output row (weights = the fractional areas). Bake W as a constant.
#   2. DEVICE: out = W @ x   -> ONE static-addressed nc_matmul on TensorE.
#      psum = nisa.nc_matmul(W_tile, x_tile)   # W [K<=128, n_out<=128] stationary
#                                              # x [K<=128, n_in_free<=512] moving
# W is sparse but STATIC -> no indirect DMA, no GpSimd. TensorE 2.4% -> 67%.""",
    "From the UniVR shutter-unroll optimization: F.interpolate 758.9->106 ms (7x) by "
    "host-static addressing, then ->55.6 ms (1.92x) by this gather-as-matmul, 13x "
    "cumulative. STATIC access patterns ONLY (static-pattern-only landmine).",
)


# ---------------------------------------------------------------------------
# the knowledge entries — one per op family
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KnowledgeEntry:
    """The retrieved, op-aware knowledge for one op family.

    Fields carry KEYS into ``TECHNIQUES`` / ``SIGNATURES`` / ``LANDMINES`` (so the
    data stays DRY and a test can assert the right keys are attached), plus the
    concrete ``examples``. ``render_knowledge_section`` expands the keys to text.
    """

    family: str
    summary: str
    techniques: tuple[str, ...]
    signatures: tuple[str, ...]
    landmines: tuple[str, ...]
    examples: tuple[WorkedExample, ...] = field(default_factory=tuple)


KNOWLEDGE: dict[str, KnowledgeEntry] = {
    ELEMENTWISE: KnowledgeEntry(
        ELEMENTWISE,
        "A pointwise map (activation, gate, scale, softcap, RoPE). Bound by HBM "
        "traffic — the whole win is fusion: one load, compute in SBUF, one store.",
        ("loop-fusion", "activation-reduce-fusion", "wide-aligned-tiles",
         "bf16-in-fp32-accumulate", "double-buffer"),
        ("activation", "broadcast", "iota"),
        ("no-dst-param", "no-bare-float-mul", "host-side-cast",
         "no-python-controlflow"),
        (_EX_ELEMENTWISE,),
    ),
    REDUCTION: KnowledgeEntry(
        REDUCTION,
        "A free-axis reduce (sum / mean / max / var). Keep the result 2-D and fuse "
        "the elementwise pre-op into the reduce.",
        ("keepdims-2d", "activation-reduce-fusion", "delayed-softmax-division",
         "explicit-broadcast", "engine-overlap"),
        ("reduce", "activation", "broadcast", "reciprocal"),
        ("no-1d-collapse", "implicit-partition-broadcast", "broadcast-freefn",
         "size-1-partition"),
        (_EX_RMSNORM,),
    ),
    NORMALIZATION: KnowledgeEntry(
        NORMALIZATION,
        "RMSNorm / LayerNorm / add+norm: a reduce (mean / mean-square) -> inverse "
        "norm -> broadcast -> affine. Compose the reduction + broadcast idioms; "
        "hoist gamma/beta.",
        ("activation-reduce-fusion", "keepdims-2d", "explicit-broadcast",
         "hoist-invariant", "loop-fusion", "bf16-in-fp32-accumulate"),
        ("activation", "reduce", "rsqrt", "broadcast"),
        ("no-1d-collapse", "implicit-partition-broadcast", "broadcast-freefn",
         "host-side-cast"),
        (_EX_RMSNORM,),
    ),
    MATMUL: KnowledgeEntry(
        MATMUL,
        "A GEMM on the systolic array. Contraction on the PARTITION axis; tile the "
        "moving free dim to <=512 and accumulate natively in PSUM.",
        ("isa-return-form", "tile-to-hw-limits", "psum-native-accumulation",
         "bf16-in-fp32-accumulate", "downcast-before-transpose",
         "fast-weight-load", "partition-vectorize", "transpose-swap-for-layout"),
        ("nc_matmul", "nc_transpose"),
        ("no-dst-param", "moving-free-512", "partition-le-128"),
        (_EX_TILED_MATMUL,),
    ),
    SOFTMAX: KnowledgeEntry(
        SOFTMAX,
        "Row softmax: subtract row-max for stability, exp, sum, then divide ONCE. "
        "Fuse the exp + denominator into one activation call; delay the division.",
        ("delayed-softmax-division", "activation-reduce-fusion", "keepdims-2d",
         "explicit-broadcast", "loop-fusion"),
        ("reduce", "activation", "reciprocal", "broadcast"),
        ("no-1d-collapse", "implicit-partition-broadcast", "no-dst-param"),
        (_EX_SOFTMAX,),
    ),
    ATTENTION: KnowledgeEntry(
        ATTENTION,
        "QK^T -> softmax -> PV. Two matmuls around an online (flash) softmax; tile "
        "K/V by 512 and carry a running max + denominator so the full score row "
        "never materializes.",
        ("isa-return-form", "tile-to-hw-limits", "delayed-softmax-division",
         "activation-reduce-fusion", "psum-native-accumulation",
         "downcast-before-transpose", "negated-max-online-softmax",
         "sequential-range-for-scan", "fast-weight-load",
         "transpose-swap-for-layout"),
        ("nc_matmul", "nc_transpose", "activation", "reduce", "reciprocal"),
        ("no-dst-param", "moving-free-512", "no-1d-collapse", "partition-le-128",
         "size-1-partition", "attn-scores-on-partition", "packed-axis-dma-alias",
         "partial-rope-zero-freq"),
        (_EX_ATTENTION, _EX_SOFTMAX),
    ),
    SCAN: KnowledgeEntry(
        SCAN,
        "SSM / linear-attention / GatedDeltaNet-KDA: a recurrence over the sequence. "
        "Two forms: a CHUNKED parallel form (intra-chunk matmul + SBUF state carried "
        "across chunks) and a SEQUENTIAL form (nl.sequential_range over T). On this "
        "neuronx-cc the SEQUENTIAL form is preferred for GatedDeltaNet — see "
        "sequential-gdn-not-chunked.",
        ("chunked-scan", "sequential-range-for-scan", "sequential-gdn-not-chunked",
         "tensor-tensor-scan", "paged-32bit-safe-addressing", "isa-return-form",
         "tile-to-hw-limits", "psum-native-accumulation", "keepdims-2d"),
        ("nc_matmul", "activation", "reduce"),
        ("chunk-partition-limit", "no-dst-param", "partition-le-128",
         "no-1d-collapse", "rmsnormgated-plain-weight"),
        (_EX_SCAN,),
    ),
    MOE_ROUTER: KnowledgeEntry(
        MOE_ROUTER,
        "Router: logits = x@w -> gate activation -> top-K selection + affinity "
        "scatter. Top-K (K<=8) is sort-free (K max+mask passes).",
        ("sort-free-topk", "isa-return-form", "keepdims-2d",
         "delayed-softmax-division", "tile-to-hw-limits", "partition-vectorize"),
        ("nc_matmul", "reduce", "iota", "activation", "reciprocal"),
        ("topk-k-limit", "no-1d-collapse", "no-dst-param", "moving-free-512"),
        (_EX_MOE_ROUTER,),
    ),
    INDIRECT_GATHER: KnowledgeEntry(
        INDIRECT_GATHER,
        "A gather / interpolate / resample / upsample / scatter with a STATIC "
        "(data-independent) access pattern. The naive lowering is an indirect DMA "
        "on GpSimd — slow (~153 GB/s/dir) and serial while TensorE idles. THE win: "
        "compute the access indices on the host (static addressing), then express "
        "the whole linear resample as ONE matmul `out = W @ x` with a precomputed "
        "mostly-zero weight matrix — moving the work onto TensorE. Static patterns "
        "ONLY (not KV-paging / MoE dispatch).",
        ("gather-as-matmul", "host-static-addressing", "isa-return-form",
         "tile-to-hw-limits", "psum-native-accumulation", "wide-aligned-tiles"),
        ("nc_matmul", "nc_transpose"),
        ("static-pattern-only", "no-dst-param", "moving-free-512", "partition-le-128"),
        (_EX_GATHER_MATMUL,),
    ),
}


# ---------------------------------------------------------------------------
# classification — op name / notes -> op-family key
# ---------------------------------------------------------------------------
# Ordered most-specific first; the FIRST family whose keywords hit wins. Keyed on
# the op NAME and (secondarily) the spec ``notes`` string. ``family`` on the spec
# is a MODEL family (e.g. "dense_causal_lm"), NOT an op family, so it is used only
# as a weak tie-breaker hint, never as the primary signal.
_FAMILY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # NOTE: bare "qk" is intentionally NOT a keyword — it would swallow "qkv_proj"
    # (a matmul). Use the score-specific "qk^t" / "softmax-pv" instead.
    (ATTENTION,   ("attn", "attention", "sdpa", "flash", "qk^t", "softmax-pv")),
    (SCAN,        ("scan", "ssm", "ssd", "mamba", "deltanet", "delta_net",
                   "linear_attention", "linear-attention", "recurr", "kda",
                   "gateddelta", "gated_delta", "delta_rule", "gdn",
                   "selective_scan", "state space", "state-space")),
    (MOE_ROUTER,  ("router", "topk", "top_k", "top-k", "moe", "expert",
                   "gating", "gate_logits", "affinit")),
    # indirect_gather BEFORE the generic families so interpolate/upsample/gather
    # ops don't fall through to elementwise (they are a high-value authoring
    # target — the gather-as-matmul win — not a cheap pointwise op).
    (INDIRECT_GATHER, ("interpolate", "grid_sample", "grid-sample", "resample",
                       "upsample", "bilinear", "index_select", "gather", "scatter",
                       "warp", "remap")),
    (SOFTMAX,     ("softmax",)),
    (NORMALIZATION, ("rmsnorm", "layernorm", "rms_norm", "layer_norm", "groupnorm",
                     "group_norm", "batchnorm", "norm")),
    (MATMUL,      ("matmul", "gemm", "linear", "_mm", "proj", "dense",
                   "fc", "bmm")),
    (REDUCTION,   ("reduce", "sum", "mean", "variance", "var", "cumsum", "argmax",
                   "l2norm")),
    (ELEMENTWISE, ("gelu", "silu", "swiglu", "geglu", "softcap", "rope", "gate",
                   "act", "cast", "add", "mul", "rotary", "elementwise", "tanh",
                   "sigmoid", "relu", "erf")),
)


def _name_of(spec_or_name) -> str:
    if isinstance(spec_or_name, str):
        return spec_or_name
    return getattr(spec_or_name, "name", "") or ""


def classify_op(name: str, family: str | None = None,
                notes: str | None = None) -> str:
    """Map an op to its op-family key (one of ``OP_FAMILIES``).

    Matching is on lowercased NAME first, then NOTES. ``family`` (the spec's MODEL
    family) is not authoritative and is ignored here. Falls back to ``ELEMENTWISE``
    — the safe default (its knowledge is the generic fuse-one-load/one-store
    guidance that applies to any pointwise op) — when nothing matches, so retrieval
    never returns nothing.
    """
    hay_name = (name or "").lower()
    hay_notes = (notes or "").lower()
    # NAME is the strong signal — check it against every family first.
    for fam, kws in _FAMILY_KEYWORDS:
        if any(kw in hay_name for kw in kws):
            return fam
    # Then fall back to NOTES (weaker — descriptive prose).
    for fam, kws in _FAMILY_KEYWORDS:
        if any(kw in hay_notes for kw in kws):
            return fam
    return ELEMENTWISE


def retrieve(spec_or_name, family: str | None = None,
             notes: str | None = None) -> KnowledgeEntry:
    """Return the ``KnowledgeEntry`` for an op.

    Accepts either an ``OpSpec`` (reads ``.name`` / ``.family`` / ``.notes``) or a
    bare op-name string (+ optional ``notes``). Always returns an entry (the
    ELEMENTWISE fallback if the op is unrecognized), so the prompt builder can
    unconditionally render a section.
    """
    name = _name_of(spec_or_name)
    if family is None:
        family = getattr(spec_or_name, "family", None)
    if notes is None:
        notes = getattr(spec_or_name, "notes", None)
    fam = classify_op(name, family=family, notes=notes)
    return KNOWLEDGE[fam]


# ---------------------------------------------------------------------------
# rendering — the prompt block the author sees
# ---------------------------------------------------------------------------
def render_knowledge_section(entry: KnowledgeEntry,
                             max_examples: int = 2) -> str:
    """Render a retrieved ``KnowledgeEntry`` as the "Relevant verified techniques &
    worked examples" prompt section. Deterministic and side-effect free, so the
    wiring in ``build_author_prompt`` stays unit-testable.

    Only the KEYS present on the entry are expanded (unknown keys are skipped
    defensively), so the section always reflects exactly the op-relevant subset —
    not the whole registry.
    """
    lines: list[str] = []
    lines.append("## Relevant verified techniques & worked examples "
                 f"(op family: {entry.family})")
    lines.append(entry.summary)

    lines.append("\nTechniques that matter for THIS op (verified levers):")
    for k in entry.techniques:
        desc = TECHNIQUES.get(k)
        if desc:
            lines.append(f"  * [{k}] {desc}")

    lines.append("\nSignatures you will need (NKI 0.6.0 return-form):")
    for k in entry.signatures:
        sig = SIGNATURES.get(k)
        if sig:
            lines.append(f"  * {sig}")

    lines.append("\nLandmines to avoid for this op family:")
    for k in entry.landmines:
        lm = LANDMINES.get(k)
        if lm:
            lines.append(f"  * {lm}")

    ex = entry.examples[:max_examples]
    if ex:
        lines.append("\nWorked example(s) — on-device-idiom-correct, "
                     "pattern-match on these:")
        for e in ex:
            lines.append(f"\n### {e.title}")
            lines.append("```python")
            lines.append(e.code)
            lines.append("```")
            lines.append(f"note: {e.note}")

    return "\n".join(lines)


def knowledge_for_prompt(spec_or_name, family: str | None = None,
                         notes: str | None = None,
                         max_examples: int = 2) -> str:
    """Convenience one-liner: retrieve + render for an op. This is the single call
    ``kernel_author.build_author_prompt`` makes."""
    return render_knowledge_section(
        retrieve(spec_or_name, family=family, notes=notes),
        max_examples=max_examples,
    )
