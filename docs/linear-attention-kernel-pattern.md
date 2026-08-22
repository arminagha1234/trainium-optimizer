# The chunked linear-attention kernel pattern (and how to write one)

Why this doc: models like **Qwen3.5 / Qwen3-Next (GatedDeltaNet)**, Mamba/SSM,
RWKV, mLSTM, RGLRU and friends use a *linear-attention recurrence* that
`neuronx-cc` cannot auto-lower from the reference PyTorch. Our preflight used to
mark the whole class "⛔ skipped — unsupported." That is the wrong conclusion:
the reference *algorithm* is compiler-hostile, but the primitive is a **solved
kernel-authoring problem**. This doc captures the reusable pattern and the
NKI-specific gotchas so we can author (or harvest) these kernels methodically
instead of treating them as blockers.

This is methodology + math only — no proprietary kernel source. Actual kernels
plug in externally via `kernel_registry` (see its IP-boundary note).

## 1. Why the naive graph fails

The HF reference (`torch_chunk_gated_delta_rule`) has two compiler-hostile
structures:

1. **In-place strided scatter inside a loop** — `attn[..., i, :i] = ...` across a
   `for i in range(1, chunk_size)` forward-substitution. This lowers to a
   select/scatter op (`TensorScalarAffineSelect`) the Neuron ISA rejects.
2. **Sequential scans** — the intra-chunk forward substitution and the
   cross-chunk state recurrence are data-dependent loops that unroll into long
   dependency chains; even if they lowered, they would be latency-bound.

Measured cost of the naive *recurrent* form (per-token `<S,k>` + rank-1
`k⊗v` + `<S,q>`): each op uses ~**1/128** of the 128×128 PE array and the T
outputs form a strict dependency chain → at batch=1, ~**0.004% MFU**
(host-dispatch bound). So "make it compile" and "make it fast" are the same fix.

## 2. The fix: the chunked dual (UT/WY transform)

Process a **chunk of C tokens with dense block matmuls**, carrying only the
inter-chunk K×V state across ⌈T/C⌉ sequential steps. The forward-substitution
loop is replaced by a **nilpotent matrix inverse**, so it becomes matmul-only.

Per value-head, over a chunk of length C, with per-token scalar log-decay
`b_g = -exp(A_log)·softplus(g + dt_bias)` and cumulative inclusive decay
`a_i = exp(Σ_{j≤i} b_g)`:

```
RHS_i = β_i v_i − β_i a_i (k_i · S0)                 # [C, V]
L_ij  = β_i (a_i / a_j)(k_i · k_j)   for j < i        # strictly-lower [C, C]
(I + L) U = RHS  →  U = Σ_{p=0}^{C-1} (−L)^p RHS      # L nilpotent ⇒ EXACT, matmul-only
o_i   = a_i (q_i · S0) + Σ_{j≤i} (a_i / a_j)(q_i · k_j) u_j
S_new = a_{C-1} S0 + Σ_j (a_{C-1} / a_j) k_j ⊗ u_j    # inter-chunk carry
```

Everything heavy — the `k·k` and `q·k` Gram matrices, the `K·S0`/`Q·S0`
contractions, the `(−L)^p` sweep, the `A·U`/`Kdec·U` contractions — is a dense
`nc_matmul` that fills the PE array. Only ⌈T/C⌉ steps stay sequential (vs T).

The **same shape** works across the whole family: Mamba-2 (SSD), RWKV6/7,
mLSTM, LightningAttn, RGLRU, KDA — each has a `*_chunked_prefill` kernel + a
`*_recurrent` decode kernel. Learn it once, reuse the structure.

## 3. Prefill vs decode

- **Prefill (T tokens):** the chunked kernel above. Fills the PE array.
- **Decode (T=1):** stays on the recurrent kernel — a 1-step chunk *is* the
  recurrence; carry state `S[K,V]` in SBUF with no HBM round-trip per token.

Route accordingly: one kernel per phase, different access patterns.

## 4. NKI gotchas that actually bite (hard-won)

1. **Contraction on the PARTITION axis via `nc_matmul`.** Use
   `nisa.nc_matmul(stationary=[K,M], moving=[K,N]) -> [M,N]` (a Kᵀ-contract).
   `nl.matmul` on an `[M,K]` layout silently contracts the **wrong** axis — the
   "rel_fro = 1.0" trap (looks like a kernel, produces garbage). Keep scores
   transposed `sT[Sk,Sq]` so both QKᵀ and P@V are a single `nc_matmul` with no
   in-kernel transpose; the softmax denominator is a ones-matmul (reduce over
   the partition axis).
2. **Decay-overflow / NaN trap.** `a_i/a_j = exp(cum_i − cum_j)`. For kept
   lower-triangular entries `cum` is monotone-decreasing (exponent ≤ 0, safe);
   **clamp the masked upper entries to ≤ 0 *before* `exp`** (`nl.minimum(diff, 0)`)
   so `exp` can't overflow to `+inf` and the subsequent 0-mask can't make
   `0·inf = NaN`.
3. **HW-safe broadcast.** Every per-token scalar (β_i, a_i, inverse-L2-norm)
   lives on the **token (C) partition axis** and broadcasts along a FREE axis
   (per-partition free broadcast = HW-safe). A single partition-0 scalar (e.g.
   `a_{C-1}` applied to the K-partition state) must be lifted onto all K
   partitions through the PE array (`ones[1,K]ᵀ · scalar`) — **never** a bare
   partition-0 broadcast.
4. **Affine head ids, never floordiv a loop index.** For GVA/GQA nest
   `(h, grp)` so the value-head id `hv = h·groups + grp` is an affine
   expression. `floordiv`-ing a loop index breaks lowering.
5. **Static-shape contract.** `CHUNK ≤ 128` and `K, V ≤ 128` (partition limit).
   `T` must be a multiple of `CHUNK`; the caller pads the prefill bucket and
   drops the padded tail.
6. **Load-bearing scale.** Linear-attention q-scaling is `K**-0.5` (a
   compile-time float). A `1.0` default silently emits `√K`×-too-large output —
   a *finite, token-emitting* fake-GREEN that passes a smoke test but is wrong.
7. **Output tensors are kernel ARGUMENTS, not returns** (NKI top-level
   constraint); dual-SDK convention: prod `neuronx-cc` 2.x treats a passed
   `o`/`final_S` as immutable and calls without them (`o is None` → allocate
   `shared_hbm` + return); the simulate/baremetal harness passes them and the
   kernel writes in place.
8. **In-place `S[...] = ...` state updates** avoid scope rebinding — this is how
   you express the update without the compiler-hostile scatter.

## 5. How to write one (the pipeline that works)

Follow the incremental pipeline — **validate at every step**, never skip:

1. **Reference** (HF/torch ground truth) → 2. **numpy** parity → 3. **NKI-lang**
   (`nl.load`/`nl.store`) → 4. **NKI-ISA** (`nisa.nc_matmul`, explicit engines)
   → 5. **tiling** (128/512 multiples) → 6. **masking** (`nl.mgrid`) → 7.
   **optimize** (only after correctness).

Translate the **math, not the code** (read a Triton/CUDA reference like
`fla.ops.*` for the algorithm, re-derive in NKI primitives — cleaner and IP-safe).

Validate in tiers, and heed the one rule that matters most:
**`nki.simulate` passing OVERSTATES hardware readiness.** A selective-scan that
simulated to `2e-7` ran ~**67 max_abs_diff** off on real Trn2. CPU-oracle →
simulate → **on-device** are three distinct gates; only an on-device pass counts
as HW-ready. (This is exactly why the invent engine's on-device speed race must
time both sides on the same device — a simulate/CPU-timed "win" is not real.)

## 6. Where this plugs into the framework

- `preflight.kernel_route()` maps a linear-attention model → the `DeltaNet`
  kernel and reports whether one is registered (`kernel_registry`), turning the
  old dead-end skip into a named, actionable route.
- `invent_engine` does **prior-art / Harvest first**: if a usable kernel is
  registered for the op's `primitive`, it is reused (never re-invented); only an
  unmatched primitive goes to authoring.
- The reusable target `CHUNK`-based structure above is the "invent" template for
  any new linear-recurrent primitive that shows up next.
