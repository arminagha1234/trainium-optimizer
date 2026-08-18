# Architecture

Three components. Each is useful on its own; together they compound.

```
    +------------------+       +-------------------+
    |   Model source   |       |  Reference corpus |
    |  (HF Hub, local) |       |  (vLLM, SGLang,   |
    +--------+---------+       |   TRT-LLM, Neuron |
             |                 |   docs, our repo) |
             v                 +---------+---------+
   +-------------------+                 |
   |   Autoport baseline (existing agent)|
   +---------+---------+                 |
             |                           |
             v                           v
   +-----------------------------------------------------+
   |                Optimizer outer loop                 |
   |                                                     |
   |   proposer  --->  compile  --->  equivalence  ---.  |
   |      ^              (NEFF)         (existing)    |  |
   |      |                                           v  |
   |      |                                        measure|
   |      |                                        (perf) |
   |      |                                           |  |
   |      +---------- update -----------+-------------+  |
   |                                    |                |
   +------------------------------------|----------------+
                                        |
                                        v
                          +-------------+-------------+
                          |     Knowledge Bank        |
                          |   (structured lessons +   |
                          |    seed prior + static    |
                          |    site render)           |
                          +-------------+-------------+
                                        |
                                        v
                          +-------------+-------------+
                          |     Leaderboard runner    |
                          |  (top-100 orchestrator,   |
                          |   per-family adapters,    |
                          |   refresh scheduler)      |
                          +---------------------------+
```

## Component 1: Optimizer outer loop

The core new engineering. Everything else is either existing (autoport,
equivalence, NKI agents) or downstream artifacts (leaderboard).

### Responsibilities

1. Start from an autoport-produced baseline.
2. Consult the knowledge bank for applicable priors.
3. Propose candidate configurations to try.
4. Compile each, gate on equivalence, measure against the chosen objective.
5. Emit lessons on both successes and failures.
6. Respect the compute budget. Stop when no-improvement streak or budget
   exhausted.

### Config space (indicative, expands per architecture family)

- **Sharding**: `tp_degree ∈ {1, 2, 4, 8, 16, 32}` (bounded by `num_kv_heads`
  for GQA), `cp_degree ∈ {1, 2, 4}`, and `dp_degree` — **derived, not searched**:
  the fill planner (`hardware.py`) sets `dp = cores // (tp*cp)` so the whole
  instance is used rather than just the TP group. See
  `optimization-stages.md` (Stage 1, "Fill the instance").
- **Precision**: `weights ∈ {fp32, bf16, fp8, int8-w8a8}`,
  `activations ∈ {bf16, fp8}`, KV cache dtype
- **Attention**: dense / paged / flash / GQA-optimized / MoE-aware
- **Batching**: static batch / dynamic batch / continuous batching / max-tokens
- **Sequence layout**: contiguous / paged blocks / prefix-cached
- **Custom NKI kernels**: swap-in points for hot ops (RoPE, RMSNorm,
  attention, sampling, etc.)

The proposer navigates this space. In v0 it's greedy with human priors; in
v1+ (phase 4) it consults a learned cost model.

### Proposer — v0 policy

Hand-priored greedy, roughly:

```
def next_candidates(current_best, model_family, size, banks_matches):
    # 1. If the bank has a high-confidence entry for this
    #    architecture+size, propose that as candidate 1.
    priors = bank.query(model_family, size, sdk_version)

    # 2. Delta-search: perturb one axis at a time from current_best.
    #    Order axes by expected impact (from bank aggregates or hand ordering).
    deltas = order_axes(current_best, priors)

    # 3. Anti-patterns short-circuit: if bank has a "never do X for family Y"
    #    entry, prune it.
    return prune_by_antipatterns(deltas, priors.antipatterns)
```

Not fancy. Deliberately so — it fits in a page and it works.

### Measurement contract

Every candidate produces the same schema:

```yaml
config: { ... }                # the full config that produced this NEFF
compile_time_seconds: N
neff_size_bytes: N
equivalence: { passed: bool, max_abs_diff: float, tokens_verified: N }
measurements:
  throughput:
    tokens_per_sec_batch1: N
    tokens_per_sec_batch16: N
    ...
  latency:
    ttft_p50_ms: N
    ttft_p99_ms: N
    per_token_p50_ms: N
    ...
  memory:
    hbm_peak_gb: N
    hbm_available_gb: N
environment:
  instance_type: trn2.48xlarge
  neuron_sdk: 2.28.0
  timestamp: ISO-8601
```

This schema is also what a lesson's `evidence` field references. Same
struct, reused everywhere. See `knowledge-bank.md`.

### Budget management

`--budget-hours N` is a first-class arg. The optimizer tracks:

- Cumulative compile time (each NEFF)
- Cumulative measurement time (running each NEFF for enough tokens to be
  reliable)
- Wall time on the trn2 instance

Stops when any of: budget exhausted, N iterations with no improvement,
convergence within tolerance to a known upper bound.

## Component 2: Knowledge bank

Structured store of what has and hasn't worked. See
[`knowledge-bank.md`](./knowledge-bank.md) for full detail.

### Interface the optimizer uses

```python
class KnowledgeBank:
    def query(model_family, param_count, seq_len, batch, sdk_version) -> Lessons
    def add_positive(config, evidence, provenance) -> LessonId
    def add_negative(config, reason, provenance) -> LessonId
    def add_kernel(op_signature, nki_source, evidence) -> LessonId
    def verify_still_valid(lesson_id, sdk_version) -> bool
```

### Interface for humans

- Static-site render, browsable per architecture family + per topic
- Search + filter by SDK version, model family, confidence
- Each lesson is a markdown file with YAML front-matter (simple to author,
  easy to grep, works with `git blame` for provenance)

## Component 3: Leaderboard runner

Ops layer. Given the "top 100" list, run the optimizer for each entry with
a per-model budget, publish results.

### Responsibilities

- Materialize the top-100 list from the current definition
- Schedule optimization runs (respect concurrency limits, capacity, cost caps)
- Publish per-model: recipe, measurements, reproducibility script
- Track SDK versions; re-verify on release
- Diff per refresh: what improved, what regressed

### Per-family adapters

Concrete Python modules that plug into the outer loop:

```
leaderboard/
  adapters/
    dense_causal_lm.py       # Llama, Qwen, Mistral, Phi, Gemma
    moe_causal_lm.py         # Mixtral, DeepSeek-MoE, Qwen-MoE
    encoder_only.py          # BERT, ModernBERT
    diffusion.py             # SD, SDXL, FLUX
    speech.py                # Whisper, Voxtral
```

Each adapter defines: config axes, equivalence tolerance, benchmark
protocol, kernel-swap points.

### Publishing

Static site (Hugo or MkDocs). Per model:

- Metrics: tok/s at various batch sizes, TTFT, TTL, HBM usage, $/M-tokens
- Recipe: config YAML + list of custom NKI kernels used + link to bank entries
- Reproducibility: `bash scripts/reproduce.sh <model-name>` gives the same
  numbers within tolerance
- Comparison baseline: same model on H100 with vLLM, same tokens, same
  temperature — for perf-per-dollar honesty

## Failure isolation

Each component fails independently:

- Optimizer can run without a bank (just no priors → slower)
- Bank can be curated by humans only (no optimizer needed)
- Leaderboard can consume optimizer output directly without the bank

Which means: we can ship any one of these alone if scope has to shrink.
Order of standalone value:

1. Optimizer with 3 models (this is the "prove it" artifact)
2. Bank populated by humans (this is the "documented Neuron optimization
   wisdom" artifact — valuable even if the optimizer never ships)
3. Leaderboard (this is the "public flywheel" artifact — needs 1+2 to be
   sustainable)

---

# Backend adapters

The serving backend is behind an adapter, the same way model families are. This
is what makes the eventual XLA → native-PyTorch migration an added file rather
than a rewrite.

## Why this matters now

vLLM-Neuron today is built on XLA. Native PyTorch (TorchNeuron) is in beta and
the serving stack is expected to move onto it. Roughly 80% of this framework is
backend-independent; the adapter isolates the other 20%.

## What lives behind the adapter

```
backends/
  base.py                  # the interface below
  vllm_neuron_xla.py       # V1 primary — proven, production-representative
  nxdi_xla.py              # autoport baseline producer
  native_pytorch.py        # added when the beta is viable
```

```python
class Backend(Protocol):
    name: str                        # "vllm-neuron-xla" | "native-pytorch" | ...

    def build_baseline(model_id: str) -> Artifact:
        """Stage 0. Produce a runnable, correct implementation."""

    def config_axes() -> dict[str, list]:
        """Stage 1. Backend-specific knob names and legal values."""

    def apply_config(artifact, config) -> Artifact: ...

    def compile(artifact) -> Neff:
        """No-op for eager-mode backends."""

    def measure(neff, shape, batch) -> Measurements: ...

    def profile(neff, shape) -> Profile: ...

    def kernel_swap_points(artifact) -> list[OpSite]:
        """Stage 2/3/4. Where a NKI kernel can be substituted."""

    def graph_passes() -> list[Pass]:
        """Stage 5. XLA passes vs torch.compile passes — highly backend-specific."""

    def toolchain_stamp() -> dict:
        """Full version capture for reproducibility."""
```

## What is *not* behind the adapter

Deliberately kept backend-agnostic, because these are the bulk of the value:

- Knowledge bank (schema, storage, retrieval, staleness, tiers)
- The search loop (beam, plan/implement split, tournament, promotion rules)
- Guardrails (shapes, HBM ceiling, stopping criteria, measurement tiers)
- Trajectory ledger, chart, and report
- Discovery job and leaderboard runner
- **NKI kernels themselves** — they sit below the framework boundary

## Migration plan when the beta lands

1. Add `backends/native_pytorch.py`. One file.
2. Query the bank for `migration_risk: high` and re-verify only those.
3. Run both backends on the same seed model; publish the comparison. That
   comparison is itself a useful public artifact — "here is what the migration
   costs or buys, measured."
4. Flip the leaderboard's primary backend only when native PyTorch matches or
   beats XLA on the seed set *and* TP works at the degrees our models need.

## Backend hooks for hardware-aware fill (handoff to the backend owner)

The core is now hardware-aware: the proposer attaches a fill plan to **every**
candidate config, so alongside `tp_degree` each config carries `cp_degree`,
`dp_degree`, `kv_replication`, `cores_used`, and `cores_available` (computed by
`hardware.fill_plan`). This is backend-independent and already exercised by the
mock. To make it real on device, the native-PyTorch backend
(`backends/native_pytorch.py` + `neuron_worker.py`) needs three changes — all
additive, and until they land the extra keys ride along in the config
harmlessly (the box just isn't filled yet, so nothing breaks):

1. **Launch `dp_degree` replicas.** `apply_config`/the worker should read
   `dp_degree` and start that many independent model replicas (each a full
   `tp × cp` group on its own cores), fan the request stream across them, and
   **sum** throughput. `dp` does not change per-core HBM — it uses *other*
   cores — so it is the cheap way to fill the box for the throughput track.
   Set `NEURON_RT_NUM_CORES` / `torchrun --nproc_per_node` to `tp*cp*dp`.
2. **Honor `cp_degree` and `kv_replication`.** `cp_degree > 1` enables context
   parallelism (the long-context/latency lever). `kv_replication > 1` means the
   search chose `tp > num_kv_heads`; the worker's `num_kv_heads % tp == 0`
   assertion (`neuron_worker.py:64`) should become "either divides, or
   replicate KV heads `kv_replication`×" — a **testable** path, gated by
   measurement, not a hard reject.
3. **Report occupancy.** `measure()` should set `Measurements.cores_used`
   (`tp*cp*dp`) and `cores_available` (the instance's cores, from
   `hardware.budget_for(instance_type)` or the runtime count), and compute
   `mfu_percent` against the **full** instance. That is what makes the
   utilization guardrail and the ledger's `[under-util: …]` flag work on real
   hardware exactly as they do against the mock.

**Heads-up: the proposer now sweeps `tp_degree` to the core count itself**
(1, 2, 4, …, 64 on trn2.48xlarge), deriving `dp_degree` per TP to sweep the
full TP×DP grid. So the worker *will* be handed high TP values — including
`tp > num_kv_heads` with `kv_replication > 1`. It must either run them (replicate
KV heads) or reject cleanly with `invalid_tp` (which is recorded as a failed
data point). Today it hard-fails the divisibility assertion; honoring
`kv_replication` is what lets the sweep actually measure the high-TP end. The
"TP≥16 spills" anti-pattern is verify-first (validated on XLA only), so it will
NOT pre-prune those points on native — they're meant to be measured here.
`config_axes()` still only needs to add `cp_degree` (for the latency /
long-context track); `tp_degree` and `dp_degree` are handled by the planner.

## Blocking unknown, restated

Cross-chip TP (degree >= 4) is documented failing on **Trn1** under native
PyTorch (`Failed to execute the device barrier 1`). Whether it works on
**Trn2** is untested and decisive — our seed models want TP=8. This is a
half-day experiment and it gates any decision to make native PyTorch primary.
