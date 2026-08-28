# Autonomous Trainium Model Optimizer + Leaderboard

### Best multi-stage result — Qwen3-1.7B, optimized end-to-end on one trn2.3xlarge

![Qwen3-1.7B optimization trajectory](optimized_models/qwen3-1-7b/optimization_timeline.png)

*Gains across **two** stages, not just config: **Stage 1 (Config)** climbs eager → `torch.compile(backend="neuron")` (+527%) → TP → batch to 44.9k tok/s, then **Stage 5 (Graph-Rewrite)** adds a further **+14%** (`--optlevel 3`) to **51,278 tok/s = 17.2× baseline**, correctness-verified. The staircase continues past Stage 1 — the honest picture of where each stage helps (blue steps = kept gains; grey ✗ = discarded; stages 2/3/4/6 walked with no gain over config for this model). Per-model recipes + charts live in [`optimized_models/`](./optimized_models/).*

An ambitious extension of the current Neuron autoport/NKI tooling. The idea in
one paragraph:

> Take our existing autoport + NKI writer/debugger + equivalence agents and
> wrap them in an outer loop that autonomously optimizes any HuggingFace
> model on Trainium along two objectives (max throughput, min latency),
> steals patterns from vLLM/SGLang/TensorRT-LLM under license-compliant
> terms, writes its own NKI kernels when nothing off-the-shelf fits, and
> writes every proven trick to a **knowledge bank** so the N+1'th model is
> cheaper to optimize than the N'th. Then point it at the top 100 open
> models and publish a **Neuron-optimized leaderboard** that refreshes on
> every SDK release.

## 📊 Progress toward the north star

**North star:** point it at *any* HF model → optimize on Trn2 → verify correctness → publish a recipe → get cheaper every model — **zero humans in the loop.**

```
Overall autonomy   ███████████████████████████████████████████████░░  ~92%
```

**The 5-station factory line** (LOAD → COMPILE → AUTO-FIX → AUTHOR → BANK), all built and validated on real silicon:

```
                        0%       25%       50%       75%      100%
 1. LOAD model          ████████████████████████████████████████ 100%  ✅
 2. COMPILE → NEFF      ████████████████████████████████████████ 100%  ✅
 3. AUTO-FIX errors     ███████████████████████████████████████░  97%  ✅ device-proven
 4. AUTHOR kernels      ██████████████████████████████████████░░  95%  ✅ device-proven
 5. BANK / compound     ████████████████████████████████████████ 100%  ✅ device-proven
```

**Capability milestones:**

```
 A  recipe → correct on device      ██████████████████████████████ 100%  ✅
 B  logprob/KL correctness gate     ████████████████████████████░░  92%  ✅ (on-device grader run pending)
 C  attention authoring on silicon  ██████████████████████████████ 100%  ✅
 D  auto-promotion / compounding    ██████████████████████████████ 100%  ✅
 E  full UNATTENDED run (capstone)  ███████████████████████████░░░  90%  ✅ ran live (4.41×, published + leaderboard)
```

**Proven on real hardware** (trn2.3xlarge): the **full unattended loop ran end-to-end** — `Qwen3-0.6B` optimized to **164 tok/s = 4.41×** at 100% correctness, recipe published, leaderboard synced, **zero human intervention**. Both compiler-weak wins — **FlashAttention** and **DeltaNet / GatedDeltaNet** — are banked, on-device-validated, and **auto-harvest** (no authoring) when a model exposes the primitive. Device-measured **385 GB/s/core** peak drives a live **%SOL profitability gate**; the full Stage-4 pipeline (`author → compile → race → %SOL → mutator`) executed on silicon incl. the live Opus-5 author writing a numerically-correct attention kernel first try; a real **logprob/KL task-eval** correctness gate composes with the publish gate; a `--max-configs` backstop keeps single-chip runs efficient. The **AWShtokoyo/vllm-neuron** contributed models (GLM-5.2, Qwen3.6-GatedDeltaNet, Gemma-4, Ministral3) are harvested into the bank — 8 live rewrites + cross-model NKI idioms. Honest limit: hand-written NKI loses to the compiler on standard small ops — wins come from the **compiler-weak regime** (long-context attention, sparse, IO-bound), exactly where FlashAttention & DeltaNet live.

**Remaining to 100%:** the on-device task-eval grader run on a full model (B), and — the real frontier — an **autonomous opportunity-sweep** that profiles a model's ops by %SOL and auto-targets the compiler-weak ones, removing the last human judgment of *what* to optimize.

## Current state (2026-08-27, honest) — Stage-4 hardened + validated end-to-end on silicon

A large hardening pass on the kernel/optimization stage, with **every claim below re-verified on a real NeuronCore** (trn2.3xlarge).

- **%SOL profitability gate is live and device-grounded.** The framework measured its own single-core memory-bandwidth ceiling on-device (streaming-copy sweep → **~385 GB/s/core**) and now computes each authored kernel's **% of speed-of-light** from a device-timed latency. A kernel already ≥80% of SOL **skips the perf loop** (no headroom); a far-from-SOL op is flagged an *opportunity*. The gate **refuses torch-wallclock latencies** — measured on-device, wallclock read ~55 GB/s (6–10× low, pure host overhead) where the device sustains ~385, so %SOL must be device-timed. (`roofline.py`, wired into `_device_race` + the perf gate.)
- **The `activation_reduce` idiom, fixed across every authoring surface.** On-device testing found the fused sum-of-squares / softmax-denominator reduce was being emitted in a return-form (`nisa.activation(op, reduce_op=)`) that does **not** compile on this stack (NCC_INIC902 / wrong shape). The correct idiom — `nisa.activation_reduce(..., reduce_res=<out-param>)`, harvested from the internal NKI-Autotune reference and validated cos 1.0 (~11% faster) — is now taught consistently in the mutator, the author knowledge base, and the repair-hint map.
- **The whole Stage-4 pipeline ran end-to-end on hardware.** `author → compile → device-race → correctness-oracle → %SOL → mutator` executed on a real NeuronCore for elementwise, norm, **and attention** ops — including the live **Bedrock Opus-5 author writing a numerically-correct `attn_decode` kernel first try, no repair**. The framework honestly declines to bank a kernel that doesn't beat its baseline.
- **Honest finding — where the first *win* comes from.** Re-measured on-device, standard small ops (rmsnorm, gelu, attn_decode) are **correct but slower than the compiler baseline** (0.37–0.70×) — hand-written NKI, even LLM-authored, loses on ops the compiler already does well. NKI wins **only in the compiler-weak regime** (long-context attention, sparse, IO-bound), which is exactly where the banked FlashAttention kernel (below) lives. Recipe-authored ops were also *fixed to compile + run correctly* (e.g. the RMSNorm recipe's 1-D-collapse / partition-broadcast / gamma-axis bugs) so they flow cleanly through the pipeline as honest anti-patterns rather than dead rejects.
- **Compute note.** `trn2.48xlarge` (the full 128-NeuronCore node) is **Capacity-Blocks-only** — confirmed no on-demand capacity in any AZ. Day-to-day optimization runs on `trn2.3xlarge`; a full node needs a reserved Capacity Block.
- **Ran the full unattended loop, live.** On one `trn2.3xlarge` the framework optimized `Qwen3-0.6B` end-to-end with **zero human intervention** — config search found `tp=2` → **164 tok/s = 4.41×** at 100% correctness, published a recipe, and synced the leaderboard. Both compiler-weak wins (**FlashAttention** and **DeltaNet/GatedDeltaNet**) are banked, on-device-validated, and **auto-harvest** (no authoring) when a model exposes the primitive.
- **Single-chip efficiency (`--max-configs`).** On a small (4-core) box the Stage-1 beam otherwise over-explores a long refinement tail (each vllm-serve config is a ~2-min NEFF compile). A hard config-budget backstop bounds the search so unattended single-chip runs stay fast without dropping the high-value levers.
- **Harvested the AWShtokoyo/vllm-neuron contributed models** (GLM-5.2, Qwen3.6-GatedDeltaNet, Gemma-4, Ministral3; Apache-2.0) into the bank: 8 live compiler-error→fix rewrites, cross-model NKI idioms/laws (sequential-scan, 32-bit-safe paged addressing, fp8 e4m3 saturation, negated-max flash softmax), and new arch routing — see [`docs/vllm-neuron-harvest.md`](./docs/vllm-neuron-harvest.md).

## Current state (2026-08-21, honest)

Live and running on **two `trn2.3xlarge` boxes in parallel**, pooling one knowledge bank.

- **Verified wins on real hardware** (native-pytorch-beta3, correctness-gated): Qwen3-0.6B **~28×**, Qwen2.5-0.5B **15.4×**, Qwen3-4B **~13×**, Qwen2.5-3B **12.4×** (see leaderboard). Dominant lever: Stage-1 config search — `torch.compile(backend="neuron")`, TP=4, bf16, batching.
- **Full pipeline runs per model**: Stage 0 baseline → 1 config → 2–3 compiler-flag rewrites (+ a fused-MoE NKI kernel *borrow* for MoE models) → 5 graph-rewrite (an `--optlevel 1/2/3` + auto-cast sweep) → 6 profile-loop. On native PyTorch the neuronx-cc compiler already does kernel selection/fusion when `torch.compile` runs, so beyond config the real lever is **compiler flags** — stages 2/3/5 race flagsets (spellings on-device-verified against neuronx-cc), each gated by the same equivalence + guardrails as config. The one genuine kernel *swap* is the MoE-family fused-NKI-megakernel borrow in Stage 3.
- **Two-box parallel**: shard 0 + shard 1 over a **40-model × 10-pass** rolling queue across diverse architectures (Qwen, Gemma-2, Phi, Mistral, OPT, Pythia, GPT-NeoX, StableLM, OLMo, Falcon, StarCoder, Granite, BLOOM…), with bidirectional bank-sync.
- **Knowledge bank compounding**: **25 lessons (10 verified / 15 provisional).** Priors promote provisional→verified once ≥2 models agree, then seed the beam. Early signal that it's *improving* not just growing: configs-to-win trending down (Qwen3-4B 88→83).
- **Durable**: near-continuous (2-min) snapshots to the [`bank-snapshots`](../../tree/bank-snapshots) branch via a repo-scoped deploy key — accumulated learning survives box loss (a fresh box restores with `git fetch origin bank-snapshots`).
- **Three backends behind one loop**: native-pytorch (throughput), diffusion (text-to-image; SD-Turbo validated), vllm-serve (latency-SLA).
- **Learns from failure too**: losses banked as anti-patterns. Novel primitives the compiler can't auto-lower (linear-attention / GatedDeltaNet) are **no longer dead-skipped** — the pre-flight gate now *routes* them to a named kernel need (linear-attention → the `DeltaNet` kernel) via `kernel_registry`, turning "⛔ unsupported" into an actionable work item. And for **Qwen3-Next / Qwen3.5** specifically, the pre-flight gate now *admits the model outright* (`--rewrites-wired`): the framework's permanent graph-rewrite bundle (sort→argmax router, tril→const-mask, dense-MoE dispatch, int64 fp32-sort) makes it compile + be correct **without** the DeltaNet kernel at the scales the loop verifies — the kernel is the large-scale perf path only. (Previously the gate skipped Qwen3.5 for lacking a kernel it doesn't need at that scale.)

## Kernel-stage breakthrough (2026-08-23) — the first validated *invented* kernel, on silicon

Two of the previously-unproven links (below) are now closed, and the framework produced its first genuinely-useful invented kernel. Every number here was measured on a real NeuronCore, against the right baseline.

- **The LLM author is LIVE.** `kernel_author` is wired to Bedrock (`claude-opus-5`); it writes idiomatic NKI, and a **repair-hint map** turns known `neuronx-cc` errors into targeted, imperative corrections fed back to the next attempt. The author is a *thinking* model, so hard ops (attention / matmul / scan) get an **op-aware token budget** — more completion headroom and a trimmed prompt — so it no longer exhausts its budget on the hidden thinking pass before emitting a kernel (the wall that used to kill attention authoring outright).
- **Refinement is *structural*, not prompted — because prompting doesn't work.** Once a correct kernel exists we want the optimizer to keep the winning template and change *one thing*, not re-derive from scratch. We measured, on real silicon, that you **cannot prompt** the LLM author into this: however hard the prompt pushed "keep the winner, change one thing", Opus-5 rewrote the template every time (source overlap with the winner stayed 0.13–0.24, never a refinement). So refinement is done **in-code** by a `kernel_mutator` that emits the winning source with one mechanical edit (wider tile, delayed division, activation-reduce fusion) — variants that preserve the template by construction. The mutator only *proposes*; the perf loop re-validates correctness + speed and keeps a variant only if it is still correct **and** faster. Division of labor: the LLM is for the *initial correct kernel* and *genuine exploration*; the mutator is for *refining a known-good template*.
- **The correctness gate is now a *hardened evaluator*.** A kernel is judged not against an unreachable fp32 ideal but against **the incumbent bf16 op it replaces** — it must miss the fp32 reference on *no more elements* than the incumbent does, plus a NaN/magnitude guard. Fair to bf16, and not reward-hackable. First validated rank-4 WIN: a fused `add_rmsnorm` that is *more accurate than the torch op it replaces*.
- **🏆 FlashAttention — the framework's first banked invented kernel.** A streaming online-softmax NKI kernel that **compiles and runs correctly at sequence length 8192, where `neuronx-cc` cannot compile dense attention at all** (it OOM-thrashes the host). Optimizations, all correctness-preserving and measured on-device:
  - transpose-free `[kv,q]` score layout + fused softmax denominator → **causal S=8192: 5.16 ms → 728 µs (7.1×)**;
  - query-sequence sharding across 4 NeuronCores → non-causal S=8192 **1.22 ms → 338 µs (3.62×)**, near-ideal, bit-identical output.
  It is **registered (rank-4), routable, injectable, and validated on-device _through the framework's own retrieve→inject→run path_** — see [`implementation/src/kernels/FlashAttention/`](./implementation/src/kernels).
- **Where NKI can (and can't) win — measured, not assumed.** On standard dense matmul / MLP / elementwise, `neuronx-cc` is already at ~80% of hardware speed-of-light; hand-written NKI *ties at best*. NKI wins **only where the compiler is weak** — long-context attention (it can't compile it), non-standard/sparse patterns. Baselines are measured against **`torch.compile` (XLA-fused), not eager**, and reported as **% of speed-of-light**, so a "win" beats the real incumbent, not a strawman.
- **Then the bottleneck moves to the host.** At batch-size 1 the device is fast and >99% *idle* — the limiter becomes host dispatch, not the kernel. The framework's next frontier is therefore the serving/host path (graph reuse, batching), which we're now characterizing.

## The kernel stage (Stage 4) — how we get a model the compiler can't lower

Being built out in honest layers (see [`docs/linear-attention-kernel-pattern.md`](./docs/linear-attention-kernel-pattern.md)):

1. **Route, don't skip** — `kernel_registry` maps a primitive to its kernel (`linear_attention`→`DeltaNet`, `mamba/ssm`→`Mamba2`, `mla`→`MLA`, …); `preflight.kernel_route` turns a linear-attn skip into "needs the `DeltaNet` kernel (available: yes/no)". Proprietary NKI source stays external (`$TRN_OPT_KERNEL_DIR`); only the routing + interface are public.
2. **Cheapest fix first** — a symptom-indexed **rewrite catalog** (`kernel_rewrites`) tries a graph rewrite before any kernel. Grounded example: compiling a **full** Qwen3-Next/Qwen3.5 (GatedDeltaNet-MoE) model on trn2 (neuronx-cc 2.27.5334), the load-bearing blocker was **not** `.tril()` (that compiles fine at full-model scale) — it was the **MoE-router `torch.topk`**, which lowers to an XLA `sort` op that trn2 has no ISA support for (`NCC_EVRF029: Operation sort is not supported`). The fix that made the full model compile to a valid NEFF is a **sort-free top-k (iterative argmax) — a pure graph rewrite, no kernel**. (The `.tril()` → `TensorScalarAffineSelect` → host-materialized constant-mask rewrite is still valid and catalogued, but was not the full-model blocker on this compiler version.)
3. **Harvest before invent** — `invent_engine._prior_art` reuses an already-authored, usable kernel (recorded `Origin.HARVESTED`) instead of re-authoring.
4. **A repair loop that learns** — `kernel_repair.KernelRepairLoop` feeds the exact `neuronx-cc` error back to the next authoring attempt (bounded rounds; honest `compiled`/`exhausted`/`stalled` stops, never a fake success). The on-device gate is real — a `nki.simulate` pass is **not** trusted as hardware-ready (a Mamba scan simulated to 2e-7 ran 67 off on real Trn2).

**Honest edges (not overclaiming):**
- **The kernel stage is wired end-to-end — and (as of 2026-08-23) both previously-unproven links are now closed on real hardware.** The full pipeline: route (`kernel_registry`/`preflight`, wired into `overnight`) → escalation ladder (rewrite catalog incl. `.tril`→const-mask and `sort`→argmax; rank-aware harvest/reuse) → author (recipe table **plus a now-LIVE Bedrock LLM author**, `kernel_author.py`) → **repair loop + hint map** (feeds the exact compiler error, and a targeted fix, back to the next attempt) → tiered validation (**rank ladder** with the `simulate ≠ on-device` wall + the **fair-vs-incumbent + magnitude anti-reward-hacking gate**) → **generic serving-injection hook** (`kernel_inject`) → persist + author-time bank retrieval. Both formerly-open links are now proven: (1) the injection hook + kernel execution **have been validated on real hardware** (the FlashAttention kernel runs end-to-end through registry→inject→run at S=2048/8192), and (2) the LLM author is **wired to a real provider** (Bedrock `claude-opus-5`). A kernel win is now *measured* end-to-end (FlashAttention at S=8192). Most auto-authored kernels still lose to the compiler — the honest finding is that NKI's value is concentrated where the compiler is weak (long-context attention), plus the accumulated lessons. See [`docs/kernel-stage-deepdive.md`](./docs/kernel-stage-deepdive.md), [`docs/kernel-stage-external-prior-art.md`](./docs/kernel-stage-external-prior-art.md), [`docs/nki-optimization-playbook.md`](./docs/nki-optimization-playbook.md), and [`docs/verified_nki_idioms.py`](./docs/verified_nki_idioms.py) (on-device-verified NKI idiom corpus).
- The learning curve is bending, but **weakly so far** — the compounding is early and gated on more architecture diversity cross-validating.

## 🏆 Trainium Optimizer Leaderboard

Results published by the autonomous overnight loop on real Trainium hardware
(`native-pytorch-beta3`). Three canonical files at the repo root:
[`LEADERBOARD.md`](./LEADERBOARD.md) for current standings,
[`HISTORY.tsv`](./HISTORY.tsv) for the append-only improvement record, and
[`optimized_models/<family>/<model>/`](./optimized_models/) for per-model
recipes and trajectory charts (`optimization_timeline.png` +
`optimization_highlights.png`).

### Text-to-text (LLMs)

<!-- LEADERBOARD:START -->
| Rank | Model | Family | Params | Baseline (tok/s) | Optimized (tok/s) | Speedup | Best config | Hardware | Status |
|-----:|:------|:-------|-------:|-----------------:|------------------:|--------:|:------------|:-------------|:-------|
| 🥇 | Qwen3-0.6B | qwen3 | 0.6B | 3,333 | **85,937** | **25.788×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 🥈 | Qwen3-1.7B | qwen3 | 1.7B | 2,975 | **51,278** | **17.239×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 🥉 | Qwen2.5-0.5B-Instruct | qwen2.5 | 0.5B | 4,833 | **74,269** | **15.368×** | TP=2, torch.compile(neuron), bf16, batch=8, DP=2 | trn2.3xlarge | ✅ Verified |
| 4 | Qwen3-4B | qwen3 | 4B | 1,882 | **26,548** | **14.104×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ Verified |
| 5 | Qwen2.5-3B-Instruct | qwen2.5 | 3B | 2,856 | **35,343** | **12.375×** | TP=4, torch.compile(neuron), bf16, batch=8 | trn2.3xlarge | ✅ Verified |
| 6 | Mistral-7B-Instruct-v0.3 | mistral | 7B | 2,555 | **23,270** | **9.108×** | TP=4, torch.compile(neuron), bf16, batch=32 | trn2.3xlarge | ✅ Verified |
| 7 | Qwen3-8B | qwen3 | 8B | 1,903 | **16,876** | **8.87×** | TP=4, torch.compile(neuron), bf16, batch=8, CP=2 | trn2.3xlarge | ✅ Verified |
| 8 | Qwen2.5-7B-Instruct | qwen2.5 | 7B | 2,916 | **9,870** | **3.384×** | TP=4, bf16, batch=8 | trn2.3xlarge | ✅ Verified |
<!-- LEADERBOARD:END -->

### Text-to-image · Text-to-video · Speech (ASR / TTS)

Not yet scaffolded — modalities open once the LLM adapters are stable.

**How to read this.** *Speedup* is against the *eager* baseline on the same
instance, on the framework's fixed probe shape. The dominant Stage-1 lever
across every measurement so far is `torch.compile(backend="neuron")` — the
search reaches it because `compile_mode` is tried before every other axis.

**Status legend.** ✅ Verified = correctness-gated (top-1 token match vs the
Stage-0 baseline ≥ 75%) and reproducible via
`optimized_models/<family>/<model>/reproduce.sh`.
🛠 Adapter in progress = the model needs a family-specific tensor-parallel
or vocab-parallel adapter (see `backends/qwen38_tp.py` for the pattern).
🕒 Queued = in the seed list, awaiting the next cycle.


## Why this is worth doing

- Nobody owns "best perf-per-dollar on Trainium for open models" as a
  visible public artifact. MLPerf covers a handful; vLLM benchmarks aren't
  Neuron-focused; HF Open LLM is accuracy, not perf.
- We already have the atomic pieces (autoport, NKI writer/debugger,
  equivalence, profile-analysis agents). The gap is an orchestrator + a
  memory of what has worked.
- The knowledge bank is the flywheel: leaderboard runs generate lessons,
  lessons make future runs cheaper, which makes the leaderboard cheaper to
  maintain, which drives more lessons.

## Docs in this folder

| File | What it covers |
|------|----------------|
| [`plan.md`](./plan.md) | Phased milestones with concrete deliverables. Read first. |
| [`guardrails.md`](./guardrails.md) | Benchmark shapes per track, batch sweeps, HBM/compile limits, stopping criteria, SDK stamping |
| [`model-landscape.md`](./model-landscape.md) | Researched top open models per modality (Aug 2026 snapshot) |
| [`optimization-stages.md`](./optimization-stages.md) | The 6-stage pipeline: config → known kernels → **borrow** → **invent** → graph rewrites, plus execution discipline |
| [`references-analysis.md`](./references-analysis.md) | What we took from Autocomp, auto_research (a large (multiple-x) on Trn2), AFlow, AgentArch, ADIAS |
| [`harvest-corpus.md`](./harvest-corpus.md) | The aws-neuron org survey + **Stage 0.5: Harvest** — mine working kernels before optimizing |
| [`agent-topology.md`](./agent-topology.md) | Multi-agent design (worker vs. watcher agents) and what host this runs on |
| [`trajectory-reporting.md`](./trajectory-reporting.md) | The ledger + chart + report showing **how** it improved, not just the final number |
| [`architecture.md`](./architecture.md) | How the components fit — optimizer loop, knowledge bank, leaderboard runner |
| [`knowledge-bank.md`](./knowledge-bank.md) | Lesson schema, types of lessons, seed content, staleness handling |
| [`open-questions.md`](./open-questions.md) | Load-bearing decisions — several now resolved |
| [`references.md`](./references.md) | References we'll consult for kernel patterns + prior art |
| `websites_check` | Original reference-checking note (kept as-is) |

## Decisions locked in so far

- **Four benchmark shapes**: `chat` (1k/512), `rag` (10k/512), `generate`
  (512/10k), `stress` (64k/64k — from a real customer ask)
- **Batch sweep** 1→32 powers of two, shape-aware (`stress` caps at 4)
- **HBM ceiling 85%** measured at peak KV occupancy, **30 min compile
  timeout** — OOM avoidance is a hard gate
- **No compute budget cap.** Termination is by search-quality criteria;
  spare capacity goes to parallelism across models, not depth on one
- **Leaderboard first**, framework open-sourced later
- **Weekly multi-source model discovery** (LMArena + HF trending + HF
  downloads + Open LLM Leaderboard + AWS customer signal)
- **Provisional vs. verified** knowledge-bank tiers with human triage
- **Per-family adapters** define the search space; the outer loop is shared
- **Anti-patterns are first-class** — own folder, read by the proposer to
  prune candidates before compiling
- **Hand-priored greedy** search for v0; learned policies deferred to Phase 4
- **Seed ladder**: Gemma 4 31B → Muse Glimmer 30B → Qwen3.8-27B (all Apache
  2.0, all fit trn2.3xlarge). MiniMax dropped — H3 is video-gen with reported
  territory restrictions; M-series deferred to phase 3.
- **V1 scope is Track A (LLMs) only.** Image / video / ASR / TTS tracks are
  fully designed but deferred — keeping them specified forces the outer loop
  to stay track-agnostic, so adding one later is writing an adapter rather
  than refactoring the core.
- **Six optimization stages**, cheapest and safest first: baseline → config →
  known kernels → **borrow** → **invent** → graph rewrites.
- **Borrow before invent.** Steal patterns from vLLM / SGLang / TRT-LLM /
  FlashAttention first. A novel kernel only replaces a borrowed one if it
  beats it by **5%** — noise floor plus a risk premium for bespoke code.
  Losing inventions are recorded, not discarded.
- **Invention is instrumented**, not assumed: `invention_rate`,
  `invention_win_rate`, `roofline_attainment` tracked from run 1 so "does it
  ever create something new" gets an empirical answer.
- **Direct code borrowing** permitted from Apache-2.0 / BSD sources, with
  `THIRD_PARTY_NOTICES` + per-file provenance headers. TensorRT-LLM needs a
  compliance look before we borrow from it.
- **Full toolchain stamped on every result** (`neuronx_cc` version
  especially), with a re-verification pass on every SDK release

## TL;DR of the plan

1. Pick 3 seed models across families (dense LLM, MoE, diffusion). Build
   the optimizer's outer loop with a hand-priored greedy search + hard
   compute budget. Get end-to-end wins on all three.
2. Ship the recipes as a blog post + repo. Get external signal early.
3. Design the knowledge bank schema, bootstrap it with 30-50 lessons we
   already know from the Neuron team, integrate into the optimizer's
   proposer.
4. Pick a "top 100 open models" definition and start a rolling leaderboard.
   Publish per-model recipes + evidence.
5. Only after that: replace hand-priored greedy with a learned search
   policy.

## Non-goals for V1

- Not solving general AutoML — we're doing model→Neuron optimization only.
- Not building a training-time optimizer — inference only.
- Not chasing state-of-the-art search algorithms in phase 1 — a good greedy
  with human-designed priors beats a bad Bayesian on a small budget.
- Not competing with Nvidia's leaderboard numbers — we publish honest
  perf-per-dollar comparisons and let the numbers speak.

## Prior art this builds on

Five systems studied in detail; full analysis in
[`references-analysis.md`](./references-analysis.md).

| System | What we took |
|--------|-------------|
| [**Autocomp**](https://github.com/ucb-bar/autocomp) (UC Berkeley, NeurIPS 2025) | Beam search over LLM-generated plans; plan-then-implement two-phase prompt; optimization menu with dropout; hardware-in-the-loop every iteration. Already supports Trainium NKI. Their paper also measured **24% improvement from reusing optimization schedules across similar ops** — direct empirical support for the knowledge bank. |
| **internal-prior-optimization-run** (private) | The closest existing system — a 12-hour autonomous loop hitting **a large (multiple-x) on Tongyi-30B-A3B**. Took: phase budgets with scoped write permissions, read-only grader, git-as-state-machine, three measurement tiers, MFU normalization, the trajectory chart format, and its prompt rules. |
| [**AFlow**](https://arxiv.org/abs/2410.10762) (ICLR 2025 Oral) | MCTS over code-represented workflow space; **tree-structured experience** rather than a flat lesson store. |
| [**AgentArch**](https://arxiv.org/abs/2509.10769) (ServiceNow) | Design dimensions interact in ways component-level study misses → run an **ablation** on the first 3-5 models, not just end-to-end speedup. |
| [**ADIAS**](https://arxiv.org/html/2608.06410v1) | Candidate-centric experience stores leave repair progress implicit → index the bank **by symptom**, not only by intervention. |

Two things that changed because of this reading:

1. **Search strategy revised from greedy to beam** (Autocomp proved beam on
   Trainium specifically; greedy was too conservative).
2. **The "config carries most of the speedup" assumption was wrong** — real
   data shows config +19%, model code +405%, structural TP changes +193%.
   Corrected in `optimization-stages.md`.

## Harvest before optimize

A **Stage 0.5: Harvest** runs before any optimization: read-only, 30-minute
budget, mines the `aws-neuron` corpus and the knowledge bank for kernels that
already work, and emits a ranked candidate inventory plus an
`unmatched_ops` queue that becomes the evidence-derived Stage 3/4 work list.

Motivating case: the `auto_research` run hand-built Context Parallel in its
third round. [`nki-library`](https://github.com/aws-neuron/nki-library)
already ships *Attention KV Parallel Segmented CTE* — "KV-parallel segmented
prefill attention with online softmax merging for context parallelism." Plus
`FGCC` (fused all-gather + matmul) for the collective bottleneck they also
attacked by hand. Harvest exists so we try the shelf before the workshop.

Provenance is now four-valued, and reported separately so the invention metric
stays honest:

| `origin` | Meaning | Risk |
|----------|---------|------|
| `harvested` | Existing AWS-maintained kernel, used as-is | Low — maintained upstream |
| `borrowed` | Pattern ported from vLLM / SGLang / TRT-LLM / FlashAttention | Medium |
| `hybrid` | Borrowed algorithm, substantially restructured for Neuron | Medium-high |
| `invented` | Novel, from profile + roofline | Highest |

A run that is 90% `harvested` is a **good outcome** — the ecosystem had the
answer. It is just not invention, and conflating the two would flatter us.

## Backend strategy

vLLM-Neuron (XLA) is the V1 target. Native PyTorch / TorchNeuron is expected to
become the serving substrate later, so the backend sits behind an adapter from
day one and every bank lesson is **layer-tagged**:

| `layer` | Survives an XLA → native-PyTorch migration? | `migration_risk` |
|---------|---------------------------------------------|------------------|
| `kernel` (NKI) | **Yes** — NKI is below the framework boundary | low |
| `collective` (TP/CP/EP patterns) | Mostly | low-medium |
| `config` | Concepts yes, knob names no | medium |
| `framework` (vLLM/NxDI internals) | Often not | high |
| `graph` (XLA passes) | Likely not | high |

Two consequences:

- When the migration happens, the re-verification scope is a single query
  (`migration_risk: high`) rather than "re-test everything."
- **The proposer prefers lower layers on ties.** A NKI kernel worth +15% beats a
  framework patch worth +15%, because one survives. Encoded as a ranking
  tiebreaker, not left to instinct.

Roughly 80% of this framework is backend-independent — bank, search loop,
guardrails, ledger, reporting, discovery, leaderboard, and the NKI kernels
themselves. Waiting for the beta would protect the other 20%, which is not
worth an unbounded delay.

## Agents

Worker agents come from the existing NAD package. Two **watcher** agents are
new, and both target a specific place where the hardware oracle is weak:

| Watcher | Weakness it covers | Budget |
|---------|-------------------|--------|
| **Adversarial equivalence** | The gate proves "matches on these inputs," not "correct in general." This agent tries to *break* winning candidates — boundary shapes, ragged batches, numerical edges. Findings become permanent test cases. | ≤10% of stage time, winners only |
| **Supervisor** | The oracle scores candidates but not *direction*. Reads the ledger every 30 min and redirects when the search concentrates somewhere the profile says is not the bottleneck. | ≤2% (reads a TSV) |

The supervisor exists because of a documented failure: the reference
implementation's agent "fixated on MoE for 12 hours before being redirected" by
a human.

Deliberately **not** built: committee voting, agent debate, duplicate reviewers.
We have a hardware oracle — compile it and measure it. Those patterns
approximate a missing oracle and mostly add cost when you have one.

## Where it runs

Two layers, because a 12-hour autonomous run on a headless trn2 cannot have an
IDE in the path:

- **Core loop** — host-agnostic Python package, provider-agnostic LLM interface
  (Bedrock / Anthropic / OpenAI / local vLLM). Runs on the instance. This is
  what gets open-sourced, because public reproducibility cannot require a
  specific editor.
- **Interactive shell** — Kiro / Claude Code, for bank triage, trajectory
  review, and debugging stalled runs.

Phase 1 prototypes as NAD-style agents and skills (fastest path, reuses the 8
existing agents), then phase 2 extracts the core into the package with NAD
agents behind a `KernelWriter` interface. That interface is also how the
Autocomp calibration experiment (Q13) plugs in.
