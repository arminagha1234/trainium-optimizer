# NKI / Megakernel-Writing Stage — Improvement Plan (living doc)

> **Status:** DRAFT v3 · started 2026-09-10 · owner: Armin
> **v2 (2026-09-10):** added §6 cross-hardware research (TPU playbook from the verified
> SAIL Gemma-v6e blog + GPU algorithmic patterns), new rec R25 (silent-perf detectors).
> **v3 (2026-09-10):** began implementation. R1 landed (◐ — offline plumbing done+tested,
> device activation pending one on-device run), R17 landed (☑). See §9 implementation log.
> **v4 (2026-09-10):** R3 (◐), R25 (☑), M4 (☑), R18 (☑) landed. Full offline suite 961 passing
> (same pre-existing torch/boto3-only failures, none introduced). See §9.
> **Purpose:** a reviewable, add-to-later plan for improving the framework's Stage-4
> (invent / NKI kernel authoring) and megakernel-writing path. This is a PLAN, not a
> changelog — every recommendation has a status box so we can check things off and
> argue about them. Nothing here is implemented yet unless its status says so.

---

## 0. TL;DR — the strategic thesis

We do **not** have to build the NKI-authoring loop from scratch. Research (Sept 2026)
found that most of our author → gate → race → bank loop **already exists as first-party
AWS assets**. The highest-leverage move is **harvest + align + differentiate**, not invent:

- **NAKOS** (internal, `KaenaNeuronKernelLibrary/nkilib_agentic_dev_plugin/orchestrator`)
  is a production, unit-tested implementation of our exact loop.
- **neuron-agentic-development** (official OSS; ships in the DLAMI/DLC) is the public
  write→debug→profile→analyze version.
- **NKI CPU simulator** (`nki.simulate`) + a **run-log-arbitrated lesson-banking skill**
  fill our two biggest gaps (offline correctness, honest banking).
- **NAKB / NKIBench / OPTNKI** are ready-made scoreboards to measure whether our changes
  actually help.

Two priors that should shape effort allocation:
- **NKI-Agent paper (AWS, DL4C@ICML'26):** the compile-verify-fix *loop* was decisive
  (Opus 4.8 ~6% single-shot → 77.3% with the agent loop); SFT beat GRPO; a **binary
  compile reward failed**. → invest in the loop + tools, not prompt-stuffing or naive RL.
- **AgenticCodeOptimizer ablation:** a local NKI-knowledge **RAG did not improve results**
  while frontier models did. → validate any knowledge-injection on a benchmark before
  over-investing in it.

**Our differentiation to keep:** the forever-run across *many whole models* + config
search + honest mock/real labeling + a bank that compounds across *models*, not just
kernels. Adopt their kernel-authoring internals; don't rebuild their fleet/cred infra.

---

## 1. Scope

The stage that takes a compiler-weak primitive (or a fusable sequence of stages) and
produces a **validated NKI kernel** — authoring, offline gate, on-device race, bank —
plus the **megakernel** (fusion) path that fuses a model's stages into one SBUF-resident
trace. Files: `implementation/src/invent_engine.py`, `kernel_author.py`,
`kernel_providers.py`, `kernel_oracles.py`, `kernel_validation.py`, `kernel_anticheat.py`,
`kernel_rewrites.py`, `nki_knowledge.py`, `kernel_perf.py`, `kernel_mutator.py`,
`fusion.py`, and the Stage-4 hook in `orchestrator.py::run_deep_stages`.

---

## 2. Current state (grounded) — where it's weak

The subsystem is architecturally rich (Harvest→Borrow→Invent; offline gate; on-device
race with anti-cheat; repair loop; perf loop; bf16-fair correctness; rank ladder). But it
is effectively **dormant**, for four concrete reasons:

1. **Not wired into the live loop.** `orchestrator.run_deep_stages` Stage-4 is a stub that
   records one honest DISCARD: *"invent_engine not wired to pipeline (needs NKI-writer
   agent + validated on-device execution)."* Zero kernels are invented in the forever-run.
2. **Default author can't iterate.** `RecipeAuthor` is the default; `LLMAuthor` exists but
   `max_repair_rounds=1` / `max_perf_rounds=1` — the repair + perf loops (the code's own
   "#1 lever") are OFF.
3. **The offline math check is a tautology.** `LLMAuthor` sets `numpy_impl = spec.reference`,
   so offline parity compares a function to itself; the gate honestly refuses to call it a
   pass and defers ALL math validation to device → a device compile is spent before any
   math bug is caught.
4. **The on-device path is the least-tested part.** `_device_race` / `_device_compile_probe`
   are annotated *"built to run on .73; not exercised on a CPU box."*

Design truth to anchor on: **most invented kernels lose to neuronx-cc's own fusion.** The
value is the occasional win on a compiler-weak primitive + the anti-patterns each loss
banks. So improvements should make the stage **targeted, iterative, and cheaply-validated**.

---

## 3. Asset inventory

### 3a. Internal (code.amazon.com)

| Asset | Where | What it gives us |
|---|---|---|
| **NAKOS "pi" orchestrator** | `KaenaNeuronKernelLibrary/.../orchestrator` | Reference impl of our whole loop: beam accept-loop, greedy git-merge of multiple fixes/iter, device racing + cred-refresh + core-lock + per-measure timeout, fleet dashboard. |
| **NAKB benchmark** | `NeuronAgenticKernelsBenchmark` | Write/Optimize/Analyze tracks + scoring rubric → our regression scoreboard; swap orchestrators via `--orchestrator`. |
| **OPTNKI** | `OPTNKI` | ASPLOS'25 "NKI-ify Llama-3.2-1B" eval; scoring rewards NKI-FLOPS fraction (coverage signal). |
| **NeuronKernelNeffSim** | `NeuronKernelNeffSim` | Hardware-free perf oracle: `latency_ns` + per-engine pressure {PE/DVE/ACT/Pool}. Pre-rank/prune before device. |
| **nki.simulate test harness** | `KaenaNeuronKernelLibrary/test/utils/` | `OutputValidator`, `golden_provider`, `torch_ref_cache`, `determinism_checker`, `qor_collector`, `core_lock_manager`, `kaizen_host_source`. Drop-in correctness+QoR+fleet-concurrency. |
| **AnalyzerImage** | `KaenaNeuronKernelLibraryAnalyzerImage` | CI-ready off-hardware gating image (compiler + nki-cpu-simulator + torch-cpu + pytest). |
| **Megakernel skill** | `.../skills/neuron-nki-megakernel-writing` | SOP + `megakernel-skeleton.md` (annotated `transformer_tkg`) + `correctness-verification.md` + `sbuf-output-handoff.md` + `sbuf-allocation-modes.md` + `forking-subkernels.md` + `collective-test-ladder.md` + `blocker-catalog.md` + `templates/megakernel_template.py` + worked example. |
| **Optimization-analysis skills + 13 specialist personas** | `.../skills/*`, `.../agents/*.md` | Parallel multi-specialist analyze phase (bottleneck/cost/dma-order/roofline/sharding/…) with a pure-aggregator umbrella. |
| **Error-code catalog** | `.../skills/neuron-nki-docs/references/debugging/error-codes/` | ~28 per-code docs (EVRF/EOOM/EARG/…) → error→fix KB for the repair loop. |
| **Lesson-banking skill** | `.../skills/neuron-nki-optimization-log-to-memory` | run_log-arbitrated banking ("never trust the agent's self-report"), dedup + confidence + merge-case + public-safe schema. |
| **NeuroTile / TileStream** | `.../experimental/neurotile/`, `.../experimental/primitives/` | Composable tiling/DMA/SBUF primitives (6-knob action space) + loop-nest scaffolding. |
| **Giga-kernels** | `.../models/gpt_oss/c128_giga_kernel/`, `c512_giga_kernel/` | Full production single-trace fused decode-layer megakernels (imitation targets). |
| **Test-set minimizer** | `.../skills/nki-fast-test-minimum-set` | Greedy weighted set-cover (1200→20 tests) to keep the gate cheap. |
| **NKILib corpus + `_torch` twins** | `.../src/nkilib_src/nkilib/{core,experimental,models}/` | Harvest corpus + ready correctness oracles + few-shot exemplars. |
| **Coding guidelines / review principles** | `.../test/docs/llm_coding_guidelines_review/` | House style + review rubric to enforce on generated kernels. |
| Leads (unopened) | `ArgNeuron`, `DistributedGemmNKI`, `TorchToReadableNKI`, `NumpyToReadableNKI` | Layout solver; distributed GEMM; torch/numpy→NKI front-ends. |

### 3b. Public (AWS Neuron docs / GitHub / papers)

| Asset | URL | What it gives us |
|---|---|---|
| **NKI CPU simulator** | awsdocs-neuron .../nki/guides/nki_simulator.html | Run real NKI on CPU (NumPy). NaN-uninit gate; runtime HW-constraint exceptions; `NKI_PRECISE_FP=0/1` algorithm-vs-precision triage; LNC2 threads. |
| **neuron-agentic-development** | github.com/aws-neuron/neuron-agentic-development | Official 5-skill write→debug→profile→analyze agent suite + orchestrator. |
| **NKI-Agent paper** | arxiv.org/abs/2607.04395 | compile-verify-fix loop + NKIBench; loop >> SFT >> binary-reward GRPO. |
| **NKI Performance guide** | .../nki/nki_perf_guide.html | MBU≥60% / ~100% MFU / spill>30%⇒fuse gates + 10 named transforms. |
| **neuron-profile / Neuron Explorer** | .../about-neuron/profiling-tools.html | Real bottleneck + Source-Code viewer (`nki_source_location`) → profile→edit loop. |
| **nki-autotune** | github.com/awslabs/nki-autotune | Tile/layout sweep + metrics + results export → race/harvest backend. |
| **nkipy** | github.com/aws-neuron/nkipy | NumPy-like layer that abstracts tiling → first-draft scaffold. |
| **nki-library catalog + design specs** | .../nki/library/api/index.html | Full production kernel catalog incl. `transformer_tkg` / `attention_block_tkg` megakernels. |
| **Framework custom-op guide** | .../nki/guides/framework_custom_op.html | Drop generated kernels into torch/JAX models + autograd + HLO dump debug. |
| **Migration guides** | .../nki/migration/index.html | Deterministic version-pitfall error→fix rules (0.1→0.6 / SDK 2.27→2.32). |
| **nrtpy** | linked from nki index | Framework-free NEFF race harness. |
| **Tutorials / nki-samples** | .../nki/tutorials/, github.com/aws-neuron/nki-samples | Online-softmax/flash, PSUM tiling, fused Mamba/scan reference logic. |

---

## 4. Recommendations (tiered)

Legend: **Effort** S/M/L · **Device?** Y (needs Trainium) / N (offline-safe) · **Status** ☐ todo / ◐ in-progress / ☑ done.

### Tier 0 — crown jewels (do first)

- **R1. Adopt `nki.simulate` as the offline correctness gate.** Run the actual NKI source on
  CPU vs `spec.reference` → kills the tautology; add NaN-uninit assertion; turn HW-constraint
  exceptions into error→fix keys; run `NKI_PRECISE_FP=0` then `=1` to separate algorithm bugs
  from precision. Slots into the rank ladder as passed-simulate (rank 3). *Files:* `invent_engine.offline_gate`, `kernel_validation`. **Effort M · Device N · ◐** — *landed 2026-09-10: `kernel_simulate.py` (import-guarded runner, honest off-simulator deferral, `NKI_PRECISE_FP` algorithm-vs-precision triage, NaN/Inf reject), `offline_gate` folds the verdict in as the authoritative independent parity signal, `KernelValidation.from_simulate()` (rank-3 simulate tier). 20 tests; device-activation of the real simulate path pending one on-device run.*
- **R2. Reuse the internal `nki.simulate` test harness** (`OutputValidator`, `golden_provider`,
  `determinism_checker`, `core_lock_manager`, `kaizen_host_source`) instead of hand-rolling.
  **Effort M · Device N · ☐**
- **R3. Harvest the `nki-library` corpus + `_torch` twins.** Point `kernel_registry` at it;
  twins become oracles; kernels become few-shot exemplars. *Files:* `kernel_registry`, `invent_engine._prior_art`, `nki_knowledge`. **Effort M · Device N · ☐**
- **R4. Stand up a scoreboard (NAKB / NKIBench / OPTNKI).** Score every change; A/B our
  orchestrator vs NAKOS. **Effort M · Device Y (some) · ☐**
- **R5. Align with NAKOS + neuron-agentic-development before building.** Lift the beam
  accept-loop, greedy multi-fix git-merge, and device-race infra. **Effort L · Device Y · ☐**

### Tier 1 — high value

- **R6. Turn on the compile-verify-fix repair loop with `LLMAuthor` as default** (`max_repair_rounds>1`).
  The simulator (R1) makes it a fast CPU loop. *Files:* `invent_engine.__init__`, `kernel_author`. **Effort M · Device N · ☐**
- **R7. Build the error→fix KB** from the internal error-code catalog + migration guides →
  `kernel_rewrites`. **Effort M · Device N · ☐**
- **R8. Adopt the run-log-arbitrated lesson-banking design** (measured log arbitrates, not the
  agent's self-report). *Files:* `invent_engine._bank_*`, `overnight._emit_lesson`. **Effort M · Device N · ☐**
- **R9. Add NeuronKernelNeffSim as a HW-free perf oracle** to pre-rank/prune candidates. *Files:* `invent_engine`, `kernel_perf`. **Effort M · Device N · ☐**
- **R10. Encode the perf guide as a transform rulebook + gates** (MBU/MFU/spill + 10 transforms). *Files:* `kernel_author._PERF_PREAMBLE`, `kernel_mutator`. **Effort M · Device N · ☐**
- **R11. Wire real neuron-profile into the perf loop** (bottleneck + `nki_source_location`). *Files:* `invent_engine` profiler hook, `kernel_perf`. **Effort M · Device Y · ☐**

### Tier 2 — solid

- **R12. Author against NeuroTile / nki-autotune / nkipy** instead of raw tiling. **Effort L · Device N/Y · ☐**
- **R13. Multi-specialist analyze phase** (13 personas + pure-aggregator umbrella). **Effort L · Device N · ☐**
- **R14. Test-set minimizer** to keep the gate cheap as the set grows. **Effort S · Device N · ☐**
- **R15. Framework custom-op path** for end-to-end model validation + HLO dump. **Effort M · Device Y · ☐**
- **R16. `nrtpy` framework-free race harness.** **Effort S · Device Y · ☐**
- **R17. Grow oracle coverage + `audit_oracles()` as a CI gate** (simulator largely supersedes hand sims; keep as coverage insurance). **Effort S · Device N · ☑** — *landed 2026-09-10: added a non-vacuous FlashAttention oracle (full-softmax ref vs independent online-softmax sim), shrank the audit missing set 10→9, added the `KNOWN_UNCOVERED` documented+shrinking allowlist for the 9 primitives still needing a grounded reference (→R3), and `test_audit_oracles_is_a_ci_gate` (fresh-subprocess gate: zero vacuous, every uncovered kernel consciously allowlisted, allowlist only shrinks).*
- **R18. Enforce the NKI coding-guidelines/review rubric + hard constraints** (output=`shared_hbm`; one E4M3/trace; module-scope helpers). **Effort S · Device N · ☐**

### Tier 3 — longer-term / experimental

- **R19.** Imitate the gpt_oss giga-kernels for a full fused decode layer. **L · Y · ☐**
- **R20.** Mine the megakernel schedule-search artifacts as a schedule-search template. **M · N · ☐**
- **R21.** Evaluate TorchToReadableNKI / NumpyToReadableNKI as write-phase front-ends. **M · N · ☐**
- **R22.** Evaluate ArgNeuron (layout solver). **M · N · ☐**
- **R23.** Use AgenticCodeOptimizer `examples/nki` (before/after + evaluators) as optimizer eval material. **M · N · ☐**
- **R24.** Offline-test multi-core paths with `nki.simulate` LNC2. **M · N · ☐**
- **R25. Silent-perf-regression detector set** (from cross-HW research, §6). Both TPU and GPU
  showed the compiler *hiding* perf bugs. Add offline/profile detectors: (a) relayout/transpose-
  DMA count above a threshold, (b) KV/operand re-read multiplier (bytes moved ÷ bytes needed),
  (c) per-op roofline gap (measured ÷ floor) as the primary rank signal, (d) code-size/unroll
  blowup. *Files:* `kernel_perf`, `kernel_anticheat`, `fusion` (M4). **Effort M · Device N · ☐**

---

## 5. Megakernel-specific plan (priority)

Our `fusion.select_fusion_targets` has the right *shape* (fused OpSpec = sequential
composition) but no authoring/verification scaffold. The internal megakernel skill IS that
scaffold.

- **M1.** Replace from-scratch megakernel authoring with the skill's **template + skeleton**
  as the LLM author's scaffold (author fills stage subkernel calls, not plumbing).
- **M2.** Make the fused-OpSpec gate the **incremental fuse→compile→check** loop with a
  `_torch` twin (never one-shot — their worked example hit 8 errors, isolatable only
  incrementally). Maps onto `orchestrator._evaluate`.
- **M3.** Encode the **SBUF discipline** as hard author rules: one shared `BufferManager`;
  `pop_heap` at each stage boundary (#1 OOM fix); auto-alloc first; `out_in_sb`/
  `store_output_in_sbuf` handoff flags; `transposed_out/in` to avoid relayout DMAs;
  collective→residual order (residual AFTER reduce, never reduce the residual).
- **M4.** Add the **"fused-in-name-only" anti-pattern detector**: count `torch→@nki.jit`
  seams — a legitimate split has exactly ONE (forced by host index math). Bank it as an
  anti-pattern + check it inside `fusion.select_fusion_targets`.
- **M5.** **Harvest** `transformer_tkg` / `attention_block_tkg` / gpt_oss giga-kernels as
  starting points instead of authoring from scratch.
- **M6.** Give the author the **forking techniques** (additive default-off param →
  fork-inline for manual-SBM kernels → `__code__`-clone for shared kernels) so it can make
  subkernels fusion-ready without breaking callers.
- **The one architectural fact (bake into the author):** outer fn NOT `@nki.jit`-decorated;
  caller wraps with `nki.jit()` at the call site; subkernels run inline inside the active
  trace (SBUF-resident). A `@nki.jit` called from torch = separate kernel = the anti-pattern.

---

## 6. Cross-hardware (TPU / GPU) transferable techniques

Two research passes mined TPU and GPU kernel-optimization literature with a strict
"transferable-to-NKI, flag the ISA-specific parts" lens. TPU is weighted higher because it
is architecturally closer to Trainium (systolic matmul array + software-managed scratchpad +
a fusing compiler); GPU contributes algorithmic patterns. Everything below is mapped to a
concrete change in *our* stage; hardware facts that do **not** port are flagged inline.

### 6a. TPU — the transferable playbook

Primary source: SAIL Research, *"Optimizing Gemma 4 on TPU v6e"* (sailresearch.com/blog/
tpu-v6e-gemma), cross-checked against the JAX Scaling Book and Pallas/Mosaic TPU docs. The
blog took Gemma 4 31B prefill (8192 tokens, 2×2 v6e) from **~32% → ~63% MFU** (throughput
18,228 → 36,669 tok/s; TTFT 449.4 → 223.4 ms; speed-of-light ceiling was 58,431 tok/s /
100% MFU). Verified against the source 2026-09-10. The techniques, most-transferable first:

- **T1. Roofline-as-bug-detector — adopt as the core perf loop.** Their entire result came
  from one loop: *(1) compute what an op should cost from first principles, (2) measure what
  it actually costs, (3) treat the gap as a bug until proven a physical limit.* Without a
  roofline, a 1.539 ms attention kernel and a 788 µs AllReduce both just look like "how long
  that takes"; only against the roofline is one obviously a mis-tuned kernel and the other a
  hardware limit. → This is exactly what **R9 (NeffSim perf oracle)** + **R10 (perf rulebook)**
  + **R11 (neuron-profile)** should implement: attach a per-op roofline (PE-bound vs
  DMA-bound floor) to every raced kernel and gate/prioritize on the *gap*, not the raw
  latency. Cheap, offline, high-leverage. *Port cleanly; no ISA dependency.*
- **T2. Reuse-aware tile sizing (the single biggest win).** Their attention kernel was slow
  purely because it was **mis-tuned**: query-block size of 32 caused each KV entry to be
  re-fetched from HBM ~32× (a key is wanted by the 1024 queries in Gemma's sliding window).
  Retuning the block to 512 took the sliding layers from **1.539 ms → 261 µs (~6×)** with no
  algorithm change. → Direct mandate for **R12 (nki-autotune tile sweep)**: sweep tile/block
  sizes and *score by HBM bytes re-read*, not just latency. The SBUF analog of "keep queries
  parked in VMEM, stream keys once" is a first-class megakernel rule (M3). *Port cleanly;
  retune the specific numbers on Trn2 SBUF — do not copy 512.*
- **T3. Silent-performance-failure detectors (a whole catalog).** The bulk of the 52%→63%
  grind was hunting failures the compiler hid instead of erroring on. Each has a Trainium
  analog worth a detector/lint in `kernel_anticheat` / `kernel_perf` / the M4 "fused-in-name-
  only" check:
  - *Bad mesh/device ordering* silently cost ~2× (a ring that needs non-existent diagonal
    links). → Trn2 analog: collective/replica-group ordering vs the real NeuronLink topology.
  - *"Free" transpose that wasn't* — the compiler relabels an array column-major at load
    instead of moving bytes, then is forced to wire a real copying transpose into *every*
    forward pass (~173 µs/layer). → Trn2 analog: silent **relayout DMAs**; detect via
    transpose/relayout count in the profile.
  - *Instruction-memory (IMEM) overflow* — large blocks made Mosaic unroll a matmul to
    ~4.45 MB of code, over the ~4 MiB IMEM, causing invisible instruction-fetch stalls
    ("SyncWait"). → Trn2 analog: code-size/loop-unroll blowups; prefer compact loops over
    unrolled blocks. *Numbers are TPU-specific; the failure class ports.*
  - **Lesson for us:** the design already distrusts silent wins (mock/real labeling); extend
    that to silent *perf* losses. This is the strongest cross-HW confirmation of the M4
    detector + honest-banking direction.
- **T4. Epilogue / cross-boundary fusion into the resident kernel.** Two of the highest-value
  late wins: (a) fold `gelu(gate)·up` into the MLP kernel's output stage while still in VMEM,
  killing a redundant 176 MB HBM write + 88 MB read; (b) fold the q/k/v RMS-norms and RoPE
  into the qkv kernel and emit q/k/v as separate arrays, removing ~500 µs/layer of
  compiler-inserted copies/dtype-conversions/transposes at the kernel boundary. → This *is*
  the megakernel thesis (§5, M1–M3): keep activations SBUF-resident across stage boundaries;
  fold norms/activations/RoPE into the producing kernel; the expensive thing is the
  torch↔kernel seam (M4). *Ports cleanly to SBUF.*
- **T5. Collective-matmul overlap.** When TP collectives are on the critical path (their
  AllReduce was ~27% of a layer at ~93% of the ICI limit — i.e. *not* fixable by a faster
  collective), split AllReduce → ReduceScatter + AllGather, **defer** the AllGather to just
  before the next matmul that needs full tokens, and ring-overlap each half with the matmul
  it feeds (send a token/output slice to the neighbor while computing the next). Hides ICI
  behind compute you were running anyway. → Maps to a future distributed-megakernel rec
  (relates to R19/§5 and the internal `DistributedGemmNKI` lead). **⚠️ Flag:** this is the
  most topology-dependent technique — **NeuronLink ≠ ICI** (Trn2's interconnect and
  collective primitives differ from v6e's 2×2 torus ring), so the *pattern* (split/defer/
  overlap) ports but the slice/hop sizing and whether the compiler already does it must be
  re-derived on Trn2. Also note their XLA did this *badly* out of the box (only ~0.45 of
  ~1.3 ms/layer hidden) and they dropped to Pallas — evidence that a hand-written NKI
  collective-matmul can beat the compiler, which is our whole value thesis.
- **T6. Trust the fusing compiler first, hand-write only where it demonstrably fails.** Their
  narrative is a clean statement of our §0 thesis: XLA's out-of-the-box fusion is very strong
  (don't rebuild it); the wins came from the specific spots where the compiler silently gave
  up. → Keep targeting **compiler-weak primitives**, and use the roofline gap (T1) to *find*
  those spots rather than guessing.

**TPU non-ports / verify-on-Trn2 (do not copy blindly):**
- v6e has two **256×256** systolic arrays; Trainium's TensorEngine is **128×128** → pad/tile
  to **128**, not 256. (Pallas even enforces a 128-multiple last block dim; our NKI tiling
  already assumes 128.)
- **VMEM ≠ SBUF** capacity/bandwidth ratios — re-derive all tile sizes (T2) against Trn2 SBUF;
  the 512 block size is a v6e artifact.
- **ICI ≠ NeuronLink** (see T5) — topology, per-hop bandwidth, and collective set differ.
- IMEM/SyncWait specifics (~4 MiB) are TPU-only; only the *code-size-matters* lesson ports.

### 6b. GPU — transferable algorithmic patterns

GPU hardware (warps/SIMT, tensor cores, shared-mem/register file) does **not** map to
Trainium's engine model, so we take **algorithms and scheduling patterns**, not code. Sources:
FlashAttention (Dao et al. 2022) / FlashAttention-2 (Dao 2023), vLLM PagedAttention (Kwon et
al. 2023), CUTLASS/CuTe and Triton autotuning docs, and the FP8/MXFP4 microscaling literature.

- **G1. Online-softmax / streaming attention (FlashAttention).** Never materialize the N×N
  scores; tile over K/V, keep a running max + running denominator, rescale the accumulator as
  you go. IO-aware: read Q,K,V,O through HBM once, keep scores in on-chip memory. → We already
  have attention-sink / flash-style kernels; the transferable rule is the **streaming-reduction
  pattern** for any softmax/scan/norm primitive the compiler handles poorly, and it composes
  with SBUF residency (T4). *Algorithm ports; the tiling is ours.*
- **G2. Minimize non-matmul FLOPs + partition for the matmul engine (FlashAttention-2).**
  FA2's ~2× over FA1 (reported) came from *rearranging the algorithm so the systolic/tensor
  unit stays busy* — fewer rescales, non-matmul work off the critical path, work partitioned
  to keep the matmul engine saturated. → This is an MFU/PE-utilization rulebook item for
  **R10**: count non-matmul ops on the critical path and push them into epilogues (ties to
  T4). The "keep the matmul array fed" goal is *more* true on a systolic array than a GPU.
- **G3. Block / microscaling quant with high-precision accumulate.** FP8/MXFP4-style block
  scaling (per-block scale, MMA accumulate in higher precision) is the standard route to
  matmul throughput. → Encode as a perf lever + a correctness rule: **accumulate in fp32/bf16
  even when inputs are low-precision**, and validate with the `NKI_PRECISE_FP` algorithm-vs-
  precision split (R1). *Pattern ports; exact dtypes are Trn2-specific.*
- **G4. Megakernel-for-decode / persistent kernels.** The GPU "persistent megakernel" idea —
  one long-lived kernel that keeps state resident and walks the whole decode step to kill
  launch overhead and round-trips — is the same instinct as our SBUF-resident megakernel
  (§5) and the internal gpt_oss giga-kernels (R19). → Reinforces prioritizing the fused
  single-trace decode layer. *Instinct ports; mechanism is ours (SBUF, not CUDA graphs).*
- **G5. Autotune-and-cache per shape.** Triton/CUTLASS autotune tile/pipeline configs per
  problem shape and **cache the winner**. → Directly supports **R12** and a bank keyed by
  `(op, shape, dtype)` so the forever-run doesn't re-search a shape it already solved. Ties
  to G2/T2 (the thing you're tuning is reuse-aware tiling).
- **G6. Paged / indirect-DMA KV access (PagedAttention).** vLLM's block-table indirection
  (gather KV blocks via a lookup instead of contiguous reads) is the pattern behind efficient
  long-context/batched decode. → Transfers as an **indirect-DMA / gather-DMA** pattern for
  KV-cache kernels on Trainium; relevant when we get to decode megakernels (R19). *Pattern
  ports; it's a DMA-descriptor technique, not CUDA-specific.*

**GPU non-ports:** warp-level primitives, shared-memory bank-conflict tuning, CUDA-graph
capture, and register-file occupancy math are SIMT-specific and do not translate — ignore
them. Take the *algorithm* (G1/G2) and the *scheduling instinct* (G4/G5/G6), never the code.

### 6c. Net new/changed recommendations from cross-HW research

The cross-HW pass did not invent new subsystems — it **sharpened and re-prioritized** existing
recs, which is the honest outcome:
- **Promotes R9+R10+R11** (roofline perf loop) — T1 is the single most-endorsed idea across
  both TPU and GPU; make the roofline *gap* the ranking signal, not raw latency.
- **Promotes R12** (autotune) with a concrete objective from T2/G5: sweep tiling, score by
  **HBM/SBUF bytes re-read**, and cache the winner per `(op, shape, dtype)`.
- **Strengthens M3/M4** (megakernel SBUF discipline + fused-in-name-only detector) — T3's
  silent-failure catalog and T4's epilogue fusions are independent confirmation; add explicit
  **relayout-DMA-count** and **code-size** detectors.
- **Adds a small new rec R25 (below):** a "silent-perf-regression" detector set, since both
  architectures showed the compiler *hiding* perf bugs rather than erroring.

---

## 7. Open questions / to-validate before over-investing

- **Does knowledge-injection actually help?** ACO ablation says a local RAG didn't; the paper
  says the loop + frontier model dominate. → A/B R7/R12-corpus/R18 on NAKB before scaling them.
- **Build vs harvest boundary.** How much of NAKOS/neuron-agentic-development do we adopt
  wholesale vs keep ours? (Keep: forever-run over many models, config search, honest
  mock/real labeling, cross-model bank. Adopt: authoring internals, device-race infra.)
- **Simulator ≠ silicon.** Keep the rank ladder honest: sim pass = rank 3 → must revalidate
  on device (the Mamba 2e-7-vs-67-off lesson).
- **Device budget.** Device-needed recs (R4/R5/R11/R15/R16/R19) must be scheduled so they
  don't disturb the running soak.

---

## 8. Implementation status (check off here)

| # | Rec | Tier | Effort | Device | Status |
|---|---|---|---|---|---|
| R1 | nki.simulate offline gate | 0 | M | N | ◐ (offline done+tested; device-activation pending) |
| R2 | reuse sim test harness | 0 | M | N | ☐ |
| R3 | harvest nki-library + twins | 0 | M | N | ◐ (registry multi-dir landed; corpus indexing = device follow-up) |
| R4 | NAKB/NKIBench scoreboard | 0 | M | Y | ☐ |
| R5 | align w/ NAKOS + OSS | 0 | L | Y | ☐ |
| R6 | repair loop + LLMAuthor default | 1 | M | N | ☐ |
| R7 | error→fix KB | 1 | M | N | ☐ |
| R8 | run-log-arbitrated banking | 1 | M | N | ☐ |
| R9 | NeffSim perf oracle | 1 | M | N | ☐ |
| R10 | perf rulebook + gates | 1 | M | N | ☐ |
| R11 | neuron-profile in perf loop | 1 | M | Y | ☐ |
| R12 | NeuroTile/autotune/nkipy | 2 | L | N/Y | ☐ |
| R13 | multi-specialist analyze | 2 | L | N | ☐ |
| R14 | test-set minimizer | 2 | S | N | ☐ |
| R15 | framework custom-op e2e | 2 | M | Y | ☐ |
| R16 | nrtpy race harness | 2 | S | Y | ☐ |
| R17 | oracle coverage + CI audit | 2 | S | N | ☑ (FlashAttention oracle + audit CI gate + allowlist) |
| R18 | coding-guidelines rubric | 2 | S | N | ☑ (module-scope-helper in static_lint; fuller rubric in internal nki_lint) |
| R19 | gpt_oss giga-kernel imitation | 3 | L | Y | ☐ |
| R20 | schedule-search template | 3 | M | N | ☐ |
| R21 | Torch/Numpy→NKI front-end | 3 | M | N | ☐ |
| R22 | ArgNeuron layout solver | 3 | M | N | ☐ |
| R23 | ACO before/after eval | 3 | M | N | ☐ |
| R24 | LNC2 sim multi-core | 3 | M | N | ☐ |
| R25 | silent-perf-regression detectors | 3 | M | N | ☑ (silent_perf.py: re-read / relayout-DMA / code-size) |
| M4 | fused-in-name-only detector | — | S | N | ☑ (fusion.detect_fused_in_name_only + offline-gate wiring) |
| M1–M3,M5,M6 | megakernel scaffold/gate/rules | — | — | mixed | ☐ |

---

## Appendix A — file touch-point map

- `orchestrator.py::run_deep_stages` — Stage-4 stub → real (gated) `InventEngine.run_op` call.
- `invent_engine.py` — `offline_gate` (add sim), `__init__` (author default + rounds),
  `_device_race`/`_device_compile_probe` (harden), `_bank_*` (run-log arbitration).
- `kernel_author.py` — `LLMAuthor` default, `build_author_prompt` (perf rulebook), preambles.
- `kernel_providers.py` — provider/model/token/timeout wiring.
- `kernel_oracles.py` — coverage + `audit_oracles()` CI gate (sim reduces need for hand sims).
- `kernel_validation.py` — rank ladder (sim=rank3).
- `kernel_rewrites.py` — error→fix catalog (from error codes + migration guides).
- `nki_knowledge.py` — retrieval corpus (nki-library exemplars) — A/B its value first.
- `kernel_perf.py` / `kernel_mutator.py` — perf oracle + profile-guided levers.
- `fusion.py` — megakernel targets + fused-in-name-only detector (M4).


---

## 9. Implementation log

Chronological record of landed work (branch `feat/attention-sink-kernel`). Each
entry is offline-safe and verified against the framework pytest in the trainopt
venv (923 baseline passing; pre-existing torch/boto3-only failures untouched).
The running 72h soak is NOT disturbed — it runs from a separate checkout on the
Kaizen desktop and Stage-4 invent is still a stub there, so these edits are dormant
until Stage 4 is wired (a later rec) and a fresh launch is authorized.

- **2026-09-10 — R1 (nki.simulate offline gate) ◐.** New `kernel_simulate.py`:
  import-guarded runner that executes the REAL authored kernel on CPU via
  `nki.simulate_kernel` and compares to the reference (breaking the numpy_impl
  tautology). Honest deferral (`ran=False`) off a simulator box; `NKI_PRECISE_FP=0→1`
  algorithm-vs-precision triage; NaN/Inf reject; both invocation conventions;
  injectable `sim_fn`/`load_entry` seams for CPU testing. Wired into
  `invent_engine.offline_gate` (new `simulate_fn` seam + `_simulate` + OfflineGate
  fields `simulate_ran/ok/precise_fp`) as the authoritative independent parity
  signal. `kernel_validation.from_simulate()` maps the outcome onto the ladder
  (rank-3 simulate → REVALIDATE_ON_DEVICE per the Mamba lesson). 20 new tests, all
  green; off-simulator behaviour byte-for-byte unchanged. **Open:** the real
  `simulate_kernel` execution path needs one on-device validation (can't run `nki`
  on the dev laptop) — until then it cleanly defers.
- **2026-09-10 — R17 (oracle coverage + audit CI gate) ☑.** Added a non-vacuous
  FlashAttention oracle (full-`[S,S]`-softmax reference vs an independent streaming
  online-softmax sim; test proves it catches a dropped-K/V-block bug). Audit
  `missing` set shrank 10→9. Added `KNOWN_UNCOVERED` — a documented, shrinking
  allowlist for the 9 primitives (KDA, LightningAttn, RWKV6/7, mLSTM, sLSTM, RGLRU,
  PowerRetention, GlmMoeDsa) that still lack a GROUNDED reference (deliberately not
  faked from memory — a wrong ground truth is worse than none; they land as R3
  harvests their `_torch` twins). `test_audit_oracles_is_a_ci_gate` runs the audit in
  a fresh subprocess and enforces: zero vacuous oracles, every uncovered kernel
  consciously allowlisted, allowlist only shrinks.
- **2026-09-10 — R3 (harvest nki-library corpus) ◐.** `KernelRegistry` gained
  multi-dir harvest: `extra_dirs` arg + `$TRN_OPT_EXTRA_KERNEL_DIRS` (os.pathsep),
  so a cloned nki-library checkout is indexed ALONGSIDE the in-repo kernels (primary
  wins a name clash; a malformed manifest falls through to the next dir). nki-library
  is already in the borrow list (`kernel_sources.yaml`) and does not cover the 9
  KNOWN_UNCOVERED exotic primitives, so it cannot ground those oracles — grounded
  refs come from the model source repos (a separate harvest). Actual manifest-
  indexing of a checkout + lifting its `_torch` twins into oracles is the
  device/external follow-up. 5 tests.
- **2026-09-10 — R25 (silent-perf-regression detectors) ☑.** New `silent_perf.py`
  flags the compiler-hidden waste from the cross-HW T3 lesson: `operand_reread`
  (HBM traffic ÷ minimum-needed; the KV-re-read class), `relayout_dma` (share of the
  DMA path in compiler-inserted relayout/transpose; the "free transpose that
  wasn't"), `code_size` (instruction-footprint blowup — fires only with a caller-
  supplied limit, no baked Trn2 constant), and a `roofline_gap` companion to
  roofline.py's %SOL. Findings carry a `route_token` so `kernel_perf.classify_
  bottleneck` (now recognizing "relayout"/"re-read") routes the matching lever.
  Complements roofline.py's MFU-implausibility (fake-speedup) guard. 18 tests.
- **2026-09-10 — M4 (fused-in-name-only detector) ☑.** `fusion.detect_fused_in_
  name_only` + `count_kernel_seams`: a true megakernel is ONE trace with inline
  subkernels; ≥2 torch→NKI seams (@nki.jit/baremetal/nki_op/wrap_nki) means the
  intermediates round-trip through HBM. Wired into `offline_gate` for fused specs
  (rejected before device time, banked as an anti-pattern); non-fused ops
  unaffected. 9 tests.
- **2026-09-10 — R18 (coding-guideline hard constraints) ☑.** Added the module-
  scope-helper rule to the TRACKED `static_lint` (the linter the offline gate calls):
  a helper defined inside a kernel body (indented def) breaks the tracer's source
  introspection → rejected. The fuller review rubric (incl. one-E4M3-per-trace, the
  ~25-rule migration table) lives in the internal-only `nki_lint` (git-excluded,
  never pushed — internal-sourced material), which was also extended locally. 4
  tests + a catalog-stays-clean regression.

> **Compliance note:** `.git/info/exclude` marks `nki_lint.py`, `bank_kernels.py`,
> `blockers.py`, and parts of `knowledge-bank/` as INTERNAL-ONLY (internal-sourced /
> customer-derived) — never pushed. All committed work above is in tracked, pushable
> files and does not import an internal-only module from a tracked test.