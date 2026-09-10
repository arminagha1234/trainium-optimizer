# NKI / Megakernel-Writing Stage — Improvement Plan (living doc)

> **Status:** DRAFT v1 · started 2026-09-10 · owner: Armin
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
  from precision. Slots into the rank ladder as passed-simulate (rank 3). *Files:* `invent_engine.offline_gate`, `kernel_validation`. **Effort M · Device N · ☐**
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
- **R17. Grow oracle coverage + `audit_oracles()` as a CI gate** (simulator largely supersedes hand sims; keep as coverage insurance). **Effort S · Device N · ☐**
- **R18. Enforce the NKI coding-guidelines/review rubric + hard constraints** (output=`shared_hbm`; one E4M3/trace; module-scope helpers). **Effort S · Device N · ☐**

### Tier 3 — longer-term / experimental

- **R19.** Imitate the gpt_oss giga-kernels for a full fused decode layer. **L · Y · ☐**
- **R20.** Mine the megakernel schedule-search artifacts as a schedule-search template. **M · N · ☐**
- **R21.** Evaluate TorchToReadableNKI / NumpyToReadableNKI as write-phase front-ends. **M · N · ☐**
- **R22.** Evaluate ArgNeuron (layout solver). **M · N · ☐**
- **R23.** Use AgenticCodeOptimizer `examples/nki` (before/after + evaluators) as optimizer eval material. **M · N · ☐**
- **R24.** Offline-test multi-core paths with `nki.simulate` LNC2. **M · N · ☐**

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

> **PENDING** — two research agents are mining TPU (Pallas/Mosaic, XLA fusion, MXU tiling,
> Gemma-on-v6e serving, incl. the SAIL research blog) and GPU (flash-attention, CUTLASS/
> Triton, persistent/megakernels, FP8/MXFP4, autotuning) literature, with a strict
> "transferable-to-NKI, flag ISA-specific" lens. Findings will be appended here as §6a (TPU)
> and §6b (GPU). Priority: TPU is architecturally closer to Trainium (systolic matmul +
> scratchpad + XLA), so weight it higher; GPU contributes algorithmic patterns.

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
| R1 | nki.simulate offline gate | 0 | M | N | ☐ |
| R2 | reuse sim test harness | 0 | M | N | ☐ |
| R3 | harvest nki-library + twins | 0 | M | N | ☐ |
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
| R17 | oracle coverage + CI audit | 2 | S | N | ☐ |
| R18 | coding-guidelines rubric | 2 | S | N | ☐ |
| R19 | gpt_oss giga-kernel imitation | 3 | L | Y | ☐ |
| R20 | schedule-search template | 3 | M | N | ☐ |
| R21 | Torch/Numpy→NKI front-end | 3 | M | N | ☐ |
| R22 | ArgNeuron layout solver | 3 | M | N | ☐ |
| R23 | ACO before/after eval | 3 | M | N | ☐ |
| R24 | LNC2 sim multi-core | 3 | M | N | ☐ |
| M1–M6 | megakernel scaffold/gate/rules | — | — | mixed | ☐ |

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
