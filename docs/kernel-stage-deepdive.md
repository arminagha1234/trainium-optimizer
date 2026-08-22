# Kernel stage — deep-dive plan

Status: **PLAN.** How to turn today's Stage-4 "invent" stub + scattered parts
(`kernel_registry`, `kernel_rewrites`, `kernel_repair`, `invent_engine`) into a
kernel stage that actually takes a model the compiler can't lower and produces a
**measured, correct win on-device** — and gets *better at it over time*.

Grounded in two studies: an inventory of our own code (what's built vs designed)
and a decomposition of the Neuron team's mature "AutoFixer" kernel-authoring
system. Both point to the same thesis.

---

## 0. The reframing — what "better at writing kernels" actually means

Two findings reframe the whole question:

1. **The bottleneck is not authoring cleverness — it's the *measurement path*.**
   There is **no code path today that puts an authored or registered kernel into
   a *served* model** (the only injection hook is the hardcoded MoE megakernel
   swap, `neuron_worker.py:406-418`). So even a perfect kernel can only bank a
   lesson; it can never prove an end-to-end throughput win. Everything upstream
   of that is theater until it exists. **This is priority #1.**

2. **The best way to "write a kernel" is often to *not*.** The mature system runs
   an **escalation ladder** — torch/graph rewrite → reuse an authored kernel →
   pass a param to an existing kernel → compiler flag → author a new kernel — and
   only reaches the last rung for primitives with no faithful in-graph
   expression. Qwen3.5 proved this: the `.tril()`→`TensorScalarAffineSelect`
   reject is fixed by a constant-mask **rewrite, no kernel**.

So: **more stages, yes — but the new stages are an escalation ladder + a
validation/measurement spine, not "more authoring."**

---

## 1. Where we are (built vs designed)

- **Served pipeline puts a real kernel in exactly one place**: the MoE fused-NKI
  megakernel *borrow* (`orchestrator.py` Stage-3 → `native_pytorch.moe_kernel_candidates`
  → `neuron_worker.swap_moe_forward`). Every other "kernel stage" is a
  compiler-flag search. Stage-4 INVENT is a stub (`orchestrator.py:336-349`).
- **`invent_engine` is standalone** — never called by `orchestrator`/`overnight`;
  `author_kernel` is a **10-op recipe table, not an LLM**, and covers *none* of
  the routed primitives (DeltaNet/Mamba/MLA are Harvest-only).
- **The new parts are unwired**: `kernel_repair.KernelRepairLoop` and
  `kernel_rewrites` are tested but not called from `run_op`; `preflight_check` in
  production (`overnight.py:235`) is never passed the registry, so linear-attn is
  still skipped even when a kernel is registered; nothing retrieves bank lessons
  at author time.

We have good *parts*. We do not have a *stage*.

---

## 2. Target: Stage 4 becomes an explicit sub-pipeline (the "more stages" answer)

Model the kernel stage as a **data-driven sub-pipeline** — each sub-stage
declares `requires`/`produces` and is topologically ordered (steal AutoFixer's
`Stage(requires, produces, precondition, run)` graph, `model_bringup.py`), so
sub-stages compose and *skip by dependency* instead of a hardcoded sequence. A
model that only needs a rewrite never reaches the author; a model with a
registered rank-4 kernel skips straight to compose+measure.

```
Stage 4 = KERNEL PIPELINE
 4a DETECT & ROUTE      primitive -> kernel need     [BUILT: kernel_registry, preflight.kernel_route]
 4b ESCALATION LADDER (cheapest rung that works wins; stop as soon as one does)
    4b.1 REWRITE        graph rewrite from the catalog (.tril->const-mask, int64-topk->float)
                        [catalog BUILT: kernel_rewrites; needs to run as a stage + a model-forward patcher]
    4b.2 HARVEST/REUSE  registered kernel, RANK-AWARE: rank4 reuse; rank3 reuse + MANDATORY on-device revalidate
                        [_prior_art BUILT; add the rank ladder + reuse router]
    4b.3 KERNEL-PARAM   an existing kernel already accepts the needed knob (e.g. softcap) -> just pass it   [NEW]
    4b.4 COMPILER-FLAG  the Stage-5 optlevel/auto-cast sweep                              [BUILT]
    4b.5 AUTHOR         only if no rung above fixed it
 4c AUTHOR SUB-PIPELINE (when 4b.5 fires) — translate the MATH (not code) from a reference, then the ENFORCED gates:
       static lint  ->  toolchain preflight  ->  numerics vs CPU/HF oracle  ->  compile to a NON-EMPTY NEFF
       [static_lint BUILT in invent_kernels; the rest need consolidating into one gate]
 4d REPAIR LOOP        author -> compile -> feed exact error back -> re-author (<=N rounds)
                       [BUILT: kernel_repair; wire into run_op at invent_engine.py:484]
 4e VALIDATE IN TIERS  rank ladder with the sim!=silicon wall: rank3 = simulate-correct, rank4 = on-device
                       [NEW: the ladder + a per-primitive (reference, sim, make_inputs) oracle registry]
 4f COMPOSE & MEASURE  get the kernel into a SERVED model + measure end-to-end     [NEW — the #1 blocker]
 4g PERSIST & REUSE    lesson by symptom; reuse-vs-author routing; bank retrieval at author time
                       [partly BUILT: bank writes; add symptom retrieval + reuse router]
```

For linear-attention primitives specifically (DeltaNet/Mamba/RWKV/…), 4c–4f
carry two extra, non-negotiable steps proven necessary by the corpus:
- **Prefill and decode are TWO kernels** (chunked-parallel prefill + O(1)
  recurrent decode) + a **handoff test** asserting `prefill→decode == full` and
  that the recurrent state layout/scale matches (it is *not* guaranteed by
  construction — parallel-authored halves diverge).
- **A full-layer composition oracle + "defeaters"**: a passed *set of primitives*
  is not a passed *layer*. Validate the whole mixer/layer against an HF-faithful
  CPU oracle, and add adversarial traps (wrong slope, dropped gate, flattened
  decay) that must diverge loudly — this catches the compounding-over-N-layers
  and missing-kernel→CPU-fallback "fake-GREEN" bugs.

---

## 3. Structural ideas to adopt from the mature system (ranked)

1. **A rank ladder with a hard `simulate ≠ silicon` wall.** `analysis(0) →
   failed-compile(1) → compiled/failed-numerical(2) → passed-**simulate**(3) →
   passed-**on-device**(4)`. A `nki.simulate` pass is **not** hardware truth (a
   Mamba scan simulated to 2e-7 ran ~67 off on real Trn2). rank-3 is reusable
   only with *mandatory on-device re-validation*; only rank-4 is HW-green. This
   is the single most valuable anti-"fake-GREEN" idea and it validates our PR #24
   on-device-race fix.
2. **One consolidated correctness gate** = numerics allclose (global **and**
   per-element) AND a non-empty NEFF on disk (import-only success is rejected).
   Stops every kernel re-rolling its own validate script.
3. **A CPU-oracle registry**: `(reference, numpy-sim, make_inputs)` per primitive,
   keyed by canonical name **and all aliases** (an orphan oracle silently *skips*
   numerical verification for a whole class — add a "not vacuous" test).
4. **Persisted, timestamped lesson notes** with a fixed schema + a 1-line lesson,
   read *first* by the next round. Turns N lonely retries into a compounding
   corpus. (Link the reference, never copy source — IP.)
5. **Reuse-vs-author routing that never dead-ends in "blocker"**:
   REUSE(rank4) / REUSE-but-revalidate(rank3) / CONTINUE(rank≥1) / AUTHOR(empty).
6. **Cheap static lint + toolchain preflight before the expensive compile** —
   catch tracer-rejection classes (floordiv-on-loop-index) and version-mismatch
   false-fails in ms, not a compile cycle.
7. **Prefill/decode split + mandatory handoff/state-continuity test.**
8. **Full-layer composition oracle with explicit defeaters.**

Plus the doctrine threaded through all of it: **rewrite/patch before kernel**,
and **never label a permanent blocker without an attempt that reached the
compiler.**

---

## 4. Sequenced build order (anchored on the #1 blocker)

**Phase 1 — Unblock measurement (nothing matters until this exists).**
- Generalize the MoE-only swap into a **generic kernel-injection hook**:
  `neuron_worker.py` loads a kernel by `(entry, path)` from a resolver and
  monkeypatches the target op's forward; `native_pytorch.py` gains a config axis
  alongside `moe_kernel`; the resolver reads a `KernelSpec` (registry) or an
  `AuthoredKernel`.
- Land the **rank ladder** (4e) + a **CPU-oracle registry** (3).
- Exit criteria: a *registered* kernel (even the existing MoE one, re-expressed
  through the generic hook) produces a measured end-to-end number, tagged rank-4.

**Phase 2 — Wire the parts we already built.**
- Drive `run_op` authoring through `KernelRepairLoop` (`invent_engine.py:484`);
  `compile_fn` wraps `AuthoredKernel.build()`, feedback via `match_error`.
- Call `invent_engine` from the orchestrator Stage-4 (replace the stub,
  `orchestrator.py:336-349`); feed results through `_update_incumbent`.
- Thread a `KernelRegistry` + `kernels_wired=True` into `preflight_check`
  (`overnight.py:235`) so a registered kernel actually lets a linear-attn model
  through.
- Retrieve bank anti-patterns/gotchas by symptom **before** authoring.

**Phase 3 — The escalation ladder (4b) + reuse routing.**
- REWRITE stage that applies a catalog rewrite to the model forward (start with
  `.tril()`→const-mask, the Qwen3.5 unblock) and re-compiles; KERNEL-PARAM stage;
  rank-aware HARVEST. Cheapest rung wins.

**Phase 4 — A real author (replace the recipe table).**
- Swap the 10-op recipe dispatch (`invent_kernels.py:795-828`) for an LLM/agent
  author that translates the *math* from a reference, fed the retrieved lessons +
  repair feedback + the C1-C8/K1-K2 contract. (Fix the tautological offline
  parity for the 5/6 recipe ops along the way.)

**Phase 5 — Linear-attention depth (the DeltaNet target).**
- Prefill/decode split + handoff test + full-layer composition oracle + defeaters
  (§2). This is what turns "Qwen3.5 compiles" (Phase 3 rewrite) into "Qwen3.5 is
  *fast*" (the chunked kernel).

Dependency reality: **Phases are gated by Phase 1.** A kernel win is unprovable
without the measurement path, so resist the temptation to do Phase 4 (fun
authoring) first.

---

## 5. Risks, non-goals, honesty

- **Don't confuse "compiles" with "fast".** The `.tril` rewrite makes Qwen3.5
  *run*; the recurrent scan is still ~0.004% MFU. Running-at-all (rewrite) and
  fast (chunked kernel) are separate deliverables; report them separately.
- **On-device is the only truth.** Keep the rank-3→4 wall; never promote a
  simulate pass to HW-green.
- **IP boundary holds.** Kernel *source* stays external (`$TRN_OPT_KERNEL_DIR`);
  this repo carries orchestration, gates, the oracle registry, and lessons
  (methodology/math), never proprietary kernel bodies.
- **Non-goal:** a general NKI compiler. We route a bounded set of known
  primitives and escalate; unknown primitives get an honest, named "needs
  kernel" work item, not a fabricated result.
