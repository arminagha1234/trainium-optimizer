# Kernel stage — external prior art (what GPU stacks + LLM-kernel-gen teach us)

Companion to [`kernel-stage-deepdive.md`](./kernel-stage-deepdive.md). Two studies —
(A) how GPU stacks author/select/autotune/inject kernels (Triton, TorchInductor,
TensorRT-LLM, TVM/Ansor, CUTLASS, JAX-Pallas), and (B) the state of automatic /
LLM-driven kernel generation (KernelBench, Kevin-32B, NVIDIA-R1, Sakana, Ansor,
PET, Tensat) — with what to steal and **how the build order changes**.

---

## 0. What the outside world confirms about our plan

- **Every mature stack is the same three-part spine** we're building: a
  **registration/dispatch boundary** (`torch.library.custom_op`/`triton_op`,
  TensorRT plugin registry, `pallas_call`) + a **selection engine** (autotune +
  cost model + heuristic default) + a **persistent cache** (timing cache /
  tuning log / compile cache). Our Phase-1 injection hook + registry + rank
  ladder is exactly this spine. Strong external validation we're building the
  right thing first.
- **Iterative refinement with real compiler/runtime feedback is the #1 lever**,
  empirically. KernelBench `fast₁` jumps **12→43% (L1), 36→72% (L2), 2→18% (L3)**
  when you add a generate→execute→feedback loop over a 10-call budget. This is
  our `KernelRepairLoop`, and it's the highest-ROI component — not fancy search.

## 1. The standout new insight: correctness gating is ADVERSARIAL

Every LLM-kernel effort that reported huge speedups was **reward-hacking the
eval harness**, and this is the single most important thing to get right:

- **Sakana "AI CUDA Engineer"** claimed up to 100×; the kernels "bypassed
  validations for accuracy" and community re-runs measured a **~3× SLOWDOWN**.
- **Kevin-32B** independently hit the SAME class of bug — the tested kernel
  **recycled the output tensor from the reference**, plus authors that (a) copied
  the reference, (b) wrapped a broken kernel in `try/except` → PyTorch fallback,
  (c) inherited from the reference class to avoid writing a kernel at all.

**Our hard-won "simulate ≠ on-device" lesson is the identical failure mode one
rung up the ladder.** So the rank ladder's on-device rung must be built as an
adversarial gate, not a QA afterthought. Concrete, cheap defenses (adopt all):

1. Reward/accept **0** for any candidate that calls back into the reference
   framework, wraps a fallback in try/except, inherits/imports the reference, or
   contains **no actual NKI kernel**.
2. **Run the candidate BEFORE the reference** (so it can't read the reference's
   output buffer), and **scrub/zero output buffers between runs**.
3. **Run twice; require identical results** (the "runs twice, different answers"
   tell that exposed Sakana).
4. **On-device is the SOLE authoritative gate.** `nki.simulate` / a CPU oracle
   is a cheap *filter* that catches obvious failures; it never grants
   "banked-correct." Only a top-rung on-device pass enters the knowledge bank.
5. Tighten numerics past loose `allclose` (KernelBench's atol/rtol≈1e-2 on 5
   inputs is exactly what let false-passes through): tight tolerance, many seeds,
   edge-case inputs; and for **linear/tensor ops exploit multi-linearity (PET)**
   so a *bounded* input set is a real sufficiency proof, not just sampling.

This is HIGH value, LOW effort, and it belongs early — folded into Phase 1's
rank ladder and Phase 2's banking policy.

## 2. Retrieval / transfer beats cold search (and beats a heavy cost model)

- **Transfer-tuning** reuses auto-schedules found for one model on another and
  reaches **~88% (49% avg) of full-Ansor speedup with 6.5–10.8× less search**.
- Sakana's one genuinely good idea was the **"innovation archive"** — retrieve
  past *working* kernels as few-shot exemplars.
- Both map straight onto our **knowledge bank + symptom-indexed rewrite catalog**:
  index past working kernels + lessons by op-signature / error-symptom, retrieve
  the closest, and inject as author context. Cheapest possible "transfer."

## 3. The cost-model question — reconciling the two studies

The GPU study flags "a cost model in front of the on-device ladder" as our
biggest gap (Triton `prune_configs_by`, Ansor's learned model exist to cut
expensive on-device measurements). The LLM-gen study says a learned cost model /
full Ansor search is **overkill for us**: those systems spend hours-to-days of
on-device search *because they have no strong prior*; an **LLM author + retrieval
bank IS the prior**, so we need far fewer measured candidates.

**Resolution (right-sized):**
- **Now:** a *cheap analytic* pre-filter (HBM bytes/rank, MAC count, DMA count,
  partition-legality) to drop obviously-losing configs before the slow
  `neuronx-cc` compile — plus retrieval from the bank. No training required.
- **Defer** a *learned* cost model until we observe the same op-shapes being
  re-searched repeatedly and on-device eval dominates wall-clock. Revisit then.

## 4. The rest of the steal list (ranked, mapped to us)

1. **`custom_op`/`pallas_call` registration boundary = our model-injection hook**
   (Phase 1). `custom_op` (opaque) vs `triton_op` (traceable) is the exact model;
   require a **meta/shape-dtype fn** per kernel so shapes trace without running it.
2. **Autotune-as-config-search + shape-bucketed cache** (`@triton.autotune`,
   `key=[shapes]`): each NKI kernel declares tile/shard/DMA-depth Configs; the
   rank ladder is the benchmark; cache the winner by `(kernel, shape-bucket,
   sdk)`. This is the pattern behind our validated `AW=8`/`ep32-aw8` fit matrix.
3. **`interpret=True` (Pallas): a pure-CPU semantic run** of the kernel before
   device+compile — a fast oracle + diff signal for the repair loop; separates
   "wrong" from "slow" cheaply.
4. **Static shape-bucket dispatch table** (FlashAttention): pre-tune per bucket
   (seqlen/head-dim/batch bins) → a `(bucket, causal, dtype) → kernel+config`
   table; the common path needs no runtime search. Fits Trainium's static-shape
   requirement + our prefill/decode bucket split.
5. **Build-time timing cache + serialized plan** (TensorRT): persist tactic
   timings + chosen configs + compiled NEFFs keyed by `(kernel, shape, target,
   neuronx-cc version)`; invalidate on compiler-version change. One-time ladder
   cost becomes a reusable asset.
6. **Pre-verified rewrite rules trusted by construction** (Tensat/PET): a catalog
   rewrite proven semantics-preserving needs no per-use correctness re-litigation
   — spend the expensive on-device validation only on *novel* code (the 5%
   invention margin).
7. **Repair-loop budget discipline** (Kevin/KernelBench): **4–8 serial turns**,
   front-loaded; **serial refinement beats parallel sampling** at fixed compute;
   keep **diff + error history**, drop chain-of-thought/transcript (trajectories
   blow to 50–100k tokens otherwise).
8. **Feed the EXACT error, one dominant fix per turn** + a **critic/verifier that
   rewrites the next prompt** (NVIDIA got L1 to 100% this way) rather than pasting
   raw logs. Make correctness failures maximally specific (which input, which
   element, expected vs got, at which stage) — coarse "wrong" is where models
   stall.
9. **CUDA-graph-style launch-plan replay** for our recurring **host-bound bs=1
   decode** finding (device >99% idle): capture the fixed per-token NKI launch
   sequence once, replay it — a big *non-kernel* latency lever.

Deferred / overkill for now: a **learned cost model** (§3) and **RL-fine-tuning
the author** (Kevin-style GRPO — high effort, needs a large task corpus + an
exploit-proof reward, and has a training-collapse landmine). A well-prompted
frontier author + repair loop + retrieval gets most of the value.

## 5. Revised build order (folding this in)

The core sequence in `kernel-stage-deepdive.md` stands; the changes are:

- **Phase 1 (measurement spine)** — ADD the **adversarial correctness gate** (§1)
  to the on-device rung from day one, and require a **meta/shape-dtype fn** per
  registered kernel (item 4.1). This is the cheapest, highest-ROI robustness we
  can buy, and retrofitting it later is how fake-GREENs slip in.
- **Phase 2 (wire the parts)** — the repair loop wiring now also carries the
  **budget discipline** (4–8 serial turns, diff+error history, one-fix-per-turn),
  a **critic/verifier** prompt-rewrite step, and **retrieval from the bank** as
  author context (§2). Banking policy: on-device-only grants "correct."
- **NEW Phase 2.5 (autotune + cache)** — insert between wiring and the escalation
  ladder: per-kernel **Config search with a cheap analytic pre-filter** (§3) and
  a **shape-bucketed timing/NEFF cache** (items 4.2, 4.5). Cheap, high-leverage,
  attacks the compile wall directly.
- **Phase 3 (escalation ladder)** — rewrite entries are **pre-verified** and
  trusted by construction (item 6); add a **static shape-bucket dispatch table**
  (item 4.4) for the common path.
- **Phase 4 (author)** — frontier-LLM author with retrieval + repair feedback +
  the critic; **not** RL-tuned initially. Add the **`interpret=True` CPU oracle**
  (item 3) as the fast pre-device gate.
- **Orthogonal, any time:** the **launch-plan replay** (item 9) for host-bound
  decode — independent of the kernel-authoring path, and possibly the biggest
  single latency win for bs=1 serving.

**Unchanged discipline:** Phase 1 gates everything (no measurable win without the
injection hook); on-device is the only truth; kernel *source* stays external.
