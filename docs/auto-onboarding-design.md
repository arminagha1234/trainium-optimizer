# Auto-architecture onboarding — design (scope-first, for review)

Status: **DESIGN — not built.** Scopes what it means for the optimizer to take on
a model whose architecture it has *never seen*, without a human writing an
adapter first. Read alongside `optimization-stages.md`, the pre-flight gate
(PR #10), and the manual onboardings this automates (Gemma-4-12B, Qwen3.8-27B).

---

## 1. The gap (why "any new model" isn't true yet)

Today the loop optimizes any model of a **known family** end-to-end (add it to
`models_queue.txt` → baseline → config search → kernels → 13–28× verified). But
for a **new architecture** it does one of two things:
- **Skips it** (pre-flight gate) if it matches a known-doomed signature
  (linear-attn/GatedDeltaNet ISA-fail), or
- **Fails at baseline** (`FAIL_NO_BASELINE`) because there's no family adapter
  telling the backend how to shard it (TP/vocab plan) or handle its quirks.

Onboarding a new family — the TP plan, head layout, RoPE/norm quirks, probe
shape — is **manual today** (that's exactly the hand-work we did for
Gemma-4-12B's per-layer-config parse + single-shot prefill + BOS, and Qwen3.8's
GQA-4/vocab-parallel adapter). That manual step is the wall between "optimizes
known models" and "optimizes *any* model."

**Key realization:** most "new models" are **variations of known architectures**
(another Llama/Qwen/Mistral-shaped dense transformer, or a known-style MoE).
Those don't need a new kernel — they need the framework to *recognize the shape
and reuse the right adapter*. That's the common case, and it's automatable.
Genuinely novel primitives (a new attention math) are the rare case and still
need Stage-4/human work — the design is honest about that boundary.

---

## 2. What "auto-onboard" means

> Given only a HuggingFace model id, produce a **working, equivalence-verified
> baseline** (a `ModelSpec` + a resolved family adapter: parallelism plan +
> quirk handling + probe shape) with **no human in the loop** — so the existing
> Stage 0→6 loop can then optimize it like any other model.

Auto-onboarding produces the **adapter** (how to load + shard + run it
correctly), NOT new kernels. If a model needs a novel kernel, onboarding's job
is to get a correct (if slow) baseline and hand a precise TODO to Stage 4 /
the human — never to fake a result.

---

## 3. The approach — three tiers, cheapest first

### Tier 0 — config-driven family mapping (covers the common case)
Read the model's `config.json` (cheap, no weights) and extract a **structural
fingerprint**: `num_layers`, `hidden`, `num_attention_heads`,
`num_key_value_heads`, `head_dim`, `vocab_size`, `rope_theta`, `attention
type`, `is_moe` (+ `num_experts`/`top_k`), norm type, activation. Match that
fingerprint to a **known family adapter** by structure, not by name — a new
`SomethingForCausalLM` that is Llama-shaped (RMSNorm + GQA + RoPE + SwiGLU)
maps to the dense-causal-LM adapter and just works. This alone covers the
majority of new model releases.

### Tier 1 — adapter synthesis for near-misses
If the fingerprint doesn't match a known family but is a **parameterizable
variant** (a new GQA ratio, a new head_dim, a vocab size that needs
vocab-parallel to fit), **generate the parallelism plan from the config**: pick
`tp_degree` from `num_kv_heads` and core count, choose colwise/rowwise shards
for q/k/v/o + MLP, add vocab-parallel if the embedding is large, set the probe
shape from `max_position_embeddings`. This is the `qwen38_tp.py` adapter pattern
**parameterized by the fingerprint** instead of hand-written per family.

### Tier 2 — attempt-and-diagnose (turn "skip" into an actionable TODO)
For a genuinely novel op (linear-attn, new MoE routing, a new fused primitive),
**attempt a minimal baseline and, on failure, DIAGNOSE precisely** — capture the
exact compiler/runtime error (e.g. the `TensorScalarAffineSelect` ISA-fail on
GatedDeltaNet), map it to the offending op, and record a **structured
onboarding lesson**: `arch=X needs {kernel|adapter} for op=Y, blocked by=<error>`.
That converts today's silent skip into a ranked work-list that feeds Stage 4
(invent) or a human — the model is *queued for onboarding*, not dropped.

---

## 4. Gates (reuse the existing discipline — nothing new to trust)

- **Equivalence is still a hard gate.** An auto-onboarded baseline must pass
  top-1-token match vs a reference (HF CPU/GPU) before any of its optimization
  results count. A fast-but-wrong auto-onboard is a bug, not a win.
- **HBM ≤ 85%, 30-min compile timeout** (existing guardrails) apply to the
  onboarding baseline compile.
- **Honest status**, never a fake baseline: Tier-0/1 success → the model enters
  the normal loop; Tier-2 → recorded as `needs-onboarding` with the diagnosis,
  not as a spurious 0.

---

## 5. Where it plugs in

- **Extends the pre-flight gate (PR #10):** today it decides skip / run. Add a
  branch — *attempt auto-onboard* — that runs Tier 0→1→2 before giving up. A
  known-doomed signature still skips fast; an *unknown* one gets an onboarding
  attempt instead of a `FAIL_NO_BASELINE`.
- **`ModelSpec` / family resolution:** onboarding emits the `ModelSpec` (family,
  param_count, num_kv_heads, probe_shape) the loop already consumes.
- **Backend adapter:** Tier-1 synthesis produces the TP/vocab plan the
  `native_pytorch` backend needs.
- **Bank:** onboarding outcomes are lessons too — a successful fingerprint→family
  mapping is reusable (the next same-shape model onboards instantly); a Tier-2
  diagnosis is an anti-pattern + a work-list item.

---

## 6. Honest limits

- Auto-onboarding delivers the **adapter** (parallelism + config + quirks), not
  **new kernels**. New dense/MoE transformer *variants* (the common case) → fully
  automatable. A genuinely new *primitive* (novel attention) → Tier-2 diagnoses
  it and hands off; it can't be conjured without kernel work (Stage 4 / human).
- Tier-1 synthesis is heuristic — the equivalence gate is what keeps a wrong
  plan from counting. Expect Tier-1 to need iteration on the first few exotic
  shapes (like every adapter did by hand).

---

## 7. Phased build

- **Phase 1 (highest ROI, lowest risk): Tier-0 config-driven family mapping.**
  Fingerprint → known-family adapter. Makes *most* new model releases onboard
  automatically. Validate on a held-out model the queue hasn't seen (e.g. a new
  Llama/Qwen variant) → auto-baseline → loop optimizes it, no human.
- **Phase 2: Tier-1 adapter synthesis** for near-miss shapes (parameterized TP/
  vocab plan from the fingerprint).
- **Phase 3: Tier-2 attempt-and-diagnose** — structured onboarding-TODO lessons
  feeding Stage 4 / the human work-list.

Phase 1 alone moves the needle most: it's the difference between "I maintain a
hand-picked model list" and "I point it at *any* trending model and it just
runs." Discovery (auto-pulling new/trending models into the queue) is the
natural capstone once Phase 1 lands.

---

## 8. Open questions for review

1. **Fingerprint matching threshold** — how close must a config be to a known
   family to auto-map (Tier 0) vs synthesize (Tier 1)? Start strict, loosen with
   evidence.
2. **Reference for the equivalence gate** on a never-seen model — HF CPU eager
   (slow but always available) vs a GPU reference when present?
3. **Phase-1 validation model** — pick a recent dense model *not* in the queue to
   prove auto-onboard end-to-end.
