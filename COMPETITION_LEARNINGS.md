# Learnings borrowed from NeurIPS-Trainium-Competition

Source: `code.amazon.com/packages/NeurIPS-Trainium-Competition` (organizer design
for a 30-min single-Trn2 LLM-pretraining competition scored on `val_bpb`). Their
`autoresearch/program.md` is the reference loop this framework descends from; the
`_private/` scorer + reviewer are a rigorous anti-cheating evaluation layer.

## Already borrowed + shipped this session
- **Trusted grader** (`trusted_grader.py`) ← their `score.py` "recompute the metric
  with organizer-owned code, never trust the participant's self-reported number."
  We re-measure the winner independently; it must reproduce (±10%) + re-pass
  equivalence before it's marked `verified`.
- **Reviewer gate** (`reviewer.py`) ← their `submission_review_prompt.md` +
  `prepare.check_submission`. "Review, then execute": validate every candidate
  before the compile; cc_flags allowlist + shell-injection guard.
- **Git-as-state-machine, keep/discard, NEVER-STOP, results.tsv, timeout-kill,
  simplicity criterion** — confirmed our core loop matches their battle-tested one.

## The reviewer taxonomy (their prompt) — the spec for `review_kernel_source`
Default-deny: PASS only if you can *positively account for every construct as
honest*. Any CRITICAL ⇒ REJECT; unresolved HIGH/obfuscation ⇒ NEEDS_MANUAL_REVIEW.
Extraction pass (enumerate before judging): **E1** eval-mode detection signals
(`__name__`,`sys.modules`,`os.environ`), **E2** input shape/content conditionals,
**E3** hashes/lookup tables, **E4** mutation of trusted state (`prepare.X=`),
**E5** checkpoint load (pickle RCE → require `weights_only=True`), **E6** grad/optim
in eval (test-time adaptation), **E7** lifecycle side-effects (`atexit`,signals,
threads), **E8** filesystem touches (`glob/walk/scandir`), **E9** dynamic-exec
(`eval/exec/__import__/compile`), **E10** logit transforms keyed on input, **E11**
collective misuse, **E12** loss honesty. Cheat categories A–L: eval-mode switching,
logit dishonesty (memorized/lookup logits), causality leakage, val peeking, network
egress, subprocess, trusted-state tampering, budget/watchdog evasion, writing
outside out-dir, distributed abuse, obfuscation, kernels. → codified as the
deny-list in `review_kernel_source`; the deep pass is an LLM reviewer (future).

## Not yet borrowed — worth doing next
- **Variance-reduced metric**: they score over 5 i.i.d. shards (~√5× less variance)
  at a fixed token budget (not a full pass). Our keep/discard is single-probe —
  aggregate a few probes and gate on the reduced number. (This was improvement #3.)
- **Capacity contract**: their scorer pins lnc/rank count and hard-fails a model
  that doesn't fit the eval core (deliberate, not ambiguous). We now have real HBM;
  formalize "fits-or-fails at serve config."
- **Stateless-forward eval contract + causality check**: `load_for_eval` must return
  a consolidated, single-device, replicated model with a pure stateless forward
  (logits a function of tokens only — no collectives/hooks/state). Strong invariant
  to assert for the stateful DeltaNet (qwen3.8) sharding, beyond top-1 token match.
- **NKI kernel template** (for Stage 4): their `train.py` ships a fused `relu(x)**2`
  NKI kernel showing the 3 essentials — (1) HBM→SBUF load, `nisa.*` compute, store;
  (2) tiling over the 128-row partition limit with eager fallback; (3) autograd via
  `torch.autograd.Function` (raw NKI isn't differentiable). Use as the starting
  point when wiring real Stage-4 invention. (Their competition also allows/ships
  custom NKI kernels as first-class — validates our borrow-then-invent kernel path.)
- **Training-side techniques** (Muon for matrices + AdamW for embed/head/scalars,
  QK-norm, logit softcap, ReLU² MLP, stochastic rounding) — their baseline; not
  directly applicable to our *inference* optimizer, but the "borrow proven recipes"
  spirit is the same.

## Operational lesson (ours, not theirs)
Box-fill (#5) once launched ~60 concurrent replicas and starved `sshd` → the 48xl
went unreachable. Fixed: cap concurrent replicas (≤12), `nice` them, stagger
launches, and always leave ≥2 cores for the OS. A benchmark must never take the box
offline.
