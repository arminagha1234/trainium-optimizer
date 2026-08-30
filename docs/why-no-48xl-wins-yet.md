# Why the 48xl has verified baselines but no published wins yet

Qwen3.8-27B is the first model to complete the whole pipeline on a trn2.48xlarge and
pass the trusted grader:

```
trusted grader: verified (claimed=343 remeasured=338 drift=1.5% equivalence=ok)
box throughput: 2,097 tok/s (7/7 replicas, tp8, compile=0)
```

That also **replaces the retracted 340 tok/s figure** for this model (see
`trn2-48xl-kaizen-sweep-2026-08-28.md`, whose GatedDeltaNet numbers were all measured
before #96's device fast-path landed). 343 tok/s per replica / 2,097 tok/s per box is
measured on current code and independently re-measured by the grader.

It is **not on the leaderboard**, and that is correct: `speedup == 1.0`. The search
found nothing faster than the eager baseline, and `publish_deliverables` requires
`speedup > 1.0`. A verified baseline is a datapoint, not a win.

## What the 25 candidates actually did

| Candidate | tok/s | Outcome |
|:--|--:|:--|
| baseline (tp=8, bf16, eager) | **343.4** | kept, grader-verified |
| `attn_implementation=sdpa` | **346.9** | faster, discarded — correctness 81% + OOM at 95% HBM |
| `tp_degree=4` | **351.4** | faster, discarded — OOM at 95% HBM |
| `cp_degree=2/4/8` | 330–335 | slower |
| `compile_mode=compile-default` | 0 | **neuronx-cc NCC_IBCG901 (issue #134)** |
| `batch=8`, `batch=32` | 0 | OOM / HBM pressure |
| `weights_dtype=fp32`, `tp=1`, `tp=2` | 0 | OOM / HBM pressure |
| `tp=16/32/64` | 0 | `invalid_tp` — 24 heads not divisible |
| `cc:optlevel1/2/2+nocast` | 0 | neuronx-cc failures |
| `cc:optlevel3`, `optlevel3+transformer+nocast` | 0 | hit the 2700 s per-candidate wall |
| profile loop, rounds 1–2 | 343.4 | `compute_bound, hot=attention_prefill(45%)` |

## The causal chain

**Two candidates were genuinely faster and both were rejected for memory, not speed.**
346.9 and 351.4 tok/s are real improvements over 343.4; sdpa also dropped to 81%
top-1 correctness, and both OOM'd at 95% HBM. The model sits right at the edge of a
24 GB core at tp=8 (6.9 GB of weights per rank), so every lever that buys throughput
pushes it over.

The lever that produced the big wins on the trn2.3xlarge — `compile_mode=compile-default`,
i.e. `torch.compile` — returns 0 here because of the GatedDeltaNet compiler crash
(#134). So for GDN-hybrid models the 48xl is currently missing its main optimisation
axis. That makes #134 the critical path for this lane, not a side quest.

## The specific thing to try next, and why

**tp was under-used.** Qwen3.8-27B has 24 attention heads. Until #140/#142 the search
only proposed powers of two, so it stopped at tp=8 and the three candidates above
tp=8 were all rejected as `invalid_tp` (16, 32 and 64 do not divide 24). Every
divisor is now considered, which makes **tp=12 and tp=24 reachable for the first
time**:

| tp | weights/rank | headroom on a 24 GB core |
|--:|--:|:--|
| 8 | 6.9 GB | at the edge; batch=8 OOMs |
| 12 | 4.6 GB | |
| **24** | **2.3 GB** | 3x the room for activations |

The profile says `compute_bound` with 45% in attention prefill, so more parallelism
plus a bigger batch is exactly the indicated direction — and batch is the lever that
was worth 10–28x on the smaller box. The prediction is therefore concrete: **tp=24
with batch=8 or 32 should be the first real 48xl win for this model.** It is being
tested now.

A second, independent blocker was fixed alongside: no multi-rank run was getting
`TORCH_NEURONX_ENABLE_HOST_CC=1` (#145), so every tp=32 candidate died at NRT
execution with `NRT_RESOURCE: Failed to allocate resource` or `Invalid NEFF` despite
using under 8 GB of a 14.4 GB per-rank budget. Any conclusion drawn about tp>8 before
that fix should be treated as unmeasured rather than negative.
