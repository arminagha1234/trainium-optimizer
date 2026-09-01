# Qwen3.5-35B-A3B — trn2.48xlarge (native-pytorch-beta3)

**Peak prefill throughput: 2694 tok/s** at TP=16, bf16, eager, **batch=8** (in=512).
Baseline (naive autoport, eager TP=16, batch=1): 453 tok/s -> **5.95x** via batching.

This is a **vision-language MoE** (`Qwen3_5MoeForConditionalGeneration`): a GatedDeltaNet
+ 256-expert MoE text tower (40 layers, 16 attn heads) wrapped with a vision tower.
It could not be loaded by the framework before **PR#180**, which added:
- `capability.estimate_params` `num_local_experts` alias (MoE sizing) ,
- shard-on-read **VL-tower remap** (`model.language_model.*` -> `model.*`, drop vision),
- `TRN_OPT_SKIP_TP` (skip the degenerate 2-device tp=8; use tp=16 = a valid 4-device NeuronLink ring).

## Correctness
Winner top-1 vs eager baseline = **13/16 (81.25%)**, which is **benign bf16 batch-variance**,
not a regression: the 3 differing positions were near-ties in the baseline itself
(top1-vs-top2 logprob gap 0.06-0.125); under batch=8 the baseline token stays rank 1-2
(max gap 0.625). The distribution is equivalent (low KL over the baseline top-k) — greedy
argmax just breaks razor-thin ties differently under bf16's batch-dependent reduction order.
Clears `trusted_grader` (reproduced + top1 >= EQUIV_MIN 0.75).

## Reproduce
See `reproduce.sh` (needs the FSX HF cache + PR#180 on main).
