# Third-Party Notices

This project borrows optimization *patterns* and, where noted, *code* from the
projects below. Per the licensing decision recorded in `open-questions.md` Q4,
direct code borrowing is permitted from Apache-2.0 / BSD sources with the
attribution recorded here and a per-file provenance header at the borrow site.

As of this release the framework is the harness and design; kernel borrows are
tracked here as they land. Each borrowed kernel also carries a provenance
header in its source file and a `reference_translation` (or `nki_kernel` with
`origin: borrowed`) entry in the knowledge bank recording repo + commit.

## Sources consulted / borrowed from

| Project | License | How used |
|---------|---------|----------|
| [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | Paged attention, KV management, continuous batching patterns |
| [SGLang](https://github.com/sgl-project/sglang) | Apache-2.0 | Radix cache, batching heuristics |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | BSD-3-Clause | Tiling + online softmax patterns |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Apache-2.0 (+ NVIDIA terms) | Fused-kernel patterns — **compliance review required before code borrow** |
| [AWS Neuron nki-library](https://github.com/aws-neuron/nki-library) | see repo | Harvested production NKI kernels (Stage 0.5) |
| [jburtoft NKI kernels](https://huggingface.co/jburtoft) / [NeuronStuff](https://github.com/jimburtoft/NeuronStuff) | check per-artifact | Published NKI kernels (DeltaNet/KDA/Mamba, assorted ops) + model ports; harvest/borrow candidates |
| [neuronx-distributed-inference `contrib/`](https://github.com/aws-neuron/neuronx-distributed-inference) | see repo | Per-architecture worked port recipes (config/sharding/kernels); mined from `contrib/` and PRs |
| [Autocomp](https://github.com/ucb-bar/autocomp) | see repo | Search architecture (beam + plan/implement); no code copied |
| internal-prior-optimization-run | private | Optimization techniques (Local-Q, Context Parallel, Local-MoE); trajectory-report format |
| [KevGomes1403/nki-moe-megakernel](https://github.com/KevGomes1403/nki-moe-megakernel) | Apache-2.0 | **Code borrowed** — fused Qwen3-MoE NKI megakernel, vendored as a Stage-3 BORROW candidate (`implementation/src/kernels/moe_fused/`) |

## Per-borrow log

_(Appended as kernels are actually borrowed. Format:)_

```
### <kernel name>
- Source: <repo> @ <commit>
- License: <license>
- Taken: <what pattern/code>
- Changes: <what we modified for Neuron>
- Site: <path/to/file.py>
- Bank lesson: <lesson_id>
```

### fused MoE megakernel (Qwen3-30B-A3B)
- Source: https://github.com/KevGomes1403/nki-moe-megakernel @ 5879c39
- License: Apache-2.0 (copy at `implementation/src/kernels/moe_fused/LICENSE`)
- Taken: `moe_fused_nki.py`, `qwen_with_megakernel.py`,
  `nki_kernels/moe/components/{routed_experts_nki,moe_layer}.py`,
  `nki_kernels/moe/vendored/router_topk.py` — the fused MoE decode (TKG)
  subkernel + its self-contained expert/router pieces and the model-integration
  reference.
- Changes: NONE to kernel bodies. Prepended SPDX/attribution headers to the two
  files that upstream carried under the repo-level LICENSE only. Added a
  framework `adapter.py` (new file) that offers + gates the swap.
- Site: `implementation/src/kernels/moe_fused/` (see its `NOTICE` + `README.md`)
- Bank lesson: `moe-fused-nki-megakernel-a3b-tp4`
- On-device status: code-complete + mock-tested; NOT yet re-validated on device
  (external `nki-library` dep + NxDI/XLA decode stack + A3B/TP4-only dims —
  see the vendor `README.md` gap analysis).

## A note on model licenses

Running models for the leaderboard is separate from borrowing code. Some model
weights carry restrictive licenses (e.g. territory clauses). The discovery
job's license filter handles that; it is not covered by this file.
