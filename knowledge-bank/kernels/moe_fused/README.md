# Fused MoE megakernel — Stage-3 BORROW candidate (MoE family)

Vendored from **[KevGomes1403/nki-moe-megakernel](https://github.com/KevGomes1403/nki-moe-megakernel)** @ `5879c39` (Apache-2.0). See [`NOTICE`](./NOTICE) and [`LICENSE`](./LICENSE) for attribution.

This wires the fused Qwen3-MoE NKI megakernel into the optimizer as a **Stage-3 (BORROW)** candidate for the MoE family: when the target model is a Qwen3-MoE, the framework offers *"swap the HF MoE layer forward with the fused NKI megakernel"* as a candidate, applied to the Stage-1 winner and **gated by the existing equivalence check** (drift past tolerance → discarded; nothing is forced on-device).

## How it is wired (all reuse the existing tournament)

1. **Detection** — `native_pytorch._is_moe_model()` reads the HF config (`num_experts` / `num_local_experts` / `*Moe*` arch). `adapter.is_moe_arch()` is the shared predicate.
2. **Candidate offering** — `native_pytorch.moe_kernel_candidates()` returns `[("moe:fused-nki-megakernel", {"moe_kernel": "fused_nki"})]` for a MoE model and `[]` otherwise (graceful no-op for dense LLMs — mirrors the `placement_axes([])` contract).
3. **Evaluation** — `orchestrator.run_deep_stages()` evaluates each offered candidate as a `Stage.BORROW` / `Origin.BORROWED` / `Layer.KERNEL` row through the **same `_evaluate()`** path as every other candidate: compile → **equivalence gate** → HBM/measurement guardrails → keep-if-beats-incumbent.
4. **Application** — the worker (`neuron_worker.py --moe-kernel fused_nki`) calls `adapter.swap_moe_forward()`, which runs the precondition gauntlet and swaps or cleanly falls back.
5. **Memory** — a `nki_kernel` / `borrowed` bank lesson (`seed_bank.py`) records the win for future MoE models.

## Interface of the vendored kernel

`moe_fused_nki.run(inp, gamma, router_w, gate_up_w, down_w) -> output[T, H]` — a decode-time (TKG) fused MoE block (RMSNorm → router top-K → selective gate/up/down expert GEMMs → weighted sum → AllReduce), HBM-in / HBM-out.

## Honest on-device status (the gap)

**Code-complete + unit-tested (mock backend). NOT yet validated on-device.** The vendored kernel does not drop straight into the framework's native-PyTorch prefill measurement path; the mismatches are real and documented rather than papered over:

| Dimension | Framework native-PyTorch backend | Vendored kernel |
|---|---|---|
| Stack | plain HF `AutoModelForCausalLM` + DTensor TP | NxDI (`neuronx_distributed_inference`) + `torch_xla` trace |
| Phase | **prefill** throughput (input_len≈1024) | **decode** (TKG, T = batch) |
| Dims | any Qwen3-MoE | hardcoded **Qwen3-30B-A3B @ TP=4** (hidden=2048, experts=128, top_k=8, moe_intermediate=768) |
| Runtime dep | Neuron DLC (`nki`) | additionally the **private `nki-library`** submodule (`nkilib.core.*`) |

Because of this, `adapter.swap_moe_forward()` runs a full precondition check and **self-skips to eager** on every model except an exact A3B/TP4 deployment with `nkilib` present. On tiny models (e.g. `katuni4ka/tiny-random-qwen3moe`) it skips on the dims mismatch. Even on A3B/TP4 the remaining **weight-repack + XLA trace bridge** from HF's `Qwen3MoeSparseMoeBlock` to `moe_fused_nki.run` is not built — it is the documented next step. The adapter reports this honestly (`execution-bridge-pending`) rather than fabricating a swap or a result.

### Validation status
- ✅ Code-complete: kernel vendored + attributed; candidate offered for MoE, not for dense; equivalence-gated through the real tournament; bank lesson registered.
- ✅ Unit-tested against the mock backend (`test_moe_borrow.py`).
- ⬜ tiny-random-qwen3moe on-device: **deferred** — the kernel cannot run on tiny dims (hardcoded A3B/TP4); it self-skips (verifiable) but there is no fused-kernel path to equivalence-check on that model.
- ⬜ Qwen3-30B-A3B @ TP=4 on-device: **deferred next step** — needs a free trn2 box (TP=4, ~20 min compile), the `nki-library` package, and the HF→kernel weight-repack + XLA trace bridge.
