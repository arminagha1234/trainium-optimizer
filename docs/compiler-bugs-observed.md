# neuronx-cc bugs observed on real runs

Compiler *bugs*, kept apart from unsupported-op rejections. The difference decides
what you do next: an unsupported op is a graph to rewrite, an `INTERNAL_ERROR` is a
compiler crash on a graph it should have accepted, and no amount of config search
will route around it.

`native_pytorch._ERR_SIGNATURES` classifies these separately, so the ledger says
"escalate" rather than leaving the search to probe configs around a fixed point.

---

## NCC_IBCG901 — BIRCodeGenLoop assertion, Qwen3.5 GatedDeltaNet

**Status:** open, blocks `torch.compile` on Qwen3.5-35B-A3B
**Owner:** kernel R&D lane (scan/GDN)
**First seen:** 2026-08-29, trn2.48xlarge, workload `T_d5cd4802-3c0b-43cd-aa12-d0817cb3b382`

```
Qwen3_5MoeGatedDeltaNet[linear_attn][0]_select.326 [INTERNAL_ERROR] [NCC_IBCG901]
BIRCodeGenLoop assertion err
```

`neuronx-cc` exits 70 while compiling the **first** GatedDeltaNet layer's `select`
op. Reproduce with `compile_mode=compile-default` (i.e. `torch.compile`) on
`Qwen/Qwen3.5-35B-A3B`; `compile_mode=eager` compiles and runs fine, which is how
the baseline was established at all.

What this costs: the model runs only in eager. Since bs=1 decode on this stack is
host-bound, eager is not catastrophic, but it removes the whole
`torch.compile`-based half of the search space for every Qwen3.5 MoE model, and
those are the models the 48xl lane exists to optimise.

Notes for whoever picks it up:
- It is layer 0 of the *linear-attention* block, not the MoE block. The 40 layers
  split 10 `full_attention` / 30 `linear_attention`, and only the GDN path fails.
- `select` is the gather/index step of the delta rule. That is the same region the
  independent read of the compiler landscape already flagged as compiler-weak, so
  this is corroboration rather than a surprise: a hand-written scan kernel would
  both bypass the crash and be the higher-value fix.
- The `attn_implementation=sdpa` candidate on the same model did not crash but blew
  past a 10800s compile wall. Two different symptoms, same suspicion: the GDN graph
  is hard on this compiler.

Escalation still to do: reduce to a minimal repro (a bare `Qwen3_5MoeGatedDeltaNet`
under `torch.compile`, no MoE, no 40 layers) and file upstream with the NEFF. The
tiny-config harness in `test_qwen38_tp_geometry.py` is a reasonable starting point --
it already builds a real hybrid Qwen3.5-MoE at ~256 hidden.

---

## Recording a new one

Add a section with: the verbatim error, the workload id, the exact config that
triggers it, the config that avoids it, what capability it costs, and whether a
signature was added to `_ERR_SIGNATURES`. An entry without a workload id and a
verbatim error is not a bug report, it is a rumour.
