# Qwen/Qwen3.5-4B — trn2.48xlarge

**Peak: 20,470 tok/s** at `TP=16, torch.compile(neuron), bf16, batch=1` — **21.795× over the eager baseline** (939 tok/s), verified on real Trainium hardware (`native-pytorch-beta3`).

## Correctness
Top-1 token agreement vs the eager baseline: **16/16 (100%)**. SoL integrity check: no violation (14.2% MFU, well under the physical ceiling). The GatedDeltaNet now compiles at TP=16 via the matmul-only chunk inverse (`TRN_OPT_GDN_MATMUL_INV=1`, PR #176) plus the `qwen3_next_rewrites` — replacing the strided forward-substitution loop that made `neuronx-cc` fail (`NCC_IBCG901`/`NCC_IINAR001`) once the layer is tensor-parallel sharded.

## Config
| field | value |
|:--|:--|
| tp_degree | 16 |
| compile_mode | compile-default (`torch.compile(backend="neuron")`) |
| weights_dtype | bf16 |
| batch / input_len | 1 / 1024 |
| instance | trn2.48xlarge |

## Reproduce
See [`reproduce.sh`](./reproduce.sh). Toolchain at publish time: torch 2.12.1, torch_neuronx 2.12.3.0.1636, neuronx-cc 2.27.2878.0, nki 0.6.0.
