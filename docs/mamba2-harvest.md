# Mamba-2 on Trainium — harvested via the official SSD kernel

**Result (2026-08-24, on-device):** Mamba-2 — an SSM the neuronx-cc compiler can't lower — **runs end-to-end on Trainium** by harvesting the official AWS **SSD kernel** (`aws-neuron/nki-library` → `experimental/scan/ssd.py`). The framework's "harvest before invent" path, proven.

## Validation (trn2.3xlarge, neuronx-cc 2.27.5334, islpy 2026.1)
- **Official SSD kernel vs `ssd_torch.py` reference:** 4/4 shapes PASS, fp32-exact (max_err ~5e-8–2e-7, cosine 1.000000 on both `y` and `final_state`). Both dispatch paths exercised: head-outer (nheads<8) and chunk-outer (nheads≥8, `ssd_block`).
- **Tiny 2-layer Mamba-2 end-to-end** (d_model=128, nheads=4, headdim=64, dstate=64, seqlen=512; in_proj → causal depthwise conv1d+silu → SSD scan (kernel) → gated RMSNorm → out_proj): logits **max_err 9.5e-7, cosine 1.0, 100% argmax-token agreement** vs a CPU torch reference.
- Kernel imported the standalone `nki` package and compiled **unmodified** via the numpy-standalone path (`NEURON_PLATFORM_TARGET_OVERRIDE=trn2`) — no shim/translation needed.

## Kernel entry + constraints (for registry integration)
- `ssd(x, dt, A, B, C, chunk_size=128, D=None, initial_state=None, causal_mask=None) -> (y, final_state)` (`nkilib.experimental.scan.ssd`, `@nki.jit`)
- Layouts: `x[B,H,L,P]`, `dt[B,H,L]`, `A[H]` (negative), `B/C[B,L,N]`, `D[H]`, `initial_state[B,H,N,P]`; returns `y[B,H,L,P]`, `final_state[B,H,N,P]` (fp32).
- Constraints: `chunk_size≤128`, `dstate≤128`, `seqlen % chunk_size == 0`, **ngroups=1 only** (`dstate>256` / `ngroups>1` are kernel TODOs → block Falcon-H1 / Zamba2), `causal_mask = np.tril(ones((Q,Q)))`.
- Dispatch: nheads≥8 → chunk-outer (`ssd_block`); nheads<8 → head-outer (`ssd_head_outer`).

## Registry integration (next)
Register like FlashAttention: primitive `mamba2_ssd` / `ssm_scan` / `linear_attention` → the SSD kernel via `$TRN_OPT_KERNEL_DIR`, using the `mamba2_ssd_scan` adapter (maps model-native `[B,L,H,P]`/`[B,L,H]` ↔ kernel layout). A clean throughput/speedup number needs the kernel bound as a persistent custom-op (the standalone path is host-bound, not device-bound).

See [`docs/kernel-sources.md`](./kernel-sources.md) for the harvest source list.
