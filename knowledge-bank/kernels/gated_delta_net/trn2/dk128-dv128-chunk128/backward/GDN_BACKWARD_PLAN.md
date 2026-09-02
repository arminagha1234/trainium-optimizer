# GDN chunked backward — build plan (foundation + recipe)

Goal: NKI backward for the GatedDeltaNet chunked forward, returning dq,dk,dv,dg,dbeta,
validated per-gradient on-device against the autograd oracle. Makes GDN fully
trainable (customer_armin need). The one big remaining survey gap.

## Status
- [x] **Autograd oracle** (`gdn_bwd_oracle.py`) — differentiable torch chunk forward,
      doubling-based T-matrix (matches the NKI fwd algorithm), autograd → dq,dk,dv,dg,
      dbeta. Forward matches the recurrence oracle to 6.9e-17; gradients sane. This is
      the gradcheck reference the NKI backward validates against. VALIDATED on trn2 box.
- [ ] NKI backward kernel — build incrementally, validate each grad vs the oracle.

## Recipe (from the mamba3 BACKWARD_KERNEL_DESIGN + fla chunk_gdr_bwd + the working
## dS recurrence in deltanet_fused_chunked_bwd_batched.py:1124-1163)
Per chunk, REVERSE loop c = NC-1 … 0 (strategy-B recompute forward intermediates):
  1. recompute: gc=cumsum(g), decay, A, T=(I+A)(I+A^2)… (doubling), u=T@v_beta,
     w=T@(k_beta·exp(gc)), and the forward state trajectory S[c] (forward re-pass).
  2. dv_new = attn_intra^T @ dO_c + k_cumdecay @ dS         (attn_intra = (q@k^T)·decay·tril)
  3. dv = (T^T @ dv_new) · beta          (WY-transform backward — a 2nd transposed solve)
  4. d_attn_intra = dO_c @ v_new^T ;  dQK = d_attn_intra · decay · tril
  5. dq = dQK @ k + (dO_c·exp(gc)) @ S^T           (intra + state paths)
  6. dk = dQK^T @ q + (state-path term via w/dS)
  7. dbeta = row-reduce of (dk_beta·k + dv_beta·v)
  8. dg  = reverse-cumsum of the decay-path contributions (flip→cumsum→flip)
  9. dS reverse recurrence: dS = exp(g_last)·dS + (q·exp(gc))^T @ dO_c − w^T @ dv_new
NKI idioms (proven this session): recompute-in-kernel; nl.matmul(x,y)=x@y (fused
transpose); reductions nl.sum keepdims; dma_copy gradient columns to HBM slices;
reverse-cumsum = flip (nc_transpose or reverse slice) → tensor_tensor_scan → flip;
fp32 gate path; bounded exp(g_i−g_j). Doubling stays STABLE because k is l2-normed
(A entries ≤1) — the un-normalized case overflows (why the oracle preprocesses).

## Build order (incremental, validate each on-device vs oracle)
1. dv (needs u,w,dv_new,T^T) — validate dv cos vs oracle.
2. dq (intra dQK@k + state (dO·exp gc)@S^T) — the dS recurrence + forward S re-pass.
3. dk, dbeta.  4. dg (reverse-cumsum) — historically the stubbed piece.
Each is a separate on-device compile+validate cycle (minutes each) → multi-step build.
