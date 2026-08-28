"""kernel_rewrites.py — a SYMPTOM-indexed catalog of known compiler-hostile ops
and the cheap graph rewrite that fixes each. Tried BEFORE writing an NKI kernel.

Philosophy (Harvest -> Borrow -> Invent): most "unsupported on Neuron" failures
are ONE hostile op away from compiling. neuronx-cc rejects a specific ISA
instruction — a runtime affine-select emitted by ``torch.tril()``, an int64
top-k, a dynamic slice — and the fix is a pure GRAPH REWRITE, not a kernel. This
catalog maps a compiler ERROR SIGNATURE (and/or the offending op) to that
rewrite, so the invent/repair loop can:

    1. read a real compile-error log,
    2. look up the known fix BY SYMPTOM,
    3. apply/suggest it before spending a compile on an authored kernel.

Indexing by SYMPTOM (the error), not just by intervention, is the ADIAS lesson:
the same fix is reused across every model that trips the same instruction, and a
new failure with a matching signature is fixed instantly the next time.

The seed entries are grounded in REAL, on-device-captured failures (not guesses):
  * ``tril-to-const-mask`` — TensorScalarAffineSelect (s2d2_ts_as_valid_elem_count)
    from ``torch.tril()``/``.triu()`` at Qwen3-Next GatedDeltaNet scale. Compiler-
    only repro on trn2 (neuronx-cc 2.27.5334) went from exit-70 ISA-fail to
    "Compiler status PASS" after this rewrite — NO NKI kernel required.
  * ``int64-topk-to-float-view`` — AwsNeuronTopK rejecting an int64 top-k/sort in
    MoE routing. Route the integer key through a float32 view. (This is a DTYPE
    reject — the top-k algorithm is fine, the int64 KEY is not.)
  * ``topk-sort-to-argmax`` — a DIFFERENT top-k failure: the MoE router's
    ``torch.topk`` lowers to an XLA ``sort`` op, and ``sort`` itself is not a
    supported trn2 ISA op (NCC_EVRF029). The dtype trick does NOT help here; the
    fix is to replace the sort-based top-k with an iterative argmax (k rounds of
    ``.max(dim=-1)`` with iota-compare masking). Grounded: this rewrite is what
    made a FULL Qwen3-Next/Qwen3.5 model compile to a valid NEFF on trn2.

Every rewrite is a hypothesis with EVIDENCE and a confidence; the repair loop
still verifies by re-compiling. A catalog entry is a lead, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rewrite:
    """One known compiler-hostile pattern and the graph rewrite that fixes it."""

    name: str
    summary: str                          # one line: what to do
    # Substrings that identify THIS failure in a neuronx-cc log. Specific on
    # purpose — a generic instruction name shared by several failures is NOT a
    # good signature (it would mis-route). Case-sensitive (compiler tokens are).
    error_signatures: tuple[str, ...]
    hostile_ops: tuple[str, ...]          # aten/HLO op names that emit the pattern
    fix: str                              # the concrete transform (guidance/snippet)
    applies_at: str = "model-graph"       # "model-graph" | "nki-kernel"
    confidence: str = "medium"            # "high" once re-compile-verified in the wild
    evidence: str = ""                    # where it was observed


# The catalog. Ordered most-specific first. Extend as new hostile ops are hit —
# each production failure that a rewrite fixes should land here so the class is
# fixed instantly next time (the compounding the framework is built on).
REWRITES: tuple[Rewrite, ...] = (
    Rewrite(
        name="tril-to-const-mask",
        summary="Replace a runtime .tril()/.triu() with a host-materialized "
                "constant triangular mask + elementwise multiply.",
        error_signatures=(
            "s2d2_ts_as_valid_elem_count",   # the exact ISA assertion that fails
            "aten__tril_select",
            "aten__triu",
        ),
        hostile_ops=("aten::tril", "aten::triu", "tril", "triu"),
        fix=(
            "tri = torch.tril(torch.ones(C, C, device=dev))      # folds to a literal\n"
            "masked = x * tri                                     # was x.tril()\n"
            "strict_lower = x * torch.tril(torch.ones(C, C), -1)  # was masked_fill(triu(0),0)\n"
            "# A runtime .tril lowers to TensorScalarAffineSelect, which fails ISA\n"
            "# validation once num_heads*num_chunks pushes the partition count up.\n"
            "# A constant mask constant-folds to a plain TensorTensor multiply."
        ),
        applies_at="model-graph",
        confidence="high",
        evidence="Qwen3-Next GatedDeltaNet modeling_qwen3_next.py:418 (seq>=512, "
                 "num_v_heads=32); trn2 neuronx-cc 2.27.5334 exit-70 -> PASS after fix.",
    ),
    Rewrite(
        name="int64-topk-to-float-view",
        summary="Route an integer top-k/sort/argsort through a float32 view "
                "(AwsNeuronTopK rejects int64 keys).",
        error_signatures=("AwsNeuronTopK",),
        hostile_ops=("aten::topk", "aten::sort", "aten::argsort", "topk", "sort"),
        fix=(
            "# The reject is dtype, not algorithm: sort/topk keys arrive int64.\n"
            "idx = torch.topk(scores.float(), k).indices   # view keys as fp32\n"
            "# For expert routing, cast the routing ids through float32 before the\n"
            "# grouped topk/sort, then back to long for the gather."
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="HF transformers/integrations/moe.py grouped-experts torch.sort "
                 "on int64 expert_ids -> AwsNeuronTopK reject (MoE routing).",
    ),
    Rewrite(
        name="topk-sort-to-argmax",
        summary="Replace a sort-based MoE-router torch.topk with an iterative "
                "argmax (k rounds of masked .max) — 'sort' is not a trn2 ISA op.",
        error_signatures=(
            "NCC_EVRF029",                     # the exact op-unsupported error code
            "Operation sort is not supported",
            "sort is not supported on trn2",
        ),
        hostile_ops=("aten::sort", "aten::topk", "sort", "topk"),
        fix=(
            "# This is an OP-unsupported reject, not a dtype reject: torch.topk in\n"
            "# the router lowers to an XLA `sort`, and `sort` has no trn2 ISA op.\n"
            "# Casting keys to float32 (int64-topk-to-float-view) does NOT help.\n"
            "# Replace the sort-based top-k with a sort-free iterative argmax:\n"
            "vals = router_probs                       # (tokens, num_experts)\n"
            "idxs = []\n"
            "for _ in range(top_k):                     # k rounds, no sort\n"
            "    m = vals.max(dim=-1, keepdim=True)     # argmax = supported reduce\n"
            "    idxs.append(m.indices)\n"
            "    iota = torch.arange(vals.shape[-1], device=vals.device)\n"
            "    mask = iota.view(1, -1) == m.indices   # mask the winner out\n"
            "    vals = vals.masked_fill(mask, float('-inf'))\n"
            "top_idx = torch.cat(idxs, dim=-1)          # was torch.topk(...).indices\n"
            "# Gather the corresponding probs with the original (unmasked) tensor.\n"
            "# Pure graph rewrite; no NKI kernel. Compiler PASS at full-model scale."
        ),
        applies_at="model-graph",
        confidence="high",
        evidence="Qwen3-Next/Qwen3.5 (GatedDeltaNet-MoE) full-model compiler-only "
                 "compile on trn2 (neuronx-cc 2.27.5334): Qwen3NextTopKRouter.forward "
                 "torch.topk at modeling_qwen3_next.py:772 -> NCC_EVRF029 'sort not "
                 "supported'. Iterative-argmax rewrite -> full model PASS (valid "
                 "~56MB NEFF). This, not .tril, was the load-bearing full-model blocker.",
    ),
    Rewrite(
        name="dense-moe-static-dispatch",
        summary="Replace HF's data-dependent grouped-MoE expert dispatch with a "
                "sort-free static-shape DENSE dispatch (compute every expert, "
                "weight by a scattered gate) — the grouped path is numerically "
                "WRONG on trn2 even after it compiles.",
        # CORRECTNESS symptom, not a compile abort: after the router/tril rewrites
        # the model COMPILES but its logits diverge. Indexed by the offending
        # op/source names so match_ops (graph inspection) and an equivalence-gate
        # failure log ("moe"/"grouped_mm"/"cosine") both route here. These
        # signatures are disjoint from the compile-error entries above.
        error_signatures=(
            "grouped_mm_experts_forward",
            "grouped_mm",
            "Qwen3NextExperts",
            "moe-correctness",
        ),
        hostile_ops=(
            "aten::index_add_", "aten::nonzero", "aten::one_hot",
            "grouped_mm", "_grouped_mm", "aten::histc", "aten::bincount",
        ),
        fix=(
            "# NOT a compile abort — a CORRECTNESS break. HF's Qwen3NextExperts /\n"
            "# moe.py expert path (one_hot/nonzero/where/index_add_, or\n"
            "# sort+histc+grouped_mm) is data-dependent AND numerically wrong on\n"
            "# trn2 (cosine ~0.75). Replace it with a dense, static-shape dispatch:\n"
            "gate_full = torch.zeros(T, E, device=x.device, dtype=w.dtype)\n"
            "gate_full = gate_full.scatter(1, top_k_index, top_k_weights)  # (T,E)\n"
            "out = torch.zeros_like(x)\n"
            "for e in range(E):                       # every expert, every token\n"
            "    g, u = F.linear(x, gate_up_proj[e]).chunk(2, dim=-1)\n"
            "    h = F.linear(act_fn(g) * u, down_proj[e])\n"
            "    out = out + h * gate_full[:, e:e+1]  # 0 for non-selected experts\n"
            "# Same math as top-k routing; no sort/topk/nonzero/grouped_mm; static\n"
            "# shapes (C8 host-dispatch friendly). Exact on CPU (maxdiff 0)."
        ),
        applies_at="model-graph",
        confidence="high",
        evidence="Qwen3-Next/Qwen3.5 (GatedDeltaNet-MoE) tiny arch-proof on trn2 "
                 "(neuronx-cc 2.27.5334, transformers 5.15.0): after the sort-free "
                 "router + tril->const-mask rewrites the model COMPILED but was "
                 "numerically wrong (cosine ~0.75), isolated to the MoE expert path "
                 "(full + linear attention were both correct in isolation, cosine "
                 "0.99998/0.99999). Dense dispatch -> cosine 0.99793 vs CPU-bf16, "
                 "top-1 14/16 (the bf16 noise floor). Wired in "
                 "backends/qwen3_next_rewrites.install_qwen3_next_neuron_rewrites.",
    ),
    Rewrite(
        name="dynamic-slice-to-static-bucket",
        summary="Replace a data-dependent (dynamic) slice length with a static, "
                "bucketed shape padded on the host.",
        error_signatures=("dynamic-update-slice", "DynamicUpdateSlice",
                          "dynamic_slice", "non-constant"),
        hostile_ops=("aten::slice_scatter", "dynamic_slice", "dynamic_update_slice"),
        fix=(
            "# Neuron wants static shapes (C8: host dispatch, single-shape kernel).\n"
            "# Pad the sequence/KV to a fixed BUCKET on the host, run the static\n"
            "# shape, drop the padded tail. (Same contract mamba2_ssd_prefill uses.)\n"
            "# NOTE: dynamic-update-slice OFTEN compiles fine on trn2; only reach\n"
            "# for this when the log actually names it as the reject."
        ),
        applies_at="model-graph",
        confidence="low",
        evidence="General static-shape guardrail; unverified against a specific "
                 "captured reject (kept low-confidence until re-compile-verified).",
    ),
    # --- OFFLINE LINT symptoms (BUG #3) ------------------------------------
    # The entries above route COMPILER errors. These route the OFFLINE static
    # lint's own messages (``invent_kernels.static_lint``), so a lint failure —
    # not just a compile failure — reaches ``match_error`` and the repair loop's
    # "named fix" assist fires for the lint symptom instead of feeding the raw
    # lint string back and stalling. ``applies_at="nki-kernel"``: the fix edits
    # the authored kernel source, not the model graph. Each signature is the
    # STABLE, interpolation-free slice of the exact message ``static_lint``
    # emits; they are disjoint from every compiler-log signature above, so no
    # cross-match with the tril / int64-topk / sort / dynamic-slice entries.
    Rewrite(
        name="lint-arange-to-mgrid",
        summary="Replace nl.arange indexing with nl.mgrid.",
        error_signatures=("nl.arange",),
        hostile_ops=(),
        fix=(
            "# static_lint: 'uses nl.arange (deprecated) — use nl.mgrid'.\n"
            "# Build indices/masks with nl.mgrid, not nl.arange:\n"
            "ix = nl.mgrid[0:128, 0:H]     # was nl.arange(...)\n"
            "rows = ix.p + t * 128         # partition index\n"
            "m = rows < T                  # tail mask"
        ),
        applies_at="nki-kernel",
        confidence="high",
        evidence="invent_kernels.static_lint rule 1 (CLAUDE.md NKI section).",
    ),
    Rewrite(
        name="lint-int-cast-to-float-recip",
        summary="Drop the int() cast in the kernel body; use *1.0/n instead.",
        error_signatures=("int() cast in kernel body",),
        hostile_ops=(),
        fix=(
            "# static_lint: 'uses int() cast in kernel body — beta-3 gotcha'.\n"
            "# The beta-3 eager path rejects integer casts in the body. Use a\n"
            "# float reciprocal instead of an int op:\n"
            "ms = nl.sum(sq, axis=1) * (1.0 / H)   # was int(...) / H"
        ),
        applies_at="nki-kernel",
        confidence="high",
        evidence="invent_kernels.static_lint rule 2 (int cast).",
    ),
    Rewrite(
        name="lint-tile-not-allowed",
        summary="Remove .tile() from the kernel body (beta-3 gotcha).",
        error_signatures=("tile() in kernel body",),
        hostile_ops=(),
        fix=(
            "# static_lint: 'uses tile() in kernel body — beta-3 gotcha, avoid'.\n"
            "# Do not call .tile(...) in the body; tile explicitly with an\n"
            "# affine_range loop over 128-row partitions + nl.mgrid masking."
        ),
        applies_at="nki-kernel",
        confidence="high",
        evidence="invent_kernels.static_lint rule 2 (tile).",
    ),
    Rewrite(
        name="lint-partition-dim-over-128",
        summary="Tile the partition axis to <=128 (partition dim must be 128).",
        error_signatures=("partition dim must be 128",),
        hostile_ops=(),
        fix=(
            "# static_lint: 'partition (first) dim N > 128 — partition dim must\n"
            "# be 128'. Never allocate a first (partition) dim > 128; loop over\n"
            "# 128-row tiles instead:\n"
            "n_tiles = (T + 128 - 1) // 128\n"
            "for t in nl.affine_range(n_tiles):\n"
            "    xs = nl.load(x[t * 128:t * 128 + 128, 0:H], mask=rows < T)"
        ),
        applies_at="nki-kernel",
        confidence="high",
        evidence="invent_kernels.static_lint rule 3 (partition dim).",
    ),
    Rewrite(
        name="lint-per-index-dma-to-multipartition",
        summary="Replace a per-index DMA on the packed axis with one "
                "multi-partition DMA + on-chip transpose.",
        error_signatures=("per-index DMA",),
        hostile_ops=(),
        fix=(
            "# static_lint: 'per-index DMA on packed axis: dma_copy subscripts\n"
            "# [var, ...] inside a loop'. Do ONE multi-partition DMA of the whole\n"
            "# [0:128, ...] tile, then transpose on-chip — never a dma_copy that\n"
            "# indexes the first (partition) axis with the loop variable:\n"
            "buf = nl.load(w[0:128, 0:K])      # one multi-partition DMA\n"
            "wt = nl.transpose(buf)            # on-chip transpose, no per-index DMA"
        ),
        applies_at="nki-kernel",
        confidence="high",
        evidence="invent_kernels.static_lint rule 4 (CLAUDE.md DMA rule).",
    ),
    # --- HARVESTED from AWShtokoyo/vllm-neuron contributed models (Apache-2.0,
    # Copyright Amazon.com; harvested 2026-08-28, see docs/vllm-neuron-harvest.md).
    # These are real, model-team-captured Neuron failures + their fixes across
    # GLM-5.2 / Qwen3.6-GDN / Gemma-4 / Ministral3. confidence="medium": grounded
    # in that repo's on-device work but NOT yet re-verified in OUR loop — the
    # repair loop still re-compiles to confirm. ------------------------------
    Rewrite(
        name="repeat-interleave-to-broadcast",
        summary="Replace torch.repeat_interleave (indirect DGE, OOBMode.ERROR) "
                "with an index-FREE unsqueeze/expand/reshape broadcast.",
        error_signatures=("repeat_interleave", "OOBMode.ERROR", "indirect DGE",
                          "OOB indirect"),
        hostile_ops=("aten::repeat_interleave", "repeat_interleave"),
        fix=(
            "# repeat_interleave lowers to an INDIRECT DMA (gather) that faults\n"
            "# OOBMode.ERROR on Neuron. For the common GQA k/v head expansion the\n"
            "# repeat is STRUCTURED — do it index-free (bit-identical, no gather):\n"
            "# was: k = k.repeat_interleave(n_rep, dim=1)   # [B,Hkv,...] -> [B,Hq,...]\n"
            "k = k[:, :, None, :, :].expand(B, Hkv, n_rep, S, D).reshape(B, Hkv*n_rep, S, D)\n"
            "# LAW (harvested): rewrite an index op to a broadcast/reshape, NEVER a\n"
            "# value clamp — the clamp hides the OOB, the reshape removes it."
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron add-qwen36-moe: repeat_interleave -> "
                 "OOBMode.ERROR indirect DGE; index-free expand/reshape fix.",
    ),
    Rewrite(
        name="gelu-tanh-inline",
        summary="Inline F.gelu(approximate='tanh') as the explicit fp32 tanh "
                "polynomial (Dynamo marks the fused gelu 'skipped').",
        error_signatures=("function marked as skipped", "approximate", "gelu"),
        hostile_ops=("aten::gelu", "F.gelu"),
        fix=(
            "# torch.compile/Dynamo refuses F.gelu(approximate='tanh') ('function\n"
            "# marked as skipped'). Inline the tanh-approx GELU in fp32:\n"
            "g = 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x*x*x)))\n"
            "# (0.7978845608 = sqrt(2/pi)). Compute in fp32, downcast the result."
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron add-gemma-4: GeGLU F.gelu(tanh) Dynamo skip.",
    ),
    Rewrite(
        name="argsort-to-argmax-mask",
        summary="Replace an unlowerable argsort/topk-sort with a sort-free "
                "iterative argmax-on-equality-mask (N/8 passes).",
        error_signatures=("argsort", "sort is not", "unlowerable sort",
                          "aten::argsort"),
        hostile_ops=("aten::argsort", "aten::sort", "argsort"),
        fix=(
            "# argsort/sort do not lower on Neuron. Rank via iterative argmax with\n"
            "# an equality mask (as GLM-5.2 _torch_argsort_unstable does), or for a\n"
            "# MoE router use a sort-free rotational/cascaded topk. Complements the\n"
            "# existing topk-sort-to-argmax router rewrite for the general argsort."
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron add-glm-5-2 functional/argsort_unstable.py "
                 "+ functional/{topk,moe/router}.py sort-free ranking.",
    ),
    Rewrite(
        name="moe-dge-16-alignment",
        summary="Pad the MoE token count to a multiple of 16 before the "
                "affinity-mask indirect DMA (swDGE over-reads to the next mult-16).",
        error_signatures=("DGE out-of-bound", "swdge", "GpSimd", "out-of-bound",
                          "nrta-1006"),
        hostile_ops=("grouped_mm", "moe_tkg", "index_add_"),
        fix=(
            "# The MoE token-generation affinity-mask indirect DMA (moe_tkg) aborts\n"
            "# DGE-out-of-bound when T % 16 != 0 (the GpSimd swDGE over-reads to the\n"
            "# next multiple of 16). Pad T up to a multiple of 16, route the pad\n"
            "# rows to expert 0 with ZERO affinity, run, then slice the pad off:\n"
            "Tp = ((T + 15) // 16) * 16                       # _DGE_ALIGNMENT=16\n"
            "# pad rows -> expert 0, affinity 0; out = moe(...)[:T]\n"
            "# (At MTP verify shape T=bs*(1+gamma), gamma=1: fall back to a kernel-\n"
            "#  free bf16 einsum decode path instead — the indirect DMA over-reads.)"
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron add-glm-5-2: moe_tkg _DGE_ALIGNMENT=16 pad; "
                 "MTP gamma=1 _forward_decode_einsum fallback.",
    ),
    Rewrite(
        name="moe-padding-token-dispatch-oob",
        summary="Mask right-padding tokens out of MoE expert dispatch (they "
                "over-run the dispatch buffer, nrta-1006).",
        error_signatures=("nrta-1006", "dispatch", "expert_mask", "over-run"),
        hostile_ops=("grouped_mm", "one_hot", "index_add_"),
        fix=(
            "# MoE prefill right-pads the batch; ~all pad tokens route to real\n"
            "# experts and over-run the dispatch buffer. Mask them BEFORE the\n"
            "# expert_mask and clamp the scatter index:\n"
            "padding_mask = idx <= positions.argmax()         # pad tokens = False\n"
            "expert_mask = expert_mask & padding_mask[..., None]\n"
            "token_position_to_id = token_position_to_id.clamp(min=-1, max=T-1)"
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron add-qwen36-moe: prefill padding-token "
                 "dispatch OOB; padding_mask + clamp fix. moe_cte skip_weight=True.",
    ),
    Rewrite(
        name="fp8-promotion-to-bf16-cast",
        summary="Insert an explicit .to(bf16) where an fp8 tensor meets a "
                "promotion op (residual add) — 'Float8 promotion not supported'.",
        error_signatures=("Float8 promotion not supported", "XLAFloat8_e4m3",
                          "Float8 promotion", "e4m3"),
        hostile_ops=("aten::add", "aten::view"),
        fix=(
            "# Neuron has no implicit fp8 promotion. Where an fp8 tensor feeds a\n"
            "# promotion op (e.g. the residual add), cast explicitly (fp8->bf16 is a\n"
            "# runtime no-op on the values):\n"
            "h = h_fp8.to(torch.bfloat16) + residual\n"
            "# After a dtype .view to fp8, add .contiguous() to materialize it\n"
            "# ('Expected XLA tensor ... Got XLAFloat8_e4m3')."
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron Ministral3 (Devstral-2) static-FP8 path.",
    ),
    Rewrite(
        name="inplace-scatter-stale-read",
        summary="Gather prior KV/state BEFORE scattering the new step; write the "
                "cache AFTER attending (in-place scatter+gather of one tensor "
                "reads stale values -> period-2 decode oscillation).",
        error_signatures=("stale", "period-2", "oscillat", "decode diverges",
                          "in-place scatter"),
        hostile_ops=("aten::scatter_", "aten::index_copy_", "slice_scatter"),
        fix=(
            "# Scattering the new token into a cache and then gathering the SAME\n"
            "# tensor in the same step reads pre-write (stale) values -> outputs\n"
            "# oscillate with period 2 across decode steps. Order it:\n"
            "prior = gather(cache, slots)      # 1. read prior\n"
            "ctx   = concat(prior, new_kv)     # 2. attend against a LOCAL tensor\n"
            "out   = attend(q, ctx)\n"
            "scatter_(cache, slots, new_kv)    # 3. write cache LAST\n"
            "# Also: anchor compile-folded slot indices into the XLA graph via\n"
            "# `idx + 0*anchor_scalar`, and bind state as a graph input with an\n"
            "# in-place copy_ (attribute reassignment demotes it to a CPU constant)."
        ),
        applies_at="model-graph",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron add-qwen36-moe (GDN paged state) + "
                 "add-glm-5-2 (aliased MLA latent): scatter-before-gather ordering.",
    ),
    Rewrite(
        name="fp8-e4m3-240-saturate",
        summary="Byte-saturate OCP e4m3 codes >240 onto the trn2 ±240 grid; do "
                "NOT rescale by 240/448 (that moves every element off the grid).",
        error_signatures=("e4m3", "448", "240", "inf", "NaN from fp8"),
        hostile_ops=(),
        fix=(
            "# trn2 legacy e4m3 max is 240; OCP e4m3 max is 448. Out-of-range OCP\n"
            "# codes read back inf -> NaN. SATURATE the byte onto the 240 grid\n"
            "# (exact for in-range codes), never multiply by 240/448:\n"
            "#   for byte b: if (b & 0x7F) >= 0x78: b = (b & 0x80) | 0x77\n"
            "# i.e. clamp magnitude to the largest representable <=240, keep sign.\n"
            "# Rescaling by 240/448 shifts EVERY value off the fp8 grid (~4 orders\n"
            "# worse); byte-saturation touches only the handful of oob codes."
        ),
        applies_at="nki-kernel",
        confidence="medium",
        evidence="AWShtokoyo/vllm-neuron Ministral3 + add-qwen36-moe: e4m3 240-vs-448 "
                 "range mismatch; byte-saturation fix.",
    ),
)


def match_error(error_log: str) -> list[Rewrite]:
    """Rewrites whose error signature appears in a neuronx-cc log. Most-specific
    (catalog-order) first. Signatures are case-sensitive on purpose — compiler
    instruction/assertion tokens are exact — so a generic word never mis-routes."""
    if not error_log:
        return []
    return [r for r in REWRITES
            if any(sig in error_log for sig in r.error_signatures)]


def match_ops(op_names) -> list[Rewrite]:
    """Rewrites whose hostile op appears in a set/list of op names (from a graph
    inspection). A cheaper, pre-compile lead than a full error log."""
    names = list(op_names or [])
    hits: list[Rewrite] = []
    for r in REWRITES:
        if any(op == n or op in n for op in r.hostile_ops for n in names):
            hits.append(r)
    return hits


def describe(rewrites: list[Rewrite]) -> str:
    """A compact, actionable summary for a ledger row / lesson / author feedback."""
    if not rewrites:
        return "no known rewrite matched this failure"
    return "; ".join(f"{r.name}: {r.summary}" for r in rewrites)
