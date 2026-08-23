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
