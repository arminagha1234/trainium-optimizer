"""Tests for the symptom-indexed rewrite catalog (kernel_rewrites) and the
iterative compile-repair loop (kernel_repair).

Theme: a compile failure TEACHES the next attempt. The loop is exercised with a
mock author — one that "learns" from feedback (the way the real LLM author will),
one that ignores it, one that keeps failing differently — to prove the loop
converges when it can and stops HONESTLY when it can't (never a fake success).
"""

from __future__ import annotations

from kernel_repair import CompileResult, KernelRepairLoop
from kernel_rewrites import (
    REWRITES,
    describe,
    match_error,
    match_ops,
)


# The real, on-device-captured neuronx-cc failure for Qwen3-Next .tril().
AFFINE_SELECT_LOG = (
    'loc(fused<132406>["aten__tril_select.190"(...)]): error: [unassigned.crzke] '
    "ISA validation failed: TensorScalarAffineSelect; inst failed assertion "
    "check: 's2d2_ts_as_valid_elem_count' [INTERNAL_ERROR] [NCC_IINAR001]"
)
TOPK_LOG = "error: AwsNeuronTopK rejected int64 key tensor in grouped routing"
# The real, on-device-captured neuronx-cc failure for the Qwen3-Next MoE router's
# sort-based torch.topk. A DIFFERENT top-k failure than TOPK_LOG: this is the
# `sort` op being unsupported (NCC_EVRF029), not an int64-dtype reject.
SORT_LOG = (
    'loc(fused<772>["Qwen3NextTopKRouter.forward"("modeling_qwen3_next.py":772)]): '
    "error: [NCC_EVRF029] Operation sort is not supported on trn2. Use supported "
    "equivalent operation like TopK or replace it with an alternate implementation "
    "via NKI. (from torch.topk(router_probs, top_k))"
)
UNKNOWN_LOG = "error: something entirely unfamiliar happened in pass QuxBar"


# -- kernel_rewrites ---------------------------------------------------------

def test_match_error_routes_tril_failure():
    hits = match_error(AFFINE_SELECT_LOG)
    assert [r.name for r in hits] == ["tril-to-const-mask"]
    assert hits[0].applies_at == "model-graph"


def test_match_error_routes_int64_topk():
    hits = match_error(TOPK_LOG)
    assert [r.name for r in hits] == ["int64-topk-to-float-view"]


def test_match_error_no_false_route_on_unknown():
    assert match_error(UNKNOWN_LOG) == []
    assert match_error("") == []


def test_affine_select_does_not_also_match_topk():
    # The generic instruction name is NOT a signature; the tril + topk entries
    # must not both fire on the same log (that would mis-route the fix).
    names = [r.name for r in match_error(AFFINE_SELECT_LOG)]
    assert "int64-topk-to-float-view" not in names


def test_match_error_routes_sort_op_unsupported():
    # The full-model MoE-router blocker: torch.topk -> XLA sort -> NCC_EVRF029.
    # Must route ONLY to the sort-free argmax rewrite, not the int64-dtype one.
    hits = match_error(SORT_LOG)
    assert [r.name for r in hits] == ["topk-sort-to-argmax"]
    assert hits[0].applies_at == "model-graph"
    assert hits[0].confidence == "high"


def test_sort_and_int64_topk_do_not_cross_match():
    # Two distinct top-k failures share the topk/sort ops but have disjoint error
    # signatures. Each log must route to exactly one entry — no cross-firing.
    sort_names = [r.name for r in match_error(SORT_LOG)]
    int64_names = [r.name for r in match_error(TOPK_LOG)]
    assert "int64-topk-to-float-view" not in sort_names
    assert "topk-sort-to-argmax" not in int64_names
    assert sort_names == ["topk-sort-to-argmax"]
    assert int64_names == ["int64-topk-to-float-view"]


def test_match_ops_finds_sort_pre_compile():
    hits = match_ops(["aten::sort"])
    assert any(r.name == "topk-sort-to-argmax" for r in hits)


def test_match_ops_finds_tril_pre_compile():
    hits = match_ops(["aten::mul", "aten::tril", "aten::add"])
    assert any(r.name == "tril-to-const-mask" for r in hits)


def test_describe_is_actionable():
    text = describe(match_error(AFFINE_SELECT_LOG))
    assert "tril-to-const-mask" in text and "constant triangular mask" in text
    assert describe([]) == "no known rewrite matched this failure"


# -- KernelRepairLoop --------------------------------------------------------

class _LearningAuthor:
    """Emits a broken kernel until it sees the tril fix in the feedback trail,
    then emits a good one — the way the real LLM author consumes the error."""

    def __call__(self, trail):
        for fb in trail:
            if any(r.name == "tril-to-const-mask" for r in fb.rewrites):
                return "good-kernel"
        return "broken-kernel"


def _compile_fn(kernel):
    if kernel == "good-kernel":
        return CompileResult(True, artifact="/tmp/k.neff")
    return CompileResult(False, error_log=AFFINE_SELECT_LOG)


def test_loop_converges_when_author_learns_from_feedback():
    loop = KernelRepairLoop(max_rounds=5, stall_patience=3)
    out = loop.run(_LearningAuthor(), _compile_fn)
    assert out.ok
    assert out.rounds == 2                      # round 1 fails -> feedback -> round 2 fixes
    assert out.artifact == "/tmp/k.neff"
    assert any(r.name == "tril-to-const-mask" for r in out.suggested_rewrites)


def test_loop_stalls_when_author_ignores_the_fix():
    # Author never consumes feedback -> identical error every round -> bail early,
    # honestly, instead of burning all rounds. The suggested rewrite is still
    # surfaced (a named work item: "apply tril-to-const-mask").
    loop = KernelRepairLoop(max_rounds=8, stall_patience=2)
    out = loop.run(lambda trail: "broken-kernel", _compile_fn)
    assert not out.ok
    assert out.reason.startswith("stalled")
    assert out.rounds == 2 < 8
    assert any(r.name == "tril-to-const-mask" for r in out.suggested_rewrites)


def test_loop_exhausts_rounds_on_unrecognized_evolving_errors():
    # A different, unrecognized error each round (no stall, no matching rewrite):
    # the loop runs to the cap and reports an honest failure, not a fake success.
    def author(trail):
        return f"attempt-{len(trail)}"

    def compile_fn(kernel):
        return CompileResult(False, error_log=f"{UNKNOWN_LOG} :: {kernel}")

    loop = KernelRepairLoop(max_rounds=4, stall_patience=3)
    out = loop.run(author, compile_fn)
    assert not out.ok
    assert out.reason == "exhausted rounds"
    assert out.rounds == 4
    assert out.suggested_rewrites == []         # nothing matched -> no false lead


def test_feedback_prompt_carries_the_fix():
    loop = KernelRepairLoop(max_rounds=1)
    out = loop.run(lambda trail: "broken-kernel", _compile_fn)
    assert out.trail, "a failed round should leave feedback"
    prompt = out.trail[0].as_prompt()
    assert "tril-to-const-mask" in prompt
    assert "constant" in prompt.lower()         # the actual fix snippet is included


def test_catalog_signatures_are_nonempty():
    # Guardrail: every catalog entry must have at least one signature or op, else
    # it can never match and is dead weight.
    for r in REWRITES:
        assert r.error_signatures or r.hostile_ops, r.name


# -- BUG #3: offline lint messages must reach the rewrite catalog ------------
from invent_kernels import static_lint  # noqa: E402 — kept next to its tests


def test_lint_messages_route_to_named_fixes():
    # Each real static_lint message must route to exactly its lint rewrite, so
    # the repair loop's "named fix" assist fires for a LINT symptom (not just a
    # compile symptom) instead of feeding the raw lint string back and stalling.
    cases = {
        # crafted-bad kernel snippet        -> expected rewrite name
        "x = nl.arange(0, 128)\n":                       "lint-arange-to-mgrid",
        "y = int(3.0)\n":                                "lint-int-cast-to-float-recip",
        "z = w.tile((2, 2))\n":                          "lint-tile-not-allowed",
        "a = nl.ndarray((256, 64), dtype=x.dtype)\n":    "lint-partition-dim-over-128",
    }
    for snippet, expected in cases.items():
        msgs = static_lint(snippet)
        assert msgs, f"snippet should lint-fail: {snippet!r}"
        for msg in msgs:
            hits = [r.name for r in match_error(msg)]
            assert expected in hits, (msg, hits)


def test_per_index_dma_lint_message_routes():
    dma = (
        "for k in nl.affine_range(8):\n"
        "    nisa.dma_copy(dst=buf[k, 0:128], src=w[k, 0:128])\n"
    )
    msgs = [m for m in static_lint(dma) if "per-index DMA" in m]
    assert msgs, static_lint(dma)
    hits = [r.name for r in match_error(msgs[0])]
    assert hits == ["lint-per-index-dma-to-multipartition"], hits


def test_lint_rewrites_apply_at_nki_kernel_and_describe():
    # The lint fixes edit kernel source, not the model graph.
    lint_names = {"lint-arange-to-mgrid", "lint-int-cast-to-float-recip",
                  "lint-tile-not-allowed", "lint-partition-dim-over-128",
                  "lint-per-index-dma-to-multipartition"}
    by_name = {r.name: r for r in REWRITES}
    assert lint_names <= set(by_name), lint_names - set(by_name)
    for n in lint_names:
        assert by_name[n].applies_at == "nki-kernel", n
    # describe() renders the routed fix actionably.
    text = describe(match_error("uses nl.arange (deprecated) — use nl.mgrid"))
    assert "lint-arange-to-mgrid" in text and "nl.mgrid" in text


def test_lint_entries_do_not_cross_match_compiler_logs():
    # No regression: the new lint entries must NOT fire on the real compiler
    # logs, and the existing compiler-log entries must still route as before.
    assert [r.name for r in match_error(AFFINE_SELECT_LOG)] == ["tril-to-const-mask"]
    assert [r.name for r in match_error(TOPK_LOG)] == ["int64-topk-to-float-view"]
    assert [r.name for r in match_error(SORT_LOG)] == ["topk-sort-to-argmax"]
    for log in (AFFINE_SELECT_LOG, TOPK_LOG, SORT_LOG, UNKNOWN_LOG):
        names = [r.name for r in match_error(log)]
        assert not any(n.startswith("lint-") for n in names), (log, names)


def test_compiler_log_signatures_do_not_appear_in_lint_messages():
    # The other direction: a real lint message must not accidentally trip any
    # compiler-log entry (tril / int64-topk / sort / dynamic-slice).
    lint_msgs = (
        static_lint("x = nl.arange(0, 128)\n")
        + static_lint("y = int(3.0)\n")
        + static_lint("z = w.tile((2, 2))\n")
        + static_lint("a = nl.ndarray((256, 64), dtype=x.dtype)\n")
    )
    graph_entries = {"tril-to-const-mask", "int64-topk-to-float-view",
                     "topk-sort-to-argmax", "dynamic-slice-to-static-bucket"}
    for msg in lint_msgs:
        names = {r.name for r in match_error(msg)}
        assert not (names & graph_entries), (msg, names)
