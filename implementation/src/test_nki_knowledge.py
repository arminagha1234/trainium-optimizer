"""Tests for ``nki_knowledge`` — the op-indexed knowledge & worked-example base
(Pillar 1). All CPU-only, no Trainium, no heavy imports.

Coverage:
  * ``classify_op`` maps each framework op (and unseen names) to the right
    op-family, name-first with a notes fallback and an elementwise default.
  * ``retrieve`` returns the family entry for an op-name AND an OpSpec-like object.
  * every technique / signature / landmine KEY referenced by an entry resolves in
    its registry (no dangling keys) — the data stays consistent.
  * the worked examples obey the VERIFIED 0.6.0 return-form (no dst=/out= on
    nc_matmul / nc_transpose / activation) — the whole point of distilling from
    the expert kernels was to translate AWAY from their dst= form.
  * ``render_knowledge_section`` surfaces the op-relevant technique/signature text
    and a fenced worked example for each family.

Runnable two ways:
    python -m pytest -q test_nki_knowledge.py
    python test_nki_knowledge.py
"""

from __future__ import annotations

import nki_knowledge as K


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def test_classify_framework_ops():
    # The exact ops in invent_kernels.catalog() must land in the right family.
    cases = {
        "rmsnorm": K.NORMALIZATION,
        "add_rmsnorm": K.NORMALIZATION,
        "layernorm": K.NORMALIZATION,
        "softmax": K.SOFTMAX,
        "attn_decode": K.ATTENTION,
        "softcap": K.ELEMENTWISE,
        "gelu_tanh": K.ELEMENTWISE,
        "silu_gate": K.ELEMENTWISE,
        "rope_apply": K.ELEMENTWISE,
    }
    for name, fam in cases.items():
        assert K.classify_op(name) == fam, f"{name} -> {K.classify_op(name)} != {fam}"


def test_classify_novel_arch_ops():
    # The families the author must grow into (novel archs) classify correctly.
    assert K.classify_op("matmul") == K.MATMUL
    assert K.classify_op("qkv_proj") == K.MATMUL
    assert K.classify_op("router_topk") == K.MOE_ROUTER
    assert K.classify_op("moe_gate") == K.MOE_ROUTER
    assert K.classify_op("ssd") == K.SCAN
    assert K.classify_op("mamba2_scan") == K.SCAN
    assert K.classify_op("gated_delta_net") == K.SCAN
    assert K.classify_op("kda_chunk") == K.SCAN
    assert K.classify_op("flash_attention") == K.ATTENTION


def test_classify_prefers_name_then_notes_then_default():
    # NAME wins over notes.
    assert K.classify_op("attn_decode", notes="uses a softmax internally") == K.ATTENTION
    # A bare/opaque name falls back to the NOTES prose.
    assert K.classify_op("op7", notes="fused residual-add + RMSNorm") == K.NORMALIZATION
    assert K.classify_op("op8", notes="tiled QK^T-softmax-PV decode") == K.ATTENTION
    # Nothing recognizable -> safe elementwise default (retrieval never empty).
    assert K.classify_op("totally_unknown_op_xyz") == K.ELEMENTWISE


def test_classify_ignores_model_family_field():
    # spec.family is a MODEL family (dense_causal_lm), NOT an op family — it must
    # not steer classification.
    assert K.classify_op("softmax", family="dense_causal_lm") == K.SOFTMAX


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
class _FakeSpec:
    def __init__(self, name, family="dense_causal_lm", notes=""):
        self.name = name
        self.family = family
        self.notes = notes


def test_retrieve_from_name_and_from_spec():
    e1 = K.retrieve("rmsnorm")
    assert e1.family == K.NORMALIZATION
    e2 = K.retrieve(_FakeSpec("attn_decode", notes="QK^T softmax PV"))
    assert e2.family == K.ATTENTION
    # An OpSpec-like whose name is opaque but whose notes name the op.
    e3 = K.retrieve(_FakeSpec("op9", notes="Mamba-2 selective_scan"))
    assert e3.family == K.SCAN


def test_every_family_has_an_entry_and_examples():
    for fam in K.OP_FAMILIES:
        e = K.KNOWLEDGE[fam]
        assert e.family == fam
        assert e.techniques, f"{fam} has no techniques"
        assert e.examples, f"{fam} has no worked example"


# ---------------------------------------------------------------------------
# data consistency — no dangling keys
# ---------------------------------------------------------------------------
def test_all_referenced_keys_resolve():
    for fam, e in K.KNOWLEDGE.items():
        for k in e.techniques:
            assert k in K.TECHNIQUES, f"{fam}: unknown technique key {k!r}"
        for k in e.signatures:
            assert k in K.SIGNATURES, f"{fam}: unknown signature key {k!r}"
        for k in e.landmines:
            assert k in K.LANDMINES, f"{fam}: unknown landmine key {k!r}"


# ---------------------------------------------------------------------------
# the worked examples obey the verified 0.6.0 RETURN-form
# ---------------------------------------------------------------------------
def test_worked_examples_use_return_form_not_dst():
    # The distilled examples must NOT reintroduce the public nki-samples dst= form
    # (nc_matmul(dst=...) etc.) — that errors on this stack. This is the core
    # translation the module performs, so guard it.
    forbidden = ("nc_matmul(dst", "nc_transpose(dst", "activation(dst",
                 "dst=qk", "dst=attn", ", dst=")
    for fam, e in K.KNOWLEDGE.items():
        for ex in e.examples:
            low = ex.code
            for bad in forbidden:
                assert bad not in low, f"{fam} example reintroduced dst= form: {bad}"


def test_worked_examples_are_code_or_comment_blocks():
    for fam, e in K.KNOWLEDGE.items():
        for ex in e.examples:
            # Either a real @nki.jit kernel or a distilled comment-pseudocode block.
            assert ("@nki.jit" in ex.code) or ex.code.lstrip().startswith("#"), (
                f"{fam} example is neither a jit kernel nor a comment block")


# ---------------------------------------------------------------------------
# rendering — op-relevant content reaches the prompt
# ---------------------------------------------------------------------------
def test_render_softmax_has_delayed_division_and_example():
    s = K.knowledge_for_prompt("softmax")
    assert "op family: softmax" in s
    assert "delayed-softmax-division" in s
    assert "reciprocal" in s
    assert "```python" in s               # a worked example is embedded
    assert "keepdims" in s


def test_render_attention_has_flash_and_matmul_signature():
    s = K.knowledge_for_prompt("attn_decode")
    assert "op family: attention" in s
    low = s.lower()
    assert "flash" in low or "online" in low
    assert "nisa.nc_matmul(stationary, moving" in s   # the return-form signature
    assert "attn_decode_kernel" in s                  # the attention worked example


def test_render_reduction_has_keepdims_and_fusion():
    s = K.knowledge_for_prompt("rmsnorm")
    assert "op family: normalization" in s
    assert "keepdims-2d" in s or "keepdims" in s
    assert "activation-reduce-fusion" in s
    assert "rmsnorm_kernel" in s


def test_render_matmul_has_psum_tiling():
    s = K.knowledge_for_prompt("matmul")
    low = s.lower()
    assert "psum" in low
    assert "512" in s
    assert "matmul_kernel" in s


def test_render_scan_and_router_families():
    scan = K.knowledge_for_prompt("ssd")
    assert "chunk" in scan.lower() and "op family: scan" in scan
    router = K.knowledge_for_prompt("router_topk")
    assert "sort-free" in router.lower() and "op family: moe_router" in router


def test_render_only_includes_relevant_subset():
    # An elementwise op should NOT drag in the scan/router-specific techniques.
    s = K.knowledge_for_prompt("softcap")
    assert "op family: elementwise" in s
    assert "chunked-scan" not in s
    assert "sort-free-topk" not in s


# ===========================================================================
# standalone runner (no pytest required)
# ===========================================================================
def _run_standalone() -> int:
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(fns)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
