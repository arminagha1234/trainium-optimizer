"""
Tests for the Qwen3-Next / Qwen3.5 (GatedDeltaNet-MoE) Neuron rewrite bundle
(backends/qwen3_next_rewrites) and its catalog entry in kernel_rewrites.

Covers:
  1. Arch detection (is_qwen3_next_arch): model_type keyed, nested text_config,
     structural-marker fallback, and negative (dense) case — pure, no torch.
  2. Import-safety + graceful install: the module imports with no torch/
     transformers, and install_qwen3_next_neuron_rewrites NEVER raises.
  3. Catalog wiring: dense-moe-static-dispatch is in kernel_rewrites and both
     match_ops and match_error route to it, disjoint from the compile entries.
  4. (torch/transformers-gated) install patches the three modeling targets, is
     idempotent, and uninstall restores; the sort-free argmax router and the
     dense-MoE dispatch are NUMERICALLY EQUAL to their references on CPU.

Layers 1-3 are CPU-only (no torch, no network). Layer 4 skips automatically
when torch/transformers are unavailable (matches the moe-baseline-fix test's
"this box has no torch" contract).
"""

from __future__ import annotations

import types

import pytest

import kernel_rewrites as KR
from backends import qwen3_next_rewrites as QR


# --------------------------------------------------------------------------- #
# 1. arch detection — pure, duck-typed configs (no torch)                     #
# --------------------------------------------------------------------------- #
def test_detect_by_model_type():
    cfg = types.SimpleNamespace(model_type="qwen3_next")
    assert QR.is_qwen3_next_arch(cfg) is True


def test_detect_by_nested_text_config():
    cfg = types.SimpleNamespace(
        model_type="qwen3_next_vl",
        text_config=types.SimpleNamespace(model_type="qwen3_next"))
    assert QR.is_qwen3_next_arch(cfg) is True


def test_detect_by_structural_markers():
    # No qwen3_next in model_type, but the GatedDeltaNet linear-attn fields exist.
    cfg = types.SimpleNamespace(
        model_type="custom_remote",
        linear_key_head_dim=32, linear_num_value_heads=4,
        linear_conv_kernel_dim=4)
    assert QR.is_qwen3_next_arch(cfg) is True


def test_detect_negative_for_dense_model():
    cfg = types.SimpleNamespace(model_type="llama")
    assert QR.is_qwen3_next_arch(cfg) is False


# --------------------------------------------------------------------------- #
# 2. import-safety + graceful install (never raises)                          #
# --------------------------------------------------------------------------- #
def test_install_never_raises_and_returns_bool():
    # In a torch/transformers-less env this returns False (graceful); where they
    # exist it returns True (and patches). Either way: a bool, never an exception.
    try:
        result = QR.install_qwen3_next_neuron_rewrites(lambda *_: None)
    finally:
        QR.uninstall_qwen3_next_neuron_rewrites()
    assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
# 3. catalog wiring (kernel_rewrites) — pure                                  #
# --------------------------------------------------------------------------- #
def test_dense_moe_in_catalog():
    names = [r.name for r in KR.REWRITES]
    assert "dense-moe-static-dispatch" in names


def test_dense_moe_matched_by_ops():
    for op in ("grouped_mm", "aten::nonzero", "aten::index_add_", "aten::one_hot"):
        hits = [r.name for r in KR.match_ops([op])]
        assert "dense-moe-static-dispatch" in hits, op


def test_dense_moe_matched_by_error_log():
    hits = [r.name for r in KR.match_error("numeric drift in grouped_mm_experts_forward")]
    assert "dense-moe-static-dispatch" in hits


def test_dense_moe_signatures_disjoint_from_compile_entries():
    # The correctness entry must not steal a compile-abort log (and vice versa).
    dense = next(r for r in KR.REWRITES if r.name == "dense-moe-static-dispatch")
    for other in KR.REWRITES:
        if other.name == "dense-moe-static-dispatch":
            continue
        assert not (set(dense.error_signatures) & set(other.error_signatures)), other.name


# --------------------------------------------------------------------------- #
# 4. numerical + patch-install correctness (needs torch + transformers)       #
# --------------------------------------------------------------------------- #
def test_install_patches_and_restores_targets():
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("transformers.models.qwen3_next.modeling_qwen3_next")
    from transformers.models.qwen3_next import modeling_qwen3_next as M
    orig_router = M.Qwen3NextTopKRouter.forward
    orig_gdr = M.torch_chunk_gated_delta_rule
    orig_experts = M.Qwen3NextExperts.forward
    try:
        assert QR.install_qwen3_next_neuron_rewrites(lambda *_: None) is True
        assert M.Qwen3NextTopKRouter.forward is QR.sortfree_router_forward
        assert M.torch_chunk_gated_delta_rule is QR.chunk_gdr_constmask
        assert M.Qwen3NextExperts.forward is QR.dense_experts_forward
        # Idempotent: a second install is a no-op (returns False).
        assert QR.install_qwen3_next_neuron_rewrites(lambda *_: None) is False
    finally:
        QR.uninstall_qwen3_next_neuron_rewrites()
    assert M.Qwen3NextTopKRouter.forward is orig_router
    assert M.torch_chunk_gated_delta_rule is orig_gdr
    assert M.Qwen3NextExperts.forward is orig_experts


def test_sortfree_router_matches_topk():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    torch.manual_seed(0)
    T_tok, H, E, K = 7, 16, 8, 2
    fake = types.SimpleNamespace(
        hidden_dim=H, top_k=K, norm_topk_prob=True,
        weight=torch.randn(E, H))
    h = torch.randn(T_tok, H)
    logits, top_val, top_idx = QR.sortfree_router_forward(fake, h)
    # Reference: exactly what HF's router does with torch.topk.
    ref_logits = F.linear(h, fake.weight)
    probs = torch.softmax(ref_logits, dtype=torch.float, dim=-1)
    ref_val, ref_idx = torch.topk(probs, K)
    ref_val = (ref_val / ref_val.sum(-1, keepdim=True)).to(ref_logits.dtype)
    assert torch.equal(top_idx, ref_idx)
    assert torch.allclose(top_val, ref_val, atol=1e-6)


def test_dense_experts_matches_topk_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    torch.manual_seed(0)
    T_tok, H, I, E, K = 5, 12, 10, 6, 2
    fake = types.SimpleNamespace(
        num_experts=E, act_fn=F.silu,
        gate_up_proj=torch.randn(E, 2 * I, H) * 0.1,
        down_proj=torch.randn(E, H, I) * 0.1)
    h = torch.randn(T_tok, H)
    # Random distinct top-k selection per token + normalized gate weights.
    scores = torch.randn(T_tok, E)
    top_w, top_idx = torch.topk(torch.softmax(scores, dim=-1), K)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    out = QR.dense_experts_forward(fake, h, top_idx, top_w)
    # Reference: sum only the selected experts, weighted by the gate.
    ref = torch.zeros_like(h)
    for t in range(T_tok):
        for j in range(K):
            e = int(top_idx[t, j])
            g, u = F.linear(h[t], fake.gate_up_proj[e]).chunk(2, dim=-1)
            ref[t] += top_w[t, j] * F.linear(fake.act_fn(g) * u, fake.down_proj[e])
    assert torch.allclose(out, ref, atol=1e-5)
