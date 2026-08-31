"""Regression tests for backends.gdn_matmul_inv (the tp>1 GDN compile fix).

Self-contained (no transformers dependency): validates the matmul-only inverse
against the exact inverse in the GatedDeltaNet operating regime, and validates
the full chunk_matmul_inv drop-in against an exact-inverse reference run through
the SAME code path (isolating the inverse-approximation error). Skips cleanly if
torch is unavailable.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import gdn_matmul_inv as G  # noqa: E402


def _l2(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _gdn_A(C, d, seed, with_decay=True):
    """Build A exactly as the GDN chunk does: A = -(beta*k) @ k^T (*) decay,
    strictly lower. This is the small-spectral-radius regime the matmul-only
    inverse (banded Neumann + refinement) targets."""
    gen = torch.Generator().manual_seed(seed)
    k = _l2(torch.randn(C, d, generator=gen))
    beta = torch.rand(C, generator=gen)
    gate = -torch.nn.functional.softplus(torch.randn(C, generator=gen))
    gcum = torch.cumsum(gate, 0)
    decay = (gcum[:, None] - gcum[None, :]).exp() if with_decay else torch.ones(C, C)
    A = -((k * beta[:, None]) @ k.transpose(-1, -2)) * decay
    return A * torch.tril(torch.ones(C, C), -1)


def _fwd_subst_inverse(A):
    """The stock HF forward-substitution inverse (exact) for strict-lower A."""
    C = A.shape[-1]
    attn = A.clone()
    for i in range(1, C):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn + torch.eye(C, dtype=A.dtype)


def _cos(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize("C", [16, 32, 64])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matmul_inverse_matches_exact_on_gdn_A(C, seed):
    A = _gdn_A(C, 128, seed)
    got = G.matmul_only_inverse(A, N=3, Sres=8)
    ref = torch.linalg.inv(torch.eye(C) - A)
    assert _cos(got, ref) > 0.9999
    assert (got - ref).abs().max().item() < 5e-3


@pytest.mark.parametrize("C", [16, 64])
def test_matmul_inverse_matches_forward_substitution(C):
    A = _gdn_A(C, 128, seed=5)
    got = G.matmul_only_inverse(A, N=3, Sres=8)
    ref = _fwd_subst_inverse(A)  # exact stock algorithm
    assert _cos(got, ref) > 0.9999


def test_matmul_inverse_batched():
    # matmul_only_inverse must broadcast over leading [B,H,NC] dims
    A = torch.stack([torch.stack([_gdn_A(64, 128, s + 10 * h) for s in range(3)]) for h in range(2)])
    got = G.matmul_only_inverse(A, N=3, Sres=8)
    ref = torch.linalg.inv(torch.eye(64) - A)
    assert _cos(got, ref) > 0.9999


def _rand_gdn_inputs(B=1, H=2, L=128, d=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, L, H, d, generator=gen)
    k = torch.randn(B, L, H, d, generator=gen)
    v = torch.randn(B, L, H, d, generator=gen)
    beta = torch.rand(B, L, H, generator=gen)
    g = -torch.nn.functional.softplus(torch.randn(B, L, H, generator=gen))
    return q, k, v, g, beta


def _exact_inverse(A, N=3, Sres=8):
    C = A.shape[-1]
    return torch.linalg.inv(torch.eye(C, dtype=A.dtype, device=A.device) - A)


def test_chunk_matmul_inv_matches_exact_inverse_end_to_end():
    """chunk_matmul_inv (approx inverse) vs the SAME chunk code with the exact
    inverse -> proves the matmul-inverse swap does not change GDN output."""
    q, k, v, g, beta = _rand_gdn_inputs()
    args = dict(chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True)
    out_mm, st_mm = G.chunk_matmul_inv(q, k, v, g, beta, **args)
    orig = G.matmul_only_inverse
    G.matmul_only_inverse = _exact_inverse
    try:
        out_ex, st_ex = G.chunk_matmul_inv(q, k, v, g, beta, **args)
    finally:
        G.matmul_only_inverse = orig
    assert _cos(out_mm, out_ex) > 0.999
    assert _cos(st_mm, st_ex) > 0.999
    assert torch.isfinite(out_mm).all()


def test_chunk_matmul_inv_handles_padding():
    q, k, v, g, beta = _rand_gdn_inputs(L=200)  # not a multiple of chunk_size
    out, st = G.chunk_matmul_inv(q, k, v, g, beta, chunk_size=64, output_final_state=True,
                                 use_qk_l2norm_in_kernel=True)
    assert out.shape[1] == 200
    assert torch.isfinite(out).all()


def test_install_uninstall_returns_list():
    # install()/uninstall() are safe no-ops when no qwen modeling module present
    G.uninstall_gdn_matmul_inverse()
    patched = G.install_gdn_matmul_inverse(lambda s: None)
    assert isinstance(patched, list)
    G.uninstall_gdn_matmul_inverse()
