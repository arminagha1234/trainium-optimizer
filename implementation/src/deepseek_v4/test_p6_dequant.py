"""P6 - FP8/FP4 dequant round-trip + expert-parallel slice (CPU, CI-safe)."""
import pytest
torch = pytest.importorskip("torch")

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepseek_v4.dequant import roundtrip_metrics, expert_shard_range, slice_experts  # noqa: E402


def test_fp8_e4m3_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    m = roundtrip_metrics(w, "e4m3")
    assert m["cos"] >= 0.999, m          # 8-bit block quant
    assert m["rel_rms"] <= 0.05, m


def test_fp4_e2m1_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(256, 256, dtype=torch.bfloat16)
    m = roundtrip_metrics(w, "fp4")
    assert m["cos"] >= 0.98, m           # 4-bit block quant is coarse
    assert m["rel_rms"] <= 0.25, m


def test_expert_shard_range_and_slice():
    # 256 experts at tp=64 -> 4 experts/rank, disjoint and covering.
    seen = set()
    for r in range(64):
        a, b = expert_shard_range(256, r, 64)
        assert b - a == 4
        seen.update(range(a, b))
    assert seen == set(range(256))
    stacked = torch.arange(8).view(8, 1, 1).expand(8, 2, 2).clone()
    s = slice_experts(stacked, rank=1, tp=4)     # experts [2,3]
    assert s.shape[0] == 2 and int(s[0, 0, 0]) == 2


def test_expert_shard_indivisible_raises():
    with pytest.raises(ValueError):
        expert_shard_range(256, 0, 7)
