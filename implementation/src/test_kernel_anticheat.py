"""Tests for the adversarial correctness gate (kernel_anticheat.py) and its
integration into the verdict ladder (kernel_validation.py).

These encode the reward-hacks the LLM-kernel corpus learned the hard way:
Sakana's "100×" that was really the framework fallback, and Kevin-32B's kernels
that recycled the reference's output buffer. Each violation type must be caught,
a legitimate NKI kernel must pass clean, and a candidate the gate rejects must
NOT be able to reach a pass rank (>= 3)."""

from __future__ import annotations

from kernel_anticheat import (
    adversarial_source_check,
    require_reproducible,
    run_candidate_before_reference,
)
from kernel_validation import (
    FAILED_ADVERSARIAL,
    KernelValidation,
    reuse_decision,
    verdict,
)


# A legitimate (if toy) NKI kernel: real nl./nisa. compute, torch only for dtype.
_GOOD_KERNEL = '''
import torch  # dtype only
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def rmsnorm_kernel(x, w, out):
    xt = nl.load(x, dtype=torch.bfloat16)
    sq = nl.multiply(xt, xt)
    ms = nl.reduce_mean(sq, axis=-1)
    inv = nl.rsqrt(nl.add(ms, 1e-6))
    nl.store(out, nl.multiply(nl.multiply(xt, inv), nl.load(w)))
'''


# -- (1) static source anti-cheat --------------------------------------------

def test_legit_nki_kernel_passes_clean():
    assert adversarial_source_check(_GOOD_KERNEL) == []


def test_no_kernel_at_all_is_caught():
    src = '''
def deltanet_kernel(x, w):
    return x + w  # pure python, no NKI anywhere
'''
    reasons = adversarial_source_check(src)
    assert any("no NKI primitive" in r for r in reasons)


def test_framework_fallback_compute_path_is_caught():
    """torch/F compute op AS the compute path == the Sakana 'it's really the
    reference' hack."""
    src = '''
import torch
import torch.nn.functional as F
def attn_kernel(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)
'''
    reasons = adversarial_source_check(src)
    assert any("framework compute op" in r for r in reasons)
    # torch.matmul spelling too.
    src2 = '''
import torch
def mm_kernel(a, b):
    return torch.matmul(a, b)
'''
    assert any("framework compute op" in r
               for r in adversarial_source_check(src2))


def test_except_returns_torch_is_silent_fallback():
    """bare except that returns a torch result -> silent framework fallback."""
    src = '''
import torch
import neuronxcc.nki.language as nl
def k(x, w):
    try:
        return nl.matmul(x, w)
    except Exception:
        return torch.matmul(x, w)   # <- swallow + framework fallback
'''
    reasons = adversarial_source_check(src)
    assert any("except-handler returns a framework" in r for r in reasons)


def test_try_except_swallows_kernel_and_returns_substitute():
    """kernel body wrapped in try/except that swallows and returns a non-kernel
    (non-framework) substitute is still a reward hack."""
    src = '''
import neuronxcc.nki.language as nl
def k(x, w):
    try:
        out = nl.matmul(x, w)
        return out
    except Exception:
        return x   # swallow the failure, hand back the input
'''
    reasons = adversarial_source_check(src)
    assert any("swallows" in r for r in reasons)


def test_inherit_or_import_reference_is_caught():
    src_import = '''
from reference_impl import ReferenceDeltaNet
import neuronxcc.nki.language as nl
def k(x):
    return ReferenceDeltaNet().forward(x)
'''
    assert any("imports the reference" in r
               for r in adversarial_source_check(src_import))

    src_inherit = '''
import neuronxcc.nki.language as nl
from mymod import BaselineAttention
class MyKernel(BaselineAttention):
    def forward(self, x):
        return nl.matmul(x, x)
'''
    reasons = adversarial_source_check(src_inherit)
    assert any("inherits from the reference/baseline" in r for r in reasons)


def test_empty_and_unparseable_source():
    assert adversarial_source_check("") == ["empty kernel source (no kernel)"]
    assert adversarial_source_check("   \n  ") == ["empty kernel source (no kernel)"]
    # syntax error -> text-only fallback, and it says so.
    bad = "def k(:\n    nki broken"
    reasons = adversarial_source_check(bad)
    assert any("did not parse" in r for r in reasons)


# -- (2) reproducibility gate -------------------------------------------------

def test_deterministic_run_is_reproducible():
    ok, reason = require_reproducible(lambda: [1, 2, 3])
    assert ok and "reproducible" in reason


def test_nondeterministic_run_fails():
    seq = iter([1, 2, 3, 4])
    ok, reason = require_reproducible(lambda: next(seq))
    assert not ok and "non-reproducible" in reason


def test_run_that_raises_is_not_reproducible():
    def boom():
        raise RuntimeError("kernel exploded")
    ok, reason = require_reproducible(boom)
    assert not ok and "raised" in reason


def test_reproducible_clamps_n_to_two():
    calls = {"n": 0}
    def counted():
        calls["n"] += 1
        return 7
    ok, _ = require_reproducible(counted, n=1)   # clamped up to 2
    assert ok and calls["n"] == 2


# -- (3) buffer-order protocol ------------------------------------------------

def test_candidate_runs_before_reference():
    """The candidate MUST execute before the reference so it cannot alias the
    reference's already-filled output buffer (Kevin's recycled-output exploit)."""
    order = []
    cand_out, ref_out = run_candidate_before_reference(
        lambda: (order.append("candidate"), "C")[1],
        lambda: (order.append("reference"), "R")[1],
    )
    assert order == ["candidate", "reference"]
    assert cand_out == "C" and ref_out == "R"


# -- (4) integration: the veto cannot reach a pass rank ----------------------

def test_verdict_adversarial_ok_false_cannot_reach_pass():
    """Even with perfect numerics + NEFF + on-device, adversarial_ok=False forces
    failed-adversarial (rank 0) — never a pass (>= 3)."""
    from kernel_registry import STATUS_RANK
    status = verdict(numerics_ok=True, neff_emitted=True, on_device=True,
                     adversarial_ok=False)
    assert status == FAILED_ADVERSARIAL
    assert STATUS_RANK[status] < 3
    # default stays True -> unchanged behaviour.
    assert verdict(numerics_ok=True, neff_emitted=True, on_device=True) \
        == "passed-on-device"


def test_from_run_records_reason_and_cannot_pass():
    v = KernelValidation.from_run(
        numerics_ok=True, neff_emitted=True, on_device=True,
        adversarial_ok=False,
        adversarial_reasons=["framework compute op used as compute path"],
    )
    assert v.status == FAILED_ADVERSARIAL and v.rank == 0
    assert not v.passed and not v.hw_ready
    assert "adversarial" in v.notes and "framework compute op" in v.notes
    # and the router refuses to reuse it -> author for real.
    assert reuse_decision(v) == "AUTHOR"
