"""Tests for the CPU-oracle registry (kernel_oracles.py)."""

from __future__ import annotations

import numpy as np

from kernel_oracles import (
    Oracle,
    audit_oracles,
    get_oracle,
    register_oracle,
)


# -- alias resolution --------------------------------------------------------

def test_alias_resolution_via_primitive_map_and_explicit_aliases():
    """The DeltaNet oracle is reachable via its canonical name, its explicit
    aliases, AND any PRIMITIVE_TO_KERNEL primitive spelling that routes to it."""
    canonical = get_oracle("DeltaNet")
    assert canonical is not None and canonical.name == "DeltaNet"

    # PRIMITIVE_TO_KERNEL spellings (normalized) -> DeltaNet
    for spelling in ("gated_delta_net", "GatedDeltaNet", "gated-delta",
                     "deltanet", "delta_rule", "linear_attention", "linear_attn"):
        o = get_oracle(spelling)
        assert o is canonical, f"{spelling!r} did not resolve to the DeltaNet oracle"

    assert get_oracle("no_such_primitive_xyz") is None


# -- vacuous-oracle detection (the orphan-oracle guard) ----------------------

def test_vacuous_oracle_is_flagged():
    ref = lambda inp: inp["x"]
    # sim IS reference -> vacuous (compares a value to itself)
    same = Oracle("Vac1", ref, ref, lambda: {"x": np.zeros(4)})
    assert same.vacuous
    # sim missing -> vacuous
    missing = Oracle("Vac2", ref, None, lambda: {"x": np.zeros(4)})
    assert missing.vacuous
    # independent sim -> NOT vacuous
    good = Oracle("Good", ref, lambda inp: inp["x"] + 0.0,
                  lambda: {"x": np.zeros(4)})
    assert not good.vacuous


def test_audit_flags_registered_vacuous_oracle():
    ref = lambda inp: inp["x"]
    register_oracle("VacuousUnderTest", ref, ref, lambda: {"x": np.zeros(4)})
    report = audit_oracles()
    assert "VacuousUnderTest" in report["vacuous"]
    # audit also reports kernels named in the primitive map that have no oracle,
    # so a whole class can't be silently skipped.
    assert isinstance(report["missing"], list)


# -- a real oracle round-trips -----------------------------------------------

def test_deltanet_oracle_round_trips_make_inputs_to_reference():
    o = get_oracle("gated_delta_net")
    assert o is not None and not o.vacuous
    inp = o.make_inputs()
    out = o.reference(inp)
    assert isinstance(out, np.ndarray)
    assert out.shape == (inp["q"].shape[0], inp["q"].shape[1])
    assert np.all(np.isfinite(out))
    # deterministic: same inputs -> same reference output
    assert np.array_equal(out, o.reference(o.make_inputs()))
    # the independent sim matches the reference (a MEANINGFUL parity check)
    sim_out = o.sim(inp)
    assert np.allclose(out, sim_out, atol=1e-4, rtol=1e-4)


def test_rope_oracle_reference_and_sim_agree():
    """The reused invent_kernels RoPE pair (strided-scatter ref vs scatter-free
    sim) is registered and its two independent impls agree."""
    o = get_oracle("rope_apply")
    if o is None:          # invent_kernels unavailable -> RoPE oracle not registered
        return
    assert not o.vacuous
    inp = o.make_inputs()
    assert np.allclose(o.reference(inp), o.sim(inp), atol=1e-5)


# -- attention-sink oracle (new: GPT-OSS / DeepSeek-V4 primitive) ------------

def test_attention_sink_alias_resolution():
    """The AttentionSink oracle is reachable via its canonical name, its explicit
    aliases, and any PRIMITIVE_TO_KERNEL spelling that routes to it (attn_sink,
    gpt_oss, ...)."""
    canonical = get_oracle("AttentionSink")
    if canonical is None:      # invent_kernels unavailable -> oracle not registered
        return
    assert canonical.name == "AttentionSink"
    for spelling in ("attentionsink", "attention_sink", "attn_sink",
                     "sink_attention", "gpt_oss", "GPT-OSS", "gptoss", "sink"):
        o = get_oracle(spelling)
        assert o is canonical, f"{spelling!r} did not resolve to the AttentionSink oracle"


def test_attention_sink_oracle_reference_and_sim_agree():
    """The two independent impls (full-softmax-with-sink reference vs
    online/blocked-softmax-with-sink sim) agree — a MEANINGFUL, non-vacuous parity
    check for the sink primitive."""
    o = get_oracle("AttentionSink")
    if o is None:
        return
    assert not o.vacuous
    inp = o.make_inputs()
    out = o.reference(inp)
    assert isinstance(out, np.ndarray) and np.all(np.isfinite(out))
    assert out.shape == (1, inp["q"].shape[1])
    # deterministic + the online-softmax sim reproduces the full-softmax reference
    assert np.array_equal(out, o.reference(o.make_inputs()))
    assert np.allclose(out, o.sim(inp), atol=1e-5)


def test_attention_sink_oracle_catches_a_missing_sink():
    """The oracle must actually VALIDATE the sink: a plain softmax that DROPS the
    sink term diverges from the reference, so the parity check would reject it
    (this is what makes the oracle worth having)."""
    o = get_oracle("AttentionSink")
    if o is None:
        return
    inp = o.make_inputs()
    q, k, v = inp["q"], inp["k"], inp["v"]
    d = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d)
    e = np.exp(scores - scores.max())
    no_sink = (e / e.sum()) @ v            # softmax WITHOUT the sink term
    assert not np.allclose(o.reference(inp), no_sink, atol=1e-5)


def test_attention_sink_is_covered_by_audit():
    """audit_oracles no longer reports AttentionSink as a missing (uncovered)
    primitive."""
    o = get_oracle("AttentionSink")
    if o is None:
        return
    assert "AttentionSink" not in audit_oracles()["missing"]


# -- head_dim=256 decode oracle (validates the harvested decode_hd256 kernels) -

def test_attn_hd256_alias_resolution():
    """The AttnDecodeHD256 oracle is reachable via its canonical name, explicit
    aliases, and PRIMITIVE_TO_KERNEL spellings (hd256, decode_hd256, ...)."""
    canonical = get_oracle("AttnDecodeHD256")
    if canonical is None:      # invent_kernels unavailable -> oracle not registered
        return
    assert canonical.name == "AttnDecodeHD256"
    for spelling in ("attndecodehd256", "decode_hd256", "attn_hd256", "hd256",
                     "head_dim_256", "attention_decode_hd256"):
        assert get_oracle(spelling) is canonical, f"{spelling!r} did not resolve"


def test_attn_hd256_oracle_reference_and_sim_agree():
    """Full-head_dim softmax reference vs split-K/split-V online-softmax sim —
    the two independent derivations agree (non-vacuous parity for the primitive
    the harvested customer_armin decode_hd256 kernels implement)."""
    o = get_oracle("AttnDecodeHD256")
    if o is None:
        return
    assert not o.vacuous
    inp = o.make_inputs()
    out = o.reference(inp)
    assert isinstance(out, np.ndarray) and np.all(np.isfinite(out))
    assert out.shape == (1, inp["q"].shape[1]) and inp["q"].shape[1] == 256
    assert np.array_equal(out, o.reference(o.make_inputs()))    # deterministic
    assert np.allclose(out, o.sim(inp), atol=1e-4)             # split-K == full


def test_attn_hd256_oracle_catches_a_dropped_head_half():
    """The oracle validates that BOTH 128-halves of the split-K head dim are
    used: an impl that drops the hi half (a real split-K bug) diverges from the
    reference, so the parity check rejects it."""
    o = get_oracle("AttnDecodeHD256")
    if o is None:
        return
    inp = o.make_inputs()
    q, k, v = inp["q"], inp["k"], inp["v"]
    d = q.shape[-1]
    h = d // 2
    scores = (q[:, :h] @ k[:, :h].T) / np.sqrt(d)     # lo half only (dropped hi)
    e = np.exp(scores - scores.max())
    lo_only = (e / e.sum()) @ v
    assert not np.allclose(o.reference(inp), lo_only, atol=1e-4)


def test_attn_hd256_is_covered_by_audit():
    o = get_oracle("AttnDecodeHD256")
    if o is None:
        return
    assert "AttnDecodeHD256" not in audit_oracles()["missing"]
