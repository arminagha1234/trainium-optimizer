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
