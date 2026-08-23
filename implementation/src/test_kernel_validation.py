"""Tests for the rank ladder + reuse-vs-author router (kernel_validation.py)."""

from __future__ import annotations

from kernel_registry import STATUS_RANK, KernelSpec
from kernel_validation import (
    AUTHOR,
    CONTINUE,
    REUSE,
    REVALIDATE_ON_DEVICE,
    KernelValidation,
    reuse_decision,
    verdict,
)


# -- verdict: the single honest pass gate ------------------------------------

def test_sim_pass_vs_on_device_pass_are_different_tiers():
    """numerics + NEFF in simulation is rank-3 'passed'; the SAME on real
    silicon is the strictly higher rank-4 'passed-on-device'."""
    sim = verdict(numerics_ok=True, neff_emitted=True, on_device=False)
    dev = verdict(numerics_ok=True, neff_emitted=True, on_device=True)
    assert sim == "passed"
    assert dev == "passed-on-device"
    assert STATUS_RANK[dev] > STATUS_RANK[sim]


def test_import_only_is_not_a_pass():
    """Kernel imported / numerics fine but NO emitted NEFF -> NOT a pass.
    'It imported' never ships."""
    v = verdict(numerics_ok=True, neff_emitted=False, on_device=False)
    assert v == "failed-compile"
    assert STATUS_RANK[v] < STATUS_RANK["passed"]
    # even on a real device, no NEFF is still not a pass.
    assert verdict(numerics_ok=True, neff_emitted=False, on_device=True) \
        == "failed-compile"


def test_compiled_but_numerics_off_is_failed_numerical():
    v = verdict(numerics_ok=False, neff_emitted=True, on_device=False)
    assert v == "failed-numerical"
    assert STATUS_RANK[v] < STATUS_RANK["passed"]
    # on_device never rescues a numerics failure.
    assert verdict(numerics_ok=False, neff_emitted=True, on_device=True) \
        == "failed-numerical"


def test_kernel_validation_from_run_wires_status_rank_tier():
    sim = KernelValidation.from_run(numerics_ok=True, neff_emitted=True,
                                    on_device=False, numeric_error=2e-7)
    assert sim.status == "passed" and sim.rank == 3 and sim.tier == "simulate"
    assert sim.passed and not sim.hw_ready

    dev = KernelValidation.from_run(numerics_ok=True, neff_emitted=True,
                                    on_device=True, numeric_error=6.6e-4,
                                    artifact="/neff/x.neff")
    assert dev.status == "passed-on-device" and dev.rank == 4
    assert dev.tier == "on-device" and dev.hw_ready and dev.artifact.endswith(".neff")

    # a failed-on-device run is still tagged on-device (honest audit), not a pass
    bad = KernelValidation.from_run(numerics_ok=False, neff_emitted=True,
                                    on_device=True)
    assert bad.tier == "on-device" and not bad.passed


# -- router ------------------------------------------------------------------

def test_rank3_routes_to_revalidate_not_reuse():
    """A simulate-only pass (rank 3) must be re-proven on device, NOT reused
    blindly (the Mamba simulate!=silicon lesson)."""
    assert reuse_decision(3) == REVALIDATE_ON_DEVICE
    sim = KernelValidation.from_run(numerics_ok=True, neff_emitted=True,
                                    on_device=False)
    assert reuse_decision(sim) == REVALIDATE_ON_DEVICE


def test_rank4_reuse():
    assert reuse_decision(4) == REUSE
    dev = KernelValidation.from_run(numerics_ok=True, neff_emitted=True,
                                    on_device=True)
    assert reuse_decision(dev) == REUSE
    # a hw-ready KernelSpec from the registry routes to REUSE too.
    assert reuse_decision(KernelSpec(name="DeltaNet",
                                     status="passed-on-device")) == REUSE


def test_rank1_2_continue():
    assert reuse_decision(1) == CONTINUE          # failed-compile: keep repairing
    assert reuse_decision(2) == CONTINUE          # compiled-but-off
    assert reuse_decision(KernelSpec(name="X", status="failed-numerical")) \
        == CONTINUE


def test_empty_corpus_and_stub_route_to_author():
    """No kernel at all (empty corpus / None) -> AUTHOR; a rank-0 analysis-only
    stub is also nothing to reuse -> AUTHOR. Never 'blocker'."""
    assert reuse_decision(None) == AUTHOR
    assert reuse_decision(0) == AUTHOR
    assert reuse_decision(KernelSpec(name="X", status="analysis-only")) == AUTHOR
    # the registry lookup for an unavailable kernel returns None -> AUTHOR.
    assert reuse_decision(KernelSpec(name="X", status="written-not-compiled")) \
        == AUTHOR


def test_router_never_returns_blocker():
    for r in (None, -5, 0, 1, 2, 3, 4, 99):
        assert reuse_decision(r) in {REUSE, REVALIDATE_ON_DEVICE, CONTINUE, AUTHOR}
