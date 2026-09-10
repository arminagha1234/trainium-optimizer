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


# -- MLA (Multi-head Latent Attention) decode oracle (DeepSeek-V2/V3, GLM) ----

def test_mla_alias_resolution():
    """The MLA oracle is reachable via its canonical name, explicit aliases, and
    any PRIMITIVE_TO_KERNEL spelling that routes to it."""
    canonical = get_oracle("MLA")
    assert canonical is not None and canonical.name == "MLA"
    for spelling in ("mla", "MLA", "multi_head_latent_attention",
                     "multihead_latent_attention", "latent_attention",
                     "mla_decode", "mla_attention", "deepseek_mla"):
        o = get_oracle(spelling)
        assert o is canonical, f"{spelling!r} did not resolve to the MLA oracle"


def test_mla_oracle_reference_and_sim_agree():
    """MATERIALIZE reference (reconstruct per-head K/V from the latent) vs ABSORB
    sim (fold W_UK into Q and W_UV into the output, attend in latent space) — two
    genuinely different algorithms that must agree by matrix absorption. A
    non-vacuous parity check for MLA decode."""
    o = get_oracle("MLA")
    assert o is not None and not o.vacuous
    inp = o.make_inputs()
    out = o.reference(inp)
    assert isinstance(out, np.ndarray) and np.all(np.isfinite(out))
    # output is [H, d_v]
    assert out.shape == (inp["qn"].shape[0], inp["w_uv"].shape[1])
    assert np.array_equal(out, o.reference(o.make_inputs()))    # deterministic
    assert np.allclose(out, o.sim(inp), atol=1e-4)              # absorb == materialize


def test_mla_oracle_catches_a_dropped_rope_term():
    """The oracle must actually VALIDATE the decoupled-RoPE key path — MLA's
    distinctive, easy-to-drop piece. An impl that omits the shared rope score
    term diverges from the reference, so the parity check rejects it."""
    o = get_oracle("MLA")
    assert o is not None
    inp = o.make_inputs()
    qn, w_uk, c, w_uv = inp["qn"], inp["w_uk"], inp["c"], inp["w_uv"]
    scale = inp["scale"]
    # nope-only attention (drops the decoupled rope key contribution)
    q_abs = np.einsum("hx,hxc->hc", qn.astype(np.float64), w_uk.astype(np.float64))
    nope = np.einsum("hc,sc->hs", q_abs, c.astype(np.float64)) * scale
    e = np.exp(nope - nope.max(axis=1, keepdims=True))
    a = e / e.sum(axis=1, keepdims=True)
    ctx = np.einsum("hs,sc->hc", a, c.astype(np.float64))
    no_rope = np.einsum("hc,hvc->hv", ctx, w_uv.astype(np.float64)).astype(np.float32)
    assert not np.allclose(o.reference(inp), no_rope, atol=1e-4)


def test_mla_is_covered_by_audit():
    """audit_oracles no longer reports MLA as a missing (uncovered) primitive."""
    assert "MLA" not in audit_oracles()["missing"]


# -- Mamba2 / SSD (state-space duality) decode oracle -------------------------

def test_mamba2_alias_resolution():
    """The Mamba2 oracle is reachable via its canonical name, explicit aliases,
    and any PRIMITIVE_TO_KERNEL spelling that routes to it."""
    canonical = get_oracle("Mamba2")
    assert canonical is not None and canonical.name == "Mamba2"
    for spelling in ("mamba2", "Mamba2", "mamba_2", "ssd", "ssm", "mamba2_ssd",
                     "state_space_dual", "selective_scan"):
        o = get_oracle(spelling)
        assert o is canonical, f"{spelling!r} did not resolve to the Mamba2 oracle"


def test_mamba2_oracle_reference_and_sim_agree():
    """Sequential state-recurrence reference vs materialized 1-semiseparable sim
    (the two sides of the state-space duality) agree — a non-vacuous parity
    check for the SSD scan."""
    o = get_oracle("Mamba2")
    assert o is not None and not o.vacuous
    inp = o.make_inputs()
    out = o.reference(inp)
    assert isinstance(out, np.ndarray) and np.all(np.isfinite(out))
    assert out.shape == (inp["x"].shape[0], inp["x"].shape[1])     # [T, P]
    assert np.array_equal(out, o.reference(o.make_inputs()))       # deterministic
    assert np.allclose(out, o.sim(inp), atol=1e-4)                 # recurrence == semisep


def test_mamba2_oracle_catches_dropped_decay():
    """Dropping the state decay collapses SSD into plain causal linear attention;
    the oracle must reject that (it pins the state-space part, not just
    causality)."""
    o = get_oracle("Mamba2")
    assert o is not None
    inp = o.make_inputs()
    x = inp["x"].astype(np.float64)
    B = inp["B"].astype(np.float64)
    C = inp["C"].astype(np.float64)
    m_nodecay = np.tril(C @ B.T)           # L = 1 everywhere (decay removed)
    y_bug = (m_nodecay @ x).astype(np.float32)
    assert not np.allclose(o.reference(inp), y_bug, atol=1e-4)


def test_mamba2_is_covered_by_audit():
    """audit_oracles no longer reports Mamba2 as a missing (uncovered)
    primitive."""
    assert "Mamba2" not in audit_oracles()["missing"]


# -- FlashAttention oracle (long-context dense attention) --------------------

def test_flash_alias_resolution():
    """The FlashAttention oracle is reachable via its canonical name, explicit
    aliases, and any PRIMITIVE_TO_KERNEL spelling that routes to it (flash,
    flash_attention, sliding_window_attention, gemma4_attention, ...)."""
    canonical = get_oracle("FlashAttention")
    if canonical is None:      # invent_kernels unavailable -> oracle not registered
        return
    assert canonical.name == "FlashAttention"
    for spelling in ("flash", "flashattn", "flash_attention", "FlashAttention",
                     "attention_long_context", "sliding_window_attention",
                     "gemma4_attention", "hetero_attention", "long_context_attention"):
        o = get_oracle(spelling)
        assert o is canonical, f"{spelling!r} did not resolve to the FlashAttention oracle"


def test_flash_oracle_reference_and_sim_agree():
    """Full-[S,S]-softmax reference vs streaming online-softmax (blocked running
    max/denom) sim — two independent algorithms for the same attention, so the
    parity check is non-vacuous and pins the online-softmax rescale."""
    o = get_oracle("FlashAttention")
    if o is None:
        return
    assert not o.vacuous
    inp = o.make_inputs()
    out = o.reference(inp)
    assert isinstance(out, np.ndarray) and np.all(np.isfinite(out))
    # reference is [S, d_head]; inputs are [d_head, S]
    assert out.shape == (inp["q"].shape[1], inp["q"].shape[0])
    assert np.array_equal(out, o.reference(o.make_inputs()))    # deterministic
    assert np.allclose(out, o.sim(inp), atol=1e-4)              # online == full


def test_flash_oracle_catches_a_truncated_context():
    """The oracle validates that the kernel attends to the WHOLE key sequence: an
    impl that drops half the K/V blocks (a real flash-blocking bug) diverges from
    the full-softmax reference, so the parity check rejects it."""
    o = get_oracle("FlashAttention")
    if o is None:
        return
    inp = o.make_inputs()
    q, k, v = inp["q"], inp["k"], inp["v"]         # each [d, S]
    half = k.shape[1] // 2
    scores = q.T @ k[:, :half]                     # attend to first half only
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    truncated = p @ v[:, :half].T
    assert not np.allclose(o.reference(inp), truncated, atol=1e-4)


def test_flash_is_covered_by_audit():
    o = get_oracle("FlashAttention")
    if o is None:
        return
    assert "FlashAttention" not in audit_oracles()["missing"]


# -- audit_oracles() as a CI gate (R17) --------------------------------------

def test_audit_oracles_is_a_ci_gate():
    """The audit is a real gate. Run it in a FRESH subprocess (so cross-test
    oracle registrations like the vacuous sentinel above cannot pollute the
    production invariant) and enforce:
      (1) ZERO vacuous production oracles — the orphan-oracle bug;
      (2) every uncovered kernel is consciously in KNOWN_UNCOVERED — a NEW
          primitive with no oracle fails CI until it is covered or listed;
      (3) KNOWN_UNCOVERED contains NO already-covered kernel — the allowlist can
          only shrink as oracles land.
    This is the exact check CI / preflight should run.
    """
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json, kernel_oracles as ko;"
        "rep = ko.audit_oracles();"
        "print(json.dumps({"
        "'vacuous': rep['vacuous'],"
        "'missing': rep['missing'],"
        "'known': sorted(ko.KNOWN_UNCOVERED),"
        "'covered': sorted({o.name for o in ko._ORACLES.values()}),"
        "}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)))
    assert proc.returncode == 0, f"audit subprocess failed: {proc.stderr}"
    data = json.loads(proc.stdout.strip().splitlines()[-1])

    # (1) no vacuous production oracles
    assert data["vacuous"] == [], f"vacuous oracles present: {data['vacuous']}"
    # (2) every uncovered kernel is consciously allowlisted
    unexpected = set(data["missing"]) - set(data["known"])
    assert not unexpected, (
        f"kernels missing an oracle and not in KNOWN_UNCOVERED: {sorted(unexpected)}")
    # (3) the allowlist only shrinks: no already-covered kernel is still listed
    stale = set(data["known"]) & set(data["covered"])
    assert not stale, f"KNOWN_UNCOVERED lists already-covered kernels: {sorted(stale)}"


def test_known_uncovered_shrank_by_flashattention():
    """Regression: FlashAttention was uncovered before R17; it must no longer be
    in the allowlist (the allowlist shrank when its oracle landed)."""
    from kernel_oracles import KNOWN_UNCOVERED
    assert "FlashAttention" not in KNOWN_UNCOVERED
