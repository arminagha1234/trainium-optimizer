# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for perf_hints.py — the perf-symptom -> specific-NKI-Guide-Opt map, and
its wiring into kernel_perf + the neuron_profile / roofline threshold bars."""

from __future__ import annotations

import perf_hints
import neuron_profile as np_mod
import roofline
import kernel_perf
import nki_knowledge


# --- perf_hints: matching ---------------------------------------------------
def test_short_matmul_routes_to_fast_weight_load():
    hay = perf_hints.symptoms_from("compute_bound", op_name="attn_decode",
                                   op_family="matmul", profile_tokens=("low-mfu",))
    hits = perf_hints.match_perf_hints(hay)
    keys = [h.key for h in hits]
    assert "fast-weight-load-matvec" in keys
    assert hits[0].opt == "Opt #7"      # most specific first


def test_scan_routes_to_tensor_tensor_scan():
    hay = perf_hints.symptoms_from("single_engine", op_name="mamba_scan",
                                   op_family="scan")
    keys = [h.key for h in perf_hints.match_perf_hints(hay)]
    assert "tensor-tensor-scan-perelement" in keys


def test_dma_small_transfer_routes_to_large_dma():
    hay = perf_hints.symptoms_from("dma_blocked", profile_tokens=("small-dma", "low-mbu"))
    keys = [h.key for h in perf_hints.match_perf_hints(hay)]
    assert "large-dma-transfers" in keys


def test_spill_routes_to_fusion():
    hay = perf_hints.symptoms_from("memory_bound", profile_tokens=("spill-high",))
    keys = [h.key for h in perf_hints.match_perf_hints(hay)]
    assert "fuse-spill" in keys


def test_no_symptom_no_match():
    assert perf_hints.match_perf_hints("") == []
    # an unrelated string should not cross-fire into a hint
    assert perf_hints.match_perf_hints("everything is perfectly fine") == []


def test_format_caps_at_max_hints():
    # a haystack that hits several hints still renders at most max_hints blocks
    hay = perf_hints.symptoms_from("dma_blocked", op_name="x_decode",
                                   op_family="matmul",
                                   profile_tokens=("small-dma", "low-mfu", "spill-high"))
    hits = perf_hints.match_perf_hints(hay)
    assert len(hits) >= 3
    rendered = perf_hints.format_perf_hints(hits, max_hints=2)
    assert rendered.count(">>> PERF") == 2


def test_guidance_from_symptoms_returns_opt_prefixed_strings():
    g = perf_hints.guidance_from_symptoms("compute_bound", op_name="qkv_decode",
                                          op_family="matmul",
                                          profile_tokens=("low-mfu",))
    assert g and g[0].startswith("Opt #7")


# --- perf_hints keys stay DRY with nki_knowledge.TECHNIQUES -----------------
def test_every_hint_technique_key_exists_in_techniques():
    for h in perf_hints.HINTS:
        if h.technique:
            assert h.technique in nki_knowledge.TECHNIQUES, (
                f"perf_hint {h.key} references unknown technique {h.technique!r}")


def test_new_guide_techniques_are_wired_into_knowledge():
    # the four NKI-Guide techniques must be attached to at least one op family
    wired = set()
    for entry in nki_knowledge.KNOWLEDGE.values():
        wired.update(entry.techniques)
    for k in ("fast-weight-load", "partition-vectorize", "tensor-tensor-scan",
              "transpose-swap-for-layout"):
        assert k in nki_knowledge.TECHNIQUES
        assert k in wired, f"{k} not wired into any KNOWLEDGE entry"


# --- neuron_profile: metrics + perf_symptoms --------------------------------
def test_summarize_carries_metrics_and_appends_breach_note():
    rep = np_mod.summarize({"pe": 0.2, "dma": 0.7}, 100.0, mfu=0.5, mbu=0.4,
                           spill_ratio=0.5, dma_transfer_kib=4.0)
    assert rep.mfu == 0.5 and rep.mbu == 0.4
    assert rep.spill_ratio == 0.5 and rep.dma_transfer_kib == 4.0
    # breaches surfaced in reason (tokens perf_hints/classify greps)
    assert "spill-high" in rep.reason
    assert "small-dma" in rep.reason
    assert "low-mfu" in rep.reason


def test_perf_symptoms_tokens():
    rep = np_mod.summarize({"pe": 0.2, "dma": 0.7}, 0.0, mfu=0.5, mbu=0.4,
                           spill_ratio=0.5, dma_transfer_kib=4.0)
    toks = set(np_mod.perf_symptoms(rep))
    assert "dma-blocked" in toks
    assert {"low-mfu", "low-mbu", "spill-high", "small-dma"} <= toks
    # every emitted token is in the shared vocabulary
    assert toks <= set(perf_hints.SYMPTOM_TOKENS)


def test_perf_symptoms_empty_for_unmeasured():
    assert np_mod.perf_symptoms(np_mod.summarize({})) == ()


def test_good_metrics_emit_no_breach():
    rep = np_mod.summarize({"pe": 0.95}, 0.0, mfu=0.95, mbu=0.7,
                           spill_ratio=0.1, dma_transfer_kib=64.0)
    toks = set(np_mod.perf_symptoms(rep))
    assert "low-mfu" not in toks and "low-mbu" not in toks
    assert "spill-high" not in toks and "small-dma" not in toks


def test_parse_explorer_metrics_from_summary():
    obj = {"n_abc": {
        "tensor_engine_active_time_percent": 0.2,
        "dma_active_time_percent": 0.8,
        "mfu_estimated_percent": 0.45,
        "mbu_estimated_percent": 0.55,
        "spill_save_bytes": 300, "spill_reload_bytes": 200,
        "sb_read_bytes": 1000, "sb_write_bytes": 1000,
        "dma_total_bytes": 65536, "dma_transfer_count": 8,
    }}
    m = np_mod.parse_neuron_explorer_metrics(obj)
    assert abs(m["mfu"] - 0.45) < 1e-9
    assert abs(m["spill_ratio"] - (500 / 2000)) < 1e-9
    assert abs(m["dma_transfer_kib"] - (65536 / 8 / 1024)) < 1e-9  # 8 KiB


# --- roofline: per-bottleneck bars ------------------------------------------
def test_good_sol_bar_per_bottleneck():
    assert roofline.good_sol_bar("compute_bound") == roofline.COMPUTE_BOUND_GOOD_SOL
    assert roofline.good_sol_bar("memory_bound") == roofline.MEMORY_BOUND_GOOD_SOL
    # memory-bound op judged good at 60%, compute-bound NOT
    assert roofline.meets_good_bar(0.65, "memory_bound")
    assert not roofline.meets_good_bar(0.65, "compute_bound")
    assert roofline.meets_good_bar(0.92, "compute_bound")


# --- kernel_perf: diagnose routes to specific guidance ----------------------
class _Race:
    def __init__(self, bottleneck="", reason="", sol=0.0, mfu=-1.0,
                 roofline_ratio=0.0):
        self.bottleneck = bottleneck
        self.reason = reason
        self.sol = sol
        self.mfu = mfu
        self.roofline_ratio = roofline_ratio


def test_diagnose_uses_specific_guidance_for_decode_matmul():
    loop = kernel_perf.KernelPerfLoop(op_name="attn_decode", op_family="matmul")
    race = _Race(bottleneck="compute_bound", reason="low-mfu 40%")
    bn, guidance = loop.diagnose(race)
    assert bn == kernel_perf.SINGLE_ENGINE  # compute_bound+slow -> single_engine
    # the specific Opt #7 guidance should be surfaced, not just the coarse lever
    assert any("Opt #7" in g for g in guidance)


def test_diagnose_falls_back_to_coarse_when_nothing_specific():
    loop = kernel_perf.KernelPerfLoop(op_name="softcap", op_family="elementwise")
    race = _Race(bottleneck="memory_bound", reason="")
    bn, guidance = loop.diagnose(race)
    # memory_bound with a spill-y op still yields the fusion lever (specific or coarse)
    assert guidance
    assert bn == kernel_perf.MEMORY_BOUND


def test_converged_honors_per_bottleneck_bar():
    loop = kernel_perf.KernelPerfLoop()
    # memory-bound at 62% SOL is CONVERGED (>= 60% bar) even though mfu unknown
    assert loop._converged(_Race(bottleneck="memory_bound", sol=0.62))
    # compute-bound at 62% is NOT converged (< 90% bar)
    assert not loop._converged(_Race(bottleneck="compute_bound", sol=0.62))
