"""Capability gate tests.

The expectations here are pinned to OBSERVED trn2.48xlarge outcomes, not to
whatever the implementation happens to produce, so a regression that starts
rejecting a model we have actually run will fail the suite.
"""
from __future__ import annotations

from capability import (
    TRN2_3XLARGE,
    TRN2_48XLARGE,
    assess,
    estimate_params,
    max_clean_tp,
)


def _dense(h=2048, L=24, inter=8192, vocab=151936, heads=16, arch="LlamaForCausalLM"):
    return {"architectures": [arch],
            "hidden_size": h, "num_hidden_layers": L, "intermediate_size": inter,
            "vocab_size": vocab, "num_attention_heads": heads}


def _moe(h=2048, L=40, moe_inter=512, experts=256, vocab=151936, heads=16,
         arch="Qwen3_5MoeForConditionalGeneration"):
    return {"architectures": [arch],
            "text_config": {"hidden_size": h, "num_hidden_layers": L,
                            "moe_intermediate_size": moe_inter,
                            "intermediate_size": moe_inter,
                            "num_experts": experts, "vocab_size": vocab,
                            "num_attention_heads": heads}}


# --- the bug this module exists to fix ---------------------------------------

def test_moe_experts_are_counted():
    """A 256-expert MoE must not be sized with a dense formula.

    The dense formula native_pytorch._fit_baseline_tp uses undercounts
    Qwen3.5-35B-A3B as 7.4 GB when it is really ~72 GB, which is why the backend
    put it on one core and OOM'd.
    """
    params, bd = estimate_params(_moe())
    assert bd["is_moe"] is True
    # expert MLPs must dominate a 256-expert model
    assert bd["mlp_params"] > 5 * bd["attn_params"]
    weight_gb = params * 2 / 1e9
    assert 55 < weight_gb < 90, weight_gb          # real weights are 71.9 GB


def test_dense_and_moe_of_same_shape_differ_by_expert_count():
    dense_params, _ = estimate_params(_dense(h=2048, L=40, inter=512))
    moe_params, _ = estimate_params(_moe(h=2048, L=40, moe_inter=512, experts=256))
    assert moe_params > 8 * dense_params


# --- observed outcomes -------------------------------------------------------

def test_27b_that_ran_is_not_rejected():
    """Qwen3.8-27B ran and was grader-verified at 344 tok/s.

    A gate that rejects it is worse than no gate, so this is the regression that
    matters most: TIGHT is acceptable, TOO_LARGE is not.
    """
    v = assess(_dense(h=5120, L=64, inter=25600, heads=24,
                      arch="Qwen3_5ForConditionalGeneration"),
               TRN2_48XLARGE, weight_gb=56.0)
    assert v.ok, v.reason
    assert v.status in ("TIGHT", "RUNNABLE")


def test_35b_a3b_is_rejected_when_tp_is_constrained_low():
    """The observed OOM happened AT tp=4, so assert the physics, not the era.

    35B-A3B is ~72 GB. Squeezed onto 4 ranks that is 18 GB/rank, past the budget,
    which is the OOM we actually saw. On a 4-core box tp cannot exceed 4, so this
    reproduces that condition exactly rather than relying on a stale arch cap.
    """
    v = assess(_moe(), TRN2_3XLARGE, weight_gb=71.9)
    assert not v.ok
    assert v.status == "TOO_LARGE"
    assert v.chosen_tp <= 4
    assert "tp>=" in v.reason


def test_35b_a3b_proceeds_once_tp_can_reach_8():
    """#121 raised the Qwen3_5 cap to the head count; the gate must agree.

    72 GB over >=8 ranks is <=9 GB/rank, inside budget. Rejecting it here would
    make the gate block a model the runner is capable of running.
    """
    v = assess(_moe(), TRN2_48XLARGE, weight_gb=71.9)
    assert v.ok, v.reason
    assert v.chosen_tp >= 8


def test_model_larger_than_the_whole_node_needs_multinode():
    v = assess(_moe(h=7168, L=92, moe_inter=2048, experts=256),
               TRN2_48XLARGE, weight_gb=1560.9)
    assert not v.ok
    assert v.status == "NEEDS_MULTINODE"
    assert "nodeCount>1" in v.reason


def test_small_model_is_runnable():
    v = assess(_dense(h=1024, L=24, inter=3072), TRN2_48XLARGE, weight_gb=1.5)
    assert v.ok and v.status == "RUNNABLE"


# --- fail-open contract ------------------------------------------------------

def test_unsizeable_config_fails_open():
    """Blocking a model we cannot size would silently shrink the reachable set."""
    v = assess({"architectures": ["MysteryForCausalLM"]}, TRN2_48XLARGE)
    assert v.ok and v.status == "UNKNOWN"


def test_measured_size_overrides_a_wrong_estimate():
    """The estimate is 3.5x high on DeepSeek-V4-Flash; the measured size wins."""
    cfg = _moe(h=4096, L=43, moe_inter=2048, experts=256,
               arch="DeepseekV4ForCausalLM")
    est = assess(cfg, TRN2_48XLARGE)
    meas = assess(cfg, TRN2_48XLARGE, weight_gb=159.6)
    assert meas.weight_gb == 159.6
    assert meas.details["weight_source"] == "measured"
    assert est.details["weight_source"] == "config-estimate"
    assert meas.weight_gb < est.weight_gb


# --- tp selection must mirror the backend -----------------------------------

def test_tp_cap_matches_the_adapter_limits():
    # Qwen3_5 is bounded by the head count, matching native_pytorch after #121.
    # A stale cap of 4 here would make the gate reject models the runner runs.
    assert max_clean_tp(_dense(heads=24, arch="Qwen3_5ForConditionalGeneration"),
                        TRN2_48XLARGE) == 8      # 24 heads: 8 divides, 16 does not
    assert max_clean_tp(_dense(heads=32, arch="Gemma4ForCausalLM"),
                        TRN2_48XLARGE) == 4
    # uncapped arch may shard as wide as heads and cores allow
    assert max_clean_tp(_dense(heads=32, arch="LlamaForCausalLM"),
                        TRN2_48XLARGE) == 32


def test_tp_must_divide_head_count():
    assert max_clean_tp(_dense(heads=6, arch="LlamaForCausalLM"),
                        TRN2_48XLARGE) == 2


def test_tp_never_exceeds_physical_cores():
    assert max_clean_tp(_dense(heads=64, arch="LlamaForCausalLM"),
                        TRN2_3XLARGE) <= TRN2_3XLARGE.cores


def test_smaller_box_is_stricter():
    cfg = _dense(h=5120, L=64, inter=25600, heads=24, arch="LlamaForCausalLM")
    big = assess(cfg, TRN2_48XLARGE, weight_gb=56.0)
    small = assess(cfg, TRN2_3XLARGE, weight_gb=56.0)
    assert small.chosen_tp <= big.chosen_tp
    assert small.gb_per_rank >= big.gb_per_rank


def test_verdict_is_truthy_by_ok():
    assert bool(assess(_dense(), TRN2_48XLARGE, weight_gb=1.0)) is True
    assert bool(assess(_moe(), TRN2_48XLARGE, weight_gb=500.0)) is False


# --- quantized checkpoints expand at load ------------------------------------

def test_fp8_measured_size_is_scaled_to_bf16():
    """DeepSeek-V4-Flash ships 159.6 GB of fp8; Neuron dequantizes it to bf16.

    Sizing off the download under-counts it 2x and calls an infeasible model
    runnable, which is exactly what happened before dequant_factor existed.
    """
    cfg = _moe(h=4096, L=43, moe_inter=2048, experts=256,
               arch="DeepseekV4ForCausalLM")
    cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3",
                                  "weight_block_size": [128, 128]}
    v = assess(cfg, TRN2_48XLARGE, weight_gb=159.6)
    assert v.details["on_disk_gb"] == 159.6
    assert v.details["dequant_factor"] == 2.0
    assert abs(v.weight_gb - 319.2) < 0.5, v.weight_gb
    assert "dequantized to bf16" in v.reason


def test_unquantized_measured_size_is_untouched():
    v = assess(_moe(), TRN2_48XLARGE, weight_gb=71.9)
    assert v.details["dequant_factor"] == 1.0
    assert v.weight_gb == 71.9


def test_int4_expands_fourfold():
    cfg = _dense(arch="LlamaForCausalLM")
    cfg["quantization_config"] = {"quant_method": "awq"}
    v = assess(cfg, TRN2_48XLARGE, weight_gb=10.0)
    assert v.details["dequant_factor"] == 4.0
    assert v.weight_gb == 40.0


def test_unknown_quant_method_does_not_invent_a_factor():
    cfg = _dense(arch="LlamaForCausalLM")
    cfg["quantization_config"] = {"quant_method": "some_future_scheme"}
    v = assess(cfg, TRN2_48XLARGE, weight_gb=10.0)
    assert v.details["dequant_factor"] == 1.0   # fail open, do not guess
    assert v.weight_gb == 10.0


def test_config_estimate_is_not_double_counted():
    """The config estimate is already in compute dtype; it must not be scaled."""
    cfg = _moe()
    cfg["quantization_config"] = {"quant_method": "fp8"}
    v = assess(cfg, TRN2_48XLARGE)               # no measured size
    assert v.details["dequant_factor"] == 1.0


# --- instance-type resolution + metadata sizing -------------------------------

def test_profile_for_known_instances():
    from capability import profile_for
    assert profile_for("trn2.48xlarge").cores == 64
    assert profile_for("trn2.3xlarge").cores == 4
    assert profile_for("trn1.32xlarge").hbm_gb_per_core == 16.0


def test_profile_for_unknown_returns_none_so_the_gate_is_skipped():
    """Modelling the wrong box is worse than not gating at all.

    Too little HBM rejects models that run; too much passes models that cannot.
    None is the signal to skip, which preflight_check honours.
    """
    from capability import profile_for
    assert profile_for("p5e.48xlarge") is None
    assert profile_for("") is None
    assert profile_for(None) is None


def test_measured_weight_gb_is_best_effort():
    """A metadata lookup must never fail a run -- it returns None instead."""
    from capability import measured_weight_gb
    assert measured_weight_gb("definitely/not-a-real-model-xyz", timeout=3) is None


def test_gate_is_skipped_when_hardware_is_none():
    """The wiring must degrade to today's behaviour on an unmodelled box."""
    from preflight import preflight_check
    from orchestrator import ModelSpec
    huge = _moe(h=7168, L=92, moe_inter=2048, experts=256)
    spec = ModelSpec(model_id="X/huge", family="moe_causal_lm", param_count=1e12)
    ok_gated, why = preflight_check(spec, config=huge, hardware=TRN2_48XLARGE,
                                   weight_gb=1560.9, kernels_wired=True,
                                   rewrites_wired=True)
    ok_ungated, _ = preflight_check(spec, config=huge, hardware=None,
                                    kernels_wired=True, rewrites_wired=True)
    assert ok_gated is False and "capability:" in (why or "")
    assert ok_ungated is True


def test_gate_tp_tracks_the_runner_for_qwen35_moe():
    """The gate must not model a narrower tp than _fit_baseline_tp will pick.

    Pinned to observed hardware: 35B-A3B runs at tp=8 (~9 GB/rank), and
    122B-A10B OOM'd at 22.5 GB of a 24 GB core even at tp=16.
    """
    m35 = _moe(h=2048, L=40, moe_inter=512, experts=256, heads=16)
    assert max_clean_tp(m35, TRN2_48XLARGE) == 16          # bounded by heads
    v35 = assess(m35, TRN2_48XLARGE, weight_gb=71.9)
    assert v35.ok, v35.reason                              # was wrongly TOO_LARGE
    assert v35.chosen_tp >= 8

    m122 = _moe(h=3072, L=48, moe_inter=1024, experts=256, heads=16)
    v122 = assess(m122, TRN2_48XLARGE, weight_gb=250.2)
    assert not v122.ok                                     # matches the real OOM
    assert v122.chosen_tp == 16
