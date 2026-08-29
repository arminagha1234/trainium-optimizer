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


def test_35b_a3b_that_oomed_is_rejected():
    v = assess(_moe(), TRN2_48XLARGE, weight_gb=71.9)
    assert not v.ok
    assert v.status == "TOO_LARGE"
    assert "tp>=" in v.reason and "capped" in v.reason


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
    assert max_clean_tp(_dense(heads=24, arch="Qwen3_5ForConditionalGeneration"),
                        TRN2_48XLARGE) == 4
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
