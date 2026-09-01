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
    #
    # 24 heads yields 8, not 24. tp must divide the head count AND be a power of two,
    # because the runtime cannot form a collective at any other world size (measured:
    # world 3/5/6/12/24 all fail with "Failed to execute the device barrier 2"). So 8
    # is the widest sharding this model can RUN, and predicting 24 would over-predict
    # the runner -- admitting models that then die at init.
    assert max_clean_tp(_dense(heads=24, arch="Qwen3_5ForConditionalGeneration"),
                        TRN2_48XLARGE) == 8
    assert max_clean_tp(_dense(heads=32, arch="Gemma4ForCausalLM"),
                        TRN2_48XLARGE) == 4
    # uncapped arch may shard as wide as heads and cores allow
    assert max_clean_tp(_dense(heads=32, arch="LlamaForCausalLM"),
                        TRN2_48XLARGE) == 32


def test_tp_is_the_largest_power_of_two_dividing_the_head_count():
    """Both invariants at once, across a range of awkward head counts.

    An earlier version asserted 2 for a 6-head model. #140 called that "really
    asserting powers of two only -- 6 shards 6 heads perfectly well" and changed it
    to assert `tp == heads`. The arithmetic in that argument is correct and the
    conclusion was still wrong: 6 shards 6 heads perfectly well AND the runtime
    cannot form a 6-way collective, so the original expectation of 2 was right.

    So assert the property rather than a number, and assert BOTH halves of it: tp
    divides the head count (or the worker rejects it) and tp is a power of two (or
    `init_process_group` fails with "Failed to execute the device barrier 2").
    Neither half alone is the contract.
    """
    for heads in (6, 10, 12, 14, 18, 20, 22, 24, 40, 48):
        tp = max_clean_tp(_dense(heads=heads, arch="LlamaForCausalLM"),
                          TRN2_48XLARGE)
        assert heads % tp == 0, (heads, tp)
        assert tp <= TRN2_48XLARGE.cores
        assert tp & (tp - 1) == 0, (heads, tp)          # power of two
        nxt = tp * 2                                    # and the LARGEST such
        assert heads % nxt != 0 or nxt > TRN2_48XLARGE.cores, (heads, tp)


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


# --- host DRAM: the constraint that actually binds on a 48xl -------------------
#
# Every expectation here is pinned to an observed trn2.48xlarge outcome. The host
# term was missing entirely, which is why the gate cleared two models that then
# OOM-killed the pod during load without ever touching a core.

def test_host_dram_binds_before_hbm_above_16_ranks():
    """The two budgets pull in OPPOSITE directions and cross over at 16 ranks.

    HBM per rank wants MORE ranks; host DRAM wants FEWER, because
    ``from_pretrained`` materialises a full private copy in every rank process
    before any sharding happens. Below the crossover HBM binds, above it the host
    does -- which is why "just raise tp" could never resolve the large MoE models.
    """
    from capability import host_load_peak_gb

    hw = TRN2_48XLARGE
    assert 8 * hw.usable_gb_per_core < hw.host_ram_gb / 8      # HBM is tighter
    assert 32 * hw.usable_gb_per_core > hw.host_ram_gb / 32     # host is tighter
    assert host_load_peak_gb(100.0, 16) == 1600.0              # ranks x model


def test_35b_a3b_fits_the_host_budget_at_tp16():
    """71.9 GB x 16 ranks = 1.15 TB of 2.1 TB -- the run that got past load."""
    from capability import host_load_peak_gb

    assert host_load_peak_gb(71.9, 16) < TRN2_48XLARGE.host_ram_gb


def test_122b_a10b_is_host_limited_not_hbm_limited():
    """The verdict must name the HOST, because HBM per rank is fine at tp=32.

    250 GB over 32 ranks is 7.8 GB of a 14.4 GB budget -- comfortable. But loading
    it needs 32 full copies = 8 TB against 2.1 TB of DRAM. Reporting this as
    TOO_LARGE would send the next person off to raise tp, which makes it worse.
    """
    v = assess(_moe(h=2048, L=40, moe_inter=512, experts=256, heads=32),
               TRN2_48XLARGE, weight_gb=250.2)
    assert not v.ok
    assert v.status == "HOST_LIMITED"
    assert "host DRAM" in v.reason
    assert "WORSE" in v.reason
    assert v.gb_per_rank < v.budget_gb_per_rank    # HBM was never the problem


def test_deepseek_v4_flash_host_limit_matches_the_observed_oomkill():
    """Observed: WorkloadInterrupted-137-OOMKilled, with no device error at all."""
    cfg = _moe(h=7168, L=61, moe_inter=2048, experts=256, heads=64)
    cfg["quantization_config"] = {"quant_method": "fp8"}
    v = assess(cfg, TRN2_48XLARGE, weight_gb=159.6)
    assert not v.ok
    assert v.status == "HOST_LIMITED"
    # The dequantized size is what drives the host peak, so it must be named.
    assert "dequantized to bf16" in v.reason


def test_a_lean_loader_removes_the_rank_multiplier():
    """Shard-on-read makes the peak model-sized instead of model x ranks."""
    from capability import host_load_peak_gb

    eager = host_load_peak_gb(250.2, 32)
    lean = host_load_peak_gb(250.2, 32, lean_loader=True)
    assert eager > TRN2_48XLARGE.host_ram_gb       # OOMs today
    assert lean < TRN2_48XLARGE.host_ram_gb        # would fit
    assert lean < eager / 10


def test_host_check_is_skipped_when_the_box_is_unmodelled():
    """Fail open. host_ram_gb is set only where it was measured off the box; an
    invented DRAM size would start rejecting models on no evidence."""
    from capability import HardwareProfile

    unknown = HardwareProfile("mystery", cores=64, hbm_gb_per_core=24.0)
    assert unknown.host_ram_gb == 0.0
    assert TRN2_3XLARGE.host_ram_gb == 0.0         # never measured
    v = assess(_moe(h=2048, L=40, moe_inter=512, experts=256, heads=32),
               unknown, weight_gb=250.2)
    assert v.status != "HOST_LIMITED"


# --- how big a model can this box actually run? -------------------------------

def test_ceiling_today_is_host_bound_and_says_so():
    from capability import ceiling

    c = ceiling(TRN2_48XLARGE)
    assert c["binding"] == "host-dram"
    assert c["ranks"] == 16                  # the crossover
    assert 130 <= c["weight_gb"] <= 140      # ~134 GB
    assert 60 <= c["params_b"] <= 70         # ~67B params in bf16


def test_fixing_the_loader_moves_the_ceiling_by_almost_7x():
    """The single highest-leverage change available for large models.

    With shard-on-read the host stops multiplying by the rank count and HBM
    becomes binding, which is the regime you want to be in.
    """
    from capability import ceiling

    today = ceiling(TRN2_48XLARGE)
    lean = ceiling(TRN2_48XLARGE, lean_loader=True)
    assert lean["binding"] == "hbm-per-rank"
    assert lean["ranks"] == 64
    assert lean["params_b"] > 6 * today["params_b"]
    assert 440 <= lean["params_b"] <= 480    # ~460B


def test_a_1t_model_does_not_fit_one_node_even_with_a_perfect_loader():
    """1T in bf16 is 2 TB of weights against 1.5 TB of HBM. Arithmetic, not tuning."""
    from capability import ceiling

    lean = ceiling(TRN2_48XLARGE, lean_loader=True)
    assert lean["weight_gb"] < 2000.0
    assert ceiling(TRN2_48XLARGE, lean_loader=True, node_count=2)["params_b"] \
        > lean["params_b"]


# --- staggered loading: the setting that makes the big two loadable today ------

def test_host_limited_verdict_hands_over_the_setting_that_fixes_it():
    """A diagnosis the next person can act on beats one they have to re-derive."""
    v = assess(_moe(h=2048, L=40, moe_inter=512, experts=256, heads=32),
               TRN2_48XLARGE, weight_gb=250.2)
    assert v.status == "HOST_LIMITED"
    assert "TRN_OPT_LOAD_CONCURRENCY=2" in v.reason
    assert v.details["host_peak_stagger2_gb"] < TRN2_48XLARGE.host_ram_gb


def test_staggering_is_modelled_and_beats_the_eager_peak():
    from capability import host_load_peak_gb

    eager = host_load_peak_gb(250.2, 32)
    stag = host_load_peak_gb(250.2, 32, concurrency=2)
    assert eager > TRN2_48XLARGE.host_ram_gb
    assert stag < TRN2_48XLARGE.host_ram_gb
    # Concurrency at or above the rank count is just the eager peak.
    assert host_load_peak_gb(250.2, 32, concurrency=32) == eager
    assert host_load_peak_gb(250.2, 32, concurrency=99) == eager


def test_staggering_and_shard_on_read_agree_on_direction_not_magnitude():
    """Both fit; shard-on-read is far cheaper, which is why it is the real fix."""
    from capability import host_load_peak_gb

    stag = host_load_peak_gb(319.2, 64, concurrency=2)
    lean = host_load_peak_gb(319.2, 64, lean_loader=True)
    assert stag < TRN2_48XLARGE.host_ram_gb
    assert lean < stag


# --- the gate must model the loader the run will actually use ------------------
#
# Without this the gate skips exactly the models staggering was built to rescue:
# 122B and DeepSeek are HOST_LIMITED under the default loader, so preflight_check
# would refuse them even with TRN_OPT_LOAD_CONCURRENCY=2 set on the run.

def _big_moe():
    return _moe(h=2048, L=40, moe_inter=512, experts=256, heads=32)


def test_122b_is_rejected_under_the_default_loader():
    v = assess(_big_moe(), TRN2_48XLARGE, weight_gb=250.2)
    assert not v.ok and v.status == "HOST_LIMITED"
    assert v.details["load_concurrency"] is None


def test_122b_is_accepted_once_staggering_is_configured():
    """734 GB of 2147 -- the run the gate must not block."""
    v = assess(_big_moe(), TRN2_48XLARGE, weight_gb=250.2, load_concurrency=2)
    assert v.ok, v.reason
    assert v.details["host_peak_gb"] < TRN2_48XLARGE.host_ram_gb
    assert v.details["load_concurrency"] == 2


def test_deepseek_is_accepted_once_staggering_is_configured():
    cfg = _moe(h=7168, L=61, moe_inter=2048, experts=256, heads=64)
    cfg["quantization_config"] = {"quant_method": "fp8"}
    v = assess(cfg, TRN2_48XLARGE, weight_gb=159.6, load_concurrency=2)
    assert v.ok, v.reason


def test_the_env_var_is_read_the_same_way_the_loader_reads_it(monkeypatch):
    """Gate and loader must never disagree about what the variable means."""
    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", "2")
    assert assess(_big_moe(), TRN2_48XLARGE, weight_gb=250.2).ok

    # A typo means "no staggering" in load_stagger, so the gate must reject again
    # rather than quietly assume a setting that will not be in effect.
    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", "two")
    assert not assess(_big_moe(), TRN2_48XLARGE, weight_gb=250.2).ok


def test_the_rejection_reason_names_the_loader_that_was_modelled(monkeypatch):
    """1200 GB fits the box's HBM but cannot be loaded even one rank at a time.

    At 64 ranks, concurrency=1 still peaks at 1200 + 63*(1200/64) = 2381 GB, because
    the 63 ranks outside the load window are each holding their own shard. So the
    verdict must say the model is unloadable HERE rather than suggest a setting that
    is already in effect.
    """
    monkeypatch.setenv("TRN_OPT_LOAD_CONCURRENCY", "1")
    v = assess(_moe(h=7168, L=92, moe_inter=2048, experts=256, heads=64),
               TRN2_48XLARGE, weight_gb=1200.0)
    assert not v.ok
    assert v.status == "HOST_LIMITED"
    assert "TRN_OPT_LOAD_CONCURRENCY=1" in v.reason
    assert v.details["load_concurrency"] == 1


# --- the gate must predict the tp the runner will actually use -----------------
#
# #140 made the search consider every divisor of the head count. While max_clean_tp
# still tried only powers of two it UNDER-predicted the runner and started rejecting
# models that fit -- the same class of bug as the original over-prediction, inverted.

def test_max_clean_tp_never_predicts_a_world_size_that_cannot_be_formed():
    """48 divides 48, and tp=48 still cannot be launched.

    #140 made this return 48 so the gate would stop rejecting MiniMax-M2. That is the
    wrong fix: the runtime only forms a collective at a power-of-two world size, so
    tp=48 dies in `init_process_group` before a single token is measured. Predicting it
    converts a clean rejection into a burned checkpoint load.
    """
    cfg = _moe(h=3072, L=62, moe_inter=1536, experts=256, heads=48,
               arch="MiniMaxM2ForCausalLM")
    tp = max_clean_tp(cfg, TRN2_48XLARGE)
    assert tp == 16                      # 1,2,4,8,16 divide 48; 32 does not
    assert tp & (tp - 1) == 0, "must be a power of two"


def test_a_24_head_model_stops_at_8_not_24():
    """The 56 idle cores on a 24-head model are reachable by REPLICAS, not by tp.

    tp=12 and tp=24 divide 24 evenly and neither can form a collective, so 8 is the
    honest ceiling. Qwen3.8-27B demonstrated this the expensive way: tp=3, 6, 12 and 24
    each loaded 55 GB of weights and then failed at `init_process_group`.
    """
    cfg = _dense(h=3072, L=64, heads=24, arch="Qwen3_5ForConditionalGeneration")
    assert max_clean_tp(cfg, TRN2_48XLARGE) == 8


def test_minimax_m2_does_not_fit_by_tensor_parallelism_alone():
    """460 GB over the 16 ranks it can actually form is 29 GB/rank -- over budget.

    The previous version asserted it fit, on the strength of tp=48 giving 9.6 GB/rank.
    That world size cannot be formed, so the number was never available. Recording the
    real verdict matters more than recording a pass: MiniMax-M2 needs expert parallelism
    or a second node, and a gate that says "fits" sends a band off to spend hours
    proving otherwise.
    """
    cfg = _moe(h=3072, L=62, moe_inter=1536, experts=256, heads=48,
               arch="MiniMaxM2ForCausalLM")
    v = assess(cfg, TRN2_48XLARGE, weight_gb=460.2, load_concurrency=1)
    assert v.chosen_tp == 16
    assert not v.ok and v.status == "TOO_LARGE", v.reason
    assert v.gb_per_rank > v.budget_gb_per_rank


def test_deltanet_value_heads_bound_the_predicted_tp():
    """Value heads cannot be replicated, so they are a hard ceiling on tp."""
    cfg = _moe(h=2048, L=40, moe_inter=512, experts=256, heads=64)
    cfg["text_config"]["linear_num_value_heads"] = 8
    assert max_clean_tp(cfg, TRN2_48XLARGE) == 8


def test_powers_of_two_models_are_unchanged():
    """The common case must not move."""
    for heads, expect in ((16, 16), (32, 32), (64, 64), (8, 8)):
        cfg = _dense(h=4096, L=32, heads=heads)
        assert max_clean_tp(cfg, TRN2_48XLARGE) == expect, heads


def test_the_gemma4_hard_cap_still_binds():
    cfg = _dense(h=4096, L=48, heads=32, arch="Gemma4ForConditionalGeneration")
    assert max_clean_tp(cfg, TRN2_48XLARGE) == 4


def test_num_local_experts_counted_as_moe():
    """Qwen3-MoE configs serialize the expert count as `num_local_experts`
    (Mixtral-style) through `to_dict`, not `num_experts`. estimate_params must
    still count it as MoE -- else a 30B MoE is sized as a ~3B dense model and the
    baseline chooser puts a 60GB model on one core (tp=1) and OOMs."""
    from capability import estimate_params
    cfg = dict(hidden_size=2048, num_hidden_layers=48, vocab_size=151936,
               intermediate_size=6144, moe_intermediate_size=768,
               num_local_experts=128, num_experts_per_tok=8)
    params, bd = estimate_params(cfg)
    assert bd["is_moe"] is True
    assert bd["num_experts"] == 128
    assert params > 25e9  # ~30B, not the ~3B dense misread
