"""Capability gate — decide from the CONFIG ALONE whether a model can run here.

Why this exists
---------------
The north star is "point it at any HuggingFace model". In practice that meant
*attempting* any model and discovering the answer on hardware, which is the
expensive way to be told no. Three 256-expert MoE models each burned a
trn2.48xlarge and returned an indistinguishable ``metric=0.0``.

They were not close calls. Two of them cannot fit at any TP this backend will
choose, and that is provable from ``config.json`` in milliseconds — no weights
downloaded, no compile, no device touched.

The load-bearing fix: MoE parameter counting
--------------------------------------------
``native_pytorch._fit_baseline_tp`` sizes a model with a DENSE formula::

    params = (4*h*h + 3*h*inter) * L + 2*V*h

That ignores the expert dimension entirely, so for a 256-expert MoE it
undercounts by an order of magnitude (measured against on-disk weights):

    Qwen3.5-35B-A3B      dense  7.4 GB   MoE-aware  67.8 GB   actual  71.9 GB
    Qwen3.5-122B-A10B    dense 17.5 GB   MoE-aware 238.6 GB   actual 250.2 GB

A 7.4 GB estimate satisfies the "keep weights under ~10 GB/rank" rule at tp=1,
so the backend puts a 71.9 GB model on a single 24 GB core. The OOM was decided
before the run started.

Design notes
------------
- Pure and dependency-free: takes a config dict, returns a verdict. No torch, no
  transformers, no network, so it is trivially testable and cannot itself fail
  on an unsupported model.
- Fails OPEN. When the config is too unusual to size confidently, the verdict is
  ``UNKNOWN`` and the caller proceeds. A gate that blocks a model that would
  have worked is a worse failure than one that lets a doomed model through: the
  first silently shrinks the reachable model set, the second costs one run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Verdict",
    "HardwareProfile",
    "TRN2_48XLARGE",
    "TRN2_3XLARGE",
    "estimate_params",
    "max_clean_tp",
    "assess",
    "profile_for",
    "measured_weight_gb",
    "host_load_peak_gb",
    "ceiling",
]

# Bytes per parameter by dtype string.
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "float16": 2, "bfloat16": 2,
                "fp32": 4, "float32": 4, "fp8": 1, "int8": 1, "int4": 0.5}

# Architectures whose clean TP is capped below head-count by the adapter. Mirrors
# native_pytorch._fit_baseline_tp so the gate predicts the SAME tp the backend
# will actually pick -- a gate that models different hardware than the runner is
# worse than no gate.
# None means "no cap beyond the head count", which the `heads % tp` test already
# enforces. MUST stay in lockstep with native_pytorch._fit_baseline_tp: a gate that
# models a different tp than the runner picks is worse than no gate, because it
# either rejects models that run or passes models that cannot.
_TP_CAPS: tuple[tuple[str, int | None], ...] = (
    ("Gemma4", 4),      # HARD: Global layers use head_dim 512 with only 4 KV heads
    ("Qwen3_5", None),  # was 4 (a validation limit); #121 raised the runner to heads
)


@dataclass(frozen=True)
class HardwareProfile:
    """What a box actually offers.

    ``hbm_gb_per_core`` bounds the model AFTER sharding. ``host_ram_gb`` bounds it
    DURING THE LOAD, and on a 48xl that is the wall you hit FIRST -- see
    ``host_load_peak_gb``.
    """

    name: str
    cores: int
    hbm_gb_per_core: float
    # Total host DRAM in decimal GB, measured from /proc/meminfo where possible.
    # trn2.48xlarge: MemTotal = 2_097_112_352 kB = 2147 GB (read off the box
    # 2026-08-29). 0.0 means unmodelled, which SKIPS the host check rather than
    # guessing a number that could block a model that would have run.
    host_ram_gb: float = 0.0
    # Fraction of a core's HBM the weights may occupy; the rest is activations,
    # KV cache and compiler scratch.
    #
    # CALIBRATED against observed trn2.48xlarge outcomes, not guessed. The
    # backend's own comment suggests ~10 GB on a 24 GB core (0.42), but that is
    # demonstrably too strict:
    #
    #   Qwen3.8-27B      13.3 GB/rank at tp=4  -> RAN, grader-verified 344 tok/s
    #   Qwen3.5-35B-A3B  17.0 GB/rank at tp=4  -> OOM
    #
    # 0.60 (14.4 GB on a 24 GB core) is the only budget that puts the observed
    # pass on one side and the observed OOM on the other. A stricter value would
    # have rejected a model that works, which is the failure this gate must not
    # make. Anything above `tight_frac` still runs but is reported as tight.
    weight_budget_frac: float = 0.60
    tight_frac: float = 0.42

    @property
    def usable_gb_per_core(self) -> float:
        return self.hbm_gb_per_core * self.weight_budget_frac

    @property
    def comfortable_gb_per_core(self) -> float:
        return self.hbm_gb_per_core * self.tight_frac

    @property
    def total_hbm_gb(self) -> float:
        return self.hbm_gb_per_core * self.cores


# LNC=2: 16 devices x 4 logical cores, 24 GB per logical core.
# host_ram_gb is set ONLY where it was read off the box. The others are left
# unmodelled (0.0) so the host check is skipped rather than run against a guess:
# an invented DRAM size would start rejecting models on no evidence.
TRN2_48XLARGE = HardwareProfile("trn2.48xlarge", cores=64, hbm_gb_per_core=24.0,
                                host_ram_gb=2147.0)   # /proc/meminfo, 2026-08-29
TRN2_3XLARGE = HardwareProfile("trn2.3xlarge", cores=4, hbm_gb_per_core=24.0)
TRN1_32XLARGE = HardwareProfile("trn1.32xlarge", cores=32, hbm_gb_per_core=16.0)
TRN1_2XLARGE = HardwareProfile("trn1.2xlarge", cores=2, hbm_gb_per_core=16.0)

_PROFILES = {p.name: p for p in
             (TRN2_48XLARGE, TRN2_3XLARGE, TRN1_32XLARGE, TRN1_2XLARGE)}


def profile_for(instance_type: str | None) -> HardwareProfile | None:
    """HardwareProfile for an instance-type string, or None if unmodelled.

    None is the signal to SKIP the gate rather than guess. Modelling the wrong
    box is worse than not gating: a profile with too little HBM would reject
    models that run, and one with too much would pass models that cannot.
    """
    return _PROFILES.get((instance_type or "").strip())


def measured_weight_gb(model_id: str, timeout: float = 6.0) -> float | None:
    """Total weight bytes from the HF repo's file metadata, in GB, or None.

    Metadata only -- no weights are downloaded. This beats the config estimate,
    which is a model of the architecture and is wrong by 3.5x on DeepSeek-V4-Flash
    and 7x on Kimi-K3. Best-effort by design: any failure (offline, private repo,
    rate limit) returns None so the caller falls back to the estimate rather than
    failing a run over a metadata lookup.
    """
    import json
    import urllib.request
    try:
        url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
        total = sum(f.get("size") or 0 for f in d.get("siblings", [])
                    if str(f.get("rfilename", "")).endswith((".safetensors", ".bin")))
        return total / 1e9 if total else None
    except Exception:  # noqa: BLE001 - advisory only
        return None


@dataclass
class Verdict:
    """Outcome of a capability assessment.

    ``ok`` False means do not spend a run. ``reason`` is written to be pasted
    into a ticket: it carries the numbers, not just a category.
    """

    ok: bool
    status: str                     # RUNNABLE | TIGHT | TOO_LARGE | HOST_LIMITED
                                    # | NEEDS_MULTINODE | UNKNOWN
    reason: str = ""
    params: int = 0
    weight_gb: float = 0.0
    chosen_tp: int = 0
    gb_per_rank: float = 0.0
    budget_gb_per_rank: float = 0.0
    min_tp_needed: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # so `if assess(...):` reads naturally
        return self.ok


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal wrappers nest the LM config under ``text_config``."""
    tc = config.get("text_config")
    return tc if isinstance(tc, dict) else config


def _int(cfg: dict[str, Any], key: str, default: int = 0) -> int:
    v = cfg.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def architectures(config: dict[str, Any]) -> list[str]:
    a = (config or {}).get("architectures")
    return [str(x) for x in a] if isinstance(a, list) else []


def estimate_params(config: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Estimate total parameters, counting MoE experts.

    Returns ``(params, breakdown)``. ``params == 0`` means the config could not
    be sized (missing hidden_size/num_hidden_layers) -- callers must treat that
    as UNKNOWN and proceed, never as "small".
    """
    tc = _text_config(config)
    h = _int(tc, "hidden_size")
    L = _int(tc, "num_hidden_layers")
    if not h or not L:
        return 0, {"reason": "config lacks hidden_size / num_hidden_layers"}

    vocab = _int(tc, "vocab_size", 151936)
    inter = _int(tc, "intermediate_size", 4 * h)
    n_exp = _int(tc, "num_experts") or _int(tc, "n_routed_experts")
    n_shared = _int(tc, "n_shared_experts") or _int(tc, "num_shared_experts")
    moe_inter = _int(tc, "moe_intermediate_size", inter)

    # Attention + norms, per layer. 4*h*h covers q/k/v/o at full width; GQA makes
    # this a slight overestimate, which is the safe direction for a gate.
    attn = 4 * h * h * L
    if n_exp:
        # Every expert carries its own gate/up/down (3 matrices of h x moe_inter).
        # This is the term the dense formula drops.
        mlp = 3 * h * moe_inter * (n_exp + n_shared) * L
    else:
        mlp = 3 * h * inter * L
    embed = 2 * vocab * h          # input embedding + lm_head
    params = attn + mlp + embed
    return params, {
        "hidden_size": h, "layers": L, "vocab": vocab,
        "intermediate_size": inter, "moe_intermediate_size": moe_inter,
        "num_experts": n_exp, "num_shared_experts": n_shared,
        "attn_params": attn, "mlp_params": mlp, "embed_params": embed,
        "is_moe": bool(n_exp),
    }


# Host DRAM held by ONE rank while it streams a checkpoint shard-by-shard and
# keeps only its own slice. One safetensors file is typically 4-5 GB.
_LEAN_STREAM_GB = 5.0


def host_load_peak_gb(
    weight_gb: float,
    ranks: int,
    *,
    lean_loader: bool = False,
    stream_gb: float = _LEAN_STREAM_GB,
    concurrency: int | None = None,
) -> float:
    """Peak HOST DRAM, summed over every rank on the node, during model load.

    This is the constraint the HBM math misses, and on a trn2.48xlarge it binds
    long before HBM does.

    ``AutoModelForCausalLM.from_pretrained`` materialises the WHOLE model in the
    calling process. Tensor parallelism runs one process per core, and sharding
    happens only AFTER the model object exists, so the transient peak is
    ``ranks * weight_gb`` -- every rank holds a full private copy at the same
    moment. Nothing about a bigger tp helps; tp makes it strictly worse:

        Qwen3.5-35B-A3B    72 GB x 16 ranks =  1.2 TB of 2.1 TB   survives
        Qwen3.5-122B-A10B 250 GB x 32 ranks =  8.0 TB             OOMKilled
        DeepSeek-V4-Flash 319 GB x 64 ranks = 20.4 TB             OOMKilled (137)

    The two budgets therefore pull in OPPOSITE directions -- HBM per rank wants
    more ranks, host DRAM wants fewer -- which is why "try a bigger tp" could
    never resolve these models.

    ``concurrency=N`` models ``TRN_OPT_LOAD_CONCURRENCY`` (backends/load_stagger.py):
    only N ranks sit in the load window at once, so the peak drops to
    ``N*W + (ranks-N)*W/ranks``. This is what makes 122B and DeepSeek loadable today,
    at the cost of ``ceil(ranks/N)`` sequential load waves.

    ``lean_loader=True`` models a shard-on-read loader: build on ``meta``, stream
    one file at a time, slice this rank's portion, release the rest. Peak becomes
    the resident sharded weights plus one file in flight per rank, so the rank
    count stops multiplying the model size. Faster than staggering, more work to
    build.
    """
    ranks = max(1, ranks)
    if lean_loader:
        return weight_gb + ranks * stream_gb
    if concurrency is not None and concurrency < ranks:
        # Staggered: only `concurrency` ranks hold a full copy; the rest already
        # hold just their shard. See backends/load_stagger.py.
        c = max(1, concurrency)
        return c * weight_gb + (ranks - c) * (weight_gb / ranks)
    return ranks * weight_gb


def ceiling(
    hw: HardwareProfile = TRN2_48XLARGE,
    *,
    lean_loader: bool = False,
    bytes_per_param: float = 2.0,
    node_count: int = 1,
) -> dict[str, Any]:
    """Largest model ``hw`` can run, and which constraint binds.

    Sweeps the power-of-two rank counts the backend actually chooses and keeps the
    best. Answers "what is the biggest model that fits on this box" with the
    binding constraint named, so the answer is actionable rather than a number.
    """
    best: dict[str, Any] = {"weight_gb": 0.0, "ranks": 0, "binding": "none"}
    cores = hw.cores * max(1, node_count)
    host = hw.host_ram_gb * max(1, node_count)
    r = 1
    while r <= cores:
        hbm_cap = r * hw.usable_gb_per_core
        caps = [("hbm-per-rank", hbm_cap), ("total-hbm", hw.total_hbm_gb * max(1, node_count))]
        if host > 0:
            if lean_loader:
                caps.append(("host-dram", max(0.0, host - r * _LEAN_STREAM_GB)))
            else:
                caps.append(("host-dram", host / r))
        binding, cap = min(caps, key=lambda kv: kv[1])
        if cap > best["weight_gb"]:
            best = {"weight_gb": round(cap, 1), "ranks": r, "binding": binding}
        r *= 2
    best["params_b"] = round(best["weight_gb"] * 1e9 / bytes_per_param / 1e9, 1)
    best["hw"] = hw.name
    best["lean_loader"] = lean_loader
    return best


def max_clean_tp(config: dict[str, Any], hw: HardwareProfile) -> int:
    """Largest TP this backend will actually use for the model.

    Bounded by the adapter caps, the head count (TP must divide the query heads)
    and the physical core count.
    """
    tc = _text_config(config)
    archs = " ".join(architectures(config))
    heads = _int(tc, "num_attention_heads", 32)
    cap = hw.cores
    for needle, capped in _TP_CAPS:
        if needle.lower() in archs.lower():
            # None -> bounded only by the head count (and cores).
            cap = min(cap, capped if capped is not None else heads)
            break
    best = 1
    for tp in (1, 2, 4, 8, 16, 32, 64):
        if tp <= cap and heads and heads % tp == 0:
            best = tp
    return best


def _dtype_bytes(config: dict[str, Any], dtype: str | None) -> float:
    key = (dtype or _text_config(config).get("dtype")
           or config.get("torch_dtype") or "bf16")
    return _DTYPE_BYTES.get(str(key).lower(), 2)


def dequant_factor(config: dict[str, Any], compute_dtype: str = "bf16") -> tuple[float, str]:
    """How much a quantized checkpoint EXPANDS when loaded, and why.

    A measured on-disk size is the right input for an unquantized model, but it
    understates a quantized one, because Neuron has no fp8/int4 compute path and
    transformers says so explicitly at load:

        Using FP8 quantized models requires a GPU or XPU, we will default to
        dequantizing the model to bf16 since no GPU or XPU is available

    So DeepSeek-V4-Flash's 159.6 GB of fp8 weights become ~319 GB of bf16 in HBM.
    Sizing off the download would under-count it by 2x and call an infeasible
    model runnable -- which is exactly what happened before this existed.

    Returns ``(factor, note)``; ``(1.0, "")`` when unquantized.
    """
    qc = config.get("quantization_config") or _text_config(config).get("quantization_config")
    if not isinstance(qc, dict):
        return 1.0, ""
    method = str(qc.get("quant_method", "") or "").lower()
    stored = {"fp8": 1.0, "int8": 1.0, "gptq": 0.5, "awq": 0.5,
              "int4": 0.5, "nf4": 0.5, "bitsandbytes": 0.5}.get(method)
    if stored is None:
        return 1.0, f" (quant_method={method or 'unknown'}: expansion unknown)"
    factor = _DTYPE_BYTES.get(compute_dtype.lower(), 2) / stored
    if factor <= 1.0:
        return 1.0, ""
    return factor, (f" ({method} dequantized to {compute_dtype} at load: "
                    f"{factor:g}x the on-disk size)")


def assess(
    config: dict[str, Any],
    hw: HardwareProfile = TRN2_48XLARGE,
    *,
    dtype: str | None = None,
    node_count: int = 1,
    weight_gb: float | None = None,
) -> Verdict:
    """Decide whether ``config`` can run on ``hw`` before spending anything.

    ``weight_gb`` — measured total weight size, e.g. summed from the HF repo's
    file metadata (one cheap API call, no download). PREFER IT. The config
    estimate is a model of the architecture and it is only as good as that model:
    it tracks Qwen3.5 MoE well (68 vs 71.9 GB actual; 239 vs 250.2) but is wrong
    by 3.5x on DeepSeek-V4-Flash (564 vs 159.6, which uses compressed-KV
    attention) and 7x on Kimi-K3. Sizing a model from a formula that does not
    know its architecture is exactly the mistake this module exists to correct,
    so when the real number is available it wins.

    Never raises: an unsizeable config with no measured size yields UNKNOWN
    (fail open).
    """
    params, breakdown = estimate_params(config)
    if not params and weight_gb is None:
        return Verdict(True, "UNKNOWN",
                       reason=("cannot size this config ("
                               f"{breakdown.get('reason', 'unknown shape')}) -- "
                               "proceeding rather than blocking a model that may work"),
                       details=breakdown)

    dq, dq_note = dequant_factor(config, dtype or "bf16")
    if weight_gb is not None:
        # A measured size is the ON-DISK size. If the checkpoint is quantized and
        # gets dequantized at load, HBM sees the expanded size, so scale it.
        breakdown["weight_source"] = "measured"
        breakdown["on_disk_gb"] = round(weight_gb, 1)
        breakdown["dequant_factor"] = dq
        breakdown["config_estimate_gb"] = round(
            params * _dtype_bytes(config, dtype) / 1e9, 1) if params else None
        weight_gb = weight_gb * dq
    else:
        # The config estimate is already in compute dtype, so no expansion.
        breakdown["weight_source"] = "config-estimate"
        breakdown["dequant_factor"] = 1.0
        weight_gb = params * _dtype_bytes(config, dtype) / 1e9
        dq_note = ""
    tp = max_clean_tp(config, hw)
    ranks = tp * max(1, node_count)
    gb_per_rank = weight_gb / ranks
    budget = hw.usable_gb_per_core
    # Smallest power-of-two TP that would fit, ignoring the adapter cap.
    min_tp = 1
    while min_tp < 4096 and weight_gb / min_tp > budget:
        min_tp *= 2

    common = dict(params=params, weight_gb=round(weight_gb, 1), chosen_tp=tp,
                  gb_per_rank=round(gb_per_rank, 1),
                  budget_gb_per_rank=round(budget, 1),
                  min_tp_needed=min_tp, details=breakdown)

    host_peak = host_load_peak_gb(weight_gb, ranks)
    host_lean = host_load_peak_gb(weight_gb, ranks, lean_loader=True)
    host_stag = host_load_peak_gb(weight_gb, ranks, concurrency=2)
    host_stag1 = host_load_peak_gb(weight_gb, ranks, concurrency=1)
    host_cap = hw.host_ram_gb * max(1, node_count)
    fits_stagger = host_stag <= host_cap
    breakdown["host_ram_gb"] = hw.host_ram_gb
    breakdown["host_peak_gb"] = round(host_peak, 1)
    breakdown["host_peak_lean_gb"] = round(host_lean, 1)
    breakdown["host_peak_stagger2_gb"] = round(host_stag, 1)

    if weight_gb > hw.total_hbm_gb * max(1, node_count):
        return Verdict(
            False, "NEEDS_MULTINODE",
            reason=(f"{weight_gb:.0f} GB of weights exceeds the whole "
                    f"{hw.name} ({hw.total_hbm_gb:.0f} GB HBM across "
                    f"{hw.cores} cores){dq_note} -- needs nodeCount>1, "
                    f"not a bigger TP"),
            **common)

    if hw.host_ram_gb > 0 and host_peak > hw.host_ram_gb * max(1, node_count):
        fits_lean = host_lean <= hw.host_ram_gb * max(1, node_count)
        return Verdict(
            False, "HOST_LIMITED",
            reason=(f"loading {weight_gb:.0f} GB on {ranks} ranks needs "
                    f"{host_peak:.0f} GB of host DRAM ({ranks} full copies, one "
                    f"per rank, because from_pretrained materialises the whole "
                    f"model before sharding) but {hw.name} has "
                    f"{hw.host_ram_gb:.0f} GB -- this OOM-kills the pod during "
                    f"load, before any core is touched. A bigger tp makes it "
                    f"WORSE, not better"
                    + (f"; TRN_OPT_LOAD_CONCURRENCY=2 brings the peak to "
                       f"{host_stag:.0f} GB and fits" if fits_stagger
                       else f"; even TRN_OPT_LOAD_CONCURRENCY=1 peaks at "
                            f"{host_stag1:.0f} GB")
                    + (f"; a shard-on-read loader would need "
                       f"{host_lean:.0f} GB" if fits_lean
                       else f"; even a shard-on-read loader needs "
                            f"{host_lean:.0f} GB")
                    + dq_note),
            **common)

    if gb_per_rank > budget:
        moe = " (MoE: expert weights dominate)" if breakdown.get("is_moe") else ""
        moe += dq_note
        return Verdict(
            False, "TOO_LARGE",
            reason=(f"{weight_gb:.0f} GB of weights at tp={tp} is "
                    f"{gb_per_rank:.0f} GB/rank, over the {budget:.0f} GB/rank "
                    f"budget on {hw.name}{moe}; needs tp>={min_tp}"
                    + (f", but this architecture is capped at tp={tp}"
                       if min_tp > tp else "")),
            **common)

    if gb_per_rank > hw.comfortable_gb_per_core:
        return Verdict(
            True, "TIGHT",
            reason=(f"{weight_gb:.0f} GB at tp={tp} = {gb_per_rank:.1f} GB/rank "
                    f"fits the {budget:.0f} GB/rank ceiling but exceeds the "
                    f"{hw.comfortable_gb_per_core:.0f} GB comfortable mark -- "
                    f"expect high HBM occupancy and configs that add memory "
                    f"(larger batch, cp>1) to OOM"),
            **common)

    return Verdict(True, "RUNNABLE",
                   reason=(f"{weight_gb:.0f} GB at tp={tp} = {gb_per_rank:.1f} "
                           f"GB/rank, within the {budget:.0f} GB/rank budget"),
                   **common)
