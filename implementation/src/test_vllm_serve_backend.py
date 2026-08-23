"""
Tests for the vllm-serve config-search backend, against a hardware-free fake
(no `vllm serve` launched). Mirrors the mock-backend test style in
test_moe_borrow.py: subclass the REAL backend and override the single on-device
seam (`_serve_and_measure`) with a synthetic result, so the backend's config
axes, its Stage-0 baseline, and the orchestrator's tournament + equivalence gate
all run off-device.

What the wiring must guarantee:
  - the vllm-serve backend is SELECTED for its backend name (overnight routing),
  - config_axes() emits the SERVING search space, and drops the optional axes
    (fp8 / async-scheduling / speculative) when the stack does not support them,
  - a latency-WORSE serving config (lower decode tok/s) is discarded, never kept,
  - an output-BREAKING config (fp8 whose greedy tokens drift from the bf16
    baseline) fails the equivalence gate and is discarded despite looking faster,
  - a FASTER + output-equivalent config is kept as the incumbent.

ON-DEVICE: DEFERRED (boxes busy). These tests exercise the mechanism; the real
sweep (Qwen3-8B / Gemma-4-12B toward a 2 s SLA) is the next step once a box frees.
"""

from __future__ import annotations

from pathlib import Path

from backends.vllm_serve import VllmServeBackend
from bank import KnowledgeBank
from guardrails import Guardrails
from ledger import Ledger, Status
from orchestrator import ModelSpec, Orchestrator, always_equivalent

SPEC = ModelSpec(
    model_id="mock/gemma-4-12b-it", family="dense_causal_lm", param_count=12e9,
    parent="gemma", probe_shape="serve 2048/512", probe_batch=1, track="latency",
)

# 16-token greedy signature the bf16 baseline "emits"; output-neutral configs
# reproduce it, fp8 drifts off it.
_BASE_TOKENS = list(range(100, 116))


class _FakeServe(VllmServeBackend):
    """A VllmServeBackend whose on-device serve is replaced by a synthetic,
    deterministic latency model. tp=2 shaves a little collective overhead at
    batch-1; a bigger batched-tokens bucket slightly HURTS single-stream decode;
    fp8 and speculative decoding look faster. fp8's greedy output DRIFTS (quant
    error), so the equivalence gate — not the metric — must reject it."""

    def _serve_and_measure(self, model_id, cfg, input_len, output_len):
        tp = int(cfg.get("tp_degree", 4))
        dtype = cfg.get("weights_dtype", "bf16")
        spec = cfg.get("speculative", "off")
        bucket = int(cfg.get("num_batched_tokens", 512))

        decode = 40.0
        decode *= {4: 1.0, 2: 1.15}.get(tp, 1.0)
        decode *= {"bf16": 1.0, "fp8": 1.4}.get(dtype, 1.0)
        decode *= 1.25 if spec == "draft" else 1.0
        decode *= {512: 1.0, 1024: 0.97, 2048: 0.94}.get(bucket, 1.0)

        tokens = list(_BASE_TOKENS)
        if dtype == "fp8":
            tokens = [t + 500 for t in tokens]     # drift -> 0% match -> gate fail

        ttft_ms = 600.0
        e2e = ttft_ms / 1000.0 + (output_len - 1) / max(decode, 1e-6)
        return {
            "ok": True, "decode_tok_s": decode, "ttft_ms": ttft_ms,
            "tpot_ms": 1000.0 / decode, "e2e_seconds": e2e,
            "hits_sla": e2e <= self.sla_seconds, "top1_tokens": tokens,
            "hbm_peak_gb": 18.0, "hbm_available_gb": 24.0,
        }


def _backend(caps=None) -> _FakeServe:
    return _FakeServe(core_count=64, target_input_len=2048, target_output_len=512,
                      sla_seconds=2.0,
                      capabilities=caps if caps is not None
                      else {"fp8": True, "async_scheduling": True, "spec_decode": True})


def _orch(tmp_path: Path, backend) -> Orchestrator:
    orch = Orchestrator(
        backend=backend, bank=KnowledgeBank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(tmp_path / "run"),
        equivalence=always_equivalent, sdk_version="2.28.0",
    )
    orch.ledger.init()
    return orch


# -- selection / routing -----------------------------------------------------

def test_vllm_serve_backend_selected_by_name():
    """overnight._make_backend routes the 'vllm-serve' name to the serving
    backend, carrying the SLA target through."""
    import overnight
    b = overnight._make_backend(
        "vllm-serve", instance_type="trn2.48xlarge",
        serve_target=overnight.ServeTarget(input_len=2048, output_len=512,
                                           sla_seconds=2.0))
    assert isinstance(b, VllmServeBackend)
    assert b.name == "vllm-serve"
    assert b.target_input_len == 2048 and b.sla_seconds == 2.0


# -- config axes -------------------------------------------------------------

def test_config_axes_emit_serving_space_when_supported():
    """With a stack that supports the optional knobs, the serving search space
    includes tp (4/2), bf16+fp8, max_num_seqs=1, the batched-token buckets,
    async-scheduling, and speculative decoding."""
    axes = _backend().config_axes()
    assert axes["tp_degree"] == [4, 2]
    assert axes["weights_dtype"] == ["bf16", "fp8"]
    assert axes["max_num_seqs"] == [1]
    assert axes["num_batched_tokens"] == [512, 1024, 2048]
    assert axes["async_scheduling"] == [False, True]
    assert axes["speculative"] == ["off", "draft"]


def test_config_axes_drop_unsupported_axes_gracefully():
    """A stack that supports none of the optional knobs (e.g. a laptop / older
    vLLM) emits only the universally-available serving axes — no fp8, no
    async-scheduling, no speculative — rather than fabricating them."""
    axes = _backend(caps={}).config_axes()
    assert axes["weights_dtype"] == ["bf16"]     # fp8 dropped
    assert "async_scheduling" not in axes
    assert "speculative" not in axes
    assert axes["tp_degree"] == [4, 2]           # core axes still present
    assert axes["num_batched_tokens"] == [512, 1024, 2048]


# -- tournament: keep the faster+equivalent, drop the worse ------------------

def test_faster_equivalent_config_is_kept(tmp_path: Path):
    """Stage-1 search finds a serving config that is faster than the bf16/tp4
    baseline AND reproduces its greedy output — and keeps it as the incumbent."""
    orch = _orch(tmp_path, _backend())
    base = orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)
    assert orch.incumbent.metric > base.metric          # strictly faster
    # ...and it did NOT win by garbling output: the incumbent is not fp8.
    assert orch.incumbent.config.get("weights_dtype") != "fp8"
    assert any(r.status is Status.KEEP for r in orch.ledger.read())


def test_latency_worse_config_is_discarded(tmp_path: Path):
    """A config the latency model scores WORSE than the incumbent (here the
    2048 batched-tokens bucket, which hurts single-stream decode) is recorded as
    a discard and never becomes the incumbent."""
    orch = _orch(tmp_path, _backend(caps={}))   # no fp8/spec so the win is tp/bucket
    orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)
    rows = orch.ledger.read()
    worse = [r for r in rows if "num_batched_tokens=2048" in r.description]
    assert worse, "the 2048-bucket candidate should have been tried"
    assert all(r.status is Status.DISCARD for r in worse)
    assert orch.incumbent.config.get("num_batched_tokens") != 2048


def test_output_breaking_fp8_is_discarded(tmp_path: Path):
    """fp8 looks 1.4x faster but its greedy tokens drift from the bf16 baseline,
    so the equivalence gate rejects it — it is never kept, and the incumbent
    stays bf16 (a quant config that wins latency but garbles output loses)."""
    orch = _orch(tmp_path, _backend())
    orch.establish_baseline(SPEC)
    orch.run_stage1_config(SPEC)
    rows = orch.ledger.read()
    fp8_rows = [r for r in rows if "weights_dtype=fp8" in r.description]
    assert fp8_rows, "fp8 should have been proposed and evaluated"
    assert any("equivalence fail" in r.description for r in fp8_rows)
    assert not any(r.status is Status.KEEP for r in fp8_rows)
    assert orch.incumbent.config.get("weights_dtype") != "fp8"


# -- tied-embedding lm_head fix ---------------------------------------------
#
# Real bug: a tied-embedding model whose checkpoint does NOT materialize
# lm_head.weight (Qwen3-4B — shares model.embed_tokens.weight) crashes
# vLLM-Neuron's STRICT load_sharded_pipelined loader with
# "Checkpoint key(s) not found for parameter 'lm_head.weight'", so the baseline
# serve produces 0 tok/s -> FAIL_NO_BASELINE. The differentiator is NOT tp: the
# other three Qwen3 sizes (0.6B/1.7B tied, 8B untied) all physically ship
# lm_head.weight and load at any tp; 4B is the only one lacking it and it fails
# at every tp. Fix: for such a model the worker serves a PATCHED local checkpoint
# that adds lm_head.weight := embed_tokens.weight; all other models are
# unchanged. These tests assert the routing decision (mock the HF config) and
# that the built `vllm serve` argv uses the resolved path.

import argparse

import backends.vllm_serve_worker as w


def _serve_args(model: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=model, tp=4, dtype="bf16", quantization="none", max_num_seqs=1,
        num_batched_tokens=512, async_scheduling="0", on_device_sampling="1",
        speculative="off", draft_model="", input_len=2048, output_len=512,
        port=8000)


def test_tied_model_without_lmhead_serves_patched_checkpoint(monkeypatch, tmp_path):
    """Tied embeddings + checkpoint missing lm_head.weight (the Qwen3-4B case)
    -> the worker resolves to a patched checkpoint dir instead of the raw id."""
    monkeypatch.setattr(w, "_hf_config_dict", lambda m: {"tie_word_embeddings": True})
    monkeypatch.setattr(w, "_resolve_snapshot_dir", lambda m: str(tmp_path))
    monkeypatch.setattr(w, "_checkpoint_has_lm_head", lambda d: False)
    monkeypatch.setattr(w, "_materialize_tied_lm_head",
                        lambda m, d: str(tmp_path / "patched"))
    assert w._resolve_served_model("Qwen/Qwen3-4B") == str(tmp_path / "patched")


def test_tied_model_uses_patched_path_in_serve_cmd(monkeypatch):
    """The resolved (patched) path is what actually reaches `vllm serve`; the raw
    tied model id is replaced in the argv."""
    monkeypatch.setattr(w, "_resolve_served_model", lambda m: "/patched/qwen3-4b")
    cmd, _env = w._build_serve_cmd(_serve_args("Qwen/Qwen3-4B"))
    assert cmd[cmd.index("serve") + 1] == "/patched/qwen3-4b"
    assert "Qwen/Qwen3-4B" not in cmd


def test_untied_model_served_unchanged(monkeypatch):
    """Untied model (Qwen3-8B): never patched, served by its raw id — the fix is
    additive and leaves the working path untouched."""
    monkeypatch.setattr(w, "_hf_config_dict", lambda m: {"tie_word_embeddings": False})
    patched_called = []
    monkeypatch.setattr(w, "_materialize_tied_lm_head",
                        lambda m, d: patched_called.append((m, d)) or "/x")
    assert w._resolve_served_model("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
    assert not patched_called, "an untied model must never be patched"


def test_tied_model_with_materialized_lmhead_served_unchanged(monkeypatch, tmp_path):
    """Tied model whose checkpoint ALREADY materializes lm_head.weight
    (Qwen3-0.6B / 1.7B): served unchanged, no patch — this is why those tied
    models work today and must keep working."""
    monkeypatch.setattr(w, "_hf_config_dict", lambda m: {"tie_word_embeddings": True})
    monkeypatch.setattr(w, "_resolve_snapshot_dir", lambda m: str(tmp_path))
    monkeypatch.setattr(w, "_checkpoint_has_lm_head", lambda d: True)
    patched_called = []
    monkeypatch.setattr(w, "_materialize_tied_lm_head",
                        lambda m, d: patched_called.append(1) or "/x")
    assert w._resolve_served_model("Qwen/Qwen3-1.7B") == "Qwen/Qwen3-1.7B"
    assert not patched_called


def test_resolve_is_fail_open(monkeypatch):
    """Any error while deciding -> serve the original id (today's behavior),
    never crash the worker."""
    def boom(_m):
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(w, "_hf_config_dict", boom)
    assert w._resolve_served_model("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_materialize_adds_lm_head_equal_to_embed(tmp_path):
    """End-to-end materialization (skipped where torch/safetensors are absent,
    e.g. the laptop test venv; runs on-device): the patched dir gains
    lm_head.weight bit-identical to embed_tokens.weight, and _checkpoint_has_lm_head
    flips False->True."""
    import pytest
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors")
    from safetensors.torch import save_file, load_file
    src = tmp_path / "ckpt"
    src.mkdir()
    emb = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    save_file({w._EMBED_KEY: emb}, str(src / "model.safetensors"))
    assert w._checkpoint_has_lm_head(str(src)) is False
    patched = w._materialize_tied_lm_head("some/model", str(src))
    assert patched is not None
    assert w._checkpoint_has_lm_head(patched) is True
    # find the file carrying lm_head.weight and confirm it equals embed
    import glob
    got = None
    for fp in glob.glob(patched + "/*.safetensors"):
        d = load_file(fp)
        if w._LMHEAD_KEY in d:
            got = d[w._LMHEAD_KEY]
    assert got is not None and torch.equal(got, emb)
