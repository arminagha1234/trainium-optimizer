"""
Tests for auto-discovery (discover.py).

Mock-only, no hardware and no network: models + their configs are served by a
`MockSource`, so a whole discovery pass runs offline and deterministically.

Covers:
  - license filtering (permissive kept; gated / "other" / unknown dropped),
  - size filtering (fits the cap kept; over-cap dropped as too-big; unknown
    size dropped),
  - family-by-shape (dense vs MoE) and non-LLM rejection,
  - linear-attention arches skipped (composes with the pre-flight gate),
  - dedup against models_queue.txt (already-queued dropped),
  - queue append: correct `hf_id<TAB>family<TAB>tag` format, never duplicates,
    backs the queue up first,
  - drop-LOGGING: every drop is emitted (no silent filtering).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from discover import (
    DISCOVERED_TAG,
    Candidate,
    Filters,
    MockSource,
    RawModel,
    append_to_queue,
    discover,
    estimate_params_from_config,
    family_from_config,
    load_queued_ids,
    read_queue,
)


# -- canned configs ----------------------------------------------------------

def _dense_cfg(hidden=2048, layers=24, vocab=32000, heads=16, inter=5632):
    return {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": hidden, "num_hidden_layers": layers,
            "vocab_size": vocab, "num_attention_heads": heads,
            "intermediate_size": inter}


def _moe_cfg():
    return {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
            "hidden_size": 1024, "num_hidden_layers": 8, "vocab_size": 32000,
            "num_attention_heads": 16, "intermediate_size": 3584,
            "num_local_experts": 8}


DELTANET_CFG = {"architectures": ["Qwen3GatedDeltaNetForCausalLM"],
                "model_type": "qwen3_gated_deltanet", "hidden_size": 2048,
                "num_hidden_layers": 24, "vocab_size": 32000,
                "num_attention_heads": 16, "intermediate_size": 5632}

# Not an LLM (a diffusion pipeline config) — no *ForCausalLM arch.
DIFFUSION_CFG = {"architectures": ["FluxPipeline"], "model_type": "flux"}


def _source():
    """A mock source exercising every filter branch."""
    models = [
        RawModel("acme/good-2b", downloads=5000, trending_score=99.0,
                 pipeline_tag="text-generation", license="apache-2.0"),
        RawModel("acme/good-moe", downloads=4000, trending_score=95.0,
                 pipeline_tag="text-generation", license="mit"),
        RawModel("acme/huge-70b", downloads=9000, trending_score=90.0,
                 pipeline_tag="text-generation", license="apache-2.0"),
        RawModel("meta/gated-8b", downloads=8000, trending_score=85.0,
                 pipeline_tag="text-generation", license="apache-2.0", gated=True),
        RawModel("corp/proprietary-3b", downloads=7000, trending_score=80.0,
                 pipeline_tag="text-generation", license="other"),
        RawModel("labs/mystery-lm", downloads=6000, trending_score=75.0,
                 pipeline_tag="text-generation", license=None),
        RawModel("art/flux-image", downloads=6500, trending_score=70.0,
                 pipeline_tag="text-to-image", license="apache-2.0"),
        RawModel("exp/deltanet-2b", downloads=5500, trending_score=65.0,
                 pipeline_tag="text-generation", license="apache-2.0"),
        RawModel("acme/no-config-1b", downloads=100, trending_score=10.0,
                 pipeline_tag="feature-extraction", license="mit"),
    ]
    configs = {
        "acme/good-2b": _dense_cfg(),
        "acme/good-moe": _moe_cfg(),
        "acme/huge-70b": _dense_cfg(hidden=8192, layers=80, heads=64, inter=28672),
        "meta/gated-8b": _dense_cfg(hidden=4096, layers=32, heads=32, inter=14336),
        "corp/proprietary-3b": _dense_cfg(hidden=2560, layers=32),
        # mystery-lm: license None -> dropped before config even matters
        "art/flux-image": DIFFUSION_CFG,
        "exp/deltanet-2b": DELTANET_CFG,
        # no-config-1b: no config entry (loader returns None) + non-LLM tag
    }
    return MockSource(models, configs)


# -- shape helpers -----------------------------------------------------------

def test_family_from_config():
    assert family_from_config(_dense_cfg()) == "dense_causal_lm"
    assert family_from_config(_moe_cfg()) == "moe_causal_lm"
    assert family_from_config(DIFFUSION_CFG) is None      # non-LLM
    assert family_from_config(None) is None


def test_param_estimate_is_in_the_right_ballpark():
    # A ~1.1B TinyLlama-shaped config should estimate in the 0.8-1.6B range.
    est = estimate_params_from_config(_dense_cfg())
    assert 0.8e9 < est < 1.8e9
    # A 70B-shaped config should blow well past a 14B cap.
    assert estimate_params_from_config(
        _dense_cfg(hidden=8192, layers=80, heads=64, inter=28672)) > 14e9
    # Missing core fields -> unknown (0.0).
    assert estimate_params_from_config({"model_type": "llama"}) == 0.0


# -- filtering ---------------------------------------------------------------

def test_discover_keeps_only_safe_candidates():
    report = discover(_source(), limit=20, filters=Filters(max_params=14e9))
    kept = {c.hf_id for c in report.candidates}
    assert kept == {"acme/good-2b", "acme/good-moe"}

    fams = {c.hf_id: c.family for c in report.candidates}
    assert fams["acme/good-2b"] == "dense_causal_lm"
    assert fams["acme/good-moe"] == "moe_causal_lm"
    assert all(c.tag == DISCOVERED_TAG for c in report.candidates)


def test_every_drop_has_the_expected_reason():
    report = discover(_source(), limit=20, filters=Filters(max_params=14e9))
    reasons = {d.hf_id: d.reason for d in report.drops}
    assert reasons["acme/huge-70b"] == "too-big"
    assert reasons["meta/gated-8b"] == "gated"
    assert reasons["corp/proprietary-3b"] == "bad-license"
    assert reasons["labs/mystery-lm"] == "bad-license"      # license None
    assert reasons["art/flux-image"] == "non-llm"           # diffusion
    assert reasons["exp/deltanet-2b"] == "linear-attn"
    assert reasons["acme/no-config-1b"] in ("no-config", "non-llm")


def test_no_moe_policy_excludes_moe():
    report = discover(_source(), limit=20,
                      filters=Filters(max_params=14e9, allow_moe=False))
    kept = {c.hf_id for c in report.candidates}
    assert kept == {"acme/good-2b"}          # the MoE is now filtered
    assert any(d.hf_id == "acme/good-moe" for d in report.drops)


def test_tighter_cap_drops_more():
    report = discover(_source(), limit=20, filters=Filters(max_params=1e9))
    # good-2b (~1.1B) now exceeds a 1B cap -> too-big.
    assert not any(c.hf_id == "acme/good-2b" for c in report.candidates)
    assert any(d.hf_id == "acme/good-2b" and d.reason == "too-big"
               for d in report.drops)


def test_unknown_size_is_dropped():
    # A readable text-gen config that lacks the size fields (hidden/layers/vocab)
    # and an id with no size token -> size is UNKNOWN, which is too risky to
    # queue autonomously, so it's dropped as unknown-size.
    src = MockSource(
        [RawModel("acme/no-size", pipeline_tag="text-generation",
                  license="apache-2.0")],
        configs={"acme/no-size": {"architectures": ["LlamaForCausalLM"],
                                  "model_type": "llama"}},
    )
    report = discover(src, limit=5, filters=Filters(max_params=14e9))
    assert report.candidates == []
    assert report.drops[0].reason == "unknown-size"


def test_dedup_against_existing_queue(tmp_path: Path):
    queue = tmp_path / "models_queue.txt"
    queue.write_text("acme/good-2b\tdense_causal_lm\tseed\n")
    filters = Filters.build(max_params=14e9, queue_path=queue)
    report = discover(_source(), limit=20, filters=filters)
    kept = {c.hf_id for c in report.candidates}
    assert "acme/good-2b" not in kept                 # already queued
    assert "acme/good-moe" in kept
    assert any(d.hf_id == "acme/good-2b" and d.reason == "already-queued"
               for d in report.drops)


# -- drop logging (no silent filtering) --------------------------------------

def test_every_drop_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="discover"):
        report = discover(_source(), limit=20, filters=Filters(max_params=14e9))
    drop_logs = [r.getMessage() for r in caplog.records
                 if r.getMessage().startswith("DROP")]
    # One DROP log line per dropped model — nothing filtered silently.
    assert len(drop_logs) == len(report.drops)
    for d in report.drops:
        assert any(d.hf_id in m and d.reason in m for m in drop_logs)


# -- queue append ------------------------------------------------------------

def test_append_writes_correct_format_and_backs_up(tmp_path: Path):
    queue = tmp_path / "models_queue.txt"
    queue.write_text("existing/model\tdense_causal_lm\tseed\n")
    cands = [
        Candidate("acme/good-2b", "dense_causal_lm", param_count=1.1e9),
        Candidate("acme/good-moe", "moe_causal_lm", param_count=6e9),
    ]
    appended, backup = append_to_queue(cands, queue)

    assert {c.hf_id for c in appended} == {"acme/good-2b", "acme/good-moe"}
    # Backup made BEFORE the write, holding the pre-append content.
    assert backup is not None and backup.exists()
    assert backup.read_text() == "existing/model\tdense_causal_lm\tseed\n"

    ids = read_queue(queue)
    assert ids == ["existing/model", "acme/good-2b", "acme/good-moe"]
    # Format: hf_id<TAB>family<TAB>tag, tagged discovered.
    lines = [l for l in queue.read_text().splitlines()
             if l and not l.startswith("#")]
    assert lines[1] == "acme/good-2b\tdense_causal_lm\tdiscovered"
    assert lines[2] == "acme/good-moe\tmoe_causal_lm\tdiscovered"


def test_append_never_duplicates(tmp_path: Path):
    queue = tmp_path / "models_queue.txt"
    queue.write_text("acme/good-2b\tdense_causal_lm\tseed\n")
    cands = [
        Candidate("acme/good-2b", "dense_causal_lm"),      # already present
        Candidate("acme/good-moe", "moe_causal_lm"),       # new
        Candidate("acme/good-moe", "moe_causal_lm"),       # dup within batch
    ]
    appended, _ = append_to_queue(cands, queue)
    assert {c.hf_id for c in appended} == {"acme/good-moe"}
    assert read_queue(queue).count("acme/good-moe") == 1


def test_append_to_missing_queue_creates_it(tmp_path: Path):
    queue = tmp_path / "sub" / "models_queue.txt"   # parent also missing
    appended, backup = append_to_queue(
        [Candidate("acme/good-2b", "dense_causal_lm")], queue)
    assert backup is None                            # nothing to back up
    assert queue.exists()
    assert read_queue(queue) == ["acme/good-2b"]


def test_append_nothing_when_all_dupes(tmp_path: Path):
    queue = tmp_path / "models_queue.txt"
    queue.write_text("acme/good-2b\tdense_causal_lm\tseed\n")
    appended, backup = append_to_queue(
        [Candidate("acme/good-2b", "dense_causal_lm")], queue)
    assert appended == [] and backup is None
    # Queue untouched (single line, no dup, no spurious backup).
    assert read_queue(queue) == ["acme/good-2b"]


def test_read_queue_skips_comments_and_blanks(tmp_path: Path):
    queue = tmp_path / "models_queue.txt"
    queue.write_text("# header\n\nacme/a\tdense_causal_lm\tseed\n\nacme/b\tmoe_causal_lm\tdiscovered\n")
    assert read_queue(queue) == ["acme/a", "acme/b"]
    assert load_queued_ids(queue) == frozenset({"acme/a", "acme/b"})


def test_end_to_end_discover_then_append(tmp_path: Path):
    """The whole point: discover -> the survivors land in the queue in the exact
    format auto-onboarding consumes, tagged discovered, no human."""
    queue = tmp_path / "models_queue.txt"
    filters = Filters.build(max_params=14e9, queue_path=queue)
    report = discover(_source(), limit=20, filters=filters)
    appended, _ = append_to_queue(report.candidates, queue)
    assert {c.hf_id for c in appended} == {"acme/good-2b", "acme/good-moe"}
    lines = [l for l in queue.read_text().splitlines()
             if l and not l.startswith("#")]
    assert set(lines) == {
        "acme/good-2b\tdense_causal_lm\tdiscovered",
        "acme/good-moe\tmoe_causal_lm\tdiscovered",
    }
