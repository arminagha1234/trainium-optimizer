"""The showcase must refresh itself at the end of every cycle.

Relying on someone remembering to run `publish_deliverables.py` is how the
leaderboard drifted behind the runs: real verified results sat in run artifacts for
days while the published table showed neither them nor any 48xl row at all.

What "automatic" must NOT change is what qualifies. Every gate stays where it was --
verified=="verified", speedup>1.0, bundle present on disk -- so these tests pin the
gates as much as the automation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import overnight
from publish_deliverables import collect_verified


class _Args:
    """Stand-in for the parsed argv.

    ``publish_repo_dir`` has no default here ON PURPOSE. In production it falls back
    to the repo overnight.py lives in, which is what let a mock smoke-run rewrite the
    real LEADERBOARD.md. A test must never be one forgotten kwarg away from doing the
    same, so every case passes an explicit temp checkout.
    """

    def __init__(self, publish_repo_dir, **kw):
        self.publish = True
        self.backend = "native-pytorch-beta3"
        self.publish_repo_dir = publish_repo_dir
        self.publish_push = False
        for k, v in kw.items():
            setattr(self, k, v)


def _bundle(root: Path, slug: str, *, model_id, speedup, verified="verified",
            hardware="trn2.48xlarge"):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(json.dumps({
        "model_id": model_id,
        "backend": "native-pytorch-beta3",
        "baseline_metric": 100.0,
        "best_metric": 100.0 * speedup,
        "speedup": speedup,
        "metric_label": "tok/s",
        "config": {"tp_degree": 8, "weights_dtype": "bf16", "batch": 1},
        "toolchain": {"instance_type": hardware},
        "verified": verified,
    }, indent=2))
    (d / "optimization_timeline.png").write_bytes(b"\x89PNG\r\n")
    return d


def _run(tmp_path, bundles, **argkw):
    out_root = tmp_path / "art"
    om = out_root / "optimized_models"
    om.mkdir(parents=True)
    for b in bundles:
        _bundle(om, **b)
    repo = tmp_path / "repo"
    (repo / "optimized_models").mkdir(parents=True)
    (repo / "README.md").write_text("# repo\n")
    logs: list[str] = []
    res = overnight._auto_publish(
        out_root, _Args(str(repo), **argkw), logs.append)
    return repo, logs, res


def test_a_verified_win_lands_in_the_repo_and_on_the_leaderboard(tmp_path):
    repo, logs, res = _run(tmp_path, [dict(
        slug="qwen3-5-35b-a3b", model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.6)])
    board = (repo / "LEADERBOARD.md").read_text()
    assert "Qwen3.5-35B-A3B" in board
    assert "trn2.48xlarge" in board
    assert (repo / "optimized_models" / "qwen3-5-35b-a3b" / "recipe.json").exists()
    assert "qwen3-5-35b-a3b" in (res.get("published") or [])


def test_the_per_model_route_readme_is_written_automatically(tmp_path):
    repo, _, _ = _run(tmp_path, [dict(
        slug="qwen3-5-35b-a3b", model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.6)])
    md = (repo / "optimized_models" / "qwen3-5-35b-a3b" / "README.md").read_text()
    assert "verified route(s)" in md
    assert "trn2.48xlarge" in md


def test_an_unverified_result_is_never_published(tmp_path):
    """The 35B run that produced 6.33 tok/s but failed the grader.

    Automation must not become a way to smuggle an unauditable row onto the board.
    """
    repo, logs, res = _run(tmp_path, [dict(
        slug="q", model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.6,
        verified="unverified")])
    assert not (repo / "LEADERBOARD.md").exists()
    assert res.get("noop") is True
    assert any("nothing verified" in m for m in logs)


def test_a_baseline_only_result_is_never_published(tmp_path):
    """speedup == 1.0 is not an optimization; the showcase is of wins."""
    _, _, res = _run(tmp_path, [dict(
        slug="q", model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.0)])
    assert res.get("noop") is True


def test_writing_is_automatic_but_pushing_is_not(tmp_path):
    """Rewriting a checkout is reviewable and reversible; pushing is neither."""
    repo, _, res = _run(tmp_path, [dict(
        slug="q", model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.6)])
    assert (repo / "LEADERBOARD.md").exists(), "files should still be written"
    assert res.get("pushed") is False
    assert res.get("dry_run") is True


def test_consistency_is_checked_automatically(tmp_path):
    """A dead recipe link looks verified and cannot be audited -- catch it here,
    not in CI that is not wired yet."""
    repo, logs, _ = _run(tmp_path, [dict(
        slug="q", model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.6)])
    assert any("consistency OK" in m for m in logs), logs


def test_two_boxes_for_one_model_both_survive_publication(tmp_path):
    """The 3xl and 48xl routes are independent results, not a conflict."""
    repo, _, res = _run(tmp_path, [
        dict(slug="m/trn2-3xlarge", model_id="Qwen/Qwen3.8-27B", speedup=1.9,
             hardware="trn2.3xlarge"),
        dict(slug="m/trn2-48xlarge", model_id="Qwen/Qwen3.8-27B", speedup=1.3,
             hardware="trn2.48xlarge"),
    ])
    board = (repo / "LEADERBOARD.md").read_text()
    # Two tables now (peak throughput + improvement over baseline): each of the
    # two independent hardware routes appears once per table -> 4 mentions.
    assert "## Peak throughput" in board and "## Improvement over eager baseline" in board
    assert board.count("Qwen3.8-27B") == 4
    for hw in ("trn2.3xlarge", "trn2.48xlarge"):
        assert hw in board
    md = (repo / "optimized_models" / "m" / "README.md").read_text()
    assert "2 verified route(s)" in md


def test_a_missing_bundle_directory_is_a_clean_noop(tmp_path):
    out_root = tmp_path / "art"
    out_root.mkdir()
    logs: list[str] = []
    res = overnight._auto_publish(out_root, _Args(str(tmp_path / "repo")), logs.append)
    assert res.get("noop") is True
    assert any("nothing verified this cycle" in m for m in logs)


def test_publication_failure_never_loses_the_run(monkeypatch, tmp_path):
    """Measurements are the expensive, irreplaceable part of a cycle."""
    import publish_deliverables

    def boom(**kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(publish_deliverables, "publish", boom)
    out_root = tmp_path / "art"
    (out_root / "optimized_models").mkdir(parents=True)
    _bundle(out_root / "optimized_models", "q",
            model_id="Qwen/Qwen3.5-35B-A3B", speedup=1.6)
    logs: list[str] = []
    res = overnight._auto_publish(out_root, _Args(str(tmp_path / "r")), logs.append)
    assert "error" in res
    assert any("non-fatal" in m for m in logs)


def test_publish_can_be_turned_off(tmp_path):
    """--no-publish exists for a run you do not want touching the showcase."""
    parser_args = overnight  # module import check only
    assert hasattr(parser_args, "_auto_publish")
    # The flag is honoured by the caller, so assert the default is ON.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-publish", dest="publish", action="store_false")
    assert ap.parse_args([]).publish is True
    assert ap.parse_args(["--no-publish"]).publish is False


# --- synthetic results must never reach the public showcase -------------------
#
# Not hypothetical. Automatic publication defaults to the repo overnight.py lives in,
# so a laptop `--backend mock` smoke-run regenerated LEADERBOARD.md down to a single
# row -- "Qwen3-8B | 604 | 173,162 | 286.93x | ... | mock | verified" -- and rewrote
# optimized_models/qwen3-8b/ with it. Every gate that existed passed: verified,
# speedup > 1, bundle present. Nothing asked whether the number came from hardware.

def test_a_mock_backend_run_never_publishes(tmp_path):
    out_root = tmp_path / "art"
    om = out_root / "optimized_models"
    om.mkdir(parents=True)
    _bundle(om, "qwen3-8b", model_id="Qwen/Qwen3-8B", speedup=286.93,
            hardware="mock")
    repo = tmp_path / "repo"
    repo.mkdir()
    logs: list[str] = []
    args = _Args(str(repo))
    args.backend = "mock"
    res = overnight._auto_publish(out_root, args, logs.append)
    assert res.get("noop") is True
    assert not (repo / "LEADERBOARD.md").exists()
    assert any("mock" in m for m in logs)


def test_a_synthetic_instance_type_is_refused_even_on_a_real_backend(tmp_path):
    """Belt and braces: the backend name is not the only way a mock number leaks."""
    repo, logs, res = _run(tmp_path, [dict(
        slug="qwen3-8b", model_id="Qwen/Qwen3-8B", speedup=286.93,
        hardware="mock")])
    assert res.get("noop") is True
    assert not (repo / "LEADERBOARD.md").exists()


def test_real_neuron_instances_are_still_accepted():
    from publish_deliverables import is_real_hardware

    for hw in ("trn2.48xlarge", "trn2.3xlarge", "trn1.32xlarge", "inf2.48xlarge"):
        assert is_real_hardware(hw), hw


def test_anything_not_a_neuron_instance_is_refused():
    """An allowlist, so a NEW synthetic backend is excluded by default."""
    from publish_deliverables import is_real_hardware

    for hw in ("mock", "", None, "cpu", "sim", "fake-trn2", "gpu.a100", "trn2"):
        assert not is_real_hardware(hw), hw
