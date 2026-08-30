"""Harvesting must be able to run unattended without ever inventing a row.

A run can earn a verified result and still never publish it: overnight._auto_publish
writes into the checkout it is handed, which on a Kaizen pod is /tmp/to and dies with
the pod, and the pod cannot push anyway. This module is that missing hop, so its tests
care about two things -- that a real bundle survives the trip intact, and that nothing
else does.
"""

from __future__ import annotations

import json
from pathlib import Path

from harvest_publish import (changed_paths, find_staged_bundles, merge_bundles,
                            render_showcase)


def _recipe(model_id, *, speedup, hardware="trn2.48xlarge", verified="verified"):
    return {
        "model_id": model_id, "backend": "native-pytorch-beta3",
        "baseline_metric": 343.4, "best_metric": 343.4 * speedup,
        "speedup": speedup, "metric_label": "tok/s",
        "config": {"tp_degree": 24, "weights_dtype": "bf16", "batch": 8},
        "toolchain": {"instance_type": hardware}, "verified": verified,
    }


def _stage(root: Path, band: str, rel: str, recipe: dict, *, nested_artifacts=False):
    mid = "artifacts/optimized_models" if nested_artifacts else "optimized_models"
    d = root / band / mid / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(json.dumps(recipe, indent=2))
    (d / "results.tsv").write_text("commit\tstage\n")
    (d / "optimization_timeline.png").write_bytes(b"\x89PNG\r\n")
    return d


# --- finding bundles across the layouts runs actually produced -----------------

def test_both_bundle_layouts_are_found(tmp_path):
    """Runs of different vintages copied to <band>/optimized_models and to
    <band>/artifacts/optimized_models. Missing one silently loses results."""
    _stage(tmp_path, "sweep_mid2", "qwen3-8-27b/trn2-48xlarge",
           _recipe("Qwen/Qwen3.8-27B", speedup=1.4))
    _stage(tmp_path, "q27b", "qwen3-8-27b/trn2-48xlarge",
           _recipe("Qwen/Qwen3.8-27B", speedup=1.2), nested_artifacts=True)
    found = find_staged_bundles(tmp_path)
    assert len(found) == 2
    assert {str(b.rel_dir) for b in found} == {"qwen3-8-27b/trn2-48xlarge"}


def test_the_band_directory_is_discarded_but_slug_and_hardware_are_kept(tmp_path):
    _stage(tmp_path, "sweep_huge2", "qwen3-235b-a22b/trn2-48xlarge",
           _recipe("Qwen/Qwen3-235B-A22B", speedup=1.1))
    b = find_staged_bundles(tmp_path)[0]
    assert str(b.rel_dir) == "qwen3-235b-a22b/trn2-48xlarge"
    assert "sweep_huge2" not in str(b.rel_dir)


def test_a_flat_bundle_without_a_hardware_level_still_works(tmp_path):
    _stage(tmp_path, "band", "gpt2", _recipe("openai-community/gpt2", speedup=2.0))
    assert str(find_staged_bundles(tmp_path)[0].rel_dir) == "gpt2"


def test_nothing_staged_finds_nothing(tmp_path):
    assert find_staged_bundles(tmp_path) == []


# --- merging ------------------------------------------------------------------

def test_a_merged_bundle_keeps_every_file(tmp_path):
    _stage(tmp_path / "stage", "b", "m/trn2-48xlarge",
           _recipe("Qwen/Qwen3.8-27B", speedup=1.4))
    repo = tmp_path / "repo" / "optimized_models"
    merge_bundles(find_staged_bundles(tmp_path / "stage"), repo)
    d = repo / "m" / "trn2-48xlarge"
    for f in ("recipe.json", "results.tsv", "optimization_timeline.png"):
        assert (d / f).exists(), f


def test_a_rerun_of_the_same_model_on_the_same_box_overwrites(tmp_path):
    """Two runs of one model on one box are the same row, not two."""
    stage = tmp_path / "stage"
    _stage(stage, "old", "m/trn2-48xlarge", _recipe("M/m", speedup=1.1))
    repo = tmp_path / "repo" / "optimized_models"
    merge_bundles(find_staged_bundles(stage), repo)
    stage2 = tmp_path / "stage2"
    _stage(stage2, "new", "m/trn2-48xlarge", _recipe("M/m", speedup=1.9))
    merge_bundles(find_staged_bundles(stage2), repo)
    got = json.loads((repo / "m" / "trn2-48xlarge" / "recipe.json").read_text())
    assert got["speedup"] == 1.9


# --- rendering: the gates must still be the gates -----------------------------

def _repo_with(tmp_path, *recipes):
    repo = tmp_path / "repo"
    (repo / "optimized_models").mkdir(parents=True)
    (repo / "README.md").write_text("# repo\n")
    stage = tmp_path / "stage"
    for i, (rel, rec) in enumerate(recipes):
        _stage(stage, f"band{i}", rel, rec)
    merge_bundles(find_staged_bundles(stage), repo / "optimized_models")
    return repo


def test_a_verified_win_lands_on_the_leaderboard(tmp_path):
    repo = _repo_with(tmp_path, ("qwen3-8-27b/trn2-48xlarge",
                                 _recipe("Qwen/Qwen3.8-27B", speedup=1.4)))
    res = render_showcase(repo)
    assert [q["slug"] for q in res["qualified"]] == ["qwen3-8-27b"]
    board = (repo / "LEADERBOARD.md").read_text()
    assert "Qwen3.8-27B" in board and "trn2.48xlarge" in board
    assert "1.400" in board or "1.4" in board


def test_a_verified_baseline_at_speedup_one_is_not_a_row(tmp_path):
    """The actual Qwen3.8-27B situation: grader-verified 343 tok/s, nothing faster
    found. A verified baseline is a datapoint, not a win."""
    repo = _repo_with(tmp_path, ("qwen3-8-27b/trn2-48xlarge",
                                 _recipe("Qwen/Qwen3.8-27B", speedup=1.0)))
    res = render_showcase(repo)
    assert res["qualified"] == []
    assert not (repo / "LEADERBOARD.md").exists()


def test_an_unverified_result_is_not_a_row(tmp_path):
    repo = _repo_with(tmp_path, ("m/trn2-48xlarge",
                                 _recipe("M/m", speedup=2.0, verified="unverified")))
    assert render_showcase(repo)["qualified"] == []


def test_a_synthetic_instance_type_is_not_a_row(tmp_path):
    repo = _repo_with(tmp_path, ("m/mock", _recipe("M/m", speedup=99.0,
                                                   hardware="mock")))
    assert render_showcase(repo)["qualified"] == []


def test_an_empty_harvest_cannot_blank_an_existing_leaderboard(tmp_path):
    """The dangerous failure: harvesting nothing must not erase the board."""
    repo = _repo_with(tmp_path, ("m/trn2-48xlarge",
                                 _recipe("M/m", speedup=1.0)))
    (repo / "LEADERBOARD.md").write_text("# existing board\n| row |\n")
    render_showcase(repo)
    assert "existing board" in (repo / "LEADERBOARD.md").read_text()


def test_two_boxes_for_one_model_produce_two_rows_and_one_route_readme(tmp_path):
    repo = _repo_with(
        tmp_path,
        ("m/trn2-3xlarge", _recipe("M/m", speedup=1.8, hardware="trn2.3xlarge")),
        ("m/trn2-48xlarge", _recipe("M/m", speedup=1.3, hardware="trn2.48xlarge")),
    )
    res = render_showcase(repo)
    assert len(res["qualified"]) == 2
    md = (repo / "optimized_models" / "m" / "README.md").read_text()
    assert "2 verified route(s)" in md


def test_skipped_results_are_reported_with_a_reason(tmp_path):
    repo = _repo_with(tmp_path, ("m/trn2-48xlarge",
                                 _recipe("M/m", speedup=1.0)))
    reasons = " ".join(r for _, r in render_showcase(repo)["skipped"])
    assert "speedup" in reasons or "1.0" in reasons


# --- what a push should carry -------------------------------------------------

def test_the_push_is_scoped_to_the_showcase_and_the_qualifying_bundles(tmp_path):
    repo = _repo_with(tmp_path, ("qwen3-8-27b/trn2-48xlarge",
                                 _recipe("Qwen/Qwen3.8-27B", speedup=1.4)))
    (repo / "optimized_models" / "unrelated").mkdir()
    (repo / "optimized_models" / "unrelated" / "junk.txt").write_text("x")
    res = render_showcase(repo)
    paths = changed_paths(repo, res["qualified"])
    assert "LEADERBOARD.md" in paths and "README.md" in paths
    assert any("qwen3-8-27b/trn2-48xlarge/recipe.json" in p for p in paths)
    assert not any("unrelated" in p for p in paths), paths


def test_the_route_readme_is_included_in_the_push(tmp_path):
    repo = _repo_with(tmp_path, ("qwen3-8-27b/trn2-48xlarge",
                                 _recipe("Qwen/Qwen3.8-27B", speedup=1.4)))
    res = render_showcase(repo)
    paths = changed_paths(repo, res["qualified"])
    assert "optimized_models/qwen3-8-27b/README.md" in paths
