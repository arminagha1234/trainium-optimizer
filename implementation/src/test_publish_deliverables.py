"""
Tests for publish_deliverables — the optimized-model auto-publisher.

Everything here runs on a plain CPU box, offline, with mock recipe.json
fixtures. No git, no network, no Neuron SDK. We cover the three durability-
critical pure functions:

  * collect_verified   — verified-only + speedup>1.0 filter, dedup-by-model.
  * render_leaderboard — speedup-desc sort, medal emojis, failed-model exclusion.
  * update_readme_region — marker insertion/replacement + idempotency (no-op).
  * would_change       — content-level no-op detection.

The file is runnable under pytest (`python -m pytest -q`) AND standalone
(`python test_publish_deliverables.py`) so it can be exercised without pytest.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from publish_deliverables import (
    MARKER_END,
    MARKER_START,
    collect_verified,
    render_leaderboard,
    render_readme_table,
    update_readme_region,
    would_change,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _recipe(model_id: str, *, speedup, baseline, best, verified="verified", config=None):
    return {
        "model_id": model_id,
        "backend": "native-pytorch-beta3",
        "baseline_metric": baseline,
        "best_metric": best,
        "speedup": speedup,
        "metric_label": "tok/s",
        "config": config or {
            "tp_degree": 4, "weights_dtype": "bf16",
            "compile_mode": "compile-default", "batch": 8,
        },
        "toolchain": {"instance_type": "trn2.3xlarge"},
        "verified": verified,
    }


def _write_bundle(root: Path, slug: str, recipe: dict, *, chart=True):
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(json.dumps(recipe, indent=2))
    if chart:
        (d / "optimization_timeline.png").write_bytes(b"\x89PNG\r\n")
    return d


def _standard_tree(root: Path):
    """A mix: verified wins (incl. the NEW Qwen3-1.7B ~17.2x), an unverified one,
    and a failed 0.0-speedup one. Only the wins should ever publish."""
    _write_bundle(root, "qwen3-0-6b",
                  _recipe("Qwen/Qwen3-0.6B", speedup=27.05, baseline=3085.1, best=83450.3))
    _write_bundle(root, "qwen2-5-0-5b-instruct",
                  _recipe("Qwen/Qwen2.5-0.5B-Instruct", speedup=15.368,
                          baseline=4832.6, best=74268.5))
    _write_bundle(root, "qwen3-4b",
                  _recipe("Qwen/Qwen3-4B", speedup=13.426, baseline=1974.3, best=26507.3))
    # The NEW verified model the task calls out — must appear on the board.
    _write_bundle(root, "qwen3-1-7b",
                  _recipe("Qwen/Qwen3-1.7B", speedup=17.2, baseline=2600.0, best=44720.0))
    # Must be excluded:
    _write_bundle(root, "unverified-model",
                  _recipe("Qwen/Qwen3-8B", speedup=9.9, baseline=1000.0, best=9900.0,
                          verified="unverified"))
    _write_bundle(root, "failed-model",
                  _recipe("Qwen/Qwen3-32B", speedup=0.0, baseline=500.0, best=0.0))


# --------------------------------------------------------------------------- #
# collect_verified
# --------------------------------------------------------------------------- #
def test_collect_verified_filters_and_sorts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _standard_tree(root)
        dels = collect_verified(root)
        ids = [d.model_id for d in dels]

        # unverified + failed excluded
        assert "Qwen/Qwen3-8B" not in ids, "unverified must be skipped"
        assert "Qwen/Qwen3-32B" not in ids, "0.0-speedup failure must be skipped"
        # exactly the four verified wins
        assert len(dels) == 4, ids
        # sorted by throughput (best tok/s) desc -- the board primary metric
        bests = [d.best for d in dels]
        assert bests == sorted(bests, reverse=True), bests
        assert dels[0].model_id == "Qwen/Qwen3-0.6B"  # highest tok/s
        # NEW model present with right params/family derivation
        q17 = next(d for d in dels if d.model_id == "Qwen/Qwen3-1.7B")
        assert q17.params == "1.7B"
        assert q17.family == "qwen3"
        assert q17.speedup == 17.2


def test_collect_verified_dedup_keeps_faster():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # same model_id in flat + nested layout — faster one must win
        _write_bundle(root, "qwen3-0-6b",
                      _recipe("Qwen/Qwen3-0.6B", speedup=25.788, baseline=3332.5, best=85937.2))
        _write_bundle(root / "qwen3", "qwen3-0-6b",
                      _recipe("Qwen/Qwen3-0.6B", speedup=27.05, baseline=3085.1, best=83450.3))
        dels = collect_verified(root)
        assert len(dels) == 1
        # throughput-primary: the higher-tok/s recipe wins even though the
        # other had a higher speedup over its own baseline
        assert dels[0].best == 85937.2
        assert dels[0].speedup == 25.788


def test_collect_verified_speedup_derived_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r = _recipe("Qwen/Qwen3-0.6B", speedup=None, baseline=1000.0, best=5000.0)
        r.pop("speedup")
        _write_bundle(root, "qwen3-0-6b", r)
        dels = collect_verified(root)
        assert len(dels) == 1
        assert dels[0].speedup == 5.0


# --------------------------------------------------------------------------- #
# render_leaderboard / render_readme_table
# --------------------------------------------------------------------------- #
def test_render_leaderboard_sort_medals_and_exclusion():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _standard_tree(root)
        dels = collect_verified(root)
        md = render_leaderboard(dels)

        assert "🥇" in md and "🥈" in md and "🥉" in md
        # two-table structure: peak-throughput headline + speedup context
        assert "## Peak throughput" in md
        assert "## Improvement over eager baseline" in md
        # the MAIN table ranks by throughput (tok/s), NOT speedup:
        # Qwen2.5-0.5B (74k tok/s, 15.4x) ranks ABOVE Qwen3-1.7B (44k tok/s,
        # 17.2x) -- the reverse of a speedup sort, pinning the tput order.
        peak = md.split("## Improvement over eager baseline")[0]
        assert (peak.index("Qwen3-0.6B") < peak.index("Qwen2.5-0.5B-Instruct")
                < peak.index("Qwen3-1.7B"))
        # NEW model appears
        assert "Qwen3-1.7B" in md
        # failed / unverified never appear
        assert "Qwen3-32B" not in md
        assert "Qwen3-8B" not in md
        # medal is on the fastest row, plain number rank further down
        assert "🥇 | Qwen3-0.6B" in md


def test_readme_table_has_expected_columns():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _standard_tree(root)
        table = render_readme_table(collect_verified(root))
        for col in ("Rank", "Model", "Family", "Params", "Baseline",
                    "Optimized", "Speedup", "Best config", "Hardware", "Status"):
            assert col in table, col
        assert "✅ Verified" in table


# --------------------------------------------------------------------------- #
# update_readme_region — marker insertion, replacement, idempotency (no-op)
# --------------------------------------------------------------------------- #
_README_NO_MARKERS = """# Title

## 🏆 Trainium Optimizer Leaderboard

### Text-to-text (LLMs)

| Rank | Model | Family |
|-----:|:------|:-------|
| 1 | OldModel | old |
| 2 | StaleRow/queued | mixed |

## Current State

This section MUST be preserved untouched.

![chart](./chart.png)
"""


def test_update_readme_inserts_markers_first_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _standard_tree(root)
        dels = collect_verified(root)
        out = update_readme_region(_README_NO_MARKERS, dels)

        assert MARKER_START in out and MARKER_END in out
        # new content replaced the old table
        assert "OldModel" not in out
        assert "StaleRow/queued" not in out
        assert "Qwen3-0.6B" in out
        # untouched regions survive
        assert "## Current State" in out
        assert "This section MUST be preserved untouched." in out
        assert "![chart](./chart.png)" in out


def test_update_readme_idempotent_noop():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _standard_tree(root)
        dels = collect_verified(root)
        once = update_readme_region(_README_NO_MARKERS, dels)
        twice = update_readme_region(once, dels)
        assert once == twice, "second render must be byte-identical (git no-op)"


def test_update_readme_replaces_between_existing_markers_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _standard_tree(root)
        dels = collect_verified(root)
        seeded = (
            "# Title\n\n## 🏆 Leaderboard\n\n"
            f"{MARKER_START}\n| stale |\n{MARKER_END}\n\n"
            "## Keep Me\n\nprose\n"
        )
        out = update_readme_region(seeded, dels)
        assert "| stale |" not in out
        assert "## Keep Me" in out and "prose" in out
        # exactly one marker pair remains
        assert out.count(MARKER_START) == 1 and out.count(MARKER_END) == 1


# --------------------------------------------------------------------------- #
# would_change — content-level no-op detection
# --------------------------------------------------------------------------- #
def test_would_change_detects_noop_after_publish():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "optimized_models"
        _standard_tree(src)
        dels = collect_verified(src)

        repo = root / "repo"
        (repo / "optimized_models").mkdir(parents=True)
        # Fresh repo has nothing yet -> everything changes.
        (repo / "README.md").write_text(_README_NO_MARKERS)
        assert would_change(repo, dels), "empty repo must report changes"

        # Simulate a publish: write leaderboard, readme region, copy recipes.
        (repo / "LEADERBOARD.md").write_text(render_leaderboard(dels))
        (repo / "README.md").write_text(
            update_readme_region((repo / "README.md").read_text(), dels)
        )
        for d in dels:
            dest = repo / "optimized_models" / d.rel_dir
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "recipe.json").write_text((d.source_dir / "recipe.json").read_text())

        # Now nothing should change -> no-op.
        assert would_change(repo, dels) == [], "identical content must be a no-op"


# --------------------------------------------------------------------------- #
# check_consistency — the anti-divergence guard (the 38-rows / 9-folders bug)
# --------------------------------------------------------------------------- #
from publish_deliverables import check_consistency, leaderboard_recipe_dirs


def test_leaderboard_recipe_dirs_parses_links():
    lb = (
        "| 🥇 | A | f | 1B | 1 | 2 | 2× | c | trn2.3xlarge | ✅ | "
        "[recipe](./optimized_models/model-a/) |\n"
        "| 2 | B | f | 1B | 1 | 2 | 2× | c | trn2.3xlarge | ✅ | "
        "[recipe](./optimized_models/fam/model-b/) |\n"
    )
    assert leaderboard_recipe_dirs(lb) == ["model-a", "fam/model-b"]
    assert leaderboard_recipe_dirs("") == []


def test_check_consistency_passes_when_every_row_has_a_bundle():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _write_bundle(repo / "optimized_models", "model-a",
                      _recipe("org/Model-A", speedup=2.0, baseline=1, best=2))
        (repo / "LEADERBOARD.md").write_text(
            "x [recipe](./optimized_models/model-a/) y")
        assert check_consistency(repo) == []


def test_check_consistency_flags_dead_links():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _write_bundle(repo / "optimized_models", "model-a",
                      _recipe("org/Model-A", speedup=2.0, baseline=1, best=2))
        # leaderboard links a-and-b, but only a has a folder -> b is a dead link
        (repo / "LEADERBOARD.md").write_text(
            "[recipe](./optimized_models/model-a/) "
            "[recipe](./optimized_models/model-b/)")
        broken = check_consistency(repo)
        assert broken == ["model-b"]


def test_check_consistency_no_leaderboard_is_clean():
    with tempfile.TemporaryDirectory() as tmp:
        assert check_consistency(Path(tmp)) == []


def test_render_leaderboard_output_is_self_consistent():
    # render_leaderboard is the source of truth: what it emits must always pass
    # check_consistency against the same bundle tree it was rendered from.
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        src = repo / "optimized_models"
        _standard_tree(src)
        dels = collect_verified(src)
        (repo / "LEADERBOARD.md").write_text(render_leaderboard(dels))
        assert check_consistency(repo) == []


def test_overnight_writes_run_summary_not_leaderboard():
    # overnight's per-cycle summary must NOT clobber the canonical LEADERBOARD.md.
    import overnight
    from dataclasses import dataclass

    @dataclass
    class _R:
        slug: str; ok: bool; baseline: float; best: float
        speedup: float; attempts: int; error: str = ""; skipped: bool = False

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        p = overnight.write_leaderboard(
            [_R("m", True, 100.0, 200.0, 2.0, 1)], out, "mock", cycle=1)
        assert p.name == "RUN_SUMMARY.md"
        assert not (out / "LEADERBOARD.md").exists()


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


# --- per-model README: "I have box X, what is the best way to run this?" -------
#
# The leaderboard answers "which model is fastest". It does not answer the question
# a reader has when they land on one model and own a particular box. A model
# optimized on both a trn2.3xlarge and a trn2.48xlarge is two independent results,
# and collect_verified already keys on (model, hardware) to keep both.

from publish_deliverables import group_by_model, render_model_readme  # noqa: E402


def _hw_recipe(model_id, hardware, *, speedup, baseline, best):
    r = _recipe(model_id, speedup=speedup, baseline=baseline, best=best)
    r["toolchain"] = {"instance_type": hardware}
    return r


def _two_route_tree(root: Path):
    """One model verified on two boxes, in the nested <slug>/<hardware>/ layout."""
    _write_bundle(root, "qwen3-5-35b-a3b/trn2-3xlarge",
                  _hw_recipe("Qwen/Qwen3.5-35B-A3B", "trn2.3xlarge",
                             speedup=1.8, baseline=100.0, best=180.0))
    _write_bundle(root, "qwen3-5-35b-a3b/trn2-48xlarge",
                  _hw_recipe("Qwen/Qwen3.5-35B-A3B", "trn2.48xlarge",
                             speedup=1.4, baseline=300.0, best=420.0))
    return root


def test_the_same_model_on_two_boxes_yields_two_routes():
    with tempfile.TemporaryDirectory() as td:
        root = _two_route_tree(Path(td))
        ds = collect_verified(root)
        groups = group_by_model(ds)
        assert len(ds) == 2, [d.hardware for d in ds]
        assert list(groups) == ["qwen3-5-35b-a3b"]
        assert {d.hardware for d in groups["qwen3-5-35b-a3b"]} == {
            "trn2.3xlarge", "trn2.48xlarge"}


def test_model_readme_lists_every_route_with_its_own_numbers():
    with tempfile.TemporaryDirectory() as td:
        groups = group_by_model(collect_verified(_two_route_tree(Path(td))))
        md = render_model_readme("qwen3-5-35b-a3b", groups["qwen3-5-35b-a3b"])
    assert "# Qwen3.5-35B-A3B" in md
    assert "2 verified route(s)" in md
    for hw in ("trn2.3xlarge", "trn2.48xlarge"):
        assert f"`{hw}`" in md
    # Each row carries that box's own baseline and best, not a shared pair.
    assert "180" in md and "420" in md
    assert "100" in md and "300" in md


def test_model_readme_warns_that_speedups_are_not_comparable_across_boxes():
    """A 1.4x on a 48xl and a 1.8x on a 3xl are not the same achievement.

    Without saying so, the table invites exactly the wrong comparison -- and the
    48xl row is the LOWER speedup while being the HIGHER absolute throughput.
    """
    with tempfile.TemporaryDirectory() as td:
        groups = group_by_model(collect_verified(_two_route_tree(Path(td))))
        md = render_model_readme("qwen3-5-35b-a3b", groups["qwen3-5-35b-a3b"])
    assert "speedup is not" in md.lower()
    rows = [ln for ln in md.splitlines() if ln.startswith("| `trn2")]
    assert "48xlarge" in rows[-1], "the lower speedup must not be ranked first"
    assert "3xlarge" in rows[0]


def test_model_readme_is_written_for_single_route_models_too():
    """So the file always exists and a second route later just adds a row."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_bundle(root, "gpt2", _recipe("openai-community/gpt2",
                                            speedup=2.0, baseline=10.0, best=20.0))
        groups = group_by_model(collect_verified(root))
        md = render_model_readme("gpt2", groups["gpt2"])
    assert "1 verified route(s)" in md
    assert "trn2.3xlarge" in md


def test_model_readme_is_empty_for_no_routes():
    assert render_model_readme("nothing", []) == ""


# ---------------------------------------------------------------------------
# Publication must not depend on tools the measuring machines do not have.
#
# `publish` shelled out to `rsync -a` to place bundles. rsync is NOT installed in
# the Neuron DLC, so on every Kaizen pod this raised FileNotFoundError(2), which
# `_auto_publish` correctly treats as non-fatal -- with the result that the
# measurements survived and the row never appeared. Qwen3.5-0.8B earned a
# grader-verified 1.045x (1,143 tok/s, drift 0.5%, equivalence ok) and was silently
# not published for exactly this reason.
#
# The failure was invisible because it happened on the pod, was caught, and was
# logged as non-fatal. So the test has to assert the property directly: publication
# is pure Python and works with an empty PATH.
# ---------------------------------------------------------------------------

def _verified_bundle(root, slug="qwen3-5-0-8b", hardware="trn2.48xlarge",
                     speedup=1.045):
    import json
    d = root / slug / hardware
    d.mkdir(parents=True, exist_ok=True)
    (d / "recipe.json").write_text(json.dumps({
        "model_id": "Qwen/Qwen3.5-0.8B", "backend": "native-pytorch-beta3",
        "baseline_metric": 1094.0, "best_metric": 1143.0, "speedup": speedup,
        "metric_label": "tok/s",
        "config": {"tp_degree": 4, "weights_dtype": "bf16"},
        "toolchain": {"instance_type": "trn2.48xlarge"}, "verified": "verified",
    }))
    (d / "results.tsv").write_text("commit\tstage\n")
    (d / "optimization_timeline.png").write_bytes(b"\x89PNG\r\n")
    return d


def test_publish_needs_no_external_binaries(tmp_path, monkeypatch):
    """With PATH emptied, publishing still works. rsync's absence broke this."""
    from publish_deliverables import publish
    src = tmp_path / "art" / "optimized_models"
    _verified_bundle(src)
    repo = tmp_path / "repo"
    (repo / "optimized_models").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")
    res = publish(repo_dir=repo, deploy_key="/nonexistent",
                  optimized_models_dir=src,
                  lock_path=str(tmp_path / "lock"), dry_run=True)
    assert res["published"] == ["qwen3-5-0-8b"], res
    assert not res.get("error")


def test_the_bundle_actually_lands_in_the_checkout(tmp_path, monkeypatch):
    """A row that qualifies is worthless if its folder never arrives."""
    from publish_deliverables import publish
    src = tmp_path / "art" / "optimized_models"
    _verified_bundle(src)
    repo = tmp_path / "repo"
    (repo / "optimized_models").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")
    publish(repo_dir=repo, deploy_key="/nonexistent", optimized_models_dir=src,
            lock_path=str(tmp_path / "lock"), dry_run=True)
    dest = repo / "optimized_models" / "qwen3-5-0-8b" / "trn2.48xlarge"
    for name in ("recipe.json", "results.tsv", "optimization_timeline.png"):
        assert (dest / name).is_file(), f"{name} did not arrive"
    assert "Qwen3.5-0.8B" in (repo / "LEADERBOARD.md").read_text()


def test_copying_a_bundle_twice_is_not_an_error(tmp_path, monkeypatch):
    """Cycles re-publish the same model; the destination already exists."""
    from publish_deliverables import publish
    src = tmp_path / "art" / "optimized_models"
    _verified_bundle(src)
    repo = tmp_path / "repo"
    (repo / "optimized_models").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")
    kw = dict(repo_dir=repo, deploy_key="/nonexistent", optimized_models_dir=src,
              lock_path=str(tmp_path / "lock"), dry_run=True)
    publish(**kw)
    res = publish(**kw)
    assert res["published"] == ["qwen3-5-0-8b"]
    assert not res.get("error")


def test_a_newer_measurement_overwrites_the_bundle_in_place(tmp_path, monkeypatch):
    from publish_deliverables import publish
    import json
    src = tmp_path / "art" / "optimized_models"
    _verified_bundle(src, speedup=1.045)
    repo = tmp_path / "repo"
    (repo / "optimized_models").mkdir(parents=True)
    monkeypatch.setenv("PATH", "")
    kw = dict(repo_dir=repo, deploy_key="/nonexistent", optimized_models_dir=src,
              lock_path=str(tmp_path / "lock"), dry_run=True)
    publish(**kw)
    _verified_bundle(src, speedup=1.31)
    publish(**kw)
    got = json.loads((repo / "optimized_models" / "qwen3-5-0-8b" /
                      "trn2.48xlarge" / "recipe.json").read_text())
    assert got["speedup"] == 1.31
