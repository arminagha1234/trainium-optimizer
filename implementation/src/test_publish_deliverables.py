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
        # sorted by speedup desc
        speedups = [d.speedup for d in dels]
        assert speedups == sorted(speedups, reverse=True)
        assert dels[0].model_id == "Qwen/Qwen3-0.6B"
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
        assert dels[0].speedup == 27.05


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
        # top of board is the fastest
        assert md.index("Qwen3-0.6B") < md.index("Qwen2.5-0.5B-Instruct")
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
