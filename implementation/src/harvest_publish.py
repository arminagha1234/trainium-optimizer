"""Close the last hop: verified bundles on shared storage -> the public showcase.

Why a separate step exists at all
---------------------------------
``overnight._auto_publish`` (#138) refreshes LEADERBOARD.md, the README region and the
per-model route READMEs at the end of every cycle -- but only inside the checkout it is
handed, and it does not push. On a Kaizen pod that checkout is ``/tmp/to``, which dies
with the pod, and the pod could not push regardless: it has no deploy key, and pushes
to this repo go through the GitHub Git Data API.

So a run could earn a row and the row would still never appear. This module is the
missing hop. Runs copy their verified bundles to shared storage; this collects them,
merges them into a checkout, re-renders the showcase FROM the bundles, and reports what
qualifies so a caller can push.

It moves bundles. It does not decide what deserves a row: that stays with
``publish_deliverables.collect_verified``, which requires ``verified``,
``speedup > 1.0``, a bundle present on disk, and real Neuron hardware (#139).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

__all__ = ["StagedBundle", "find_staged_bundles", "merge_bundles",
           "render_showcase", "changed_paths"]


@dataclass(frozen=True)
class StagedBundle:
    """A recipe bundle found on shared storage, and where it belongs in the repo."""
    recipe: Path          # .../optimized_models/<slug>[/<hardware>]/recipe.json
    rel_dir: Path         # <slug>[/<hardware>]


def find_staged_bundles(stage_root: Path | str) -> list[StagedBundle]:
    """Every bundle under ``stage_root``, with its repo-relative destination.

    Bundles arrive in two layouts, because runs of different vintages copied them
    differently: ``<band>/optimized_models/...`` and
    ``<band>/artifacts/optimized_models/...``. Both are keyed off the LAST
    ``optimized_models`` component, so the band directory above it is discarded and
    the slug/hardware path below it is preserved.
    """
    root = Path(stage_root)
    out: list[StagedBundle] = []
    for recipe in sorted(root.rglob("recipe.json")):
        parts = recipe.parts
        if "optimized_models" not in parts:
            continue
        i = len(parts) - 1 - parts[::-1].index("optimized_models")
        rel = Path(*parts[i + 1:-1])
        if rel.parts:
            out.append(StagedBundle(recipe=recipe, rel_dir=rel))
    return out


def merge_bundles(bundles: list[StagedBundle], dest_root: Path | str) -> list[str]:
    """Copy each bundle into ``dest_root/<rel_dir>``. Returns the rel_dirs written.

    A later bundle for the same (slug, hardware) overwrites an earlier one: two runs
    of the same model on the same box are the same row, and the newest measurement is
    the one to keep. Which of two DIFFERENT results wins is not decided here --
    collect_verified keeps the faster one.
    """
    dest_root = Path(dest_root)
    written: list[str] = []
    for b in bundles:
        dst = dest_root / b.rel_dir
        dst.mkdir(parents=True, exist_ok=True)
        for f in b.recipe.parent.iterdir():
            if f.is_file():
                target = dst / f.name
                shutil.copy2(f, target)
                # A reproduce script that is not executable is a broken instruction:
                # every RECIPE.md says to run `./reproduce.sh`. Bundles picked up from
                # shared storage have whatever mode they were written with -- staging
                # through tar, cp or an older run loses the bit -- so restore it here
                # rather than trusting the source. Harvesting a stale bundle silently
                # un-executabled eight of these in the live repo.
                if target.suffix == ".sh":
                    target.chmod(target.stat().st_mode | 0o755)
        written.append(str(b.rel_dir))
    return written


def render_showcase(repo_dir: Path | str) -> dict:
    """Re-render LEADERBOARD.md, the README region and the route READMEs.

    Returns ``{"qualified": [...], "skipped": [...]}``. Writes nothing when nothing
    qualifies, so an empty harvest cannot blank an existing leaderboard.
    """
    from publish_deliverables import (
        _skipped_report, collect_verified, group_by_model, render_leaderboard,
        render_model_readme, update_readme_region,
    )

    repo_dir = Path(repo_dir)
    root = repo_dir / "optimized_models"
    deliverables = collect_verified(root)
    result = {
        "qualified": [{"slug": d.slug, "hardware": d.hardware,
                       "speedup": d.speedup, "best": d.best,
                       "rel_dir": d.rel_dir} for d in deliverables],
        "skipped": _skipped_report(root),
    }
    if not deliverables:
        return result

    (repo_dir / "LEADERBOARD.md").write_text(render_leaderboard(deliverables))
    readme = repo_dir / "README.md"
    if readme.exists():
        readme.write_text(update_readme_region(readme.read_text(), deliverables))
    for slug, rows in group_by_model(deliverables).items():
        model_root = root / slug
        if not model_root.is_dir():
            model_root = root / Path(rows[0].rel_dir).parts[0]
        model_root.mkdir(parents=True, exist_ok=True)
        (model_root / "README.md").write_text(render_model_readme(slug, rows))
    return result


def changed_paths(repo_dir: Path | str, qualified: list[dict]) -> list[str]:
    """Repo-relative paths a push should carry: the showcase plus qualifying bundles.

    Scoped to the models that qualified, so a harvest never sweeps up unrelated files
    that happen to be sitting in the checkout.
    """
    repo_dir = Path(repo_dir)
    paths = [p for p in ("LEADERBOARD.md", "README.md") if (repo_dir / p).exists()]
    for q in qualified:
        d = repo_dir / "optimized_models" / q["rel_dir"]
        paths += [str(f.relative_to(repo_dir)) for f in sorted(d.rglob("*"))
                  if f.is_file()]
        readme = repo_dir / "optimized_models" / Path(q["rel_dir"]).parts[0] / "README.md"
        if readme.exists():
            paths.append(str(readme.relative_to(repo_dir)))
    return sorted(set(paths))
