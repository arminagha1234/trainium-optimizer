"""
publish_deliverables.py — durable auto-publisher for the optimized-model showcase.

Every VERIFIED optimized model (the recipe bundle the loop writes under
``optimized_models/<slug>/``) plus a freshly regenerated leaderboard is pushed
to GitHub automatically, so the showcase never has to be PR'd by hand again.

Modelled on the existing ``bank_publish`` durability pattern:

  * Least privilege — pushes with a repo-scoped SSH **deploy key** only, never a
    PAT. ``GIT_SSH_COMMAND`` pins ``IdentitiesOnly=yes`` so no other agent/key is
    ever offered.
  * Blast radius — pushes to ``main`` but only ever ``git add``s the showcase
    surface: ``optimized_models/**``, ``LEADERBOARD.md``, and the README's
    marker-delimited leaderboard table. It NEVER stages framework/code files, so
    a runaway publisher cannot clobber the optimizer itself.
  * No-op when unchanged — if git sees no staged diff, nothing is committed or
    pushed. Idempotent by construction.
  * Survives main advancing — ``git pull --rebase`` before every push.
  * ``flock`` — two runs can never race the same checkout.
  * Separate checkout — the publisher works out of its own clone, never the live
    working tree the optimizer loop is writing into.

The pure, offline-testable core (``collect_verified`` / ``render_leaderboard`` /
``render_readme_table`` / ``update_readme_region``) has no git or network
dependency and is unit-tested with mock ``recipe.json`` fixtures.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Box defaults (all overridable via CLI args or env). These match the .211 box.
# --------------------------------------------------------------------------- #
DEFAULT_REPO_DIR = os.environ.get(
    "PUB_REPO_DIR", "/home/ubuntu/trainium-optimizer-publish"
)
DEFAULT_DEPLOY_KEY = os.environ.get(
    "PUB_DEPLOY_KEY", "/home/ubuntu/.ssh/gh_optimized_models_deploy"
)
DEFAULT_OPTIMIZED_MODELS_DIR = os.environ.get(
    "PUB_OPTIMIZED_MODELS_DIR", "/home/ubuntu/trainium-optimizer/optimized_models"
)
DEFAULT_LOCK_PATH = os.environ.get(
    "PUB_LOCK_PATH", "/tmp/publish_deliverables.lock"
)
DEFAULT_SSH_URL = os.environ.get(
    "PUB_SSH_URL", "git@github.com:arminagha1234/trainium-optimizer.git"
)
DEFAULT_BRANCH = os.environ.get("PUB_BRANCH", "main")

AUTHOR_NAME = "arminagha1234"
AUTHOR_EMAIL = "arminagha1234@users.noreply.github.com"

# The only paths this publisher is ever allowed to stage. Enforced with an
# explicit `git add <these>` — we never `git add -A`.
ALLOWED_PATHS = ("optimized_models", "LEADERBOARD.md", "README.md")

MARKER_START = "<!-- LEADERBOARD:START -->"
MARKER_END = "<!-- LEADERBOARD:END -->"

_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Deliverable:
    """One verified, faster-than-baseline optimized model ready for the showcase."""

    slug: str
    model_id: str
    display_name: str
    family: str
    params: str
    speedup: float
    baseline: float
    best: float
    metric_label: str
    config: dict[str, Any]
    config_summary: str
    hardware: str
    verified: str
    rel_dir: str  # path of the recipe folder relative to optimized_models_dir
    source_dir: Path  # absolute source folder (for rsync)
    chart_exists: bool


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O beyond reading recipe.json)
# --------------------------------------------------------------------------- #
def _slug(model_id: str) -> str:
    """Canonical slug from a model id — matches publish.py's convention."""
    return model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")


def _params_from_model_id(model_id: str, recipe: dict[str, Any]) -> str:
    """Prefer an explicit param_count; otherwise sniff the size token (e.g. 1.7B)."""
    pc = recipe.get("param_count")
    if pc:
        try:
            b = float(pc) / 1e9
            return f"{b:g}B"
        except (TypeError, ValueError):
            return str(pc)
    name = model_id.split("/")[-1]
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
    return f"{m.group(1)}B" if m else "—"


def _family_from_model_id(model_id: str) -> str:
    """qwen3-0.6b -> qwen3 ; qwen2.5-0.5b-instruct -> qwen2.5."""
    name = model_id.split("/")[-1].lower()
    m = re.match(r"^([a-z]+\d*(?:\.\d+)?)", name)
    return m.group(1) if m else name


def _config_summary(config: dict[str, Any]) -> str:
    """Compact, human-readable one-liner of the winning config for the table."""
    parts: list[str] = []
    tp = config.get("tp_degree")
    if tp:
        parts.append(f"TP={tp}")
    cm = str(config.get("compile_mode", ""))
    if cm.startswith("compile"):
        parts.append("torch.compile(neuron)")
    dtype = config.get("weights_dtype")
    if dtype:
        parts.append(str(dtype))
    batch = config.get("batch")
    if batch:
        parts.append(f"batch={batch}")
    for axis in ("dp_degree", "cp_degree"):
        v = config.get(axis)
        if isinstance(v, (int, float)) and v > 1:
            parts.append(f"{axis.split('_')[0].upper()}={v}")
    return ", ".join(parts) if parts else "(config-only)"


def _num(recipe: dict[str, Any], *keys: str) -> Optional[float]:
    """First present numeric value among alias keys (handles field-name drift)."""
    for k in keys:
        if k in recipe and recipe[k] is not None:
            try:
                return float(recipe[k])
            except (TypeError, ValueError):
                continue
    return None


def _deliverable_from_recipe(
    recipe_path: Path, optimized_models_dir: Path
) -> Optional[Deliverable]:
    """Build a Deliverable from a recipe.json, or None if it isn't a verified win."""
    try:
        recipe = json.loads(recipe_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if str(recipe.get("verified", "")).strip().lower() != "verified":
        return None

    baseline = _num(recipe, "baseline_metric", "baseline")
    best = _num(recipe, "best_metric", "best")
    speedup = _num(recipe, "speedup")
    if speedup is None and baseline and best:
        speedup = best / baseline
    if speedup is None or speedup <= 1.0:
        return None

    model_id = recipe.get("model_id", "")
    if not model_id:
        return None

    source_dir = recipe_path.parent
    config = recipe.get("config", {}) or {}
    toolchain = recipe.get("toolchain", {}) or {}
    return Deliverable(
        slug=_slug(model_id),
        model_id=model_id,
        display_name=model_id.split("/")[-1],
        family=_family_from_model_id(model_id),
        params=_params_from_model_id(model_id, recipe),
        speedup=round(float(speedup), 3),
        baseline=baseline or 0.0,
        best=best or 0.0,
        metric_label=recipe.get("metric_label", "tok/s"),
        config=config,
        config_summary=_config_summary(config),
        hardware=toolchain.get("instance_type", "trn2.3xlarge"),
        verified="verified",
        rel_dir=str(source_dir.relative_to(optimized_models_dir)),
        source_dir=source_dir,
        chart_exists=(source_dir / "optimization_timeline.png").exists(),
    )


def collect_verified(optimized_models_dir: Path | str) -> list[Deliverable]:
    """Scan ``optimized_models`` for recipe bundles and keep only the wins.

    Keeps a recipe iff ``verified == "verified"`` AND ``speedup > 1.0`` —
    unverified / failed / 0.0-speedup recipes are skipped (the showcase is
    *successes*). Recurses so it works for both the flat ``<slug>/`` layout the
    loop writes and any legacy nested ``<family>/<model>/`` layout. If the same
    ``model_id`` appears twice, the faster recipe wins (mirrors publish.py's
    beat-gate).
    """
    root = Path(optimized_models_dir)
    if not root.exists():
        return []
    best_by_model: dict[str, Deliverable] = {}
    for recipe_path in sorted(root.rglob("recipe.json")):
        d = _deliverable_from_recipe(recipe_path, root)
        if d is None:
            continue
        prev = best_by_model.get(d.model_id)
        if prev is None or d.speedup > prev.speedup or (
            d.speedup == prev.speedup and d.best > prev.best
        ):
            best_by_model[d.model_id] = d
    return sorted(
        best_by_model.values(), key=lambda x: (x.speedup, x.best), reverse=True
    )


def _skipped_report(optimized_models_dir: Path | str) -> list[tuple[str, str]]:
    """(rel_dir, reason) for every recipe that was NOT published — honest logging."""
    root = Path(optimized_models_dir)
    skipped: list[tuple[str, str]] = []
    if not root.exists():
        return skipped
    for recipe_path in sorted(root.rglob("recipe.json")):
        try:
            recipe = json.loads(recipe_path.read_text())
        except (json.JSONDecodeError, OSError):
            skipped.append((str(recipe_path.parent), "unreadable recipe.json"))
            continue
        rel = str(recipe_path.parent.relative_to(root))
        v = str(recipe.get("verified", "")).strip().lower()
        if v != "verified":
            skipped.append((rel, f"not verified (verified={recipe.get('verified')!r})"))
            continue
        baseline = _num(recipe, "baseline_metric", "baseline")
        best = _num(recipe, "best_metric", "best")
        speedup = _num(recipe, "speedup")
        if speedup is None and baseline and best:
            speedup = best / baseline
        if speedup is None or speedup <= 1.0:
            skipped.append((rel, f"speedup <= 1.0 (speedup={speedup})"))
    return skipped


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _rank_label(i: int) -> str:
    return _MEDALS.get(i, str(i))


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def render_readme_table(deliverables: list[Deliverable]) -> str:
    """The markdown table that lives between the README leaderboard markers."""
    header = (
        "| Rank | Model | Family | Params | Baseline (tok/s) | "
        "Optimized (tok/s) | Speedup | Best config | Hardware | Status |\n"
        "|-----:|:------|:-------|-------:|-----------------:|"
        "------------------:|--------:|:------------|:-------------|:-------|"
    )
    rows = [header]
    for i, d in enumerate(deliverables, start=1):
        rows.append(
            f"| {_rank_label(i)} | {d.display_name} | {d.family} | {d.params} | "
            f"{_fmt(d.baseline)} | **{_fmt(d.best)}** | **{d.speedup:g}×** | "
            f"{d.config_summary} | {d.hardware} | ✅ Verified |"
        )
    return "\n".join(rows)


def render_leaderboard(deliverables: list[Deliverable]) -> str:
    """Regenerate the full standalone ``LEADERBOARD.md`` (sorted, medals top 3)."""
    lines = [
        "# Trainium Optimizer — Leaderboard",
        "",
        "Verified optimized models, one row per model, sorted by speedup over the "
        "eager baseline on real Trainium hardware (`native-pytorch-beta3`). "
        "Auto-published by the optimizer loop — do not edit by hand.",
        "",
        "Recipes and trajectory charts live under "
        "[`optimized_models/`](./optimized_models/) — each folder holds "
        "`recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and "
        "`optimization_timeline.png`.",
        "",
        "| Rank | Model | Family | Params | Baseline (tok/s) | Best (tok/s) | "
        "Speedup | Best config | Hardware | Verified | Recipe |",
        "|-----:|:------|:-------|-------:|-----------------:|-------------:|"
        "--------:|:------------|:-------------|:---------|:-------|",
    ]
    for i, d in enumerate(deliverables, start=1):
        recipe_link = f"[recipe](./optimized_models/{d.rel_dir}/)"
        lines.append(
            f"| {_rank_label(i)} | {d.display_name} | {d.family} | {d.params} | "
            f"{_fmt(d.baseline)} | **{_fmt(d.best)}** | **{d.speedup:g}×** | "
            f"{d.config_summary} | {d.hardware} | ✅ verified | {recipe_link} |"
        )
    lines += [
        "",
        f"{len(deliverables)} verified model(s). "
        "Speedup is measured against the eager baseline on the same instance and "
        "probe shape. See [`HISTORY.tsv`](./HISTORY.tsv) for the append-only record.",
        "",
    ]
    return "\n".join(lines)


def update_readme_region(readme_text: str, deliverables: list[Deliverable]) -> str:
    """Replace only the marker-delimited leaderboard table in the README.

    If the markers are absent (first run), they are inserted around the existing
    leaderboard markdown table so nothing else in the README (chart embed,
    Current State, legend, prose) is touched. Idempotent: running it again with
    the same deliverables yields byte-identical text (→ git no-op).
    """
    table = render_readme_table(deliverables)
    block = f"{MARKER_START}\n{table}\n{MARKER_END}"

    if MARKER_START in readme_text and MARKER_END in readme_text:
        pattern = re.compile(
            re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END), re.DOTALL
        )
        return pattern.sub(lambda _m: block, readme_text, count=1)

    # First run: find the existing pipe-table and wrap it.
    lines = readme_text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            # walk the contiguous run of table rows
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            start, end = i, j
            break
    if start is None:
        # No table to wrap — insert after the leaderboard heading, else prepend.
        for i, line in enumerate(lines):
            if "Text-to-text" in line or "🏆" in line:
                insert_at = i + 1
                lines[insert_at:insert_at] = ["", block]
                return "\n".join(lines) + ("\n" if readme_text.endswith("\n") else "")
        return block + "\n\n" + readme_text

    new_lines = lines[:start] + [block] + lines[end:]
    return "\n".join(new_lines) + ("\n" if readme_text.endswith("\n") else "")


# --------------------------------------------------------------------------- #
# No-op detection (content level, git-free — offline testable)
# --------------------------------------------------------------------------- #
def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except OSError:
        return None


def would_change(
    repo_dir: Path | str,
    deliverables: list[Deliverable],
    *,
    check_recipes: bool = True,
) -> list[str]:
    """Which showcase files the render would change vs. what's on disk in the
    checkout. Empty list == no-op. Compares LEADERBOARD.md, the README table
    region, and (optionally) each verified recipe.json's content.
    """
    repo_dir = Path(repo_dir)
    changed: list[str] = []

    new_leaderboard = render_leaderboard(deliverables)
    if _read(repo_dir / "LEADERBOARD.md") != new_leaderboard:
        changed.append("LEADERBOARD.md")

    readme = _read(repo_dir / "README.md")
    if readme is not None:
        if update_readme_region(readme, deliverables) != readme:
            changed.append("README.md")

    if check_recipes:
        for d in deliverables:
            dest = repo_dir / "optimized_models" / d.rel_dir / "recipe.json"
            if _read(dest) != _read(d.source_dir / "recipe.json"):
                changed.append(f"optimized_models/{d.rel_dir}/")
    return changed


# --------------------------------------------------------------------------- #
# Git plumbing (deploy-key, least privilege)
# --------------------------------------------------------------------------- #
def _ssh_env(deploy_key: str) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_SSH_COMMAND"] = (
        f"ssh -i {deploy_key} -o IdentitiesOnly=yes "
        f"-o StrictHostKeyChecking=no -o BatchMode=yes"
    )
    return env


def _git(args: list[str], cwd: Path, env: dict[str, str], check: bool = True):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), env=env,
        capture_output=True, text=True, check=check,
    )


def _log(msg: str) -> None:
    print(f"[publish_deliverables {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}",
          flush=True)


def _ensure_checkout(repo_dir: Path, ssh_url: str, branch: str, env: dict[str, str]) -> None:
    if not (repo_dir / ".git").exists():
        _log(f"cloning {ssh_url} -> {repo_dir} (deploy key)")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--branch", branch, ssh_url, str(repo_dir)],
            env=env, capture_output=True, text=True, check=True,
        )
    else:
        # Pin the remote to the SSH URL so pushes always use the deploy key.
        _git(["remote", "set-url", "origin", ssh_url], repo_dir, env)


# --------------------------------------------------------------------------- #
# publish()
# --------------------------------------------------------------------------- #
def publish(
    repo_dir: Path | str,
    deploy_key: str,
    branch: str = "main",
    *,
    optimized_models_dir: Path | str = DEFAULT_OPTIMIZED_MODELS_DIR,
    ssh_url: str = DEFAULT_SSH_URL,
    lock_path: str = DEFAULT_LOCK_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish verified deliverables + refreshed leaderboard to ``branch``.

    Safe by construction: separate checkout, ``flock``, ``pull --rebase`` before
    push, scoped ``git add`` (only ``optimized_models/**`` + ``LEADERBOARD.md`` +
    README table region), deploy-key push, and a no-op when git sees no diff.
    Returns a result dict for the loop to log.
    """
    repo_dir = Path(repo_dir)
    optimized_models_dir = Path(optimized_models_dir)
    env = _ssh_env(deploy_key)

    result: dict[str, Any] = {
        "committed": False, "pushed": False, "noop": False,
        "published": [], "skipped": [], "dry_run": dry_run,
    }

    # Single-instance / anti-race lock.
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _log("another publish run holds the lock — exiting")
        result["noop"] = True
        result["reason"] = "locked"
        return result

    try:
        deliverables = collect_verified(optimized_models_dir)
        skipped = _skipped_report(optimized_models_dir)
        result["published"] = [d.slug for d in deliverables]
        result["skipped"] = skipped

        _log(f"{len(deliverables)} verified deliverable(s): "
             f"{', '.join(f'{d.slug}={d.speedup:g}x' for d in deliverables) or '(none)'}")
        for rel, reason in skipped:
            _log(f"  skip {rel}: {reason}")

        if not deliverables:
            _log("nothing verified to publish — no-op")
            result["noop"] = True
            return result

        if not dry_run:
            _ensure_checkout(repo_dir, ssh_url, branch, env)
            _git(["checkout", branch], repo_dir, env)
            _git(["pull", "--rebase", "origin", branch], repo_dir, env)

        # 1. rsync each verified recipe folder into the checkout.
        dest_models = repo_dir / "optimized_models"
        dest_models.mkdir(parents=True, exist_ok=True)
        for d in deliverables:
            dest = dest_models / d.rel_dir
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["rsync", "-a", f"{d.source_dir}/", f"{dest}/"],
                capture_output=True, text=True, check=True,
            )

        # 2. LEADERBOARD.md + README table region.
        (repo_dir / "LEADERBOARD.md").write_text(render_leaderboard(deliverables))
        readme_path = repo_dir / "README.md"
        if readme_path.exists():
            readme_path.write_text(
                update_readme_region(readme_path.read_text(), deliverables)
            )

        if dry_run:
            changed = would_change(repo_dir, deliverables)
            _log(f"dry-run: would change {changed or '(nothing)'}")
            result["noop"] = not changed
            result["changed"] = changed
            return result

        # 3. Scoped stage — ONLY the showcase surface, never code files.
        _git(["add", "--", *ALLOWED_PATHS], repo_dir, env)
        staged = _git(["diff", "--cached", "--quiet"], repo_dir, env, check=False)
        if staged.returncode == 0:
            _log("no staged diff — nothing changed, no-op (not committing/pushing)")
            result["noop"] = True
            return result

        # 4. Commit as the showcase author, then rebase-safe push via deploy key.
        n = len(deliverables)
        msg = f"auto-publish: {n} verified model{'s' if n != 1 else ''} + leaderboard"
        commit_env = dict(env)
        commit_env.update({
            "GIT_AUTHOR_NAME": AUTHOR_NAME, "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": AUTHOR_NAME, "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
        })
        _git(["commit", "-m", msg], repo_dir, commit_env)
        result["committed"] = True
        _log(f"committed: {msg}")

        _git(["pull", "--rebase", "origin", branch], repo_dir, env)  # survive main advancing
        _git(["push", "origin", f"HEAD:{branch}"], repo_dir, env)
        result["pushed"] = True
        _log(f"pushed to origin/{branch}")
        return result

    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    p.add_argument("--deploy-key", default=DEFAULT_DEPLOY_KEY)
    p.add_argument("--optimized-models-dir", default=DEFAULT_OPTIMIZED_MODELS_DIR)
    p.add_argument("--ssh-url", default=DEFAULT_SSH_URL)
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--lock-path", default=DEFAULT_LOCK_PATH)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = publish(
        repo_dir=args.repo_dir,
        deploy_key=args.deploy_key,
        branch=args.branch,
        optimized_models_dir=args.optimized_models_dir,
        ssh_url=args.ssh_url,
        lock_path=args.lock_path,
        dry_run=args.dry_run,
    )
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
