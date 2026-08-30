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

# Showcase commit identity. Overridable so this publisher is not welded to one
# person's account -- a second agent, a CI runner or a fork needs to publish
# under its own identity without editing source.
AUTHOR_NAME = os.environ.get("PUB_AUTHOR_NAME", "arminagha1234")
AUTHOR_EMAIL = os.environ.get(
    "PUB_AUTHOR_EMAIL", "arminagha1234@users.noreply.github.com"
)

# The only paths this publisher is ever allowed to stage. Enforced with an
# explicit `git add <these>` — we never `git add -A`.
ALLOWED_PATHS = ("optimized_models", "LEADERBOARD.md", "README.md")

# A row is auditable when its recipe.json exists and its bundle directory is
# real -- that is the hard requirement, because every number is read from that
# file and the row links to that directory. These extras make a bundle
# reproducible rather than merely auditable, so their absence is reported but
# does not remove a genuine verified result.
_RECOMMENDED_BUNDLE_FILES = ("reproduce.sh", "RECIPE.md", "results.tsv")

# Instance families that mean "this was measured on real Neuron silicon". Anything
# else -- most importantly the mock backend's "mock" -- is a synthetic number and
# must never reach the public showcase.
#
# This is not hypothetical. Automatic publication (overnight._auto_publish) defaults
# to the repo it lives in, so a laptop smoke-run on `--backend mock` regenerated
# LEADERBOARD.md down to a single row reading
#     Qwen3-8B | 604 | 173,162 | 286.93x | TP=4, ..., fp8, DP=16 | mock | verified
# and rewrote optimized_models/qwen3-8b/ with it. Every existing gate passed:
# verified=="verified", speedup>1, bundle present. The missing question was whether
# the number came from hardware at all.
_REAL_INSTANCE_PREFIXES = ("trn1.", "trn2.", "trn3.", "inf1.", "inf2.")


def is_real_hardware(instance_type: str) -> bool:
    """True only for a real Neuron instance type.

    Deliberately a prefix allowlist rather than a "not mock" denylist: a new
    synthetic backend should be excluded by DEFAULT, not once someone remembers to
    add its name here.
    """
    return str(instance_type or "").strip().lower().startswith(_REAL_INSTANCE_PREFIXES)

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
    # SINGLE SOURCE OF TRUTH: a row may only exist if its deliverable bundle
    # exists on disk next to the recipe it is rendered from. Historically the
    # leaderboard carried rows whose linked optimized_models/<slug>/ directory
    # was absent -- every such link 404s for a reader, and the number behind it
    # cannot be checked against anything. A row with a dead link is worse than
    # no row: it looks verified and cannot be audited.
    if not source_dir.is_dir():
        print(f"[publish] skipping {model_id or recipe_path}: bundle directory "
              f"{source_dir} does not exist -- the row's link would 404",
              file=sys.stderr)
        return None
    # Auditability is carried by recipe.json (already parsed above): every number
    # in the row comes from it. Reproducibility extras are reported but never
    # disqualify a genuine verified run -- dropping a real result over a missing
    # PNG would be worse than an incomplete bundle.
    incomplete = [f for f in _RECOMMENDED_BUNDLE_FILES if not (source_dir / f).exists()]
    if incomplete:
        print(f"[publish] {model_id}: bundle incomplete (missing {incomplete}) -- "
              f"row still listed, numbers come from recipe.json", file=sys.stderr)
    config = recipe.get("config", {}) or {}
    toolchain = recipe.get("toolchain", {}) or {}
    hardware = str(
        recipe.get("instance_type")
        or toolchain.get("instance_type")
        or "trn2.3xlarge"
    )
    # A synthetic result is not a result. The mock backend produces plausible-looking
    # recipes -- verified, speedup > 1, complete bundle -- that would otherwise
    # publish exactly like a measured one.
    if not is_real_hardware(hardware):
        print(f"[publish] skipping {model_id}: instance_type={hardware!r} is not a "
              f"real Neuron instance -- refusing to publish a synthetic result",
              file=sys.stderr)
        return None
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
        hardware=hardware,
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
    ``model_id`` appears twice **on the same hardware**, the faster recipe wins
    (mirrors publish.py's beat-gate). The same model on *different* hardware
    yields one row per instance type.
    """
    root = Path(optimized_models_dir)
    if not root.exists():
        return []
    best_by_model: dict[tuple[str, str], Deliverable] = {}
    for recipe_path in sorted(root.rglob("recipe.json")):
        d = _deliverable_from_recipe(recipe_path, root)
        if d is None:
            continue
        # Key on (model, hardware), not model alone. A model optimized on two
        # instance types yields two independent results — speedup is relative to
        # that box's eager baseline — so both deserve a row. Keying on model
        # alone silently dropped whichever box scored lower, which hid the
        # trn2.48xlarge numbers entirely behind the trn2.3xlarge ones.
        key = (d.model_id, d.hardware)
        prev = best_by_model.get(key)
        if prev is None or d.speedup > prev.speedup or (
            d.speedup == prev.speedup and d.best > prev.best
        ):
            best_by_model[key] = d
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
        tc = recipe.get("toolchain", {}) or {}
        hw = str(recipe.get("instance_type") or tc.get("instance_type") or "trn2.3xlarge")
        if not is_real_hardware(hw):
            skipped.append((rel, f"not real hardware (instance_type={hw!r})"))
            continue
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


def group_by_model(deliverables: list[Deliverable]) -> dict[str, list[Deliverable]]:
    """slug -> its results, best speedup first.

    ``collect_verified`` keys on (model, hardware), so one model can legitimately
    appear several times -- once per box it was optimized on.
    """
    by_slug: dict[str, list[Deliverable]] = {}
    for d in deliverables:
        by_slug.setdefault(d.slug, []).append(d)
    for rows in by_slug.values():
        rows.sort(key=lambda x: (x.speedup, x.best), reverse=True)
    return by_slug


def render_model_readme(slug: str, rows: list[Deliverable]) -> str:
    """Per-model README enumerating every hardware route that was verified.

    The leaderboard answers "which model is fastest". This answers the question a
    reader actually has when they land on one model: "I have box X -- what is the
    best known way to run this, and what did it get?"

    Deliberately does NOT rank across hardware. Speedup is measured against that
    box's OWN eager baseline, so a 4x on a trn2.3xlarge and a 4x on a
    trn2.48xlarge are not the same achievement and comparing them is meaningless.
    Absolute tok/s is comparable; speedup is not. That is stated in the file
    rather than left for the reader to infer.
    """
    if not rows:
        return ""
    name = rows[0].display_name
    out = [
        f"# {name}",
        "",
        f"Verified optimization routes for [`{rows[0].model_id}`]"
        f"(https://huggingface.co/{rows[0].model_id}) on Trainium.",
        "",
        f"{len(rows)} verified route(s). Auto-generated by "
        "`publish_deliverables.py` -- do not edit by hand.",
        "",
        "| Hardware | Baseline | Best | Speedup | Best config | Recipe |",
        "|:---------|---------:|-----:|--------:|:------------|:-------|",
    ]
    for d in rows:
        rel = Path(d.rel_dir).name if Path(d.rel_dir).name != slug else "."
        link = f"[recipe](./{rel}/)" if rel != "." else "[recipe](./)"
        out.append(
            f"| `{d.hardware}` | {_fmt(d.baseline)} | **{_fmt(d.best)}** | "
            f"**{d.speedup:.3f}x** | {d.config_summary} | {link} |"
        )
    out += [
        "",
        f"Metric is {rows[0].metric_label}, measured on real hardware via the "
        "`native-pytorch-beta3` backend.",
        "",
        "## Comparing these rows",
        "",
        "**Absolute throughput is comparable across rows; speedup is not.** Each "
        "speedup is relative to the eager baseline *on that same box*, and a "
        "bigger box has a slower-relative baseline for a large model (more "
        "sharding, more collectives) as well as far more memory. So a lower "
        "speedup on a larger instance can still be the higher absolute number, "
        "and for a model that does not fit the smaller box it is the only option.",
        "",
        "## Reproducing a route",
        "",
        "Each recipe folder carries `recipe.json` (every number in the row above), "
        "`RECIPE.md`, `reproduce.sh`, `results.tsv` (the full search trajectory, "
        "including the discarded candidates and why) and "
        "`optimization_timeline.png`.",
        "",
    ]
    return "\n".join(out)


def render_leaderboard(deliverables: list[Deliverable]) -> str:
    """Regenerate the full standalone ``LEADERBOARD.md`` (sorted, medals top 3)."""
    lines = [
        "# Trainium Optimizer — Leaderboard",
        "",
        "Verified optimized models, **one row per model per hardware target**, "
        "sorted by speedup over the eager baseline on real Trainium hardware "
        "(`native-pytorch-beta3`). Auto-published by the optimizer loop — do not "
        "edit by hand.",
        "",
        "Speedup is relative to the eager baseline **on the same instance**, so "
        "rows for different hardware are each internally consistent but are **not "
        "comparable to one another** — a bigger box can score a lower multiple.",
        "",
        "Recipes and trajectory charts live under "
        "[`optimized_models/`](./optimized_models/) — each folder holds "
        "`recipe.json`, `RECIPE.md`, `reproduce.sh`, `results.tsv`, and "
        "`optimization_timeline.png`.",
        "",
        "Every row is generated from a `recipe.json` in an existing bundle: no bundle, "
        "no row, and no number that is not read straight out of that file. Rows are "
        "never hand-added — a hand-added row is removed on the next publish.",
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
        f"{len(deliverables)} verified result(s) across "
        f"{len({d.model_id for d in deliverables})} model(s) and "
        f"{len({d.hardware for d in deliverables})} hardware target(s). "
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
# consistency guard — LEADERBOARD.md rows MUST have a bundle in the same repo
# --------------------------------------------------------------------------- #
# Every leaderboard recipe link is `[recipe](./optimized_models/<rel>/)`. This is
# the invariant that was silently violated (38 rows, 9 folders → 30 dead links).
_RECIPE_LINK_RE = re.compile(r"\(\.?/?optimized_models/([^)\s]+?)/?\)")


def leaderboard_recipe_dirs(leaderboard_text: str) -> list[str]:
    """The ``optimized_models/<rel>`` dirs a LEADERBOARD.md links to (one per row).
    Pure text parse — no filesystem. Empty for empty/None."""
    if not leaderboard_text:
        return []
    return [m.group(1) for m in _RECIPE_LINK_RE.finditer(leaderboard_text)]


def check_consistency(repo_dir: Path | str) -> list[str]:
    """The anti-divergence guard: return the ``optimized_models/<rel>`` dirs that
    ``LEADERBOARD.md`` links to but whose bundle (``recipe.json``) is NOT present
    in ``repo_dir``. Empty list == consistent (every row has real, takeable code).

    This is the single check that would have caught the 30 dead links: it does not
    care WHICH writer produced the leaderboard — it only enforces that a row cannot
    exist without its folder. Runs anywhere (CI, pre-push, publish preflight); no
    deploy key, no network. Never raises."""
    repo_dir = Path(repo_dir)
    lb = _read(repo_dir / "LEADERBOARD.md")
    if lb is None:
        return []
    broken: list[str] = []
    for rel in leaderboard_recipe_dirs(lb):
        if not (repo_dir / "optimized_models" / rel / "recipe.json").exists():
            broken.append(rel)
    return broken


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
    dry_run: bool = True,
) -> dict[str, Any]:
    """Publish verified deliverables + refreshed leaderboard to ``branch``.

    Safe by construction: separate checkout, ``flock``, ``pull --rebase`` before
    push, scoped ``git add`` (only ``optimized_models/**`` + ``LEADERBOARD.md`` +
    README table region), deploy-key push, and a no-op when git sees no diff.

    **``dry_run`` defaults to True.** This function rewrites the public showcase
    of the repo, so the default has to be the harmless one -- an accidental or
    mis-configured invocation should print what it *would* change and stop.
    Writing requires opting in explicitly (``dry_run=False`` / ``--push``),
    which makes every real publish an intentional act rather than a default.
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

        # 1b. Per-model README enumerating that model's hardware routes. Written
        # for every model, including single-route ones, so the file is always there
        # and a second route later just adds a row.
        for slug, rows in group_by_model(deliverables).items():
            model_root = dest_models / slug
            if not model_root.is_dir():
                # Nested layout (<slug>/<hardware>/): the model root is the parent
                # of the recipe dir. Flat layout: it IS the recipe dir.
                model_root = dest_models / Path(rows[0].rel_dir).parts[0]
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "README.md").write_text(render_model_readme(slug, rows))

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

        # 3b. ANTI-DIVERGENCE GUARD — refuse to push a LEADERBOARD.md that links to
        # a bundle not present in the repo. Belt-and-suspenders (render_leaderboard
        # only lists collect_verified folders, so this is consistent by
        # construction), but it also catches the vector that actually bit us: a
        # folder hidden from git (e.g. .gitignore) so `git add` staged the
        # leaderboard but not the folders it references. Two checks: (a) the folder
        # exists on disk, (b) git actually TRACKS the referenced recipe.json.
        broken_disk = check_consistency(repo_dir)
        tracked = set(_git(["ls-files", "optimized_models"], repo_dir, env,
                           check=False).stdout.splitlines())
        lb_text = _read(repo_dir / "LEADERBOARD.md") or ""
        broken_git = [rel for rel in leaderboard_recipe_dirs(lb_text)
                      if f"optimized_models/{rel}/recipe.json" not in tracked]
        broken = sorted(set(broken_disk) | set(broken_git))
        if broken:
            msg = ("ABORT: LEADERBOARD.md links to bundle(s) not present/tracked in "
                   f"the repo — would create dead links: {broken}. Not committing or "
                   "pushing. (Publish the folders, or check .gitignore is not hiding "
                   "optimized_models/.)")
            _log(msg)
            result["error"] = "leaderboard_folder_divergence"
            result["broken_links"] = broken
            _git(["reset"], repo_dir, env, check=False)  # unstage; leave tree clean
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
    # Writing is opt-in. --dry-run is kept for backwards compatibility but is
    # now the default, so passing it changes nothing; --push is what publishes.
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="(default) report what would change and write nothing")
    p.add_argument("--push", dest="dry_run", action="store_false",
                   help="actually commit and push the showcase")
    p.add_argument("--check", action="store_true",
                   help="CONSISTENCY GUARD ONLY: verify every LEADERBOARD.md recipe "
                        "link resolves to a bundle in --repo-dir; exit 1 on any dead "
                        "link. No deploy key / network — for CI and pre-push hooks.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    # --check is the standalone anti-divergence guard (CI / pre-push). It never
    # writes or pushes, so it needs no deploy key.
    if args.check:
        broken = check_consistency(args.repo_dir)
        if broken:
            _log(f"CONSISTENCY FAIL: {len(broken)} LEADERBOARD.md row(s) link to a "
                 f"missing bundle (dead links): {broken}")
            return 1
        _log("consistency OK: every LEADERBOARD.md row has a bundle in the repo")
        return 0
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
