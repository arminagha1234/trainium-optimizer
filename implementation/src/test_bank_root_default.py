"""Regression test: the default --bank-root must find the real bank from any CWD.

The bug this guards against was silent and expensive. `--bank-root` defaulted to
`Path("../../knowledge-bank")`, which is CWD-relative and only correct when the driver is
launched from `implementation/src/`. But `run_overnight.py` documents being launched from
`implementation/`:

    python run_overnight.py --backend mock

from where the default resolved to a path OUTSIDE the repository. `KnowledgeBank.load_all()`
returns `[]` for a missing root rather than raising, so the run proceeded with an EMPTY
bank -- no config priors, and no anti-pattern pruning at all. Anti-patterns are the
highest-ROI lesson type (each one prunes a candidate before a 5-20 min compile), so the
failure mode was "the loop quietly stops compounding", which no existing test caught.

The two defaults also disagreed with each other: `--out-root` is `Path("../artifacts")`
(correct relative to `implementation/`) while `--bank-root` assumed `implementation/src/`.
They cannot both be right from a single working directory, which is the tell.
"""

from __future__ import annotations

import os
from pathlib import Path

import overnight

REPO_ROOT = Path(overnight.__file__).resolve().parents[2]


def _default_bank_root() -> Path:
    """The parser's default for --bank-root, as the driver would see it."""
    parser = overnight._build_parser() if hasattr(overnight, "_build_parser") else None
    if parser is not None:
        for action in parser._actions:
            if "--bank-root" in getattr(action, "option_strings", []):
                return Path(action.default)
    # Fall back to reading the module default the same way main() does.
    import argparse
    import inspect
    src = inspect.getsource(overnight.main)
    assert "--bank-root" in src, "main() no longer defines --bank-root"
    ns = {"Path": Path, "argparse": argparse, "__file__": overnight.__file__}
    # Evaluate just the default expression to avoid running main().
    marker = "default=Path(__file__).resolve().parents[2]"
    assert marker in src, (
        "--bank-root default is not anchored to __file__; a CWD-relative default "
        "silently resolves outside the repo when run as documented"
    )
    return Path(overnight.__file__).resolve().parents[2] / "knowledge-bank"


def test_default_bank_root_is_inside_the_repo_and_exists() -> None:
    root = _default_bank_root()
    assert root.exists(), f"default bank root does not exist: {root}"
    assert REPO_ROOT in root.parents or root == REPO_ROOT / "knowledge-bank", (
        f"default bank root escapes the repo: {root}"
    )


def test_default_bank_root_is_cwd_independent(tmp_path: Path) -> None:
    """The whole point: the same bank must be found from any working directory."""
    here = Path.cwd()
    seen = []
    try:
        for cwd in (REPO_ROOT, REPO_ROOT / "implementation",
                    REPO_ROOT / "implementation" / "src", tmp_path):
            os.chdir(cwd)
            seen.append(_default_bank_root().resolve())
    finally:
        os.chdir(here)
    assert len(set(seen)) == 1, f"bank root varies by CWD: {seen}"


def test_default_bank_is_not_silently_empty() -> None:
    """A bank that loads zero lessons means the flywheel is off. Catch that here
    rather than discovering it as 'the optimizer stopped getting cheaper'."""
    from bank import KnowledgeBank

    lessons = KnowledgeBank(_default_bank_root()).load_all()
    assert lessons, (
        "default bank root loaded 0 lessons -- config priors and anti-pattern "
        "pruning would be silently disabled"
    )
