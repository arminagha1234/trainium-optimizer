# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""harvest_sources.py — load the canonical harvest-source list (the "borrow
before invent" inputs).

Where the framework looks for an EXISTING kernel before spending effort
authoring a new one. The list lives in ``kernel_sources.yaml`` at the repo root
(human-readable companion: ``docs/kernel-sources.md``); this module loads it so
the harvest stage — or an agent — can consult it once per run instead of leaving
it as a doc nobody reads.

Pure + dependency-light (just PyYAML). Never raises on a missing/broken file —
returns an empty structure so a harvest step degrades gracefully rather than
crashing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_FILENAME = "kernel_sources.yaml"
# Categories in priority order for "borrow before invent".
_CATEGORIES = ("kernel_repos", "official_libraries", "tutorials", "docs")


def _find_sources_file() -> Path | None:
    """Locate kernel_sources.yaml by walking up from this file to the repo root
    (works for a clone and for an editable ``pip install -e .``). Env override:
    ``TRN_OPT_KERNEL_SOURCES``."""
    env = os.environ.get("TRN_OPT_KERNEL_SOURCES")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / _FILENAME
        if cand.is_file():
            return cand
    return None


def load_sources() -> dict[str, Any]:
    """Return the parsed harvest sources (``{kernel_repos, official_libraries,
    tutorials, docs, harvest_priority}``), or ``{}`` if the file is missing /
    unparseable. Never raises."""
    path = _find_sources_file()
    if path is None:
        return {}
    try:
        import yaml  # noqa: PLC0415 — light dep, imported lazily
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 — a broken sources file must not break harvest
        return {}


def all_sources(data: dict[str, Any] | None = None) -> list[dict]:
    """Flat list of every source entry across categories (each with an added
    ``category`` key), in borrow-before-invent priority order."""
    data = load_sources() if data is None else data
    out: list[dict] = []
    for cat in _CATEGORIES:
        for entry in data.get(cat, []) or []:
            if isinstance(entry, dict):
                out.append({**entry, "category": cat})
    return out


def summary() -> str:
    """One-screen human summary the harvest stage can log once per run so the
    source list is actually consulted, not ignored."""
    data = load_sources()
    if not data:
        return ("harvest sources: none found (kernel_sources.yaml missing) — "
                "see docs/kernel-sources.md")
    lines = ["Harvest sources (borrow before invent — see docs/kernel-sources.md):"]
    for e in all_sources(data):
        covers = ", ".join(e.get("covers", []) or [])
        lines.append(f"  [{e['category']}] {e.get('name','?')}: {e.get('url','')}"
                     + (f"\n        covers: {covers}" if covers else ""))
    pri = data.get("harvest_priority") or []
    if pri:
        lines.append("  harvest priority:")
        lines += [f"    {i+1}. {p}" for i, p in enumerate(pri)]
    return "\n".join(lines)


def main() -> int:
    print(summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
