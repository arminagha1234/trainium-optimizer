"""Shared helpers for the optimizer.* CLIs: backend construction, --set/kv
parsing, and recipe.json resolution.

The published ``reproduce.sh`` runs from inside a recipe bundle dir (which
contains ``recipe.json``), so when a flag is omitted we fall back to the recipe:
``optimizer.apply`` passes the config via ``--set`` explicitly, while
``optimizer.measure`` has no ``--set`` and reads model/backend/config from
``./recipe.json``. Either source works; CLI flags win over the recipe file.
"""

from __future__ import annotations

import json
import os
from typing import Any


def make_backend(name: str, instance_type: str | None = None):
    """Build a backend by name via the framework's own factory (lazy import so a
    laptop `mock` run needs no on-device deps)."""
    from overnight import _make_backend  # noqa: PLC0415 (src on path via package __init__)
    return _make_backend(name, instance_type)


def _coerce(v: str) -> Any:
    """Coerce a --set string value to bool / int / float / str."""
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_set(pairs: list[str] | None) -> dict:
    """``["tp_degree=4", "weights_dtype=bf16"]`` -> ``{"tp_degree":4, ...}``."""
    cfg: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        cfg[k.strip()] = _coerce(v)
    return cfg


def load_recipe(path: str = "recipe.json") -> dict | None:
    """The recipe.json in the current dir (the recipe bundle), or None."""
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None
    return None


def resolve(args, *, need_config: bool) -> tuple[str, str, dict, dict | None]:
    """Merge CLI flags with ./recipe.json into (model_id, backend, config, recipe).
    CLI wins. Raises SystemExit with an actionable message if model/backend can't
    be determined."""
    recipe = load_recipe()
    model = getattr(args, "model", None) or (recipe or {}).get("model_id")
    backend = getattr(args, "backend", None) or (recipe or {}).get("backend")
    config = parse_set(getattr(args, "set", None))
    if not config and recipe:
        config = recipe.get("config") or {}
    if not model or not backend:
        raise SystemExit(
            "optimizer: need --model and --backend (or run from a recipe dir "
            "containing recipe.json). Got model=%r backend=%r." % (model, backend))
    return model, backend, config, recipe
