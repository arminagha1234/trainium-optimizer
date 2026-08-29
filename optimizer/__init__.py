"""optimizer — the installable CLI package the published recipes call.

Every recipe's ``reproduce.sh`` runs ``python -m optimizer.apply`` /
``python -m optimizer.measure`` / ``python -m optimizer.run``. Historically that
package did NOT exist (the code lives flat in ``implementation/src/``), so the
flagship "reproducible recipe" deliverable was non-runnable. This package is the
thin, honest bridge: it maps those three commands onto the framework's real
backend / orchestrator logic.

Import side effect: locate ``implementation/src`` relative to this package (works
for an editable ``pip install -e .`` and for a plain clone) and put it on
``sys.path`` so the flat ``from orchestrator import ...`` modules resolve without
a manual ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "implementation", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

__all__ = ["apply", "measure", "run"]
