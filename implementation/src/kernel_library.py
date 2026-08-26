"""kernel_library.py — the IN-REPO validated NKI kernel library.

The framework already (a) HARVESTS external kernels and (b) AUTHORS new ones —
but it did not durably KEEP the good kernels it produced. A kernel we validated
on-device (correct + fast) was lost to the next run, which breaks the compounding
premise for kernels specifically. This module closes that loop: a versioned,
in-repo store of kernels WE wrote, keyed by (primitive, arch, shape_class),
keeping the BEST validated kernel per key.

It is the public, source-included complement to ``kernel_registry`` (which reads
manifests of PROPRIETARY external kernels, never their source). Consult order for
"is there a kernel for this primitive?": in-repo library (ours, source here) →
external registry ($TRN_OPT_KERNEL_DIR, proprietary) → author from scratch.

Layout (under ``knowledge-bank/kernels/``):
    <primitive>/<arch>/<shape_class>/
        kernel.py         # the @nki.jit source (importable, Apache-2.0)
        manifest.yaml     # metadata + correctness + performance (schema below)
        reference.py      # (optional) numpy oracle it was validated against
        results.tsv       # (optional) raw on-device measurements

Three integration points (all extend existing machinery):
  1. ROUTE/HARVEST reads it FIRST (before external + before authoring) — via
     ``KernelLibrary.lookup`` / ``to_kernel_spec`` (bridges to registry.KernelSpec).
  2. BANK-ON-WIN: ``KernelLibrary.bank`` stores an authored kernel as the new best
     for its key ONLY if it beats the current best speedup AND clears the
     correctness gate (keep-winner, mirroring the lesson bank).
  3. AUTHOR RETRIEVAL: banked sources can be surfaced to the LLM author as
     op-family worked-examples (see nki_knowledge integration).

Pure-stdlib + PyYAML (already a repo dep). No numpy/torch/nki import, so it is
cheap to import and trivially unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # noqa: BLE001 — degrade to no-library rather than break import
    yaml = None  # type: ignore

# Reuse the registry's primitive normalization + name map so the library and the
# external registry agree on what "gated_delta_net" etc. resolve to.
try:
    from kernel_registry import _norm, kernel_for_primitive, STATUS_RANK, KernelSpec
except Exception:  # noqa: BLE001 — standalone fallback (tests without registry on path)
    def _norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def kernel_for_primitive(primitive: str) -> str | None:  # type: ignore
        return None

    STATUS_RANK = {"passed": 3, "passed-on-device": 4, "hardware-validated": 4}
    KernelSpec = None  # type: ignore

_MIN_USABLE_RANK = 3          # >= "passed" (simulate-correct) is reusable
DEFAULT_ROOT = "knowledge-bank/kernels"


@dataclass
class LibKernel:
    """One banked kernel: its manifest + where the source lives on disk."""

    primitive: str
    arch: str
    shape_class: str
    name: str
    entry: str                          # @nki.jit fn name in kernel.py
    status: str = "passed-on-device"
    dtype: str = "fp32"
    speedup: float = 0.0                # best measured speedup vs its baseline
    baseline: str = ""
    cosine: float = 0.0
    max_abs_err: float = 0.0
    provenance: str = ""
    license: str = ""
    sdk: dict = field(default_factory=dict)
    validated_on: str = ""
    date: str = ""
    kernel_family: str = ""             # canonical registry name (PRIMITIVE_TO_KERNEL)
    dir: Path | None = None             # the entry directory
    raw: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return STATUS_RANK.get(self.status, 0)

    @property
    def usable(self) -> bool:
        return self.rank >= _MIN_USABLE_RANK

    @property
    def hw_ready(self) -> bool:
        return self.rank >= STATUS_RANK.get("passed-on-device", 4)

    @property
    def source_path(self) -> Path | None:
        return (self.dir / "kernel.py") if self.dir else None

    def source(self) -> str:
        p = self.source_path
        return p.read_text() if (p and p.is_file()) else ""


def _key(primitive: str, arch: str, shape_class: str) -> tuple[str, str, str]:
    return (_norm(primitive), _norm(arch), _norm(shape_class))


class KernelLibrary:
    """Load / look up / bank validated in-repo kernels.

    Never raises on a bad/absent entry — a malformed manifest is skipped, a
    missing root yields an empty library, so the framework runs unchanged when no
    kernels are banked yet.
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    # -- load ---------------------------------------------------------------
    def _iter_manifests(self):
        if yaml is None or not self.root.is_dir():
            return
        for mpath in self.root.glob("*/*/*/manifest.yaml"):
            try:
                data = yaml.safe_load(mpath.read_text()) or {}
                yield mpath.parent, data
            except Exception:  # noqa: BLE001 — a bad manifest is skipped
                continue

    def _to_lib(self, d: Path, data: dict) -> LibKernel | None:
        try:
            perf = data.get("performance", {}) or {}
            corr = data.get("correctness", {}) or {}
            return LibKernel(
                primitive=str(data.get("primitive", "")),
                arch=str(data.get("arch", "")),
                shape_class=str(data.get("shape_class", "")),
                name=str(data.get("name", d.name)),
                entry=str(data.get("entry", "")),
                status=str(data.get("status", "passed-on-device")),
                dtype=str(data.get("dtype", "fp32")),
                speedup=float(perf.get("speedup", 0.0) or 0.0),
                baseline=str(perf.get("baseline", "")),
                cosine=float(corr.get("cosine", 0.0) or 0.0),
                max_abs_err=float(corr.get("max_abs_err", 0.0) or 0.0),
                provenance=str(data.get("provenance", "")),
                license=str(data.get("license", "")),
                sdk=dict(data.get("sdk", {}) or {}),
                validated_on=str(data.get("validated_on", "")),
                date=str(data.get("date", "")),
                kernel_family=str(data.get("kernel_family", "")
                                  or kernel_for_primitive(data.get("primitive", "")) or ""),
                dir=d, raw=data,
            )
        except Exception:  # noqa: BLE001
            return None

    def all(self) -> list[LibKernel]:
        out = []
        for d, data in self._iter_manifests():
            lk = self._to_lib(d, data)
            if lk is not None:
                out.append(lk)
        return out

    def all_for_primitive(self, primitive: str) -> list[LibKernel]:
        """Every banked kernel whose primitive OR canonical family matches."""
        p = _norm(primitive)
        fam = _norm(kernel_for_primitive(primitive) or "")
        hits = []
        for lk in self.all():
            if _norm(lk.primitive) == p or (fam and _norm(lk.kernel_family) == fam):
                hits.append(lk)
        return hits

    def lookup(self, primitive: str, arch: str | None = None,
               shape_class: str | None = None) -> LibKernel | None:
        """The BEST usable banked kernel for a primitive.

        Filters to the requested arch / shape_class when given (a None matches
        any), requires ``usable`` (>= simulate-correct), and returns the highest
        (rank, speedup) — an on-device-validated fast kernel beats a
        simulate-only one, and among equals the faster wins.
        """
        cands = [lk for lk in self.all_for_primitive(primitive) if lk.usable]
        if arch:
            cands = [lk for lk in cands if _norm(lk.arch) == _norm(arch)]
        if shape_class:
            exact = [lk for lk in cands if _norm(lk.shape_class) == _norm(shape_class)]
            cands = exact or cands            # fall back to any shape if no exact match
        if not cands:
            return None
        return max(cands, key=lambda lk: (lk.rank, lk.speedup))

    def to_kernel_spec(self, lk: LibKernel):
        """Bridge a LibKernel to a registry ``KernelSpec`` so the existing
        route/preflight code can consume an in-repo kernel identically to an
        external one. ``path`` points at the in-repo source dir."""
        if KernelSpec is None or lk is None:
            return None
        return KernelSpec(
            name=lk.kernel_family or lk.name,
            status=lk.status,
            entry=f"kernel:{lk.entry}",
            path=str(lk.source_path or (lk.dir or "")),
            variants=[lk.primitive],
            tolerances=({"max_abs_err": lk.max_abs_err} if lk.max_abs_err else {}),
            backend=str((lk.sdk or {}).get("backend", "")),
            notes=f"in-repo kernel-lib: {lk.provenance}"[:400],
        )

    # -- bank-on-win --------------------------------------------------------
    def bank(self, manifest: dict, source: str, *, reference: str = "",
             results_tsv: str = "", min_gain: float = 0.0) -> bool:
        """Store a validated kernel as the best for its (primitive, arch,
        shape_class) key — KEEP-WINNER: write ONLY if it clears the correctness
        gate AND (no incumbent OR strictly beats the incumbent speedup by
        ``min_gain``). Returns True if it was banked.

        Correctness gate: status must be reusable (>= 'passed') and cosine, if
        given, must be >= 0.99 — never bank a kernel that is not at least
        simulate-correct (mirrors the anti-reward-hack posture of the bank).
        """
        if yaml is None:
            return False
        prim = str(manifest.get("primitive", ""))
        arch = str(manifest.get("arch", ""))
        shp = str(manifest.get("shape_class", ""))
        if not (prim and arch and shp and source.strip()):
            return False
        status = str(manifest.get("status", "passed-on-device"))
        if STATUS_RANK.get(status, 0) < _MIN_USABLE_RANK:
            return False
        corr = manifest.get("correctness", {}) or {}
        cos = float(corr.get("cosine", 1.0) or 0.0)
        if cos and cos < 0.99:
            return False
        new_speedup = float((manifest.get("performance", {}) or {}).get("speedup", 0.0) or 0.0)

        incumbent = self.lookup(prim, arch, shp)
        if incumbent is not None and _key(incumbent.primitive, incumbent.arch,
                                          incumbent.shape_class) == _key(prim, arch, shp):
            # only replace if strictly better (rank first, then speedup+min_gain)
            new_rank = STATUS_RANK.get(status, 0)
            if new_rank < incumbent.rank:
                return False
            if new_rank == incumbent.rank and new_speedup <= incumbent.speedup + min_gain:
                return False

        d = self.root / _slug(prim) / _slug(arch) / _slug(shp)
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel.py").write_text(source)
        (d / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        if reference:
            (d / "reference.py").write_text(reference)
        if results_tsv:
            (d / "results.tsv").write_text(results_tsv)
        return True


def _slug(s: str) -> str:
    """Filesystem-safe directory name preserving readability (kept, not fully
    normalized, so `dk128-dv128-chunk128` stays legible)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "unknown"
