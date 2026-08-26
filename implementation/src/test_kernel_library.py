"""Tests for kernel_library — the in-repo validated NKI kernel store. CPU-only,
no Trainium: exercises load/lookup/bank (keep-winner) + the registry bridge.

Runnable: python -m pytest -q test_kernel_library.py  OR  python test_kernel_library.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from kernel_library import KernelLibrary, LibKernel
from kernel_registry import KernelRegistry


def _write_entry(root: Path, primitive, arch, shape, *, entry, speedup, status,
                 cosine=1.0, source="import neuronxcc.nki as nki\n@nki.jit\ndef k(x):\n    return x\n"):
    d = root / primitive / arch / shape
    d.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": f"{primitive}_{shape}", "primitive": primitive, "arch": arch,
        "shape_class": shape, "entry": entry, "status": status,
        "kernel_family": "DeltaNet" if "delta" in primitive else "",
        "correctness": {"cosine": cosine, "max_abs_err": 5e-7},
        "performance": {"speedup": speedup, "baseline": "ref"},
        "sdk": {"nki": "0.6.0"},
    }
    (d / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (d / "kernel.py").write_text(source)
    return d


# ---------------------------------------------------------------------------
# load + lookup
# ---------------------------------------------------------------------------
def test_lookup_returns_best_usable():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_entry(root, "gated_delta_net", "trn2", "dk128-c128",
                     entry="gdn_fwd", speedup=2.9, status="passed-on-device")
        lib = KernelLibrary(root)
        lk = lib.lookup("gated_delta_net")
        assert lk is not None
        assert lk.entry == "gdn_fwd"
        assert lk.usable and lk.hw_ready
        assert "def k" in lk.source()          # source is retrievable


def test_lookup_prefers_on_device_then_speedup():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # a fast simulate-only vs a slower on-device: on-device (higher rank) wins
        _write_entry(root, "gated_delta_net", "trn2", "simfast",
                     entry="sim", speedup=9.0, status="passed")
        _write_entry(root, "gated_delta_net", "trn2", "devslow",
                     entry="dev", speedup=2.0, status="passed-on-device")
        lk = KernelLibrary(root).lookup("gated_delta_net")
        assert lk.entry == "dev"               # rank beats raw speedup
        # among equal rank, faster wins
        _write_entry(root, "gated_delta_net", "trn2", "devfast",
                     entry="devf", speedup=5.0, status="passed-on-device")
        assert KernelLibrary(root).lookup("gated_delta_net").entry == "devf"


def test_primitive_spelling_and_family_match():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_entry(root, "gated_delta_net", "trn2", "c128",
                     entry="gdn", speedup=3.0, status="passed-on-device")
        lib = KernelLibrary(root)
        # normalized spellings all resolve
        for spelling in ("GatedDeltaNet", "gated-delta-net", "gateddelta"):
            assert lib.lookup(spelling) is not None, spelling


def test_empty_or_missing_root_is_safe():
    lib = KernelLibrary("/nonexistent/path/kernels")
    assert lib.all() == []
    assert lib.lookup("gated_delta_net") is None


# ---------------------------------------------------------------------------
# bank-on-win (keep-winner)
# ---------------------------------------------------------------------------
def _manifest(primitive, arch, shape, entry, speedup, status="passed-on-device", cosine=1.0):
    return {"name": f"{primitive}", "primitive": primitive, "arch": arch,
            "shape_class": shape, "entry": entry, "status": status,
            "correctness": {"cosine": cosine}, "performance": {"speedup": speedup}}


def test_bank_stores_then_replaces_only_if_faster():
    with tempfile.TemporaryDirectory() as tmp:
        lib = KernelLibrary(tmp)
        src = "import neuronxcc.nki as nki\n@nki.jit\ndef k(x):\n    return x\n"
        # first bank
        assert lib.bank(_manifest("gdn", "trn2", "c128", "v1", 2.0), src) is True
        assert lib.lookup("gdn").entry == "v1"
        # slower -> NOT banked (keep-winner)
        assert lib.bank(_manifest("gdn", "trn2", "c128", "v_slow", 1.5), src) is False
        assert lib.lookup("gdn").entry == "v1"
        # faster -> banked, becomes the new best
        assert lib.bank(_manifest("gdn", "trn2", "c128", "v_fast", 3.5), src) is True
        assert lib.lookup("gdn").entry == "v_fast"
        assert abs(lib.lookup("gdn").speedup - 3.5) < 1e-9


def test_bank_rejects_incorrect_or_uncompiled():
    with tempfile.TemporaryDirectory() as tmp:
        lib = KernelLibrary(tmp)
        src = "import neuronxcc.nki as nki\n@nki.jit\ndef k(x):\n    return x\n"
        # cosine below the correctness gate -> rejected
        assert lib.bank(_manifest("gdn", "trn2", "c128", "bad", 9.0, cosine=0.5), src) is False
        # not-reusable status -> rejected
        assert lib.bank(_manifest("gdn", "trn2", "c128", "nc", 9.0, status="failed-compile"), src) is False
        # empty source -> rejected
        assert lib.bank(_manifest("gdn", "trn2", "c128", "e", 9.0), "") is False
        assert lib.lookup("gdn") is None


def test_bank_on_device_beats_banked_simulate():
    with tempfile.TemporaryDirectory() as tmp:
        lib = KernelLibrary(tmp)
        src = "import neuronxcc.nki as nki\n@nki.jit\ndef k(x):\n    return x\n"
        lib.bank(_manifest("gdn", "trn2", "c128", "sim", 9.0, status="passed"), src)
        # a slower on-device kernel still replaces a faster simulate-only one (rank)
        assert lib.bank(_manifest("gdn", "trn2", "c128", "dev", 2.0,
                                  status="passed-on-device"), src) is True
        assert lib.lookup("gdn").entry == "dev"


# ---------------------------------------------------------------------------
# registry bridge: route consults the in-repo library FIRST
# ---------------------------------------------------------------------------
def test_registry_prefers_in_repo_library():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_entry(root, "gated_delta_net", "trn2", "c128",
                     entry="gdn_fwd", speedup=3.0, status="passed-on-device")
        lib = KernelLibrary(root)
        # registry with NO external dir but WITH the library -> resolves via library
        reg = KernelRegistry(kernel_dir=None, library=lib)
        spec = reg.for_primitive("GatedDeltaNet")
        assert spec is not None
        assert spec.entry == "kernel:gdn_fwd"
        assert spec.usable and spec.hw_ready
        # without the library, no external dir -> None (unchanged behaviour)
        assert KernelRegistry(kernel_dir=None).for_primitive("GatedDeltaNet") is None


def test_registry_default_unchanged_when_no_library():
    # Byte-for-byte prior behaviour: no library, no kernel dir -> empty.
    reg = KernelRegistry()
    assert reg.for_primitive("gated_delta_net") is None
    assert reg.library is None


# ===========================================================================
def _run_standalone() -> int:
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    p = f_ = 0
    for n, f in fns:
        try:
            f(); print(f"  PASS  {n}"); p += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {n}"); traceback.print_exc(); f_ += 1
    print(f"\n{p} passed, {f_} failed (of {len(fns)})")
    return 1 if f_ else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
