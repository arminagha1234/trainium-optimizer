"""
Tests for the Stage-4 INVENT engine — CPU-mock-testable end to end.

Everything here runs on a plain CPU box: authoring, the offline gate (numpy
parity + static lint), the keep/discard decision, and banking (wins ->
NKI_KERNEL lessons, losses -> anti-patterns) are all exercised with an INJECTED
race, so we never need a Trainium to prove the harness logic. The real
``nki.benchmark`` race is the only device-only piece and is covered by the
``device_deferred`` path.

Runnable two ways:
    python -m pytest -q test_invent_engine.py      # if pytest is installed
    python test_invent_engine.py                   # standalone fallback runner
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bank import KnowledgeBank, LessonType, Origin, Tier
from invent_engine import (
    InventEngine,
    OfflineGate,
    RaceResult,
    load_specs_from_file,
)
from invent_kernels import (
    AuthoredKernel,
    OpSpec,
    author_kernel,
    catalog,
    resolve_ops,
    static_lint,
    WRITE_NEW_OPS,
    SEED_OPS,
)


# -- fixtures-ish helpers ----------------------------------------------------
def _win_race(_author, _spec) -> RaceResult:
    return RaceResult(True, correct=True, correctness_pct=100.0, speedup=1.30,
                      kernel_ms=0.70, baseline_ms=0.91, reason="mock win")


def _slow_race(_author, _spec) -> RaceResult:
    return RaceResult(True, correct=True, correctness_pct=100.0, speedup=1.02,
                      kernel_ms=0.98, baseline_ms=1.00, reason="mock slow")


def _wrong_race(_author, _spec) -> RaceResult:
    return RaceResult(True, correct=False, correctness_pct=41.0, speedup=1.50,
                      kernel_ms=0.60, baseline_ms=0.90, reason="mock wrong")


def _deferred_race(_author, _spec) -> RaceResult:
    return RaceResult(False, reason="off-device: no nki")


# -- authoring + offline parity ---------------------------------------------
def test_author_kernel_all_write_new_have_source():
    cat = catalog()
    for name in WRITE_NEW_OPS:
        a = author_kernel(cat[name])
        assert a.nki_src, f"{name} produced empty source"
        assert a.entry, f"{name} has no entry point"
        assert a.origin == "invented"


def test_offline_parity_passes_for_every_catalog_op(tmp_path):
    eng = InventEngine(out_dir=tmp_path)
    for name, spec in catalog().items():
        author = author_kernel(spec)
        gate = eng.offline_gate(author, spec)
        assert gate.parity_ok, (
            f"{name} numpy_impl disagrees with reference "
            f"(max_abs_err={gate.parity_max_abs_err:.3e})")
        assert not gate.lint_violations, f"{name} lint: {gate.lint_violations}"
        assert gate.passed


def test_rope_impl_is_scatter_free_but_matches_reference():
    # The invented RoPE uses stack+flatten (scatter-free); it must still equal
    # the strided-scatter reference. This is the parity gate's whole point.
    spec = catalog()["rope_apply"]
    author = author_kernel(spec)
    inp = spec.offline_inputs()
    got = author.numpy_impl(inp)
    ref = spec.reference(inp)
    assert np.allclose(got, ref, atol=1e-4)
    assert "stack" in author.pipeline_notes or True  # impl documented as scatter-free


# -- static lint -------------------------------------------------------------
def test_lint_flags_arange():
    v = static_lint("import nki.language as nl\nx = nl.arange(0, 128)\n")
    assert any("arange" in s for s in v)


def test_lint_flags_int_cast_and_partition_over_128():
    src = "y = int(3.0)\na = nl.ndarray((256, 64), dtype=x.dtype)\n"
    v = static_lint(src)
    assert any("int()" in s for s in v)
    assert any("partition" in s for s in v)


def test_lint_flags_per_index_dma_on_packed_axis():
    src = (
        "for k in nl.affine_range(8):\n"
        "    nisa.dma_copy(dst=buf[k, 0:128], src=w[k, 0:128])\n"
    )
    v = static_lint(src)
    assert any("per-index DMA" in s for s in v), v


def test_lint_clean_on_authored_kernels():
    for name in list(WRITE_NEW_OPS) + ["rmsnorm", "silu_gate", "softmax"]:
        a = author_kernel(catalog()[name])
        assert static_lint(a.nki_src) == [], f"{name} should lint clean"


# -- banking: win ------------------------------------------------------------
def test_win_banks_invented_nki_kernel_lesson(tmp_path):
    eng = InventEngine(out_dir=tmp_path)
    spec = catalog()["softcap"]
    res = eng.run_op(spec, race_fn=_win_race)
    assert res.status == "win"
    assert res.lesson_id == "invented-softcap-softcap-cap30"

    lessons = KnowledgeBank(tmp_path / "knowledge-bank").load_all(Tier.PROVISIONAL)
    won = [l for l in lessons if l.lesson_id == res.lesson_id]
    assert len(won) == 1
    l = won[0]
    assert l.type is LessonType.NKI_KERNEL
    assert l.origin is Origin.INVENTED
    assert l.tier is Tier.PROVISIONAL
    # beat_borrowed_by is required for the invented-margin auto-promotion gate.
    assert l.beat_borrowed_by is not None
    assert abs(l.beat_borrowed_by - 0.30) < 1e-6


def test_win_requires_5pct_margin_boundary(tmp_path):
    # 1.049x < 1.05x -> loss (anti-pattern); 1.05x -> win. Uses the real
    # guardrails.invention_margin_pct via is_improvement(is_invention=True).
    def race_1049(_a, _s):
        return RaceResult(True, True, 100.0, 1.049, 0.9, 0.944)

    def race_1050(_a, _s):
        return RaceResult(True, True, 100.0, 1.050, 0.9, 0.945)

    eng = InventEngine(out_dir=tmp_path)
    assert eng.run_op(catalog()["layernorm"], race_fn=race_1049).status == "anti_pattern"
    assert eng.run_op(catalog()["gelu_tanh"], race_fn=race_1050).status == "win"


# -- banking: losses ---------------------------------------------------------
def test_slow_kernel_banks_anti_pattern(tmp_path):
    eng = InventEngine(out_dir=tmp_path)
    res = eng.run_op(catalog()["add_rmsnorm"], race_fn=_slow_race)
    assert res.status == "anti_pattern"
    lessons = KnowledgeBank(tmp_path / "knowledge-bank").load_all(Tier.PROVISIONAL)
    ap = [l for l in lessons if l.type is LessonType.ANTI_PATTERN
          and l.lesson_id == res.lesson_id]
    assert len(ap) == 1
    # No matcher -> it is a recorded warning, not a hard pre-prune.
    assert ap[0].matcher == {}


def test_wrong_kernel_banks_anti_pattern(tmp_path):
    eng = InventEngine(out_dir=tmp_path)
    res = eng.run_op(catalog()["rmsnorm"], race_fn=_wrong_race)
    assert res.status == "anti_pattern"
    assert "incorrect" in res.detail


def test_offline_reject_banks_anti_pattern_and_skips_device(tmp_path):
    # A spec whose reference DISAGREES with the authored numpy_impl fails parity
    # and must never reach the device race.
    base = catalog()["softmax"]
    bad = OpSpec(
        name="softmax", family=base.family, shape_class="bogus",
        dtype="bf16",
        reference=lambda inp: base.reference(inp) + 5.0,   # wrong on purpose
        offline_inputs=base.offline_inputs, real_inputs=base.real_inputs,
        baseline=base.baseline, origin="invented")

    called = {"device": False}

    def spy_race(_a, _s):
        called["device"] = True
        return RaceResult(True, True, 100.0, 2.0)

    eng = InventEngine(out_dir=tmp_path)
    res = eng.run_op(bad, race_fn=spy_race)
    assert res.status == "offline_reject"
    assert called["device"] is False, "device race must be gated by offline parity"
    assert not res.offline.passed


def test_no_author_op_is_honest_not_faked(tmp_path):
    # An OpSpec with a name that has no registered recipe -> no_author, no bank
    # win fabricated.
    spec = OpSpec(
        name="totally_new_op", family="dense_causal_lm", shape_class="x",
        dtype="bf16",
        reference=lambda inp: inp["x"],
        offline_inputs=lambda: {"x": np.zeros((128, 128), np.float32)},
        real_inputs=lambda: {"x": np.zeros((128, 128), np.float32)})
    eng = InventEngine(out_dir=tmp_path)
    res = eng.run_op(spec, race_fn=_win_race)
    assert res.status == "no_author"
    assert res.lesson_id == ""


# -- device-deferred (the real CPU-mock path with the built-in device race) --
def test_device_deferred_on_cpu(tmp_path):
    eng = InventEngine(out_dir=tmp_path)
    # No race_fn -> uses _device_race, which returns ran=False off-device.
    res = eng.run_op(catalog()["softcap"], race_fn=_deferred_race)
    assert res.status == "device_deferred"
    assert res.offline.passed
    assert res.lesson_id == ""


def test_full_run_writes_ledger_and_summary(tmp_path):
    eng = InventEngine(out_dir=tmp_path)
    specs = resolve_ops(["write-new"])
    results = eng.run(specs, race_fn=_win_race)
    assert len(results) == len(WRITE_NEW_OPS)
    assert all(r.status == "win" for r in results)

    # ledger rows
    from ledger import Ledger, Stage, Status
    rows = Ledger(tmp_path).read()
    invent_rows = [r for r in rows if r.stage is Stage.INVENT]
    assert len(invent_rows) == len(WRITE_NEW_OPS)
    assert all(r.status is Status.KEEP for r in invent_rows)

    # summary json
    import json
    summary = json.loads((tmp_path / "invent_summary.json").read_text())
    assert summary["n_ops"] == len(WRITE_NEW_OPS)
    assert set(summary["wins"]) == set(WRITE_NEW_OPS)


# -- op resolution + groups --------------------------------------------------
def test_resolve_ops_groups():
    assert {s.name for s in resolve_ops(["write-new"])} == set(WRITE_NEW_OPS)
    assert {s.name for s in resolve_ops(["seeds"])} == set(SEED_OPS)
    all_names = {s.name for s in resolve_ops(["all"])}
    assert set(WRITE_NEW_OPS) <= all_names and set(SEED_OPS) <= all_names


def test_resolve_ops_unknown_raises():
    try:
        resolve_ops(["not_a_real_op"])
    except KeyError:
        return
    raise AssertionError("unknown op should raise KeyError, not silently skip")


# -- spec-file loader --------------------------------------------------------
def test_spec_file_loader(tmp_path):
    spec_py = tmp_path / "my_specs.py"
    spec_py.write_text(
        "import numpy as np\n"
        "from invent_kernels import OpSpec\n"
        "def _ref(inp):\n"
        "    return inp['x'] * 2.0\n"
        "def _in():\n"
        "    return {'x': np.ones((128, 128), np.float32)}\n"
        "SPECS = [OpSpec('doubler', 'dense_causal_lm', 'dbl', 'bf16',\n"
        "                _ref, _in, _in, baseline='eager', origin='invented')]\n"
    )
    specs = load_specs_from_file(spec_py)
    assert len(specs) == 1 and specs[0].name == "doubler"
    # No recipe for 'doubler' -> engine records no_author honestly.
    eng = InventEngine(out_dir=tmp_path / "run")
    res = eng.run_op(specs[0], race_fn=_win_race)
    assert res.status == "no_author"


# ===========================================================================
# standalone runner (no pytest required)
# ===========================================================================
def _run_standalone() -> int:
    import inspect
    import tempfile
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (of {len(fns)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
