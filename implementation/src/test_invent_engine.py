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
    _load_entry_from_file,
    _authored_module_name,
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


def test_offline_gate_passes_for_every_catalog_op(tmp_path):
    # Every catalog op must clear the offline gate (lint clean + impl runs at the
    # right shape) so it is eligible for the on-device race. NOTE (BUG #4 fix):
    # ``parity_ok`` is NOT asserted for every op anymore — it is only a MEANINGFUL
    # pass when the recipe supplies an INDEPENDENT numpy_impl. All catalog ops but
    # rope_apply reuse spec.reference as their numpy_impl, so their offline
    # "parity" was a function-compared-to-itself tautology; the gate now reports
    # those as parity_independent=False / parity_ok=False (verified nothing) while
    # still passing them through to the real device gate. See the dedicated
    # tautology test below.
    eng = InventEngine(out_dir=tmp_path)
    for name, spec in catalog().items():
        author = author_kernel(spec)
        gate = eng.offline_gate(author, spec)
        assert not gate.lint_violations, f"{name} lint: {gate.lint_violations}"
        assert gate.passed, f"{name} should be eligible for the device race"
        if gate.parity_independent:
            # rope_apply — the one op with a genuine re-derivation — must match.
            assert gate.parity_ok, (
                f"{name} numpy_impl disagrees with reference "
                f"(max_abs_err={gate.parity_max_abs_err:.3e})")
        else:
            # Tautological comparison: must NOT be counted as a parity pass.
            assert not gate.parity_ok, (
                f"{name} numpy_impl is spec.reference — a tautology must not be "
                f"reported as a parity pass")


def test_tautological_numpy_impl_is_not_counted_as_parity_pass(tmp_path):
    # BUG #4 regression. For the ops whose recipe reuses spec.reference as
    # numpy_impl, the offline "parity" compared a function to ITSELF: always
    # passing, validating nothing. The gate must now report those as NOT
    # independently verified (parity_independent False, parity_ok False) while
    # still letting the op through to the REAL on-device gate (passed True).
    # Only rope_apply carries a genuinely independent re-derivation.
    eng = InventEngine(out_dir=tmp_path)
    cat = catalog()

    g_soft = eng.offline_gate(author_kernel(cat["softcap"]), cat["softcap"])
    assert g_soft.parity_independent is False
    assert g_soft.parity_ok is False        # tautology is NOT a claimed pass
    assert g_soft.passed is True            # still eligible for the device gate
    assert "tautology" in g_soft.reason

    g_rope = eng.offline_gate(author_kernel(cat["rope_apply"]), cat["rope_apply"])
    assert g_rope.parity_independent is True
    assert g_rope.parity_ok is True         # genuine independent check passed
    assert g_rope.passed is True


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


# -- BUG #1 regression: static_lint must be comment/string-blind -------------
def test_lint_ignores_forbidden_tokens_in_comments_and_docstrings():
    # The exact trap that stalled the LLM author: helpful comments/docstrings
    # that NAME the forbidden constructs must NOT flag. Real code below uses only
    # nl.mgrid / *1.0/n, so this kernel is genuinely clean.
    src = (
        "import nki.language as nl\n"
        "@nki.jit\n"
        "def k(x):\n"
        '    """Index via nl.mgrid only (no nl.arange). No int(...) / no .tile(...)."""\n'
        "    # indexing via nl.mgrid only (no nl.arange)\n"
        "    # no int( cast, no .tile( in the body — use *1.0/n\n"
        "    ix = nl.mgrid[0:128, 0:64]\n"
        "    ms = nl.sum(x, axis=1) * (1.0 / 64)\n"
        "    return ms\n"
    )
    assert static_lint(src) == [], static_lint(src)


def test_lint_still_flags_forbidden_tokens_in_real_code():
    # Same tokens, but now in EXECUTABLE code — every rule must still fire.
    src = (
        "import nki.language as nl\n"
        "@nki.jit\n"
        "def k(x):\n"
        "    idx = nl.arange(0, 128)\n"          # rule 1
        "    n = int(3.0)\n"                       # rule 2 (int cast)
        "    y = x.tile((2, 2))\n"                 # rule 2 (tile)
        "    a = nl.ndarray((256, 64), dtype=x.dtype)\n"  # rule 3 (partition > 128)
        "    return a\n"
    )
    v = static_lint(src)
    assert any("arange" in s for s in v), v
    assert any("int()" in s for s in v), v
    assert any("tile()" in s for s in v), v
    assert any("partition" in s for s in v), v


def test_lint_comment_only_dma_reference_does_not_flag_but_real_one_does():
    # The DMA rule is a line/indentation scan — prove the comment-blind path
    # keeps it working: a dma_copy mentioned only in a comment is clean; a real
    # per-index dma_copy inside a loop still flags.
    clean = (
        "for k in nl.affine_range(8):\n"
        "    # avoid a per-index dma_copy(dst=buf[k, 0:128]) here — use one DMA\n"
        "    buf = nl.load(w[0:128, 0:128])\n"
    )
    assert static_lint(clean) == [], static_lint(clean)

    dirty = (
        "for k in nl.affine_range(8):\n"
        "    nisa.dma_copy(dst=buf[k, 0:128], src=w[k, 0:128])\n"
    )
    assert any("per-index DMA" in s for s in static_lint(dirty))


def test_lint_falls_back_gracefully_on_partial_source():
    # A not-yet-valid kernel (tokenize will choke) must still be lint-checkable —
    # the scrubber falls back to the raw source rather than crashing, so a real
    # nl.arange in the partial code is still caught.
    partial = "def k(:\n    idx = nl.arange(0, 128)\n"
    v = static_lint(partial)
    assert any("arange" in s for s in v), v


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


# -- BUG #3 regression: the on-device speed race must be FAIR ----------------
def test_fair_speedup_rejects_mismatched_timing():
    # The core of BUG #3: a speedup may only be computed from two measurements
    # taken by the SAME method on the SAME device. Anything else must return None
    # so the race defers instead of banking a physically meaningless ratio.
    from invent_engine import _fair_speedup

    # The exact pre-fix bug: kernel timed by nki.benchmark DEVICE latency vs a
    # baseline timed by a CPU wallclock -> not comparable.
    assert _fair_speedup(0.70, 0.91, "nki.benchmark@device", "wallclock@cpu") is None
    # Same method but the baseline is still on CPU -> not comparable.
    assert _fair_speedup(0.70, 0.91, "wallclock@device", "wallclock@cpu") is None
    # A non-positive timing is not a real measurement.
    assert _fair_speedup(0.0, 0.91, "wallclock@device", "wallclock@device") is None
    # ONLY a same-method, same-(on-)device pair yields a real speedup.
    s = _fair_speedup(0.70, 0.91, "wallclock@device", "wallclock@device")
    assert s is not None and abs(s - 0.91 / 0.70) < 1e-9


def test_unfair_race_defers_and_cannot_bank_a_win(tmp_path):
    # End-to-end honesty (BUG #3): when a fair on-device comparison cannot be
    # established the race defers (ran=False) and run_op records device_deferred
    # — never a banked NKI_KERNEL win. This is what _device_race now does when it
    # has no Neuron device handle to run the baseline on the same device.
    def _unfair_defers(_a, _s):
        return RaceResult(False, reason="fair same-device timing unavailable")

    eng = InventEngine(out_dir=tmp_path)
    res = eng.run_op(catalog()["softcap"], race_fn=_unfair_defers)
    assert res.status == "device_deferred"
    assert res.lesson_id == ""
    lessons = KnowledgeBank(tmp_path / "knowledge-bank").load_all(Tier.PROVISIONAL)
    assert not [l for l in lessons if l.type is LessonType.NKI_KERNEL], (
        "a deferred (unfair) race must never bank an NKI_KERNEL win")


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


# -- file-backed loader (the "entry function not found" fix) -----------------
# These exercise the loader MECHANISM on CPU (no nki needed): the fix is that
# the authored kernel is a REAL, importable, file-backed module-level object
# whose source linecache can read — that is what lets the compiler resolve the
# entry symbol on device. We prove the CPU-testable half here.
def _with_authored_dir(tmp_path):
    """Point the loader's on-disk dir at tmp_path; return a restore callable.

    Manual (not monkeypatch) so the standalone runner — which injects only
    tmp_path — exercises these too.
    """
    import invent_kernels
    prev = invent_kernels._AUTHORED_DIR
    invent_kernels._AUTHORED_DIR = tmp_path
    return lambda: setattr(invent_kernels, "_AUTHORED_DIR", prev)


def test_file_backed_loader_returns_real_module_function(tmp_path):
    restore = _with_authored_dir(tmp_path)
    try:
        src = "def my_entry(a, b):\n    return a + b\n"
        fn = _load_entry_from_file(src, "my_entry", "adder")
        assert callable(fn)
        assert fn(2, 3) == 5
        # It is a GENUINE file-backed module object: real __file__ on disk, and
        # linecache/inspect can read its source (the property the device
        # compiler needs to register <module>.<fn>_kernel — a synthetic
        # exec-dict lacked it).
        import inspect
        assert fn.__module__.startswith("invent_authored_adder_")
        assert Path(fn.__globals__["__file__"]).exists()
        assert "def my_entry" in inspect.getsource(fn)
    finally:
        restore()


def test_file_backed_loader_missing_entry_is_none(tmp_path):
    restore = _with_authored_dir(tmp_path)
    try:
        assert _load_entry_from_file("x = 1\n", "nope", "op") is None
        assert _load_entry_from_file("", "e", "op") is None
    finally:
        restore()


def test_file_backed_loader_build_error_is_data_not_crash(tmp_path):
    restore = _with_authored_dir(tmp_path)
    try:
        # A syntax error must degrade to None (recorded as "could not build"),
        # never propagate — an un-compilable authored kernel is the common case.
        assert _load_entry_from_file("def broken(:\n", "broken", "op") is None
    finally:
        restore()


def test_authored_module_name_is_content_addressed():
    # Different source -> different module name (so a source edit yields a fresh
    # module/file and is never masked by Python's import cache).
    a = _authored_module_name("silu_gate", "def k(): return 1\n")
    b = _authored_module_name("silu_gate", "def k(): return 2\n")
    assert a != b
    assert a == _authored_module_name("silu_gate", "def k(): return 1\n")


# -- self-test target (execution-path validation on a known-good seed) -------
def test_self_test_off_device_defers_gracefully(tmp_path):
    # Off-device (no nki), the self-test authors + offline-gates the seed and
    # reports a graceful deferral (executed=True as "deferred", not a failure).
    eng = InventEngine(out_dir=tmp_path)
    res, executed, verdict = eng.self_test("silu_gate")
    assert res.status == "device_deferred"
    assert executed is True
    assert "OFF-DEVICE" in verdict


def test_self_test_flags_entry_not_found_as_non_execution(tmp_path):
    # If a race reports the exact "entry function not found" wall, the execution
    # classifier must NOT count it as executed (that is the failure we target).
    from invent_engine import _executed_on_device
    win = RaceResult(True, correct=True, correctness_pct=100.0, speedup=1.2)
    wall = RaceResult(True, correct=False, correctness_pct=0.0, speedup=0.0,
                      reason="device race error: entry function "
                             "'invent_authored_silu_gate.silu_gate_kernel' not found")
    from invent_engine import InventResult, OfflineGate
    ok_res = InventResult("silu_gate", "s", "seed", "win",
                          OfflineGate(True, True, 0.0), win)
    bad_res = InventResult("silu_gate", "s", "seed", "anti_pattern",
                           OfflineGate(True, True, 0.0), wall)
    assert _executed_on_device(ok_res) is True
    assert _executed_on_device(bad_res) is False


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


# -- FIX 1: a missing torch_neuronx.nki_hop must NEVER abort the device path --
# On torch-neuronx 2.9 the ``nki_hop`` module was removed. An eager, unused
# ``from torch_neuronx import nki_hop`` at the top of _device_race raised
# ImportError and aborted the race before ANY device work; the wrap_nki fallback
# in _invoke_kernel imported it unguarded too. Both must survive its absence.
def test_device_race_has_no_eager_nki_hop_import():
    # The unused eager import is gone (it aborted the whole race on 2.9).
    import inspect
    src = inspect.getsource(InventEngine._device_race)
    assert "import nki_hop" not in src
    assert "from torch_neuronx import nki_hop" not in src


def test_invoke_kernel_direct_call_needs_no_nki_hop():
    # The PROVEN path is a direct positional call — it must never touch nki_hop,
    # so a directly-callable kernel returns cleanly even with torch_neuronx absent
    # (it is not installed in this env). Result is element [0] of a tuple return.
    from invent_engine import _invoke_kernel
    assert _invoke_kernel(lambda *a: (7, "meta"), [1, 2]) == 7
    assert _invoke_kernel(lambda *a: 9, [1]) == 9


def test_invoke_kernel_fallback_missing_nki_hop_is_runtimeerror_not_importerror():
    # When the direct call raises TypeError and the kernel is not a tuple builder,
    # the last-resort wrap_nki fallback lives in the removed-in-2.9
    # torch_neuronx.nki_hop. The guarded import must degrade to a clear
    # RuntimeError (recorded as race data by _device_race), NEVER an ImportError
    # that aborts the race.
    from invent_engine import _invoke_kernel

    def _wants_kwargs(*args):
        raise TypeError("kernel wants keyword args")

    try:
        _invoke_kernel(_wants_kwargs, [1])
    except ImportError:  # pragma: no cover
        raise AssertionError("nki_hop absence must not surface as ImportError")
    except RuntimeError as e:
        assert "wrap_nki" in str(e) and "nki_hop" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected the guarded fallback to raise RuntimeError")


# -- FIX 1(b): an out-param (destination-passing) kernel must surface the REAL
# arity error, not the masked generic "no invocation path" message, so the
# repair loop can learn to drop the out= param and RETURN the tensor instead.
def test_invoke_kernel_out_param_kernel_surfaces_actionable_arity_error():
    from invent_engine import _invoke_kernel

    # The LLM wrote a destination-passing kernel: 3 required args, writes into
    # `out`. The harness calls it with only the op's 2 input tensors (x, gamma),
    # so Python raises "missing 1 required positional argument: 'out'".
    def rmsnorm_kernel(x, gamma, out):        # noqa: ARG001 - signature is the point
        return out

    try:
        _invoke_kernel(rmsnorm_kernel, [1, 2])   # 2 inputs, kernel needs 3
    except RuntimeError as e:
        msg = str(e)
        # The REAL TypeError text (the actionable cause) is surfaced verbatim...
        assert "positional argument" in msg
        assert "out" in msg
        # ...and the contract is named so the model knows what to change.
        assert "return" in msg.lower()
        assert "out param" in msg.lower() or "out=" in msg
        # It is NOT masked behind the old generic non-actionable phrasing alone.
        assert "got 2 positional args" in msg
    else:  # pragma: no cover
        raise AssertionError("expected a RuntimeError surfacing the arity mismatch")


# -- FIX 2: the repair-loop compile gate runs the REAL compile ---------------
# build() only IMPORTS/traces (a @nki.jit fn is lowered by neuronx-cc lazily, on
# first invocation), so a real "failed to resolve name"/ISA error used to escape
# the repair window and die at race time instead of teaching a round-2 rewrite.
def test_repair_loop_feeds_compile_error_back_and_converges(tmp_path):
    # The seam: an injected compile_fn fails round 1 with a COMPILER error and
    # succeeds round 2; the author consumes the fed-back error and rewrites. We
    # prove the compile error reaches repair feedback AND the loop converges,
    # then goes through the SAME _finish gates as single-shot (-> win).
    from kernel_repair import CompileResult
    RESOLVE_ERR = ("device compile failed: RuntimeError('neuronx-cc: failed to "
                   "resolve name softcap_kernel')")
    seen = {"feedback": ""}

    class _LearningLLM:  # KernelAuthor-shaped: .author(spec, lessons, feedback)
        def author(self, spec, lessons, feedback):
            trail = "".join(fb.error_log for fb in (feedback or []))
            seen["feedback"] = trail
            fixed = "resolve name" in trail        # learned from the fed-back error
            src = ("import neuronxcc.nki as nki\n"
                   f"@nki.jit\ndef {spec.name}_kernel(x):\n    return x\n")
            return AuthoredKernel(op=spec.name, origin="invented",
                                  numpy_impl=spec.reference, nki_src=src,
                                  entry=f"{spec.name}_kernel",
                                  pipeline_notes="good" if fixed else "bad")

    rounds = {"n": 0}

    def compile_fn(kernel):
        rounds["n"] += 1
        if kernel.pipeline_notes == "good":
            return CompileResult(True, artifact="/tmp/k.neff")
        return CompileResult(False, error_log=RESOLVE_ERR)

    eng = InventEngine(out_dir=tmp_path, author=_LearningLLM(), max_repair_rounds=4)
    res = eng.run_op(catalog()["softcap"], race_fn=_win_race, compile_fn=compile_fn)
    assert rounds["n"] == 2                       # failed once, converged on round 2
    assert "resolve name" in seen["feedback"]     # the compile error reached the author
    assert res.status == "win"                    # converged kernel cleared _finish


def test_compile_gate_runs_real_compile_and_surfaces_error(tmp_path):
    # The DEFAULT _compile must, on device, run the real compile via
    # _device_compile_probe and turn a compiler error into the CompileResult
    # error_log the repair loop feeds back. The probe itself is device-only (needs
    # torch + a NeuronCore, exactly like _device_race), so we stub it and force
    # nki_available()->True; the assertion is that _compile WIRES the probe result.
    import invent_engine as ie
    spec = catalog()["softcap"]
    kernel = author_kernel(spec)          # clean kernel: clears the offline gate
    eng = InventEngine(out_dir=tmp_path)
    prev = ie.nki_available
    ie.nki_available = lambda: True       # pretend we are on device
    kernel.build = lambda: (lambda *a: a)  # build() succeeds (import/trace only)
    try:
        # (1) probe reports a REAL neuronx-cc error -> compile FAILS with it.
        RESOLVE = "failed to resolve name 'softcap_kernel'"
        eng._device_compile_probe = lambda fn, s: RESOLVE
        r_fail = eng._compile(kernel, spec)
        assert not r_fail.ok
        assert RESOLVE in r_fail.error_log
        assert "device compile failed" in r_fail.error_log
        # (2) probe reports None (it compiled) -> compile SUCCEEDS.
        eng._device_compile_probe = lambda fn, s: None
        r_ok = eng._compile(kernel, spec)
        assert r_ok.ok and r_ok.artifact == kernel.entry
    finally:
        ie.nki_available = prev


def test_compile_gate_off_device_stays_offline_only(tmp_path):
    # Regression: off device (no nki) the compile gate keeps today's behavior — an
    # offline-gate PASS is the honest best-effort; it must NOT require a device.
    eng = InventEngine(out_dir=tmp_path)
    spec = catalog()["softcap"]
    r = eng._compile(author_kernel(spec), spec)
    assert r.ok and r.artifact.startswith("offline-only:")


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
