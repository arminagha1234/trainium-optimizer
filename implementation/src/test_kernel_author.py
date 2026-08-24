"""
Tests for the pluggable authoring seam (``kernel_author``) and the REAL repair
loop wired into ``invent_engine.run_op`` — all CPU-only, no Trainium.

Coverage:
  * ``RecipeAuthor`` reproduces the current recipe-table authoring exactly.
  * ``build_author_prompt`` / ``LLMAuthor`` surface the EXACT compiler error and
    the matched rewrite NAME from prior-round feedback (the #1 lever).
  * an ``LLMAuthor`` whose mock ``complete_fn`` emits a BROKEN kernel until it
    sees a rewrite hint in the prompt, then a GOOD one, CONVERGES under
    ``run_op(max_repair_rounds=4)`` via the repair loop — where the single-shot
    (``max_repair_rounds=1``) author is rejected.

Runnable two ways:
    python -m pytest -q test_kernel_author.py
    python test_kernel_author.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bank import KnowledgeBank, LessonType, Tier
from invent_engine import InventEngine, RaceResult
from invent_kernels import OpSpec, author_kernel, catalog, WRITE_NEW_OPS
from kernel_author import (
    LLMAuthor,
    RecipeAuthor,
    build_author_prompt,
    extract_entry,
    extract_nki_source,
)
from kernel_repair import CompileResult, Feedback
from kernel_rewrites import match_error


# -- race fns ----------------------------------------------------------------
def _win_race(_author, _spec) -> RaceResult:
    return RaceResult(True, correct=True, correctness_pct=100.0, speedup=1.30,
                      kernel_ms=0.70, baseline_ms=0.91, reason="mock win")


# ---------------------------------------------------------------------------
# RecipeAuthor reproduces today's authoring
# ---------------------------------------------------------------------------
def test_recipe_author_matches_author_kernel():
    ra = RecipeAuthor()
    for name in WRITE_NEW_OPS:
        spec = catalog()[name]
        got = ra.author(spec, lessons=None, feedback=None)
        ref = author_kernel(spec)
        assert got.op == ref.op
        assert got.entry == ref.entry
        assert got.nki_src == ref.nki_src
        assert got.origin == ref.origin


def test_recipe_author_ignores_feedback():
    ra = RecipeAuthor()
    spec = catalog()["softcap"]
    fb = [Feedback(1, "some error", match_error("s2d2_ts_as_valid_elem_count"))]
    with_fb = ra.author(spec, lessons=None, feedback=fb)
    without_fb = ra.author(spec, lessons=None, feedback=None)
    # Recipe is fixed — feedback cannot change the output.
    assert with_fb.nki_src == without_fb.nki_src


# ---------------------------------------------------------------------------
# prompt builder surfaces the compiler error + matched rewrite
# ---------------------------------------------------------------------------
def test_prompt_surfaces_compiler_error_and_matched_rewrite():
    spec = catalog()["attn_decode"]
    # The captured real failure: .tril() -> TensorScalarAffineSelect ISA assert.
    err = ("neuronx-cc: ERROR: ISA validation failed: "
           "s2d2_ts_as_valid_elem_count assertion at aten__tril_select")
    rewrites = match_error(err)
    assert [r.name for r in rewrites] == ["tril-to-const-mask"], (
        "precondition: the captured error must match the tril rewrite")
    fb = [Feedback(round=1, error_log=err, rewrites=rewrites)]

    prompt = build_author_prompt(spec, lessons=None, feedback=fb)

    # The EXACT compiler error signature must appear verbatim...
    assert "s2d2_ts_as_valid_elem_count" in prompt
    # ...and the matched rewrite NAME (the actionable fix) must be named.
    assert "tril-to-const-mask" in prompt
    # round 1 context + the op entry-naming instruction are present.
    assert "Round 1" in prompt
    assert "attn_decode_kernel" in prompt


def test_prompt_re_diagnoses_when_feedback_has_no_rewrites():
    # A Feedback constructed without pre-diagnosed rewrites still gets the fix
    # surfaced (the builder re-runs match_error defensively).
    spec = catalog()["softcap"]
    err = "Compiler status FAIL: s2d2_ts_as_valid_elem_count"
    fb = [Feedback(round=1, error_log=err, rewrites=[])]
    prompt = build_author_prompt(spec, lessons=None, feedback=fb)
    assert "tril-to-const-mask" in prompt


def test_prompt_no_feedback_is_round_one():
    spec = catalog()["softcap"]
    prompt = build_author_prompt(spec, lessons=None, feedback=None)
    assert "round 1" in prompt.lower()


# ---------------------------------------------------------------------------
# FIX 1(a): the prompt states the invocation contract + the op's inputs
# ---------------------------------------------------------------------------
def test_prompt_states_return_a_tensor_no_out_param_contract():
    spec = catalog()["rmsnorm"]
    prompt = build_author_prompt(spec, lessons=None, feedback=None)
    lower = prompt.lower()
    # The harness contract: RETURN the output tensor, take no out= param.
    assert "return" in lower
    assert "out=" in prompt or "destination" in lower
    assert "no `out=`" in prompt or "do not take an" in lower or "NO `out=`" in prompt
    # The op's actual _arg_order inputs are listed so the LLM writes a matching
    # signature. rmsnorm's inputs are x, gamma.
    assert "x, gamma" in prompt
    assert "rmsnorm_kernel(x, gamma)" in prompt


def test_prompt_lists_arg_order_for_multi_input_op():
    spec = catalog()["attn_decode"]           # inputs: q, k, v
    prompt = build_author_prompt(spec, lessons=None, feedback=None)
    assert "q, k, v" in prompt
    assert "attn_decode_kernel(q, k, v)" in prompt


# ---------------------------------------------------------------------------
# FIX 2: full cross-round error history + NKI pitfalls in the preamble
# ---------------------------------------------------------------------------
def test_prompt_includes_all_prior_round_errors():
    spec = catalog()["softcap"]
    # A multi-round trail: each round has a DISTINCT error signature.
    trail = [
        Feedback(round=1, error_log="ERR_ROUND_ONE: unresolved nki.language.mgrid name",
                 rewrites=[]),
        Feedback(round=2, error_log="ERR_ROUND_TWO: expecting simple variable",
                 rewrites=[]),
        Feedback(round=3, error_log="ERR_ROUND_THREE: partition dim exceeds 128",
                 rewrites=[]),
    ]
    prompt = build_author_prompt(spec, lessons=None, feedback=trail)
    # EVERY round's error must be present (not just the latest), each labeled.
    assert "ERR_ROUND_ONE" in prompt
    assert "ERR_ROUND_TWO" in prompt
    assert "ERR_ROUND_THREE" in prompt
    assert "Round 1" in prompt and "Round 2" in prompt and "Round 3" in prompt


def test_preamble_carries_nki_pitfalls():
    # The observed regressions must be called out in the standing preamble.
    spec = catalog()["softcap"]
    prompt = build_author_prompt(spec, lessons=None, feedback=None)
    lower = prompt.lower()
    # mgrid lowering pitfall (worded as "use with care", not "does not exist").
    assert "mgrid" in lower
    assert "trace-only" in lower or "does not always lower" in lower or "unresolved" in lower
    # tuple-unpack pitfall.
    assert "expecting simple variable" in lower
    assert "tuple-unpack" in lower or "for (a, b)" in lower


def test_preamble_carries_nki06_compile_rules():
    # The concrete gen3/trn2 compile rules learned on real silicon must reach
    # the model via the authoring prompt (build_author_prompt embeds _NKI_PREAMBLE).
    spec = catalog()["softcap"]
    prompt = build_author_prompt(spec, lessons=None, feedback=None)
    lower = prompt.lower()
    # (1) nc_matmul moving free-dim <= 512 tiling rule.
    assert "512" in prompt
    assert "moving free dim" in lower or "moving free dimension" in lower
    assert "psum" in lower  # accumulate-in-PSUM tiling guidance
    # (2) reductions must stay >= 2-D (keepdims), never collapse to 1-D.
    assert "keepdims" in lower
    assert "1-d" in lower and "2-d" in lower
    # (3) the exact nc_matmul 0.6.0 signature string — returns the tile, no dst=.
    assert "nisa.nc_matmul(stationary, moving" in prompt
    assert "is_transpose" in prompt and "tile_position" in prompt
    assert "`dst=`/`out=`" in prompt and "returns the result tile" in lower
    # (4) correct broadcast + scalar-dtype forms.
    assert "nl.broadcast_to(tile, shape)" in prompt
    assert "nl.multiply" in prompt


# ---------------------------------------------------------------------------
# extraction helpers
# ---------------------------------------------------------------------------
def test_extract_nki_source_and_entry_from_fenced_block():
    completion = (
        "Here is the kernel:\n"
        "```python\n"
        "import nki\n"
        "@nki.jit\n"
        "def foo_kernel(x):\n"
        "    return x\n"
        "```\n"
        "Hope that helps!\n"
    )
    src = extract_nki_source(completion)
    assert "def foo_kernel" in src
    assert "Hope that helps" not in src          # prose stripped
    assert extract_entry(src) == "foo_kernel"    # @nki.jit entry wins


def test_extract_source_empty_completion_is_empty():
    assert extract_nki_source("") == ""
    assert extract_nki_source(None) == ""


# ---------------------------------------------------------------------------
# LLMAuthor + real repair loop convergence
# ---------------------------------------------------------------------------
# A GOOD kernel: lint-clean (no nl.arange), a resolvable @nki.jit entry.
_GOOD_SRC = (
    "import nki\n"
    "import nki.language as nl\n"
    "@nki.jit\n"
    "def softcap_kernel(x, cap):\n"
    "    return x\n"
)
# A BROKEN kernel: uses nl.arange, which (a) trips the static lint (so the
# single-shot offline gate REJECTS it) and (b) our stand-in compiler rejects
# with the captured tril ISA assertion (the teacher error the loop feeds back).
_BROKEN_SRC = (
    "import nki\n"
    "import nki.language as nl\n"
    "@nki.jit\n"
    "def softcap_kernel(x, cap):\n"
    "    idx = nl.arange(0, 128)\n"   # deprecated -> lint reject + compile reject
    "    return x\n"
)


def _mock_complete(prompt: str) -> str:
    """Mock LLM: emit BROKEN until the prompt carries the tril rewrite hint (i.e.
    the repair loop fed back the matched fix), then emit the GOOD kernel."""
    if "tril-to-const-mask" in prompt:
        return "```python\n" + _GOOD_SRC + "```"
    return "```python\n" + _BROKEN_SRC + "```"


def _mock_compile(kernel) -> CompileResult:
    """Stand-in neuronx-cc: an un-fixed (nl.arange) kernel fails with the real
    captured TensorScalarAffineSelect ISA assertion (which the catalog maps to
    tril-to-const-mask); a fixed kernel compiles."""
    if "nl.arange" in kernel.nki_src:
        return CompileResult(
            False,
            error_log=("neuronx-cc 2.27.5334: Compiler status FAIL: ISA "
                       "validation failed: s2d2_ts_as_valid_elem_count "
                       "assertion (exit 70)"))
    return CompileResult(True, artifact=kernel.entry)


def test_llm_author_repair_loop_converges(tmp_path):
    author = LLMAuthor(_mock_complete)
    eng = InventEngine(out_dir=tmp_path, author=author, max_repair_rounds=4)
    spec = catalog()["softcap"]

    res = eng.run_op(spec, race_fn=_win_race, compile_fn=_mock_compile)

    # The repair loop re-authored after reading the fed-back rewrite and the
    # kernel then passed the offline gate + the (mocked) on-device race.
    assert res.status == "win", res.detail
    assert res.race.correct is True
    # A win banks a provisional NKI_KERNEL lesson for the repaired kernel.
    lessons = KnowledgeBank(tmp_path / "knowledge-bank").load_all(Tier.PROVISIONAL)
    assert any(l.type is LessonType.NKI_KERNEL for l in lessons)


def test_single_shot_rejects_the_broken_kernel(tmp_path):
    # Same author, but max_repair_rounds=1 (today's single-shot). With no repair
    # loop the author never sees feedback, only ever emits the BROKEN kernel, and
    # the offline lint gate REJECTS it — proving the repair loop is what unlocks
    # convergence, not something already true single-shot.
    author = LLMAuthor(_mock_complete)
    eng = InventEngine(out_dir=tmp_path, author=author, max_repair_rounds=1)
    spec = catalog()["softcap"]
    res = eng.run_op(spec, race_fn=_win_race)
    assert res.status == "offline_reject"
    assert res.status != "win"


def test_repair_loop_stalls_when_error_never_matches(tmp_path):
    # A compiler error with NO catalog rewrite -> no fed-back hint -> the mock
    # never flips -> identical error every round -> the loop stalls honestly
    # (never a fabricated win). Recorded as an offline_reject, not a win.
    def _no_match_complete(_prompt: str) -> str:
        return "```python\n" + _BROKEN_SRC + "```"

    def _opaque_compile(kernel) -> CompileResult:
        return CompileResult(False, error_log="totally opaque compiler error XYZ")

    author = LLMAuthor(_no_match_complete)
    eng = InventEngine(out_dir=tmp_path, author=author, max_repair_rounds=4)
    res = eng.run_op(catalog()["softcap"], race_fn=_win_race,
                     compile_fn=_opaque_compile)
    assert res.status == "offline_reject"
    assert res.lesson_id != ""            # the loss is banked as data


# ---------------------------------------------------------------------------
# defaults unchanged: engine with defaults behaves exactly as today
# ---------------------------------------------------------------------------
def test_default_engine_uses_recipe_author_single_shot(tmp_path):
    # No author + default max_repair_rounds=1 -> recipe author, single-shot.
    eng = InventEngine(out_dir=tmp_path)
    assert isinstance(eng.author, RecipeAuthor)
    assert eng.max_repair_rounds == 1
    res = eng.run_op(catalog()["softcap"], race_fn=_win_race)
    assert res.status == "win"
    assert res.lesson_id == "invented-softcap-softcap-cap30"


# ---------------------------------------------------------------------------
# FIX A: the author prompt carries the PERFORMANCE rules (not only correctness)
# ---------------------------------------------------------------------------
def test_author_prompt_contains_performance_rules():
    # The old preamble was ALL correctness rules and zero perf, so authored
    # kernels came out serialized/slow. The prompt must now surface the ranked
    # perf levers so the model writes for speed on the first draft.
    prompt = build_author_prompt(catalog()["rmsnorm"], lessons=None, feedback=None)
    low = prompt.lower()
    assert "performance rules" in low
    assert "correct-but-slow" in low                 # the framing: slow == a loss
    assert "fuse" in low                             # (1) fuse into one kernel
    assert "nisa.activation" in low                  # (2) scalar-engine fused reduce
    assert "reduce_op" in low
    assert "hoist" in low                            # (3) hoist loop-invariant loads
    assert "keep the pe busy" in low                 # (4) engine overlap
    assert "bf16-in" in low and "fp32-accumulate" in low  # (5) accumulate policy
    assert "double-buffer" in low                    # (7) pipeline DMA behind compute
    assert "reciprocal" in low                       # (8) delayed division


# ---------------------------------------------------------------------------
# Pillar 1: op-aware knowledge retrieval is wired into the prompt
# ---------------------------------------------------------------------------
def test_prompt_includes_op_aware_knowledge_by_default():
    # rmsnorm (normalization) must pull the reduction/normalization knowledge:
    # keepdims + activation-reduce-fusion + a fenced worked example naming the op.
    prompt = build_author_prompt(catalog()["rmsnorm"], lessons=None, feedback=None)
    assert "Relevant verified techniques & worked examples" in prompt
    assert "op family: normalization" in prompt
    assert "activation-reduce-fusion" in prompt
    assert "rmsnorm_kernel" in prompt          # the worked example is embedded


def test_prompt_knowledge_is_op_specific():
    # An attention op gets the flash/online-softmax example; a normalization op
    # does NOT — the retrieval is op-aware, not one-size-fits-all.
    attn = build_author_prompt(catalog()["attn_decode"], lessons=None, feedback=None)
    assert "op family: attention" in attn
    assert "flash" in attn.lower() or "online" in attn.lower()
    norm = build_author_prompt(catalog()["rmsnorm"], lessons=None, feedback=None)
    assert "op family: attention" not in norm
    # The scan-only technique never leaks into a normalization prompt.
    assert "chunked-scan" not in norm


def test_include_knowledge_false_reproduces_baseline_prompt():
    # The A/B measurement flips exactly this switch: include_knowledge=False must
    # remove the retrieved section and leave the rest byte-identical.
    from kernel_author import _NKI_PREAMBLE
    spec = catalog()["softmax"]
    enriched = build_author_prompt(spec, lessons=None, feedback=None)
    baseline = build_author_prompt(spec, lessons=None, feedback=None,
                                   include_knowledge=False)
    assert "Relevant verified techniques & worked examples" in enriched
    assert "Relevant verified techniques & worked examples" not in baseline
    # The baseline is the enriched prompt with ONLY the knowledge block removed.
    assert len(baseline) < len(enriched)
    # The standing preamble + op section are present in BOTH.
    assert "## Op to author" in baseline and "## Op to author" in enriched
    assert _NKI_PREAMBLE.strip() in baseline


def test_perf_preamble_is_ordered_after_correctness_preamble():
    # Correctness rules first (they gate compile), perf rules second — both must
    # be present and the perf block must not have displaced the NKI contract.
    from kernel_author import _NKI_PREAMBLE, _PERF_PREAMBLE
    prompt = build_author_prompt(catalog()["softcap"], lessons=None, feedback=None)
    assert _NKI_PREAMBLE.strip() in prompt
    assert _PERF_PREAMBLE.strip() in prompt
    assert prompt.index(_NKI_PREAMBLE[:40]) < prompt.index(_PERF_PREAMBLE[:40])


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
