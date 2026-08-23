"""Tests for the targeted SDK-API repair-hint map (repair_hints) and its wiring
into the repair prompt (kernel_author._fmt_feedback / Feedback.as_prompt) and the
KernelRepairLoop stall guard.

Theme: a KNOWN compiler-error signature (e.g. nc_matmul missing `moving`) becomes
a loud, imperative correction PREPENDED to the next-round prompt — teaching the
real SDK API — while an UNKNOWN error falls back to raw-error-only (unchanged).
"""

from __future__ import annotations

from kernel_repair import CompileResult, Feedback, KernelRepairLoop
from repair_hints import HINTS, format_hints, match_hints


# Real (or faithfully wrapped) error strings this SDK emits, keyed to each hint.
NC_MATMUL_ERR = ('device compile failed: TypeError("nc_matmul() missing value '
                 'for required argument \'moving\'")')
NC_TRANSPOSE_ERR = ("device compile failed: TypeError(\"nc_transpose() missing "
                    "value for required argument 'data'\")")
ACTIVATION_DTYPE_ERR = ("device compile failed: RuntimeError(\"error: failed to "
                        "compile NKI kernel:\\n- [x1] error: activation() got an "
                        "unexpected keyword argument 'dtype'\")")
ACTIVATION_DATA_ERR = ("device compile failed: RuntimeError(\"error: activation() "
                       "missing value for required argument 'data'\")")
ISMP902_ERR = ("device compile failed: RuntimeError('[NCC_ISMP902] Simplifier "
               "internal error in is_subset during dtype convert fusion')")
REDUCTION_1D_ERR = ("device compile failed: ValueError('tile must have at least "
                    "2 dimensions; got a 1-D result from nl.sum(axis=1)')")
BROADCAST_ERR = ("device build/trace failed: AttributeError(\"'Tensor' object "
                 "has no attribute 'broadcast_to'\")")
PARTITION_BCAST_ERR = ("device build/trace failed: AssertionError('Unexpected "
                       "partition broadcast!')")
UNKNOWN_ERR = "error: something entirely unfamiliar happened in pass QuxBar"
INFER_TILE_ERR = ('device compile failed: RuntimeError("authored kernel not '
                  'directly callable: Failed to infer tile from tensor k_res, '
                  'used by parameter a of nki api a = b: the first dimension of '
                  'the tile is not the partition dimension of the tensor.")')
OUTPUT_DEPS_ERR = ("device compile failed: SyntaxError('Unexpected output "
                   "dependencies, missing indices in the dst access: j.')")
TOO_MANY_POS_ERR = ('device compile failed: RuntimeError("authored kernel not '
                    'directly callable: too many positional arguments. The harness '
                    'invokes the kernel as out = kernel(*inputs) ... do NOT take an '
                    'out param.")')
ISFV902_ERR = ('device compile failed: RuntimeError("RunNeuronCCImpl: error '
               'condition error != 0: [INTERNAL_ERROR] [NCC_ISFV902] SFKVectorizer '
               'error: gist(): incompatible function arguments.")')

_SEEDED = {
    "nc_matmul-missing-moving": (NC_MATMUL_ERR,
                                 "nisa.nc_matmul(stat, mov)"),
    "nc_transpose-missing-data": (NC_TRANSPOSE_ERR,
                                  "nisa.nc_transpose(data=src)"),
    "activation-signature": (ACTIVATION_DTYPE_ERR,
                             "nisa.activation(nl.square, x"),
    "simplifier-ismp902-host-cast": (ISMP902_ERR, "cast on the HOST"),
    "reduction-collapse-1d": (REDUCTION_1D_ERR, "keepdims=True"),
    "broadcast-to-freefn": (BROADCAST_ERR, "nl.broadcast_to(tile, shape=(P, F))"),
    "unexpected-partition-broadcast": (PARTITION_BCAST_ERR, "Broadcast the [1,F]"),
    "infer-tile-partition-dim": (INFER_TILE_ERR, "keep the partition dim first"),
    "unexpected-output-dependencies": (OUTPUT_DEPS_ERR,
                                       "nl.store(out[:, a:b], value=tile)"),
    "too-many-positional-return-form": (TOO_MANY_POS_ERR, "RETURN-FORM contract"),
    "sfkvectorizer-gist-isfv902": (ISFV902_ERR, "partition dim >=2"),
}


def test_activation_data_error_does_not_misroute_to_nc_transpose():
    # nisa.activation and nisa.nc_transpose BOTH have a `data` arg and emit the
    # identical "missing required argument 'data'" phrase. The activation error
    # must route to the activation hint ONLY — never to nc_transpose.
    names = [h.key for h in match_hints(ACTIVATION_DATA_ERR)]
    assert "activation-signature" in names
    assert "nc_transpose-missing-data" not in names


def test_partition_broadcast_routes_and_teaches_explicit_broadcast():
    names = [h.key for h in match_hints(PARTITION_BCAST_ERR)]
    assert names == ["unexpected-partition-broadcast"]
    text = format_hints(match_hints(PARTITION_BCAST_ERR))
    assert "nl.broadcast_to" in text and "shape=(P, F)" in text


# -- the map matches each seeded signature and injects the right text --------

def test_every_seeded_signature_matches_and_injects_its_fix():
    for key, (err, needle) in _SEEDED.items():
        hits = match_hints(err)
        names = [h.key for h in hits]
        assert key in names, (err, names)
        text = format_hints(hits)
        assert "COMPILER SAID" in text
        assert needle in text, (key, needle, text)


def test_nc_matmul_hint_is_imperative_and_specific():
    hits = match_hints(NC_MATMUL_ERR)
    assert [h.key for h in hits] == ["nc_matmul-missing-moving"]
    text = format_hints(hits)
    # Teaches the REAL signature: return-form (stationary, moving) -> tile, no dst.
    assert "moving" in text and "stationary" in text
    assert "RETURNS the result tile" in text


# -- unmatched error -> raw-error-only fallback (unchanged behaviour) --------

def test_unknown_error_matches_nothing():
    assert match_hints(UNKNOWN_ERR) == []
    assert match_hints("") == []
    assert format_hints([]) == ""


def test_every_hint_has_patterns_and_fix():
    # Guardrail: a hint with no pattern can never fire; with no fix teaches nothing.
    for h in HINTS:
        assert h.patterns, h.key
        assert h.fix.strip(), h.key
        assert h.title.strip(), h.key


# -- the hint appears in the actually-built repair prompt --------------------

def test_hint_text_appears_in_built_author_prompt():
    from invent_kernels import catalog
    from kernel_author import build_author_prompt

    spec = catalog()["rmsnorm"]
    fb = [Feedback(1, NC_MATMUL_ERR, [])]
    prompt = build_author_prompt(spec, None, fb)
    assert "COMPILER SAID" in prompt
    assert "nisa.nc_matmul(stat, mov)" in prompt
    # Raw error is STILL present (hint is in ADDITION, not a replacement).
    assert "missing value for required argument 'moving'" in prompt


def test_feedback_as_prompt_carries_the_hint():
    fb = Feedback(2, NC_TRANSPOSE_ERR, [])
    prompt = fb.as_prompt()
    assert "COMPILER SAID" in prompt
    assert "nisa.nc_transpose(data=src)" in prompt


def test_unmatched_feedback_prompt_has_no_hint_banner():
    from invent_kernels import catalog
    from kernel_author import build_author_prompt

    spec = catalog()["rmsnorm"]
    prompt = build_author_prompt(spec, None, [Feedback(1, UNKNOWN_ERR, [])])
    assert "COMPILER SAID" not in prompt
    assert "QuxBar" in prompt          # raw error still fed back


# -- stall guard: a NEW hint grants extra rounds; no hint bails as before -----

def test_matched_hint_error_gets_extra_rounds_before_stalling():
    # Author keeps emitting the SAME nc_matmul error (ignores the fix). With a
    # matched hint, the loop grants hint_bonus extra rounds before bailing —
    # it does NOT bail at stall_patience like an unmatched repeat would.
    loop = KernelRepairLoop(max_rounds=8, stall_patience=2, hint_bonus=2)
    out = loop.run(lambda trail: "broken",
                   lambda k: CompileResult(False, error_log=NC_MATMUL_ERR))
    assert not out.ok
    assert out.reason.startswith("stalled")
    # stall_patience(2) + hint_bonus(2) = bail at round 4, not round 2.
    assert out.rounds == 4, out.rounds


def test_unmatched_repeat_still_bails_at_stall_patience():
    # No hint matches -> no bonus -> unchanged: bail at stall_patience.
    loop = KernelRepairLoop(max_rounds=8, stall_patience=2, hint_bonus=2)
    out = loop.run(lambda trail: "broken",
                   lambda k: CompileResult(False, error_log=UNKNOWN_ERR))
    assert not out.ok
    assert out.reason.startswith("stalled")
    assert out.rounds == 2, out.rounds


def test_hint_bonus_bounded_by_max_rounds():
    # A matched hint never causes an unbounded loop: capped at max_rounds.
    loop = KernelRepairLoop(max_rounds=3, stall_patience=2, hint_bonus=5)
    out = loop.run(lambda trail: "broken",
                   lambda k: CompileResult(False, error_log=NC_MATMUL_ERR))
    assert not out.ok
    assert out.rounds == 3
    assert out.reason == "exhausted rounds"


# -- NEW attn_decode-run signatures: unique routing + no cross-fire ----------

def test_infer_tile_routes_uniquely_and_teaches_partition_first():
    names = [h.key for h in match_hints(INFER_TILE_ERR)]
    assert names == ["infer-tile-partition-dim"], names
    text = format_hints(match_hints(INFER_TILE_ERR))
    assert "PARTITION axis" in text and "FREE axis" in text


def test_output_deps_routes_uniquely_and_teaches_full_slice_store():
    names = [h.key for h in match_hints(OUTPUT_DEPS_ERR)]
    assert names == ["unexpected-output-dependencies"], names
    text = format_hints(match_hints(OUTPUT_DEPS_ERR))
    assert "nl.store(out[:, a:b], value=tile)" in text


def test_too_many_positional_routes_to_return_form_and_teaches_no_dst():
    names = [h.key for h in match_hints(TOO_MANY_POS_ERR)]
    assert names == ["too-many-positional-return-form"], names
    text = format_hints(match_hints(TOO_MANY_POS_ERR))
    assert "NO out=/dst= parameter" in text
    assert "nl.ndarray(shape, dtype, buffer=nl.shared_hbm)" in text
    assert "nisa.nc_matmul(stationary, moving)" in text


def test_isfv902_routes_uniquely_and_does_not_misroute_to_ismp902():
    # NCC_ISFV902 (vectorizer gist crash) and NCC_ISMP902 (Simplifier is_subset)
    # are DIFFERENT compiler crashes with DIFFERENT fixes — must not cross-fire.
    names = [h.key for h in match_hints(ISFV902_ERR)]
    assert names == ["sfkvectorizer-gist-isfv902"], names
    assert "simplifier-ismp902-host-cast" not in names
    # and the ISMP902 host-cast error must NOT hit the new isfv902 hint.
    assert "sfkvectorizer-gist-isfv902" not in [h.key for h in match_hints(ISMP902_ERR)]


def test_new_hint_appears_in_built_author_prompt():
    from invent_kernels import catalog
    from kernel_author import build_author_prompt

    spec = catalog()["attn_decode"]
    prompt = build_author_prompt(spec, None, [Feedback(1, TOO_MANY_POS_ERR, [])])
    assert "COMPILER SAID" in prompt
    assert "RETURN-FORM contract" in prompt
    # Raw error still present (hint is in ADDITION, not a replacement).
    assert "too many positional arguments" in prompt


def test_new_hint_error_gets_extra_rounds_before_stalling():
    # The exact stall the first live attn_decode run hit: the author repeats the
    # SAME "too many positional arguments" error. With the NEW hint matched, the
    # loop grants hint_bonus extra rounds before bailing (was: bail at round 2).
    loop = KernelRepairLoop(max_rounds=8, stall_patience=2, hint_bonus=2)
    out = loop.run(lambda trail: "broken",
                   lambda k: CompileResult(False, error_log=TOO_MANY_POS_ERR))
    assert not out.ok
    assert out.reason.startswith("stalled")
    assert out.rounds == 4, out.rounds
