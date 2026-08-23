"""kernel_author.py — the PLUGGABLE authoring seam for Stage 4.

Today ``invent_kernels.author_kernel`` is a recipe table: given an op name it
returns a hand-written ``AuthoredKernel``. That is the correct DEFAULT (it is
grounded, lint-clean, and deterministic), but it cannot iterate — it authors the
same kernel every time, ignoring both the retrieved bank lessons and any
compiler feedback from a prior round.

This module introduces the ``KernelAuthor`` protocol — the single seam the
engine authors through — so authoring can be swapped without touching the
engine's gate/bank/race machinery:

    author(spec, lessons, feedback) -> AuthoredKernel

  * ``spec``     — the OpSpec (what kernel to write).
  * ``lessons``  — banked anti-patterns / prior wins the engine RETRIEVED as
                   relevant to this op (``InventEngine._retrieve_lessons``).
  * ``feedback`` — the accumulated ``kernel_repair.Feedback`` from prior repair
                   rounds: the EXACT compiler error each round plus the rewrite
                   catalog's matched fix. Empty on the first (single-shot) round.

Two implementations ship:

  * ``RecipeAuthor`` — wraps ``invent_kernels.author_kernel``; ignores feedback.
    This is TODAY'S behaviour, and the engine's default, so a single-shot run is
    byte-for-byte unchanged.
  * ``LLMAuthor``    — builds a prompt from {spec, retrieved lessons, prior
    errors + matched rewrites, and a short NKI-gotcha preamble} and calls an
    INJECTED ``complete_fn(prompt) -> str``. The completion function is the
    provider-agnostic seam the README describes (Bedrock / Anthropic / OpenAI /
    local vLLM): the real network lives behind it, so this class is fully
    mockable and never talks to a model in tests.

The prompt-builder (``build_author_prompt``) is the load-bearing part: it
surfaces, per round, the exact compiler error and the matched rewrite fix — the
proven #1 lever (a captured ``TensorScalarAffineSelect`` error taught the
``.tril()`` -> constant-mask fix; round 2 that reads it compiles).
"""

from __future__ import annotations

import re
from typing import Callable, Protocol, runtime_checkable

from invent_kernels import AuthoredKernel, OpSpec, author_kernel
from kernel_repair import Feedback
from kernel_rewrites import match_error
from repair_hints import format_hints, match_hints


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------
@runtime_checkable
class KernelAuthor(Protocol):
    """The one authoring interface the engine depends on.

    An author consumes the op spec, the retrieved bank lessons, and the
    accumulated repair feedback, and returns an ``AuthoredKernel``. How it does
    so — recipe table, LLM, search, human — is opaque to the engine.
    """

    def author(self, spec: OpSpec, lessons: list | None,
               feedback: list[Feedback] | None) -> AuthoredKernel:
        ...


class RecipeAuthor:
    """Today's author: the ``invent_kernels`` recipe table.

    Ignores ``feedback`` (the recipe is fixed, it cannot iterate on a compiler
    error) and forwards ``lessons`` to ``author_kernel`` exactly as the engine
    does today. Using this author with ``max_repair_rounds=1`` reproduces the
    engine's current single-shot behaviour precisely.
    """

    def author(self, spec: OpSpec, lessons: list | None = None,
               feedback: list[Feedback] | None = None) -> AuthoredKernel:
        return author_kernel(spec, lessons=lessons)


# ---------------------------------------------------------------------------
# prompt builder — the valuable part of the LLM author
# ---------------------------------------------------------------------------
# The mandatory NKI rules the static lint enforces, stated up front so the model
# does not have to rediscover them from a lint reject. Mirrors CLAUDE.md / the
# ``static_lint`` rules in invent_kernels.
_NKI_PREAMBLE = """\
You are authoring a single NKI (Neuron Kernel Interface) kernel for a Trainium2
device. Follow these MANDATORY rules (each is enforced by a static lint before
any compile):
  * top-level `import nki`, `import nki.isa as nisa`, `import nki.language as nl`;
    decorate the entry with `@nki.jit`.
  * partition (first) dim is ALWAYS 128; tile the free axis at 128/512.
  * NEVER use `nl.arange` (deprecated) — use `nl.mgrid` for indexing/masking.
  * NEVER use an `int(...)` cast or `.tile(...)` in the kernel body (beta-3
    eager gotcha) — use `* (1.0 / n)` instead of integer ops.
  * one multi-partition DMA per operand — never a per-index single-slice DMA on
    a packed (partition) axis inside a loop.

INVOCATION CONTRACT (how the harness runs your kernel):
  The kernel is called as `out = kernel(<inputs, in the order listed below>)`
  and MUST RETURN the output tensor. Do NOT take an `out=` / destination
  parameter and do NOT write into a passed-in output buffer — a kernel that
  declares an extra `out` arg fails to invoke (wrong arity) because the harness
  passes ONLY the input tensors positionally. Your entry signature must accept
  exactly those positional inputs and `return` the result.

Known NKI-0.6.0 pitfalls (observed on real silicon — do not repeat these):
  * `nl.mgrid` is trace-only: it exists as a Python object but can leave an
    unresolved `nki.language.mgrid` name at compile (it does not always lower).
    Use it with care; prefer explicit index tiles / `nl.arange`-style index
    ranges that actually lower to hardware indexing.
  * NO Python tuple-unpacking in NKI loops: `for (a, b) in ...` fails with
    "expecting simple variable" — iterate a single simple loop variable instead.
  * Mind tile/partition bounds: the partition (first) dim must be <= 128, and
    never index the output past its declared dim size.

NKI 0.6.0 — rules that make kernels COMPILE on gen3/trn2 (from real-silicon
neuronx-cc errors during LLM authoring — each bullet is a verbatim compiler
reject turned into a DO/DON'T):
  * nc_matmul moving free-dim <= 512 (one PSUM bank): DON'T pass a `moving`
    operand whose free dim > 512 — `nc_matmul moving free dimension 4096 exceeds
    max 512 for nc_version=gen3`. DO tile any larger free dim into <=512 chunks
    and loop, accumulating in PSUM. Also stationary free (M) <= 128 and the
    contraction (partition) dim <= 128.
  * nc_matmul call signature (0.6.0) — it RETURNS the result tile; there is NO
    `dst=`/`out=` parameter. Exact signature:
      nisa.nc_matmul(stationary, moving, *, is_stationary_onezero=False,
        is_moving_onezero=False, is_transpose=False, tile_position=(),
        tile_size=(), mask=None) -> tile
    DO assign the RETURNED tile: `psum = nisa.nc_matmul(stat_tile, mov_tile)`
    (stationary and moving are the only positionals; the rest are keyword-only).
    stationary [K,M], moving [K,N] -> result [M,N]; contraction K on the
    partition axis (K, M <= 128; moving free dim N <= 512). DO NOT pass a
    `dst=`/`out=` (there is none — a 3rd positional errors "too many positional
    arguments") and DO NOT expect in-place; USE the return value.
  * nc_transpose call signature (0.6.0) — RETURNS the transposed tile; there is
    NO `dst`. Exact signature:
      nisa.nc_transpose(data, *, mask=None, dtype=None, engine=...) -> tile
    DO call it as `t = nisa.nc_transpose(data=src_tile)` (data [P,F] -> [F,P],
    P, F each <= 128) and assign the RETURN value. The high-level
    `nl.transpose(x)` also RETURNS a tile and is often simpler. For attention-
    style kernels prefer feeding an already-[K,N] moving operand into nc_matmul
    over transposing on the fly.
  * Reductions MUST stay 2-D: `nl.sum(axis=1)` that collapses to a 1-D tensor
    fails — SBUF/PSUM tiles need >= 2 dims. DO keep a [P,1]-shaped result
    (`nl.sum(x, axis=1, keepdims=True)`); DON'T let a tile collapse to 1-D. Same
    for `nl.max`.
  * Broadcast: DO use the free-function form `nl.broadcast_to(tile, shape)`; the
    tensor-method `tile.broadcast_to(...)` does not resolve in 0.6.0.
  * Scalar literals must match the tile dtype: `cap * 1.0` (tile x bare python
    float) is an object x float type error. DO multiply via
    `nl.multiply(tile, s)` where `s` is a matching-dtype scalar tile, not a bare
    python float.
  * NO Python control-flow tricks in the NKI body: no `try`/`except`, no inner
    (nested) function definitions or calls, and no tuple-unpacking — keep
    straight-line, traceable code.
Return ONLY the kernel source in a single ```python code block.
"""


# The PERFORMANCE half of the standing contract. The rules above make a kernel
# COMPILE and be CORRECT; these make it FAST — and on this engine a correct-but-
# slow kernel is banked as an anti_pattern (a loss), so speed is not a follow-up
# pass, it is a first-draft requirement. Ranked by ROI, distilled from
# ../../docs/nki-optimization-playbook.md (§11 fusion, §5 engine routing, §3/§4
# HBM traffic, §6 pipelining, §8 numerics, §10 roofline). Kept to terse bullets
# so it is cheap in-prompt and the model can act on it directly.
_PERF_PREAMBLE = """\
PERFORMANCE RULES (write for speed from the first draft — a correct-but-slow
kernel is a loss):
  1. FUSE the whole op into ONE kernel. Intermediates stay in SBUF; do ONE load
     per input and ONE store of the output — never round-trip a temporary
     through HBM (there is no HW cache; a spilled intermediate is pure loss).
  2. FUSE instructions onto the Scalar engine via `nisa.activation`. REAL
     signature: `nisa.activation(op, data, *, bias=None, scale=1.0,
     reduce_op=None, dtype=None, ...) -> tile` — op is FIRST, data (the input
     tile) is SECOND (the only two positionals; rest keyword-only); it RETURNS
     `op(scale*data + bias)` as a tile (ASSIGN it — there is no `dst`/`out`).
     With `reduce_op=` it also does a free-axis reduce in the SAME instruction.
     Assign the return value —
       * rmsnorm: `ms = nisa.activation(nl.square, x, reduce_op=nl.add)` gets
         mean-square in one pass; do NOT materialize a squared tile then sum it
         separately.
       * softmax: `e = nisa.activation(nl.exp, x, bias=neg_rowmax,
         reduce_op=nl.add)` gives `exp(x-max)` + the running denominator at once.
       * softcap: `t = nisa.activation(nl.tanh, x, scale=1/cap)` then
         one multiply by cap.
  3. HOIST loop-invariant loads (gamma / beta / cap / row-max operands) OUT of
     the tile loop — there is no HW cache, so a per-tile re-DMA of an invariant
     is wasted bandwidth. Load once, keep resident in SBUF, reuse every tile.
  4. KEEP THE PE BUSY — overlap the engines instead of serializing one. Route the
     reduce to the Scalar engine and the elementwise apply to the Vector engine
     so they run concurrently, and broadcast gamma via a TensorE matmul-against-
     ones so the otherwise-idle PE does the broadcast (a 3-engine pipeline, not
     one serial engine doing everything).
  5. bf16-in / fp32-accumulate. Read bf16, accumulate in fp32 (PSUM/Scalar are
     fp32), and cast back to bf16 only at the FINAL store (the Scalar engine's
     embedded cast pipelines it for free).
  6. WIDE, ALIGNED tiles: partition dim = 128; free dim >= 512 (bf16 >= 1024) so
     each DMA moves >= 2 KiB/partition and all 16 DMA engines stay busy — small
     tiles are packet-rate bound, not bandwidth bound.
  7. DOUBLE-BUFFER: structure the loop so tile n+1's DMA overlaps tile n's
     compute (buffer rotation), driving latency toward `max(compute, dma)` rather
     than `compute + dma`.
  8. Keep reductions 2-D with keepdims and DELAY the division: reduce to a [P,1]
     tile and apply `1/sum` to the FINAL result via `nisa.reciprocal` (one op on
     the small output), never divide every element mid-stream.
"""


def _fmt_lessons(lessons: list | None) -> str:
    if not lessons:
        return "(none retrieved)"
    out = []
    for l in lessons:
        lid = getattr(l, "lesson_id", "?")
        reason = (getattr(l, "reason", "") or "").strip()
        if len(reason) > 300:
            reason = reason[:297] + "..."
        out.append(f"  - [{lid}] {reason}")
    return "\n".join(out)


def _fmt_feedback(feedback: list[Feedback] | None) -> str:
    """One block per prior round: the EXACT compiler error + the matched rewrite
    fix. This is the #1 lever — a named, actionable fix keyed off the real error,
    not an opaque 'it failed'."""
    if not feedback:
        return ("(no prior compile attempts — this is round 1)")
    blocks: list[str] = []
    for fb in feedback:
        # Prefer the rewrites the loop already diagnosed; re-diagnose defensively
        # if a Feedback was constructed without them, so the matched fix is
        # ALWAYS surfaced when the error signature is known.
        rewrites = fb.rewrites or match_error(fb.error_log or "")
        err = (fb.error_log or "").strip()
        block = [f"--- Round {fb.round} compiler error (verbatim) ---"]
        # TARGETED SDK-API repair hints: when the error matches a KNOWN
        # signature (e.g. nc_matmul missing `moving`), PREPEND a loud, imperative
        # correction ("COMPILER SAID X — DO THIS:") ABOVE the raw error so the
        # model acts on the exact fix instead of re-emitting the same broken call
        # under repair pressure. Injected IN ADDITION to the verbatim error and
        # the catalog rewrites below — see repair_hints.py.
        hint_text = format_hints(match_hints(fb.error_log or ""))
        if hint_text:
            block.append(hint_text)
        block.append(err or "(empty error log)")
        if rewrites:
            block.append("Matched rewrite(s) from the catalog — APPLY before "
                         "re-authoring:")
            for r in rewrites:
                block.append(f"  [{r.name}] {r.summary}")
                if r.fix:
                    block.append("  fix:\n    " + r.fix.replace("\n", "\n    "))
        else:
            block.append("No catalog rewrite matched this error — reason about "
                         "the compiler message directly.")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _op_input_order(spec: OpSpec) -> list[str] | None:
    """The exact positional inputs the harness will pass to the kernel, in order.

    Mirrors ``invent_engine._arg_order`` (the single source of truth for the
    invocation order) via a late import so ``kernel_author`` stays importable
    before ``invent_engine`` (which imports THIS module) finishes loading. Falls
    back to the spec's own offline-input keys, then to ``None`` if neither is
    resolvable — the prompt then simply omits the concrete list rather than
    guessing."""
    try:
        from invent_engine import _arg_order  # noqa: PLC0415 — late to avoid a cycle
    except Exception:  # noqa: BLE001
        _arg_order = None
    inp: dict = {}
    try:
        got = spec.offline_inputs()
        if isinstance(got, dict):
            inp = got
    except Exception:  # noqa: BLE001 — never let prompt-building fail on input gen
        inp = {}
    if _arg_order is not None:
        try:
            order = _arg_order(spec.name, inp)
            if order:
                return list(order)
        except Exception:  # noqa: BLE001
            pass
    return list(inp.keys()) or None


def build_author_prompt(spec: OpSpec, lessons: list | None,
                        feedback: list[Feedback] | None) -> str:
    """Assemble the authoring prompt from the NKI preamble, the op spec, the
    retrieved bank lessons, and the prior-round compiler errors + matched
    rewrites. Deterministic and side-effect free (given the spec), so it is
    directly unit-testable (the tests assert the compiler error and the matched
    rewrite NAME both appear in the returned string, and that the FULL multi-round
    error history is present)."""
    inputs = _op_input_order(spec)
    if inputs:
        inputs_line = ", ".join(inputs)
        sig_hint = f"{spec.name}_kernel({inputs_line})"
    else:
        inputs_line = "(the op's input tensors, in declared order)"
        sig_hint = f"{spec.name}_kernel(...)"
    return (
        f"{_NKI_PREAMBLE}\n"
        f"{_PERF_PREAMBLE}\n"
        f"## Op to author\n"
        f"  name        : {spec.name}\n"
        f"  entry naming: define `@nki.jit def {sig_hint}`\n"
        f"  invocation  : the harness calls `out = {spec.name}_kernel({inputs_line})` "
        f"and expects the output tensor RETURNED — take exactly these positional "
        f"inputs, NO `out=`/destination param.\n"
        f"  inputs      : {inputs_line}\n"
        f"  family      : {spec.family}\n"
        f"  shape_class : {spec.shape_class}\n"
        f"  dtype       : {spec.dtype}\n"
        f"  baseline to beat (on device): {spec.baseline}\n"
        f"  notes       : {spec.notes or '(none)'}\n\n"
        f"## Relevant banked lessons (anti-patterns / prior wins to heed)\n"
        f"{_fmt_lessons(lessons)}\n\n"
        f"## Prior compile attempts — ALL rounds, learn from EVERY error "
        f"(do not repeat a fix that already failed a prior round)\n"
        f"{_fmt_feedback(feedback)}\n"
    )


# ---------------------------------------------------------------------------
# LLM author
# ---------------------------------------------------------------------------
CompleteFn = Callable[[str], str]

# @nki.jit def <name>(...)  — the entry the compiler will look for.
_JIT_ENTRY_RE = re.compile(r"@nki\.jit\b[^\n]*\n\s*def\s+(\w+)\s*\(")
_DEF_RE = re.compile(r"\bdef\s+(\w+)\s*\(")
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_nki_source(completion: str) -> str:
    """Pull the kernel source out of a model completion.

    Uses the first fenced ```python code block if present (the format the prompt
    asks for); otherwise treats the whole completion as source. Returns "" for an
    empty/None completion so the engine records ``no_author`` honestly rather than
    banking a fake kernel."""
    if not completion:
        return ""
    m = _FENCE_RE.search(completion)
    src = m.group(1) if m else completion
    return src.strip()


def extract_entry(nki_src: str) -> str:
    """The name of the jitted entry fn in ``nki_src`` (the symbol the compiler
    resolves). Prefer an `@nki.jit`-decorated def; fall back to the first def."""
    m = _JIT_ENTRY_RE.search(nki_src)
    if m:
        return m.group(1)
    m = _DEF_RE.search(nki_src)
    return m.group(1) if m else ""


class LLMAuthor:
    """A provider-agnostic LLM/agent author.

    ``complete_fn(prompt) -> str`` is INJECTED: it is the single place a real
    provider (Bedrock / Anthropic / OpenAI / local vLLM) is wired in, and the
    single place tests mock. This class owns the valuable, provider-independent
    logic: build the feedback-aware prompt, call the completion, extract the NKI
    source + entry, and hand back an ``AuthoredKernel`` the engine can gate.

    ``numpy_impl`` defaults to ``spec.reference``: the LLM produces device source,
    not a second numpy re-derivation, so the offline gate's parity check is a
    tautology (correctly reported as ``parity_independent=False``) and the real
    math correctness is decided by the on-device gate — exactly the honest
    contract the engine already applies to the recipe ops that reuse the
    reference.
    """

    def __init__(self, complete_fn: CompleteFn,
                 build_prompt: Callable[..., str] = build_author_prompt) -> None:
        self._complete = complete_fn
        self._build_prompt = build_prompt

    def author(self, spec: OpSpec, lessons: list | None = None,
               feedback: list[Feedback] | None = None) -> AuthoredKernel:
        prompt = self._build_prompt(spec, lessons, feedback)
        completion = self._complete(prompt)
        nki_src = extract_nki_source(completion)
        entry = extract_entry(nki_src)
        rounds = len(feedback or [])
        notes = (f"LLM-authored via injected complete_fn "
                 f"(provider-agnostic); repair round {rounds + 1}, "
                 f"{len(lessons or [])} lesson(s) consulted")
        return AuthoredKernel(
            op=spec.name,
            origin="invented",
            numpy_impl=spec.reference,   # device source; math deferred to device gate
            nki_src=nki_src,
            entry=entry,
            pipeline_notes=notes,
        )
