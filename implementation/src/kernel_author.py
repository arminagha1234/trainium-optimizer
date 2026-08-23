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
Return ONLY the kernel source in a single ```python code block.
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
        block = [f"--- Round {fb.round} compiler error (verbatim) ---",
                 err or "(empty error log)"]
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


def build_author_prompt(spec: OpSpec, lessons: list | None,
                        feedback: list[Feedback] | None) -> str:
    """Assemble the authoring prompt from the NKI preamble, the op spec, the
    retrieved bank lessons, and the prior-round compiler errors + matched
    rewrites. Deterministic and side-effect free, so it is directly unit-testable
    (the tests assert the compiler error and the matched rewrite NAME both appear
    in the returned string)."""
    return (
        f"{_NKI_PREAMBLE}\n"
        f"## Op to author\n"
        f"  name        : {spec.name}\n"
        f"  entry naming: define `@nki.jit def {spec.name}_kernel(...)`\n"
        f"  family      : {spec.family}\n"
        f"  shape_class : {spec.shape_class}\n"
        f"  dtype       : {spec.dtype}\n"
        f"  baseline to beat (on device): {spec.baseline}\n"
        f"  notes       : {spec.notes or '(none)'}\n\n"
        f"## Relevant banked lessons (anti-patterns / prior wins to heed)\n"
        f"{_fmt_lessons(lessons)}\n\n"
        f"## Prior compile attempts (learn from each error)\n"
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
