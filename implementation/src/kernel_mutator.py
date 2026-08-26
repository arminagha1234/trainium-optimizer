"""kernel_mutator.py — STRUCTURAL refinement of a winning kernel, in-code.

Why this exists (the on-device finding, 2026-08-25): once a CORRECT kernel
exists, we want the optimizer to *refine* it — keep the winning template and
change ONE thing — not throw it away and re-derive from scratch. We tried to get
the LLM author to do this by prompting ("KEEP-THE-WINNER … change exactly one
thing"). It does not work: measured on real silicon, the Bedrock Opus-5 author
rewrote the template regardless of how hard the prompt pushed (source overlap
with the winner stayed at 0.13–0.24, nowhere near a refinement). An LLM
kernel-author reconstructs from scratch every call; "refine, don't rewrite" is
not a promptable behavior.

So refinement is done HERE, structurally: given the winning kernel's *source*
and (optionally) the diagnosed roofline bottleneck, produce a small set of
variants that are each the winning template with ONE localized, mechanical edit
(a wider tile, a delayed division, an activation-reduce fusion). By construction
every variant preserves the template — its ``refinement_ratio`` vs the winner is
~1.0 — which is exactly the property the prompt could not deliver.

Contract with the perf loop (``kernel_perf.KernelPerfLoop``):
  * The mutator only PROPOSES. Each proposed variant is re-validated by the
    loop's ``measure_fn`` (compile + on-device race + the fair correctness gate)
    and adopted ONLY if it is still correct AND faster. A mutation that breaks
    correctness or does not help is cheaply rejected — it never costs a wasted
    LLM authoring round.
  * ``MutatingAuthor`` adapts to the loop's ``AuthorFn`` seam
    (``author_fn(trail: list[PerfFeedback]) -> kernel``): it hands back the next
    template-preserving variant, prioritized by the latest diagnosed bottleneck,
    as an ``AuthoredKernel`` the loop measures like any other.

Division of labor this encodes: the LLM is for the INITIAL correct kernel and
genuine exploration; the mutator is for incremental refinement of a known-good
template — the two things the on-device A/B showed each tool is actually good at.

Pure-python, stdlib-only (``re`` / ``ast``): no numpy/torch/nki import, so it is
trivially unit-testable and cheap to import inside the loop.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable

from invent_kernels import AuthoredKernel, OpSpec

# Bottleneck labels — mirror kernel_perf so a mutation can be routed to the
# roofline state it addresses. Imported defensively so a kernel_perf refactor
# never breaks importing this module (the strings are the contract, not the
# symbols).
try:
    from kernel_perf import MEMORY_BOUND, SINGLE_ENGINE, DMA_BLOCKED
except Exception:  # noqa: BLE001
    MEMORY_BOUND, SINGLE_ENGINE, DMA_BLOCKED = (
        "memory_bound", "single_engine", "dma_blocked")


@dataclass
class MutatedKernel:
    """One template-preserving variant: the winner's source with ONE edit.

    ``label``      — what single lever was changed (for logs / notes).
    ``bottleneck`` — the roofline state this mutation targets (routing hint).
    ``nki_src``    — the mutated source (guaranteed to DIFFER from the input, and
                     to still parse as Python — a mutation that cannot apply
                     cleanly is dropped, never emitted as a no-op)."""

    label: str
    bottleneck: str
    nki_src: str


# ---------------------------------------------------------------------------
# individual mutations — each: source -> mutated source (or None if inapplicable)
# ---------------------------------------------------------------------------
# A mutation returns the edited source ONLY when its pattern is present AND the
# edit changes the source; otherwise None (do not propose a no-op).

def _widen_tile(src: str) -> str | None:
    """DMA_BLOCKED lever: widen the free-dim tile chunk 512 -> 1024 so each DMA
    moves >= 2 KiB/partition and all 16 DMA engines stay busy. Rewrites the
    standalone integer literal 512 (a tile/loop chunk on these kernels) to 1024.
    Only fires when a bare ``512`` token is present."""
    if not re.search(r"(?<![\w.])512(?![\w.])", src):
        return None
    return re.sub(r"(?<![\w.])512(?![\w.])", "1024", src)


def _narrow_tile(src: str) -> str | None:
    """DMA_BLOCKED / occupancy lever: the opposite tiling try, 512 -> 256, for
    kernels where a smaller tile improves double-buffer overlap. Only fires when
    a bare ``512`` token is present."""
    if not re.search(r"(?<![\w.])512(?![\w.])", src):
        return None
    return re.sub(r"(?<![\w.])512(?![\w.])", "256", src)


# nl.sum(t * t, axis=..., keepdims=True)  -> the mean-square in ONE Scalar pass
_SQUARE_SUM_RE = re.compile(
    r"nl\.sum\(\s*([A-Za-z_]\w*)\s*\*\s*\1\s*,\s*axis=\d+\s*,\s*keepdims=True\s*\)")


def _fuse_square_reduce(src: str) -> str | None:
    """MEMORY_BOUND / SINGLE_ENGINE lever: collapse ``nl.sum(t * t, axis=1,
    keepdims=True)`` (materialize a squared tile, then reduce) into ONE
    Scalar-engine instruction ``nisa.activation(nl.square, t, reduce_op=nl.add)``
    (the activation-reduce fusion). Only fires when the square-then-sum pattern
    is present."""
    if not _SQUARE_SUM_RE.search(src):
        return None
    return _SQUARE_SUM_RE.sub(
        r"nisa.activation(nl.square, \1, reduce_op=nl.add)", src)


# X / <denominator>  where the denominator is a reduced sum / a *den*-named tile
_DELAYED_DIV_RE = re.compile(
    r"([A-Za-z_]\w*)\s*/\s*(nl\.sum\([^()]*\)|[A-Za-z_]*(?:den|sum)[A-Za-z_]*)")


def _delayed_division(src: str) -> str | None:
    """MEMORY_BOUND lever: turn an element-wise divide by a reduced denominator
    ``X / den`` into ``X * nl.reciprocal(den)`` — one reciprocal on the small
    [P,1] denominator + a multiply, instead of a full-tensor divide every
    element (delayed-softmax-division). Deliberately conservative: only matches a
    denominator that is an ``nl.sum(...)`` or a ``*den*``/``*sum*``-named tile, so
    a scalar divide like ``1.0 / F`` is never touched. Only fires on a match."""
    if not _DELAYED_DIV_RE.search(src):
        return None
    return _DELAYED_DIV_RE.sub(r"\1 * nl.reciprocal(\2)", src)


# Ordered mutation table: (label, fn, bottleneck it targets). Order is the
# default proposal order when no bottleneck is diagnosed.
_MUTATIONS: tuple[tuple[str, Callable[[str], "str | None"], str], ...] = (
    ("fuse-square-reduce", _fuse_square_reduce, MEMORY_BOUND),
    ("delayed-division", _delayed_division, MEMORY_BOUND),
    ("activation-fuse", _fuse_square_reduce, SINGLE_ENGINE),
    ("widen-tile-1024", _widen_tile, DMA_BLOCKED),
    ("narrow-tile-256", _narrow_tile, DMA_BLOCKED),
)


def _valid_python(src: str) -> bool:
    """A mutation must still parse — a regex edit that produced a syntax error is
    dropped rather than handed to the loop to fail on."""
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def mutate(nki_src: str, bottleneck: str | None = None) -> list[MutatedKernel]:
    """Return the template-preserving variants of ``nki_src``, most-relevant
    first. Each variant is the source with ONE mechanical edit; a mutation whose
    pattern is absent, whose edit does not change the source, or whose result
    does not parse is dropped (never a no-op). When ``bottleneck`` is given, the
    mutations that target it are proposed FIRST (the loop diagnosed that lever),
    then the rest — so the highest-ROI edit is measured first. Deduplicated by
    resulting source so two mutations that collapse to the same edit are proposed
    once."""
    if not nki_src or not nki_src.strip():
        return []

    def _order_key(entry: tuple) -> int:
        # 0 = targets the diagnosed bottleneck (propose first), 1 = the rest.
        return 0 if (bottleneck and entry[2] == bottleneck) else 1

    ordered = sorted(_MUTATIONS, key=_order_key)
    out: list[MutatedKernel] = []
    seen: set[str] = set()
    for label, fn, bn in ordered:
        try:
            new_src = fn(nki_src)
        except Exception:  # noqa: BLE001 — a mutation must never raise into the loop
            new_src = None
        if not new_src or new_src == nki_src or new_src in seen:
            continue
        if not _valid_python(new_src):
            continue
        seen.add(new_src)
        out.append(MutatedKernel(label=label, bottleneck=bn, nki_src=new_src))
    return out


# ---------------------------------------------------------------------------
# MutatingAuthor — adapts mutate() to the perf loop's AuthorFn seam
# ---------------------------------------------------------------------------
class MutatingAuthor:
    """A ``kernel_perf`` ``AuthorFn`` (``author_fn(trail) -> kernel``) that emits
    the next STRUCTURAL variant of a winning kernel instead of calling an LLM.

    Seeded with the winning kernel's source + entry. On each call it proposes the
    next unused template-preserving variant, prioritized by the bottleneck in the
    most recent ``PerfFeedback`` (the loop's own diagnosis). The returned
    ``AuthoredKernel`` is measured by the loop like any other; because every
    variant preserves the template, the loop's keep-gate is choosing among
    refinements of the winner — never a from-scratch rewrite.

    When the variant queue is exhausted it returns the seed unchanged, which the
    loop reads as "no new lever" and stops honestly (``no_gain``) rather than
    fabricating a candidate."""

    def __init__(self, seed_src: str, entry: str, *, op: str = "",
                 reference: Callable | None = None) -> None:
        self._seed_src = seed_src or ""
        self._entry = entry or ""
        self._op = op
        self._reference = reference
        self._served: set[str] = set()   # variant sources already proposed

    def _next_variant(self, bottleneck: str | None) -> MutatedKernel | None:
        for mk in mutate(self._seed_src, bottleneck=bottleneck):
            if mk.nki_src not in self._served:
                return mk
        return None

    def _as_kernel(self, src: str, note: str) -> AuthoredKernel:
        return AuthoredKernel(
            op=self._op, origin="invented", numpy_impl=self._reference,
            nki_src=src, entry=self._entry,
            pipeline_notes=f"structural-mutation: {note}")

    def author_fn(self, trail: list) -> AuthoredKernel:
        """The seam ``KernelPerfLoop.run(author_fn=...)`` calls each round."""
        bottleneck = None
        if trail:
            bottleneck = getattr(trail[-1], "bottleneck", None)
        mk = self._next_variant(bottleneck)
        if mk is None:
            # No unused lever left — hand back the seed unchanged so the loop's
            # keep-gate sees no improvement and stops honestly.
            return self._as_kernel(self._seed_src, "exhausted (seed unchanged)")
        self._served.add(mk.nki_src)
        return self._as_kernel(mk.nki_src, mk.label)

    # Convenience so the same object also satisfies a KernelAuthor-style call
    # (spec, lessons, feedback, perf_feedback) if wired that way — it ignores the
    # spec/lessons/feedback and drives off perf_feedback, mirroring author_fn.
    def author(self, spec: OpSpec | None = None, lessons: list | None = None,
               feedback: list | None = None,
               perf_feedback: list | None = None) -> AuthoredKernel:
        return self.author_fn(perf_feedback or [])
