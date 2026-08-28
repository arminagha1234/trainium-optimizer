# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""kernel_compose.py — DECOMPOSE-AND-COMPOSE authoring for the hard ops.

The on-device finding this closes: the LLM author STALLS when asked to write a
whole hard kernel in one shot (flash-attention from scratch was ``offline_reject``ed;
the from-scratch attn_decode was wrong). Humans don't write FlashAttention
monolithically either — they COMPOSE it from a small set of idioms they already
trust: an online-softmax step, a tiled ``nc_matmul`` with PSUM accumulation, a
KV-tile loop. Each idiom is individually simple and individually verified; the
kernel is their assembly.

This module gives the author that vocabulary. It ships a ``PrimitiveLibrary`` of
VERIFIED NKI building blocks (each an on-device-validated idiom, not a full
kernel), a per-op-family DECOMPOSITION (which primitives compose that op), and a
prompt transform that hands the author the relevant verified blocks with a single
instruction: **compose from these, do NOT reinvent them.** That converts the hard
task ("invent a whole flash kernel") into the easy one ("assemble three idioms
you've been given"), which is exactly the failure mode we observed — too much at
once.

Design (composes with the existing seams, changes nothing else):
  * ``make_compose_prompt_fn(library)`` returns a drop-in ``build_prompt`` callable
    with the SAME signature as ``kernel_author.build_author_prompt``. For a HARD op
    with a known decomposition and verified primitives available, it PREPENDS the
    building-blocks section; for every other op it delegates to the base prompt
    UNCHANGED (byte-identical), so a composing author is safe to use for all ops.
  * ``ComposingAuthor`` is a ``KernelAuthor`` that owns an ``LLMAuthor`` wired with
    that compose prompt fn — so it reuses all of LLMAuthor's extraction / op-budget
    / provider plumbing and only differs in the prompt.

Honesty: only VERIFIED primitives are offered as building blocks (you do not tell
a model to trust an unvalidated idiom). The composed kernel is STILL gated by the
engine's offline + on-device correctness race exactly as any authored kernel —
composition changes how the author is prompted, never what is trusted. A
primitive earns ``verified=True`` only from an on-device validation (the default
library's blocks were validated during this project; ``PrimitiveLibrary.add`` /
``from_kernel_library`` let the bank contribute more as they are proven).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Op families that author monolithically-hard kernels — the ones that benefit
# from decomposition. Matches kernel_author._HARD_OP_FAMILIES (attention here is
# spelled to match nki_knowledge.classify_op labels).
_HARD_FAMILIES = frozenset({"attention", "scan", "matmul"})


@dataclass(frozen=True)
class Primitive:
    """One verified NKI building block — an idiom, not a whole kernel.

    ``families`` are the op-families this block composes into. ``nki_src`` is the
    compact, lint-compliant idiom the author is shown as a trusted block.
    ``verified`` gates whether it is OFFERED (only on-device-validated idioms are
    handed to the author as trusted). ``note`` is the one-line "what it does /
    why it's trusted" the prompt shows above the snippet."""

    name: str
    families: frozenset
    nki_src: str
    entry: str = ""
    verified: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# the default library — idioms validated on real silicon during this project
# ---------------------------------------------------------------------------
# Compact idioms (NOT full kernels): each is a straight-line snippet the author
# assembles into the target kernel. They obey the _NKI_PREAMBLE rules (return-form
# nc_matmul, keepdims-2D reductions, activation_reduce fusion) so a composed
# kernel starts from lint-clean building blocks.
_ONLINE_SOFTMAX = Primitive(
    name="online_softmax_step",
    families=frozenset({"attention", "scan"}),
    note="numerically-stable running softmax over a K-tile (negated-max trick, "
         "keepdims-2D) — validated on-device; the flash-attention inner step",
    nki_src=(
        "# online-softmax step over one scores tile s_ij [P, Ntile] (fp32):\n"
        "#   m_new = max(m_prev, rowmax(s_ij)); p = exp(s_ij - m_new)\n"
        "#   l_new = exp(m_prev - m_new)*l_prev + rowsum(p); rescale acc by\n"
        "#   exp(m_prev - m_new). Reductions stay [P,1] (keepdims).\n"
        "m_cur = nl.max(s_ij, axis=1, keepdims=True)          # [P,1]\n"
        "m_new = nl.maximum(m_prev, m_cur)                     # [P,1]\n"
        "# exp(x - m_new) + running denom in ONE Scalar-engine pass:\n"
        "p = nisa.activation(nl.exp, s_ij, bias=nl.negate(m_new), reduce_op=nl.add)\n"
        "alpha = nisa.activation(nl.exp, m_prev, bias=nl.negate(m_new))  # rescale\n"
    ),
)
_TILED_MATMUL = Primitive(
    name="tiled_psum_matmul",
    families=frozenset({"attention", "scan", "matmul"}),
    note="tiled nc_matmul with PSUM accumulation, moving free-dim <=512 — the "
         "return-form call that actually lowers on trn2 (validated)",
    nki_src=(
        "# stationary [K,M] (K,M<=128), moving [K,N] (N<=512) -> psum [M,N].\n"
        "# Tile any free dim > 512 into <=512 chunks and ACCUMULATE in PSUM.\n"
        "# nc_matmul RETURNS the tile (no dst=/out=); assign it.\n"
        "psum = nl.zeros((M, N), dtype=nl.float32, buffer=nl.psum)\n"
        "for k0 in nl.affine_range(0, K, 128):\n"
        "    stat = lhsT[k0:k0+128, :]        # [Ktile, M]\n"
        "    mov  = rhs[k0:k0+128, :]         # [Ktile, N]\n"
        "    psum += nisa.nc_matmul(stat, mov)   # accumulate contraction in PSUM\n"
    ),
)
_KV_TILE_LOOP = Primitive(
    name="kv_tile_loop",
    families=frozenset({"attention"}),
    note="the flash outer loop: stream K/V in [P,dk]/[P,dv] tiles, one DMA per "
         "tile, carry (acc, m, l) in SBUF across tiles (no HBM round-trip)",
    nki_src=(
        "# carry the running (acc [P,dv], m [P,1], l [P,1]) across KV tiles in\n"
        "# SBUF; ONE multi-partition DMA per K/V tile (never a per-index slice).\n"
        "acc = nl.zeros((P, dv), dtype=nl.float32)\n"
        "m_prev = nl.full((P, 1), -1e30, dtype=nl.float32)\n"
        "l_prev = nl.zeros((P, 1), dtype=nl.float32)\n"
        "for j0 in nl.affine_range(0, S_kv, TILE):\n"
        "    k_tile = nl.load(K[j0:j0+TILE, :])   # one DMA\n"
        "    v_tile = nl.load(V[j0:j0+TILE, :])   # one DMA\n"
        "    # scores = q @ k_tile.T (tiled_psum_matmul), then online_softmax_step,\n"
        "    # then acc = acc*alpha + p @ v_tile ; update m_prev,l_prev = m_new,l_new\n"
    ),
)
_ACTIVATION_REDUCE = Primitive(
    name="activation_reduce_fuse",
    families=frozenset({"scan"}),
    note="fused square-then-reduce in ONE Scalar-engine instruction with a [P,1] "
         "out-param — the ONLY compiling fused-reduce form on 0.6.0 (validated)",
    nki_src=(
        "# mean-square in one pass: allocate the [P,1] reduce_res out-param and\n"
        "# call activation_reduce (return-form does NOT return the reduction).\n"
        "ms = nl.zeros((P, 1), dtype=nl.float32)\n"
        "nisa.activation_reduce(op=nl.square, data=x, reduce_op=nl.add, reduce_res=ms)\n"
        "inv = nl.rsqrt(nl.multiply(ms, 1.0 / H))    # 1/rms, [P,1]\n"
    ),
)

# Which verified primitives compose each hard op family, in assembly order.
DECOMPOSITIONS: dict[str, list[str]] = {
    "attention": ["kv_tile_loop", "tiled_psum_matmul", "online_softmax_step"],
    "scan": ["activation_reduce_fuse", "tiled_psum_matmul", "online_softmax_step"],
    "matmul": ["tiled_psum_matmul"],
}


class PrimitiveLibrary:
    """A registry of verified NKI building blocks, queryable by op-family."""

    def __init__(self, primitives: list[Primitive] | None = None) -> None:
        self._by_name: dict[str, Primitive] = {}
        for p in (primitives or []):
            self.add(p)

    def add(self, prim: Primitive) -> None:
        self._by_name[prim.name] = prim

    def get(self, name: str) -> Primitive | None:
        return self._by_name.get(name)

    def for_family(self, family: str) -> list[Primitive]:
        """The VERIFIED primitives that compose ``family``, in decomposition
        order. Only verified blocks are returned (unverified idioms are never
        offered as trusted). Unknown family -> []."""
        order = DECOMPOSITIONS.get((family or "").lower(), [])
        out: list[Primitive] = []
        for name in order:
            p = self._by_name.get(name)
            if p is not None and p.verified:
                out.append(p)
        return out

    def from_kernel_library(self, kernel_library: object) -> "PrimitiveLibrary":
        """Ingest additional verified idioms the bank has proven (best-effort).
        A ``kernel_library`` exposing ``primitives()`` (each a ``Primitive`` or a
        mapping with the same fields) contributes them; anything else is ignored.
        Returns self for chaining. Never raises."""
        try:
            getter = getattr(kernel_library, "primitives", None)
            for rec in (getter() if callable(getter) else []):
                if isinstance(rec, Primitive):
                    if rec.verified:
                        self.add(rec)
                elif isinstance(rec, dict) and rec.get("verified", True):
                    self.add(Primitive(
                        name=rec["name"],
                        families=frozenset(rec.get("families", [])),
                        nki_src=rec.get("nki_src", ""),
                        entry=rec.get("entry", ""),
                        verified=True, note=rec.get("note", "")))
        except Exception:  # noqa: BLE001 — bank ingestion is best-effort
            pass
        return self


def default_library() -> PrimitiveLibrary:
    """The built-in library of on-device-validated idioms."""
    return PrimitiveLibrary([
        _ONLINE_SOFTMAX, _TILED_MATMUL, _KV_TILE_LOOP, _ACTIVATION_REDUCE])


# ---------------------------------------------------------------------------
# prompt transform
# ---------------------------------------------------------------------------
def _family_of(spec: object) -> str:
    """The op-family for a spec, via nki_knowledge.classify_op (best-effort)."""
    try:
        from nki_knowledge import classify_op
        return classify_op(getattr(spec, "name", "") or "",
                           getattr(spec, "family", None),
                           getattr(spec, "notes", None))
    except Exception:  # noqa: BLE001
        return (getattr(spec, "family", "") or "").lower()


def compose_section(spec: object, library: PrimitiveLibrary) -> str:
    """The "compose from these VERIFIED building blocks" block for a hard op, or
    "" when the op is not a hard family / has no verified primitives (in which
    case the caller uses the base prompt unchanged)."""
    fam = _family_of(spec)
    if fam not in _HARD_FAMILIES:
        return ""
    prims = library.for_family(fam)
    if not prims:
        return ""
    order = " -> ".join(p.name for p in prims)
    blocks = []
    for p in prims:
        blocks.append(f"### [{p.name}] {p.note}\n```python\n{p.nki_src}```")
    return (
        "## COMPOSE — do NOT write this kernel from scratch\n"
        f"This is a `{fam}` op: hard to author monolithically. You are given the "
        f"VERIFIED building blocks below (each an on-device-validated idiom). "
        f"ASSEMBLE the kernel from them in this order: {order}. Reuse each block's "
        f"idiom verbatim where it applies; do NOT reinvent an online-softmax, a "
        f"tiled matmul, or a KV-tile loop that is already given to you. Adapt only "
        f"the shapes/indices to this op's inputs. The blocks obey every MANDATORY "
        f"rule already (return-form nc_matmul, keepdims-2D reductions, fused "
        f"activation_reduce), so composing from them starts you lint-clean.\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
    )


def make_compose_prompt_fn(library: PrimitiveLibrary | None = None,
                           base: Callable[..., str] | None = None
                           ) -> Callable[..., str]:
    """Return a drop-in ``build_prompt`` (same signature as
    ``kernel_author.build_author_prompt``) that PREPENDS the verified
    building-blocks section for hard ops and delegates to ``base`` unchanged for
    every other op. Use it as ``LLMAuthor(complete_fn, build_prompt=<this>)``."""
    lib = library or default_library()
    if base is None:
        from kernel_author import build_author_prompt as base  # noqa: PLC0415

    def build_prompt(spec, lessons, feedback, perf_feedback=None, **kwargs) -> str:
        head = compose_section(spec, lib)
        body = base(spec, lessons, feedback, perf_feedback, **kwargs)
        return f"{head}{body}" if head else body

    return build_prompt


class ComposingAuthor:
    """A ``KernelAuthor`` that composes hard kernels from verified primitives.

    Owns an ``LLMAuthor`` wired with the compose prompt fn, so it reuses all of
    LLMAuthor's extraction / op-budget / provider plumbing and differs ONLY in the
    prompt (hard ops get the building-blocks section; other ops are byte-identical
    to a plain LLMAuthor). Construct with the same ``complete_fn`` you would give
    an ``LLMAuthor``.
    """

    def __init__(self, complete_fn: Callable[[str], str],
                 library: PrimitiveLibrary | None = None) -> None:
        from kernel_author import LLMAuthor  # noqa: PLC0415 — avoid import cycle
        self.library = library or default_library()
        self._llm = LLMAuthor(complete_fn,
                              build_prompt=make_compose_prompt_fn(self.library))

    def author(self, spec, lessons=None, feedback=None, perf_feedback=None):
        return self._llm.author(spec, lessons, feedback, perf_feedback)
