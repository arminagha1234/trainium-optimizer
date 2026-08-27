"""op_signature.py — robust resolution of a model's architecture *primitive* to
the canonical kernel-corpus name, so cross-model kernel reuse ("N+1 inherits
N's kernel") is not defeated by a near-miss spelling.

The problem this fixes (Station-5 bank-compounding, retrieval side):
``kernel_registry.kernel_for_primitive`` maps a FIXED set of normalized
descriptor spellings to a kernel name via an exact dict hit. That is correct but
BRITTLE — a model exposing a primitive string that *embeds* a known descriptor
but is not itself a dict key silently misses:

    "qwen3next_gated_delta"  -> _norm -> "qwen3nextgateddelta"  (NOT a key)
    "Mamba2SSDMixer"         -> _norm -> "mamba2ssdmixer"       (NOT a key)

so ``_prior_art`` returns None and the model re-invents a kernel the corpus
already has. This module adds a SAFE fallback: after the exact hit fails, look
for a known descriptor that appears as a substring of the normalized primitive
(and, optionally, of the op name), preferring the LONGEST match.

Safety rules that keep the fuzzy fallback from mis-routing a kernel (mis-routing
is the dangerous failure — a Mamba kernel is not a DeltaNet kernel):
  * exact match ALWAYS wins and is tried first (byte-identical to today);
  * only descriptors of length >= ``_MIN_FUZZY_LEN`` are eligible as substrings,
    so short/ambiguous tokens ("mla", "kda", "ssm", "flash") match ONLY exactly,
    never as a loose substring of an unrelated word;
  * the LONGEST eligible descriptor wins, so "gateddeltalinearattention" is
    preferred over "gateddelta" when both are present;
  * a fuzzy hit only ever resolves to an EXISTING corpus kernel name — it can
    turn a former miss into a (still primitive-specific) hit, never change an
    existing exact hit, and the perf/repair loop re-validates on device anyway.

Pure stdlib; imports the name map from ``kernel_registry`` (the single source of
truth for descriptor -> kernel-name) so the two never drift.
"""

from __future__ import annotations

from kernel_registry import PRIMITIVE_TO_KERNEL, _norm, kernel_for_primitive

# A descriptor must be at least this many normalized chars to be eligible for
# the SUBSTRING fallback. Short descriptors ("mla"=3, "kda"=3, "ssm"=3,
# "flash"=5, "mamba"=5, "rwkv6"=5) are excluded from fuzzy matching — they still
# resolve on an exact hit, but never as a loose substring of an unrelated token.
_MIN_FUZZY_LEN = 6

# Descriptors eligible for the substring fallback, longest first (so the most
# specific match wins). Computed once from the shared name map.
_FUZZY_KEYS: tuple[str, ...] = tuple(sorted(
    (k for k in PRIMITIVE_TO_KERNEL if len(k) >= _MIN_FUZZY_LEN),
    key=len, reverse=True))


def _fuzzy_lookup(normalized: str) -> str | None:
    """The canonical kernel name for the LONGEST eligible descriptor that is a
    substring of ``normalized``, or None. ``normalized`` must already be
    ``_norm``-ed."""
    if not normalized:
        return None
    for key in _FUZZY_KEYS:            # longest-first
        if key in normalized:
            return PRIMITIVE_TO_KERNEL[key]
    return None


def resolve_kernel_name(primitive: str, op_name: str = "") -> str | None:
    """Canonical kernel-corpus name for a primitive descriptor, robust to
    near-miss spellings, or None if nothing plausibly matches.

    Resolution order (each strictly more permissive than the last):
      1. EXACT ``kernel_for_primitive(primitive)`` — unchanged, byte-identical.
      2. SUBSTRING of the normalized primitive against eligible descriptors
         (longest wins) — catches "qwen3next_gated_delta", "Mamba2SSDMixer", ...
      3. SUBSTRING of the normalized OP NAME, if given — a last resort for when
         the primitive field is empty/unhelpful but the op name carries the
         family (e.g. op "gated_delta_rule" with primitive "").

    Never raises; returns None on no match so callers keep their "no kernel
    available -> author one" path."""
    exact = kernel_for_primitive(primitive)
    if exact is not None:
        return exact
    hit = _fuzzy_lookup(_norm(primitive))
    if hit is not None:
        return hit
    if op_name:
        return _fuzzy_lookup(_norm(op_name))
    return None
