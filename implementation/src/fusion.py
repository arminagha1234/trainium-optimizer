# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""fusion.py — CROSS-OP FUSION GROUPS: author ONE megakernel for adjacent ops.

The framework authors op-by-op, but the largest wins on Trainium come from fusing
ACROSS ops: a pre-norm folded into the attention it feeds, a MoE router folded
into the expert matmul it dispatches, an attention output folded into its residual
add. Each boundary the compiler leaves un-fused is a full intermediate
round-tripped through HBM — and there is no hardware cache, so that traffic is
pure loss. Fusing the group into one kernel keeps the intermediate resident in
SBUF (the #1 perf lever the author preamble already preaches, applied at the GRAPH
level instead of within a single op).

This module is the graph-level analogue of ``opportunity.select_targets``: instead
of ranking single ops, it finds ADJACENT op runs that share data and are worth
fusing, and materializes each as a single ``OpSpec`` "megakernel" target the
existing author/gate/bank pipeline can consume unchanged.

  * ``FUSABLE_PAIRS``          — the ordered (producer_family, consumer_family)
                                 boundaries worth fusing on trn2 (each removes an
                                 HBM round-trip and/or feeds a compiler-weak op).
  * ``detect_fusion_groups``   — over an EXECUTION-ORDERED spec list, find maximal
                                 runs whose every consecutive boundary is fusable.
  * ``fused_spec``             — compose a group into one ``OpSpec`` (name
                                 ``fused_<a>_<b>``, reference = members applied in
                                 sequence, inputs from the head op).
  * ``rank_fusion_groups`` /
    ``select_fusion_targets``  — score groups (fusion always removes a round-trip;
                                 a group feeding a compiler-weak op scores higher)
                                 and return the fused megakernel specs to author.

Pure-python; composes with ``opportunity`` (family scores) and
``nki_knowledge.classify_op`` (families). No device dependency. Honest: the fused
reference is the exact sequential composition of the members' references, so the
engine's correctness gate validates the megakernel against real ground truth —
fusion changes WHAT we author, never what we trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from invent_kernels import OpSpec

# Ordered (producer_family -> consumer_family) boundaries worth fusing on trn2.
# Each is a real megakernel opportunity: the producer's output feeds the consumer
# directly, so fusing keeps it in SBUF instead of spilling it to HBM. Families use
# nki_knowledge.classify_op labels.
FUSABLE_PAIRS: frozenset = frozenset({
    ("normalization", "attention"),   # pre-norm folded into attention
    ("normalization", "matmul"),      # norm folded into the projection it feeds
    ("attention", "elementwise"),     # attention output + residual add
    ("attention", "normalization"),   # post-attn norm
    ("moe_router", "matmul"),         # router dispatch folded into the expert GEMM
    ("matmul", "elementwise"),        # GEMM + bias/activation epilogue
    ("elementwise", "normalization"), # residual + next-block norm
    ("softmax", "matmul"),            # attention probs @ V
})

# A group that FEEDS one of these compiler-weak families is extra-worth fusing
# (the fused kernel also lands in the regime where authoring beats the compiler).
_COMPILER_WEAK = frozenset({"attention", "scan"})


@dataclass(frozen=True)
class FusionGroup:
    """A run of adjacent ops worth fusing into one megakernel."""
    members: tuple                    # tuple[OpSpec, ...], in execution order
    reason: str = ""

    @property
    def names(self) -> list[str]:
        return [getattr(m, "name", "?") for m in self.members]

    @property
    def size(self) -> int:
        return len(self.members)


def _family_of(spec: Any) -> str:
    try:
        from nki_knowledge import classify_op
        return classify_op(getattr(spec, "name", "") or "",
                           getattr(spec, "family", None),
                           getattr(spec, "notes", None))
    except Exception:  # noqa: BLE001
        return (getattr(spec, "family", "") or "").lower()


def _is_fusable(prod: Any, cons: Any,
                fusable: frozenset = FUSABLE_PAIRS) -> bool:
    """True when producer->consumer is a fusable boundary."""
    return (_family_of(prod), _family_of(cons)) in fusable


def detect_fusion_groups(specs: list, fusable: frozenset = FUSABLE_PAIRS,
                         max_group: int = 3) -> list[FusionGroup]:
    """Find maximal runs of adjacent specs whose every consecutive boundary is
    fusable, over an EXECUTION-ORDERED ``specs`` list. Returns groups of size >= 2
    (a single op is not a fusion), each capped at ``max_group`` members (a
    megakernel over too many ops explodes the authoring difficulty; 3 is a sane
    single-chip bound). Order-preserving; never raises on odd specs."""
    groups: list[FusionGroup] = []
    i, n = 0, len(specs)
    while i < n - 1:
        run = [specs[i]]
        j = i
        while (j + 1 < n and len(run) < max_group
               and _is_fusable(specs[j], specs[j + 1], fusable)):
            run.append(specs[j + 1])
            j += 1
        if len(run) >= 2:
            fams = " -> ".join(_family_of(m) for m in run)
            groups.append(FusionGroup(
                members=tuple(run),
                reason=(f"adjacent fusable boundary chain [{fams}] — fusing keeps "
                        f"{len(run) - 1} intermediate(s) in SBUF instead of "
                        f"round-tripping through HBM")))
            i = j + 1           # consume the whole run
        else:
            i += 1
    return groups


def _primary_input_key(spec: Any) -> str | None:
    """The head op's primary input key (first key of its offline inputs) — the
    slot each member's previous output is threaded into. None if unavailable."""
    try:
        inp = spec.offline_inputs()
        if isinstance(inp, dict) and inp:
            return next(iter(inp))
    except Exception:  # noqa: BLE001
        pass
    return None


def fused_spec(group: FusionGroup) -> OpSpec:
    """Compose a ``FusionGroup`` into ONE ``OpSpec`` megakernel target.

    The fused reference applies the members' references in sequence: member 0 runs
    on the original inputs; each subsequent member runs on the original input dict
    with ITS primary input key overridden by the previous member's output (the
    common producer->consumer data flow). Inputs / dtype / shape_class come from
    the head op. The result is a genuine ground-truth reference, so the engine's
    correctness gate validates the megakernel exactly as any op — no trust is
    assumed from fusion."""
    members = list(group.members)
    head = members[0]
    fused_name = "fused_" + "_".join(m.name for m in members)

    def reference(inp: dict) -> Any:
        out = members[0].reference(inp)
        for m in members[1:]:
            key = _primary_input_key(m)
            sub = dict(inp)
            if key is not None:
                sub[key] = out
            out = m.reference(sub)
        return out

    fams = " -> ".join(_family_of(m) for m in members)
    return OpSpec(
        name=fused_name,
        family=getattr(head, "family", "fused"),
        shape_class=getattr(head, "shape_class", "fused"),
        dtype=getattr(head, "dtype", "bf16"),
        reference=reference,
        offline_inputs=head.offline_inputs,
        real_inputs=head.real_inputs,
        baseline=getattr(head, "baseline", "torch-eager"),
        notes=(f"FUSED megakernel over [{fams}]. {group.reason}. Author ONE kernel "
               f"that computes the whole chain, keeping the {len(members) - 1} "
               f"intermediate(s) resident in SBUF (no HBM round-trip)."),
        primitive="fused",
    )


@dataclass(frozen=True)
class FusionTarget:
    """A scored fusion group + its materialized megakernel spec."""
    group: FusionGroup
    spec: OpSpec
    score: float
    reason: str


def _group_score(group: FusionGroup) -> float:
    """0..1 worth-of-fusing score. Base credit for each removed round-trip
    (more members fused == more traffic saved), plus a bonus when the chain feeds
    a compiler-weak family (attention/scan) where the authored kernel also beats
    the compiler. Uses opportunity's per-family scores when available."""
    try:
        from opportunity import _FAMILY_OPPORTUNITY  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        _FAMILY_OPPORTUNITY = {}
    fams = [_family_of(m) for m in group.members]
    fam_score = max((_FAMILY_OPPORTUNITY.get(f, 0.2) for f in fams), default=0.2)
    # Each additional member removes one HBM round-trip: +0.15 each, capped.
    traffic = min(0.45, 0.15 * (group.size - 1))
    weak_bonus = 0.20 if any(f in _COMPILER_WEAK for f in fams) else 0.0
    return max(0.0, min(1.0, fam_score * 0.5 + traffic + weak_bonus))


def rank_fusion_groups(specs: list, fusable: frozenset = FUSABLE_PAIRS,
                       max_group: int = 3) -> list[FusionTarget]:
    """Detect fusion groups over ``specs`` and rank them, most-worth first — each
    with its materialized megakernel ``OpSpec``."""
    out: list[FusionTarget] = []
    for g in detect_fusion_groups(specs, fusable, max_group):
        score = _group_score(g)
        out.append(FusionTarget(
            group=g, spec=fused_spec(g), score=score,
            reason=f"{g.reason} (score={score:.2f})"))
    out.sort(key=lambda t: t.score, reverse=True)
    return out


def select_fusion_targets(specs: list, max_targets: int | None = None,
                          fusable: frozenset = FUSABLE_PAIRS,
                          max_group: int = 3) -> list[OpSpec]:
    """The fused megakernel ``OpSpec``s the framework should author, highest-worth
    first, optionally capped at ``max_targets``. Returns [] when no adjacent ops
    are fusable (an honest 'nothing to fuse here')."""
    ranked = rank_fusion_groups(specs, fusable, max_group)
    specs_out = [t.spec for t in ranked]
    return specs_out[:max_targets] if max_targets is not None else specs_out
