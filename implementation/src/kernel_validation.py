"""kernel_validation.py — the RANK LADDER + reuse-vs-author router.

Once a kernel is injected (backends.kernel_inject) and run, SOMETHING has to
decide: is this kernel trustworthy enough to reuse, or does the pipeline need to
keep working / author from scratch? That decision is the whole point of the
outcome ladder in `kernel_registry.STATUS_RANK`, and this module makes it
operational — it turns a raw run outcome into a ranked status, and a ranked
status into a routing decision.

Two pieces, both consolidated here so there is ONE definition of "what counts as
a pass" and ONE router:

  1. ``verdict(numerics_ok, neff_emitted, on_device) -> status`` — the single
     honest gate. A kernel PASSES only when its numerics match (allclose) AND it
     actually emitted a non-empty NEFF. Importing the kernel without a compiled
     NEFF is NOT a pass (the classic overstatement: "it imported, ship it"). The
     same pass on a REAL device is a strictly higher tier than a simulate-only
     pass — the single most important lesson from the AutoFixer corpus, where a
     Mamba scan simulated to 2e-7 ran ~67 max_abs_diff off on silicon.

  2. ``reuse_decision(spec_or_rank) -> REUSE | REVALIDATE_ON_DEVICE | CONTINUE |
     AUTHOR`` — the router the orchestrator consults, mirroring the AutoFixer
     "search prior art before authoring" flow. Crucially it NEVER returns
     "blocker": a novel primitive is always at worst an AUTHOR work item, never a
     dead end (see kernel_registry's module doc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernel_registry import (
    MIN_HW_READY_RANK,
    MIN_USABLE_RANK,
    STATUS_RANK,
)

# The adversarial anti-cheat layer lives in its own module (pure-Python, no
# Trainium) but is re-exported here so `verdict`'s `adversarial_ok` veto and the
# checks that feed it share one import surface for callers.
from kernel_anticheat import (  # noqa: F401 — re-exported for callers
    adversarial_source_check,
    require_reproducible,
    run_candidate_before_reference,
)

# Router verdicts. Named constants (not bare strings) so a typo is a NameError,
# not a silently-wrong route.
REUSE = "REUSE"
REVALIDATE_ON_DEVICE = "REVALIDATE_ON_DEVICE"
CONTINUE = "CONTINUE"
AUTHOR = "AUTHOR"

# Canonical status strings the verdict emits (all present in STATUS_RANK).
PASSED = "passed"                       # rank 3 — numerics + NEFF, in simulation
PASSED_ON_DEVICE = "passed-on-device"   # rank 4 — same, on real silicon
FAILED_NUMERICAL = "failed-numerical"   # rank 2 — compiled/NEFF but numerics off
FAILED_COMPILE = "failed-compile"       # rank 1 — no NEFF (incl. import-only)
FAILED_ADVERSARIAL = "failed-adversarial"  # rank 0 — rejected by the anti-cheat


def verdict(numerics_ok: bool, neff_emitted: bool, on_device: bool,
            adversarial_ok: bool = True) -> str:
    """Consolidate one run outcome into a ladder status string.

    The truth table (see kernel_registry.STATUS_RANK for the ranks):

        adversarial_ok is False                -> "failed-adversarial" (0)
        numerics_ok & neff_emitted & on_device -> "passed-on-device"   (4)
        numerics_ok & neff_emitted             -> "passed"             (3)
        neff_emitted (numerics failed)         -> "failed-numerical"   (2)
        no NEFF (import-only / empty NEFF)     -> "failed-compile"     (1)

    The invariants this encodes:
      * A NEFF is REQUIRED for a pass. ``numerics_ok`` alone (e.g. the kernel
        imported and a numpy-level check passed) but with NO emitted NEFF is
        rank-1 "failed-compile" — NOT a pass. "It imported" never ships.
      * ``on_device`` only ELEVATES an already-passing result (3 -> 4). It never
        rescues a numerics/compile failure — a device run that miscompiles is
        still a failure.
      * ``adversarial_ok`` is the reward-hack veto (see kernel_anticheat). It
        defaults True so honest callers and existing tests are unchanged. When
        False — the candidate failed the static anti-cheat or the reproducibility
        gate — it OVERRIDES everything and forces "failed-adversarial" (rank 0),
        so a kernel that gamed the numerics gate (Sakana's silent framework
        fallback, Kevin's recycled output buffer) can NEVER reach a pass rank,
        no matter how good its faked numerics looked.
    """
    if not adversarial_ok:
        return FAILED_ADVERSARIAL
    if numerics_ok and neff_emitted:
        return PASSED_ON_DEVICE if on_device else PASSED
    if neff_emitted:
        return FAILED_NUMERICAL
    return FAILED_COMPILE


@dataclass
class KernelValidation:
    """The result of validating one kernel run.

    ``status`` is a ladder string (keys of STATUS_RANK); ``rank`` is its numeric
    rank; ``tier`` records WHICH environment produced it ("simulate" vs
    "on-device"), because a simulate pass and a device pass are the same
    ``status`` string only at different ranks and must never be conflated when
    deciding reuse. ``numeric_error`` is the measured max_abs_diff (or similar);
    ``artifact`` is the NEFF handle / path when one was emitted.
    """

    status: str
    rank: int
    tier: str                       # "simulate" | "on-device"
    numeric_error: float = float("inf")
    artifact: str = ""
    notes: str = ""

    @property
    def passed(self) -> bool:
        """Reusable at all (>= simulate-correct)."""
        return self.rank >= MIN_USABLE_RANK

    @property
    def hw_ready(self) -> bool:
        """Reusable as hardware-ready without re-verification."""
        return self.rank >= MIN_HW_READY_RANK

    @classmethod
    def from_run(cls, *, numerics_ok: bool, neff_emitted: bool,
                 on_device: bool, numeric_error: float = float("inf"),
                 artifact: str = "", notes: str = "",
                 adversarial_ok: bool = True,
                 adversarial_reasons: list[str] | None = None
                 ) -> "KernelValidation":
        """Build a KernelValidation directly from a run's raw signals, using the
        single ``verdict`` gate so status/rank/tier can never disagree.

        ``adversarial_ok`` (default True) is the anti-cheat veto; pass the
        concatenated output of ``kernel_anticheat.adversarial_source_check`` /
        ``require_reproducible`` as ``adversarial_reasons`` and set
        ``adversarial_ok=False`` to reject a reward-hacking candidate. The reason
        is RECORDED in ``notes`` (honest audit trail) and the status is forced to
        rank-0 "failed-adversarial", so it can never route to reuse."""
        status = verdict(numerics_ok, neff_emitted, on_device, adversarial_ok)
        # tier reflects WHERE the run happened, independent of pass/fail, so a
        # failed-on-device result is still tagged on-device (honest audit trail).
        tier = "on-device" if on_device else "simulate"
        if not adversarial_ok:
            why = "; ".join(adversarial_reasons or ["adversarial gate rejected"])
            notes = f"{notes} [adversarial: {why}]".strip()
        return cls(status=status, rank=STATUS_RANK.get(status, 0), tier=tier,
                   numeric_error=numeric_error, artifact=artifact, notes=notes)


def _rank_of(spec_or_rank: Any) -> int | None:
    """Extract a numeric rank from a KernelSpec, a KernelValidation, a bare int,
    or None. Returns None when there is nothing to rank (empty corpus / no
    kernel), which the router treats as "author from scratch"."""
    if spec_or_rank is None:
        return None
    if isinstance(spec_or_rank, bool):   # guard: bool is an int subclass
        return int(spec_or_rank)
    if isinstance(spec_or_rank, int):
        return spec_or_rank
    r = getattr(spec_or_rank, "rank", None)
    if isinstance(r, int):
        return r
    return None


def reuse_decision(spec_or_rank: Any) -> str:
    """Route a kernel (by its rank) to the next action — mirroring AutoFixer:

        rank >= 4 (passed-on-device)  -> REUSE
                                          silicon-validated; reuse as HW-ready.
        rank == 3 (passed, simulate)  -> REVALIDATE_ON_DEVICE
                                          simulate != silicon (the Mamba lesson):
                                          a sim pass must be re-proven on device
                                          before it is trusted, not reused blind.
        rank >= 1 (some compile attempt) -> CONTINUE
                                          a failed-compile / compiled-but-off
                                          kernel exists; keep iterating the repair
                                          loop rather than starting over.
        none / rank < 1               -> AUTHOR
                                          no kernel (empty corpus) or only an
                                          analysis-only stub: author from scratch.

    Never returns a "blocker" — a novel primitive is always at worst an AUTHOR
    work item (kernel_registry's founding rule).
    """
    rank = _rank_of(spec_or_rank)
    if rank is None:
        return AUTHOR
    if rank >= MIN_HW_READY_RANK:            # 4
        return REUSE
    if rank == MIN_USABLE_RANK:              # 3 — simulate-only pass
        return REVALIDATE_ON_DEVICE
    if rank >= 1:                            # 1..2 — a real compile attempt exists
        return CONTINUE
    return AUTHOR                            # rank 0 (analysis-only) — nothing to reuse
