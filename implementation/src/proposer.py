"""
Config proposer — beam search over configs, seeded by bank priors.

Revised from the original "hand-priored greedy" after reading Autocomp, which
proved beam search on Trainium/NKI specifically. Beam is barely more complex
than greedy (keep top-k rather than top-1) and is far more robust to the local
optima that greedy falls into when config axes interact — which they do (a
fusion that helps at TP=8/bf16 can hurt at TP=16/fp8).

We keep the hand-priored part: bank config_priors seed the initial beam and
bias which axes get perturbed first.

This proposer covers Stage 1 (config). Stages 2-5 (kernels) are driven by the
harvest manifest and profile-guided candidate generation, not this module.

See ../../references-analysis.md and ../../optimization-stages.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bank import KnowledgeBank, Lesson, LessonType, Tier
from hardware import ComputeBudget, fill_plan
from ledger import Layer, LAYER_DURABILITY


@dataclass
class Candidate:
    config: dict[str, Any]
    metric: float = 0.0
    parent: dict[str, Any] | None = None
    provenance: str = ""       # what changed vs. parent, for the ledger description
    layer: Layer = Layer.CONFIG

    def key(self) -> tuple:
        return tuple(sorted(self.config.items()))


class BeamProposer:
    """Beam search over the backend's config axes.

    Parameters mirror Autocomp's defaults, which are tuned for this task:
      beam_size=4, num_plan_candidates=8.
    """

    def __init__(
        self,
        axes: dict[str, list[Any]],
        bank: KnowledgeBank | None = None,
        beam_size: int = 4,
        plans_per_parent: int = 8,
        budget: ComputeBudget | None = None,
        num_kv_heads: int | None = None,
        track: str = "throughput",
        long_context: bool = False,
    ) -> None:
        self.axes = dict(axes)
        self.bank = bank
        self.beam_size = beam_size
        self.plans_per_parent = plans_per_parent
        # Hardware awareness: when a budget is given, every candidate is filled
        # to use the whole instance (see _fill). Without it, behavior is
        # unchanged — this keeps the mock/unit path and any budget-less caller
        # working exactly as before.
        self.budget = budget
        self.num_kv_heads = num_kv_heads
        self.track = track
        # Context parallelism becomes a search axis for the long-context track
        # AND the latency track. Rationale: DP replicas raise *aggregate*
        # throughput but do nothing for a single request's latency, so the
        # "fastest possible" (latency) track fills the box with tp/cp — more
        # cores working on one request — instead. For a plain short-shape
        # throughput run, cp just burns search budget since DP fill already
        # uses every core.
        if (budget is not None and (long_context or track == "latency")
                and "cp_degree" not in self.axes):
            self.axes["cp_degree"] = [c for c in (1, 2, 4, 8) if c <= budget.num_cores]
        # Try high-impact axes FIRST. The overnight run showed the greedy stop
        # firing before it ever reached compile_mode — the single biggest lever
        # (torch.compile was ~3-6x while tp/dtype were single-digit %). Ordering
        # by expected impact means the win is found in round 1, not missed.
        self.axes = dict(sorted(self.axes.items(),
                                key=lambda kv: self._AXIS_PRIORITY.get(kv[0], 50)))
        self._seen: set[tuple] = set()

    # Lower = tried earlier. Compile mode dominates; sharding/precision are
    # cheap tie-breakers that interact, so they come after the big lever.
    _AXIS_PRIORITY = {
        "compile_mode": 0,
        "attn_implementation": 1, "attention_kernel": 1,
        "batching": 2,
        "weights_dtype": 3, "activations_dtype": 3, "kv_cache_dtype": 4,
        "sequence_layout": 5,
        "tp_degree": 6, "cp_degree": 7,
    }

    def _fill(self, config: dict[str, Any]) -> dict[str, Any]:
        """Attach the parallelism fill-plan (dp/cp/util + kv_replication) so a
        candidate uses the whole instance, not just its TP group. No-op when no
        budget was supplied."""
        if self.budget is None:
            return config
        tp = int(config.get("tp_degree", 1) or 1)
        cp = int(config.get("cp_degree", 1) or 1)
        plan = fill_plan(self.budget, tp=tp, cp=cp,
                         num_kv_heads=self.num_kv_heads, track=self.track)
        cfg = dict(config)
        cfg.update(plan.as_config())
        return cfg

    # -- seeding -------------------------------------------------------------

    def seed(
        self,
        baseline: dict[str, Any],
        family: str,
        param_count: float,
        seq_len: int,
        batch: int,
        sdk_version: str,
    ) -> list[Candidate]:
        """Initial beam: baseline plus any bank config_priors that apply.

        A good prior can land within a few percent of the Stage-1 optimum on
        the first candidate, which is the whole point of the knowledge bank.
        """
        beam = [Candidate(config=self._fill(dict(baseline)), provenance="baseline")]
        self._seen.add(beam[0].key())

        if self.bank is not None:
            priors = self.bank.query_interventions(
                family=family, param_count=param_count, seq_len=seq_len,
                batch=batch, sdk_version=sdk_version,
                types=(LessonType.CONFIG_PRIOR,),
            )
            for lesson in priors[: self.beam_size]:
                cfg = self._fill({**baseline, **lesson.intervention.get("spec", {})})
                cand = Candidate(
                    config=cfg, parent=baseline,
                    provenance=f"prior:{lesson.lesson_id}",
                )
                if cand.key() not in self._seen:
                    self._seen.add(cand.key())
                    beam.append(cand)
        return beam

    # -- expansion -----------------------------------------------------------

    def expand(self, beam: list[Candidate]) -> list[Candidate]:
        """Generate the next round of candidates by perturbing one axis at a
        time from each beam member.

        Single-axis perturbation is deliberate: it keeps credit assignment
        clean. When a candidate improves, we know which axis did it — the
        feedback-richness-vs-credit-assignment tension the ADAS survey names.
        """
        out: list[Candidate] = []
        for parent in beam:
            # Generate EVERY unseen single-axis neighbour, in priority order.
            # Generating all of them (rather than a capped subset) is what
            # guarantees the decisive axis — compile_mode, or tp — is always
            # tried in the round, instead of being starved by a high-fan-out
            # axis. The overnight run halted before torch.compile precisely
            # because the capped expansion never emitted a compile_mode
            # candidate. The beam still keeps only top-k, so fan-out is bounded
            # downstream, and the round-level stop check evaluates the whole
            # round before deciding to stop.
            for axis, values in self.axes.items():       # dict is priority-sorted
                for v in values:
                    if parent.config.get(axis) == v:
                        continue
                    # Re-fill after the perturbation: changing tp (or cp)
                    # changes how many DP replicas fit, so dp/util are
                    # recomputed rather than inherited from the parent.
                    cfg = self._fill({**parent.config, axis: v})
                    cand = Candidate(
                        config=cfg, parent=parent.config,
                        provenance=f"{axis}={v}",
                    )
                    if cand.key() in self._seen:
                        continue
                    self._seen.add(cand.key())
                    out.append(cand)
        return out

    # -- selection -----------------------------------------------------------

    def select(self, scored: list[Candidate]) -> list[Candidate]:
        """Keep the top beam_size, breaking ties toward more durable layers.

        The layer tiebreaker is the migration-risk preference made concrete: a
        kernel-level win beats a framework-level win of equal magnitude,
        because one survives the XLA -> native-PyTorch migration.
        """
        return sorted(
            scored,
            key=lambda c: (-c.metric, LAYER_DURABILITY.get(c.layer, 9)),
        )[: self.beam_size]

    def exhausted(self, beam: list[Candidate]) -> bool:
        """True when every single-axis move from the beam has been seen."""
        for parent in beam:
            for axis, values in self.axes.items():
                for v in values:
                    if parent.config.get(axis) == v:
                        continue
                    cfg = self._fill({**parent.config, axis: v})
                    if tuple(sorted(cfg.items())) not in self._seen:
                        return False
        return True
