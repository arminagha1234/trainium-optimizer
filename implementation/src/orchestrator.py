"""
Orchestrator — walks the stage pipeline, runs the tournament, enforces
guardrails, applies keep/discard, and records everything to the ledger.

This is the backend-independent core. It knows nothing about XLA, NKI, or
tokens — it talks to a Backend, a KnowledgeBank, Guardrails, and a Ledger.

V1 implements Stage 0 (baseline), Stage 0.5 (harvest — stubbed), and Stage 1
(config search via beam). Stages 2-5 (kernel work) reuse the same tournament
shell with different candidate generators; they are wired but delegate kernel
authoring to the worker agents, which are not part of this module.

The tournament per stage:
    generate candidates
    -> prune anti-patterns (zero cost, before any compile)
    -> for each survivor: guardrail-check, compile, [equivalence], measure
    -> keep if it beats the incumbent (Stage-4 uses the invention margin)
    -> record every attempt to the ledger, keep or discard

See ../../optimization-stages.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backends.base import Backend, Measurements
from bank import KnowledgeBank
from guardrails import Guardrails, StoppingState
from ledger import (
    Layer,
    Ledger,
    Origin,
    Row,
    Stage,
    Status,
)
from proposer import BeamProposer, Candidate


@dataclass
class ModelSpec:
    """What the optimizer needs to know about the target model."""

    model_id: str
    family: str                 # dense_causal_lm | moe_causal_lm | ...
    param_count: float
    parent: str | None = None   # llama | qwen | ...
    probe_shape: str = "chat 1k/512"
    probe_batch: int = 1
    seq_len: int = 1024
    # Hardware-fit inputs. num_kv_heads bounds the clean TP degree (GQA rule);
    # the fill plan uses it to decide when DP replicas (throughput) or KV
    # replication (a testable option) are needed. track picks throughput vs
    # latency fill behavior.
    num_kv_heads: int | None = None
    track: str = "throughput"   # "throughput" | "latency"
    long_context: bool = False  # enables the context-parallel search axis


@dataclass
class EquivalenceResult:
    passed: bool
    correctness_pct: float = 100.0
    notes: str = ""


# An equivalence checker is injected so the orchestrator stays backend- and
# family-agnostic. Real impl calls the NAD equivalence agent; the mock passes.
EquivalenceFn = Callable[[Any, ModelSpec], EquivalenceResult]


def always_equivalent(_neff: Any, _spec: ModelSpec) -> EquivalenceResult:
    return EquivalenceResult(passed=True, correctness_pct=100.0)


@dataclass
class Orchestrator:
    backend: Backend
    bank: KnowledgeBank
    guards: Guardrails
    ledger: Ledger
    equivalence: EquivalenceFn = always_equivalent
    sdk_version: str = "2.28.0"
    # When set, the proposer fills the whole instance (DP replicas / CP) rather
    # than leaving cores idle beyond the TP group. Left None it's a no-op, so
    # existing budget-less callers and tests are unchanged. Real runs set it
    # (e.g. "trn2.48xlarge") — see overnight.py.
    instance_type: str | None = None

    # populated during a run
    incumbent: Candidate | None = None

    def establish_baseline(self, spec: ModelSpec) -> Candidate:
        """Stage 0. Measure the autoport baseline and set it as the incumbent.

        The baseline is the equivalence *reference* — it defines correctness,
        so it is NOT equivalence-gated (it is trivially equivalent to itself).
        Every later candidate is measured against it. This is also why the loop
        always has an incumbent even if every subsequent change fails its gate.
        """
        artifact = self.backend.build_baseline(spec.model_id)
        neff = self.backend.compile(artifact)
        m = self.backend.measure(neff, spec.probe_shape, spec.probe_batch)
        base = Candidate(config=artifact.config, provenance="baseline",
                         layer=Layer.NONE, metric=m.metric)
        self._record(base, Stage.BASELINE, Origin.NONE, Layer.NONE, source="",
                     metric=m.metric, correctness=100.0,
                     compile_s=neff.compile_seconds, status=Status.KEEP,
                     desc="baseline", mfu=m.mfu_percent)
        self.incumbent = base
        return base

    def run_stage1_config(self, spec: ModelSpec) -> Candidate:
        """Stage 1: config search via beam. Returns the stage's incumbent."""
        if self.incumbent is None:
            self.establish_baseline(spec)

        axes = self.backend.config_axes()
        budget = None
        if self.instance_type:
            from hardware import budget_for
            budget = budget_for(self.instance_type)
        proposer = BeamProposer(
            axes=axes, bank=self.bank, budget=budget,
            num_kv_heads=spec.num_kv_heads, track=spec.track,
            long_context=spec.long_context,
        )

        beam = proposer.seed(
            baseline=self.incumbent.config,
            family=spec.family, param_count=spec.param_count,
            seq_len=spec.seq_len, batch=spec.probe_batch,
            sdk_version=self.sdk_version,
        )

        # Score the seed beam (skip the pure-baseline member; already measured).
        scored_seed = []
        for c in beam:
            if c.provenance == "baseline":
                c.metric = self.incumbent.metric
                scored_seed.append(c)
                continue
            evaluated = self._evaluate(c, spec, Stage.CONFIG)
            if evaluated is not None:
                scored_seed.append(evaluated)
                self._update_incumbent(evaluated)
        beam = proposer.select(scored_seed) if scored_seed else [self.incumbent]

        state = StoppingState(guards=self.guards)
        # Don't let the SOFT stopping criteria (no-improvement / marginal) end
        # the search before every config axis has been tried at least once.
        # The overnight run halted before ever reaching compile_mode — the
        # single biggest lever (~3-6x) — because tp/dtype came first and
        # plateaued the streak. max_iterations stays a hard backstop regardless.
        all_axes = set(proposer.axes)
        explored: set[str] = set()

        def _stop_now(reason: str) -> bool:
            if reason == "max_iterations":
                return True                        # hard backstop always wins
            return explored.issuperset(all_axes)   # soft stops wait for coverage

        while True:
            stop, reason = state.should_stop()
            if stop and _stop_now(reason):
                break
            candidates = proposer.expand(beam)
            if not candidates:
                break

            # Anti-pattern pruning — zero cost, before any compile.
            family_dir = spec.family.replace("_", "-")
            configs = [c.config for c in candidates]
            _survivors, pruned = self.bank.prune(
                configs, family_dir, self.sdk_version, backend=self.backend.name)
            pruned_keys = {tuple(sorted(cfg.items())) for cfg, _ in pruned}
            for cfg, prune_reason in pruned:
                self._record_pruned(cfg, prune_reason)
            candidates = [c for c in candidates if c.key() not in pruned_keys]

            # Evaluate the WHOLE round before deciding to stop, so a round that
            # contains the decisive candidate (compile_mode, or tp=8) always
            # tries it. Stopping is a round-level decision, not per-candidate —
            # that per-candidate check is what let the streak fire mid-round,
            # before the big lever, in the overnight run.
            scored = []
            round_improved = False
            round_best_gain = 0.0
            for cand in candidates:
                if "=" in cand.provenance:          # "axis=value" -> axis seen
                    explored.add(cand.provenance.split("=", 1)[0])
                evaluated = self._evaluate(cand, spec, Stage.CONFIG)
                if evaluated is None:
                    continue
                scored.append(evaluated)
                gain = self._gain_pct(evaluated)    # vs the incumbent so far
                if self._update_incumbent(evaluated):
                    round_improved = True
                    round_best_gain = max(round_best_gain, gain)

            state.record(improved=round_improved, gain_pct=round_best_gain)
            beam = proposer.select(beam + scored)

        assert self.incumbent is not None
        return self.incumbent

    # -- tournament primitives ----------------------------------------------

    def _evaluate(
        self, cand: Candidate, spec: ModelSpec, stage: Stage,
        origin: Origin = Origin.NONE, layer: Layer = Layer.CONFIG,
        source: str = "",
    ) -> Candidate | None:
        """Compile -> guardrails -> equivalence -> measure. Records the attempt.

        Returns the scored candidate, or None if it failed a gate (which is
        itself recorded as a discard). Equivalence is a HARD gate: a faster
        config that produces different output is a bug, not a win.
        """
        artifact = self.backend.apply_config(
            self.backend.build_baseline(spec.model_id), cand.config
        )
        neff = self.backend.compile(artifact)

        # Compile-timeout guardrail.
        if not self.guards.compile_ok(neff.compile_seconds):
            self._record(cand, stage, origin, layer, source, metric=0.0,
                         correctness=0.0, compile_s=neff.compile_seconds,
                         status=Status.DISCARD,
                         desc=f"{cand.provenance} (compile timeout)")
            return None

        # Equivalence — hard gate, before performance is even considered.
        eq = self.equivalence(neff, spec)
        if not eq.passed:
            self._record(cand, stage, origin, layer, source, metric=0.0,
                         correctness=eq.correctness_pct,
                         compile_s=neff.compile_seconds, status=Status.DISCARD,
                         desc=f"{cand.provenance} (equivalence fail: {eq.notes})")
            return None

        m: Measurements = self.backend.measure(neff, spec.probe_shape, spec.probe_batch)

        # HBM + measurement-quality guardrails.
        if not self.guards.hbm_ok(m):
            self._record(cand, stage, origin, layer, source, metric=m.metric,
                         correctness=eq.correctness_pct,
                         compile_s=neff.compile_seconds, status=Status.DISCARD,
                         desc=f"{cand.provenance} (OOM {m.hbm_utilization:.0%} HBM)")
            return None
        if not self.guards.measurement_trustworthy(m):
            self._record(cand, stage, origin, layer, source, metric=m.metric,
                         correctness=eq.correctness_pct,
                         compile_s=neff.compile_seconds, status=Status.DISCARD,
                         desc=f"{cand.provenance} (noisy measurement)")
            return None

        cand.metric = m.metric
        cand.layer = layer
        # Keep/discard is decided against the incumbent by the caller path;
        # here we record the measured result. The status is finalized in
        # _update_incumbent via a follow-up row only when it becomes incumbent.
        is_invention = stage is Stage.INVENT
        beats = self._beats_incumbent(cand, is_invention)
        # Under-utilization is a soft flag, not a gate: record it in the
        # description so "left the box idle" is visible in the ledger and chart,
        # but never discard a correct, faster candidate for it.
        desc = cand.provenance
        if not self.guards.utilization_ok(m):
            desc = (f"{desc} [under-util: {m.device_utilization:.0%} of "
                    f"{m.cores_available} cores]")
        self._record(
            cand, stage, origin, layer, source, metric=m.metric,
            correctness=eq.correctness_pct, compile_s=neff.compile_seconds,
            mfu=m.mfu_percent,
            status=Status.KEEP if beats else Status.DISCARD,
            desc=desc,
        )
        return cand

    def _beats_incumbent(self, cand: Candidate, is_invention: bool = False) -> bool:
        if self.incumbent is None:
            return True
        return self.guards.is_improvement(
            cand.metric, self.incumbent.metric, is_invention=is_invention
        )

    def _update_incumbent(self, cand: Candidate) -> bool:
        if self._beats_incumbent(cand):
            self.incumbent = cand
            return True
        return False

    def _gain_pct(self, cand: Candidate) -> float:
        if self.incumbent is None or self.incumbent.metric <= 0:
            return 0.0
        return (cand.metric / self.incumbent.metric - 1.0) * 100.0

    # -- ledger writes -------------------------------------------------------

    def _record(
        self, cand: Candidate, stage: Stage, origin: Origin, layer: Layer,
        source: str, metric: float, correctness: float, compile_s: float,
        status: Status, desc: str, mfu: float = -1.0,
    ) -> None:
        from ledger import current_commit
        self.ledger.append(Row(
            commit=current_commit(self.ledger.run_dir),
            stage=stage, origin=origin, layer=layer, source=source,
            metric=metric, mfu=mfu, correctness=correctness,
            compile_s=compile_s, status=status, description=desc,
        ))

    def _record_pruned(self, config: dict[str, Any], reason: str) -> None:
        """Anti-pattern prune: a zero-cost negative result, still logged."""
        self.ledger.append(Row(
            commit="pruned", stage=Stage.CONFIG, origin=Origin.NONE,
            layer=Layer.CONFIG, source="", metric=0.0, mfu=-1.0,
            correctness=0.0, compile_s=0.0, status=Status.DISCARD,
            description=f"pruned: {reason}",
        ))
