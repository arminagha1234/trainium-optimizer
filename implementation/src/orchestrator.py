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

from backends.base import Backend, Measurements, OpSite, Profile
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


class NoBaselineError(RuntimeError):
    """Raised when Stage 0 cannot establish a baseline (the worker crashed or
    produced 0 throughput). The optimization run is void: there is no incumbent
    and no equivalence reference, so every later stage would be meaningless.

    optimize_one_model() catches this and reports the model as FAILED (never as
    a benign "ok, 0.000" row). The distinct type lets callers that legitimately
    tolerate a missing baseline (e.g. bank_hygiene's canary re-measure) classify
    it as a transient crash instead of a demotion signal.
    """


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
    # Stage-0 baseline top-1 token signature; the real equivalence reference.
    _baseline_tokens: list = field(default_factory=list)

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

        # HONESTY GATE: a baseline whose worker CRASHED (no result file) or
        # produced 0 throughput is NOT a valid incumbent — it never ran. Record
        # it as FAIL_NO_BASELINE (not KEEP with a benign 0) and abort the run, so
        # a crash surfaces as FAILED rather than "ok, 0.000". This is what makes
        # a crashed MoE baseline honest: the int64-topk crash used to slip
        # through here as metric=0/status=keep. (See ledger.Status.FAIL_NO_BASELINE.)
        if m.metric <= 0.0:
            self._record(base, Stage.BASELINE, Origin.NONE, Layer.NONE, source="",
                         metric=0.0, correctness=0.0,
                         compile_s=neff.compile_seconds,
                         status=Status.FAIL_NO_BASELINE,
                         desc="FAIL_NO_BASELINE: baseline worker produced no "
                              "throughput (crash / 0 tok/s) — run is void")
            raise NoBaselineError(
                f"FAIL_NO_BASELINE: {spec.model_id} baseline produced no "
                f"throughput (metric={m.metric}); the worker crashed or "
                f"returned 0 tok/s, so there is no incumbent to optimize.")

        # Capture the baseline's top-1 token signature — this IS the correctness
        # reference every later candidate is gated against.
        self._baseline_tokens = list(getattr(m, "top1_tokens", []) or [])
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
            # Scope priors to this run's execution stack: don't seed a native
            # beam with vllm-serve priors or vice-versa. mock matches all.
            backend=self.backend.name,
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

    def run_deep_stages(self, spec: ModelSpec) -> Candidate:
        """Stages 2-5 on top of the Stage-1 winner.

        On native PyTorch the neuronx-cc compiler already does kernel selection,
        fusion, and graph rewrites when torch.compile runs — so the real lever
        beyond config is *compiler flags* (NEURON_CC_FLAGS). Each candidate here
        recompiles the Stage-1 winner with a different flag set and is gated by
        the same real equivalence + guardrails as Stage 1.

        Stage 2 (known kernels) / 3 (borrow) / 5 (graph rewrite) use real flags.
        Stage 4 (invent) is entered but honestly records that novel-NKI
        invention needs the NKI-writer agent — matching the docs' expectation
        that early runs are borrow-dominated and invention is ~0.
        """
        if self.incumbent is None:
            return self.incumbent
        base_cfg = dict(self.incumbent.config)
        # cc-flags only affect the compiled path; ensure the deep stages compile.
        base_cfg["compile_mode"] = "compile-default"

        stage_plan = [
            (Stage.KNOWN_KERNEL, Origin.HARVESTED, Layer.KERNEL,
             [("cc:model-type-transformer", "--model-type transformer")]),
            (Stage.BORROW, Origin.BORROWED, Layer.KERNEL,
             [("cc:auto-cast-none", "--auto-cast none")]),
            (Stage.GRAPH_REWRITE, Origin.NONE, Layer.GRAPH,
             [("cc:optlevel3", "--optlevel 3"),
              ("cc:optlevel3+transformer+nocast",
               "--optlevel 3 --model-type transformer --auto-cast none")]),
        ]
        for stage, origin, layer, flagsets in stage_plan:
            for label, flags in flagsets:
                cand = Candidate(config={**base_cfg, "cc_flags": flags},
                                 provenance=label, layer=layer)
                evaluated = self._evaluate(cand, spec, stage, origin=origin,
                                           layer=layer, source="neuronx-cc")
                if evaluated is not None:
                    self._update_incumbent(evaluated)

        # Stage 3 (BORROW) — family-specific kernel candidates. For the MoE
        # family the backend offers "swap the HF MoE layer forward with the
        # vendored fused NKI megakernel" (nki-moe-megakernel). Dense LLMs offer
        # nothing here, so this is a graceful no-op for them. Each candidate is
        # evaluated through the SAME _evaluate() path — real equivalence gate
        # included — so a kernel whose output drifts past tolerance, or that is
        # not faster, is discarded rather than forced. Backends predating this
        # method simply expose no candidates (getattr default []).
        moe_candidates = getattr(self.backend, "moe_kernel_candidates", None)
        if callable(moe_candidates):
            from kernels.moe_fused import KERNEL_SOURCE
            base_artifact = self.backend.build_baseline(spec.model_id)
            for label, patch in moe_candidates(base_artifact):
                cand = Candidate(config={**base_cfg, **patch},
                                 provenance=label, layer=Layer.KERNEL)
                evaluated = self._evaluate(
                    cand, spec, Stage.BORROW, origin=Origin.BORROWED,
                    layer=Layer.KERNEL, source=KERNEL_SOURCE)
                if evaluated is not None:
                    self._update_incumbent(evaluated)

        # Stage 4 — invent: entered, but no auto-generated NKI kernel this run.
        self._record(
            Candidate(config=base_cfg, provenance="stage4-invent", layer=Layer.KERNEL),
            Stage.INVENT, Origin.NONE, Layer.KERNEL, source="",
            metric=0.0, correctness=0.0, compile_s=0.0, status=Status.DISCARD,
            desc="Stage 4 entered: no auto-invention (needs NKI-writer agent)")
        return self.incumbent

    # -- Stage 6 -------------------------------------------------------------

    # A bottleneck is "dominant, attackable" when the hottest op owns at least
    # this share of step time. Below it, the profile is flat and another pass of
    # the deep stages has nothing to bite on, so the loop should stop rather than
    # burn compiles chasing noise.
    _PROFILE_LOOP_DOMINANCE_SHARE = 0.30

    def run_profile_loop(
        self, spec: ModelSpec, max_rounds: int = 3, patience: int = 2,
    ) -> Candidate | None:
        """Stage 6: bounded, profile-guided re-entry loop.

        The pipeline above is linear — Stage 1 then the deep stages once. This
        closes the loop: re-profile the current incumbent and, while the profile
        still shows a dominant, attackable bottleneck, re-enter the deep stages
        to attack it. It reuses the SAME incumbent / tournament / equivalence /
        guardrail machinery (run_deep_stages) — it does not evaluate anything
        itself, it only decides whether another pass is worth the compiles.

        Bounding (must not loop forever). The loop stops on EITHER:
          (a) no improvement past the guardrail's noise margin for `patience`
              consecutive rounds (the incumbent stopped moving), OR
          (b) the `max_rounds` cap, OR
          (c) the profile no longer shows a dominant, attackable bottleneck.

        Each round is recorded to the ledger as a PROFILE_LOOP row (profile ->
        re-entry -> gained/didn't), so the bank can learn which profiled
        bottlenecks re-entry actually helps.
        """
        if self.incumbent is None:
            return self.incumbent

        no_improvement = 0
        for round_idx in range(1, max_rounds + 1):
            prof = self._profile_incumbent(spec)
            hot = self._dominant_op(prof)
            if hot is None:
                self._record(
                    self.incumbent, Stage.PROFILE_LOOP, Origin.NONE, Layer.NONE,
                    source="", metric=self.incumbent.metric, correctness=100.0,
                    compile_s=0.0, status=Status.DISCARD,
                    desc=(f"round {round_idx}: no dominant bottleneck "
                          f"({prof.bottleneck or 'flat'}) -> loop done"))
                break

            prev_metric = self.incumbent.metric
            # Re-enter the deep stages to attack the profiled bottleneck. All the
            # real evaluation (compile/equivalence/guardrails/keep) happens there.
            self.run_deep_stages(spec)
            gain = (
                (self.incumbent.metric / prev_metric - 1.0) * 100.0
                if prev_metric > 0 else 0.0
            )
            improved = self.guards.is_improvement(
                self.incumbent.metric, prev_metric)
            self._record(
                self.incumbent, Stage.PROFILE_LOOP, Origin.NONE, Layer.NONE,
                source="", metric=self.incumbent.metric, correctness=100.0,
                compile_s=0.0,
                status=Status.KEEP if improved else Status.DISCARD,
                desc=(f"round {round_idx}: profile={prof.bottleneck or '?'} "
                      f"hot={hot.op_name}({hot.cost_share:.0%}) -> re-entered "
                      f"deep stages -> {'+' if gain >= 0 else ''}{gain:.1f}% "
                      f"({'improved' if improved else 'no gain'})"))

            if improved:
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break

        return self.incumbent

    def _profile_incumbent(self, spec: ModelSpec) -> Profile:
        """Re-profile the current incumbent. Cheap: it builds+``compile``s the
        incumbent config and asks the backend to classify the bottleneck — it
        does NOT re-measure on device (the profile is a symptom key, not a run)."""
        artifact = self.backend.apply_config(
            self.backend.build_baseline(spec.model_id), self.incumbent.config
        )
        neff = self.backend.compile(artifact)
        return self.backend.profile(neff, spec.probe_shape)

    def _dominant_op(self, prof: Profile) -> OpSite | None:
        """The hottest op site, if it dominates enough to be worth re-attacking."""
        hottest = prof.hottest(1)
        if not hottest:
            return None
        top = hottest[0]
        return top if top.cost_share >= self._PROFILE_LOOP_DOMINANCE_SHARE else None

    # -- tournament primitives ----------------------------------------------

    def _equivalence(self, m: Measurements, spec: ModelSpec, neff) -> EquivalenceResult:
        """Real correctness gate: fraction of top-1 tokens matching the Stage-0
        baseline signature. No baseline signature (mock) -> injected checker."""
        ref = self._baseline_tokens
        cand = list(getattr(m, "top1_tokens", []) or [])
        if not ref:
            return self.equivalence(neff, spec)     # mock / no reference
        if not cand:
            return EquivalenceResult(passed=False, correctness_pct=0.0,
                                     notes="no output tokens (run failed/OOM)")
        n = min(len(ref), len(cand))
        match = sum(1 for i in range(n) if ref[i] == cand[i]) / n
        return EquivalenceResult(passed=match >= 0.75, correctness_pct=match * 100.0,
                                 notes=f"top1 match {match:.0%} vs baseline")

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
        # REVIEW GATE ("review, then execute") — reject malformed or unsafe
        # candidates before the 5-20 min compile. Legit proposer candidates
        # always PASS; this catches bad values + cc_flags injection.
        try:
            from reviewer import review_config
            verdict, reason = review_config(cand.config)
        except Exception:  # noqa: BLE001 — reviewer must never crash the loop
            verdict, reason = "PASS", "reviewer-unavailable"
        if verdict != "PASS":
            self._record(cand, stage, origin, layer, source, metric=0.0,
                         correctness=0.0, compile_s=0.0, status=Status.DISCARD,
                         desc=f"{cand.provenance} (review REJECT: {reason})")
            return None

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

        # Measure first — this also yields the top-1 token signature the real
        # equivalence gate needs.
        m: Measurements = self.backend.measure(neff, spec.probe_shape, spec.probe_batch)

        # Zero-throughput is a SILENT failure, not a benign result. A run that
        # returns 0 img/s (a diffusion backend that produced no image) or 0 tok/s
        # can still "pass" equivalence — the mock checker passes unconditionally,
        # and an empty output signature trivially matches another empty one — so
        # without this gate a 0-metric candidate could be recorded as a verified
        # (or unverified) 0 rather than the failure it is. Record it as an
        # explicit anti-pattern-style FAIL and discard, before equivalence.
        if m.metric <= 0.0:
            self._record(cand, stage, origin, layer, source, metric=0.0,
                         correctness=0.0, compile_s=neff.compile_seconds,
                         status=Status.DISCARD,
                         desc=f"{cand.provenance} (metric=0 -> backend produced no throughput)")
            return None

        # Equivalence — HARD gate. Compares this config's top-1 tokens against
        # the Stage-0 baseline signature; a config that changes the output is a
        # bug, not a win. (Falls back to the injected checker when no signature
        # is available, e.g. the mock backend.)
        eq = self._equivalence(m, spec, neff)
        if not eq.passed:
            self._record(cand, stage, origin, layer, source, metric=0.0,
                         correctness=eq.correctness_pct,
                         compile_s=neff.compile_seconds, status=Status.DISCARD,
                         desc=f"{cand.provenance} (equivalence fail: {eq.notes})")
            return None

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
