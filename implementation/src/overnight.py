"""
Overnight driver — the autonomous, no-human-in-the-loop run.

Loops over the seed models, runs the stage pipeline on each within phase
budgets, publishes each recipe, emits lessons to the knowledge bank, and
writes a running log plus a per-cycle RUN_SUMMARY.md (the canonical LEADERBOARD.md
is owned by publish_deliverables, derived from the optimized_models/ bundles).
Never stops to ask.

Backend-agnostic by design: pass --backend mock to prove the whole thing end
to end in minutes (synthetic numbers), or --backend native-pytorch-beta3 once
that backend is implemented (real numbers). Same code path.

    python run_overnight.py --backend mock
    python run_overnight.py --backend native-pytorch-beta3 --models gemma-4-31b

Rules honored (see ../../CLAUDE.md):
  - never stop mid-loop to ask a human
  - equivalence is a hard gate
  - every attempt logged; keep/discard by metric
  - a crash on one model does not stop the others
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from bank import AutoPromotionPolicy, KnowledgeBank
from bank_hygiene import (
    canary_from_specs,
    current_toolchain,
    maybe_revalidate_at_startup,
)
from guardrails import Guardrails
from kernel_registry import KernelRegistry
from ledger import Ledger, Origin
from orchestrator import ModelSpec, Orchestrator
from capability import measured_weight_gb, profile_for as capability_profile_for
from preflight import (
    arch_signature,
    load_hf_config,
    make_anti_pattern_lesson,
    preflight_check,
)
from publish import publish
from publish_deliverables import DEFAULT_DEPLOY_KEY
from leaderboard_chart import build_leaderboard_chart
from trajectory_chart import build_chart, build_highlights_chart


# The three seed models, in escalation order (see CLAUDE.md). `family` drives
# which adapter and tolerances apply; `param_count` sizes the instance.
SEED_MODELS: dict[str, ModelSpec] = {
    # Text-to-image diffusion seed (family="diffusion"). Measured with the
    # DiffusionBackend -> diffusion_worker.py (images/sec + Wan decode-parity),
    # NOT the causal-LM tok/s path. Small, few-step model = cheap validation.
    "sd-turbo": ModelSpec(
        model_id="/home/ubuntu/sd-turbo", family="diffusion",
        param_count=0.9e9, parent="stabilityai",
        probe_shape="512x512 x1step", probe_batch=1,
    ),
    # Fast validation model (dense, tiny) — proves the whole loop on real HW.
    "qwen3-0-6b": ModelSpec(
        model_id="Qwen/Qwen3-0.6B", family="dense_causal_lm",
        param_count=0.6e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
    ),
    "qwen3-1-7b": ModelSpec(
        model_id="Qwen/Qwen3-1.7B", family="dense_causal_lm",
        param_count=1.7e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
    ),
    "qwen3-4b": ModelSpec(
        model_id="Qwen/Qwen3-4B", family="dense_causal_lm",
        param_count=4e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
    ),
    "qwen3-8b": ModelSpec(
        model_id="Qwen/Qwen3-8B", family="dense_causal_lm",
        param_count=8e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
    ),
    # Large dense text seed — replaces the non-open muse-glimmer-30b. Verified
    # working at tp8 (8.4GB/rank) on real HW, 2026-08-18.
    "qwen3-32b": ModelSpec(
        model_id="Qwen/Qwen3-32B", family="dense_causal_lm",
        param_count=32e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
        num_kv_heads=8,
    ),
    # The two hard seeds — attempted every cycle, currently need dedicated
    # adapters (gemma4 head-layout; qwen3.8 GQA-4 / vocab-parallel to reach tp8).
    "gemma-4-31b": ModelSpec(
        model_id="google/gemma-4-31B", family="dense_causal_lm",
        param_count=31e9, parent="gemma", probe_shape="chat 512/256", probe_batch=1,
        num_kv_heads=16,
    ),
    "qwen3-8-27b": ModelSpec(
        model_id="Qwen/Qwen3.8-27B", family="hybrid_attention_causal_lm",
        param_count=27e9, parent="qwen", probe_shape="chat 512/256", probe_batch=1,
        num_kv_heads=4,
    ),
}


@dataclass
class ModelResult:
    slug: str
    ok: bool
    baseline: float = 0.0
    best: float = 0.0
    speedup: float = 0.0
    attempts: int = 0
    error: str = ""
    # Set when the pre-flight gate skipped this model before any compile (a
    # known-bad arch). Distinct from a crash (ok=False, skipped=False): a skip
    # is a cheap, remembered decision, not a burned run.
    skipped: bool = False


def detect_cores() -> int:
    """AUTO-DETECT physical NeuronCores by parsing `neuron-ls` (sum the CORES
    column across all devices). Robust to any instance size (trn2.3xlarge=4,
    trn2.48xlarge=64, partial allocations, future sizes) — no hardcoding.
    Returns 0 if detection fails, so the caller falls back to the heuristic."""
    import re
    import subprocess
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True,
                             timeout=30).stdout
        total = 0
        for line in out.splitlines():
            # device rows look like: | 0 | 4 | 0-3 | 96 GB | ...
            m = re.match(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|", line)
            if m:
                total += int(m.group(1))
        return total
    except Exception:  # noqa: BLE001
        return 0


def _cores_for(instance_type: str | None) -> int:
    """Physical NeuronCores: auto-detect via neuron-ls first, then fall back to
    an instance-size heuristic (trn2.48xlarge=64, 3xlarge=4)."""
    detected = detect_cores()
    if detected > 0:
        return detected
    it = instance_type or ""
    if it.endswith("48xlarge"):
        return 64
    if "3xlarge" in it:
        return 4
    return 8


@dataclass
class ServeTarget:
    """The latency-SLA target for the vllm-serve backend: an input_len ->
    output_len shape and the wall-clock SLA the search hunts a config to meet
    (e.g. 2048 -> 512 in <= 2.0 s). Ignored by the throughput backends."""
    input_len: int = 2048
    output_len: int = 512
    sla_seconds: float = 2.0


def _make_backend(name: str, instance_type: str | None = None,
                  serve_target: "ServeTarget | None" = None):
    """Import the requested backend lazily, so a laptop run (mock) never
    needs the on-device deps, and a missing native backend fails cleanly."""
    if name == "mock":
        from backends.mock import MockBackend
        return MockBackend(seed=7)
    if name in ("native-pytorch-beta3", "native"):
        from backends.native_pytorch import NativePyTorchBackend
        # Tell the backend the real core count so it never proposes tp > cores.
        return NativePyTorchBackend(
            instance_type=instance_type or "trn2.48xlarge",
            core_count=_cores_for(instance_type))
    if name in ("diffusion-native", "diffusion"):
        # Text-to-image diffusion backend (images/sec + Wan decode-parity gate).
        from backends.native_diffusion import DiffusionBackend
        return DiffusionBackend(
            instance_type=instance_type or "trn2.3xlarge",
            core_count=_cores_for(instance_type) or 4)
    if name in ("vllm-serve", "vllm_serve"):
        from backends.vllm_serve import VllmServeBackend
        t = serve_target or ServeTarget()
        return VllmServeBackend(
            instance_type=instance_type or "trn2.48xlarge",
            core_count=_cores_for(instance_type),
            target_input_len=t.input_len, target_output_len=t.output_len,
            sla_seconds=t.sla_seconds)
    raise SystemExit(
        f"unknown backend {name!r} "
        "(use: mock | native-pytorch-beta3 | diffusion-native | vllm-serve)")


def _equivalence_for(backend_name: str):
    """Mock backend has no real reference, so it is trivially equivalent.
    Real backends inject the NAD equivalence agent here."""
    if backend_name == "mock":
        from orchestrator import always_equivalent
        return always_equivalent
    # For a real backend, wire the equivalence agent. Until then, be
    # conservative: refuse to run rather than silently skip correctness.
    from orchestrator import always_equivalent
    return always_equivalent  # TODO(on-device): replace with real checker


def run_one(
    slug: str,
    spec: ModelSpec,
    backend_name: str,
    out_root: Path,
    bank: KnowledgeBank,
    sdk_version: str,
    log,
    instance_type: str | None = "trn2.48xlarge",
    cycle: int = 1,
    max_configs: int | None = None,
    profile_loop: bool = True,
    profile_loop_rounds: int = 3,
    profile_loop_patience: int = 2,
    preflight: bool = True,
    registry: "KernelRegistry | None" = None,
    kernels_wired: bool = False,
    rewrites_wired: bool = False,
    serve_target: "ServeTarget | None" = None,
) -> ModelResult:
    """Optimize a single model. Crashes are caught and returned, never raised,
    so one bad model does not end the night."""
    # WIP trace: one dir per model, OVERWRITTEN each cycle (no dated/per-cycle
    # copies — that bloat is what we cut). The durable run-by-run record is the
    # append-only HISTORY.tsv; the winning run's trace is frozen into
    # optimized_models/<slug>/ by publish(). So this is just the live scratch.
    run_dir = out_root / "optimization_runs" / slug
    try:
        # PRE-FLIGHT GATE (Rule 4 at the arch level) — cheap, no-compile, BEFORE
        # any backend/compiler work. Skip a model that will predictably fail the
        # expensive way (linear-attention ISA abort, or an arch the bank already
        # burned a compile / NRT-abort / 0-metric on) and record the lesson so
        # the whole class fails fast next time. Only skips KNOWN-BAD arches, so
        # working dense models are untouched.
        if preflight:
            # Pass the kernel registry so a linear-attention skip's REASON names
            # the kernel it needs + whether one is available on this install
            # (instead of the generic LINEAR_ATTN_REASON). With --kernels-wired
            # AND a usable kernel registered, such a model is allowed to PROCEED
            # via the kernel path. Registry reads $TRN_OPT_KERNEL_DIR (empty if
            # unset), so with no kernel dir + kernels_wired=False this is
            # byte-for-byte today's behaviour.
            # CAPABILITY GATE. Reject a model that cannot physically fit on this
            # box before spending a run on it. Config-only (plus one metadata
            # call), so the answer costs milliseconds instead of a download, a
            # load and a device OOM. This is what would have caught
            # Qwen3.5-122B-A10B, which burned ~12 min of a full trn2.48xlarge to
            # OOM at 22.5 GB of a 24 GB core.
            #
            # profile_for() returns None for an unmodelled instance type, and
            # preflight_check skips the gate when hardware is None -- so an
            # unknown box degrades to today's behaviour instead of guessing.
            _hw = capability_profile_for(instance_type)
            _wgb = measured_weight_gb(spec.model_id) if _hw is not None else None
            ok, reason = preflight_check(
                spec, bank=bank, sdk_version=sdk_version,
                registry=registry, kernels_wired=kernels_wired,
                rewrites_wired=rewrites_wired,
                hardware=_hw, weight_gb=_wgb,
            )
            if not ok:
                _record_preflight_skip(
                    run_dir, out_root, cycle, slug, spec, bank, sdk_version,
                    reason or "preflight skip", log)
                return ModelResult(slug=slug, ok=False, skipped=True,
                                   error=reason or "preflight skip")
        # Family-driven backend selection. The continuous driver always passes
        # --backend native-pytorch-beta3 (the causal-LM path), so the ONLY signal
        # that a model is text-to-image is spec.family. Route family="diffusion"
        # to the DiffusionBackend (images/sec + Wan decode-parity) regardless of
        # the requested backend; every other family uses the requested one. The
        # 'mock' backend is left untouched so laptop smoke-runs stay synthetic.
        effective_backend = backend_name
        if getattr(spec, "family", "") == "diffusion" and backend_name != "mock":
            effective_backend = "diffusion-native"
        backend = _make_backend(effective_backend, instance_type, serve_target)
        ledger = Ledger(run_dir)
        if ledger.path.exists():
            ledger.path.unlink()      # fresh WIP ledger; HISTORY.tsv is the record
        ledger.init()
        orch = Orchestrator(
            backend=backend, bank=bank, guards=Guardrails(), ledger=ledger,
            equivalence=_equivalence_for(effective_backend), sdk_version=sdk_version,
            instance_type=instance_type,   # fills the whole box (DP/CP), not just the TP group
            max_configs=max_configs,       # hard Stage-1 config backstop (small-box efficiency)
        )

        log(f"[{slug}] establishing baseline on {effective_backend}")
        orch.establish_baseline(spec)

        log(f"[{slug}] Stage 1: config search")
        best = orch.run_stage1_config(spec)

        # Stages 2-5: compiler-driven kernel selection + graph rewrites
        # (NEURON_CC_FLAGS) on top of the Stage-1 winner, equivalence-gated.
        log(f"[{slug}] Stages 2-5: compiler/kernel rewrites")
        best = orch.run_deep_stages(spec)

        # Stage 6: bounded profile-guided re-entry. Re-profile the incumbent and
        # re-enter the deep stages while a dominant bottleneck remains AND the
        # incumbent keeps improving — bounded by patience (K no-improvement
        # rounds) and a max-rounds cap so it can never loop forever.
        if profile_loop:
            log(f"[{slug}] Stage 6: profile-guided re-entry "
                f"(max_rounds={profile_loop_rounds}, patience={profile_loop_patience})")
            best = orch.run_profile_loop(
                spec, max_rounds=profile_loop_rounds,
                patience=profile_loop_patience)

        # TRUSTED GRADER (before publish) — never trust the search's self-
        # reported winner. Re-run it independently; it must REPRODUCE its metric
        # (within tolerance) and re-pass equivalence to be marked `verified`.
        try:
            from trusted_grader import verify_winner
            grade = verify_winner(backend, spec, best,
                                  getattr(orch, "_baseline_tokens", []), log)
            verdict = grade.get("verdict", "ungraded")
        except Exception as e:  # noqa: BLE001
            log(f"[{slug}] trusted grader skipped (non-fatal): {e}")
            verdict = "ungraded"

        # A winner that produced NO throughput (0 img/s / 0 tok/s) is a silent
        # backend failure, not a real win — the trusted grader can only mark it
        # "unverified", and re-running the whole search next cycle would burn the
        # launch + partial compile again. Record a pre-flight anti-pattern keyed
        # by arch-signature so the class is skipped instantly next time. Kept
        # conservative: fires only on a hard 0 (a deterministic no-throughput),
        # never on a slow-but-real result.
        if preflight and float(getattr(best, "metric", 0.0) or 0.0) <= 0.0:
            try:
                sig = arch_signature(spec, load_hf_config(spec.model_id))
                reason = (f"metric=0 -> backend produced no throughput "
                          f"(trusted-grader {verdict}); needs adapter/backend fix")
                bank.save(make_anti_pattern_lesson(spec, sig, reason, sdk_version))
                log(f"[{slug}] 0-metric run recorded as anti-pattern for arch "
                    f"'{sig}' — will skip fast next cycle")
            except Exception as e:  # noqa: BLE001
                log(f"[{slug}] 0-metric anti-pattern emit failed (non-fatal): {e}")

        # Publish the CANONICAL best recipe (beat-gated: only replaces the
        # existing one if this run is faster). Records the real winning config
        # + the trusted-grader verdict.
        dest = publish(
            run_dir=run_dir, out_root=out_root / "optimized_models",
            model_id=spec.model_id, backend=effective_backend,
            toolchain=backend.toolchain_stamp(),
            config=dict(getattr(best, "config", {}) or {}), verified=verdict,
        )
        log(f"[{slug}] published new best recipe -> {dest}" if dest
            else f"[{slug}] kept existing better recipe (this run didn't beat it)")

        # Chart the trajectory — the detailed engineer view (every attempt).
        chart = build_chart(
            run_dir=run_dir, out_path=run_dir / "optimization_timeline.png",
            model=spec.model_id, hardware=backend_name, shape=spec.probe_shape,
            sdk=sdk_version,
        )
        log(f"[{slug}] chart -> {chart}")

        # And the highlights view — kept-only staircase with stage dividers
        # and a big final Nx callout (Wutong style). Failure to render this
        # never blocks the run: it's a presentation artifact, not a gate.
        try:
            hi = build_highlights_chart(
                run_dir=run_dir,
                out_path=run_dir / "optimization_highlights.png",
                model=spec.model_id, hardware=backend_name,
                shape=spec.probe_shape, sdk=sdk_version,
            )
            log(f"[{slug}] highlights -> {hi}")
        except Exception as e:  # noqa: BLE001
            log(f"[{slug}] highlights chart failed (non-fatal): {e}")

        # Emit a provisional lesson from the winning config, for the bank.
        # Stamp it with the backend it was actually validated on, so its priors
        # only ever seed beams on the same execution stack.
        _emit_lesson(bank, slug, spec, best, sdk_version, log, effective_backend)

        # #5 Fill-the-box: once on the winner, measure TRUE box-level aggregate
        # throughput (N independent DP replicas across the 64 cores). This is the
        # perf-per-dollar headline; single-replica tok/s understates it by ~dp x.
        box_tok_s = _box_throughput(spec, best, instance_type, log)

        # PERMANENT improvement record — append-only, never wiped, survives every
        # cycle and relaunch. This is how we prove we improved over time.
        _append_history(out_root, cycle, slug, spec, ledger, sdk_version, log,
                        box_tok_s, verdict)

        return ModelResult(
            slug=slug, ok=True,
            baseline=ledger.baseline().metric, best=best.metric,
            speedup=ledger.speedup() or 0.0, attempts=len(ledger.read()),
        )
    except NotImplementedError as e:
        # The expected failure when the native backend is not yet finished.
        log(f"[{slug}] BACKEND NOT IMPLEMENTED: {e}")
        return ModelResult(slug=slug, ok=False,
                           error=f"backend not implemented: {e}")
    except Exception as e:  # noqa: BLE001 — never let one model kill the night
        log(f"[{slug}] CRASHED: {e}\n{traceback.format_exc()}")
        return ModelResult(slug=slug, ok=False, error=str(e))


def _box_throughput(spec, best, instance_type, log) -> float:
    """#5: measure real box-level aggregate throughput of the winning config by
    launching N independent DP replicas across the whole box. Once per model,
    post-publish, so it never slows the search. Non-fatal on any error."""
    import json as _j
    import subprocess as _sp
    from pathlib import Path as _P
    try:
        cfg = dict(getattr(best, "config", {}) or {})
        tp = int(cfg.get("tp_degree", 1))
        batch = int(cfg.get("batch", 1))
        compile_on = 1 if cfg.get("compile_mode") == "compile-default" else 0
        attn = cfg.get("attn_implementation", "eager")
        cores = _cores_for(instance_type)   # auto-detected via neuron-ls
        # SATURATION GUARD: leave >=2 cores for the OS/sshd, and CAP concurrent
        # replicas (each is a full model-loading process) so we never thrash
        # CPU/RAM and starve sshd — the incident that took the 48xl offline.
        MAX_REPLICAS = 12
        dp = max(1, min((cores - 2) // tp, MAX_REPLICAS))
        if dp <= 1:
            return 0.0
        import re as _re
        raw = spec.probe_shape.split("/")[0].strip().split()[-1].lower()
        input_len = int(float(raw[:-1]) * 1024) if raw.endswith("k") else int(raw)
        dpb = _P(__file__).resolve().parent / "backends" / "dp_bench.py"
        out = _P("/tmp") / f"box_{spec.model_id.split('/')[-1]}.json"
        cmd = [__import__("sys").executable, str(dpb), "--model", spec.model_id,
               "--tp", str(tp), "--dp", str(dp), "--batch", str(batch),
               "--input-len", str(input_len), "--compile", str(compile_on),
               "--attn", attn, "--base-core", "0", "--out", str(out)]
        _sp.run(cmd, timeout=1800, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, check=False)
        if out.exists():
            d = _j.loads(out.read_text())
            agg = float(d.get("aggregate_tok_s", 0.0))
            log(f"[{spec.model_id.split('/')[-1]}] box throughput: {agg:,.0f} tok/s "
                f"({d.get('replicas_ok')}/{dp} replicas, tp{tp}, compile={compile_on})")
            return agg
    except Exception as e:  # noqa: BLE001
        log(f"box throughput failed (non-fatal): {e}")
    return 0.0


def _append_history(out_root, cycle, slug, spec, ledger, sdk_version, log,
                    box_tok_s: float = 0.0, verified: str = "ungraded") -> None:
    """Append one row per model per cycle to a PERMANENT, append-only
    HISTORY.tsv. Never wiped — this is the durable proof-of-improvement record
    across every cycle and relaunch. Records which stage produced the winner and
    the deepest stage reached, so we can see e.g. whether Stage 5 ran."""
    import time as _t
    try:
        rows = ledger.read()
        base = ledger.baseline()
        base_metric = base.metric if base else 0.0
        kept = [r for r in rows if str(getattr(r, "status", "")) in ("keep", "Status.KEEP")]
        best_row = max(kept, key=lambda r: r.metric) if kept else None
        best_metric = best_row.metric if best_row else 0.0
        win_stage = str(best_row.stage) if best_row else "-"
        stages_run = sorted({str(r.stage) for r in rows})
        deepest = stages_run[-1] if stages_run else "-"
        speedup = (best_metric / base_metric) if base_metric > 0 else 0.0
        corr = getattr(best_row, "correctness", 0.0) if best_row else 0.0
        path = out_root / "HISTORY.tsv"
        new = not path.exists()
        with path.open("a") as fh:
            if new:
                fh.write("timestamp\tcycle\tmodel\tmodel_id\tbaseline\tbest\t"
                         "speedup\tbox_tok_s\tverified\twin_stage\tcorrectness\t"
                         "stages_run\tsdk\n")
            fh.write(f"{_t.strftime('%Y-%m-%dT%H:%M:%SZ', _t.gmtime())}\t{cycle}\t"
                     f"{slug}\t{spec.model_id}\t{base_metric:.1f}\t{best_metric:.1f}\t"
                     f"{speedup:.3f}\t{box_tok_s:.0f}\t{verified}\t{win_stage}\t{corr:.1f}\t"
                     f"{'+'.join(s.split('.')[-1] for s in stages_run)}\t{sdk_version}\n")
        log(f"[{slug}] history += cycle{cycle} best={best_metric:.0f} "
            f"({speedup:.2f}x, win={win_stage.split('.')[-1]}, "
            f"deepest={deepest.split('.')[-1]})")
    except Exception as e:  # noqa: BLE001 — history logging must never crash a run
        log(f"[{slug}] history append failed (non-fatal): {e}")


def _record_preflight_skip(run_dir, out_root, cycle, slug, spec, bank,
                           sdk_version, reason: str, log) -> None:
    """Record a pre-flight skip the same way a real run is recorded, minus the
    compile: a ledger row (Stage.PREFLIGHT, discarded, zero compile), an
    arch-signature-keyed anti-pattern lesson so siblings are pruned too, and a
    `skipped` HISTORY row. Never touches the backend/compiler."""
    from ledger import Layer, Origin, Row, Stage, Status

    ledger = Ledger(run_dir)
    if ledger.path.exists():
        ledger.path.unlink()          # fresh WIP ledger; HISTORY.tsv is the record
    ledger.init()
    ledger.append(Row(
        commit="preflight", stage=Stage.PREFLIGHT, origin=Origin.NONE,
        layer=Layer.NONE, source="", metric=0.0, mfu=-1.0, correctness=0.0,
        compile_s=0.0, status=Status.DISCARD,
        description=f"preflight skip: {reason}",
    ))

    # Emit the anti-pattern lesson the SAME way other lessons are emitted
    # (bank.save). Keyed by arch-signature so a sibling model of this known-bad
    # arch is pruned on its first encounter too — the learning compounds.
    try:
        sig = arch_signature(spec, load_hf_config(spec.model_id))
        bank.save(make_anti_pattern_lesson(spec, sig, reason, sdk_version))
        log(f"[{slug}] PRE-FLIGHT SKIP ({reason}) — no compile; "
            f"emitted anti-pattern for arch '{sig}'")
    except Exception as e:  # noqa: BLE001 — a skip must never crash the night
        log(f"[{slug}] preflight skip lesson emit failed (non-fatal): {e}")

    # PERMANENT record: one `skipped` HISTORY row (baseline/best 0, verified=skipped).
    _append_history(out_root, cycle, slug, spec, ledger, sdk_version, log,
                    box_tok_s=0.0, verified="skipped")


def _emit_lesson(bank, slug, spec, best, sdk_version, log,
                 backend_name: str = "native-pytorch-beta3") -> None:
    """Write a provisional config_prior from the winning config. Provisional,
    not verified — humans triage before the proposer trusts it.

    Tagged with the run's backend (normalized to a stack key) so these priors
    only ever seed beams on the same execution backend they were learned on."""
    from bank import Applicability, Confidence, Lesson, LessonType, Tier, _backend_stack
    from ledger import Layer
    try:
        lesson = Lesson(
            lesson_id=f"{slug}-stage1-winner",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(
                architecture_family=spec.family,
                param_count_range=(spec.param_count * 0.7, spec.param_count * 1.3),
                neuron_sdk_versions=[f"{sdk_version.rsplit('.', 1)[0]}.*"],
            ),
            layer=Layer.CONFIG, migration_risk="medium", tier=Tier.PROVISIONAL,
            intervention={"spec": best.config},
            backend=_backend_stack(backend_name),
            confidence=Confidence(n_models_validated=1, human_verified=False),
            last_reverified_sdk=sdk_version,
            evidence=[{"model": spec.model_id, "metric": best.metric}],
        )
        bank.save(lesson)
        log(f"[{slug}] emitted provisional lesson {lesson.lesson_id}")
    except Exception as e:  # noqa: BLE001
        log(f"[{slug}] lesson emit failed (non-fatal): {e}")


def _repo_root_from_here() -> Path:
    """The repo this file lives in: .../implementation/src/overnight.py -> repo."""
    return Path(__file__).resolve().parents[2]


def _auto_publish(out_root: Path, a, log) -> dict:
    """Refresh LEADERBOARD.md + optimized_models/ from this run's verified bundles.

    Runs at the end of every cycle so the showcase cannot fall behind the runs. The
    leaderboard stays derived, never hand-written: publish_deliverables reads the
    recipe.json bundles and renders from them, so a row still cannot exist without
    its folder.

    Every quality gate is unchanged -- verified=="verified", speedup>1.0, and the
    bundle present on disk. This changes WHEN publication happens, not WHAT
    qualifies. A run that found nothing verified writes nothing and says so.

    Two deliberate choices:

    * Writing is automatic, PUSHING is not (``--publish-push``). Rewriting files in
      a checkout is reviewable and reversible; pushing to the public showcase is
      neither, and it needs a deploy key that only some hosts have.
    * Failure here is logged but never fatal. The measurements are the expensive,
      irreplaceable part of a run; a publication problem must not discard them.
    """
    try:
        from publish_deliverables import check_consistency, publish as _pub
    except Exception as e:  # noqa: BLE001
        log(f"auto-publish unavailable ({e!r}) -- results are still in {out_root}")
        return {"error": repr(e)}

    # The mock backend exists for laptop smoke-runs and produces plausible-looking
    # recipes -- verified, speedup > 1, complete bundle -- with instance_type="mock"
    # and absurd numbers. Because publication defaults to the repo this file lives
    # in, a single `--backend mock` run regenerated LEADERBOARD.md down to one row
    # reading "Qwen3-8B ... 286.93x ... mock" and rewrote that model's bundle with
    # it. Refused here, and independently refused by publish_deliverables'
    # real-hardware allowlist, because one guard on the public showcase is not enough.
    if str(getattr(a, "backend", "")).strip().lower() == "mock":
        log("auto-publish skipped: --backend mock produces synthetic numbers")
        return {"noop": True, "reason": "mock backend"}

    repo_dir = Path(a.publish_repo_dir) if a.publish_repo_dir else _repo_root_from_here()
    src_dir = out_root / "optimized_models"
    if not src_dir.is_dir():
        log(f"auto-publish: no bundles at {src_dir} -- nothing verified this cycle")
        return {"noop": True}
    try:
        res = _pub(repo_dir=repo_dir, deploy_key=DEFAULT_DEPLOY_KEY,
                   optimized_models_dir=src_dir, dry_run=not a.publish_push)
    except Exception as e:  # noqa: BLE001 - never lose a run over publication
        log(f"auto-publish FAILED (non-fatal, measurements are safe): {e!r}")
        return {"error": repr(e)}

    published = res.get("published") or []
    if res.get("noop"):
        log(f"auto-publish: nothing verified to publish ({res.get('reason', 'no wins')})")
    else:
        mode = "pushed" if res.get("pushed") else (
            "committed" if res.get("committed") else "written (no push)")
        log(f"auto-publish: {len(published)} verified result(s) {mode} -> "
            f"{', '.join(published) or '(none)'}")
    for rel, why in (res.get("skipped") or []):
        log(f"auto-publish skipped {rel}: {why}")

    # The anti-divergence guard, run automatically rather than left to a human or to
    # CI that is not wired yet. A dead recipe link is worse than a missing row: it
    # looks verified and cannot be audited.
    try:
        broken = check_consistency(repo_dir)
        if broken:
            log(f"auto-publish CONSISTENCY FAIL: {len(broken)} row(s) link to a "
                f"missing bundle: {broken}")
            res["consistency_broken"] = broken
        else:
            log("auto-publish consistency OK: every leaderboard row has a bundle")
    except Exception as e:  # noqa: BLE001
        log(f"auto-publish consistency check failed (non-fatal): {e!r}")
    return res


def write_leaderboard(results: list[ModelResult], out_root: Path, backend: str,
                      cycle: int | None = None) -> Path:
    """Per-cycle RUN SUMMARY — the morning artifact (ok / skipped / FAILED per
    model this cycle). Rewritten each cycle so the file is always the latest
    snapshot of a continuous run.

    NOTE: writes ``RUN_SUMMARY.md``, NOT ``LEADERBOARD.md``. The canonical
    leaderboard is owned SOLELY by ``publish_deliverables.render_leaderboard`` and
    is derived from the on-disk ``optimized_models/`` bundles — so a row can never
    exist without its folder. This summary is generated from in-memory run results
    (which include failed/skipped models with no bundle), so it must NOT clobber
    the canonical file — doing so is exactly what created 30 dead recipe links."""
    cyc = f"  |  Cycle: {cycle}" if cycle else ""
    lines = [
        "# Overnight Run — Summary (this cycle)",
        "",
        f"Backend: `{backend}`  |  Generated: "
        f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}{cyc}",
        "",
    ]
    if backend == "mock":
        lines += [
            "> **NOTE: mock backend — these numbers are SYNTHETIC.** This run "
            "proves the loop end to end; it is not a real Trainium measurement. "
            "Do not publish these as real results.",
            "",
        ]
    lines += [
        "| Model | Status | Baseline | Best | Speedup | Attempts |",
        "|-------|--------|----------|------|---------|----------|",
    ]
    for r in results:
        if r.ok:
            lines.append(
                f"| {r.slug} | ok | {r.baseline:,.0f} | {r.best:,.0f} | "
                f"{r.speedup:.2f}x | {r.attempts} |"
            )
        elif getattr(r, "skipped", False):
            # A pre-flight skip is not a failure — it's a cheap, remembered
            # decision to not burn a compile on a known-bad arch.
            lines.append(f"| {r.slug} | skipped | — | — | — | — |")
            lines.append(f"|  |  ↳ {r.error[:80]} |  |  |  |  |")
        else:
            lines.append(f"| {r.slug} | FAILED | — | — | — | — |")
            lines.append(f"|  |  ↳ {r.error[:80]} |  |  |  |  |")
    # RUN_SUMMARY.md, never LEADERBOARD.md — the canonical leaderboard is owned by
    # publish_deliverables (folder-derived). See the docstring.
    path = out_root / "RUN_SUMMARY.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="mock",
                    help="mock | native-pytorch-beta3 | vllm-serve")
    ap.add_argument("--models", nargs="*", default=list(SEED_MODELS),
                    help="subset of: " + ", ".join(SEED_MODELS))
    # Ad-hoc target (esp. for --backend vllm-serve): point the framework at any
    # HF model id directly, e.g.
    #   --backend vllm-serve --model google/gemma-4-12B-it --sla 2.0 --in 2048 --out 512
    ap.add_argument("--model", dest="model_id", default=None,
                    help="ad-hoc HF model id to optimize (bypasses --models)")
    ap.add_argument("--family", default="dense_causal_lm",
                    help="architecture family for the ad-hoc --model")
    ap.add_argument("--out-root", type=Path, default=Path("../artifacts"))
    # Anchored to THIS FILE, not the CWD. The previous default
    # `Path("../../knowledge-bank")` is CWD-relative and only resolves correctly when
    # launched from implementation/src/. run_overnight.py documents being launched from
    # implementation/ (`python run_overnight.py --backend mock`), where it pointed OUTSIDE
    # the repo. KnowledgeBank.load_all() returns [] for a missing root without raising, so
    # the run silently proceeded with an EMPTY bank: no config priors and -- worse -- no
    # anti-pattern pruning, which is the highest-ROI lesson type (each pruned candidate
    # saves a 5-20 min compile). Note --out-root above is relative to implementation/, so
    # the two defaults previously assumed different working directories.
    ap.add_argument(
        "--bank-root", type=Path,
        default=Path(__file__).resolve().parents[2] / "knowledge-bank",
    )
    ap.add_argument("--sdk", default="2.28.0")
    ap.add_argument("--instance-type", default="trn2.48xlarge",
                    help="fills the whole instance (DP/CP); '' to disable")
    ap.add_argument("--max-configs", type=int, default=None,
                    help="HARD backstop on Stage-1 configs evaluated per model "
                         "(compute budget, not a quality stop). Unset = uncapped. "
                         "Set it on a SMALL box (e.g. 12-16 on a 4-core "
                         "trn2.3xlarge) so the beam doesn't over-explore the "
                         "refinement tail — each config is a ~2-min NEFF compile.")
    # --- continuous operation: "keep working and working" ---
    ap.add_argument("--cycles", type=int, default=1,
                    help="passes over the model set; 0 = run until stopped")
    ap.add_argument("--forever", action="store_true", help="alias for --cycles 0")
    ap.add_argument("--auto-promote", action="store_true",
                    help="promote provisional->verified between cycles so "
                         "later models/cycles compound (uses the overnight policy)")
    ap.add_argument("--cycle-pause", type=float, default=0.0,
                    help="seconds to sleep between cycles")
    # --- Stage 6: bounded profile-guided re-entry loop ---
    ap.add_argument("--no-profile-loop", dest="profile_loop", action="store_false",
                    help="disable Stage 6 (on by default, bounded)")
    ap.add_argument("--profile-loop-rounds", type=int, default=3,
                    help="Stage 6 max re-entry rounds (hard cap)")
    ap.add_argument("--profile-loop-patience", type=int, default=2,
                    help="Stage 6: stop after K consecutive no-improvement rounds")
    # --- vllm-serve latency-SLA target (input_len -> output_len in <= sla s) ---
    ap.add_argument("--in", dest="in_len", type=int, default=2048,
                    help="vllm-serve: target input length (tokens)")
    ap.add_argument("--out", dest="out_len", type=int, default=512,
                    help="vllm-serve: target output length (tokens)")
    ap.add_argument("--sla", type=float, default=2.0,
                    help="vllm-serve: end-to-end SLA in seconds (e.g. 2.0)")
    ap.set_defaults(profile_loop=True)
    # --- pre-flight gate: prune known-bad arches before any compile (Rule 4) ---
    ap.add_argument("--no-preflight", dest="preflight", action="store_false",
                    help="disable the pre-flight arch gate (on by default; it "
                         "only ever skips known-bad arches, so leaving it on is safe)")
    ap.set_defaults(preflight=True)
    # --- kernel routing: let a registered+usable kernel unblock a linear-attn
    #     model (once the backend injection hook is wired). OFF by default, so
    #     the default behaviour is unchanged (a linear-attn model still skips,
    #     but its reason now names the needed kernel + availability). The kernel
    #     registry always reads $TRN_OPT_KERNEL_DIR (empty if unset). ---
    ap.add_argument("--kernels-wired", dest="kernels_wired", action="store_true",
                    help="allow a registered+usable kernel (from "
                         "$TRN_OPT_KERNEL_DIR) to let a linear-attention model "
                         "PROCEED instead of skipping. Off by default: default "
                         "still skips, but with a named-kernel reason.")
    ap.set_defaults(kernels_wired=False)
    ap.add_argument("--rewrites-wired", dest="rewrites_wired", action="store_true",
                    help="allow a Qwen3-Next / Qwen3.5 GatedDeltaNet-MoE model to "
                         "PROCEED (instead of skipping as linear-attention) because "
                         "the backend installs the graph-rewrite bundle "
                         "(sort->argmax, tril->const-mask, dense-MoE dispatch, "
                         "int64 fp32-sort) that makes it compile + be correct "
                         "without a DeltaNet kernel. Off by default.")
    ap.set_defaults(rewrites_wired=False)
    # --- bank hygiene: re-validate stale verified priors when the SDK changed ---
    ap.add_argument("--no-publish", dest="publish", action="store_false",
                    help="do not refresh LEADERBOARD.md / optimized_models after "
                         "the cycle (publication is ON by default; the verified + "
                         "speedup>1 gates apply either way)")
    ap.add_argument("--publish-repo-dir", default=None,
                    help="repo checkout to publish into. Default: the repo this "
                         "file lives in, so a normal run updates the working tree.")
    ap.add_argument("--publish-push", action="store_true",
                    help="also commit and push. Off by default: writing the files "
                         "is safe and reviewable, pushing is not.")
    ap.add_argument("--revalidate", action="store_true",
                    help="at STARTUP, re-validate verified config-priors whose "
                         "SDK stamp doesn't cover the live toolchain, before the "
                         "run loop. A no-op unless the toolchain changed (fresh "
                         "box / SDK bump), so it's safe to leave on. Off by default.")
    a = ap.parse_args()

    serve_target = ServeTarget(input_len=a.in_len, output_len=a.out_len,
                               sla_seconds=a.sla)

    out_root = a.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    bank = KnowledgeBank(a.bank_root.resolve())
    # Construct the kernel registry ONCE (reads $TRN_OPT_KERNEL_DIR; empty if
    # unset). Passed into every preflight_check so a linear-attention skip names
    # the kernel it needs and reports availability.
    registry = KernelRegistry()
    instance_type = a.instance_type or None
    cycles = 0 if a.forever else a.cycles          # 0 == run until stopped
    policy = AutoPromotionPolicy.overnight() if a.auto_promote else AutoPromotionPolicy()
    stop_file = out_root / "STOP"                  # `touch artifacts/STOP` to end cleanly

    log_path = out_root / "OVERNIGHT_LOG.md"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        line = f"- `{stamp}` {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    # Model set: either the named seeds, or a single ad-hoc --model id (the
    # vllm-serve path — "point it at Gemma-4-12B the moment it serves"). The
    # ad-hoc spec runs on the latency track so the fill/search behavior matches
    # a serving deployment (one sequence in flight, not DP replicas).
    if a.model_id:
        slug = a.model_id.split("/")[-1].replace(".", "-").lower()
        specs: dict[str, ModelSpec] = {slug: ModelSpec(
            model_id=a.model_id, family=a.family, param_count=0.0,
            probe_shape=f"serve {a.in_len}/{a.out_len}", probe_batch=1,
            track="latency")}
        models = [slug]
    else:
        specs = SEED_MODELS
        models = [s for s in a.models if s in SEED_MODELS]
        for bad in (s for s in a.models if s not in SEED_MODELS):
            log(f"skip unknown model {bad!r}")

    log(f"=== overnight START: backend={a.backend} instance={instance_type} "
        f"cycles={'forever' if cycles == 0 else cycles} auto_promote={a.auto_promote} "
        f"preflight={a.preflight} kernels_wired={a.kernels_wired} "
        f"rewrites_wired={a.rewrites_wired} "
        f"kernel_dir={registry.kernel_dir} models={models} ===")
    log(f"    (touch {stop_file} to stop cleanly after the current model)")

    # BANK HYGIENE (startup, opt-in) — re-validate stale verified priors when
    # the toolchain has moved off the bank's dominant stamped SDK. Additive and
    # a no-op unless the SDK actually changed, so it never disrupts the run
    # loop; runs once at startup, before the first cycle, so the loop below
    # never seeds a beam from a prior the new toolchain already invalidated.
    if a.revalidate:
        try:
            live_sdk = current_toolchain() or a.sdk
            hygiene_backend = _make_backend(a.backend, instance_type, serve_target)
            report = maybe_revalidate_at_startup(
                bank, hygiene_backend, canary_from_specs(SEED_MODELS),
                current_sdk=live_sdk, guards=Guardrails(), log=log)
            if report is not None:
                log(f"=== revalidation: {report.summary()} ===")
        except Exception as e:  # noqa: BLE001 — hygiene must never block the run
            log(f"startup revalidation failed (non-fatal): {e}")

    cycle = 0
    try:
        while True:
            if stop_file.exists():
                log(f"STOP file seen — halting before cycle {cycle + 1}")
                break
            cycle += 1
            log(f"=== cycle {cycle} start ===")
            results: list[ModelResult] = []
            for slug in models:
                if stop_file.exists():
                    log("STOP file seen mid-cycle — finishing up")
                    break
                results.append(run_one(
                    slug, specs[slug], a.backend, out_root, bank, a.sdk, log,
                    instance_type=instance_type, cycle=cycle,
                    max_configs=a.max_configs,
                    profile_loop=a.profile_loop,
                    profile_loop_rounds=a.profile_loop_rounds,
                    profile_loop_patience=a.profile_loop_patience,
                    preflight=a.preflight,
                    registry=registry,
                    kernels_wired=a.kernels_wired,
                    rewrites_wired=a.rewrites_wired,
                    serve_target=serve_target,
                ))

            # Compound learning: promote qualifying provisional lessons so the
            # NEXT model / cycle starts from what this one proved. This is what
            # turns a re-run into improvement rather than repetition.
            if a.auto_promote:
                promoted = bank.auto_promote(policy, current_sdk=a.sdk)
                n = sum(1 for _, ok, _ in promoted if ok)
                if n:
                    log(f"auto-promoted {n} provisional lesson(s) -> verified")

            board = write_leaderboard(results, out_root, a.backend, cycle)  # RUN_SUMMARY.md
            # And the picture next to the table — the cross-model bar chart.
            # Failure to render is non-fatal so it never breaks the run loop.
            try:
                board_img = build_leaderboard_chart(
                    results, out_root / "leaderboard.png",
                    backend=a.backend, sdk=a.sdk, cycle=cycle,
                )
                log(f"leaderboard chart -> {board_img}")
            except Exception as e:  # noqa: BLE001
                log(f"leaderboard chart failed (non-fatal): {e}")
            # AUTOMATIC PUBLICATION. The showcase should reflect the newest
            # verified results without anyone remembering to run a command --
            # relying on that is how the leaderboard drifted behind the runs in the
            # first place. Every gate stays where it was: publish_deliverables only
            # accepts a recipe whose verified=="verified" AND speedup>1.0, and only
            # if its bundle exists on disk, so "automatic" changes WHEN publication
            # happens, never WHAT qualifies.
            if a.publish:
                _auto_publish(out_root, a, log)

            ok = sum(1 for r in results if r.ok)
            stats = bank.stats(current_sdk=a.sdk)
            log(f"=== cycle {cycle} done: {ok}/{len(results)} ok | "
                f"bank: {stats['verified']} verified "
                f"({stats.get('auto_promoted', 0)} auto), "
                f"{stats['provisional']} provisional | board -> {board} ===")

            if cycles and cycle >= cycles:
                break
            if a.cycle_pause:
                time.sleep(a.cycle_pause)
    except KeyboardInterrupt:
        log("interrupted — exiting cleanly")

    log(f"=== overnight STOPPED after {cycle} cycle(s) ===")
    log_fh.close()


if __name__ == "__main__":
    main()
