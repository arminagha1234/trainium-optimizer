"""
Seed the knowledge bank with initial verified lessons.

Do not make the optimizer rediscover things the Neuron ecosystem already
knows. These lessons come from the auto_research run's real measured findings
(the three all_gather-elimination rewrites that carried it to a large (multiple-x)) plus
well-established config priors and anti-patterns.

All are backend-relevant and tagged by layer. The three op_rewrites are
COLLECTIVE-layer, so they survive an XLA -> native-PyTorch migration (they are
about the math, not the framework).

    python seed_bank.py --bank-root ../../knowledge-bank

See ../../knowledge-bank.md for the schema and ../../references-analysis.md
for where these came from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bank import Applicability, Confidence, KnowledgeBank, Lesson, LessonType, Symptom, Tier
from ledger import Layer, Origin


def seed_lessons() -> list[Lesson]:
    sdk = ["2.26.*", "2.27.*", "2.28.*"]
    dense = "dense_causal_lm"
    moe = "moe_causal_lm"
    hybrid = "hybrid_attention_causal_lm"

    return [
        # --- collective-layer op_rewrites (the auto_research wins) ----------
        Lesson(
            lesson_id="local-q-eliminate-hidden-allgather",
            type=LessonType.OP_REWRITE,
            applicability=Applicability(dense, (7e9, 200e9), neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            origin=Origin.BORROWED,
            intervention={"spec": {"attention_sharding": "local_q"}},
            reason=(
                "Standard TP all-gathers full hidden states then each rank "
                "computes QKV on a shard. Local-Q inverts it: each rank "
                "computes full QKV on its local tokens (seq/TP), then "
                "all-gathers only the small K/V. Saves the hidden all_gather "
                "and cuts QKV compute by TP x."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="collective_bound",
                signature="all_gather of hidden states dominates step time",
                observed_via="CC engine busy, PE idle in profile",
            )],
            source="internal-prior-optimization-run",
            confidence=Confidence(n_models_validated=4, architecture_diversity=2,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
            evidence=[{"note": "part of the +193% Round 3 on Tongyi-30B-A3B"}],
        ),
        Lesson(
            lesson_id="context-parallel-kv-split",
            type=LessonType.OP_REWRITE,
            applicability=Applicability(dense, (7e9, 200e9),
                                        seq_len_range=(8192, 10_000_000),
                                        neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            origin=Origin.BORROWED,
            intervention={"spec": {"context_parallel": True}},
            reason=(
                "Prior KV cache split across ranks; each rank attends to 1/TP "
                "of prior context, merged via online softmax reduction (exact). "
                "Cuts prior-attention compute by TP x. Biggest wins at long "
                "context where attention dominates."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="compute_bound",
                signature="prior-context attention scan dominates at long seq",
                observed_via="PE busy in attention, cost grows with context len",
            )],
            source="internal-prior-optimization-run + nki-library kv_parallel_segmented",
            confidence=Confidence(n_models_validated=4, architecture_diversity=2,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
        Lesson(
            lesson_id="local-moe-eliminate-input-allgather",
            type=LessonType.OP_REWRITE,
            applicability=Applicability(moe, (20e9, 300e9), neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            origin=Origin.BORROWED,
            intervention={"spec": {"moe_sharding": "local_moe"}},
            reason=(
                "Each rank keeps full MoE/MLP weights, processes only local "
                "tokens, all-reduces output. Eliminates the MoE/MLP input "
                "all_gather entirely. MoE models gained most from this (11-17x)."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="collective_bound",
                signature="MoE input all_gather dominates",
                observed_via="CC engine busy at expert dispatch",
            )],
            source="internal-prior-optimization-run",
            confidence=Confidence(n_models_validated=3, architecture_diversity=1,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),

        # --- config priors --------------------------------------------------
        Lesson(
            lesson_id="dense-30b-tp8-bf16",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(dense, (20e9, 40e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            intervention={"spec": {"tp_degree": 8, "weights_dtype": "bf16",
                                   "kv_cache_dtype": "bf16"}},
            reason="~30B dense fits well at TP=8 bf16 on trn2; good starting point.",
            confidence=Confidence(n_models_validated=3, architecture_diversity=1,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
        Lesson(
            lesson_id="dense-7b-tp2-bf16",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(dense, (3e9, 10e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            intervention={"spec": {"tp_degree": 2, "weights_dtype": "bf16"}},
            reason="Small dense models: TP=2 avoids collective overhead of higher TP.",
            confidence=Confidence(n_models_validated=2, human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
        Lesson(
            lesson_id="hybrid-attn-27b-tp-by-kvheads-then-fill-dp",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(hybrid, (20e9, 40e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            intervention={"spec": {"tp_degree": 4, "weights_dtype": "bf16",
                                   "kv_cache_dtype": "bf16"}},
            reason=(
                "Hybrid-attention ~27B (Qwen3.x-style GQA, often 4 KV heads) is "
                "capped at TP=num_kv_heads=4 for clean attention sharding. That "
                "is only 4 of 64 logical cores on trn2.48xlarge (LNC=2). Do NOT "
                "stop there: the search sweeps the whole TPxDP grid (TP=4xDP=16, "
                "TP=8xDP=8, ... TP=64xDP=1 — every box-filling partition) and "
                "MEASURES each; a good default is low-TP x many-DP for "
                "throughput, but the winner is decided by measurement. TP>kv_heads "
                "is possible via KV "
                "replication but must be measured, not assumed — it trades "
                "redundant KV work for tensor-parallel width and only sometimes "
                "wins. Leaving 12 cores idle is the failure mode to avoid."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="under_utilized",
                signature="TP group << instance cores; most of the box idle",
                observed_via="device_utilization low; MFU low despite fast per-replica",
            )],
            confidence=Confidence(n_models_validated=2, architecture_diversity=1,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),

        Lesson(
            lesson_id="latency-track-fill-tp-cp-not-dp",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(dense, (1e9, 300e9), neuron_sdk_versions=sdk),
            layer=Layer.COLLECTIVE, migration_risk="low-medium",
            intervention={"spec": {"track": "latency", "dp_degree": 1,
                                   "cp_degree": 2, "batching": "static"}},
            reason=(
                "For the 'fastest possible' (lowest per-request latency) track, "
                "fill the instance with tensor/context parallelism, NOT "
                "data-parallel replicas: DP raises aggregate tok/s but does "
                "nothing for a single request's latency. So set dp=1 and put "
                "more cores on the one request — raise tp (up to the KV-head "
                "cap, or beyond via KV replication) and cp_degree (splits the "
                "sequence, wins most at long context). Prefer smaller batch. "
                "This is the mirror image of the throughput prior, which fills "
                "with DP replicas instead."
            ),
            symptoms_addressed=[Symptom(
                bottleneck="latency_bound",
                signature="single-request TTFT/'/token latency is the objective",
                observed_via="customer SLA on p50/p99 latency, batch small",
            )],
            confidence=Confidence(n_models_validated=2, architecture_diversity=1,
                                  human_verified=True),
            last_reverified_sdk="2.28.0",
        ),

        # --- anti-patterns (prune before compile) ---------------------------
        Lesson(
            lesson_id="tp16-spill-under-30b",
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability(dense, (0, 30e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            matcher={"tp_degree": {"gte": 16}},
            reason=("At TP>=16 with under 30B params, per-core weight shards get "
                    "small enough that collective overhead dominates and the "
                    "compiler spills. ~3x slower than TP=8. NOTE: verified on "
                    "XLA only — on other backends this is measured, not "
                    "assumed, so the TPxDP sweep can confirm it on native."),
            confidence=Confidence(n_models_validated=3, human_verified=True),
            last_reverified_sdk="2.28.0",
            # Verified on vLLM-Neuron/XLA. On native PyTorch it does NOT
            # pre-prune — the TPxDP sweep measures TP>=16 to verify the prior
            # on the new backend before trusting it.
            backend_validated=["vllm-neuron-xla"],
        ),
        Lesson(
            lesson_id="placement-device-scheduler-bf16-drift",
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability("diffusion", (0, 300e9),
                                        neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            # No matcher ON PURPOSE: this must NOT pre-prune. Moving a component
            # to the device is sometimes the right call (the text-encoder was a
            # 65s -> 0.7s win), so placement is decided by measure() + the
            # equivalence gate, not assumed away before compile. The lesson is
            # the recorded WARNING that a device placement of a sequential
            # bf16 solver needs the correctness gate, and why.
            reason=(
                "Moving a bf16 reduction/solver op (e.g. a diffusion scheduler "
                "step) to the device can drift accuracy over many sequential "
                "steps — placement must be correctness-gated, not assumed. Wan "
                "2.2 (50 steps): scheduler on device measured 72.3s (NOT faster "
                "than 71.2s on CPU) AND degraded output to PSNR 34.7 dB vs the "
                "CPU scheduler's 56.2 dB. Conversely the T5 text-encoder on "
                "device was 65s -> 0.7s with no drift. So placement is "
                "per-component: keep the sequential solver on CPU, put the "
                "one-shot encoder on device — and let the equivalence gate "
                "confirm each, rather than forcing everything on-device."),
            symptoms_addressed=[Symptom(
                bottleneck="accuracy_drift",
                signature=("output quality (PSNR/token-match) degrades over many "
                           "sequential steps after a component moved to device"),
                observed_via="equivalence gate fails a fast device-placement candidate",
            )],
            confidence=Confidence(n_models_validated=1, human_verified=True),
            last_reverified_sdk="2.28.0",
            evidence=[{"model": "Wan2.2-TI2V-5B", "steps": 50,
                       "scheduler_device_s": 72.3, "scheduler_cpu_s": 71.2,
                       "scheduler_device_psnr_db": 34.7, "scheduler_cpu_psnr_db": 56.2,
                       "text_encoder_cpu_s": 65.0, "text_encoder_device_s": 0.7}],
        ),
        Lesson(
            lesson_id="fp8-activations-rmsnorm-heavy",
            type=LessonType.ANTI_PATTERN,
            applicability=Applicability(dense, (0, 200e9), neuron_sdk_versions=sdk),
            layer=Layer.CONFIG, migration_risk="medium",
            matcher={"activations_dtype": "fp8"},
            reason=("FP8 activations on RMSNorm-heavy models can accumulate error "
                    "past equivalence tolerance. Only enable with per-op rtol "
                    "tuning validated."),
            confidence=Confidence(n_models_validated=2, human_verified=True),
            last_reverified_sdk="2.28.0",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-root", type=Path, default=Path("../../knowledge-bank"))
    a = ap.parse_args()

    bank = KnowledgeBank(a.bank_root.resolve())
    lessons = seed_lessons()
    for lesson in lessons:
        # Seed lessons are the hand-authored, human-curated bootstrap — they
        # go straight to verified so the proposer trusts them from run 1.
        lesson.tier = Tier.VERIFIED
        lesson.confidence.human_verified = True
        p = bank.save(lesson)
        print(f"  seeded {lesson.tier.value}/{lesson.type.value}: {lesson.lesson_id}")
    stats = bank.stats(current_sdk="2.28.0")
    print(f"\nbank now: {stats['verified']} verified, {stats['provisional']} provisional")
    print(f"by type: {stats['by_type']}")


if __name__ == "__main__":
    main()
