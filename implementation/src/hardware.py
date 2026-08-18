"""
Hardware topology — how much compute the instance actually has, so the search
FILLS it instead of leaving cores idle.

The motivating bug: a 27B model with 4 KV heads is capped at TP=4 by the GQA
sharding rule (num_kv_heads % tp == 0, enforced in the worker). On a
trn2.48xlarge that is 4 of 64 logical NeuronCores — ~94% of a ~$21.50/hr box
sitting idle. TP being bounded by KV heads does NOT mean the instance must be
under-used: fill the rest with data-parallel replicas (throughput) or context
parallelism (long context / latency). This module computes that fill plan.

Core counts here are the LNC=2 logical-NeuronCore counts (the schedulable
ranks a TP/CP/DP group draws from), matching the validated benchmarks in
Armin-Neuron. The runtime count is authoritative — prefer detect_num_cores()
on-device; the static table is the off-device planning default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeBudget:
    """The schedulable compute on one instance."""

    instance_type: str
    num_cores: int              # schedulable NeuronCores (ranks) at the default LNC
    hbm_gib_per_core: float
    lnc: int = 1
    note: str = ""

    @property
    def total_hbm_gib(self) -> float:
        return self.num_cores * self.hbm_gib_per_core


# Static defaults. Runtime detection (below) overrides num_cores when we're
# actually on the box. Numbers verified against the user's own Armin-Neuron
# benchmark docs where available; others are marked verify-on-device.
_INSTANCES: dict[str, ComputeBudget] = {
    "trn2.48xlarge": ComputeBudget(
        "trn2.48xlarge", 64, 24.0, lnc=2,
        note="16 chips x 8 physical NeuronCores = 128 physical; 64 LOGICAL at "
             "the default LNC=2 (24 GiB each, 1.5 TB total). LNC=1 exposes 128. "
             "Runtime count is authoritative.",
    ),
    "trn2.3xlarge": ComputeBudget(
        "trn2.3xlarge", 4, 24.0, lnc=2,
        note="1 Trn2 chip: 8 physical -> 4 logical at LNC=2; verify on-device",
    ),
    "trn1.32xlarge": ComputeBudget(
        "trn1.32xlarge", 32, 16.0, lnc=1,
        note="16 chips x 2 cores; 16 GiB/core",
    ),
    "inf2.48xlarge": ComputeBudget(
        "inf2.48xlarge", 12, 16.0, lnc=1,
        note="6 Inferentia2 x 2 cores; verify on-device",
    ),
}

DEFAULT_INSTANCE = "trn2.48xlarge"


def detect_num_cores() -> int | None:
    """Authoritative on-device rank count, or None when off-device.

    The real runtime also exposes this via `neuron-ls` and torch_neuronx; the
    env vars cover the torchrun/NEURON_RT path we actually launch under.
    """
    for var in ("NEURON_RT_NUM_CORES", "WORLD_SIZE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return None


def budget_for(
    instance_type: str = DEFAULT_INSTANCE, num_cores: int | None = None
) -> ComputeBudget:
    """Resolve a compute budget. Precedence: explicit num_cores > runtime
    detection > static table > single-core fallback."""
    base = _INSTANCES.get(instance_type)
    cores = num_cores or detect_num_cores() or (base.num_cores if base else 1)
    if base is None:
        return ComputeBudget(instance_type, cores, 24.0,
                             note="unknown instance; relied on runtime/explicit cores")
    if cores != base.num_cores:
        return ComputeBudget(base.instance_type, cores, base.hbm_gib_per_core,
                             base.lnc, note=f"{base.note} (overridden to {cores})")
    return base


@dataclass(frozen=True)
class FillPlan:
    """How a (tp, cp) parallel group is replicated to fill the instance."""

    tp: int
    cp: int
    dp: int
    cores_available: int
    kv_replication: int = 1     # >1 when tp > num_kv_heads (KV heads replicated)

    @property
    def cores_used(self) -> int:
        return self.tp * self.cp * self.dp

    @property
    def utilization(self) -> float:
        return self.cores_used / self.cores_available if self.cores_available else 0.0

    def as_config(self) -> dict:
        """Fields to merge into a candidate config so the whole pipeline (and
        the ledger) can see how the box is being used."""
        return {
            "tp_degree": self.tp,
            "cp_degree": self.cp,
            "dp_degree": self.dp,
            "kv_replication": self.kv_replication,
            "cores_used": self.cores_used,
            "cores_available": self.cores_available,
        }


def fill_plan(
    budget: ComputeBudget,
    tp: int,
    cp: int = 1,
    num_kv_heads: int | None = None,
    track: str = "throughput",
) -> FillPlan:
    """Given a base (tp, cp) group, decide replication to use the whole box.

    throughput -> maximize data-parallel replicas: dp = cores // (tp*cp).
                  Each replica is an independent model on its own cores, so
                  aggregate tok/s scales ~linearly and per-core HBM is
                  unchanged (dp uses *other* cores, not more memory per core).
    latency    -> dp = 1. Replicas do not cut single-request latency; the box
                  is filled via larger tp/cp instead (see the latency track).
    """
    tp = max(1, tp)
    cp = max(1, cp)
    per_replica = min(tp * cp, budget.num_cores)   # clamp oversized groups
    dp = max(1, budget.num_cores // per_replica) if track == "throughput" else 1
    kv_rep = 1
    if num_kv_heads and tp > num_kv_heads:
        # TP beyond the KV-head count requires replicating KV heads across
        # ranks. This is a *testable* option, not a hard ceiling — the worker
        # must opt in (see the backend handoff note).
        kv_rep = math.ceil(tp / num_kv_heads)
    return FillPlan(tp=tp, cp=cp, dp=dp,
                    cores_available=budget.num_cores, kv_replication=kv_rep)
