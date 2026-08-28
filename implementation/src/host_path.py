# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""host_path.py — the HOST-PATH optimizer: the axis kernel work ignores.

The sharpest on-device finding of the whole project: at batch-size 1 the NeuronCore
is >99% IDLE — the real limiter is HOST dispatch (Python/XLA enqueue, mark_step
barriers, graph recompiles), not device compute. The framework optimizes DEVICE
kernels (roofline / %SOL / NKI authoring), but for the common single-stream serving
case the device is already waiting on the host. Squeezing a kernel that runs while
the device sits 99% idle moves nothing end-to-end.

This module is the complementary axis to ``roofline.py``. Roofline answers "is this
DEVICE op near speed-of-light?"; host_path answers "is the device even the
bottleneck, or is the host?" — and if it's the host, WHICH host lever to pull
(graph reuse, async scheduling, larger batch, fewer mark_steps). Critically it acts
as a ROUTER: on a device-bound profile it defers to the kernel optimizer (authoring
is worth it); on a host-bound profile it says so plainly, so the framework spends
effort where the latency actually is.

Pure-python and unit-testable — it reasons over a ``HostProfile`` (device-busy
fraction + per-step host/device times + graph shape), which the serving backend
supplies. ``from_measurement`` tolerantly builds one from a metrics dict (the
schema drifts), so a missing field degrades to a conservative verdict rather than
a crash. Honest: it NEVER claims a speedup — it recommends levers and names the
one axis (host vs device) worth optimizing, with the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Below this device-busy fraction the device is idle enough that the host is the
# limiter — optimizing kernels cannot help end-to-end (the on-device bs=1 regime).
_HOST_BOUND_BELOW = 0.50
# A graph re-dispatched with more mark_steps than roughly one-per-few-ops is
# barrier-heavy: each mark_step is a host<->device round-trip that serializes.
_MARKSTEP_PER_OP_HEAVY = 0.20


@dataclass(frozen=True)
class HostProfile:
    """One serving/run measurement, host-vs-device. All times are per decode step
    (ms); ``device_busy_frac`` is the fraction of wall-time the NeuronCore was
    actually executing (0..1). Fields default to a neutral value so a partial
    measurement still analyzes."""
    device_busy_frac: float = -1.0    # -1 => unknown (derive from times if possible)
    device_ms: float = 0.0            # device-execution time per step
    dispatch_ms: float = 0.0          # host dispatch/enqueue time per step
    n_ops: int = 0                    # ops in the (per-step) graph
    n_mark_steps: int = 0             # mark_step / graph-break barriers per step
    batch_size: int = 1
    recompiles: int = 0               # graph recompiles observed in the window

    @property
    def busy(self) -> float:
        """Device-busy fraction, derived from device/dispatch times when not given
        directly. Unknown (no signal at all) -> -1.0."""
        if self.device_busy_frac >= 0.0:
            return self.device_busy_frac
        total = self.device_ms + self.dispatch_ms
        if total > 0.0:
            return self.device_ms / total
        return -1.0


@dataclass(frozen=True)
class Recommendation:
    """One host-side lever, with why it applies and its qualitative effect."""
    lever: str
    rationale: str
    effect: str = ""


@dataclass(frozen=True)
class HostVerdict:
    """The host-vs-device read: which axis is the bottleneck + ranked host levers."""
    axis: str                          # "host" | "device" | "unknown"
    device_idle_frac: float            # 1 - busy (0 when unknown)
    recommendations: list = field(default_factory=list)
    reason: str = ""

    @property
    def host_bound(self) -> bool:
        return self.axis == "host"


# ---------------------------------------------------------------------------
# the ranked host levers (each keyed to a condition in ``analyze``)
# ---------------------------------------------------------------------------
def _levers(p: HostProfile) -> list[Recommendation]:
    """Host-side levers that APPLY to this profile, highest-leverage first."""
    recs: list[Recommendation] = []
    # 1. Recompiles are the most expensive host stall — a recompile is seconds, not
    #    microseconds. Kill them first (fixed shapes / bucketing / graph cache).
    if p.recompiles > 0:
        recs.append(Recommendation(
            "eliminate graph recompiles",
            f"{p.recompiles} recompile(s) in the window — each is a full compiler "
            "invocation on the host serialize path (dynamic shapes / uncached graph)",
            "use fixed/bucketed shapes and a warmed graph cache so steady-state "
            "steps hit a compiled graph"))
    # 2. Barrier-heavy: too many mark_steps per op serializes host<->device.
    if p.n_ops > 0 and p.n_mark_steps / max(1, p.n_ops) >= _MARKSTEP_PER_OP_HEAVY:
        recs.append(Recommendation(
            "reduce mark_step / graph-break frequency",
            f"{p.n_mark_steps} mark_step(s) over {p.n_ops} ops — each barrier is a "
            "host<->device round-trip that stalls the pipeline",
            "coalesce work between barriers so one mark_step covers more ops"))
    # 3. Dispatch dominates device: overlap host enqueue with device execution.
    if p.dispatch_ms > 0.0 and p.dispatch_ms >= p.device_ms:
        recs.append(Recommendation(
            "async scheduling (overlap dispatch with execution)",
            f"host dispatch {p.dispatch_ms:.2f} ms >= device {p.device_ms:.2f} ms — "
            "the device waits on the host enqueue",
            "enable async execution / a dispatch-ahead queue so step n+1 enqueues "
            "while step n runs on-device"))
    # 4. bs=1 with an idle device: fill it with batching (throughput lever).
    if p.batch_size <= 1 and p.busy >= 0.0 and p.busy < _HOST_BOUND_BELOW:
        recs.append(Recommendation(
            "increase batch size / continuous batching",
            f"batch_size={p.batch_size} and device only {p.busy*100:.0f}% busy — "
            "the device has idle capacity a larger batch would fill",
            "batch requests (continuous/inflight batching) to amortize the fixed "
            "per-step host cost across more tokens"))
    return recs


def analyze(profile: HostProfile) -> HostVerdict:
    """Classify the bottleneck axis and recommend host levers.

    device-busy unknown -> ``axis="unknown"`` (no signal; do not guess).
    device-busy < ``_HOST_BOUND_BELOW`` (or dispatch >= device) -> ``axis="host"``
    with ranked levers — kernel authoring cannot help end-to-end here.
    Otherwise -> ``axis="device"``: the device IS the bottleneck, defer to the
    kernel optimizer (roofline / NKI authoring). Never raises."""
    busy = profile.busy
    if busy < 0.0 and profile.dispatch_ms <= 0.0 and profile.device_ms <= 0.0:
        return HostVerdict("unknown", 0.0, [],
                           "no host/device timing available — cannot route")
    idle = max(0.0, 1.0 - busy) if busy >= 0.0 else 0.0
    host_bound = (0.0 <= busy < _HOST_BOUND_BELOW) or (
        profile.dispatch_ms > 0.0 and profile.dispatch_ms >= profile.device_ms
        and profile.device_ms >= 0.0 and profile.dispatch_ms > 0.0)
    if host_bound:
        recs = _levers(profile)
        return HostVerdict(
            "host", idle, recs,
            f"HOST-BOUND: device {busy*100:.0f}% busy ({idle*100:.0f}% idle) — "
            f"{len(recs)} host lever(s) beat kernel work here; the device is "
            f"already waiting on the host")
    return HostVerdict(
        "device", idle, [],
        f"DEVICE-BOUND: device {busy*100:.0f}% busy — the NeuronCore is the "
        f"limiter; defer to the kernel optimizer (roofline / NKI authoring)")


def optimize_axis(profile: HostProfile) -> str:
    """The single axis worth optimizing for this profile: "host" | "device" |
    "unknown". The router the framework consults before spending authoring
    effort — no point authoring a kernel for a host-bound stream."""
    return analyze(profile).axis


# ---------------------------------------------------------------------------
# tolerant construction from a serving metrics dict
# ---------------------------------------------------------------------------
def from_measurement(m: Any) -> HostProfile:
    """Build a ``HostProfile`` from a metrics mapping (tolerant of the several
    spellings a serving backend emits). Missing fields default neutrally so a
    partial measurement still analyzes; a non-mapping -> an all-default profile
    (which ``analyze`` routes to "unknown"). Never raises."""
    if not isinstance(m, dict):
        return HostProfile()

    def _num(*keys, default=0.0):
        for k in keys:
            if k in m:
                try:
                    return float(m[k])
                except (TypeError, ValueError):
                    continue
        return default

    def _int(*keys, default=0):
        return int(_num(*keys, default=default))

    busy = _num("device_busy_frac", "device_busy", "neuroncore_busy",
                "device_utilization", default=-1.0)
    if busy > 1.0:                       # a percentage
        busy = busy / 100.0
    return HostProfile(
        device_busy_frac=busy,
        device_ms=_num("device_ms", "device_time_ms", "on_device_ms"),
        dispatch_ms=_num("dispatch_ms", "host_ms", "host_dispatch_ms",
                         "enqueue_ms"),
        n_ops=_int("n_ops", "num_ops", "graph_ops"),
        n_mark_steps=_int("n_mark_steps", "mark_steps", "num_mark_steps"),
        batch_size=_int("batch_size", "batch", "bs", default=1),
        recompiles=_int("recompiles", "num_recompiles", "graph_recompiles"))
