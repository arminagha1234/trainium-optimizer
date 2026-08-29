# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""neuron_profile.py — PROFILE-GUIDED bottleneck diagnosis for the perf loop.

Today ``kernel_perf.classify_bottleneck`` diagnoses a slow kernel from the
ANALYTIC roofline alone: a shape-derived ``memory_bound``/``compute_bound`` label
+ arithmetic-intensity ratio. That is a *guess* — it says "this op *should* be
memory-bound", not "on THIS silicon the DMA engine was 71% busy while the PE sat
at 9%". The on-device finding that motivates this module: the real profiler
(``neuron-profile``) knows exactly which engine (PE / Act / Pool / DMA) dominated
the measured latency, and feeding THAT to the author turns perf-tuning from
guessing into surgery ("your K-tile DMA is 60% of latency → double-buffer it").

This module is the seam between a real per-engine profile and the perf loop's
existing bottleneck vocabulary. ``kernel_perf.classify_bottleneck`` already scans
``race.reason`` for the keywords ``dma`` / ``bandwidth`` / ``single`` / ``serial``
/ ``engine`` / ``spill`` BEFORE falling back to the analytic label — so the
integration is: profile the measured kernel, ``summarize`` the per-engine busy
fractions into a ``reason`` string carrying the right keyword, and attach it to
the ``RaceResult``. The analytic path stays as the fail-open fallback when no
profiler is available (off-device, or ``neuron-profile`` absent) — this module
NEVER raises and returns ``None`` when it cannot profile, so the loop degrades to
today's analytic behaviour rather than breaking.

The valuable, load-bearing part — ``summarize`` (per-engine busy → dominant
bottleneck + human reason) and ``parse_profile_json`` (tolerant extraction of
engine busy fractions from a ``neuron-profile`` summary) — is PURE and
unit-testable off-device. Only ``profile_kernel`` (which actually shells out to
``neuron-profile`` / uses the profiling API) touches the device, and it is fully
guarded.

The canonical bottleneck labels match ``kernel_perf`` exactly
(``memory_bound`` / ``single_engine`` / ``dma_blocked``) so a caller can route on
``ProfileReport.dominant`` directly, while ``ProfileReport.reason`` carries the
keyword form for the grep path already wired in ``classify_bottleneck``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Canonical bottleneck labels — kept identical to kernel_perf's constants so a
# ProfileReport.dominant can be consumed there without translation.
MEMORY_BOUND = "memory_bound"
SINGLE_ENGINE = "single_engine"
DMA_BLOCKED = "dma_blocked"

# The Trainium2 compute/movement engines a neuron-profile reports a busy fraction
# for. Keys are the normalized names this module uses; ``parse_profile_json``
# maps the many spellings a profiler emits (PE/Per/TensorE, Act/Scalar,
# Pool/Vector, DMA/DVE/SP) onto them.
_COMPUTE_ENGINES = ("pe", "act", "pool")   # the three on-core compute engines
_MOVEMENT_ENGINES = ("dma",)               # off-core data movement

# A compute engine is "dominant" (one engine serializes the pipeline) when it is
# the busiest AND no engine is near-saturated — i.e. the kernel is latency-bound
# on a single serial engine while the rest sit idle. If the busiest engine is
# itself near-saturated the kernel is simply compute-bound on that engine (still
# routed to SINGLE_ENGINE guidance: fuse + overlap), so this threshold only
# separates "serialized" from "genuinely saturated" for the REASON prose.
_SATURATED = 0.80
# A compute engine must be at least this busy to count as "the pipeline serializes
# on it". Below this floor NOTHING is meaningfully busy — the device is starved
# waiting on data (memory-bound), not serialized on an engine. Without this floor
# a kernel where every engine sits at ~15% would be mislabelled single-engine
# just because one compute engine edged out the others.
_SERIALIZE_MIN = 0.30
# DMA counts as the bottleneck when it is the busiest engine by at least this
# margin over the busiest compute engine (a clear data-movement bind, not a tie).
_DMA_MARGIN = 1.10

# --- NKI Performance Guide threshold bars (docs/nki-perf-guide.md) ------------
# The guide's "good" targets, used to turn a raw metric into a pass/fail symptom
# token the perf loop / perf_hints route on. A metric BELOW its bar is a symptom.
MFU_GOOD = 0.90          # matmul-dominated compute-bound kernel: MFU >= 90% is good
MBU_GOOD = 0.60          # memory-bound kernel: MBU >= 60% is good
COMPUTE_GOOD = 0.90      # compute-bound: busiest engine active >= 90% is good
SPILL_INVESTIGATE = 0.30 # spill traffic > 30% of SBUF<->device traffic -> investigate
DMA_IDEAL_KIB = 32.0     # a DMA transfer smaller than this is packet-rate bound


@dataclass(frozen=True)
class ProfileReport:
    """A per-engine profile of one measured kernel, reduced to the perf loop's
    bottleneck vocabulary.

    ``engine_busy`` maps normalized engine name -> busy fraction (0..1) of the
    measured window. ``dominant`` is the canonical bottleneck label
    (``memory_bound`` / ``single_engine`` / ``dma_blocked``) matching
    ``kernel_perf``. ``reason`` is the human, keyword-bearing string the perf
    loop's ``classify_bottleneck`` greps (so the analytic fallback is bypassed in
    favour of this real measurement). ``device_us`` is the profiled device
    latency in microseconds (0.0 if the profiler did not report one).

    ``mfu`` / ``mbu`` are the model-FLOPs / memory-bandwidth utilization fractions
    the profiler reports (-1.0 = not reported). ``spill_ratio`` is spill traffic
    as a fraction of SBUF<->device traffic (-1.0 = unknown). ``dma_transfer_kib``
    is the average DMA transfer size in KiB (-1.0 = unknown). These are the NKI
    Performance Guide signals ``perf_symptoms`` turns into routing tokens."""

    engine_busy: dict = field(default_factory=dict)
    dominant: str = MEMORY_BOUND
    reason: str = ""
    device_us: float = 0.0
    mfu: float = -1.0
    mbu: float = -1.0
    spill_ratio: float = -1.0
    dma_transfer_kib: float = -1.0

    @property
    def measured(self) -> bool:
        """True when this report carries real per-engine data (not an empty /
        fail-open placeholder)."""
        return bool(self.engine_busy)


def _busiest(engine_busy: dict, names: tuple) -> tuple[str, float]:
    """(name, busy) of the busiest engine among ``names`` present in
    ``engine_busy`` — ("", 0.0) if none are present."""
    best_name, best = "", 0.0
    for n in names:
        v = float(engine_busy.get(n, 0.0) or 0.0)
        if v > best:
            best_name, best = n, v
    return best_name, best


def summarize(engine_busy: dict, device_us: float = 0.0, *, mfu: float = -1.0,
              mbu: float = -1.0, spill_ratio: float = -1.0,
              dma_transfer_kib: float = -1.0) -> ProfileReport:
    """Reduce a per-engine busy map to a ``ProfileReport`` — the PURE core.

    Decision (best-first, mirrors the physical bind):
      * DMA the clear busiest engine (>= ``_DMA_MARGIN`` over the busiest compute
        engine) -> ``dma_blocked``: the load/store path is the limiter; the fix is
        double-buffering + wider tiles.
      * one compute engine busiest but NOTHING near-saturated -> ``single_engine``:
        the pipeline serializes on one engine while the others sit idle; the fix
        is activation-fusion + engine overlap.
      * otherwise (all engines lightly used, or a compute engine saturated with
        low arithmetic intensity) -> ``memory_bound``: the op is bandwidth-bound;
        the fix is fusion + hoisting invariant loads.

    The optional ``mfu``/``mbu``/``spill_ratio``/``dma_transfer_kib`` metrics (the
    NKI Perf Guide signals; -1.0 = not reported) are stored on the report and
    their threshold breaches appended to ``reason`` (so ``classify_bottleneck``'s
    grep and ``perf_symptoms`` both see them).

    Empty input -> a fail-open ``memory_bound`` report with ``measured=False`` so
    the caller falls back to the analytic label rather than trusting an empty
    profile. Never raises."""
    metrics = dict(mfu=mfu, mbu=mbu, spill_ratio=spill_ratio,
                   dma_transfer_kib=dma_transfer_kib)
    if not engine_busy:
        return ProfileReport({}, MEMORY_BOUND,
                             "no per-engine profile — analytic fallback",
                             device_us, **metrics)
    busy = {k: float(v or 0.0) for k, v in engine_busy.items()}
    dma_name, dma = _busiest(busy, _MOVEMENT_ENGINES)
    ce_name, ce = _busiest(busy, _COMPUTE_ENGINES)
    peak = max(busy.values(), default=0.0)
    extra = _metric_breach_note(metrics)

    if dma > 0.0 and dma >= ce * _DMA_MARGIN and dma >= ce:
        return ProfileReport(
            busy, DMA_BLOCKED,
            f"dma-blocked: DMA engine {dma*100:.0f}% busy vs compute "
            f"{ce_name.upper() or 'PE'} {ce*100:.0f}% — data movement is the "
            f"limiter (double-buffer + widen tiles){extra}", device_us, **metrics)

    if ce >= _SERIALIZE_MIN and peak < _SATURATED and ce >= dma:
        idle = ", ".join(f"{n.upper()} {busy.get(n, 0.0)*100:.0f}%"
                         for n in _COMPUTE_ENGINES if n != ce_name)
        return ProfileReport(
            busy, SINGLE_ENGINE,
            f"single-engine serialize: {ce_name.upper()} {ce*100:.0f}% busy while "
            f"{idle or 'other engines'} sit idle — no engine saturated, the "
            f"pipeline serializes on one engine (activation-fuse + overlap){extra}",
            device_us, **metrics)

    return ProfileReport(
        busy, MEMORY_BOUND,
        f"memory-bound: busiest engine only {peak*100:.0f}% busy — bandwidth-bound, "
        f"likely spilling an intermediate through HBM (fuse + hoist invariant "
        f"loads){extra}", device_us, **metrics)


def _metric_breach_note(metrics: dict) -> str:
    """A short ' | ...' suffix naming the NKI-Guide threshold breaches present in
    ``metrics`` (spill > 30%, small DMA, low MFU/MBU) so the reason string carries
    them for the grep/token path. Empty when nothing is reported or in-bounds."""
    notes = []
    sr = metrics.get("spill_ratio", -1.0)
    if sr is not None and sr >= 0.0 and sr > SPILL_INVESTIGATE:
        notes.append(f"spill {sr*100:.0f}% > {SPILL_INVESTIGATE*100:.0f}% (spill-high)")
    kib = metrics.get("dma_transfer_kib", -1.0)
    if kib is not None and kib >= 0.0 and kib < DMA_IDEAL_KIB:
        notes.append(f"DMA {kib:.0f} KiB < {DMA_IDEAL_KIB:.0f} KiB (small-dma)")
    mfu = metrics.get("mfu", -1.0)
    if mfu is not None and mfu >= 0.0 and mfu < MFU_GOOD:
        notes.append(f"MFU {mfu*100:.0f}% < {MFU_GOOD*100:.0f}% (low-mfu)")
    mbu = metrics.get("mbu", -1.0)
    if mbu is not None and mbu >= 0.0 and mbu < MBU_GOOD:
        notes.append(f"MBU {mbu*100:.0f}% < {MBU_GOOD*100:.0f}% (low-mbu)")
    return (" | " + "; ".join(notes)) if notes else ""


def perf_symptoms(report: ProfileReport) -> tuple[str, ...]:
    """The NKI-Guide symptom tokens for a report — the vocabulary ``perf_hints``
    routes on (shared with ``perf_hints.SYMPTOM_TOKENS``). Combines the coarse
    bottleneck label with any threshold breach (low-mfu/low-mbu/spill-high/
    small-dma). Empty tuple for an unmeasured report. Never raises."""
    if not report.measured:
        return ()
    toks: list[str] = []
    dom = (report.dominant or "").replace("_", "-")
    if dom:
        toks.append(dom)                                  # memory-bound|single-engine|dma-blocked
    if 0.0 <= report.mfu < MFU_GOOD:
        toks.append("low-mfu")
    if 0.0 <= report.mbu < MBU_GOOD:
        toks.append("low-mbu")
    if report.spill_ratio >= 0.0 and report.spill_ratio > SPILL_INVESTIGATE:
        toks.append("spill-high")
    if report.dma_transfer_kib >= 0.0 and report.dma_transfer_kib < DMA_IDEAL_KIB:
        toks.append("small-dma")
    return tuple(toks)


# ---------------------------------------------------------------------------
# tolerant parse of a neuron-profile summary
# ---------------------------------------------------------------------------
# neuron-profile emits engine names in several spellings across SDK versions; map
# each onto this module's normalized {pe, act, pool, dma}. Substring match, first
# hit wins, so "PE utilization" / "TensorEngine" both land on "pe".
_ENGINE_ALIASES = (
    ("dma", ("dma", "dve", " sp ", "sp_", "movement", "hbm", "bandwidth")),
    ("pe", ("pe", "tensor", "matmul", "pooling_matmul")),
    ("act", ("act", "scalar", "activation")),
    ("pool", ("pool", "vector", "gpsimd")),
)


def _normalize_engine(name: str) -> str:
    """Map a profiler's engine name onto {pe, act, pool, dma}; "" if unknown."""
    n = f" {(name or '').lower()} "
    for norm, aliases in _ENGINE_ALIASES:
        if any(a in n for a in aliases):
            return norm
    return ""


def _as_fraction(v: Any) -> float:
    """Coerce a busy value to a 0..1 fraction. Accepts a fraction (0.71), a
    percentage (71.0 -> 0.71), or a string ("71%" / "0.71"). Non-numeric -> 0.0."""
    try:
        if isinstance(v, str):
            v = v.strip().rstrip("%")
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0:            # a percentage
        f = f / 100.0
    return max(0.0, min(1.0, f))


# --- neuron-explorer summary-json schema ------------------------------------
# `neuron-profile` is DEPRECATED -> `neuron-explorer`. Its `view
# --output-format=summary-json` emits ``{"n_<hash>": {<metrics>}}`` where the
# per-engine utilization lives in SPECIFIC keys (values are 0..1 fractions
# despite the ``_percent`` suffix). We map ONLY those keys — a naive flatten
# would misread e.g. ``tensor_engine_instruction_count=55`` as 5500% util.
# Captured from a real trn2 run (neuron-explorer 2.32.0, 2026-08-28).
_EXPLORER_ENGINE_KEYS = {
    "pe": "tensor_engine_active_time_percent",
    "act": "scalar_engine_active_time_percent",
    "pool": "gpsimd_engine_active_time_percent",
    "dma": "dma_active_time_percent",
}
# Extra roofline signals neuron-explorer reports (compute / memory-bandwidth
# utilization) — surfaced on the report reason when present.
_EXPLORER_MFU_KEY = "mfu_estimated_percent"
_EXPLORER_MBU_KEY = "mbu_estimated_percent"
# Spill / DMA-size signals (Opt #2 / Opt #9). The spill and SBUF-traffic byte
# counters vary in spelling across SDK versions; try a few. dma transfer size is
# reported directly by some versions, else derived from total DMA bytes / count.
_EXPLORER_SPILL_KEYS = ("spill_save_bytes", "spill_reload_bytes", "spill_bytes")
_EXPLORER_SB_KEYS = ("sb_read_bytes", "sb_write_bytes", "sbuf_bytes")
_EXPLORER_DMA_KIB_KEYS = ("dma_avg_transfer_kib", "dma_transfer_kib",
                          "avg_dma_transfer_size_kib")
_EXPLORER_DMA_BYTES_KEYS = ("dma_total_bytes", "dma_bytes")
_EXPLORER_DMA_COUNT_KEYS = ("dma_transfer_count", "dma_count", "dma_instruction_count")


def _first_num(node: dict, keys: tuple, default: float = -1.0) -> float:
    """First numeric value among ``keys`` present in ``node`` (fraction-coerced
    only via float, NOT _as_fraction — these are byte/count/percent raw values).
    ``default`` when none present. Never raises."""
    for k in keys:
        if k in node:
            try:
                return float(node[k])
            except (TypeError, ValueError):
                continue
    return default


def parse_neuron_explorer_metrics(obj: Any) -> dict:
    """Extract the NKI-Guide scalar metrics (mfu, mbu, spill_ratio,
    dma_transfer_kib) from a neuron-explorer summary object. Same node-selection
    as ``parse_neuron_explorer_summary``. Returns only the metrics actually present
    (missing -> omitted, so the caller keeps the -1.0 'unknown' default). Never
    raises; ``{}`` on any miss."""
    try:
        node = obj
        if isinstance(obj, dict) and not _is_explorer_node(obj):
            nodes = [v for v in obj.values() if _is_explorer_node(v)]
            if not nodes:
                return {}
            node = max(nodes, key=lambda n: (
                _as_fraction(n.get(_EXPLORER_ENGINE_KEYS["dma"], 0)) +
                _as_fraction(n.get(_EXPLORER_ENGINE_KEYS["pe"], 0))))
        if not isinstance(node, dict):
            return {}
        out: dict = {}
        if _EXPLORER_MFU_KEY in node:
            out["mfu"] = _as_fraction(node[_EXPLORER_MFU_KEY])
        if _EXPLORER_MBU_KEY in node:
            out["mbu"] = _as_fraction(node[_EXPLORER_MBU_KEY])
        # spill_ratio = spill traffic / SBUF<->device traffic (guide's >30% bar).
        spill = sum(max(0.0, _first_num(node, (k,), 0.0)) for k in _EXPLORER_SPILL_KEYS
                    if k in node)
        sb = sum(max(0.0, _first_num(node, (k,), 0.0)) for k in _EXPLORER_SB_KEYS
                 if k in node)
        if sb > 0.0 and any(k in node for k in _EXPLORER_SPILL_KEYS):
            out["spill_ratio"] = spill / sb
        # dma transfer size: direct if reported, else total_bytes / count / 1024.
        kib = _first_num(node, _EXPLORER_DMA_KIB_KEYS)
        if kib < 0.0:
            b = _first_num(node, _EXPLORER_DMA_BYTES_KEYS)
            c = _first_num(node, _EXPLORER_DMA_COUNT_KEYS)
            if b >= 0.0 and c > 0.0:
                kib = (b / c) / 1024.0
        if kib >= 0.0:
            out["dma_transfer_kib"] = kib
        return out
    except Exception:  # noqa: BLE001
        return {}


def _is_explorer_node(d: Any) -> bool:
    """True when ``d`` is a neuron-explorer per-node metrics dict."""
    return isinstance(d, dict) and any(k in d for k in _EXPLORER_ENGINE_KEYS.values())


def parse_neuron_explorer_summary(obj: Any) -> dict:
    """Extract the normalized ``{pe,act,pool,dma: fraction}`` map from a
    neuron-explorer ``summary-json`` object (``{"n_<id>": {...}}`` or a bare
    per-node metrics dict). Reads ONLY the known engine-utilization keys, so
    instruction counts / times never leak in as bogus utilization. Uses the
    busiest node when several are present. Never raises; ``{}`` on any miss."""
    try:
        node = obj
        if isinstance(obj, dict) and not _is_explorer_node(obj):
            # {"n_<id>": {...}, ...} — pick the node with the highest DMA+PE busy.
            nodes = [v for v in obj.values() if _is_explorer_node(v)]
            if not nodes:
                return {}
            node = max(nodes, key=lambda n: (
                _as_fraction(n.get(_EXPLORER_ENGINE_KEYS["dma"], 0)) +
                _as_fraction(n.get(_EXPLORER_ENGINE_KEYS["pe"], 0))))
        if not _is_explorer_node(node):
            return {}
        return {eng: _as_fraction(node.get(key, 0.0))
                for eng, key in _EXPLORER_ENGINE_KEYS.items()
                if key in node}
    except Exception:  # noqa: BLE001
        return {}


def parse_profile_json(obj: Any) -> dict:
    """Extract a normalized ``{engine: busy_fraction}`` map from a neuron-profile /
    neuron-explorer summary object (already JSON-parsed). Tolerant of the several
    shapes the profiler emits — the neuron-explorer ``{"n_<id>": {...}}`` schema
    (handled first, schema-aware), a flat ``{"PE busy %": 71, ...}``, a nested
    ``{"engines": {"TensorEngine": {"utilization": 0.71}}}``, or a list of
    ``{"name":..., "busy":...}`` records — because the exact schema drifts across
    SDK versions and a strict parse would fail-closed on every bump. Unknown /
    unparseable input -> ``{}`` (which ``summarize`` treats as fail-open). Never
    raises."""
    # neuron-explorer summary-json takes precedence when recognized (its keys are
    # specific and a naive flatten would mis-scale its instruction-count fields).
    if isinstance(obj, dict):
        if _is_explorer_node(obj) or any(
                isinstance(v, dict) and _is_explorer_node(v) for v in obj.values()):
            got = parse_neuron_explorer_summary(obj)
            if got:
                return got
    out: dict = {}

    def _put(name: str, val: Any) -> None:
        eng = _normalize_engine(name)
        if eng:
            out[eng] = max(out.get(eng, 0.0), _as_fraction(val))

    try:
        if isinstance(obj, dict):
            # nested {"engines": {...}} takes precedence if present
            engines = obj.get("engines") or obj.get("engine_utilization")
            if isinstance(engines, dict):
                for name, rec in engines.items():
                    if isinstance(rec, dict):
                        _put(name, rec.get("utilization", rec.get("busy",
                             rec.get("busy_pct", 0.0))))
                    else:
                        _put(name, rec)
            elif isinstance(engines, list):
                for rec in engines:
                    if isinstance(rec, dict):
                        _put(rec.get("name", ""), rec.get("utilization",
                             rec.get("busy", rec.get("busy_pct", 0.0))))
            else:
                # flat dict of "<engine> ..." -> value
                for name, val in obj.items():
                    if isinstance(val, (int, float, str)):
                        _put(name, val)
        elif isinstance(obj, list):
            for rec in obj:
                if isinstance(rec, dict):
                    _put(rec.get("name", ""), rec.get("utilization",
                         rec.get("busy", rec.get("busy_pct", 0.0))))
    except Exception:  # noqa: BLE001 — a parse failure must fall back to analytic
        return {}
    return out


# ---------------------------------------------------------------------------
# the device path (guarded)
# ---------------------------------------------------------------------------
def profile_kernel(run_fn: Callable[[], Any], device: Any = None, *,
                   profiler: Callable[[Callable], dict] | None = None
                   ) -> ProfileReport | None:
    """Profile one measured kernel run and return a ``ProfileReport``, or ``None``
    when profiling is unavailable (off-device / no ``neuron-profile``) so the
    caller falls back to the analytic bottleneck.

    ``profiler`` is the INJECTED seam: a callable ``(run_fn) -> engine_busy_dict``
    (or a dict already parsed) that actually invokes the profiler. Injected so
    this function is unit-testable with a fake profiler and so the real
    ``neuron-profile`` integration lives in exactly one place (the caller wires
    it). With no ``profiler`` injected this returns ``None`` (there is no default
    device profiler here — the honest "cannot profile" signal), NEVER a fabricated
    report. Never raises: a profiler that throws degrades to ``None``."""
    if profiler is None:
        return None
    try:
        raw = profiler(run_fn)
    except Exception:  # noqa: BLE001 — a broken profiler is "cannot profile", not a crash
        return None
    if raw is None:
        return None
    busy = raw if _looks_normalized(raw) else parse_profile_json(raw)
    device_us = 0.0
    metrics: dict = {}
    if isinstance(raw, dict):
        try:
            device_us = float(raw.get("device_us", raw.get("total_us", 0.0)) or 0.0)
        except (TypeError, ValueError):
            device_us = 0.0
        # Pull the NKI-Guide scalar metrics (mfu/mbu/spill/dma-size) when present;
        # a pre-normalized busy dict carries no metrics (they stay unknown).
        if not _looks_normalized(raw):
            metrics = parse_neuron_explorer_metrics(raw)
    if not busy:
        return None
    return summarize(busy, device_us, **metrics)


def _looks_normalized(d: Any) -> bool:
    """True when ``d`` is already a normalized ``{pe|act|pool|dma: fraction}`` map
    (so we skip ``parse_profile_json`` and use it directly)."""
    if not isinstance(d, dict) or not d:
        return False
    keys = set(d.keys())
    return keys.issubset({"pe", "act", "pool", "dma"}) and bool(
        keys & {"pe", "act", "pool", "dma"})


def capture_profiler(neff_path: str, *, core: int = 1,
                     explorer: str = "neuron-explorer",
                     workdir: str = "/tmp") -> Callable[[Callable], dict] | None:
    """Build a real ``profiler(run_fn) -> engine_busy`` backed by the
    ``neuron-explorer`` capture→view flow (``neuron-profile`` is deprecated).

    Given a compiled ``.neff`` (obtainable from the neuronx-cc compile cache, e.g.
    ``/var/tmp/neuron-compile-cache/.../model.neff``), the returned profiler runs
    ``neuron-explorer capture -n <neff> -s <ntff> --io-from=neff`` on ``core`` and
    ``neuron-explorer view -n <neff> -s <ntff> --output-format=summary-json``, then
    parses the summary via ``parse_neuron_explorer_summary``. The ``run_fn`` arg is
    accepted for interface parity but not used (the NEFF is executed by the capture
    itself). Returns ``None`` (so ``profile_kernel`` fails open) when ``explorer``
    is not on PATH or ``neff_path`` is missing; the profiler itself returns ``{}``
    on any capture/view/parse failure. Never raises at build time."""
    import os
    import shutil
    if shutil.which(explorer) is None or not (neff_path and os.path.exists(neff_path)):
        return None

    import json
    import subprocess

    def profiler(_run_fn: Callable) -> dict:
        try:
            ntff = os.path.join(workdir, "topt_profile.ntff")
            env = dict(os.environ, NEURON_RT_VISIBLE_CORES=str(core))
            subprocess.run(
                [explorer, "capture", "-n", neff_path, "-s", ntff, "--io-from=neff"],
                check=True, capture_output=True, timeout=180, env=env)
            view = subprocess.run(
                [explorer, "view", "-n", neff_path, "-s", ntff,
                 "--output-format=summary-json"],
                check=True, capture_output=True, timeout=120, env=env)
            return json.loads(view.stdout.decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 — a capture/view/parse failure is "cannot profile"
            return {}

    return profiler


def latest_neff(cache_root: str = "/var/tmp/neuron-compile-cache") -> str | None:
    """The most-recently-modified ``model.neff`` under the neuronx-cc compile
    cache (the artifact a just-compiled kernel produced), or ``None`` if none
    exists. A convenience for wiring ``capture_profiler`` right after a kernel
    compiles. Never raises."""
    import os
    try:
        best, best_mtime = None, -1.0
        for root, _dirs, files in os.walk(cache_root):
            for f in files:
                if f.endswith(".neff"):
                    p = os.path.join(root, f)
                    m = os.path.getmtime(p)
                    if m > best_mtime:
                        best, best_mtime = p, m
        return best
    except Exception:  # noqa: BLE001
        return None
