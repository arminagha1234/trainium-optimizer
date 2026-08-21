"""
Auto-discovery — "a new model comes out -> it gets pulled into the queue
automatically", so the user never hand-maintains the model list.

This is the natural capstone of auto-onboarding (see
../../docs/auto-onboarding-design.md). The two compose:

  * DISCOVERY (this module) FINDS models — pulls recent/trending open
    text-generation models from HuggingFace, filters them down to ones that are
    *safe and sensible* to hand an autonomous loop (open license, fits the
    target instance, actually an LLM, not already queued), and appends the
    survivors to `models_queue.txt` tagged `discovered`.
  * AUTO-ONBOARDING (Tier-0 config-driven family mapping) then makes them
    RUNNABLE — it reads each discovered model's config, maps its structural
    fingerprint to a known family adapter, and produces an equivalence-verified
    baseline the Stage 0->6 loop can optimize.

The whole point: a discovered model that onboards + optimizes end to end with
**no human in the loop**. Discovery only picks the target by shape here (a
best-effort `family` from the config, `dense_causal_lm` / `moe_causal_lm`);
auto-onboarding refines it by the true structural fingerprint. So discovery is
allowed to be approximate on family — it must NOT be approximate on safety.

Design constraints (this feeds an AUTONOMOUS loop, so filters are conservative):
  - open/permissive license ONLY (apache-2.0 / mit / bsd / ... — skip gated,
    "other", or unknown-license models),
  - a param estimate that FITS the target instance (<= a cap, e.g. ~14B for a
    trn2.3xlarge bf16 run) — unknown size is treated as too-risky and dropped,
  - text-generation dense / MoE only (skip non-LLM and known-doomed
    linear-attention arches the pre-flight gate already rejects),
  - never re-queue something already in `models_queue.txt` (dedup by hf_id).

Honest reporting is a first-class requirement: EVERY drop is logged with a
reason (gated / bad-license / unknown-size / too-big / non-llm / linear-attn /
already-queued) — nothing is filtered silently.

Network-dependent (the HuggingFace hub) BY THE PRIMARY SOURCE ONLY. All hub
imports are gated inside the source, so this module imports cleanly on a laptop
with no network and is fully unit-testable against a MOCK source. It is a TOOL
the user / loop invokes (`python discover.py ...`); it does NOT auto-run inside
the overnight loop yet.

    # dry run: show what it WOULD discover + queue, nothing written
    python discover.py --limit 20 --max-params 14e9

    # commit the survivors to the queue (backs the queue up first)
    python discover.py --limit 20 --max-params 14e9 --append
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

# Reuse the pre-flight gate's config helpers so discovery and onboarding agree
# on what a config *means* (same arch extraction, same linear-attn detection).
# These are pure-python (no torch / no hub) and already unit-tested.
from preflight import _architectures, is_linear_attention_arch, load_hf_config

_LOG = logging.getLogger("discover")

# A config loader turns a model_id into a plain config dict (or None if it can't
# be read). Injected so the module never hard-depends on the hub and stays
# testable with canned configs. Mirrors preflight.ConfigLoader.
ConfigLoader = Callable[[str], "dict[str, Any] | None"]


# ---------------------------------------------------------------------------
# license policy (conservative — permissive/open only)
# ---------------------------------------------------------------------------

# Truly permissive licenses we are willing to hand an autonomous loop. Anything
# NOT in this set (including None, "other", "unknown", and the various
# gated/community licenses like the Llama/Gemma custom terms) is dropped. Start
# strict; loosen only with evidence (mirrors the design doc's "start strict").
DEFAULT_OPEN_LICENSES: frozenset[str] = frozenset({
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
})

# A trn2.3xlarge bf16 run tops out around here; keep discovery's default cap
# conservative so we never queue something that can't even baseline.
DEFAULT_MAX_PARAMS: float = 14e9

DISCOVERED_TAG = "discovered"


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class RawModel:
    """A model as a SOURCE reports it, before any filtering. Everything here is
    cheap listing metadata — no config fetch, no weights."""
    hf_id: str
    downloads: int = 0
    trending_score: float = 0.0
    likes: int = 0
    pipeline_tag: str | None = None
    license: str | None = None
    gated: bool = False          # HF "gated" (auto/manual) — needs access grant
    private: bool = False
    tags: tuple[str, ...] = ()
    # Some sources (HF `full=True`) report an exact param total from the
    # safetensors index — the most reliable size signal when present.
    param_count: float = 0.0


@dataclass
class Candidate:
    """A model that SURVIVED filtering — safe to queue for the autonomous loop.
    `family`/`tag` are exactly the queue columns; `family` is best-effort by
    shape and auto-onboarding refines it."""
    hf_id: str
    family: str                  # dense_causal_lm | moe_causal_lm
    tag: str = DISCOVERED_TAG
    param_count: float = 0.0
    license: str | None = None
    arch: str = ""               # HF architecture class name, for the report
    downloads: int = 0
    trending_score: float = 0.0

    def queue_line(self) -> str:
        """The `hf_id<TAB>family<TAB>tag` row appended to models_queue.txt."""
        return f"{self.hf_id}\t{self.family}\t{self.tag}"


@dataclass
class Drop:
    """A model that was FILTERED OUT, with a machine-readable reason. Collected
    so the CLI can print an honest breakdown and tests can assert on it."""
    hf_id: str
    reason: str                  # gated | bad-license | unknown-size | too-big |
    #                              non-llm | linear-attn | already-queued | no-config
    detail: str = ""


@dataclass
class Filters:
    """The conservative filter policy. Defaults are safe for an autonomous
    loop; the CLI can widen the cap or the license set explicitly."""
    max_params: float = DEFAULT_MAX_PARAMS
    allowed_licenses: frozenset[str] = DEFAULT_OPEN_LICENSES
    allow_moe: bool = True
    # Normalized (lower-cased) hf_ids already in the queue — dedup source.
    queued_ids: frozenset[str] = frozenset()

    @classmethod
    def build(
        cls,
        max_params: float = DEFAULT_MAX_PARAMS,
        allowed_licenses: Iterable[str] | None = None,
        allow_moe: bool = True,
        queue_path: "str | Path | None" = None,
    ) -> "Filters":
        licenses = (frozenset(l.lower() for l in allowed_licenses)
                    if allowed_licenses is not None else DEFAULT_OPEN_LICENSES)
        queued = load_queued_ids(queue_path) if queue_path else frozenset()
        return cls(max_params=max_params, allowed_licenses=licenses,
                   allow_moe=allow_moe, queued_ids=queued)


@dataclass
class DiscoveryReport:
    """Everything one discovery pass produced — kept candidates AND every drop
    (with reasons). `candidates` is the primary `list[Candidate]` result."""
    candidates: list[Candidate] = field(default_factory=list)
    drops: list[Drop] = field(default_factory=list)
    scanned: int = 0

    def drops_by_reason(self) -> dict[str, list[Drop]]:
        out: dict[str, list[Drop]] = {}
        for d in self.drops:
            out.setdefault(d.reason, []).append(d)
        return out


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

class ModelSource(Protocol):
    """A source of candidate models. `name` is for reporting; `list_models`
    returns raw listing metadata. A source MAY also expose a `config_loader`
    used to fetch each model's config during filtering."""
    name: str

    def list_models(self, limit: int, sort: str) -> list[RawModel]: ...


class HFHubSource:
    """PRIMARY source: recent/trending open text-generation models from the
    HuggingFace hub. The `huggingface_hub` import is GATED here, so importing
    this module never requires the hub — only *using* this source does. Fails
    gracefully (clear RuntimeError) when the hub is unavailable/offline."""

    name = "huggingface"

    def __init__(self, task: str = "text-generation"):
        self.task = task

    def list_models(self, limit: int, sort: str) -> list[RawModel]:
        try:
            from huggingface_hub import list_models as hf_list_models
        except Exception as e:  # noqa: BLE001 — offline / not installed
            raise RuntimeError(
                "huggingface_hub is unavailable (offline or not installed); "
                "use --source mock for an offline dry run"
            ) from e
        try:
            infos = hf_list_models(
                task=self.task, sort=sort, direction=-1, limit=limit,
                full=True, cardData=True,
            )
        except Exception as e:  # noqa: BLE001 — network / API failure
            raise RuntimeError(f"HF hub list_models failed: {e}") from e
        return [self._to_raw(mi) for mi in infos]

    def config_loader(self, model_id: str) -> "dict[str, Any] | None":
        """Cheap, weight-free config fetch via the hub (falls back to the shared
        loader, which also handles a local model dir)."""
        return load_hf_config(model_id)

    @staticmethod
    def _to_raw(mi: Any) -> RawModel:
        card = getattr(mi, "cardData", None) or {}
        lic = card.get("license") if isinstance(card, dict) else None
        # Param total from the safetensors index when present (most reliable).
        params = 0.0
        st = getattr(mi, "safetensors", None)
        if st is not None:
            total = getattr(st, "total", None)
            if total is None and isinstance(st, dict):
                total = st.get("total")
            if total:
                params = float(total)
        gated = bool(getattr(mi, "gated", False))  # HF: False | "auto" | "manual"
        return RawModel(
            hf_id=getattr(mi, "id", "") or getattr(mi, "modelId", ""),
            downloads=int(getattr(mi, "downloads", 0) or 0),
            trending_score=float(getattr(mi, "trending_score", 0.0) or 0.0),
            likes=int(getattr(mi, "likes", 0) or 0),
            pipeline_tag=getattr(mi, "pipeline_tag", None),
            license=str(lic).lower() if lic else None,
            gated=gated,
            private=bool(getattr(mi, "private", False)),
            tags=tuple(getattr(mi, "tags", ()) or ()),
            param_count=params,
        )


class MockSource:
    """Offline, deterministic source for tests and dry demos. Carries the raw
    listing AND a config map so a whole discovery pass runs with no network."""

    name = "mock"

    def __init__(self, models: list[RawModel],
                 configs: "dict[str, dict[str, Any]] | None" = None):
        self._models = list(models)
        self._configs = dict(configs or {})

    def list_models(self, limit: int, sort: str) -> list[RawModel]:
        key = {
            "downloads": lambda m: m.downloads,
            "likes": lambda m: m.likes,
        }.get(sort, lambda m: m.trending_score)
        return sorted(self._models, key=key, reverse=True)[:limit]

    def config_loader(self, model_id: str) -> "dict[str, Any] | None":
        return self._configs.get(model_id)


# ---------------------------------------------------------------------------
# config -> shape (family + param estimate)
# ---------------------------------------------------------------------------

def _text_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Multimodal configs nest the LM under `text_config`; look there first for
    LM fields, falling back to the top level."""
    tc = config.get("text_config")
    return tc if isinstance(tc, dict) else config


def _get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    tc = _text_cfg(config)
    for src in (tc, config):
        for k in keys:
            v = src.get(k)
            if v is not None:
                return v
    return default


def _moe_experts(config: dict[str, Any]) -> int:
    """Number of routed experts if this is an MoE, else 0."""
    n = _get(config, "num_local_experts", "num_experts", "n_routed_experts",
             "moe_num_experts", default=0)
    try:
        return int(n or 0)
    except (TypeError, ValueError):
        return 0


def family_from_config(config: dict[str, Any] | None) -> str | None:
    """Best-effort family by SHAPE (not by name), matching the queue's vocab:
      - "moe_causal_lm"   -> config declares routed experts,
      - "dense_causal_lm" -> a *ForCausalLM decoder with no experts,
      - None              -> not a text-generation LM (encoder / diffusion / ...)
                             so discovery drops it as non-llm.
    Auto-onboarding's Tier-0 refines this from the full structural fingerprint;
    discovery only needs the coarse dense-vs-MoE split for the queue column."""
    if not config:
        return None
    archs = _architectures(config)
    is_causal_lm = any(a.endswith("ForCausalLM") for a in archs)
    # model_type alone isn't enough to call it an LM, but a CausalLM arch is.
    if not is_causal_lm:
        # A couple of configs omit `architectures`; fall back to a decoder-ish
        # model_type + the presence of core LM fields.
        mt = str(_get(config, "model_type", default="")).lower()
        has_lm_fields = _get(config, "vocab_size") and _get(
            config, "num_hidden_layers", "n_layer", "num_layers")
        if not (mt and has_lm_fields):
            return None
    return "moe_causal_lm" if _moe_experts(config) > 0 else "dense_causal_lm"


def estimate_params_from_config(config: dict[str, Any] | None) -> float:
    """Rough parameter count from config alone (no weights). Good enough for a
    conservative <= cap gate. Returns 0.0 when the config lacks the core fields
    (the caller then treats size as UNKNOWN and drops it — conservative)."""
    if not config:
        return 0.0
    hidden = _get(config, "hidden_size", "n_embd", "d_model")
    layers = _get(config, "num_hidden_layers", "n_layer", "num_layers")
    vocab = _get(config, "vocab_size")
    if not (hidden and layers and vocab):
        return 0.0
    hidden = int(hidden)
    layers = int(layers)
    vocab = int(vocab)
    heads = int(_get(config, "num_attention_heads", "n_head",
                     default=max(1, hidden // 128)))
    kv_heads = int(_get(config, "num_key_value_heads", default=heads))
    head_dim = int(_get(config, "head_dim",
                        default=(hidden // heads if heads else 0)))
    inter = int(_get(config, "intermediate_size", "ffn_dim", "n_inner",
                     default=4 * hidden))

    # Attention block: q + k + v + o (GQA shrinks k/v).
    q = hidden * heads * head_dim
    kv = 2 * hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    attn = q + kv + o

    n_experts = _moe_experts(config)
    if n_experts > 0:
        moe_inter = int(_get(config, "moe_intermediate_size",
                             "intermediate_size", default=inter))
        # Modern gated MLP = 3 matrices (gate, up, down); all experts count
        # toward the *total* parameter size (what the instance must hold).
        mlp = n_experts * 3 * hidden * moe_inter
        shared = int(_get(config, "n_shared_experts", "num_shared_experts",
                          default=0) or 0)
        if shared:
            mlp += shared * 3 * hidden * moe_inter
        mlp += hidden * n_experts  # router
    else:
        mlp = 3 * hidden * inter   # gated SwiGLU dense MLP

    per_layer = attn + mlp
    embed = vocab * hidden
    total = embed + layers * per_layer
    if not _get(config, "tie_word_embeddings", default=True):
        total += vocab * hidden  # separate lm_head
    return float(total)


def estimate_params(raw: RawModel, config: dict[str, Any] | None) -> float:
    """Best available param estimate: the source's exact safetensors total if
    present, else a config-based estimate, else parse a size hint from the id
    (e.g. `...-7B`, `...-1.5b`, `...-800M`). 0.0 means genuinely unknown."""
    if raw.param_count and raw.param_count > 0:
        return raw.param_count
    est = estimate_params_from_config(config)
    if est > 0:
        return est
    return _params_from_id(raw.hf_id)


def _params_from_id(hf_id: str) -> float:
    """Parse a trailing size token from the model id as a LAST resort."""
    import re
    best = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([bBmM])", hf_id):
        val = float(num) * (1e9 if unit in "bB" else 1e6)
        best = max(best, val)
    return best


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------

def _drop(report: DiscoveryReport, hf_id: str, reason: str, detail: str = "",
          logger: logging.Logger | None = None) -> None:
    """Record + LOG one drop. No model is ever filtered silently."""
    report.drops.append(Drop(hf_id=hf_id, reason=reason, detail=detail))
    (logger or _LOG).info("DROP  %-14s %s%s", reason, hf_id,
                          f"  ({detail})" if detail else "")


def discover(
    sources: "ModelSource | list[ModelSource]",
    limit: int = 20,
    filters: Filters | None = None,
    *,
    sort: str = "trendingScore",
    config_loader: ConfigLoader | None = None,
    logger: logging.Logger | None = None,
) -> DiscoveryReport:
    """Pull recent/trending open text-generation models and filter them down to
    safe, autonomous-loop-ready candidates.

    Returns a `DiscoveryReport`: `.candidates` is the primary `list[Candidate]`
    result, `.drops` records every filtered model with a reason. Every drop is
    also logged (no silent filtering).

    `config_loader` overrides how a model's config is fetched (defaults to the
    source's own `.config_loader`, then the shared `preflight.load_hf_config`).
    """
    log = logger or _LOG
    flt = filters or Filters()
    src_list = [sources] if not isinstance(sources, list) else sources
    report = DiscoveryReport()

    # Merge sources, dedup by id (first source wins), cap at limit.
    seen: set[str] = set()
    raws: list[tuple[RawModel, ModelSource]] = []
    for src in src_list:
        for raw in src.list_models(limit=limit, sort=sort):
            key = raw.hf_id.lower()
            if not raw.hf_id or key in seen:
                continue
            seen.add(key)
            raws.append((raw, src))
    raws = raws[:limit]
    report.scanned = len(raws)

    for raw, src in raws:
        hf_id = raw.hf_id

        # 1. gated / private — needs an access grant; an autonomous loop can't
        #    click through, so it would only ever FAIL_NO_BASELINE.
        if raw.gated or raw.private:
            _drop(report, hf_id, "gated",
                  "private" if raw.private else "gated repo", log)
            continue

        # 2. license — permissive/open ONLY. Unknown/None/"other" is a drop.
        lic = (raw.license or "").lower()
        if lic not in flt.allowed_licenses:
            _drop(report, hf_id, "bad-license", lic or "unknown", log)
            continue

        # 3. config fetch — needed for family + a real size estimate. If we
        #    can't read it, we can't reason about it -> drop (conservative).
        loader = config_loader or getattr(src, "config_loader", None) or load_hf_config
        cfg = loader(hf_id)
        if cfg is None and (not raw.param_count):
            _drop(report, hf_id, "no-config", "config unreadable", log)
            continue

        # 4. arch class — must be a text-gen LM, and NOT a known-doomed
        #    linear-attention arch the pre-flight gate already rejects.
        if is_linear_attention_arch(cfg):
            _drop(report, hf_id, "linear-attn",
                  "neuronx-cc ISA-unsupported (needs adapter)", log)
            continue
        family = family_from_config(cfg)
        if family is None:
            # No usable config but a pipeline_tag says text-generation: treat as
            # a dense LM by default (onboarding will refine). Otherwise non-LLM.
            if cfg is None and raw.pipeline_tag == "text-generation":
                family = "dense_causal_lm"
            else:
                _drop(report, hf_id, "non-llm",
                      raw.pipeline_tag or "no CausalLM arch", log)
                continue
        if family == "moe_causal_lm" and not flt.allow_moe:
            _drop(report, hf_id, "non-llm", "moe excluded by policy", log)
            continue

        # 5. size — must FIT the target instance. Unknown size is too risky.
        params = estimate_params(raw, cfg)
        if params <= 0:
            _drop(report, hf_id, "unknown-size",
                  "no param estimate from config/id", log)
            continue
        if params > flt.max_params:
            _drop(report, hf_id, "too-big",
                  f"~{params/1e9:.1f}B > {flt.max_params/1e9:.0f}B cap", log)
            continue

        # 6. dedup — never re-queue something already in models_queue.txt.
        if hf_id.lower() in flt.queued_ids:
            _drop(report, hf_id, "already-queued", "in models_queue.txt", log)
            continue

        archs = _architectures(cfg or {})
        cand = Candidate(
            hf_id=hf_id, family=family, tag=DISCOVERED_TAG,
            param_count=params, license=lic,
            arch=archs[0] if archs else "",
            downloads=raw.downloads, trending_score=raw.trending_score,
        )
        report.candidates.append(cand)
        log.info("KEEP  %-14s %s  (~%.1fB, %s, %s)", family, hf_id,
                 params / 1e9, lic, cand.arch or "?")

    return report


# ---------------------------------------------------------------------------
# queue I/O
# ---------------------------------------------------------------------------

def default_queue_path() -> Path:
    """The live queue at the repo root: <repo>/models_queue.txt. This module
    lives at <repo>/implementation/src/discover.py, so the root is two up."""
    return Path(__file__).resolve().parents[2] / "models_queue.txt"


def read_queue(queue_path: "str | Path") -> list[str]:
    """Return the hf_ids already in the queue (first TAB-separated column),
    skipping blanks and `#` comments. Missing file -> []."""
    p = Path(queue_path)
    if not p.exists():
        return []
    ids: list[str] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split("\t")[0].strip())
    return ids


def load_queued_ids(queue_path: "str | Path | None") -> frozenset[str]:
    """Normalized (lower-cased) set of queued hf_ids, for dedup."""
    if not queue_path:
        return frozenset()
    return frozenset(i.lower() for i in read_queue(queue_path))


def backup_queue(queue_path: "str | Path") -> Path | None:
    """Copy the queue to a timestamped `.bak-<UTC>` alongside it, BEFORE any
    append. Returns the backup path, or None if there was nothing to back up."""
    p = Path(queue_path)
    if not p.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    bak = p.with_name(f"{p.name}.bak-{stamp}")
    shutil.copy2(p, bak)
    return bak


def append_to_queue(
    candidates: list[Candidate],
    queue_path: "str | Path | None" = None,
    *,
    backup: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[list[Candidate], Path | None]:
    """Append NEW candidates to the queue in `hf_id<TAB>family<TAB>tag` format,
    never duplicating an id already present (double-checks against the file even
    if discovery already deduped), backing the queue up first.

    Returns (appended_candidates, backup_path). `appended` excludes anything
    that was already in the queue.
    """
    log = logger or _LOG
    path = Path(queue_path) if queue_path else default_queue_path()

    existing = {i.lower() for i in read_queue(path)}
    fresh: list[Candidate] = []
    seen_this_batch: set[str] = set()
    for c in candidates:
        key = c.hf_id.lower()
        if key in existing or key in seen_this_batch:
            log.info("SKIP-APPEND  %s already in queue", c.hf_id)
            continue
        seen_this_batch.add(key)
        fresh.append(c)

    if not fresh:
        log.info("append_to_queue: nothing new to append")
        return [], None

    bak = backup_queue(path) if backup else None
    if bak:
        log.info("backed up queue -> %s", bak)

    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open("a") as fh:
        if header_needed:
            fh.write("# models_queue.txt — hf_id<TAB>family<TAB>tag "
                     "(family refined by auto-onboarding)\n")
        for c in fresh:
            fh.write(c.queue_line() + "\n")
    log.info("appended %d model(s) to %s", len(fresh), path)
    return fresh, bak


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_source(name: str) -> ModelSource:
    if name == "hf":
        return HFHubSource()
    if name == "mock":
        # A tiny built-in demo set so `--source mock` works with no network and
        # shows the filtering in action (one keep + a couple of drops).
        return _demo_mock_source()
    raise SystemExit(f"unknown --source {name!r} (use: hf | mock)")


def _demo_mock_source() -> MockSource:
    models = [
        RawModel("acme/tinylm-1b", downloads=5000, trending_score=90.0,
                 pipeline_tag="text-generation", license="apache-2.0"),
        RawModel("acme/huge-70b", downloads=9000, trending_score=80.0,
                 pipeline_tag="text-generation", license="mit"),
        RawModel("meta/gated-8b", downloads=8000, trending_score=70.0,
                 pipeline_tag="text-generation", license="other", gated=True),
    ]
    configs = {
        "acme/tinylm-1b": {"architectures": ["LlamaForCausalLM"],
                           "model_type": "llama", "hidden_size": 2048,
                           "num_hidden_layers": 22, "vocab_size": 32000,
                           "num_attention_heads": 16, "intermediate_size": 5632},
        "acme/huge-70b": {"architectures": ["LlamaForCausalLM"],
                          "model_type": "llama", "hidden_size": 8192,
                          "num_hidden_layers": 80, "vocab_size": 32000,
                          "num_attention_heads": 64, "intermediate_size": 28672},
    }
    return MockSource(models, configs)


def _report_text(report: DiscoveryReport, appended: list[Candidate] | None,
                 backup: Path | None, dry_run: bool) -> str:
    lines: list[str] = []
    lines.append(f"scanned {report.scanned} model(s); "
                 f"kept {len(report.candidates)}, dropped {len(report.drops)}")
    lines.append("")
    lines.append("KEPT (candidates):")
    if report.candidates:
        for c in report.candidates:
            lines.append(f"  + {c.hf_id}\t{c.family}\t~{c.param_count/1e9:.1f}B"
                         f"\t{c.license}\t{c.arch or '?'}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("DROPPED (with reasons):")
    by_reason = report.drops_by_reason()
    if by_reason:
        for reason in sorted(by_reason):
            ds = by_reason[reason]
            lines.append(f"  {reason} ({len(ds)}):")
            for d in ds:
                lines.append(f"    - {d.hf_id}"
                             + (f"  ({d.detail})" if d.detail else ""))
    else:
        lines.append("  (none)")
    lines.append("")
    if dry_run:
        lines.append("DRY RUN — would append these NEW rows "
                     "(re-run with --append to write):")
        for c in report.candidates:
            lines.append(f"  {c.queue_line()}")
    else:
        lines.append(f"APPENDED {len(appended or [])} new row(s)"
                     + (f"; backup -> {backup}" if backup else ""))
        for c in (appended or []):
            lines.append(f"  {c.queue_line()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="hf", help="hf (HuggingFace hub) | mock")
    ap.add_argument("--limit", type=int, default=20,
                    help="how many trending/recent models to pull + consider")
    ap.add_argument("--sort", default="trendingScore",
                    help="hub sort key: trendingScore | downloads | likes")
    ap.add_argument("--max-params", type=float, default=DEFAULT_MAX_PARAMS,
                    help="param cap in absolute count (e.g. 14e9). Bigger models "
                         "are dropped as too-big.")
    ap.add_argument("--licenses", nargs="*", default=None,
                    help="override the open-license allowlist "
                         f"(default: {sorted(DEFAULT_OPEN_LICENSES)})")
    ap.add_argument("--no-moe", action="store_true",
                    help="exclude MoE models (keep dense only)")
    ap.add_argument("--queue", type=Path, default=None,
                    help="queue path (default: <repo>/models_queue.txt)")
    ap.add_argument("--append", action="store_true",
                    help="append survivors to the queue (default is a dry run)")
    ap.add_argument("--quiet", action="store_true",
                    help="don't stream per-model KEEP/DROP log lines")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(message)s")

    queue_path = a.queue or default_queue_path()
    filters = Filters.build(
        max_params=a.max_params, allowed_licenses=a.licenses,
        allow_moe=not a.no_moe, queue_path=queue_path,
    )

    try:
        source = _build_source(a.source)
        report = discover(source, limit=a.limit, filters=filters, sort=a.sort)
    except RuntimeError as e:
        # Network-dependent source failed — fail gracefully, don't crash.
        print(f"discovery unavailable: {e}")
        return 2

    appended: list[Candidate] = []
    backup: Path | None = None
    if a.append:
        appended, backup = append_to_queue(report.candidates, queue_path)

    print()
    print(_report_text(report, appended, backup, dry_run=not a.append))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
