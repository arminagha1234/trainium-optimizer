"""
Trajectory chart — "how did it improve", generated from the ledger.

Modeled on `internal-prior-optimization-run`'s generate_chart.py,
with one significant change: that version hardcodes its data arrays
("simplified from ~100 experiments"). This reads the ledger directly, so the
chart can never drift from the record and it scales to 100 models.

Design goals (clean light "report figure", modeled on the Tongyi-30B-A3B
optimization-timeline reference):
  - white background, dark text, high contrast, generous margins, large fonts
  - x-axis = the optimization steps in order, grouped into labeled STAGE bands
    that span the WHOLE pipeline (Baseline 0 .. Profile 6). Every stage the
    optimizer walked gets a band, even one that produced no winning candidate,
    so the viewer sees the full pipeline was explored honestly.
  - one strong accent for the winning path: a stepped tok/s line, each kept
    step a big dot labelled with its "+NN% . idea" and stage
  - discarded candidates drawn faintly (light-gray x) so they never fight the
    trajectory — the previous chart's clutter was the main complaint
  - a subtle MFU strip below, on its own axis, so it can't confuse the story
  - final headline badge (tok/s + speedup) and a conditions/correctness line

Usage:
    python -m trajectory_chart --run-dir optimization_runs/gemma-4-31b \\
        --model "Gemma 4 31B" --hardware trn2.48xlarge --tp 8 \\
        --shape "chat 1k/512" --sdk 2.28.0 --roofline 48000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ledger import Ledger, Origin, Row, Stage  # noqa: E402


# -- theme -------------------------------------------------------------------

THEME = {
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#c9d0d8",
    "text.color": "#1b2733",
    "axes.labelcolor": "#3d4a57",
    "xtick.color": "#54606c",
    "ytick.color": "#54606c",
    "grid.color": "#e3e8ee",
    "grid.alpha": 1.0,
    "font.family": "DejaVu Sans",
    "font.size": 11,
}

# -- clean light-theme palette for build_chart -------------------------------
# One strong accent for the winning path; everything else is muted context so
# the tok/s improvement story reads instantly.
INK = "#1b2733"            # primary text
SUBTLE_INK = "#5b6875"     # secondary text / conditions line
ACCENT = "#1668dc"         # winning trajectory line + kept dots
ACCENT_DK = "#0d47a1"      # darker accent for the headline badge
GAIN_GREEN = "#1a7f37"     # "+NNN%" gain labels
HELD_GRAY = "#98a4b3"      # flat "incumbent held" line through no-win stages
DISCARD_GRAY = "#b9c1cc"   # faded discarded-candidate marks

# Band tints: gain-producing stages get a faint blue wash; stages walked but
# producing no winning candidate get a neutral gray wash — so the viewer sees
# the whole pipeline was explored honestly, not that stages were hidden.
BAND_WIN_TINT = "#eaf2fd"
BAND_NOWIN_TINT = "#f3f4f6"
BAND_EDGE = "#dfe4ea"

# Canonical pipeline order + display metadata (number, short label). Every
# stage present in the ledger gets a band in this order, so stages 0..6 are
# all visible even when a later stage produced no promoted candidate.
STAGE_META: dict[Stage, tuple[str, str]] = {
    Stage.PREFLIGHT: ("", "Preflight"),
    Stage.BASELINE: ("0", "Baseline"),
    Stage.HARVEST: ("0.5", "Harvest"),
    Stage.CONFIG: ("1", "Config"),
    Stage.KNOWN_KERNEL: ("2", "Known-Kernel"),
    Stage.BORROW: ("3", "Borrow"),
    Stage.INVENT: ("4", "Invent"),
    Stage.GRAPH_REWRITE: ("5", "Graph-Rewrite"),
    Stage.PROFILE_LOOP: ("6", "Profile"),
}
_PIPELINE_ORDER: list[Stage] = list(STAGE_META.keys())

STAGE_COLOR: dict[Stage, str] = {
    Stage.BASELINE: "#8b949e",
    Stage.HARVEST: "#a371f7",
    Stage.CONFIG: "#58a6ff",
    Stage.KNOWN_KERNEL: "#39c5cf",
    Stage.BORROW: "#3fb950",
    Stage.INVENT: "#f0883e",
    Stage.GRAPH_REWRITE: "#f85149",
    Stage.PROFILE_LOOP: "#d29922",
}

# Provenance is encoded as marker shape so it reads at a glance alongside the
# stage color.
ORIGIN_MARKER: dict[Origin, str] = {
    Origin.NONE: "s",        # square — config, no kernel provenance
    Origin.HARVESTED: "o",   # circle — off the shelf
    Origin.BORROWED: "^",    # triangle — ported from a reference
    Origin.HYBRID: "P",      # plus — borrowed then restructured
    Origin.INVENTED: "D",    # diamond — novel
}

# Below this, annotating every gain makes the chart unreadable.
ANNOTATE_THRESHOLD_PCT = 5.0

# Execution-mode encoding (eager vs torch.compile). Marker-edge ring colors.
COMPILED_EDGE = "#3fb950"   # bright green ring = torch.compile(backend="neuron")
EAGER_EDGE = "#6e7681"      # muted grey ring = eager

# First-forward time (seconds) above which we treat a point as compiled when
# the description carries no explicit compile_mode signal. Eager's lazy build
# is seconds; a NEFF compile is minutes.
COMPILE_S_HEURISTIC = 60.0


def _fmt_thousands(v: float, _pos: int | None = None) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}K"
    return f"{int(v)}"


def _stage_bands(kept: list[Row]) -> list[tuple[float, Stage]]:
    """X positions where the stage changes, for the separator lines."""
    bands: list[tuple[float, Stage]] = []
    prev: Stage | None = None
    for i, r in enumerate(kept):
        if r.stage is not prev:
            bands.append((i - 0.5, r.stage))
            prev = r.stage
    return bands


def _pipeline_bands(rows: list[Row]) -> list[dict]:
    """Lay every stage the optimizer walked onto the x-axis, in canonical
    pipeline order, as contiguous bands.

    Each kept row becomes a "win" step (a real improvement point). A stage that
    was walked but promoted nothing still gets one flat placeholder slot, so the
    full Baseline->Profile pipeline is always visible and the trajectory holds
    at the incumbent through those stages instead of hiding them.

    x is a continuous cursor. A win step takes unit width; a no-win stage takes
    a wider slot so its header (e.g. "Known-Kernel") never collides with its
    neighbour's. Band dicts carry left/right edges (x0, x1) directly.

    Returns a list of band dicts:
        {stage, number, label, x0, x1, has_win,
         steps: [{x, row|None, y, is_win}], best_discarded_y}
    """
    win_w = 1.0
    nowin_w = 1.9
    present = [s for s in _PIPELINE_ORDER if any(r.stage is s for r in rows)]
    bands: list[dict] = []
    cursor = 0.0
    incumbent = 0.0
    for s in present:
        stage_rows = [r for r in rows if r.stage is s]
        kept_in = [r for r in stage_rows if r.kept]
        disc_pos = [r.metric for r in stage_rows if not r.kept and r.metric > 0]
        num, label = STAGE_META[s]
        x0 = cursor
        steps: list[dict] = []
        if kept_in:
            for r in kept_in:
                incumbent = r.metric
                steps.append({"x": cursor + win_w / 2, "row": r,
                              "y": r.metric, "is_win": True})
                cursor += win_w
        else:
            # walked, no promoted candidate: one flat slot at the incumbent
            steps.append({"x": cursor + nowin_w / 2, "row": None,
                          "y": incumbent, "is_win": False})
            cursor += nowin_w
        bands.append({
            "stage": s, "number": num, "label": label,
            "x0": x0, "x1": cursor, "has_win": bool(kept_in), "steps": steps,
            "best_discarded_y": max(disc_pos) if disc_pos else None,
        })
    return bands


def _idea_short(row: Row) -> str:
    """One short prose phrase for a kept step's label (idea, not knob name)."""
    idea = _ideafy(row).replace("\n", " ")
    return " ".join(idea.split())


def build_chart(
    run_dir: Path,
    out_path: Path,
    model: str,
    hardware: str = "",
    tp: int | None = None,
    shape: str = "",
    sdk: str = "",
    roofline: float | None = None,
    metric_label: str = "tok/s",
) -> Path:
    """Clean, white, report-style optimization trajectory.

    The winning tok/s path is a stepped accent line climbing left->right; every
    kept step is a labelled dot ("+NN% . idea"); the full Baseline->Profile
    pipeline is drawn as labelled stage bands even where a stage produced no
    winner; discards are faint; MFU is a subtle strip below. Signature is
    unchanged so the overnight loop keeps calling it as-is.
    """
    led = Ledger(run_dir)
    rows = led.read()
    if not rows:
        raise SystemExit(f"ledger is empty: {led.path}")

    kept = [r for r in rows if r.kept]
    discarded = [r for r in rows if not r.kept]
    if not kept:
        raise SystemExit("no kept rows — nothing to plot")

    bands = _pipeline_bands(rows)
    steps = [st for b in bands for st in b["steps"]]
    xs = [st["x"] for st in steps]
    ys = [st["y"] for st in steps]
    n = len(steps)
    ymax_data = max(ys)

    # First step where a win was recorded (baseline) and the incumbent step.
    win_steps = [st for st in steps if st["is_win"]]
    baseline_y = win_steps[0]["y"] if win_steps else ys[0]
    incumbent_y = ymax_data
    incumbent_x = max((st["x"] for st in win_steps if st["y"] == incumbent_y),
                      default=xs[-1])

    plt.rcParams.update(THEME)
    has_mfu = any(r.mfu >= 0 for r in kept)

    fig = plt.figure(figsize=(15, 8.2))
    if has_mfu:
        gs = GridSpec(2, 1, height_ratios=[5.2, 1.0], hspace=0.08, figure=fig)
        ax = fig.add_subplot(gs[0])
        axm = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = fig.add_subplot(1, 1, 1)
        axm = None
    fig.subplots_adjust(top=0.83, bottom=0.16, left=0.075, right=0.975)

    # -- title + conditions --------------------------------------------------
    fig.suptitle(
        f"{model} — Optimization Trajectory",
        fontsize=20, fontweight="bold", color=INK, x=0.075, ha="left", y=0.965,
    )
    min_corr = min((r.correctness for r in kept), default=0.0)
    cond = "   ·   ".join(
        p for p in (
            hardware,
            f"TP={tp}" if tp else "",
            shape,
            f"SDK {sdk}" if sdk else "",
            f"correctness ≥ {min_corr:.3g}% on every kept step",
        ) if p
    )
    fig.text(0.075, 0.905, cond, ha="left", fontsize=11.5, color=SUBTLE_INK)

    # -- headroom above the data for band labels + step labels ---------------
    ymax = ymax_data * 1.42
    x_right = bands[-1]["x1"]
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.15, x_right + 0.15)

    band_label_y = 0.965       # axes fraction — stage-name row along the top
    band_note_y = 0.905

    # -- stage bands: shaded, labelled, spanning the whole pipeline ----------
    # No-win stages get a wider slot (see _pipeline_bands) so their header
    # never collides with the neighbouring stage's.
    xtrans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for i, b in enumerate(bands):
        left, right = b["x0"], b["x1"]
        tint = BAND_WIN_TINT if b["has_win"] else BAND_NOWIN_TINT
        ax.axvspan(left, right, color=tint, zorder=0)
        if i > 0:
            ax.axvline(left, color=BAND_EDGE, linewidth=1.1, zorder=1)
        cx = (left + right) / 2
        num, label = b["number"], b["label"]
        head = f"{num}   {label}" if num else label
        ax.text(cx, band_label_y, head, transform=xtrans,
                ha="center", va="top", fontsize=12, fontweight="bold",
                color=INK if b["has_win"] else SUBTLE_INK)
        if not b["has_win"]:
            ax.text(cx, band_note_y, "walked\nno gain over config",
                    transform=xtrans, ha="center", va="top", fontsize=8.5,
                    color="#93a0ad", style="italic", linespacing=1.1)

    # -- discarded candidates, faint (search width, without the clutter) -----
    rng = np.random.default_rng(7)
    for b in bands:
        stage_disc = [r for r in discarded
                      if r.stage is b["stage"] and r.metric > 0]
        if not stage_disc:
            continue
        span = max(0.28, (b["x1"] - b["x0"]) * 0.5)
        for r in stage_disc:
            jx = (b["x0"] + b["x1"]) / 2 + rng.uniform(-span, span)
            ax.scatter(jx, r.metric, marker="x", s=34, linewidths=1.1,
                       color=DISCARD_GRAY, alpha=0.45, zorder=2)

    # -- the winning path ----------------------------------------------------
    # Solid accent line over the gain-producing steps; a lighter dashed line
    # continues flat through the no-win stages to show the incumbent "held".
    solid_x = [st["x"] for st in steps if st["x"] <= incumbent_x]
    solid_y = [st["y"] for st in steps if st["x"] <= incumbent_x]
    ax.plot(solid_x, solid_y, color=ACCENT, linewidth=3.0, zorder=4,
            solid_capstyle="round", solid_joinstyle="round")
    ax.fill_between(solid_x, solid_y, color=ACCENT, alpha=0.06, zorder=1)
    if incumbent_x < xs[-1]:
        held_x = [st["x"] for st in steps if st["x"] >= incumbent_x]
        held_y = [st["y"] for st in steps if st["x"] >= incumbent_x]
        ax.plot(held_x, held_y, color=HELD_GRAY, linewidth=2.0,
                linestyle=(0, (5, 3)), zorder=3)

    # kept-step dots
    for st in win_steps:
        ax.scatter(st["x"], st["y"], s=150, color=ACCENT, edgecolors="white",
                   linewidths=1.8, zorder=6)
    # flat placeholder markers for no-win stages: hollow ring on the held line
    for st in steps:
        if not st["is_win"]:
            ax.scatter(st["x"], st["y"], s=70, facecolors="white",
                       edgecolors=HELD_GRAY, linewidths=1.6, zorder=5)

    # -- per-step gain + idea labels -----------------------------------------
    prev_y = None
    for idx, st in enumerate(steps):
        if not st["is_win"]:
            prev_y = st["y"]
            continue
        r = st["row"]
        if prev_y is None:
            # baseline sits near the axis floor: label above-right of the dot
            # so it is never clipped by the bottom margin / MFU strip.
            ax.annotate(
                f"baseline\n{st['y']:,.0f} {metric_label}",
                xy=(st["x"], st["y"]), xytext=(14, 24),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, color=SUBTLE_INK, linespacing=1.2,
            )
        else:
            pct = (st["y"] / prev_y - 1.0) * 100.0 if prev_y > 0 else 0.0
            sign = "+" if pct >= 0 else ""
            ax.annotate(
                f"{sign}{pct:.0f}%",
                xy=(st["x"], st["y"]), xytext=(0, 30),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=GAIN_GREEN,
            )
            ax.annotate(
                _idea_short(r),
                xy=(st["x"], st["y"]), xytext=(0, 15),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9.5, color=INK,
            )
        prev_y = st["y"]

    # -- headline badge on the incumbent -------------------------------------
    speedup = led.speedup() or (incumbent_y / baseline_y if baseline_y > 0 else 0.0)
    headline = f"{incumbent_y:,.0f} {metric_label}"
    if speedup:
        headline += f"\n{speedup:.1f}× baseline"
    ax.annotate(
        headline,
        xy=(incumbent_x, incumbent_y),
        xytext=(0.985, 0.60), textcoords="axes fraction",
        ha="right", va="center", fontsize=15, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=ACCENT_DK,
                  edgecolor="none"),
        arrowprops=dict(arrowstyle="-", color=ACCENT_DK, lw=1.4, alpha=0.7,
                        shrinkB=10),
        zorder=8,
    )

    # -- roofline ceiling (optional) -----------------------------------------
    if roofline:
        ax.axhline(roofline, color="#b0472b", linestyle=":", linewidth=1.6,
                   alpha=0.8, zorder=2)
        ax.text(n - 0.5, roofline, f"  roofline · {incumbent_y / roofline * 100:.0f}% attained",
                fontsize=9.5, color="#b0472b", va="bottom", ha="right")

    # -- primary axis cosmetics ----------------------------------------------
    ax.set_ylabel(f"throughput ({metric_label})", fontsize=12.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.grid(True, axis="y", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if has_mfu:
        _blank_step_axis(ax, steps, xlabel=False)
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        _blank_step_axis(ax, steps, xlabel=True)

    # -- MFU strip below -----------------------------------------------------
    if has_mfu:
        m_y = []
        cur = 0.0
        for st in steps:
            if st["is_win"] and st["row"].mfu >= 0:
                cur = st["row"].mfu
            m_y.append(cur)
        axm.fill_between(xs, m_y, color="#8a94a3", alpha=0.18, zorder=1)
        axm.plot(xs, m_y, color="#6b7684", linewidth=1.8, zorder=2)
        axm.scatter([st["x"] for st in win_steps],
                    [st["row"].mfu if st["row"].mfu >= 0 else 0 for st in win_steps],
                    s=28, color="#6b7684", zorder=3)
        axm.set_ylabel("MFU %", fontsize=10, color="#6b7684")
        axm.set_ylim(0, max(m_y) * 1.5 if max(m_y) > 0 else 1)
        axm.tick_params(axis="y", labelsize=8.5, colors="#6b7684")
        axm.grid(True, axis="y", alpha=0.7)
        axm.set_axisbelow(True)
        for side in ("top", "right"):
            axm.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axm.spines[side].set_color("#c9d0d8")
        _blank_step_axis(axm, steps, xlabel=True)

    # -- legend --------------------------------------------------------------
    handles = [
        Line2D([], [], color=ACCENT, linewidth=3, marker="o", markersize=9,
               markerfacecolor=ACCENT, markeredgecolor="white",
               label="kept step (promoted)"),
        Line2D([], [], color=HELD_GRAY, linewidth=2, linestyle=(0, (5, 3)),
               marker="o", markersize=8, markerfacecolor="white",
               markeredgecolor=HELD_GRAY, label="incumbent held (no gain)"),
        Line2D([], [], color=DISCARD_GRAY, marker="x", linestyle="none",
               markersize=8, label="discarded candidate"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10.5, frameon=True,
              facecolor="white", edgecolor="#d5dbe2", framealpha=0.95,
              borderpad=0.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path


def _blank_step_axis(axis, steps: list[dict], xlabel: bool = False) -> None:
    """Blank the per-step x ticks — the idea for each kept step is already
    labelled above its dot, and the stage bands label the groups, so a second
    set of tick labels would just re-clutter the axis. Optionally add a single
    clean axis caption naming what the x-axis is."""
    axis.set_xticks([st["x"] for st in steps])
    axis.set_xticklabels([""] * len(steps))
    axis.tick_params(axis="x", length=0)
    if xlabel:
        axis.set_xlabel(
            "optimization steps  (left → right = the order the optimizer tried them)",
            fontsize=11, color=SUBTLE_INK,
        )



def _nearest_x_for_stage(kept: list[Row], stage: Stage) -> int | None:
    """X position of the last kept row in this stage, for placing failures."""
    idxs = [i for i, r in enumerate(kept) if r.stage is stage]
    return idxs[-1] if idxs else None


def _mode_signal_present(rows: list[Row]) -> bool:
    """True if this run has any compile-mode signal at all. Keeps the eager/
    compiled labeling native-PyTorch-specific — mock/XLA runs (no compile_mode
    axis) get no labels and render exactly as before."""
    for r in rows:
        d = r.description.lower()
        if "compile_mode=" in d or "compile-default" in d or "eager" in d:
            return True
    return False


def _row_mode(desc: str) -> str | None:
    """Explicit mode from a row's provenance description, or None if it doesn't
    change the mode (e.g. a tp_degree=8 delta inherits the current mode)."""
    d = desc.lower()
    if "compile_mode=compile-default" in d or "compile-default" in d:
        return "compiled"
    if "compile_mode=eager" in d:
        return "eager"
    return None


def _classify_modes(kept: list[Row]) -> list[str]:
    """Label every kept point eager/compiled by carrying the mode forward along
    the winning path. Baseline is eager (the native backend's naive start);
    the mode flips only when a compile_mode candidate is promoted, and later
    points (tp/dtype changes) inherit whatever mode was in effect."""
    modes: list[str] = []
    current = "eager"                      # native baseline is eager
    for r in kept:
        explicit = _row_mode(r.description)
        if explicit is not None:
            current = explicit
        elif r.description.strip().lower() == "baseline":
            current = "eager"
        elif r.compile_s >= COMPILE_S_HEURISTIC and _row_mode(r.description) is None:
            # weak fallback: a minutes-long first forward implies a NEFF compile
            current = "compiled"
        modes.append(current)
    return modes


def _short_label(desc: str, width: int = 22) -> str:
    """Two-line compact axis labels, in the style of the reference chart."""
    if len(desc) <= width:
        return desc
    words, lines, cur = desc.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
            if len(lines) == 2:
                break
        else:
            cur = f"{cur} {w}".strip()
    if cur and len(lines) < 2:
        lines.append(cur)
    return "\n".join(lines)


# -- highlights chart (kept-only staircase, the "presentation" view) --------

# Stage labels for the top-of-plot annotations, styled after wutong's
# "Round 1 (Params)". Renamed to our stage vocabulary so the two vocabularies
# don't drift.
_STAGE_LABEL: dict[Stage, str] = {
    Stage.BASELINE: "Baseline",
    Stage.HARVEST: "Stage 0.5\n(Harvest)",
    Stage.CONFIG: "Stage 1\n(Config)",
    Stage.KNOWN_KERNEL: "Stage 2\n(Known Kernels)",
    Stage.BORROW: "Stage 3\n(Borrow)",
    Stage.INVENT: "Stage 4\n(Invent)",
    Stage.GRAPH_REWRITE: "Stage 5\n(Graph Rewrite)",
    Stage.PROFILE_LOOP: "Stage 6\n(Profile Loop)",
}


def _stage_label(stage: Stage) -> str:
    return _STAGE_LABEL.get(stage, stage.value)


# Row.description -> prose idea. Explicit map for the common axis=value combos
# so the staircase reads as ideas ("torch.compile", "NKI flash-attention"), not
# parameter names ("compile_mode=compile-default"). Falls back to a light
# pretty-print for anything unmapped so new axes never crash the chart.
_IDEA_MAP: dict[str, str] = {
    "compile_mode=compile-default": "torch.compile",
    "compile_mode=compile-max": "torch.compile\n(max-autotune)",
    "compile_mode=compile-reduce-overhead": "torch.compile\n(reduce-overhead)",
    "compile_mode=eager": "Eager mode",
    "attn_implementation=sdpa": "SDPA attention",
    "attn_implementation=flash": "Flash attention",
    "attn_implementation=eager_math": "Math attention",
    "attention_kernel=nki_flash": "NKI\nflash-attention",
    "attention_kernel=paged": "Paged attention",
    "attention_kernel=kv_parallel": "KV-parallel\nattention",
    "weights_dtype=bf16": "BF16 weights",
    "weights_dtype=fp8": "FP8 weights",
    "weights_dtype=int8": "INT8 weights",
    "kv_cache_dtype=bf16": "BF16 KV cache",
    "kv_cache_dtype=fp8": "FP8 KV cache",
    "batching=paged": "Paged batching",
    "batching=continuous": "Continuous batching",
    "batching=static": "Static batching",
    "sequence_layout=bshd": "BSHD layout",
    "sequence_layout=sbhd": "SBHD layout",
    "track=latency": "Latency track\n(dp=1)",
    "track=throughput": "Throughput track",
}

_PRETTY_AXIS: dict[str, str] = {
    "tp_degree": "TP",
    "cp_degree": "CP",
    "dp_degree": "DP",
    "kv_replication": "KV-rep",
}


def _ideafy(row: Row) -> str:
    """A row's provenance rendered as a short prose idea for the x-axis of the
    highlights chart. Explicit map first, then light pretty-print, then a
    stage/origin fallback for the kernel stages."""
    # Drop the under-utilization annotation the orchestrator may have appended;
    # it belongs on the busy chart, not the presentation one.
    desc = row.description.split("[under-util:")[0].strip()
    if not desc or desc.lower() == "baseline":
        return "Baseline"
    if desc in _IDEA_MAP:
        return _IDEA_MAP[desc]
    # bank prior:lesson-id — show the lesson id, truncated
    if desc.startswith("prior:"):
        lid = desc.split(":", 1)[1]
        return f"Bank prior:\n{lid[:22]}"
    # single axis=value not in the map: pretty-print the axis
    if "=" in desc:
        axis, value = desc.split("=", 1)
        return f"{_PRETTY_AXIS.get(axis, axis.replace('_', ' '))}={value}"
    # stage/origin fallback for the kernel stages
    if row.stage is Stage.BORROW and row.source:
        return f"Borrowed:\n{row.source.split('@')[0][:20]}"
    if row.stage is Stage.INVENT:
        return "Invented\nkernel"
    if row.stage is Stage.KNOWN_KERNEL and row.origin is Origin.HARVESTED:
        return "nkilib kernel"
    if row.stage is Stage.HARVEST:
        return "Harvested\nfrom corpus"
    return desc[:22]


def build_highlights_chart(
    run_dir: Path,
    out_path: Path,
    model: str,
    hardware: str = "",
    tp: int | None = None,
    shape: str = "",
    sdk: str = "",
    metric_label: str = "tok/s",
) -> Path:
    """Wutong-style kept-path staircase — the "presentation" chart.

    Complementary to build_chart: that one shows every attempt with stage
    colors, provenance markers, MFU, and discards — the engineer view. This
    one shows only the *story*: the winning path stepped up, labelled with
    the idea (not the axis value), stage sections divided by dashed lines
    with big prose labels on top, and a giant final Nx callout in red.

    Modeled on wutongabc/auto_research_for_AWS_Neuron_optimization's
    "Every Optimization Step" chart. The 5-6× compile-mode cliff on Qwen3-8B
    is exactly the kind of jump this view was designed to sell.
    """
    led = Ledger(run_dir)
    rows = led.read()
    if not rows:
        raise SystemExit(f"ledger is empty: {led.path}")
    kept = [r for r in rows if r.kept]
    if not kept:
        raise SystemExit("no kept rows — nothing to plot")

    plt.rcParams.update(THEME)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    # top pad leaves room for the stage-name row; bottom pad for prose labels.
    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.08, right=0.94)

    fig.suptitle(
        f"{model}: Every Optimization Step",
        fontsize=17, fontweight="bold", color=INK, y=0.965,
    )
    cond = " │ ".join(
        p for p in (
            hardware,
            f"TP={tp}" if tp else "",
            shape,
            f"SDK {sdk}" if sdk else "",
        ) if p
    )
    if cond:
        fig.text(0.5, 0.895, cond, ha="center", fontsize=10, color=SUBTLE_INK)

    xs = list(range(len(kept)))
    ys = [r.metric for r in kept]

    # -- the staircase itself ------------------------------------------------
    # steps-post = horizontal-then-vertical: each improvement "holds" until
    # the next win. This is the visual signature of Wutong's chart.
    ax.plot(xs, ys, drawstyle="steps-post", color=ACCENT,
            linewidth=2.5, alpha=0.95, zorder=2)
    ax.scatter(xs, ys, s=80, color=ACCENT,
               edgecolors="white", linewidths=1.4, zorder=3)

    # -- per-point tok/s values (blue text, always above the dot) -----------
    # Alternating above/below sounds nice but breaks the moment two adjacent
    # points are close in y (the classic small-config-win-then-another-small
    # -config-win pattern) — an above label of point i then collides with the
    # below label of point i+1. Consistent "above" keeps every label at the
    # same offset relative to its dot, so labels never invade each other's
    # airspace. The ylim below gives them room.
    ymax = max(ys)
    for i, r in enumerate(kept):
        ax.annotate(
            f"{r.metric:,.0f}",
            xy=(i, r.metric),
            xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold", color=ACCENT_DK,
        )

    # -- stage dividers + labels at the top (wutong's "Round 1 (Params)") ---
    # The plot has three horizontal bands stacked above the data:
    #   data (staircase, incl. above-dot value labels)  ->  ends near ymax*1.02
    #   speedup callout band                            ->  around ymax*1.20
    #   stage-name row                                  ->  around ymax*1.45 (top_y)
    #   headroom (ylim caps at ymax * 1.60)
    # Any two of these overlap the moment top_y is squeezed. Keep them separate.
    bands = _stage_bands(kept)
    top_y = ymax * 1.45
    for i, (xpos, stage) in enumerate(bands):
        # Vertical dashed line between stages — never at the plot's left edge.
        if i > 0:
            ax.axvline(xpos, color=BAND_EDGE, linestyle="--",
                       linewidth=1.2, alpha=0.9)
        # Centered stage label above the band it names.
        next_xpos = bands[i + 1][0] if i + 1 < len(bands) else len(kept) - 0.5
        ax.text(
            (xpos + next_xpos) / 2, top_y,
            _stage_label(stage),
            fontsize=10.5, color=SUBTLE_INK, ha="center", va="top",
            style="italic",
        )

    # -- giant final Nx callout ---------------------------------------------
    # Anchored to the top-right axes corner so it never collides with the
    # last data point's tok/s label (they used to fight for the same pixels).
    # A drawn arrow points from the callout down to the last data point.
    speedup = led.speedup() or (ys[-1] / ys[0] if ys[0] > 0 else 0.0)
    if speedup and speedup > 1.05:
        # Callout in its own middle band — clear of the last data point (at
        # axes-frac ~0.63 with ylim=ymax*1.60) AND the stage-name row (top_y at
        # axes-frac ~0.91). shrinkB stops the arrow short of the last point so
        # it doesn't spear the "N,NNN" value label above it.
        ax.annotate(
            f"{speedup:.1f}×",
            xy=(len(kept) - 1, ys[-1]),
            xytext=(0.985, 0.78),
            xycoords="data", textcoords="axes fraction",
            fontsize=26, fontweight="bold", color="#f85149",
            ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color="#f85149", lw=2.0,
                            connectionstyle="arc3,rad=0", shrinkB=18),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_ideafy(r) for r in kept],
        fontsize=9, rotation=38, ha="right",
    )
    ax.set_ylabel(
        f"{metric_label}" + (f"  ({shape})" if shape else ""),
        fontsize=11,
    )
    # ylim upper is generous on purpose: it creates the three-band vertical
    # layout (data / callout / stage names) that keeps them from fighting.
    ax.set_ylim(0, ymax * 1.60)
    ax.set_xlim(-0.6, len(kept) - 0.4)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.grid(True, axis="y", alpha=0.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--shape", default="")
    ap.add_argument("--sdk", default="")
    ap.add_argument("--roofline", type=float, default=None)
    ap.add_argument("--metric-label", default="tok/s")
    ap.add_argument("--style", choices=("detailed", "highlights"), default="detailed",
                    help="detailed = every attempt (engineer view); "
                         "highlights = kept-only staircase (presentation view)")
    a = ap.parse_args()

    if a.style == "highlights":
        out = a.out or (a.run_dir / "optimization_highlights.png")
        p = build_highlights_chart(
            run_dir=a.run_dir, out_path=out, model=a.model, hardware=a.hardware,
            tp=a.tp, shape=a.shape, sdk=a.sdk, metric_label=a.metric_label,
        )
    else:
        out = a.out or (a.run_dir / "optimization_timeline.png")
        p = build_chart(
            run_dir=a.run_dir, out_path=out, model=a.model, hardware=a.hardware,
            tp=a.tp, shape=a.shape, sdk=a.sdk, roofline=a.roofline,
            metric_label=a.metric_label,
        )
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
