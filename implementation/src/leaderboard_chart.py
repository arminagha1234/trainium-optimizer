"""
Leaderboard chart — cross-model baseline vs optimized, one bar pair per model.

Complements LEADERBOARD.md (which is a table) with the *picture* — the shape
of wutongabc/auto_research_for_AWS_Neuron_optimization's
`speedup_comparison.png`, kept in our dark theme so it sits next to the
trajectory chart without a visual jolt.

Written each cycle by overnight.py so the artifact refreshes as more cycles
complete. Failed models are annotated in place rather than dropped — the
absence of a bar is a data point too.

Usage:
    from leaderboard_chart import build_leaderboard_chart
    build_leaderboard_chart(results, out_path, backend="native-pytorch-beta3")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


THEME = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "text.color": "#c9d1d9",
    "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "font.family": "DejaVu Sans",
    "font.size": 10,
}

BASELINE_COLOR = "#6e7681"     # muted grey — same as the trajectory chart's baseline
OPTIMIZED_COLOR = "#58a6ff"    # blue — matches the highlights staircase
FAIL_COLOR = "#f85149"


class _Result(Protocol):
    """Structural type — anything with these attrs works. Kept a Protocol so
    this module has zero import coupling with overnight.py (avoids a cycle)."""
    slug: str
    ok: bool
    baseline: float
    best: float
    speedup: float
    error: str


@dataclass(frozen=True)
class SimpleResult:
    """Adapter for callers that don't already have a compatible object (tests,
    scripts). Matches overnight.ModelResult's shape exactly."""
    slug: str
    ok: bool
    baseline: float = 0.0
    best: float = 0.0
    speedup: float = 0.0
    error: str = ""


def _fmt_thousands(v: float, _pos: int | None = None) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}K"
    return f"{int(v)}"


def _wrap_short(text: str, width: int, max_lines: int) -> str:
    """Word-wrap text into at most `max_lines` lines of ~`width` chars each.
    Ellipsizes the last line when the input is longer than the budget so a
    truncation is *visible* rather than silent — which mattered on the first
    render, where "invalid_tp: kv_heads=4 (needs KV replication)" was cut off
    at "needs" with no signal that more text existed."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    truncated = False
    for w in words:
        # `> width`, not `>= width`, so a line that exactly hits width stays.
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = ""
            if len(lines) >= max_lines:
                truncated = True
                break
        candidate = f"{cur} {w}".strip()
        if len(candidate) > width:      # a single word that alone exceeds width
            if cur:
                lines.append(cur)
            if len(lines) >= max_lines:
                truncated = True
                break
            cur = w[:width - 1] + "…"
            truncated = True
            continue
        cur = candidate
    if cur:
        if len(lines) < max_lines:
            lines.append(cur)
        else:
            truncated = True
    if truncated and lines:
        last = lines[-1]
        if not last.endswith("…"):
            lines[-1] = (last + " …") if len(last) + 2 <= width else last[:width - 1] + "…"
    return "\n".join(lines[:max_lines])


def build_leaderboard_chart(
    results: list[_Result],
    out_path: Path,
    backend: str = "",
    sdk: str = "",
    metric_label: str = "prefill tok/s",
    cycle: int | None = None,
) -> Path:
    """Baseline vs optimized bars, one pair per model, speedup labelled above.

    Zero results is not an error — the chart just renders a "no data yet"
    placeholder so callers can invoke it unconditionally after every cycle.
    """
    plt.rcParams.update(THEME)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.3 * max(1, len(results)) + 3), 6.2))
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.08, right=0.96)

    # -- title + conditions --------------------------------------------------
    fig.suptitle(
        "Leaderboard: Baseline vs Optimized (per model)",
        fontsize=16, fontweight="bold", color="#f0f6fc", y=0.965,
    )
    cond = " │ ".join(
        p for p in (
            backend,
            f"SDK {sdk}" if sdk else "",
            f"cycle {cycle}" if cycle else "",
        ) if p
    )
    if cond:
        fig.text(0.5, 0.905, cond, ha="center", fontsize=9.5, color="#8b949e")

    if not results:
        ax.text(0.5, 0.5, "no results yet — run at least one model",
                ha="center", va="center", fontsize=12, color="#8b949e",
                transform=ax.transAxes)
        ax.set_axis_off()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # -- bars ---------------------------------------------------------------
    slugs = [r.slug for r in results]
    xs = list(range(len(results)))
    width = 0.36
    baselines = [r.baseline if r.ok else 0.0 for r in results]
    bests = [r.best if r.ok else 0.0 for r in results]

    ax.bar([x - width / 2 for x in xs], baselines, width,
           label="Baseline", color=BASELINE_COLOR, edgecolor="#0d1117",
           linewidth=0.5, zorder=2)
    ax.bar([x + width / 2 for x in xs], bests, width,
           label="Optimized", color=OPTIMIZED_COLOR, edgecolor="#0d1117",
           linewidth=0.5, zorder=2)

    # -- per-model annotations (value on each bar; speedup above the pair) --
    ymax = max([*bests, *baselines, 1.0])
    for i, r in enumerate(results):
        if not r.ok:
            # Failed model: red X in place of bars + short reason underneath.
            # Wrap long reasons to 2 lines so they don't run off the axis on
            # the rightmost model (where ha="center" truncates at the plot
            # edge). Keep it short — this is a picture, the .md has the full
            # error.
            ax.text(i, ymax * 0.35, "×", fontsize=42, color=FAIL_COLOR,
                    ha="center", va="center", fontweight="bold")
            reason = _wrap_short(r.error or "failed", width=20, max_lines=3)
            ax.text(i, ymax * 0.18, reason, fontsize=7.5, color=FAIL_COLOR,
                    ha="center", va="center", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#f8514912", edgecolor="#f8514940"))
            continue

        # Values inside each bar (white if there's room, muted below).
        if r.baseline > ymax * 0.08:
            ax.text(i - width / 2, r.baseline / 2, f"{r.baseline:,.0f}",
                    ha="center", va="center", fontsize=8.5, color="#c9d1d9",
                    fontweight="bold")
        else:
            ax.text(i - width / 2, r.baseline + ymax * 0.015, f"{r.baseline:,.0f}",
                    ha="center", va="bottom", fontsize=8.5, color=BASELINE_COLOR)

        if r.best > ymax * 0.08:
            ax.text(i + width / 2, r.best / 2, f"{r.best:,.0f}",
                    ha="center", va="center", fontsize=8.5, color="#0d1117",
                    fontweight="bold")
        else:
            ax.text(i + width / 2, r.best + ymax * 0.015, f"{r.best:,.0f}",
                    ha="center", va="bottom", fontsize=8.5, color=OPTIMIZED_COLOR)

        # Speedup label above the taller bar. Bold red when >= 2x, else muted.
        if r.speedup > 1.02:
            top = max(r.baseline, r.best)
            big = r.speedup >= 2.0
            ax.text(i, top + ymax * 0.055,
                    f"{r.speedup:.1f}×",
                    ha="center", va="bottom",
                    fontsize=14 if big else 11,
                    fontweight="bold",
                    color=FAIL_COLOR if big else "#8b949e")

    ax.set_xticks(xs)
    ax.set_xticklabels(slugs, fontsize=9.5, rotation=18, ha="right")
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_ylim(0, ymax * 1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    ax.grid(True, axis="y", alpha=0.35)
    # Upper right: the corner over the last model. In the throughput picture
    # the rightmost slot is either the largest speedup (already annotated) or
    # a failed model (whose × sits low), so this corner is the safest anchor.
    ax.legend(loc="upper right", fontsize=9, frameon=True,
              facecolor="#161b22", edgecolor="#30363d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
