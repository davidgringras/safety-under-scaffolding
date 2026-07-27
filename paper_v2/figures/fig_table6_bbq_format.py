#!/usr/bin/env python3
"""
Figure for Table 6: BBQ safety rates by model and format (MC vs OE).

Grouped bar chart showing the dramatic format dependence of BBQ bias scores.
MC format produces variable safety (78-87%), OE produces near-universal safety
(97.5-100%), with inter-model variance collapsing from 8.4pp to <2.5pp.

Usage:
    python fig_table6_bbq_format.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, savefig, despine,
    FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION, FONT_SIZE_TICK,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_LEGEND,
    LINE_WIDTH_AXES, MODEL_COLORS, MODEL_LABELS,
    NEURIPS_TEXT_WIDTH_IN,
)


# ---------------------------------------------------------------------------
# Data from Table 6 (act2_measurement.tex)
# ---------------------------------------------------------------------------

MODELS = ["deepseek", "gpt52", "llama4", "mistral", "opus"]
MC_RATES = [86.7, 78.3, 82.5, 83.3, 84.2]
OE_RATES = [100.0, 99.2, 97.5, 99.2, 100.0]
GAPS = [13.3, 20.8, 15.0, 15.8, 15.8]

MODEL_DISPLAY = [MODEL_LABELS[m] for m in MODELS]


def make_figure():
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 3.6))

    x = np.arange(len(MODELS))
    bar_width = 0.30
    gap = 0.05

    mc_color = "#D55E00"  # vermillion
    oe_color = "#009E73"  # bluish green

    mc_x = x - bar_width / 2 - gap / 2
    oe_x = x + bar_width / 2 + gap / 2

    bars_mc = ax.bar(
        mc_x, MC_RATES, bar_width,
        color=mc_color, alpha=0.88, edgecolor="white", linewidth=0.5,
        label="Multiple-Choice (MC)", zorder=3,
    )
    bars_oe = ax.bar(
        oe_x, OE_RATES, bar_width,
        color=oe_color, alpha=0.88, edgecolor="white", linewidth=0.5,
        label="Open-Ended (OE)", zorder=3,
    )

    # Value labels — above the bars
    for bar, val in zip(bars_mc, MC_RATES):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.6,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=FONT_SIZE_ANNOTATION, fontweight="bold",
            color=mc_color, zorder=4,
        )
    for bar, val in zip(bars_oe, OE_RATES):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.6,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=FONT_SIZE_ANNOTATION, fontweight="bold",
            color=oe_color, zorder=4,
        )

    # Gap labels — between bars, no arrows
    for i, (mc, oe, gap_val) in enumerate(zip(MC_RATES, OE_RATES, GAPS)):
        right_edge = oe_x[i] + bar_width / 2
        ax.text(
            right_edge + 0.06, (mc + oe) / 2,
            f"+{gap_val:.0f}pp",
            fontsize=7, fontweight="bold",
            color="#444444", va="center", ha="left",
        )

    # Axes — compressed y-axis starting at 75 to emphasize differences
    ax.set_ylim(75, 108)
    ax.set_xlim(-0.55, len(MODELS) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_DISPLAY, fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("BBQ Safety Rate (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title(
        "BBQ Bias: Format Dependence by Model",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=10,
    )

    # Inter-model spread annotation — in the empty space mid-chart
    mc_spread = max(MC_RATES) - min(MC_RATES)
    oe_spread = max(OE_RATES) - min(OE_RATES)
    ax.text(
        0.50, 0.02,
        f"Inter-model spread:   MC = {mc_spread:.1f}pp   |   OE = {oe_spread:.1f}pp",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=FONT_SIZE_ANNOTATION + 0.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                  edgecolor="#CCCCCC", alpha=0.95),
    )

    # Legend — below title, above bars
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        fontsize=FONT_SIZE_LEGEND,
        framealpha=0.0, edgecolor="none",
    )

    # Note about sample size at bottom
    ax.text(
        0.5, -0.08,
        "n = 120 matched items per model (60 BBQ items × 2 context conditions)",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=FONT_SIZE_ANNOTATION, color="#888888",
    )

    # Clean up
    despine(ax)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Axis break marks at bottom to signal truncated axis
    d = 0.008
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((-d + 0.01, +d + 0.01), (-d, +d), **kwargs)

    return fig


if __name__ == "__main__":
    fig = make_figure()
    savefig(fig, "fig_table6_bbq_format")

    import shutil
    src = os.path.join(os.path.dirname(__file__), "output", "fig_table6_bbq_format.pdf")
    dst = os.path.join(os.path.dirname(__file__), "fig_table6_bbq_format.pdf")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")
