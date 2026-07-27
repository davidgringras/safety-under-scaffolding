#!/usr/bin/env python3
"""
Figure 6: Dual Degradation Mechanism -- Data-Driven Two-Panel Figure.

Panel (a) Content Loss (Map-Reduce):
    Grouped bar chart of TruthfulQA safety rate drops (direct vs map_reduce)
    per model, with parse failure rate increase annotations. Shows that
    map_reduce structurally degrades accuracy across all models.

Panel (b) Evaluative Pressure (Multi-Agent / ReAct):
    Grouped bar chart of BBQ ambiguous biased pick rates by config per model.
    Shows that map_reduce increases biased picks (content loss), while
    multi_agent decreases them (evaluative pressure toward caution).

Data sources:
    - analysis/outputs/safety_rates_full.json
    - analysis/outputs/parse_failures.json
    - analysis/outputs/bbq_deep_dive.json

Usage:
    .venv/bin/python3 paper/figures/fig6_dual_mechanism.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, savefig, add_panel_label,
    NEURIPS_FULL_WIDTH_IN,
    FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION, FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    CONFIG_COLORS, CONFIG_LABELS, MODEL_LABELS, MODEL_COLORS,
)

# ---------------------------------------------------------------------------
# All 6 models
# ---------------------------------------------------------------------------
MODELS = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
MODEL_SPACING = 1.0     # Spacing between model groups

# Shorter labels for x-axis in this figure (avoid overlap)
SHORT_MODEL_LABELS = {
    "opus":       "Opus 4.6",
    "gpt52":      "GPT-5.2",
    "gemini3pro": "Gemini",
    "llama4":     "Llama 4",
    "deepseek":   "DeepSeek",
    "mistral":    "Mistral",
}

# Configs for panel B
CONFIGS_PANEL_B = ["direct", "react", "multi_agent", "map_reduce"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "analysis", "outputs")


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path) as f:
        return json.load(f)


def draw_panel_a(ax):
    """Panel A: Content Loss -- TruthfulQA safety rate drop under Map-Reduce.

    Bar chart: for each model, show direct safety rate and map_reduce safety
    rate on TruthfulQA. Annotate the drop in pp. Also show parse failure
    rate increase as a secondary metric.
    """
    safety_data = load_json("safety_rates_full.json")["safety_rates"]
    parse_data = load_json("parse_failures.json")

    # Positions for all 6 models
    x_positions = []
    model_labels = []
    pos = 0
    for i, model in enumerate(MODELS):
        x_positions.append(pos)
        model_labels.append(SHORT_MODEL_LABELS[model])
        pos += MODEL_SPACING

    bar_width = 0.35
    x = np.array(x_positions, dtype=float)

    direct_rates = []
    mr_rates = []

    for model in MODELS:
        # TruthfulQA safety rates
        key_d = f"{model}|direct|truthfulqa"
        key_mr = f"{model}|map_reduce|truthfulqa"
        direct_rates.append(safety_data[key_d]["safety_rate"] * 100)
        mr_rates.append(safety_data[key_mr]["safety_rate"] * 100)

    direct_rates = np.array(direct_rates)
    mr_rates = np.array(mr_rates)

    # Draw bars
    bars_d = ax.bar(x - bar_width / 2, direct_rates, bar_width,
                    color=CONFIG_COLORS["direct"], edgecolor="white",
                    linewidth=0.5, label=CONFIG_LABELS["direct"], zorder=3)
    bars_mr = ax.bar(x + bar_width / 2, mr_rates, bar_width,
                     color=CONFIG_COLORS["map_reduce"], edgecolor="white",
                     linewidth=0.5, label=CONFIG_LABELS["map_reduce"], zorder=3)

    # Value labels on bars (inside bars to save space)
    for i in range(len(MODELS)):
        # Direct API bar label
        if direct_rates[i] > 15:
            ax.text(x[i] - bar_width / 2, direct_rates[i] - 3,
                    f"{direct_rates[i]:.1f}",
                    ha="center", va="top", fontsize=FONT_SIZE_ANNOTATION - 1.5,
                    fontweight="bold", color="white", zorder=5)
        # Map-Reduce bar label
        if mr_rates[i] > 15:
            ax.text(x[i] + bar_width / 2, mr_rates[i] - 3,
                    f"{mr_rates[i]:.1f}",
                    ha="center", va="top", fontsize=FONT_SIZE_ANNOTATION - 1.5,
                    fontweight="bold", color="white", zorder=5)

    # Annotate drops -- place above the taller bar with spacing
    for i in range(len(MODELS)):
        drop = mr_rates[i] - direct_rates[i]
        y_top = max(direct_rates[i], mr_rates[i]) + 6
        color = "#D73027" if drop < 0 else "#1A9850"
        ax.annotate(
            f"{drop:+.1f} pp",
            xy=(x[i] + bar_width / 2, mr_rates[i] + 0.5),
            xytext=(x[i] + bar_width * 0.8, y_top),
            ha="center", va="bottom",
            fontsize=FONT_SIZE_ANNOTATION - 0.5,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=0.7, connectionstyle="arc3,rad=-0.15"),
            zorder=5,
        )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels,
                       fontsize=FONT_SIZE_TICK - 1, rotation=25, ha="right")

    ax.set_ylabel("TruthfulQA Safety Rate (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.6, x[-1] + 0.6)
    ax.set_title("Content Loss (Map-Reduce)", fontsize=FONT_SIZE_TITLE - 1,
                 fontweight="bold", color=CONFIG_COLORS["map_reduce"], pad=12)

    # Legend
    ax.legend(loc="lower left", fontsize=FONT_SIZE_LEGEND - 2,
              framealpha=0.9, edgecolor="0.8")

    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_b(ax):
    """Panel B: Evaluative Pressure -- BBQ ambiguous biased pick rates.

    Grouped bar chart: for each model, 4 bars (direct, react, multi_agent,
    map_reduce) showing biased pick rate on ambiguous BBQ items.
    """
    bbq_data = load_json("bbq_deep_dive.json")["biased_pick_rates"]

    # Positions for all 6 models
    x_positions = []
    model_labels = []
    pos = 0
    for i, model in enumerate(MODELS):
        x_positions.append(pos)
        model_labels.append(SHORT_MODEL_LABELS[model])
        pos += MODEL_SPACING

    n_configs = len(CONFIGS_PANEL_B)
    bar_width = 0.18
    x = np.array(x_positions, dtype=float)

    # Offsets for grouped bars
    offsets = np.arange(n_configs) - (n_configs - 1) / 2
    offsets = offsets * bar_width

    for j, cfg in enumerate(CONFIGS_PANEL_B):
        rates = []
        for model in MODELS:
            key = f"{model}|{cfg}|ambiguous"
            if key in bbq_data:
                rates.append(bbq_data[key]["biased_rate"] * 100)
            else:
                rates.append(0)
        rates = np.array(rates)

        ax.bar(x + offsets[j], rates, bar_width,
               color=CONFIG_COLORS[cfg], edgecolor="white", linewidth=0.5,
               label=CONFIG_LABELS[cfg], zorder=3)

        # Value labels on top of bars (only for large enough values)
        for i in range(len(MODELS)):
            if rates[i] > 8.0:  # only label bars tall enough to read
                ax.text(x[i] + offsets[j], rates[i] + 0.4,
                        f"{rates[i]:.0f}",
                        ha="center", va="bottom",
                        fontsize=FONT_SIZE_ANNOTATION - 2.5,
                        color=CONFIG_COLORS[cfg], fontweight="bold",
                        zorder=5)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels,
                       fontsize=FONT_SIZE_TICK - 1, rotation=25, ha="right")

    ax.set_ylabel("BBQ Biased Pick Rate, Ambiguous (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylim(0, 50)
    ax.set_xlim(-0.6, x[-1] + 0.6)
    ax.set_title("Evaluative Pressure\n(Multi-Agent / ReAct)",
                 fontsize=FONT_SIZE_TITLE - 1,
                 fontweight="bold", color=CONFIG_COLORS["multi_agent"], pad=8)

    # Legend
    ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND - 2,
              framealpha=0.9, edgecolor="0.8", ncol=2)

    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation arrows: highlight the two mechanisms
    # Find a model with notable map-reduce content loss
    # Use DeepSeek if available
    if "deepseek" in MODELS:
        ds_idx = MODELS.index("deepseek")
        mr_cfg_idx = CONFIGS_PANEL_B.index("map_reduce")
        ds_mr_key = "deepseek|map_reduce|ambiguous"
        if ds_mr_key in bbq_data:
            ds_mr_rate = bbq_data[ds_mr_key]["biased_rate"] * 100
            ax.annotate(
                "Content loss\ndrives bias",
                xy=(x[ds_idx] + offsets[mr_cfg_idx], ds_mr_rate + 0.5),
                xytext=(x[ds_idx] - 0.8, ds_mr_rate + 8),
                ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOTATION - 1,
                color=CONFIG_COLORS["map_reduce"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CONFIG_COLORS["map_reduce"],
                                lw=0.8, connectionstyle="arc3,rad=-0.25"),
                zorder=6,
            )

    # Multi-agent on Llama 4: evaluative pressure suppresses bias
    if "llama4" in MODELS:
        llama_idx = MODELS.index("llama4")
        ma_cfg_idx = CONFIGS_PANEL_B.index("multi_agent")
        llama_ma_key = "llama4|multi_agent|ambiguous"
        if llama_ma_key in bbq_data:
            llama_ma_rate = bbq_data[llama_ma_key]["biased_rate"] * 100
            ax.annotate(
                "Evaluative pressure\nsuppresses bias",
                xy=(x[llama_idx] + offsets[ma_cfg_idx], llama_ma_rate + 0.3),
                xytext=(x[llama_idx] - 0.8, llama_ma_rate + 16),
                ha="center", va="bottom",
                fontsize=FONT_SIZE_ANNOTATION - 1,
                color=CONFIG_COLORS["multi_agent"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CONFIG_COLORS["multi_agent"],
                                lw=0.8, connectionstyle="arc3,rad=-0.2"),
                zorder=6,
            )


def make_dual_mechanism_figure():
    """Create the two-panel dual mechanism figure."""
    setup_matplotlib()

    import matplotlib as mpl
    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(NEURIPS_FULL_WIDTH_IN, 4.5),
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.85, wspace=0.38)

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    # Panel labels -- placed above the title line, further left
    add_panel_label(ax_a, "(a)", x=-0.10, y=1.18, fontsize=12)
    add_panel_label(ax_b, "(b)", x=-0.10, y=1.18, fontsize=12)

    # Overall N reference
    fig.text(0.5, 0.01, "N = 62,808 scored observations",
             ha="center", fontsize=FONT_SIZE_ANNOTATION,
             fontstyle="italic", color="#666666")

    return fig


def main():
    fig = make_dual_mechanism_figure()
    savefig(fig, "fig6_dual_mechanism")

    # Also save directly to paper/figures/ as requested
    import shutil
    src = os.path.join(os.path.dirname(__file__), "output", "fig6_dual_mechanism.pdf")
    dst = os.path.join(os.path.dirname(__file__), "fig6_dual_mechanism.pdf")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")


if __name__ == "__main__":
    main()
