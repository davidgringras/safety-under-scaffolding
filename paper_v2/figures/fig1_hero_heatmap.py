#!/usr/bin/env python3
"""
Figure 1: Hero heatmap of safety rates.

Model x Config heatmap showing overall safety rates for all 6 models.
Data source: analysis/outputs/safety_rates_full.json

Usage:
    python fig1_hero_heatmap.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# Ensure the figures directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, get_figure_size, savefig,
    MODEL_LABELS, CONFIG_LABELS,
    FONT_SIZE_ANNOTATION, FONT_SIZE_TITLE,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK,
    NEURIPS_TEXT_WIDTH_IN, HEATMAP_CMAP,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Model ordering for heatmap rows (top to bottom) -- all 6 models
MODELS = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
CONFIGS = ["direct", "react", "multi_agent", "map_reduce"]
BENCHMARKS = ["bbq", "sycophancy", "truthfulqa", "xstest_orbench"]

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_safety_rates():
    """Load real safety rate data from analysis/outputs/safety_rates_full.json.

    Returns a dict: matrix[model][config] = weighted-average safety rate (0-100).
    """
    data_path = os.path.join(PROJECT_ROOT, "analysis", "outputs", "safety_rates_full.json")
    with open(data_path) as f:
        raw = json.load(f)

    sr = raw["safety_rates"]

    # Compute weighted average safety rate across benchmarks for each model x config
    matrix = {}
    for model in MODELS:
        matrix[model] = {}
        for config in CONFIGS:
            total_safe = 0
            total_n = 0
            for bm in BENCHMARKS:
                key = f"{model}|{config}|{bm}"
                if key in sr:
                    entry = sr[key]
                    total_safe += entry["n_safe"]
                    total_n += entry["n_total"]
            if total_n > 0:
                matrix[model][config] = (total_safe / total_n) * 100
            else:
                matrix[model][config] = float("nan")

    return matrix


def make_hero_heatmap():
    """Create the hero heatmap figure with all 6 models."""
    setup_matplotlib()

    matrix = load_safety_rates()

    # Build the data matrix: rows = models, cols = configs
    n_models = len(MODELS)
    n_configs = len(CONFIGS)
    data_matrix = np.full((n_models, n_configs), np.nan)

    for i, model in enumerate(MODELS):
        for j, config in enumerate(CONFIGS):
            data_matrix[i, j] = matrix[model].get(config, np.nan)

    # Color scale: RdYlGn, centered at midpoint of data range
    valid_vals = data_matrix[~np.isnan(data_matrix)]
    vmin = np.floor(np.min(valid_vals))
    vmax = np.ceil(np.max(valid_vals))

    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=72.0, vmax=vmax)

    # Figure size
    fig_width = NEURIPS_TEXT_WIDTH_IN
    fig_height = 4.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Plot the heatmap using vector Rectangle patches (not rasterized imshow)
    for i in range(n_models):
        for j in range(n_configs):
            val = data_matrix[i, j]
            if np.isnan(val):
                fc = "#CCCCCC"
            else:
                fc = cmap(norm(val))
            rect = mpatches.FancyBboxPatch(
                (j - 0.5, i - 0.5), 1.0, 1.0,
                boxstyle="square,pad=0",
                facecolor=fc, edgecolor="none", zorder=1,
            )
            ax.add_patch(rect)

    # Set axis limits to match the grid
    ax.set_xlim(-0.5, n_configs - 0.5)
    ax.set_ylim(n_models - 0.5, -0.5)

    # Create a ScalarMappable for the colorbar (no raster image needed)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Annotate all cells with safety rate percentages
    for i in range(n_models):
        for j, config in enumerate(CONFIGS):
            val = data_matrix[i, j]
            if not np.isnan(val):
                # Choose text color based on background luminance
                bg_color = cmap(norm(val))
                luminance = (
                    0.299 * bg_color[0]
                    + 0.587 * bg_color[1]
                    + 0.114 * bg_color[2]
                )
                text_color = "white" if luminance < 0.5 else "black"

                # Bold the direct baseline column
                weight = "bold" if j == 0 else "normal"

                label = f"{val:.1f}%"

                ax.text(
                    j, i, label,
                    ha="center", va="center",
                    fontsize=FONT_SIZE_ANNOTATION + 1,
                    fontweight=weight,
                    color=text_color,
                )

    # Axes formatting
    row_labels = [MODEL_LABELS[m] for m in MODELS]
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZE_TICK)

    ax.set_xticks(range(n_configs))
    ax.set_xticklabels(
        [CONFIG_LABELS[c] for c in CONFIGS],
        fontsize=FONT_SIZE_TICK,
    )

    # Move x-axis labels to top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Title
    ax.set_title(
        "Safety Rates by Model and Deployment Configuration",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=24,
    )

    # Grid lines (white, separating cells)
    for x_pos in range(1, n_configs):
        ax.axvline(x_pos - 0.5, color="white", linewidth=2)
    for y_pos in range(1, n_models):
        ax.axhline(y_pos - 0.5, color="white", linewidth=2)

    # Disable default grid
    ax.grid(False)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Tick params
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.06, shrink=0.75)
    cbar.set_label("Safety Rate (%)", fontsize=FONT_SIZE_AXIS_LABEL - 1)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK - 1)

    # Caption reference
    fig.text(
        0.02, -0.04,
        "N = 62,808 scored observations across 6 models, 4 configurations, and 4 benchmarks.",
        fontsize=FONT_SIZE_ANNOTATION,
        fontstyle="italic",
        color="#666666",
        ha="left",
    )

    return fig


def main():
    fig = make_hero_heatmap()

    # Save to both the expected location (paper/figures/) and output/
    figures_dir = os.path.dirname(os.path.abspath(__file__))

    # Save as PDF directly in paper/figures/
    pdf_path = os.path.join(figures_dir, "fig1_hero_heatmap.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    print(f"  Saved: {pdf_path}")

    # Also save via savefig helper (goes to paper/figures/output/)
    savefig(fig, "fig1_hero_heatmap")


if __name__ == "__main__":
    main()
