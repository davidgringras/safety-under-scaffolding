#!/usr/bin/env python3
"""
Figure 2: Error Direction Analysis.

Stacked bar chart showing error direction breakdown by scaffolded config
(react, multi_agent, map_reduce). Uses real data from
analysis/outputs/error_direction.json.

Categories: factual_error, factual_correction, sycophantic, anti_sycophantic,
over_cautious, under_cautious, format_noncompliance, other.

Usage:
    python fig2_error_direction.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, savefig,
    CONFIG_LABELS,
    FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    NEURIPS_TEXT_WIDTH_IN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Scaffolded configs only (exclude direct baseline)
SCAFFOLD_CONFIGS = ["react", "multi_agent", "map_reduce"]
MODELS = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
BENCHMARKS = ["bbq", "sycophancy", "truthfulqa", "xstest_orbench"]

# Error categories in display order
ERROR_CATEGORIES = [
    "factual_error",
    "factual_correction",
    "sycophantic",
    "anti_sycophantic",
    "over_cautious",
    "under_cautious",
    "format_noncompliance",
    "other",
]

# Human-readable labels
ERROR_LABELS = {
    "factual_error":        "Factual Error",
    "factual_correction":   "Factual Correction",
    "sycophantic":          "Sycophantic",
    "anti_sycophantic":     "Anti-Sycophantic",
    "over_cautious":        "Over-Cautious",
    "under_cautious":       "Under-Cautious",
    "format_noncompliance": "Format Non-compliance",
    "other":                "Other",
}

# Distinct colors for each error category
ERROR_COLORS = {
    "factual_error":        "#C4392A",  # red
    "factual_correction":   "#2E8B57",  # green (sea green)
    "sycophantic":          "#E8601C",  # vivid orange
    "anti_sycophantic":     "#7B68EE",  # medium slate blue
    "over_cautious":        "#3B6FB6",  # vivid blue
    "under_cautious":       "#5AAECC",  # teal-blue
    "format_noncompliance": "#999999",  # grey
    "other":                "#DAA520",  # goldenrod
}


def load_error_direction_data():
    """Load real error direction data from analysis/outputs/error_direction.json.

    Returns a dict: aggregated[config] = {error_type: count} aggregated across
    all models and benchmarks.
    """
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "analysis", "outputs", "error_direction.json",
    )
    with open(data_path) as f:
        raw = json.load(f)

    distributions = raw["error_distributions"]

    # Aggregate by config across all models and benchmarks
    aggregated = {cfg: defaultdict(int) for cfg in SCAFFOLD_CONFIGS}

    for key, counts in distributions.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        model, config, benchmark = parts
        if config not in SCAFFOLD_CONFIGS:
            continue
        if model not in MODELS:
            continue
        for err_type, n in counts.items():
            aggregated[config][err_type] += n

    return aggregated


def make_error_direction_figure():
    """Create the stacked bar chart of error direction breakdown by config."""
    setup_matplotlib()

    aggregated = load_error_direction_data()

    # Prepare data for plotting
    # For each config, compute fractions of each error type
    config_labels = [CONFIG_LABELS[c] for c in SCAFFOLD_CONFIGS]
    config_totals = []
    config_fractions = []

    for cfg in SCAFFOLD_CONFIGS:
        counts = aggregated[cfg]
        total = sum(counts.values())
        config_totals.append(total)
        fracs = {}
        for cat in ERROR_CATEGORIES:
            fracs[cat] = (counts.get(cat, 0) / total * 100) if total > 0 else 0
        config_fractions.append(fracs)

    # Figure
    fig_width = NEURIPS_TEXT_WIDTH_IN
    fig_height = 3.0
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    y_positions = np.arange(len(SCAFFOLD_CONFIGS))
    bar_height = 0.55

    # Build stacked horizontal bars
    lefts = np.zeros(len(SCAFFOLD_CONFIGS))
    bar_handles = []

    for cat in ERROR_CATEGORIES:
        vals = np.array([cf[cat] for cf in config_fractions])
        if np.sum(vals) == 0:
            continue

        color = ERROR_COLORS[cat]
        bars = ax.barh(
            y_positions, vals, bar_height, left=lefts,
            color=color, edgecolor="white", linewidth=0.5,
            label=ERROR_LABELS[cat],
        )
        bar_handles.append(bars)

        # Annotate segments >= 8%
        for j, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 8:
                ax.text(
                    l + v / 2, j,
                    f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=FONT_SIZE_ANNOTATION - 0.5,
                    fontweight="bold",
                    color="white",
                )

        lefts += vals

    # Y-axis: config labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(config_labels, fontsize=FONT_SIZE_TICK)

    # X-axis
    ax.set_xlim(0, 108)
    ax.set_xlabel("Error Composition (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_xticks([0, 25, 50, 75, 100])

    # Title
    ax.set_title(
        "Error Direction Breakdown by Scaffolding Configuration",
        fontsize=FONT_SIZE_TITLE,
        fontweight="bold",
        pad=12,
    )

    # Add total N annotations on the right side
    for j, (cfg, total) in enumerate(zip(SCAFFOLD_CONFIGS, config_totals)):
        ax.text(
            102, j,
            f"N={total:,}",
            ha="left", va="center",
            fontsize=FONT_SIZE_ANNOTATION,
            color="#555555",
        )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=4,
        fontsize=FONT_SIZE_LEGEND - 1.0,
        frameon=True,
        edgecolor="0.8",
        fancybox=True,
        handlelength=1.0,
        handleheight=0.7,
        columnspacing=0.8,
    )

    # Clean up grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.2, linewidth=0.3)
    ax.yaxis.grid(False)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def main():
    fig = make_error_direction_figure()

    # Save to both the expected location (paper/figures/) and output/
    figures_dir = os.path.dirname(os.path.abspath(__file__))

    # Save as PDF directly in paper/figures/
    pdf_path = os.path.join(figures_dir, "fig2_error_direction.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    print(f"  Saved: {pdf_path}")

    # Also save via savefig helper (goes to paper/figures/output/)
    savefig(fig, "fig2_error_direction")

    # Print summary
    aggregated = load_error_direction_data()
    print("\n  Error direction summary:")
    for cfg in SCAFFOLD_CONFIGS:
        counts = aggregated[cfg]
        total = sum(counts.values())
        print(f"\n    {CONFIG_LABELS[cfg]} (N={total}):")
        for cat in ERROR_CATEGORIES:
            n = counts.get(cat, 0)
            if n > 0:
                pct = 100 * n / total
                print(f"      {ERROR_LABELS[cat]:25s}: {n:4d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
