#!/usr/bin/env python3
"""
Figure 8: Sycophancy resistance heatmap by model and scaffold configuration.

Regenerated from canonical dataset to fix GPT-5.2 data contamination
(old figure used stale data with wrong N and rates).

Usage:
    python fig_sycophancy_heatmap.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, savefig, despine,
    FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION, FONT_SIZE_TICK,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_LEGEND,
    MODEL_LABELS, CONFIG_LABELS,
    NEURIPS_TEXT_WIDTH_IN,
)


# ---------------------------------------------------------------------------
# Load data from canonical dataset
# ---------------------------------------------------------------------------

CANONICAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "results", "canonical_primary_dataset.jsonl"
)

MODELS_ALL = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
MODELS_TWEET = ["opus", "gpt52", "llama4", "deepseek", "mistral"]
CONFIGS = ["direct", "react", "multi_agent", "map_reduce"]


def load_sycophancy_data(models):
    """Load sycophancy rates from canonical dataset."""
    counts = defaultdict(lambda: {"safe": 0, "total": 0})

    with open(CANONICAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["benchmark_id"] != "sycophancy":
                continue
            key = (r["model_id"], r["config_id"])
            counts[key]["total"] += 1
            if r.get("is_safe"):
                counts[key]["safe"] += 1

    # Build rate matrix
    rates = np.zeros((len(models), len(CONFIGS)))
    ns = np.zeros((len(models), len(CONFIGS)), dtype=int)
    for i, model in enumerate(models):
        for j, config in enumerate(CONFIGS):
            d = counts[(model, config)]
            ns[i, j] = d["total"]
            rates[i, j] = d["safe"] / d["total"] * 100 if d["total"] > 0 else 0

    return rates, ns


def make_figure(models, note_text, height=3.2):
    setup_matplotlib()

    rates, ns = load_sycophancy_data(models)

    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, height))

    # Custom colormap: red (low) -> yellow (mid) -> green (high)
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=100)

    im = ax.imshow(rates, cmap=cmap, norm=norm, aspect="auto")

    # Cell annotations
    for i in range(len(models)):
        for j in range(len(CONFIGS)):
            val = rates[i, j]
            # Dark text on light backgrounds, white on dark
            text_color = "white" if val < 25 else "black"
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                fontsize=FONT_SIZE_ANNOTATION + 1, fontweight="bold",
                color=text_color,
            )

    # Axes
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(len(models)))
    ylabels = [MODEL_LABELS[m] + ("*" if m == "gemini3pro" else "") for m in models]
    ax.set_yticklabels(ylabels, fontsize=FONT_SIZE_TICK)

    ax.set_title(
        "Sycophancy Resistance by Model and Scaffold Configuration",
        fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=10,
    )

    # Note at bottom
    ax.text(
        0.5, -0.10,
        note_text,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=FONT_SIZE_ANNOTATION - 0.5, color="#666666",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Non-Sycophancy Rate (%)", fontsize=FONT_SIZE_AXIS_LABEL - 1)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    return fig


if __name__ == "__main__":
    import shutil

    # Paper version (all 6 models, ITT with Gemini caveat)
    fig = make_figure(
        MODELS_ALL,
        "n = 500 per cell (ITT scoring).  *Gemini 3 Pro: 70.5% unparseable; PP rates: 31%, 46%, 40%, 46%.",
    )
    savefig(fig, "fig_sycophancy_heatmap")
    src = os.path.join(os.path.dirname(__file__), "output", "fig_sycophancy_heatmap.pdf")
    dst = os.path.join(os.path.dirname(__file__), "fig_sycophancy_heatmap.pdf")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")

    # Tweet version (5 models, Gemini excluded)
    fig2 = make_figure(
        MODELS_TWEET,
        "n = 500 per cell.  Gemini 3 Pro excluded (response truncation made >70% of items unscorable).",
        height=2.9,
    )
    savefig(fig2, "fig_sycophancy_heatmap_tweet")
