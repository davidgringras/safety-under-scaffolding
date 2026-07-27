#!/usr/bin/env python3
"""
Figure 3: Model-Benchmark Sensitivity Matrix.

Panel of 4 grouped bar charts (one per benchmark), each showing
safety rates for all 6 models x 4 configs. Significance stars from
pairwise comparisons (BH-corrected).

Data source: analysis/outputs/safety_rates_full.json

Usage:
    .venv/bin/python3 paper/figures/fig3_sensitivity_capability.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, savefig,
    CONFIG_ORDER, BENCHMARK_ORDER,
    MODEL_LABELS, CONFIG_LABELS, BENCHMARK_LABELS,
    CONFIG_COLORS, FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION,
    NEURIPS_FULL_WIDTH_IN,
)

# All 6 models
ALL_MODELS = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]

# Shorter model labels to prevent overlap on x-axis
SHORT_MODEL_LABELS = {
    "opus": "Opus",
    "gpt52": "GPT-5.2",
    "gemini3pro": "Gemini",
    "llama4": "Llama 4",
    "deepseek": "DeepSeek",
    "mistral": "Mistral",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load safety rates and pairwise comparisons from JSON."""
    data_path = Path(__file__).resolve().parent.parent.parent / "analysis" / "outputs" / "safety_rates_full.json"
    with open(data_path) as f:
        raw = json.load(f)
    return raw["safety_rates"], raw.get("pairwise_comparisons", [])


def build_significance_lookup(comparisons):
    """Build a lookup dict: (model, benchmark, comparison) -> significant_bh05."""
    lookup = {}
    for c in comparisons:
        key = (c["model"], c["benchmark"], c["comparison"])
        lookup[key] = c["significant_bh05"]
    return lookup


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_fig3(safety_rates, sig_lookup):
    """Create the 4-panel grouped bar chart (one panel per benchmark)."""
    setup_matplotlib()

    import matplotlib as mpl
    mpl.rcParams["figure.constrained_layout.use"] = False

    n_models = len(ALL_MODELS)
    n_configs = len(CONFIG_ORDER)

    fig, axes = plt.subplots(1, 4, figsize=(NEURIPS_FULL_WIDTH_IN, 5.0), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.24, top=0.87, wspace=0.08)

    bar_width = 0.13
    # Positions for each model group within a panel
    group_positions = np.arange(n_models)

    for ax_idx, bm in enumerate(BENCHMARK_ORDER):
        ax = axes[ax_idx]

        for cfg_idx, cfg in enumerate(CONFIG_ORDER):
            offset = (cfg_idx - (n_configs - 1) / 2) * bar_width
            color = CONFIG_COLORS[cfg]

            for m_idx, model in enumerate(ALL_MODELS):
                key = f"{model}|{cfg}|{bm}"
                entry = safety_rates.get(key, None)
                if entry is None:
                    continue

                rate = entry["safety_rate"] * 100  # convert to percentage
                ci_lo = entry["ci_lower"] * 100
                ci_hi = entry["ci_upper"] * 100
                yerr_lo = rate - ci_lo
                yerr_hi = ci_hi - rate

                x = group_positions[m_idx] + offset

                ax.bar(x, rate, width=bar_width * 0.9,
                       color=color, edgecolor="white", linewidth=0.3,
                       alpha=0.88, zorder=3)
                ax.errorbar(x, rate, yerr=[[yerr_lo], [yerr_hi]],
                            fmt="none", ecolor="black", alpha=0.35,
                            capsize=1.5, capthick=0.6, linewidth=0.6, zorder=4)

                # Significance star for scaffold vs direct
                if cfg != "direct":
                    comp_name = f"{cfg}_vs_direct"
                    sig_key = (model, bm, comp_name)
                    if sig_lookup.get(sig_key, False):
                        star_y = ci_hi + 2.5
                        ax.text(x, star_y, "*", ha="center", va="bottom",
                                fontsize=7, fontweight="bold", color="#333333",
                                zorder=5)

        # Panel formatting
        ax.set_title(BENCHMARK_LABELS[bm], fontsize=10, fontweight="bold", pad=8)

        model_labels_for_axis = [SHORT_MODEL_LABELS[m] for m in ALL_MODELS]
        ax.set_xticks(group_positions)
        ax.set_xticklabels(model_labels_for_axis, fontsize=6, rotation=40, ha="right")

        ax.set_ylim(0, 108)
        ax.axhline(50, color="#CCCCCC", linewidth=0.4, linestyle=":", alpha=0.6, zorder=0)

        if ax_idx == 0:
            ax.set_ylabel("Safety Rate (%)", fontsize=9)

    # Shared legend
    legend_patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
                      for c in CONFIG_ORDER]

    # Add significance note to legend entries
    sig_marker = mlines.Line2D([], [], color="none", marker="$*$", markersize=6,
                                markeredgecolor="#333333", linestyle="None",
                                label="p < 0.05 (BH-corrected) vs. Direct")
    legend_patches.append(sig_marker)

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=6.5, frameon=True,
               bbox_to_anchor=(0.5, -0.005),
               handlelength=1.0, handletextpad=0.3, columnspacing=0.8)

    # Compute total N from data
    total_n = sum(entry["n_total"] for entry in safety_rates.values())
    fig.suptitle(f"Safety Rate by Model, Benchmark, and Scaffold Configuration (N = {total_n:,})",
                 fontsize=FONT_SIZE_TITLE - 1, fontweight="bold", y=0.95)

    return fig


def main():
    safety_rates, comparisons = load_data()
    sig_lookup = build_significance_lookup(comparisons)
    fig = make_fig3(safety_rates, sig_lookup)

    # Save to output/ directory
    savefig(fig, "fig3_sensitivity_capability")

    # Also copy to paper/figures/ for LaTeX
    import shutil
    fig_dir = Path(__file__).resolve().parent
    src = fig_dir / "output" / "fig3_sensitivity_capability.pdf"
    dst = fig_dir / "fig3_sensitivity_capability.pdf"
    if src.exists():
        shutil.copy2(str(src), str(dst))
        print(f"  Copied: {dst}")


if __name__ == "__main__":
    main()
