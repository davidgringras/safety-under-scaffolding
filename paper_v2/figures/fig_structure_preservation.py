#!/usr/bin/env python3
"""
Figure: Structure Preservation Dose-Response

Plots the relationship between task-structure preservation (MC option propagation)
and safety outcome across scaffold conditions. Shows that more structure
preserved leads to better safety outcomes.

Data sources:
  - Main experiment: safety_rates_full.json (direct, react, multi_agent, map_reduce)
  - CoT pilot: pilot3_cot_scaled.json (N=200, 6 models)
  - Option-preserving MR: pilot71_final.json
  - Propagation analysis: propagation_analysis.json

Only MC-format benchmarks (TruthfulQA, BBQ, Sycophancy) are used.

Usage:
    python fig_structure_preservation.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import (
    setup_matplotlib, get_figure_size, savefig,
    MODEL_COLORS, MODEL_LABELS, MODEL_ORDER,
    NEURIPS_TEXT_WIDTH_IN, NEURIPS_FULL_WIDTH_IN,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK, FONT_SIZE_ANNOTATION,
    FONT_SIZE_LEGEND,
)

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis", "outputs")
PILOTS_DIR = os.path.join(PROJECT_ROOT, "pilots")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# ──────────────────────────────────────────────────────────────────────
# Condition ordering (left to right = increasing structure preservation)
# ──────────────────────────────────────────────────────────────────────
CONDITION_ORDER = [
    "cot",
    "direct",
    "react",
    "multi_agent",
    "map_reduce_option_preserving",
    "map_reduce",
]

# True structure preservation percentages (for Spearman and annotation)
STRUCTURE_PRESERVATION = {
    "map_reduce":                     5,
    "map_reduce_option_preserving":  70,
    "multi_agent":                   90,
    "react":                         95,
    "direct":                       100,
    "cot":                          100,
}

# X-axis tick labels: condition name + structure %
CONDITION_TICK_LABELS = {
    "map_reduce":                   "Standard\nMR (5%)",
    "map_reduce_option_preserving": "Opt-Pres.\nMR (70%)",
    "multi_agent":                  "Multi-\nAgent (90%)",
    "react":                        "ReAct\n(95%)",
    "direct":                       "Direct\n(100%)",
    "cot":                          "CoT\n(100%)",
}

# Models to include
MODELS_MAIN = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
MC_BENCHMARKS = ["truthfulqa", "bbq"]  # Only MC-format benchmarks (sycophancy is open-ended)


def load_main_experiment_mc_rates():
    """
    Load safety rates for MC benchmarks only from the main experiment.
    Returns dict: model -> config -> safety_rate (as %)
    """
    with open(os.path.join(ANALYSIS_DIR, "safety_rates_full.json")) as f:
        data = json.load(f)

    rates = {}
    for model in MODELS_MAIN:
        rates[model] = {}
        for config in ["direct", "react", "multi_agent", "map_reduce"]:
            n_safe_total = 0
            n_total_total = 0
            for bm in MC_BENCHMARKS:
                key = f"{model}|{config}|{bm}"
                if key in data["safety_rates"]:
                    entry = data["safety_rates"][key]
                    n_safe_total += entry["n_safe"]
                    n_total_total += entry["n_total"]
            if n_total_total > 0:
                rates[model][config] = 100.0 * n_safe_total / n_total_total
    return rates


def load_cot_rates():
    """
    Load CoT safety rates from scaled pilot (N=200, MC benchmarks only).
    Returns dict: model -> safety_rate (as %)
    """
    with open(os.path.join(PILOTS_DIR, "pilot3_cot_scaled.json")) as f:
        data = json.load(f)

    rates = {}
    for model in MODELS_MAIN:
        if model in data["safety_rates"]:
            m_data = data["safety_rates"][model]
            if "cot" in m_data:
                cot = m_data["cot"]
                n_safe = 0
                n_total = 0
                for bm in MC_BENCHMARKS:
                    if bm in cot and cot[bm]["n_total"] > 0:
                        n_safe += cot[bm]["n_safe"]
                        n_total += cot[bm]["n_total"]
                if n_total > 0:
                    rates[model] = 100.0 * n_safe / n_total
    return rates


def load_option_preserving_rates():
    """
    Load option-preserving MR rates from pilot71_final.json.
    Returns dict: model -> safety_rate (as %) for MC benchmarks.
    """
    final_path = os.path.join(ANALYSIS_DIR, "pilot71_final.json")
    with open(final_path) as f:
        data = json.load(f)

    rates = {}
    agg = data["comparison_aggregate"]
    for model in agg:
        op_data = agg[model].get("map_reduce_option_preserving", {})
        if op_data.get("n_total", 0) > 0:
            rates[model] = 100.0 * op_data["safety_rate"]
    return rates


def main():
    setup_matplotlib()

    # Load all data
    main_rates = load_main_experiment_mc_rates()
    cot_rates = load_cot_rates()
    op_rates = load_option_preserving_rates()

    # Evenly-spaced x positions for conditions
    x_positions = {cond: i for i, cond in enumerate(CONDITION_ORDER)}

    # Assemble all points: (model, condition, x_pos, structure%, safety%)
    all_points = []

    for model in MODELS_MAIN:
        # Main experiment
        for config in ["direct", "react", "multi_agent", "map_reduce"]:
            if model in main_rates and config in main_rates[model]:
                xp = x_positions[config]
                sp = STRUCTURE_PRESERVATION[config]
                y = main_rates[model][config]
                all_points.append((model, config, xp, sp, y))

        # CoT
        if model in cot_rates:
            xp = x_positions["cot"]
            sp = STRUCTURE_PRESERVATION["cot"]
            y = cot_rates[model]
            all_points.append((model, "cot", xp, sp, y))

        # Option-preserving MR
        if model in op_rates:
            xp = x_positions["map_reduce_option_preserving"]
            sp = STRUCTURE_PRESERVATION["map_reduce_option_preserving"]
            y = op_rates[model]
            all_points.append((model, "map_reduce_option_preserving", xp, sp, y))

    # ──────────────────────────────────────────────────────────────
    # Compute Spearman correlation (on true structure preservation %)
    # ──────────────────────────────────────────────────────────────
    sp_vals = [p[3] for p in all_points]
    safety_vals = [p[4] for p in all_points]
    rho, p_val = stats.spearmanr(sp_vals, safety_vals)
    print(f"Spearman rho = {rho:.3f}, p = {p_val:.2e}, N = {len(sp_vals)}")

    # ──────────────────────────────────────────────────────────────
    # Create figure
    # ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(NEURIPS_FULL_WIDTH_IN, 3.6))

    # Use same marker for all conditions -- x-axis labels already identify them
    cond_markers = {
        "direct": "o",
        "cot": "o",
        "react": "o",
        "multi_agent": "o",
        "map_reduce_option_preserving": "o",
        "map_reduce": "o",
    }

    # Small y-direction jitter (no x-jitter -- use dodge instead)
    # Model dodge: small horizontal offset per model to avoid overlap
    n_models = len(MODELS_MAIN)
    dodge_width = 0.5  # total spread
    dodge_offsets = {}
    for i, model in enumerate(MODELS_MAIN):
        dodge_offsets[model] = -dodge_width / 2 + dodge_width * i / (n_models - 1)

    # Plot each model as connected line + markers
    for model in MODELS_MAIN:
        color = MODEL_COLORS.get(model, "#333333")
        dodge = dodge_offsets[model]

        # Gather points for this model
        model_pts = []
        for (m, c, xp, sp, y) in all_points:
            if m == model:
                model_pts.append((c, xp + dodge, y))

        # Sort by x position
        model_pts.sort(key=lambda t: t[1])

        if not model_pts:
            continue

        # Draw connecting line (thin, semi-transparent)
        line_xs = [p[1] for p in model_pts]
        line_ys = [p[2] for p in model_pts]
        ax.plot(line_xs, line_ys,
                color=color, linewidth=1.0, alpha=0.45, zorder=2)

        # Draw markers
        for cond, cx, cy in model_pts:
            marker = cond_markers.get(cond, "o")
            ax.scatter([cx], [cy],
                       color=color, marker=marker, s=55, zorder=4,
                       edgecolors="white", linewidths=0.5)

    # ──────────────────────────────────────────────────────────────
    # Add a light shaded region for "structure lost" zone
    # ──────────────────────────────────────────────────────────────
    ax.axvspan(len(CONDITION_ORDER) - 1.5, len(CONDITION_ORDER) - 0.4,
               alpha=0.05, color="#D73027", zorder=0)

    # ──────────────────────────────────────────────────────────────
    # Axes formatting
    # ──────────────────────────────────────────────────────────────
    ax.set_xticks(list(range(len(CONDITION_ORDER))))
    ax.set_xticklabels([CONDITION_TICK_LABELS[c] for c in CONDITION_ORDER],
                       fontsize=FONT_SIZE_TICK - 1, ha="center")

    ax.set_xlabel("Deployment Condition (task-structure preservation %)",
                  fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Safety Rate (%)", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_xlim(-0.6, len(CONDITION_ORDER) - 0.4)
    ax.set_ylim(44, 96)

    # Arrow annotation below x-axis labels (in figure coordinates)
    ax.annotate("",
                xy=(len(CONDITION_ORDER) - 0.5, 46.0),
                xytext=(-0.4, 46.0),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4,
                                mutation_scale=12),
                annotation_clip=False)
    ax.text(len(CONDITION_ORDER) / 2 - 0.5, 46.5,
            "more aggressive decomposition \u2192",
            fontsize=FONT_SIZE_ANNOTATION, ha="center", va="bottom",
            color="#444444", fontweight="bold", style="italic")

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    ax.grid(axis="x", alpha=0.0)  # no vertical grid

    # ──────────────────────────────────────────────────────────────
    # Correlation annotation
    # ──────────────────────────────────────────────────────────────
    p_str = (f"$p < 10^{{{int(np.floor(np.log10(p_val)))}}}$"
             if p_val < 0.001
             else f"$p = {p_val:.3f}$")
    ax.text(0.02, 0.03,
            f"Spearman $\\rho$ = {rho:.2f}, {p_str}, $N$ = {len(sp_vals)}",
            transform=ax.transAxes,
            fontsize=FONT_SIZE_ANNOTATION, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    # ──────────────────────────────────────────────────────────────
    # Legends
    # ──────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    # Model legend (colors) — placed in upper right to avoid data overlap
    model_handles = []
    for model in MODELS_MAIN:
        model_handles.append(
            Line2D([0], [0], marker="o", color=MODEL_COLORS[model],
                   label=MODEL_LABELS[model], markersize=5, linewidth=0.8,
                   markeredgecolor="white", markeredgewidth=0.3)
        )

    leg1 = ax.legend(handles=model_handles, loc="upper right",
                     fontsize=FONT_SIZE_LEGEND - 1, framealpha=0.92,
                     edgecolor="0.8",
                     bbox_to_anchor=(1.0, 1.0),
                     borderpad=0.4, handletextpad=0.4,
                     labelspacing=0.3)

    # ──────────────────────────────────────────────────────────────
    # Save
    # ──────────────────────────────────────────────────────────────
    savefig(fig, "fig_structure_preservation")

    # ──────────────────────────────────────────────────────────────
    # Save caption
    # ──────────────────────────────────────────────────────────────
    caption = (
        "Safety outcomes as a function of task-structure preservation across "
        "scaffold conditions. Each point represents one model under one deployment "
        "condition. Conditions ordered by the degree to which original task structure "
        "(MC options, context, system prompt) is preserved in model sub-calls. "
        "Direct and CoT conditions preserve full structure (right); standard "
        "map-reduce strips MC options and decomposes tasks (left); option-preserving "
        "map-reduce preserves MC options while still decomposing (middle). "
        "The monotonic relationship suggests structure preservation is a first-order "
        "determinant of scaffold safety outcomes."
    )
    caption_path = os.path.join(OUTPUT_DIR, "fig_structure_preservation_caption.txt")
    with open(caption_path, "w") as f:
        f.write(caption)
    print(f"  Caption: {caption_path}")

    # Print summary table
    print("\n  === Data Summary (MC benchmarks only) ===")
    print(f"  {'Model':<15} {'Condition':<30} {'Struct%':>7} {'Safety%':>8}")
    print(f"  {'-'*62}")
    for model, cond, xp, sp, y in sorted(all_points, key=lambda t: (t[0], t[3])):
        print(f"  {MODEL_LABELS[model]:<15} {CONDITION_TICK_LABELS[cond].replace(chr(10),' '):<30} {sp:>6}% {y:>7.1f}%")


if __name__ == "__main__":
    main()
