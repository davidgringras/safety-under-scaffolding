#!/usr/bin/env python3
"""
Figure 5: Number Needed to Harm (NNH) Forest Plot.

Forest plot showing NNH computed from H3 interaction model risk differences
for each model x config comparison (react, multi_agent, map_reduce vs direct).
Log-scale x-axis. Dashed threshold at NNH=10. Color-coded by config.
All 6 models shown as full data rows.

Data source: analysis/outputs/confirmatory_results.json (H3 model_effects)
NNH = 1 / |RD|, direction = "harm" if RD < 0, "benefit" if RD > 0.
CIs derived from RD +/- 1.96 * RD_SE (NNH CIs inverted from RD CIs).

Usage:
    .venv/bin/python3 paper/figures/fig5_nnh.py
"""

from __future__ import annotations

import json
import math
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
    CONFIG_ORDER, CONFIG_LABELS, CONFIG_COLORS,
    MODEL_LABELS, FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION,
    NEURIPS_TEXT_WIDTH_IN,
)

# All 6 models
ALL_MODELS = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
ALL_MODEL_LABELS = {
    "opus": "Opus 4.6",
    "gpt52": "GPT-5.2",
    "gemini3pro": "Gemini 3 Pro",
    "llama4": "Llama 4",
    "deepseek": "DeepSeek V3.2",
    "mistral": "Mistral Large 2",
}
SCAFFOLD_CONFIGS = ["react", "multi_agent", "map_reduce"]

# Cap for display purposes: NNH values beyond this are effectively "no effect"
NNH_DISPLAY_CAP = 500
NNH_CI_CAP = 2000

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_h3_data():
    """Load H3 model_effects from confirmatory_results.json and compute NNH."""
    data_path = (
        Path(__file__).resolve().parent.parent.parent
        / "analysis" / "outputs" / "confirmatory_results.json"
    )
    with open(data_path) as f:
        raw = json.load(f)

    model_effects = raw["H3"]["model_effects"]

    rows = []
    for model in ALL_MODELS:
        if model not in model_effects:
            continue
        for cfg in SCAFFOLD_CONFIGS:
            if cfg not in model_effects[model]:
                continue

            entry = model_effects[model][cfg]
            rd = entry["RD"]
            rd_se = entry.get("RD_SE", None)

            # NNH = 1 / |RD|
            if abs(rd) < 1e-10:
                nnh = float("inf")
            else:
                nnh = 1.0 / abs(rd)

            # Direction
            if rd < -1e-10:
                direction = "harm"
            elif rd > 1e-10:
                direction = "benefit"
            else:
                direction = "harm"  # default for zero

            # Compute NNH CIs from RD CIs if SE is available
            nnh_lo = None
            nnh_hi = None
            if rd_se is not None and rd_se > 0:
                rd_ci_lo = rd - 1.96 * rd_se
                rd_ci_hi = rd + 1.96 * rd_se

                abs_rd_vals = [abs(rd_ci_lo), abs(rd_ci_hi)]
                if rd_ci_lo <= 0 <= rd_ci_hi:
                    abs_rd_max = max(abs_rd_vals)
                    nnh_lo = 1.0 / abs_rd_max if abs_rd_max > 1e-10 else float("inf")
                    nnh_hi = float("inf")
                else:
                    abs_rd_min = min(abs_rd_vals)
                    abs_rd_max = max(abs_rd_vals)
                    nnh_lo = 1.0 / abs_rd_max if abs_rd_max > 1e-10 else float("inf")
                    nnh_hi = 1.0 / abs_rd_min if abs_rd_min > 1e-10 else float("inf")

            rows.append({
                "model": model,
                "config": cfg,
                "nnh": nnh,
                "nnh_lo": nnh_lo,
                "nnh_hi": nnh_hi,
                "direction": direction,
                "rd": rd,
                "label": f"{ALL_MODEL_LABELS[model]}  |  {CONFIG_LABELS[cfg]}",
            })

    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_fig5(rows):
    """Create NNH forest plot from H3 interaction model estimates."""
    setup_matplotlib()

    # Sort: group by config, then by NNH within each config
    config_sort = {"map_reduce": 0, "multi_agent": 1, "react": 2}
    rows.sort(key=lambda r: (config_sort.get(r["config"], 9),
                              r["nnh"] if r["nnh"] is not None else 9999))

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(NEURIPS_TEXT_WIDTH_IN, 0.45 * n_rows + 1.6))

    y_positions = np.arange(n_rows)

    for i, row in enumerate(rows):
        y = n_rows - 1 - i  # top-to-bottom ordering
        cfg = row["config"]
        color = CONFIG_COLORS.get(cfg, "#888888")

        nnh_val = row["nnh"]
        nnh_lo = row["nnh_lo"]
        nnh_hi = row["nnh_hi"]

        # Cap for display
        nnh_display = min(nnh_val, NNH_DISPLAY_CAP)

        # Determine marker shape based on direction
        if row["direction"] == "harm":
            marker = "v"  # triangle down = harmful
        else:
            marker = "^"  # triangle up = benefit
        marker_color = color

        # Plot point estimate
        ax.plot(nnh_display, y, marker=marker, markersize=8,
                color=marker_color, markeredgecolor="white",
                markeredgewidth=0.7, zorder=5)

        # CI whiskers (if available)
        if nnh_lo is not None and nnh_hi is not None:
            ci_lo_display = min(nnh_lo, NNH_CI_CAP) if not np.isinf(nnh_lo) else NNH_CI_CAP
            ci_hi_display = min(nnh_hi, NNH_CI_CAP) if not np.isinf(nnh_hi) else NNH_CI_CAP

            ci_left = min(ci_lo_display, ci_hi_display)
            ci_right = max(ci_lo_display, ci_hi_display)

            # Only draw CI if finite and meaningful
            if ci_right < NNH_CI_CAP and ci_left > 0:
                ax.plot([ci_left, ci_right], [y, y],
                        color=marker_color, linewidth=1.2, alpha=0.5, zorder=3)
                # Caps
                cap_h = 0.15
                ax.plot([ci_left, ci_left], [y - cap_h, y + cap_h],
                        color=marker_color, linewidth=0.8, alpha=0.5, zorder=3)
                ax.plot([ci_right, ci_right], [y - cap_h, y + cap_h],
                        color=marker_color, linewidth=0.8, alpha=0.5, zorder=3)
            elif ci_right >= NNH_CI_CAP:
                # Draw arrow to indicate CI extends to infinity
                ax.annotate("", xy=(NNH_DISPLAY_CAP * 0.95, y),
                            xytext=(nnh_display, y),
                            arrowprops=dict(arrowstyle="->", color=marker_color,
                                            lw=1.0, alpha=0.4))

        # NNH value label — offset to avoid marker/CI overlap
        if nnh_val < NNH_DISPLAY_CAP:
            direction_symbol = "H" if row["direction"] == "harm" else "B"
            nnh_str = f"{int(math.ceil(nnh_val))} ({direction_symbol})"
            # Place label to the right of CI endpoint or marker, whichever is further right
            rightmost = nnh_display
            if nnh_hi is not None and not np.isinf(nnh_hi) and nnh_hi < NNH_CI_CAP:
                rightmost = max(rightmost, min(nnh_hi, NNH_DISPLAY_CAP))
            label_x = rightmost * 1.4
            ax.text(label_x, y, nnh_str,
                    ha="left", va="center", fontsize=6.5,
                    color="#444444", zorder=5)
        else:
            ax.text(NNH_DISPLAY_CAP * 0.65, y, f"NNH>{NNH_DISPLAY_CAP}",
                    ha="right", va="center", fontsize=6.5,
                    color="#999999", style="italic", zorder=5)

    # Y-axis labels
    labels = [rows[n_rows - 1 - i]["label"] for i in range(n_rows)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.5)

    # Log scale x-axis
    ax.set_xscale("log")
    ax.set_xlim(1.5, NNH_DISPLAY_CAP * 1.5)

    # Reference lines with labels placed using axes-fraction y coordinate
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # NNH = 10 threshold
    ax.axvline(10, color="#D73027", linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)
    ax.text(10, 1.02, "NNH=10", fontsize=6,
            ha="center", va="bottom", color="#D73027", fontweight="bold",
            transform=trans)

    # NNH = 50
    ax.axvline(50, color="#E69F00", linewidth=0.7, linestyle=":", alpha=0.4, zorder=1)
    ax.text(50, 1.02, "NNH=50", fontsize=5.5,
            ha="center", va="bottom", color="#E69F00",
            transform=trans)

    # NNH = 100
    ax.axvline(100, color="#1A9850", linewidth=0.7, linestyle=":", alpha=0.4, zorder=1)
    ax.text(100, 1.02, "NNH=100", fontsize=5.5,
            ha="center", va="bottom", color="#1A9850",
            transform=trans)

    # Section dividers between config groups
    prev_cfg = None
    for i, row in enumerate(rows):
        y = n_rows - 1 - i
        if prev_cfg is not None and row["config"] != prev_cfg:
            ax.axhline(y + 0.5, color="#CCCCCC", linewidth=0.6, linestyle="-", alpha=0.6)
        prev_cfg = row["config"]

    ax.set_xlabel("Number Needed to Harm (NNH, log scale)", fontsize=9)
    ax.set_title("Number Needed to Harm by Model and Scaffold (N = 62,808)",
                 fontsize=FONT_SIZE_TITLE - 1, fontweight="bold", pad=20)

    # Legend
    legend_patches = [mpatches.Patch(color=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
                      for c in SCAFFOLD_CONFIGS]

    # Direction markers
    harm_marker = mlines.Line2D([], [], color="#666666", marker="v", linestyle="None",
                                 markersize=6, label="Harm (scaffold worse)")
    benefit_marker = mlines.Line2D([], [], color="#666666", marker="^", linestyle="None",
                                    markersize=6, label="Benefit (scaffold better)")
    legend_patches.extend([harm_marker, benefit_marker])

    ax.legend(handles=legend_patches, loc="lower right", fontsize=7,
              frameon=True, ncol=2, handlelength=1.5,
              borderpad=0.4, columnspacing=1.0)

    # Interpretation note
    ax.text(0.01, -0.10,
            "NNH = 1/|RD| from H3 interaction model (config x model). "
            "H = harm, B = benefit. CIs from RD +/- 1.96*SE.",
            transform=ax.transAxes, fontsize=6,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                      edgecolor="0.7", alpha=0.9),
            style="italic")

    return fig


def main():
    rows = load_h3_data()
    fig = make_fig5(rows)

    # Save to paper/figures/output/ (default behavior of savefig)
    fig_dir = Path(__file__).resolve().parent
    savefig(fig, "fig5_nnh", output_dir=str(fig_dir / "output"))

    # Also copy to paper/figures/ for direct inclusion
    import shutil
    src_pdf = fig_dir / "output" / "fig5_nnh.pdf"
    dst_pdf = fig_dir / "fig5_nnh.pdf"
    if src_pdf.exists():
        shutil.copy2(str(src_pdf), str(dst_pdf))
        print(f"  Copied: {dst_pdf}")


if __name__ == "__main__":
    main()
