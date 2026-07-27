"""
Figure: E4/EXP-6 Side-by-Side Dose-Response Mirror

Shows that the SAME chain architecture produces opposite effects
depending on prompt CONTENT: degradation on BBQ (bias) vs improvement
on TruthfulQA (misconceptions).

Data sources:
  - docs/sequential_e4_variants.md (DeepSeek BBQ)
  - docs/exp6_tqa_variants.md (DeepSeek TruthfulQA)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from style import setup_matplotlib, savefig, add_panel_label, FONT_SIZE_ANNOTATION

setup_matplotlib()

# ── Data from experiment files ──────────────────────────────────────
# BBQ (E4) - DeepSeek
bbq_labels = ["Passthrough", "Minimal\n(neutral)", "Moderate\n(bias-focused)", "Aggressive\n(bias-focused)"]
bbq_rates = [94.0, 94.0, 88.0, 82.0]
bbq_deltas = [0.0, 0.0, -6.0, -12.0]
bbq_baseline = 94.0

# TruthfulQA (EXP-6) - DeepSeek
tqa_labels = ["Passthrough", "Minimal\n(neutral)", "Moderate\n(misc.-focused)", "Aggressive\n(misc.-focused)"]
tqa_rates = [74.0, 76.0, 82.0, 92.0]
tqa_deltas = [0.0, +2.0, +8.0, +18.0]
tqa_baseline = 74.0

# ── Color gradient: blue -> orange -> red ───────────────────────────
bar_colors = ["#808080", "#4575B4", "#F46D43", "#D73027"]

# ── Create figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, rates, deltas, baseline, labels, title, ylabel, direction in [
    (axes[0], bbq_rates, bbq_deltas, bbq_baseline, bbq_labels,
     "BBQ (Bias Benchmark)", "Safety Rate (%)", "down"),
    (axes[1], tqa_rates, tqa_deltas, tqa_baseline, tqa_labels,
     "TruthfulQA", "Accuracy (%)", "up"),
]:
    x = np.arange(len(rates))
    bars = ax.bar(x, rates, width=0.6, color=bar_colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    # Baseline dashed line (passthrough = direct API call, no scaffolding)
    ax.axhline(y=baseline, color="#888888", linestyle="--", linewidth=0.7,
               zorder=2, alpha=0.5)

    # Delta labels on bars
    for i, (rate, delta) in enumerate(zip(rates, deltas)):
        if i == 0:
            # Passthrough - just show the rate
            ax.text(i, rate + 0.8, f"{rate:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color="#333333")
        else:
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:.0f}pp"
            color = "#D73027" if delta < 0 else "#1A9850" if delta > 0 else "#555555"
            ax.text(i, rate + 0.8, delta_str,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_ylim(60, 102)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.grid(axis="x", visible=False)

# Panel labels
add_panel_label(axes[0], "A", x=-0.10, y=1.06)
add_panel_label(axes[1], "B", x=-0.10, y=1.06)

# ── Annotation spanning both panels ─────────────────────────────────
fig.text(0.5, -0.02,
         "Chain architecture identical \u2014 prompt content drives the effect",
         ha="center", va="top", fontsize=11, fontstyle="italic",
         color="#333333")

# ── Save ────────────────────────────────────────────────────────────
savefig(fig, "fig_e4_exp6_mirror")
print("Done: fig_e4_exp6_mirror")
