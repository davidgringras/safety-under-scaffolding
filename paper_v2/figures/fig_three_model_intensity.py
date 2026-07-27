"""
Figure: Three-Model Prompt Intensity Comparison

Shows encoding depth as a continuous moderator with two distinct
failure architectures: linear degradation (DeepSeek) vs threshold
collapse (GPT-5.2), contrasted with immune/improving (Opus).

Data sources:
  - docs/sequential_e4_variants.md (DeepSeek)
  - docs/exp1_e4_opus.md (Opus)
  - docs/exp2_e4_gpt52.md (GPT-5.2)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from style import (setup_matplotlib, savefig, MODEL_COLORS, MODEL_LABELS,
                   FONT_SIZE_ANNOTATION, FONT_SIZE_LEGEND,
                   MARKER_SIZE_PRIMARY, LINE_WIDTH_PRIMARY)

setup_matplotlib()

# ── Data: Risk difference from passthrough baseline (pp) ────────────
# X-axis conditions (excluding passthrough, which is the 0 baseline)
conditions = ["Neutral\nchain", "Moderate\nbias-focused", "Aggressive\nbias-focused"]
x = np.arange(len(conditions))

# Opus: Minimal=+2.0pp, Moderate=+4.0pp, Aggressive=+4.0pp
opus_rd = [+2.0, +4.0, +4.0]

# DeepSeek: Minimal=+0.0pp, Moderate=-6.0pp, Aggressive=-12.0pp
deepseek_rd = [0.0, -6.0, -12.0]

# GPT-5.2: Minimal=+0.0pp, Moderate=-6.0pp, Aggressive=-22.0pp
gpt52_rd = [0.0, -6.0, -22.0]

# ── Colors and styles ───────────────────────────────────────────────
opus_color = "#009E73"       # green/teal (from style: gpt52 green works for Opus here)
deepseek_color = "#0072B2"   # blue
gpt52_color = "#D73027"      # red

# ── Create figure ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Baseline at 0pp
ax.axhline(y=0, color="#555555", linestyle="--", linewidth=0.9,
           zorder=2, alpha=0.6, label="_nolegend_")
ax.text(2.35, 0.6, "Passthrough baseline",
        fontsize=FONT_SIZE_ANNOTATION, color="#555555", ha="right", va="bottom")

# Plot lines
ax.plot(x, opus_rd, color=opus_color, linestyle="-", linewidth=LINE_WIDTH_PRIMARY + 0.5,
        marker="o", markersize=MARKER_SIZE_PRIMARY + 2, markeredgecolor="white",
        markeredgewidth=1.0, zorder=5, label="Opus 4.6")

ax.plot(x, deepseek_rd, color=deepseek_color, linestyle="--", linewidth=LINE_WIDTH_PRIMARY + 0.5,
        marker="s", markersize=MARKER_SIZE_PRIMARY + 1, markeredgecolor="white",
        markeredgewidth=1.0, zorder=5, label="DeepSeek V3.2")

ax.plot(x, gpt52_rd, color=gpt52_color, linestyle=":", linewidth=LINE_WIDTH_PRIMARY + 0.5,
        marker="D", markersize=MARKER_SIZE_PRIMARY + 1, markeredgecolor="white",
        markeredgewidth=1.0, zorder=5, label="GPT-5.2")

# ── Annotations ─────────────────────────────────────────────────────
# Opus: "Immune / improves"
ax.annotate("Immune / improves",
            xy=(2, opus_rd[2]), xytext=(1.55, 8.5),
            fontsize=FONT_SIZE_ANNOTATION + 1, color=opus_color,
            fontstyle="italic", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=opus_color,
                            lw=0.8, connectionstyle="arc3,rad=-0.15"),
            ha="center")

# DeepSeek: "Linear degradation"
ax.annotate("Linear degradation",
            xy=(2, deepseek_rd[2]), xytext=(0.9, -15.5),
            fontsize=FONT_SIZE_ANNOTATION + 1, color=deepseek_color,
            fontstyle="italic", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=deepseek_color,
                            lw=0.8, connectionstyle="arc3,rad=0.15"),
            ha="center")

# GPT-5.2: "Threshold collapse"
ax.annotate("Threshold collapse",
            xy=(2, gpt52_rd[2]), xytext=(0.85, -25.5),
            fontsize=FONT_SIZE_ANNOTATION + 1, color=gpt52_color,
            fontstyle="italic", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=gpt52_color,
                            lw=0.8, connectionstyle="arc3,rad=0.3"),
            ha="center")

# ── Data point labels ───────────────────────────────────────────────
for i, (o, d, g) in enumerate(zip(opus_rd, deepseek_rd, gpt52_rd)):
    offset_y = 1.2
    # Opus values
    sign_o = "+" if o >= 0 else ""
    ax.text(i + 0.08, o + offset_y, f"{sign_o}{o:.0f}pp",
            fontsize=8, color=opus_color, ha="left", va="bottom")
    # DeepSeek values
    sign_d = "+" if d >= 0 else ""
    if i == 0:
        # At neutral, DeepSeek and GPT-5.2 overlap at 0pp; show only DeepSeek below
        ax.text(i + 0.08, d - offset_y - 0.5, f"{sign_d}{d:.0f}pp",
                fontsize=8, color=deepseek_color, ha="left", va="top")
    elif d == g:
        # Same value: offset them vertically to avoid overlap
        ax.text(i + 0.12, d + offset_y + 0.3, f"{sign_d}{d:.0f}pp",
                fontsize=8, color=deepseek_color, ha="left", va="bottom")
        sign_g = "+" if g >= 0 else ""
        ax.text(i + 0.12, g - offset_y - 1.0, f"{sign_g}{g:.0f}pp",
                fontsize=8, color=gpt52_color, ha="left", va="top")
    else:
        ax.text(i + 0.10, d - offset_y - 0.5, f"{sign_d}{d:.0f}pp",
                fontsize=8, color=deepseek_color, ha="left", va="top")
        # GPT-5.2 values
        sign_g = "+" if g >= 0 else ""
        ax.text(i - 0.10, g - offset_y - 0.5, f"{sign_g}{g:.0f}pp",
                fontsize=8, color=gpt52_color, ha="right", va="top")

# ── Axes formatting ─────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylabel("Risk Difference from Passthrough (pp)", fontsize=12)
ax.set_xlabel("Review Prompt Intensity", fontsize=12)
ax.set_ylim(-28, 14)
ax.set_xlim(-0.4, 2.6)

ax.grid(axis="y", alpha=0.2, linewidth=0.4)
ax.grid(axis="x", visible=False)

# Legend
ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.92,
          edgecolor="0.8")

# ── Save ────────────────────────────────────────────────────────────
savefig(fig, "fig_three_model_intensity")
print("Done: fig_three_model_intensity")
