#!/usr/bin/env python3
"""
Configuration-Aware Safety Scorecard
=====================================
Publication-quality heatmap/table figure showing:
  Panel A: Pooled delta safety rate (all models) with case-cluster bootstrap 95% CI
  Panel B: Per-model delta breakdown as annotated heatmap

Rows: Safety benchmarks (BBQ, XSTest+OR-Bench, TruthfulQA, Sycophancy)
Cols: Scaffold configurations (ReAct, Multi-Agent, Map-Reduce)
Delta = scaffold rate - direct rate (percentage points)

Data: safety_rates_full.json, confirmatory_results.json, experiment_results_clean.jsonl
"""

import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# Style imports
sys.path.insert(0, str(Path(__file__).parent))
from style import (
    setup_matplotlib, MODEL_ORDER,
    FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK,
    FONT_SIZE_ANNOTATION, NEURIPS_TEXT_WIDTH_IN,
)

setup_matplotlib()
mpl.rcParams["figure.constrained_layout.use"] = False

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent.parent / "analysis" / "outputs"
with open(DATA_DIR / "safety_rates_full.json") as f:
    sr_data = json.load(f)["safety_rates"]
with open(DATA_DIR / "confirmatory_results.json") as f:
    cr = json.load(f)

SCAFFOLD_CONFIGS = ["react", "multi_agent", "map_reduce"]
BENCH_ORDER = ["bbq", "xstest_orbench", "truthfulqa", "sycophancy"]

BENCH_LABELS_A = {
    "bbq":            "BBQ\n(Bias)",
    "xstest_orbench": "XSTest+OR-Bench\n(Refusal)",
    "truthfulqa":     "TruthfulQA\n(Truthfulness)",
    "sycophancy":     "Sycophancy",
}
BENCH_LABELS_B = {
    "bbq":            "BBQ (Bias)",
    "xstest_orbench": "XSTest (Refusal)",
    "truthfulqa":     "TruthfulQA (Truth.)",
    "sycophancy":     "Sycophancy",
}
SCAFFOLD_LABELS = {
    "react": "ReAct", "multi_agent": "Multi-Agent", "map_reduce": "Map-Reduce",
}
MODEL_SHORT = {
    "opus": "Opus", "gpt52": "GPT", "gemini3pro": "Gem.",
    "llama4": "Lla.", "deepseek": "DSk.", "mistral": "Mis.",
}

# ---------------------------------------------------------------------------
# Statistics — case-cluster bootstrap CIs for RD
# ---------------------------------------------------------------------------

# Load pre-computed bootstrap CIs (from analysis/compute_scorecard_bootstrap.py)
BOOT_CI_PATH = DATA_DIR / "scorecard_bootstrap_cis.json"
_boot_cis = None

if BOOT_CI_PATH.exists():
    with open(BOOT_CI_PATH) as f:
        _boot_data = json.load(f)
    _boot_cis = _boot_data.get("results", {})
    print(f"Loaded bootstrap CIs ({_boot_data.get('n_boot', '?')} replicates)")
else:
    print(f"WARNING: {BOOT_CI_PATH} not found; falling back to Newcombe-Wilson CIs.")
    print(f"  Run: python analysis/compute_scorecard_bootstrap.py")


def wilson_ci_fallback(ns_s, nt_s, ns_d, nt_d, z=1.96):
    """Newcombe-Wilson fallback when bootstrap CIs are unavailable."""
    def _wci(n_s, n_t):
        if n_t == 0:
            return 0.0, 0.0, 0.0
        p = n_s / n_t
        d = 1 + z**2 / n_t
        c = (p + z**2 / (2 * n_t)) / d
        m = z * math.sqrt((p * (1 - p) + z**2 / (4 * n_t)) / n_t) / d
        return p, max(0, c - m), min(1, c + m)
    p1, p1_lo, p1_hi = _wci(ns_s, nt_s)
    p2, p2_lo, p2_hi = _wci(ns_d, nt_d)
    rd = p1 - p2
    lo = rd - math.sqrt((p1 - p1_lo)**2 + (p2_hi - p2)**2)
    hi = rd + math.sqrt((p1_hi - p1)**2 + (p2 - p2_lo)**2)
    return rd, lo, hi


# Pooled deltas (across all models)
pooled = {}
for bench in BENCH_ORDER:
    for cfg in SCAFFOLD_CONFIGS:
        key = f"{bench}|{cfg}"
        if _boot_cis and key in _boot_cis:
            bc = _boot_cis[key]
            pooled[(bench, cfg)] = (bc["rd"] * 100, bc["ci_lo"] * 100, bc["ci_hi"] * 100)
        else:
            ns_s = ns_d = nt_s = nt_d = 0
            for model in MODEL_ORDER:
                ks = f"{model}|{cfg}|{bench}"
                kd = f"{model}|direct|{bench}"
                if ks in sr_data and kd in sr_data:
                    ns_s += sr_data[ks]["n_safe"]; nt_s += sr_data[ks]["n_total"]
                    ns_d += sr_data[kd]["n_safe"]; nt_d += sr_data[kd]["n_total"]
            rd, lo, hi = wilson_ci_fallback(ns_s, nt_s, ns_d, nt_d)
            pooled[(bench, cfg)] = (rd * 100, lo * 100, hi * 100)

# Per-model deltas
per_model = {}
for model in MODEL_ORDER:
    for bench in BENCH_ORDER:
        for cfg in SCAFFOLD_CONFIGS:
            ks = f"{model}|{cfg}|{bench}"
            kd = f"{model}|direct|{bench}"
            if ks in sr_data and kd in sr_data:
                per_model[(model, bench, cfg)] = (
                    sr_data[ks]["safety_rate"] - sr_data[kd]["safety_rate"]) * 100

# Significance lookup
sig_lookup = {}
for pc in cr.get("pairwise_comparisons", []):
    cfg = pc["comparison"].replace("_vs_direct", "")
    key = (pc["benchmark"], cfg)
    sig_lookup.setdefault(key, {"n_sig": 0, "n_total": 0})
    sig_lookup[key]["n_total"] += 1
    if pc.get("significant_bh05", False):
        sig_lookup[key]["n_sig"] += 1


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def cell_color(rd):
    if rd >= -2.0:   return "#c7e9c0"  # green
    elif rd >= -5.0: return "#ffffb2"  # yellow
    elif rd >= -10.0: return "#fdae61"  # orange
    else:            return "#f46d43"  # red

def text_color(rd):
    return "white" if rd < -10.0 else "#1a1a1a"


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
n_bench = len(BENCH_ORDER)
n_cfg = len(SCAFFOLD_CONFIGS)
n_model = len(MODEL_ORDER)
n_cols_b = n_cfg * n_model

fig = plt.figure(figsize=(NEURIPS_TEXT_WIDTH_IN, 8.8))

# Axes: [left, bottom, width, height]
ax_a = fig.add_axes([0.24, 0.60, 0.70, 0.34])    # Panel A scorecard
ax_b = fig.add_axes([0.24, 0.08, 0.57, 0.31])    # Panel B heatmap
ax_cb = fig.add_axes([0.835, 0.10, 0.017, 0.27])  # colorbar

# ===================================================================
# PANEL A: Pooled Scorecard
# ===================================================================
ax_a.set_xlim(-0.5, n_cfg - 0.5)
ax_a.set_ylim(-0.5, n_bench - 0.5)
ax_a.invert_yaxis()
ax_a.axis("off")

# Column headers
for j, cfg in enumerate(SCAFFOLD_CONFIGS):
    ax_a.text(j, -0.78, SCAFFOLD_LABELS[cfg],
              ha="center", va="bottom",
              fontsize=FONT_SIZE_AXIS_LABEL, fontweight="bold")

# Row headers
for i, bench in enumerate(BENCH_ORDER):
    ax_a.text(-0.65, i, BENCH_LABELS_A[bench],
              ha="right", va="center",
              fontsize=FONT_SIZE_TICK, linespacing=1.15)

# Draw cells
pad = 0.05
for i, bench in enumerate(BENCH_ORDER):
    for j, cfg in enumerate(SCAFFOLD_CONFIGS):
        rd_pp, lo_pp, hi_pp = pooled[(bench, cfg)]
        fc = cell_color(rd_pp)
        tc = text_color(rd_pp)

        # Background rectangle
        rect = mpatches.FancyBboxPatch(
            (j - 0.5 + pad, i - 0.5 + pad),
            1.0 - 2 * pad, 1.0 - 2 * pad,
            boxstyle="round,pad=0.03", facecolor=fc, edgecolor="#aaaaaa",
            linewidth=0.6, zorder=2)
        ax_a.add_patch(rect)

        # Delta value
        sign = "+" if rd_pp >= 0 else ""
        ax_a.text(j, i - 0.10,
                  f"{sign}{rd_pp:.1f} pp",
                  ha="center", va="center",
                  fontsize=13, fontweight="bold", color=tc, zorder=3)

        # 95% CI
        ax_a.text(j, i + 0.25,
                  f"[{lo_pp:+.1f}, {hi_pp:+.1f}]",
                  ha="center", va="center",
                  fontsize=FONT_SIZE_ANNOTATION, color=tc, alpha=0.7,
                  zorder=3)

        # Significance marker (dagger if >= 3/6 models individually significant)
        sl = sig_lookup.get((bench, cfg), {"n_sig": 0})
        if sl.get("n_sig", 0) >= 3:
            ax_a.text(j + 0.40, i - 0.40, "\u2020",
                      ha="right", va="top", fontsize=9,
                      color=tc, alpha=0.55, zorder=3)

# Title
ax_a.text(-0.65, -1.25,
          "A   Pooled Safety Change vs. Direct Baseline (6 Models, $N$=62,808)",
          fontsize=FONT_SIZE_TITLE, fontweight="bold", va="bottom")

# Legend -- positioned between panels using figure coordinates
legend_items = [
    mpatches.Patch(facecolor="#c7e9c0", edgecolor="#aaa",
                   label="Equivalent ($\\geq$ \u22122 pp)"),
    mpatches.Patch(facecolor="#ffffb2", edgecolor="#aaa",
                   label="Small (\u22122 to \u22125 pp)"),
    mpatches.Patch(facecolor="#fdae61", edgecolor="#aaa",
                   label="Moderate (\u22125 to \u221210 pp)"),
    mpatches.Patch(facecolor="#f46d43", edgecolor="#aaa",
                   label="Large (< \u221210 pp)"),
]
leg = fig.legend(handles=legend_items,
                 loc="center", bbox_to_anchor=(0.56, 0.56),
                 ncol=4, fontsize=7, frameon=True,
                 handlelength=1.0, handletextpad=0.3, columnspacing=0.7,
                 edgecolor="#cccccc", fancybox=True)


# ===================================================================
# PANEL B: Per-Model Heatmap
# ===================================================================
matrix = np.full((n_bench, n_cols_b), np.nan)
for i, bench in enumerate(BENCH_ORDER):
    for j, cfg in enumerate(SCAFFOLD_CONFIGS):
        for k, model in enumerate(MODEL_ORDER):
            col = j * n_model + k
            matrix[i, col] = per_model.get((model, bench, cfg), np.nan)

# Colormap
vmin, vmax = -40, 17
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.cm.RdYlGn

# Draw heatmap using vector Rectangle patches instead of rasterized imshow
for i_row in range(n_bench):
    for j_col in range(n_cols_b):
        val = matrix[i_row, j_col]
        if np.isnan(val):
            fc = "#CCCCCC"
        else:
            fc = cmap(norm(val))
        rect = mpatches.FancyBboxPatch(
            (j_col - 0.5, i_row - 0.5), 1.0, 1.0,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor="none", zorder=1,
        )
        ax_b.add_patch(rect)
ax_b.set_xlim(-0.5, n_cols_b - 0.5)
ax_b.set_ylim(n_bench - 0.5, -0.5)
# ScalarMappable for colorbar
import matplotlib.cm as mcm
im = mcm.ScalarMappable(cmap=cmap, norm=norm)
im.set_array([])

# Y-axis
ax_b.set_yticks(range(n_bench))
ax_b.set_yticklabels([BENCH_LABELS_B[b] for b in BENCH_ORDER],
                     fontsize=FONT_SIZE_TICK - 1)

# X-axis
x_labels = []
for cfg in SCAFFOLD_CONFIGS:
    for model in MODEL_ORDER:
        x_labels.append(MODEL_SHORT[model])
ax_b.set_xticks(range(n_cols_b))
ax_b.set_xticklabels(x_labels, fontsize=6.5, rotation=50, ha="right")

# Config group headers (above heatmap)
for j, cfg in enumerate(SCAFFOLD_CONFIGS):
    cx = j * n_model + (n_model - 1) / 2
    ax_b.text(cx, -0.95, SCAFFOLD_LABELS[cfg],
              ha="center", va="bottom", fontsize=FONT_SIZE_TICK,
              fontweight="bold", clip_on=False)
    # Bracket
    xl = j * n_model - 0.4
    xr = (j + 1) * n_model - 0.6
    yb = -0.62
    ax_b.plot([xl, xl, xr, xr], [yb + 0.08, yb, yb, yb + 0.08],
              color="#555", linewidth=0.6, clip_on=False)

# Vertical group separators
for j in range(1, n_cfg):
    xv = j * n_model - 0.5
    ax_b.axvline(xv, color="white", linewidth=3)
    ax_b.axvline(xv, color="#555", linewidth=0.7)

# Horizontal separators
for i in range(1, n_bench):
    ax_b.axhline(i - 0.5, color="white", linewidth=1.2)

# Cell text
for i in range(n_bench):
    for jc in range(n_cols_b):
        val = matrix[i, jc]
        if np.isnan(val):
            continue
        sign = "+" if val >= 0 else ""
        tc = "white" if val < -18 else "black"
        ax_b.text(jc, i, f"{sign}{val:.0f}",
                  ha="center", va="center", fontsize=6.5,
                  color=tc, fontweight="medium")

# Colorbar
cbar = fig.colorbar(im, cax=ax_cb)
cbar.set_label("$\\Delta$ Safety Rate (pp)", fontsize=FONT_SIZE_ANNOTATION)
cbar.ax.tick_params(labelsize=6.5)

# Title
ax_b.text(-0.5, -1.65,
          "B   Per-Model Safety Change by Benchmark ($\\Delta$ pp vs. Direct)",
          fontsize=FONT_SIZE_TITLE, fontweight="bold",
          va="bottom", ha="left", clip_on=False)

# Cleanup
ax_b.tick_params(axis="both", which="both", length=0)
for sp in ax_b.spines.values():
    sp.set_visible(False)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = Path(__file__).parent / "output"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = str(out / f"safety_scorecard.{ext}")
    fig.savefig(p, format=ext, dpi=300, bbox_inches="tight", pad_inches=0.12)
    print(f"  Saved: {p}")
plt.close(fig)
print("\nDone.")
