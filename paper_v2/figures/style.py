"""
Shared style definitions for Safety Under Scaffolding figures.

Provides:
  - Colorblind-safe palettes (Okabe-Ito with luminance separation)
  - NeurIPS-compatible font sizes and figure dimensions
  - Consistent model colors, config markers, benchmark colors
  - Formatting helpers (savefig, panel labels, percentage formatting)
  - Pilot data constants for prototyping

All figure scripts import from this module to guarantee visual consistency.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# NeurIPS formatting constraints
# ---------------------------------------------------------------------------
# NeurIPS D&B template: text width ~5.5in, full page width ~6.75in
NEURIPS_TEXT_WIDTH_IN = 5.5
NEURIPS_FULL_WIDTH_IN = 6.75
NEURIPS_COL_WIDTH_IN = 3.25  # for two-column layouts

# Standard aspect ratios
ASPECT_GOLDEN = 1.618
ASPECT_WIDE = 2.0
ASPECT_SQUARE = 1.0

# ---------------------------------------------------------------------------
# Font sizes -- NeurIPS publication standard
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE = 14
FONT_SIZE_SUPTITLE = 14
FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOTATION = 8

# ---------------------------------------------------------------------------
# Line widths and marker sizes -- publication quality
# ---------------------------------------------------------------------------
LINE_WIDTH_PRIMARY = 1.5       # main data lines
LINE_WIDTH_SECONDARY = 1.0     # secondary / reference lines
LINE_WIDTH_REFERENCE = 0.7     # grid, reference dashes
LINE_WIDTH_AXES = 0.8          # axis spines
MARKER_SIZE_PRIMARY = 6        # main scatter / line markers
MARKER_SIZE_SECONDARY = 4      # secondary markers
MARKER_SIZE_LARGE = 10         # emphasis markers (Fig 3 scatter)

# ---------------------------------------------------------------------------
# Colorblind-safe palette
# Based on Okabe-Ito (2008) / Wong 2011, verified for deuteranopia,
# protanopia, and tritanopia. Distinct luminance values for greyscale.
#
# Model assignments follow the spec:
#   Opus = blue, GPT-5.2 = green, Gemini = red/orange,
#   Llama4 = purple, DeepSeek = teal
# ---------------------------------------------------------------------------

# -- Model colors (5 distinct hues with different luminance) --
MODEL_COLORS = {
    "opus":        "#0072B2",  # blue         (lum ~0.36)
    "gpt52":       "#009E73",  # bluish green (lum ~0.50)
    "gemini3pro":  "#D55E00",  # vermillion   (lum ~0.40)
    "llama4":      "#CC79A7",  # reddish purple (lum ~0.55)
    "deepseek":    "#56B4E9",  # sky blue/teal (lum ~0.65)
    "mistral":     "#E69F00",  # amber/yellow  (lum ~0.70) — exploratory
}

# -- Config colors: sequential intensity (light to dark by complexity) --
CONFIG_COLORS = {
    "direct":      "#4575B4",  # medium blue (baseline)
    "react":       "#74ADD1",  # light blue
    "multi_agent": "#F46D43",  # orange
    "map_reduce":  "#D73027",  # deep red (most complex)
}

# -- Config markers: consistent shapes for all line/scatter plots --
CONFIG_MARKERS = {
    "direct":      "o",   # circle
    "react":       "s",   # square
    "multi_agent": "D",   # diamond
    "map_reduce":  "^",   # triangle-up
}

# -- Config colors: greyscale-safe gradient --
CONFIG_COLORS_GREY = {
    "direct":      "#B0B0B0",  # lightest
    "react":       "#808080",  # medium-light
    "multi_agent": "#505050",  # medium-dark
    "map_reduce":  "#202020",  # darkest
}

# -- Benchmark colors --
BENCHMARK_COLORS = {
    "sycophancy":     "#0072B2",  # blue
    "bbq":            "#E69F00",  # orange
    "xstest_orbench": "#009E73",  # green
    "truthfulqa":     "#CC79A7",  # pink
}

# -- Error-direction colors (for systematic error analysis) --
ERROR_DIRECTION_COLORS = {
    "over_cautious":   "#4575B4",  # blue
    "over_refusal":    "#74ADD1",  # light blue
    "sycophantic":     "#F46D43",  # orange
    "misconception":   "#D73027",  # red
    "correct":         "#1A9850",  # green
    "parse_failure":   "#999999",  # grey
}

# -- Diverging colormap for heatmaps (red = bad, green = good) --
HEATMAP_CMAP = "RdYlGn"

# -- Diverging colormap for delta heatmaps centered on zero --
# Custom: use RdBu_r so that blue = improvement and red = degradation
HEATMAP_DELTA_CMAP = "RdYlGn"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "opus":        "Opus 4.6",
    "gpt52":       "GPT-5.2",
    "gemini3pro":  "Gemini 3 Pro",
    "llama4":      "Llama 4",
    "deepseek":    "DeepSeek V3.2",
    "mistral":     "Mistral Large 2",
}

MODEL_LABELS_FULL = {
    "opus":        "Claude Opus 4.6",
    "gpt52":       "GPT-5.2",
    "gemini3pro":  "Gemini 3 Pro",
    "llama4":      "Llama 4 Maverick",
    "deepseek":    "DeepSeek V3.2",
    "mistral":     "Mistral Large 2",
}

CONFIG_LABELS = {
    "direct":      "Direct API",
    "react":       "ReAct",
    "multi_agent": "Multi-Agent",
    "map_reduce":  "Map-Reduce",
}

BENCHMARK_LABELS = {
    "sycophancy":     "Sycophancy",
    "bbq":            "BBQ (Bias)",
    "xstest_orbench": "XSTest+OR-Bench",
    "truthfulqa":     "TruthfulQA",
}

BENCHMARK_LABELS_SHORT = {
    "sycophancy":     "Syco.",
    "bbq":            "BBQ",
    "xstest_orbench": "XSTest",
    "truthfulqa":     "TQA",
}

# Canonical ordering
MODEL_ORDER = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
CONFIG_ORDER = ["direct", "react", "multi_agent", "map_reduce"]
BENCHMARK_ORDER = ["sycophancy", "bbq", "xstest_orbench", "truthfulqa"]

# ---------------------------------------------------------------------------
# Pilot data (from MASTER_PLAN.md -- 4 models, Gemini excluded)
# ---------------------------------------------------------------------------
# Format: PILOT_DATA[model][config] = safety_rate (%)
PILOT_DATA = {
    "deepseek": {"direct": 77.8, "react": 82.4, "multi_agent": 74.5, "map_reduce": 55.7},
    "gpt52":    {"direct": 81.5, "react": 81.5, "multi_agent": 78.7, "map_reduce": 68.9},
    "llama4":   {"direct": 79.6, "react": 75.9, "multi_agent": 82.2, "map_reduce": 71.0},
    "opus":     {"direct": 82.4, "react": 82.4, "multi_agent": 82.4, "map_reduce": 78.5},
}

# Per-benchmark pilot data (from MASTER_PLAN trace analysis, approximate)
# These are synthetic estimates based on the pilot findings narrative
PILOT_PER_BENCHMARK = {
    "sycophancy": {
        "deepseek": {"direct": 70.0, "react": 71.0, "multi_agent": 69.5, "map_reduce": 69.0},
        "gpt52":    {"direct": 72.0, "react": 72.5, "multi_agent": 71.0, "map_reduce": 71.5},
        "llama4":   {"direct": 68.0, "react": 67.5, "multi_agent": 69.0, "map_reduce": 67.0},
        "opus":     {"direct": 70.2, "react": 70.0, "multi_agent": 70.5, "map_reduce": 69.5},
    },
    "bbq": {
        "deepseek": {"direct": 88.0, "react": 85.0, "multi_agent": 78.0, "map_reduce": 52.0},
        "gpt52":    {"direct": 85.0, "react": 84.0, "multi_agent": 80.0, "map_reduce": 65.0},
        "llama4":   {"direct": 83.0, "react": 80.0, "multi_agent": 82.0, "map_reduce": 70.0},
        "opus":     {"direct": 87.5, "react": 87.0, "multi_agent": 87.7, "map_reduce": 75.0},
    },
    "xstest_orbench": {
        "deepseek": {"direct": 75.0, "react": 80.0, "multi_agent": 68.0, "map_reduce": 55.0},
        "gpt52":    {"direct": 80.0, "react": 78.0, "multi_agent": 72.0, "map_reduce": 62.0},
        "llama4":   {"direct": 78.0, "react": 72.0, "multi_agent": 80.0, "map_reduce": 68.0},
        "opus":     {"direct": 82.0, "react": 82.0, "multi_agent": 81.0, "map_reduce": 78.0},
    },
    "truthfulqa": {
        "deepseek": {"direct": 77.5, "react": 78.0, "multi_agent": 75.0, "map_reduce": 40.0},
        "gpt52":    {"direct": 82.5, "react": 83.0, "multi_agent": 80.0, "map_reduce": 67.5},
        "llama4":   {"direct": 70.0, "react": 68.0, "multi_agent": 72.0, "map_reduce": 55.0},
        "opus":     {"direct": 97.5, "react": 97.0, "multi_agent": 98.3, "map_reduce": 69.3},
    },
}

# Gemini placeholder (experiment in progress, use mean of other models)
def _add_gemini_estimates():
    """Add Gemini 3 Pro estimates as mean of other 4 models for plotting purposes."""
    for bm in PILOT_PER_BENCHMARK:
        gemini = {}
        for cfg in CONFIG_ORDER:
            vals = [PILOT_PER_BENCHMARK[bm][m][cfg]
                    for m in ["deepseek", "gpt52", "llama4", "opus"]]
            gemini[cfg] = round(np.mean(vals), 1)
        PILOT_PER_BENCHMARK[bm]["gemini3pro"] = gemini

    # Aggregate
    gemini_agg = {}
    for cfg in CONFIG_ORDER:
        vals = [PILOT_DATA[m][cfg] for m in PILOT_DATA]
        gemini_agg[cfg] = round(np.mean(vals), 1)
    PILOT_DATA["gemini3pro"] = gemini_agg

_add_gemini_estimates()


# ---------------------------------------------------------------------------
# Matplotlib RC setup
# ---------------------------------------------------------------------------

def setup_matplotlib():
    """Configure matplotlib for publication-quality NeurIPS figures.

    Sets serif fonts (Times / Computer Modern), appropriate sizes for
    NeurIPS column width, and clean grid styling.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    rc = {
        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",

        # Font sizes
        "font.size": FONT_SIZE_TICK,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_AXIS_LABEL,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "figure.titlesize": FONT_SIZE_SUPTITLE,

        # Resolution
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,

        # Lines
        "lines.linewidth": LINE_WIDTH_PRIMARY,
        "lines.markersize": MARKER_SIZE_PRIMARY,

        # Axes
        "axes.linewidth": LINE_WIDTH_AXES,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,

        # Legend
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.5,

        # Figure
        "figure.constrained_layout.use": True,
    }
    mpl.rcParams.update(rc)


def get_figure_size(width_fraction=1.0, aspect=None):
    """Return (width, height) in inches for NeurIPS format.

    Args:
        width_fraction: Fraction of text width (1.0 = full width, 0.5 = half)
        aspect: Height/width ratio. Defaults to 1/golden ratio.

    Returns:
        (width, height) tuple in inches
    """
    if aspect is None:
        aspect = 1.0 / ASPECT_GOLDEN
    width = NEURIPS_TEXT_WIDTH_IN * width_fraction
    height = width * aspect
    return (width, height)


def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=12):
    """Add a bold panel label (A, B, C, ...) to a subplot axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


def format_pct(val, decimals=1):
    """Format a percentage value for annotation."""
    return f"{val:.{decimals}f}%"


def despine(ax, top=True, right=True, left=False, bottom=False):
    """Remove spines from an axis for cleaner look."""
    for spine, hide in [("top", top), ("right", right),
                        ("left", left), ("bottom", bottom)]:
        ax.spines[spine].set_visible(not hide)


def savefig(fig, name, output_dir=None):
    """Save figure as both PDF and PNG.

    Applies tight_layout before saving. Saves to paper/figures/output/
    by default.

    Args:
        fig: matplotlib Figure
        name: filename stem (no extension)
        output_dir: output directory (defaults to paper/figures/output/)
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pdf_path = str(Path(output_dir) / f"{name}.pdf")
    png_path = str(Path(output_dir) / f"{name}.png")

    # Apply tight layout as a final pass (if constrained_layout is off)
    try:
        fig.tight_layout()
    except Exception:
        pass  # constrained_layout already active

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.08)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")
    plt.close(fig)
