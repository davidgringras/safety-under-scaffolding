#!/usr/bin/env python3
"""
Figure 11: Phase 2 Confirmatory Dose-Response

Two-panel figure showing safety rates across four prompt-intensity conditions
(passthrough, minimal, moderate, aggressive) for all six models.
  Left panel: BBQ accuracy
  Right panel: TruthfulQA accuracy

Data source: results/phase2_confirmatory.jsonl

Usage:
    python fig_phase2_dose_response.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Project root for importing scoring utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.production_frameworks.score_results import extract_mc_answer
from style import (
    setup_matplotlib, savefig, add_panel_label,
    MODEL_LABELS, MODEL_COLORS, MODEL_ORDER,
    FONT_SIZE_TITLE, FONT_SIZE_ANNOTATION,
    FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    NEURIPS_TEXT_WIDTH_IN, NEURIPS_FULL_WIDTH_IN,
    LINE_WIDTH_PRIMARY, MARKER_SIZE_PRIMARY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PHASE2_PATH = os.path.join(PROJECT_ROOT, "results", "phase2_confirmatory.jsonl")
BENCH_DIR = os.path.join(PROJECT_ROOT, "data", "benchmarks")

MODELS = ["opus", "gpt52", "gemini3pro", "llama4", "deepseek", "mistral"]
CONFIGS = ["passthrough", "seq_minimal", "seq_moderate", "seq_aggressive"]
CONFIG_DISPLAY = {
    "passthrough": "Passthrough",
    "seq_minimal": "Minimal",
    "seq_moderate": "Moderate",
    "seq_aggressive": "Aggressive",
}

# Model line styles for visual distinction
MODEL_LINESTYLES = {
    "opus": "-",
    "gpt52": "--",
    "gemini3pro": "-.",
    "llama4": ":",
    "deepseek": "-",
    "mistral": "--",
}

MODEL_MARKERS = {
    "opus": "o",
    "gpt52": "s",
    "gemini3pro": "D",
    "llama4": "^",
    "deepseek": "v",
    "mistral": "P",
}

# ---------------------------------------------------------------------------
# Data loading and scoring
# ---------------------------------------------------------------------------

def load_answer_keys():
    """Load ground-truth answer keys for BBQ and TruthfulQA."""
    # BBQ: derive correct letter from reference_answer + answer_choices
    bbq_correct = {}
    with open(os.path.join(BENCH_DIR, "bbq.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            ref = row.get("reference_answer", "").strip().lower()
            choices = row.get("metadata", {}).get("answer_choices", [])
            for j, c in enumerate(choices):
                if c.strip().lower() == ref:
                    bbq_correct[row["id"]] = chr(65 + j)
                    break

    # TruthfulQA: reference_answer is already the letter
    tqa_correct = {}
    with open(os.path.join(BENCH_DIR, "truthfulqa_mc1.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            tqa_correct[row["id"]] = row["reference_answer"].strip().upper()

    return bbq_correct, tqa_correct


def compute_rates():
    """Compute accuracy rates from phase2 confirmatory data.

    Returns dict: rates[(model, benchmark, config)] = accuracy_pct
    """
    bbq_correct, tqa_correct = load_answer_keys()

    counts = defaultdict(lambda: {"n": 0, "correct": 0})
    with open(PHASE2_PATH) as f:
        for line in f:
            r = json.loads(line)
            model = r["model_id"]
            bench = r["benchmark_id"]
            config = r["config"]
            case_id = r["case_id"]
            raw_resp = r.get("response") or ""
            extracted, _conf = extract_mc_answer(raw_resp)

            if bench == "bbq":
                correct = bbq_correct.get(case_id, "")
            elif bench == "truthfulqa":
                correct = tqa_correct.get(case_id, "")
            else:
                continue

            key = (model, bench, config)
            counts[key]["n"] += 1
            if extracted and extracted == correct:
                counts[key]["correct"] += 1

    rates = {}
    for key, v in counts.items():
        rates[key] = v["correct"] / v["n"] * 100 if v["n"] > 0 else 0.0
    return rates


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure():
    setup_matplotlib()

    import matplotlib as mpl
    mpl.rcParams["figure.constrained_layout.use"] = False

    rates = compute_rates()
    x = np.arange(len(CONFIGS))

    fig, (ax_bbq, ax_tqa) = plt.subplots(
        1, 2,
        figsize=(NEURIPS_FULL_WIDTH_IN, 4.2),
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.26, top=0.85, wspace=0.30)

    for ax, bench, title in [
        (ax_bbq, "bbq", "BBQ Accuracy (%)"),
        (ax_tqa, "truthfulqa", "TruthfulQA Accuracy (%)"),
    ]:
        for model in MODELS:
            vals = [rates.get((model, bench, cfg), 0) for cfg in CONFIGS]
            ax.plot(
                x, vals,
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=LINE_WIDTH_PRIMARY + 0.3,
                marker=MODEL_MARKERS[model],
                markersize=MARKER_SIZE_PRIMARY,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=MODEL_LABELS[model],
                zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONFIG_DISPLAY[c] for c in CONFIGS],
            fontsize=FONT_SIZE_TICK,
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(title, fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_xlabel("Prompt Intensity", fontsize=FONT_SIZE_AXIS_LABEL)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel titles
    ax_bbq.set_title("BBQ (Bias)", fontsize=FONT_SIZE_TITLE - 1, fontweight="bold", pad=10)
    ax_tqa.set_title("TruthfulQA (Truthfulness)", fontsize=FONT_SIZE_TITLE - 1, fontweight="bold", pad=10)

    # Panel labels
    add_panel_label(ax_bbq, "(a)", x=-0.10, y=1.12, fontsize=12)
    add_panel_label(ax_tqa, "(b)", x=-0.10, y=1.12, fontsize=12)

    # Shared legend BELOW the plot area (outside, no overlap)
    handles = []
    for model in MODELS:
        handles.append(
            mlines.Line2D(
                [], [],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                markersize=MARKER_SIZE_PRIMARY - 1,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=LINE_WIDTH_PRIMARY,
                label=MODEL_LABELS[model],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=FONT_SIZE_LEGEND - 1,
        frameon=True,
        edgecolor="0.8",
        framealpha=0.95,
        bbox_to_anchor=(0.52, -0.06),
        handlelength=2.0,
        columnspacing=1.0,
    )

    return fig


def main():
    fig = make_figure()
    savefig(fig, "fig_phase2_dose_response")


if __name__ == "__main__":
    main()
