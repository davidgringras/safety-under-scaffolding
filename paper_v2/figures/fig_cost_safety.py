#!/usr/bin/env python3
"""
Safety vs. Inference Cost tradeoff figure.
Each point = one model × configuration combination (24 points total).
X-axis: Inference cost per case ($), log scale.
Y-axis: Safety rate (%).
Lines connect the same model across configurations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data: 6 models × 4 configs ---
# From confirmatory_results.json (H2 model effects) and cost_analysis.json

DATA = {
    'Opus 4.6': {
        'direct':      (0.00489, 83.87),
        'react':       (0.00972, 83.61),
        'multi_agent': (0.02074, 83.45),
        'map_reduce':  (0.03080, 76.54),
    },
    'DeepSeek V3.2': {
        'direct':      (0.000037, 77.61),
        'react':       (0.000187, 77.54),
        'multi_agent': (0.000158, 77.45),
        'map_reduce':  (0.000220, 59.88),
    },
    'GPT-5.2': {
        'direct':      (0.001108, 79.75),
        'react':       (0.001351, 79.71),
        'multi_agent': (0.004817, 76.92),
        'map_reduce':  (0.008214, 65.07),
    },
    'Llama 4': {
        'direct':      (0.000065, 80.02),
        'react':       (0.000140, 81.54),
        'multi_agent': (0.000228, 83.45),
        'map_reduce':  (0.000272, 75.05),
    },
    'Mistral Large 2': {
        'direct':      (0.001191, 75.39),
        'react':       (0.001385, 74.81),
        'multi_agent': (0.005279, 71.99),
        'map_reduce':  (0.007106, 70.34),
    },
    'Gemini 3 Pro': {
        'direct':      (0.000325, 81.58),
        'react':       (0.000602, 75.96),
        'multi_agent': (0.000710, 82.00),
        'map_reduce':  (0.001418, 80.21),
    },
}

CONFIG_ORDER = ['direct', 'react', 'multi_agent', 'map_reduce']
CONFIG_LABELS = {'direct': 'Direct', 'react': 'ReAct',
                 'multi_agent': 'Multi-agent', 'map_reduce': 'Map-reduce'}
CONFIG_MARKERS = {'direct': 'o', 'react': 's', 'multi_agent': '^', 'map_reduce': 'D'}

# Color palette — distinct, colorblind-friendly
MODEL_COLORS = {
    'Opus 4.6':        '#1f77b4',
    'GPT-5.2':         '#ff7f0e',
    'Gemini 3 Pro':    '#2ca02c',
    'Llama 4':         '#d62728',
    'DeepSeek V3.2':   '#9467bd',
    'Mistral Large 2': '#8c564b',
}

fig, ax = plt.subplots(figsize=(8, 5.5))

# Plot lines connecting same model across configs
for model, configs in DATA.items():
    costs = [configs[c][0] for c in CONFIG_ORDER]
    rates = [configs[c][1] for c in CONFIG_ORDER]
    ax.plot(costs, rates, '-', color=MODEL_COLORS[model], alpha=0.4, linewidth=1.2, zorder=1)

# Plot points
for model, configs in DATA.items():
    for cfg in CONFIG_ORDER:
        cost, rate = configs[cfg]
        ax.scatter(cost, rate, marker=CONFIG_MARKERS[cfg], color=MODEL_COLORS[model],
                   s=60, zorder=3, edgecolors='white', linewidths=0.5)

# Legend: models (colors)
model_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=MODEL_COLORS[m],
                             markersize=8, label=m) for m in DATA.keys()]

# Legend: configs (shapes)
config_handles = [plt.Line2D([0], [0], marker=CONFIG_MARKERS[c], color='w',
                              markerfacecolor='grey', markersize=8,
                              label=CONFIG_LABELS[c]) for c in CONFIG_ORDER]

# Two-column legend
leg1 = ax.legend(handles=model_handles, loc='lower left', fontsize=7,
                 title='Model', title_fontsize=7.5, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=config_handles, loc='lower right', fontsize=7,
          title='Configuration', title_fontsize=7.5, framealpha=0.9)

ax.set_xscale('log')
ax.set_xlabel('Inference cost per case ($)', fontsize=10)
ax.set_ylabel('Safety rate (%)', fontsize=10)
ax.set_title('Safety Rate vs. Inference Cost', fontsize=11, fontweight='bold')

# Annotate the map-reduce cluster
ax.annotate('Map-reduce:\nhigher cost,\nlower safety',
            xy=(0.008, 65), fontsize=7, fontstyle='italic', color='#555555',
            ha='center')

ax.set_ylim(55, 88)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.tick_params(labelsize=8)

plt.tight_layout()
outpath = Path(__file__).parent / 'output' / 'fig_cost_safety.pdf'
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=300, bbox_inches='tight')
print(f"Saved to {outpath}")

# Also save PNG for quick review
plt.savefig(outpath.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Saved PNG to {outpath.with_suffix('.png')}")
