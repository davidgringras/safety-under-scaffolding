"""
Export and reporting utilities.

Generates standardised output files from evaluation results:
- JSON results archive
- CSV safety rate tables
- Text and JSON scorecards
- LaTeX table fragments
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scaffold_safety.analysis.scorecard import ScaffoldSafetyScorecard
from scaffold_safety.analysis.statistics import SafetyRate


def export_results(
    results: list[dict[str, Any]],
    rates: list[SafetyRate],
    output_dir: Path,
) -> dict[str, Path]:
    """Export evaluation results to standardised files.

    Parameters
    ----------
    results : list[dict]
        Raw scored results.
    rates : list[SafetyRate]
        Computed safety rates.
    output_dir : Path
        Output directory.

    Returns
    -------
    dict[str, Path]
        Mapping of file type to path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Raw results as JSONL
    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    paths["results_jsonl"] = results_path

    # Safety rates as CSV
    rates_path = output_dir / "safety_rates.csv"
    with open(rates_path, "w") as f:
        f.write("benchmark_id,model_id,config_id,n_total,n_safe,safety_rate,se\n")
        for r in rates:
            f.write(
                f"{r.benchmark_id},{r.model_id},{r.config_id},"
                f"{r.n_total},{r.n_safe},{r.safety_rate:.6f},{r.se:.6f}\n"
            )
    paths["safety_rates_csv"] = rates_path

    # Summary JSON
    models = sorted({r.model_id for r in rates})
    configs = sorted({r.config_id for r in rates})
    benchmarks = sorted({r.benchmark_id for r in rates})

    summary = {
        "n_results": len(results),
        "n_rated": len(rates),
        "models": models,
        "configs": configs,
        "benchmarks": benchmarks,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    paths["summary_json"] = summary_path

    return paths


def export_scorecard(
    scorecard: ScaffoldSafetyScorecard,
    output_dir: Path,
) -> dict[str, Path]:
    """Export a scorecard to text and JSON formats.

    Parameters
    ----------
    scorecard : ScaffoldSafetyScorecard
        The scorecard to export.
    output_dir : Path
        Output directory.

    Returns
    -------
    dict[str, Path]
        Mapping of format to path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # JSON
    json_path = output_dir / f"scorecard_{scorecard.model_id}.json"
    with open(json_path, "w") as f:
        json.dump(scorecard.to_dict(), f, indent=2)
    paths["json"] = json_path

    # Text
    text_path = output_dir / f"scorecard_{scorecard.model_id}.txt"
    with open(text_path, "w") as f:
        f.write(scorecard.to_text())
    paths["text"] = text_path

    return paths


def generate_latex_table(rates: list[SafetyRate]) -> str:
    """Generate a LaTeX table fragment from safety rates.

    Returns a complete tabular environment ready for inclusion
    in a LaTeX document.
    """
    config_labels = {
        "direct": "Direct API", "react": "ReAct",
        "multi_agent": "Multi-Agent", "map_reduce": "Map-Reduce",
    }

    benchmarks = sorted({r.benchmark_id for r in rates})
    models = sorted({r.model_id for r in rates})
    configs = sorted({r.config_id for r in rates})

    # Build lookup
    lookup: dict[tuple[str, str, str], SafetyRate] = {
        (r.benchmark_id, r.model_id, r.config_id): r for r in rates
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Safety rates (\%) by model, configuration, and benchmark.}",
        r"\label{tab:safety_rates}",
        r"\small",
        r"\begin{tabular}{ll" + "r" * len(configs) + r"r}",
        r"\toprule",
    ]

    header = r"Benchmark & Model"
    for cid in configs:
        header += f" & {config_labels.get(cid, cid)}"
    header += r" & $\Delta_{\max}$ \\"
    lines.append(header)
    lines.append(r"\midrule")

    for bid in benchmarks:
        first = True
        for mid in models:
            bname = bid if first else ""
            first = False
            row = f"{bname} & {mid}"
            rate_vals = []
            for cid in configs:
                key = (bid, mid, cid)
                if key in lookup:
                    rate = lookup[key].safety_rate * 100
                    rate_vals.append(rate)
                    row += f" & {rate:.1f}"
                else:
                    row += r" & ---"
            if len(rate_vals) >= 2:
                delta = max(rate_vals) - min(rate_vals)
                row += f" & {delta:.1f}"
            else:
                row += r" & ---"
            row += r" \\"
            lines.append(row)
        lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)
