#!/usr/bin/env python3
"""
Compute inter-rater agreement between human ratings and automated scores.

Reads the completed rating_sheet.csv and answer_key.jsonl, then computes:
- Cohen's kappa (overall and per-benchmark)
- Percent agreement (overall and per-benchmark)
- Detailed disagreement analysis
- Confusion matrix

Usage:
    python3 analysis/human_eval/compute_agreement.py
    python3 analysis/human_eval/compute_agreement.py --rating-sheet PATH
    python3 analysis/human_eval/compute_agreement.py --multi-rater  # for 2+ raters
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RATING_SHEET = SCRIPT_DIR / "rating_sheet.csv"
ANSWER_KEY_FILE = SCRIPT_DIR / "answer_key.jsonl"
CASES_FILE = SCRIPT_DIR / "cases.jsonl"
OUTPUT_REPORT = SCRIPT_DIR / "agreement_report.txt"

BENCHMARKS = ["bbq", "truthfulqa", "xstest_orbench", "sycophancy"]


# ---------------------------------------------------------------------------
# Cohen's kappa implementation (standalone, no scipy/sklearn dependency)
# ---------------------------------------------------------------------------

def cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Compute Cohen's kappa for binary ratings (0/1).

    Parameters
    ----------
    rater1, rater2 : list[int]
        Binary labels (0 or 1) from each rater, aligned by index.

    Returns
    -------
    float
        Cohen's kappa coefficient. Returns 0.0 if undefined (e.g., all same label).
    """
    n = len(rater1)
    if n == 0:
        return 0.0

    # Build confusion matrix
    a = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 1)
    b = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 1 and r2 == 0)
    c = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 1)
    d = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == 0 and r2 == 0)

    # Observed agreement
    p_o = (a + d) / n

    # Expected agreement by chance
    p_yes = ((a + b) / n) * ((a + c) / n)
    p_no = ((c + d) / n) * ((b + d) / n)
    p_e = p_yes + p_no

    if p_e >= 1.0:
        return 0.0  # Undefined; all agreement is by chance

    kappa = (p_o - p_e) / (1.0 - p_e)
    return kappa


def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value using Landis & Koch (1977) scale."""
    if kappa >= 0.81:
        return "almost perfect"
    elif kappa >= 0.61:
        return "substantial"
    elif kappa >= 0.41:
        return "moderate"
    elif kappa >= 0.21:
        return "fair"
    elif kappa >= 0.0:
        return "slight"
    else:
        return "poor (less than chance)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_answer_key() -> dict[str, dict[str, Any]]:
    """Load answer_key.jsonl, keyed by case_id."""
    keys = {}
    with open(ANSWER_KEY_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys[obj["case_id"]] = obj
    return keys


def load_cases() -> dict[str, dict[str, Any]]:
    """Load cases.jsonl, keyed by case_id."""
    cases = {}
    with open(CASES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases[obj["case_id"]] = obj
    return cases


def load_ratings(rating_sheet_path: Path) -> dict[str, dict[str, Any]]:
    """Load human ratings from completed rating_sheet.csv.

    Returns dict keyed by case_id with human_score (int) and confidence.
    Skips rows where human_score is empty.
    """
    ratings = {}
    with open(rating_sheet_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("case_id", "").strip()
            human_score_raw = row.get("human_score", "").strip()
            if not case_id or not human_score_raw:
                continue
            try:
                human_score = int(human_score_raw)
            except ValueError:
                # Try interpreting as safe/unsafe
                if human_score_raw.lower() in ("safe", "yes", "true", "1"):
                    human_score = 1
                elif human_score_raw.lower() in ("unsafe", "no", "false", "0"):
                    human_score = 0
                else:
                    print(f"  Warning: cannot parse human_score '{human_score_raw}' for {case_id}")
                    continue

            ratings[case_id] = {
                "human_score": human_score,
                "confidence": row.get("confidence", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
    return ratings


# ---------------------------------------------------------------------------
# Agreement computation
# ---------------------------------------------------------------------------

def compute_agreement(
    answer_key: dict[str, dict],
    ratings: dict[str, dict],
    cases: dict[str, dict],
) -> dict[str, Any]:
    """Compute agreement metrics between automated scores and human ratings."""

    # Align data
    automated_labels = []
    human_labels = []
    case_ids = []
    benchmarks = []
    disagreements = []

    for case_id in sorted(answer_key.keys()):
        if case_id not in ratings:
            continue

        auto_safe = 1 if answer_key[case_id]["automated_score_is_safe"] else 0
        human_safe = ratings[case_id]["human_score"]

        automated_labels.append(auto_safe)
        human_labels.append(human_safe)
        case_ids.append(case_id)
        benchmarks.append(answer_key[case_id]["benchmark"])

        if auto_safe != human_safe:
            case_info = cases.get(case_id, {})
            disagreements.append({
                "case_id": case_id,
                "benchmark": answer_key[case_id]["benchmark"],
                "model_id": answer_key[case_id]["model_id"],
                "config_id": answer_key[case_id]["config_id"],
                "automated_score": auto_safe,
                "human_score": human_safe,
                "confidence": ratings[case_id].get("confidence", ""),
                "notes": ratings[case_id].get("notes", ""),
                "prompt_preview": case_info.get("prompt", "")[:100],
                "response_preview": case_info.get("response", "")[:100],
            })

    n_rated = len(automated_labels)

    if n_rated == 0:
        return {"error": "No rated cases found. Has the rating_sheet.csv been filled in?"}

    # Overall metrics
    overall_kappa = cohens_kappa(automated_labels, human_labels)
    overall_agreement = sum(a == h for a, h in zip(automated_labels, human_labels)) / n_rated

    # Confusion matrix
    tp = sum(1 for a, h in zip(automated_labels, human_labels) if a == 1 and h == 1)
    fp = sum(1 for a, h in zip(automated_labels, human_labels) if a == 1 and h == 0)
    fn = sum(1 for a, h in zip(automated_labels, human_labels) if a == 0 and h == 1)
    tn = sum(1 for a, h in zip(automated_labels, human_labels) if a == 0 and h == 0)

    # Per-benchmark metrics
    per_benchmark = {}
    for bid in BENCHMARKS:
        indices = [i for i, b in enumerate(benchmarks) if b == bid]
        if not indices:
            continue
        auto_sub = [automated_labels[i] for i in indices]
        human_sub = [human_labels[i] for i in indices]
        n_sub = len(indices)
        kappa = cohens_kappa(auto_sub, human_sub)
        agreement = sum(a == h for a, h in zip(auto_sub, human_sub)) / n_sub

        per_benchmark[bid] = {
            "n": n_sub,
            "kappa": kappa,
            "kappa_interpretation": interpret_kappa(kappa),
            "agreement": agreement,
            "n_disagree": sum(1 for a, h in zip(auto_sub, human_sub) if a != h),
        }

    # By confidence level
    by_confidence = {}
    for conf in ["high", "medium", "low"]:
        indices = [
            i for i, cid in enumerate(case_ids)
            if ratings.get(cid, {}).get("confidence", "").lower() == conf
        ]
        if not indices:
            continue
        auto_sub = [automated_labels[i] for i in indices]
        human_sub = [human_labels[i] for i in indices]
        n_sub = len(indices)
        agreement = sum(a == h for a, h in zip(auto_sub, human_sub)) / n_sub
        by_confidence[conf] = {
            "n": n_sub,
            "agreement": agreement,
        }

    result = {
        "n_rated": n_rated,
        "n_total": len(answer_key),
        "overall_kappa": overall_kappa,
        "overall_kappa_interpretation": interpret_kappa(overall_kappa),
        "overall_agreement": overall_agreement,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "per_benchmark": per_benchmark,
        "by_confidence": by_confidence,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(result: dict[str, Any]) -> str:
    """Generate a human-readable agreement report."""
    lines = []
    lines.append("=" * 70)
    lines.append("HUMAN EVALUATION AGREEMENT REPORT")
    lines.append("Safety Under Scaffolding")
    lines.append("=" * 70)
    lines.append("")

    if "error" in result:
        lines.append(f"ERROR: {result['error']}")
        return "\n".join(lines)

    lines.append(f"Cases rated: {result['n_rated']} / {result['n_total']}")
    lines.append("")

    lines.append("--- OVERALL AGREEMENT ---")
    lines.append(f"  Cohen's kappa:     {result['overall_kappa']:.3f} ({result['overall_kappa_interpretation']})")
    lines.append(f"  Percent agreement: {result['overall_agreement']:.1%}")
    lines.append(f"  Disagreements:     {result['n_disagreements']}")
    lines.append("")

    cm = result["confusion_matrix"]
    lines.append("--- CONFUSION MATRIX (Automated vs Human) ---")
    lines.append("                    Human: SAFE    Human: UNSAFE")
    lines.append(f"  Auto: SAFE        {cm['true_positive']:>6}          {cm['false_positive']:>6}")
    lines.append(f"  Auto: UNSAFE      {cm['false_negative']:>6}          {cm['true_negative']:>6}")
    lines.append("")

    lines.append("--- PER-BENCHMARK AGREEMENT ---")
    for bid in BENCHMARKS:
        info = result.get("per_benchmark", {}).get(bid)
        if info is None:
            lines.append(f"  {bid}: no rated cases")
            continue
        lines.append(f"  {bid}:")
        lines.append(f"    n = {info['n']}")
        lines.append(f"    Kappa:     {info['kappa']:.3f} ({info['kappa_interpretation']})")
        lines.append(f"    Agreement: {info['agreement']:.1%}")
        lines.append(f"    Disagree:  {info['n_disagree']}")
    lines.append("")

    if result.get("by_confidence"):
        lines.append("--- AGREEMENT BY CONFIDENCE ---")
        for conf, info in result["by_confidence"].items():
            lines.append(f"  {conf}: {info['agreement']:.1%} agreement (n={info['n']})")
        lines.append("")

    if result["n_disagreements"] > 0:
        lines.append("--- DISAGREEMENT CASES ---")
        for d in result["disagreements"]:
            lines.append(f"  {d['case_id']} [{d['benchmark']}] "
                        f"auto={d['automated_score']} human={d['human_score']} "
                        f"conf={d['confidence']}")
            if d.get("notes"):
                lines.append(f"    Notes: {d['notes']}")
            lines.append(f"    Prompt: {d['prompt_preview']}...")
            lines.append(f"    Response: {d['response_preview']}...")
            lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute human-automated agreement for Safety Under Scaffolding"
    )
    parser.add_argument(
        "--rating-sheet", type=Path, default=DEFAULT_RATING_SHEET,
        help="Path to completed rating_sheet.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_REPORT,
        help="Path for output report file",
    )
    parser.add_argument(
        "--multi-rater", action="store_true",
        help="If set, compute inter-rater agreement between multiple human raters "
             "(expects rating_sheet_rater1.csv, rating_sheet_rater2.csv, etc.)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading answer key...")
    answer_key = load_answer_key()
    print(f"  {len(answer_key)} cases in answer key")

    print("Loading cases...")
    cases = load_cases()

    if args.multi_rater:
        # Multi-rater mode: find all rating_sheet_rater*.csv files
        rater_files = sorted(SCRIPT_DIR.glob("rating_sheet_rater*.csv"))
        if len(rater_files) < 2:
            print("ERROR: multi-rater mode requires at least 2 files named "
                  "rating_sheet_rater1.csv, rating_sheet_rater2.csv, etc.")
            sys.exit(1)

        print(f"\nFound {len(rater_files)} rater files:")
        all_ratings = {}
        for rf in rater_files:
            print(f"  Loading {rf.name}...")
            ratings = load_ratings(rf)
            all_ratings[rf.stem] = ratings
            print(f"    {len(ratings)} ratings loaded")

        # Compute pairwise agreement
        rater_names = sorted(all_ratings.keys())
        for i, r1_name in enumerate(rater_names):
            for r2_name in rater_names[i+1:]:
                print(f"\n--- {r1_name} vs {r2_name} ---")
                # Create virtual answer key from rater1 as "automated"
                r1_as_key = {}
                for cid, data in all_ratings[r1_name].items():
                    r1_as_key[cid] = {
                        "case_id": cid,
                        "benchmark": answer_key.get(cid, {}).get("benchmark", "unknown"),
                        "model_id": answer_key.get(cid, {}).get("model_id", ""),
                        "config_id": answer_key.get(cid, {}).get("config_id", ""),
                        "automated_score_is_safe": bool(data["human_score"]),
                    }
                result = compute_agreement(r1_as_key, all_ratings[r2_name], cases)
                if "error" not in result:
                    print(f"  Kappa: {result['overall_kappa']:.3f}")
                    print(f"  Agreement: {result['overall_agreement']:.1%}")

        # Also compute each rater vs automated
        for r_name in rater_names:
            print(f"\n--- {r_name} vs Automated ---")
            result = compute_agreement(answer_key, all_ratings[r_name], cases)
            report = generate_report(result)
            output_path = args.output.parent / f"agreement_report_{r_name}.txt"
            with open(output_path, "w") as f:
                f.write(report)
            print(f"  Report saved to: {output_path}")
            if "error" not in result:
                print(f"  Kappa: {result['overall_kappa']:.3f}")
                print(f"  Agreement: {result['overall_agreement']:.1%}")

    else:
        # Single rater mode
        print(f"Loading ratings from {args.rating_sheet}...")
        ratings = load_ratings(args.rating_sheet)
        print(f"  {len(ratings)} ratings loaded")

        if len(ratings) == 0:
            print("\nNo ratings found. Has the rating_sheet.csv been filled in?")
            print("The human_score column should contain 1 (SAFE) or 0 (UNSAFE).")
            sys.exit(1)

        print("\nComputing agreement...")
        result = compute_agreement(answer_key, ratings, cases)

        # Generate report
        report = generate_report(result)
        print(report)

        # Save report
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")

        # Also save raw results as JSON
        json_output = args.output.with_suffix(".json")
        with open(json_output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Raw results saved to: {json_output}")


if __name__ == "__main__":
    main()
