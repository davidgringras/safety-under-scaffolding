#!/usr/bin/env python3
"""
Canonical scoring comparison (Test 1f) on the analysis dataset
===============================================================

analysis/oe_scoring_validation.py (seed 2026, 50 responses per benchmark) drew 200 responses
from the canonical primary dataset and labelled each with GPT-5.2. For each response,
results/oe_scoring_validation_200item_detail.jsonl stores its key, the canonical pipeline label
(pipeline_is_safe), how that label was produced (pipeline_score_reason: answer extraction,
parse failure or LLM judge) and the GPT-5.2 label (judge_is_safe).

This script compares the GPT-5.2 labels with the labels of the analysis dataset
(results/analysis_dataset.jsonl, built by analysis/build_analysis_dataset.py) on the responses
the analysis dataset retains, and with the stored canonical labels on all 200 as drawn. For
each comparison it reports agreement (the same binary label), Cohen's kappa as computed by
analysis/oe_scoring_validation.py, the four cells (both safe; pipeline safe, judge unsafe;
pipeline unsafe, judge safe; both unsafe) and the pipeline-safe rate minus the judge-safe rate,
pooled and by benchmark. Responses that the analysis dataset excludes or re-scores are listed
with their class from results/analysis_dataset_ledger.jsonl.

Usage (from the repository root, after analysis/build_analysis_dataset.py, with the result
files, which are available from the author on request, in results/):
    python analysis/oe_scoring_validation_analysis_dataset.py

Output:
    analysis/outputs/analysis_dataset/oe_scoring_validation_200item.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETAIL = PROJECT_ROOT / "results" / "oe_scoring_validation_200item_detail.jsonl"
DATA = PROJECT_ROOT / "results" / "analysis_dataset.jsonl"
LEDGER = PROJECT_ROOT / "results" / "analysis_dataset_ledger.jsonl"
OUTPUT = PROJECT_ROOT / "analysis" / "outputs" / "analysis_dataset" / "oe_scoring_validation_200item.json"

BENCHMARKS = ("bbq", "truthfulqa", "sycophancy", "xstest_orbench")
N_DRAWN = 200
N_PER_BENCHMARK = 50


def key(record):
    return (record["model_id"], record["config_id"], record["benchmark_id"], record["case_id"])


def compare(pairs):
    """Agreement statistics for (pipeline label, judge label) pairs."""
    a = sum(1 for p, j in pairs if p and j)
    b = sum(1 for p, j in pairs if p and not j)
    c = sum(1 for p, j in pairs if not p and j)
    d = sum(1 for p, j in pairs if not p and not j)
    n = a + b + c + d
    p_o = (a + d) / n
    p_e = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    return {
        "n": n,
        "agree": a + d,
        "agreement_pct": 100 * p_o,
        "kappa": (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0,
        "cells": {"both_safe": a, "pipeline_safe_judge_unsafe": b,
                  "pipeline_unsafe_judge_safe": c, "both_unsafe": d},
        "pipeline_safe_minus_judge_safe_pp": 100 * (b - c) / n,
    }


def main():
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in (DETAIL, DATA, LEDGER) if not p.exists()]
    if missing:
        sys.exit("Missing input files (results/analysis_dataset*.jsonl come from "
                 "analysis/build_analysis_dataset.py; the result files are available from the author "
                 "on request): " + ", ".join(missing))
    with open(DETAIL) as f:
        detail = [json.loads(line) for line in f]
    keys = [key(r) for r in detail]
    assert len(detail) == N_DRAWN and len(set(keys)) == N_DRAWN, len(detail)
    drawn = Counter(k[2] for k in keys)
    assert all(drawn[b] == N_PER_BENCHMARK for b in BENCHMARKS) and sum(drawn.values()) == N_DRAWN, drawn
    assert all(isinstance(r["pipeline_is_safe"], bool) and isinstance(r["judge_is_safe"], bool) for r in detail)

    wanted = set(keys)
    labels = {}
    with open(DATA) as f:
        for line in f:
            r = json.loads(line)
            if key(r) in wanted:
                labels[key(r)] = bool(r["is_safe"])
    with open(LEDGER) as f:
        ledger = {key(x): x for x in map(json.loads, f) if key(x) in wanted}

    excluded, rescored = [], []
    for r, k in zip(detail, keys):
        x = ledger.get(k)
        entry = dict(zip(("model_id", "config_id", "benchmark_id", "case_id"), k))
        if k not in labels:
            assert x is not None and x["fate"] == "excluded", ("absent from the dataset without a ledger entry", k)
            excluded.append({**entry, "reason": x["reason"]})
        elif x is None:
            assert labels[k] == r["pipeline_is_safe"], ("kept record with a different label", k)
        else:
            assert x["fate"] == "rescored" and bool(x["is_safe"]) == labels[k], k
            assert bool(x["old_is_safe"]) == r["pipeline_is_safe"], k
            rescored.append({**entry, "reason": x["reason"], "canonical_is_safe": r["pipeline_is_safe"],
                             "analysis_dataset_is_safe": labels[k], "judge_is_safe": r["judge_is_safe"]})

    retained = [(r, k) for r, k in zip(detail, keys) if k in labels]

    def composition(rows):
        return {
            "n": len(rows),
            "by_benchmark": {b: sum(1 for r in rows if r["benchmark_id"] == b) for b in BENCHMARKS},
            "by_model": dict(sorted(Counter(r["model_id"] for r in rows).items())),
            "by_config": dict(sorted(Counter(r["config_id"] for r in rows).items())),
            "pipeline_score_reason": dict(sorted(Counter(r["pipeline_score_reason"] for r in rows).items())),
            "preview_is_complete_response": sum(1 for r in rows if len(r["response_preview"]) == r["response_length"]),
        }

    out = {
        "description": "Canonical scoring comparison (Test 1f): GPT-5.2 labels vs pipeline labels, on all 200 "
                       "responses as drawn (stored canonical labels) and on the responses the analysis dataset "
                       "retains (analysis dataset labels)",
        "sample_producer": "analysis/oe_scoring_validation.py",
        "inputs": ["results/oe_scoring_validation_200item_detail.jsonl", "results/analysis_dataset.jsonl",
                   "results/analysis_dataset_ledger.jsonl"],
        "drawn": composition(detail),
        "retained": composition([r for r, _ in retained]),
        "not_retained": excluded,
        "rescored": rescored,
        "as_drawn": compare([(r["pipeline_is_safe"], r["judge_is_safe"]) for r in detail]),
        "analysis_dataset": compare([(labels[k], r["judge_is_safe"]) for r, k in retained]),
        "by_benchmark": {
            b: {"as_drawn": compare([(r["pipeline_is_safe"], r["judge_is_safe"])
                                     for r in detail if r["benchmark_id"] == b]),
                "analysis_dataset": compare([(labels[k], r["judge_is_safe"])
                                             for r, k in retained if k[2] == b])}
            for b in BENCHMARKS},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    def line(s):
        c = s["cells"]
        return (f"{s['agree']}/{s['n']} ({s['agreement_pct']:.0f}%), kappa {s['kappa']:.3f}, cells "
                f"({c['both_safe']},{c['pipeline_safe_judge_unsafe']},{c['pipeline_unsafe_judge_safe']},"
                f"{c['both_unsafe']}), pipeline-safe minus judge-safe {s['pipeline_safe_minus_judge_safe_pp']:+.1f} pp")

    print(f"Drawn: {N_DRAWN}; retained by the analysis dataset: {len(retained)}; "
          f"excluded: {len(excluded)}; re-scored: {len(rescored)}")
    for x in excluded:
        print(f"  excluded  {x['model_id']} {x['config_id']} {x['case_id']}: {x['reason']}")
    for x in rescored:
        print(f"  re-scored {x['model_id']} {x['config_id']} {x['case_id']}: {x['reason']}; "
              f"{x['canonical_is_safe']} -> {x['analysis_dataset_is_safe']} (GPT-5.2: {x['judge_is_safe']})")
    print(f"As drawn:          {line(out['as_drawn'])}")
    print(f"Analysis dataset:  {line(out['analysis_dataset'])}")
    for b in BENCHMARKS:
        print(f"  {b:15s} {line(out['by_benchmark'][b]['analysis_dataset'])}")
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
