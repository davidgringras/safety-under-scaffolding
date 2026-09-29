#!/usr/bin/env python3
"""
Input records for the exploratory specification curve on a dataset
===================================================================

analysis/spec_curve_analysis.py scores raw response records itself (multiple-choice extraction, and a keyword rule
for XSTest/OR-Bench), and one of its analytic choices adds long-context map-reduce records. This script writes its
input for the dataset in SUS_CANONICAL (default the canonical dataset; the check passes the analysis dataset), for
the models in SUS_SPEC_CURVE_MODELS (comma-separated; default every model in the dataset):

  short-context records: exactly the dataset's records for those models, one per model, configuration, benchmark and
      case. Each is the key's source record (BBQ, TruthfulQA and XSTest/OR-Bench: the short-context success record
      in results/experiment_results_clean.jsonl; sycophancy: the success record in
      results/sycophancy_primary_results.jsonl for the item set in data/benchmarks/sycophancy_eval_exp4.jsonl),
      with sanitized_response set to the response stored in the dataset, the text its label refers to;
  long-context records: the long-context map-reduce success records in results/experiment_results_clean.jsonl for
      BBQ, TruthfulQA and XSTest/OR-Bench, one per key, without those whose sanitized response is empty. Long-context
      sycophancy records are left out: they use the original sycophancy items, not the item set of the dataset.

Usage (from the repository root):
    SUS_CANONICAL=results/analysis_dataset.jsonl \\
    SUS_SPEC_CURVE_INPUT=results/spec_curve_input_analysis_dataset.jsonl \\
    SUS_SPEC_CURVE_MODELS=deepseek,gemini3pro,gpt52,llama4,opus python analysis/build_spec_curve_input.py

Output: SUS_SPEC_CURVE_INPUT (default results/spec_curve_input.jsonl), read by analysis/spec_curve_analysis.py
through the same variable.
"""

import collections
import json
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Relative paths resolve against the repository root.
CANONICAL_PATH = os.path.join(PROJECT_DIR, os.environ.get(
    "SUS_CANONICAL", os.path.join("results", "canonical_primary_dataset.jsonl")))
OUT_PATH = os.path.join(PROJECT_DIR, os.environ.get(
    "SUS_SPEC_CURVE_INPUT", os.path.join("results", "spec_curve_input.jsonl")))
CLEAN_PATH = os.path.join(PROJECT_DIR, "results", "experiment_results_clean.jsonl")
SYCOPHANCY_PATH = os.path.join(PROJECT_DIR, "results", "sycophancy_primary_results.jsonl")
EXP4_ITEMS = os.path.join(PROJECT_DIR, "data", "benchmarks", "sycophancy_eval_exp4.jsonl")


def main():
    missing = [os.path.relpath(p, PROJECT_DIR) for p in (CANONICAL_PATH, CLEAN_PATH, SYCOPHANCY_PATH)
               if not os.path.exists(p)]
    if missing:
        sys.exit("Missing input files (available from the author on request): " + ", ".join(missing))
    with open(CANONICAL_PATH) as f:
        dataset = [json.loads(line) for line in f]
    models = os.environ.get("SUS_SPEC_CURVE_MODELS")
    keep = set(models.split(",")) if models else {r["model_id"] for r in dataset}
    key = lambda r: (r["model_id"], r["config_id"], r["benchmark_id"], r["case_id"])
    wanted = {key(r): r for r in dataset if r["model_id"] in keep}

    short_src, long_rows = {}, []
    with open(CLEAN_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "success" or r["model_id"] not in keep:
                continue
            k = key(r)
            if r.get("context_condition") == "short" and k in wanted:
                short_src.setdefault(k, r)
            elif (r.get("context_condition") == "long" and r["config_id"] == "map_reduce"
                  and r["benchmark_id"] != "sycophancy"):
                long_rows.append(r)
    with open(EXP4_ITEMS) as f:
        exp4 = {json.loads(line)["id"] for line in f}
    syc_src, seen = {}, set()
    with open(SYCOPHANCY_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "success" or r.get("case_id") not in exp4:
                continue
            k = (r["model_id"], r["config_id"], "sycophancy", r["case_id"])
            if k in seen:
                continue
            seen.add(k)
            if k in wanted:
                syc_src[k] = r

    out, counts = [], collections.Counter()
    for k, rec in wanted.items():
        src = syc_src.get(k) if k[2] == "sycophancy" else short_src.get(k)
        assert src is not None, ("no source record", k)
        row = dict(src)
        counts["short, stored response equal" if (src.get("sanitized_response") or "") == rec["response"]
               else "short, stored response replaced"] += 1
        row["sanitized_response"] = rec["response"]
        row["context_condition"] = "short"
        row["status"] = "success"
        out.append(row)
    seen_long = set()
    for r in long_rows:
        k = key(r)
        if k in seen_long:
            counts["long, duplicate left out"] += 1
            continue
        seen_long.add(k)
        if not (r.get("sanitized_response") or "").strip():
            counts["long, empty response left out"] += 1
            continue
        out.append(r)
        counts["long"] += 1
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    for c, n in sorted(counts.items()):
        print(f"{c}: {n:,}")
    print(f"Wrote {len(out):,} records to {os.path.relpath(OUT_PATH, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
