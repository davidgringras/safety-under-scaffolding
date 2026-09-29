#!/usr/bin/env python3
"""
Matched option-preserving map-reduce comparison
===============================================

The option-preserving map-reduce arm appends the multiple-choice options to every worker prompt. It was run on 200
cases (100 BBQ and 100 TruthfulQA) for five models (not Gemini 3 Pro); its responses are in
analysis/outputs/pilot71_final_checkpoint.jsonl, one record per model and case (a result file available from the
author on request). For each model this script compares the three arms on the same 200 cases:

  direct, map-reduce     the case's label in the dataset (SUS_CANONICAL; default the canonical dataset);
  option-preserving      the stored response re-scored with the canonical multiple-choice extractor
                         (extract_mc_answer in analysis/build_canonical_dataset.py) against the case's correct letter,

so the three arms share one scorer, and a response whose answer does not parse counts as unsafe in every arm. Per model
it reports the safe counts of the three arms, OP minus MR in percentage points and the recovery (OP - MR) / (Direct -
MR), each with a 95% interval from a paired case bootstrap (10,000 resamples of the 200 cases, seed 42; resamples with
Direct - MR <= 0 are left out of the recovery interval and counted), and the exact two-sided McNemar p for OP vs MR.

Usage (from the repository root):
    python analysis/option_preserving_matched.py
    SUS_CANONICAL=results/analysis_dataset.jsonl \\
    SUS_OUT_DIR=analysis/outputs/analysis_dataset python analysis/option_preserving_matched.py

Output: <SUS_OUT_DIR>/option_preserving_matched.json (default analysis/outputs/)
"""

import json
import math
import os
import random
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
from analysis.build_canonical_dataset import extract_mc_answer  # noqa: E402

# SUS_CANONICAL and SUS_OUT_DIR redirect input and output (relative paths resolve against the repository root).
CANONICAL_PATH = os.path.join(PROJECT_DIR, os.environ.get(
    "SUS_CANONICAL", os.path.join("results", "canonical_primary_dataset.jsonl")))
OUT_DIR = os.path.join(PROJECT_DIR, os.environ.get("SUS_OUT_DIR", os.path.join("analysis", "outputs")))
OP_PATH = os.path.join(PROJECT_DIR, "analysis", "outputs", "pilot71_final_checkpoint.jsonl")
N_BOOT = 10000
SEED = 42
N_CASES = 200


def mcnemar_exact(k, n):
    """Exact two-sided McNemar p: twice the smaller binomial tail of k discordant pairs out of n, capped at 1."""
    if n == 0:
        return 1.0
    m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)


def quantile(v, a):
    return v[int(a * len(v))] if a < 0.5 else v[int(a * len(v)) - 1]


def main():
    missing = [os.path.relpath(p, PROJECT_DIR) for p in (CANONICAL_PATH, OP_PATH) if not os.path.exists(p)]
    if missing:
        sys.exit("Missing input files (available from the author on request): " + ", ".join(missing))
    with open(OP_PATH) as f:
        op = [json.loads(line) for line in f]
    labels = {}
    with open(CANONICAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["config_id"] in ("direct", "map_reduce"):
                labels[(r["model_id"], r["config_id"], r["case_id"])] = bool(r["is_safe"])

    out = {}
    for model in sorted({r["model_id"] for r in op}):
        rows = []
        for r in op:                                   # file order, which fixes the bootstrap draws
            if r["model_id"] != model:
                continue
            valid = set("ABCD") if r["benchmark_id"] == "truthfulqa" else set("ABC")
            pred = extract_mc_answer(r["final_response"], valid)
            op_safe = pred is not None and pred == r["correct_answer"].strip().upper()[:1]
            d_key, m_key = (model, "direct", r["case_id"]), (model, "map_reduce", r["case_id"])
            assert d_key in labels and m_key in labels, ("case missing from the dataset", model, r["case_id"])
            rows.append((labels[d_key], labels[m_key], op_safe))
        n = len(rows)
        assert n == N_CASES, (model, n)
        d = sum(x[0] for x in rows)
        mr = sum(x[1] for x in rows)
        o = sum(x[2] for x in rows)
        b = sum(1 for x in rows if x[2] and not x[1])      # OP safe, MR unsafe
        c = sum(1 for x in rows if x[1] and not x[2])      # MR safe, OP unsafe
        rng = random.Random(SEED)
        diffs, recs, skipped = [], [], 0
        for _ in range(N_BOOT):
            s = [rows[rng.randrange(n)] for _ in range(n)]
            dd = sum(x[0] for x in s)
            mm = sum(x[1] for x in s)
            oo = sum(x[2] for x in s)
            diffs.append(100 * (oo - mm) / n)
            if dd - mm <= 0:
                skipped += 1
            else:
                recs.append((oo - mm) / (dd - mm))
        diffs.sort()
        recs.sort()
        out[model] = {
            "n_cases": n, "direct_safe": d, "map_reduce_safe": mr, "option_preserving_safe": o,
            "op_minus_mr_pp": 100 * (o - mr) / n,
            "op_minus_mr_ci95_pp": [quantile(diffs, 0.025), quantile(diffs, 0.975)],
            "direct_minus_mr_pp": 100 * (d - mr) / n,
            "recovery_pct": 100 * (o - mr) / (d - mr) if d != mr else None,
            "recovery_ci95_pct": [100 * quantile(recs, 0.025), 100 * quantile(recs, 0.975)],
            "bootstrap_resamples_direct_minus_mr_le_0": skipped,
            "mcnemar_discordant_op_safe_mr_unsafe": b, "mcnemar_discordant_mr_safe_op_unsafe": c,
            "mcnemar_exact_p": mcnemar_exact(b, b + c),
        }

    result = {
        "description": "Direct, map-reduce and option-preserving map-reduce on the same 200 cases per model, "
                       "one scorer",
        "inputs": [os.path.relpath(CANONICAL_PATH, PROJECT_DIR), os.path.relpath(OP_PATH, PROJECT_DIR)],
        "n_boot": N_BOOT, "seed": SEED, "models": out,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "option_preserving_matched.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    print(f"{'model':10s} {'direct':>6s} {'MR':>4s} {'OP':>4s}  OP-MR pp [95% CI]       "
          f"recovery % [95% CI]   McNemar p")
    for m, v in out.items():
        lo, hi = v["op_minus_mr_ci95_pp"]
        rec = v["recovery_pct"]
        rlo, rhi = v["recovery_ci95_pct"]
        print(f"{m:10s} {v['direct_safe']:6d} {v['map_reduce_safe']:4d} {v['option_preserving_safe']:4d}  "
              f"{v['op_minus_mr_pp']:+5.1f} [{lo:5.1f}, {hi:5.1f}]  "
              f"{'n/a' if rec is None else f'{rec:5.1f}'} [{rlo:6.1f}, {rhi:6.1f}]  {v['mcnemar_exact_p']:.3g}")
    print(f"Wrote {os.path.relpath(path, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
