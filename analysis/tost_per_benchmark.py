#!/usr/bin/env python3
"""
Per-benchmark equivalence tests (TOST at a 2 pp margin)
=======================================================

analysis/cluster_bootstrap_tost.py tests the pooled H1 comparisons; this script tests each scaffold against direct
within each benchmark. For a benchmark and scaffold, the risk difference (scaffold minus direct, percentage points) is
taken over the model-case pairs observed in both arms; its 90% interval comes from a case-cluster bootstrap that
resamples the benchmark's cases with replacement, each case carrying its paired model differences (B = 2,000, seed
42). The comparison is equivalent at the pre-registered margin when the 90% interval lies inside (-2, +2) pp.

Usage (from the repository root):
    python analysis/tost_per_benchmark.py
    SUS_CANONICAL=results/analysis_dataset.jsonl \\
    SUS_OUT_DIR=analysis/outputs/analysis_dataset python analysis/tost_per_benchmark.py

Output: <SUS_OUT_DIR>/tost_per_benchmark.json (default analysis/outputs/)
"""

import json
import os
import random
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# SUS_CANONICAL and SUS_OUT_DIR redirect input and output (relative paths resolve against the repository root).
CANONICAL_PATH = os.path.join(PROJECT_DIR, os.environ.get(
    "SUS_CANONICAL", os.path.join("results", "canonical_primary_dataset.jsonl")))
OUT_DIR = os.path.join(PROJECT_DIR, os.environ.get("SUS_OUT_DIR", os.path.join("analysis", "outputs")))
SCAFFOLDS = ("react", "multi_agent", "map_reduce")
MARGIN_PP = 2.0
N_BOOT = 2000
SEED = 42


def main():
    if not os.path.exists(CANONICAL_PATH):
        sys.exit("Missing input file (available from the author on request): "
                 + os.path.relpath(CANONICAL_PATH, PROJECT_DIR))
    safe = {}
    with open(CANONICAL_PATH) as f:
        for line in f:
            r = json.loads(line)
            key = (r["benchmark_id"], r["case_id"], r["model_id"], r["config_id"])
            safe.setdefault(key, int(bool(r["is_safe"])))      # first record per key
    benchmarks = sorted({k[0] for k in safe})
    out = {}
    for bench in benchmarks:
        cases = sorted({k[1] for k in safe if k[0] == bench})
        models = sorted({k[2] for k in safe if k[0] == bench})
        for cfg in SCAFFOLDS:
            s, n = [], []                                     # per case: sum of paired differences, number of pairs
            for case in cases:
                diffs = [safe[(bench, case, m, cfg)] - safe[(bench, case, m, "direct")] for m in models
                         if (bench, case, m, cfg) in safe and (bench, case, m, "direct") in safe]
                s.append(sum(diffs))
                n.append(len(diffs))
            L = len(cases)
            rd = 100 * sum(s) / sum(n)
            rng = random.Random(SEED)
            boot = []
            for _ in range(N_BOOT):
                idx = [rng.randrange(L) for _ in range(L)]
                boot.append(100 * sum(s[i] for i in idx) / sum(n[i] for i in idx))
            boot.sort()
            lo, hi = boot[int(0.05 * N_BOOT)], boot[int(0.95 * N_BOOT) - 1]
            out[f"{bench}|{cfg}"] = {"rd_pp": rd, "ci90_pp": [lo, hi], "equivalent": lo > -MARGIN_PP and hi < MARGIN_PP,
                                     "n_pairs": sum(n), "n_cases": L}

    result = {
        "description": "Per-benchmark TOST: scaffold minus direct risk difference with a 90% case-cluster bootstrap "
                       "interval; equivalent when the interval lies inside (-2, +2) pp",
        "input": os.path.relpath(CANONICAL_PATH, PROJECT_DIR),
        "margin_pp": MARGIN_PP, "n_boot": N_BOOT, "seed": SEED, "comparisons": out,
        "n_equivalent": sum(v["equivalent"] for v in out.values()), "n_comparisons": len(out),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "tost_per_benchmark.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    for k, v in out.items():
        bench, cfg = k.split("|")
        lo, hi = v["ci90_pp"]
        print(f"{bench:15s} {cfg:12s} RD {v['rd_pp']:+6.2f}  90% [{lo:+6.2f}, {hi:+6.2f}]"
              f"{'  equivalent' if v['equivalent'] else ''}")
    print(f"{result['n_equivalent']} of {result['n_comparisons']} comparisons equivalent at +/-{MARGIN_PP:g} pp. "
          f"Wrote {os.path.relpath(path, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
