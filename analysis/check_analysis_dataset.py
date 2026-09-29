#!/usr/bin/env python3
"""
Check the analysis dataset against the values printed in the paper
==================================================================

One command, from the repository root, with the result files in results/, logs/
and analysis/outputs/:
    python analysis/check_analysis_dataset.py --run

--run first executes, in order,
    python analysis/build_analysis_dataset.py
    python analysis/oe_scoring_validation_analysis_dataset.py
    SUS_CANONICAL=results/analysis_dataset.jsonl \\
    SUS_OUT_DIR=analysis/outputs/analysis_dataset python analysis/variance_decomposition.py
    SUS_CANONICAL=... SUS_OUT_DIR=... python analysis/master_reanalysis.py
    SUS_CANONICAL=... SUS_OUT_DIR=... python analysis/tost_per_benchmark.py
    SUS_CANONICAL=... SUS_OUT_DIR=... python analysis/option_preserving_matched.py
    SUS_CANONICAL=... SUS_SPEC_CURVE_INPUT=... SUS_SPEC_CURVE_MODELS=... \\
    python analysis/build_spec_curve_input.py
    SUS_SPEC_CURVE_INPUT=... SUS_SPEC_CURVE_MODELS=... SUS_SPEC_CURVE_SYCOPHANCY_ITEMS=... \\
    SUS_OUT_DIR=analysis/outputs/analysis_dataset/spec_curve_exploratory python analysis/spec_curve_analysis.py
and then checks their outputs; without --run it checks existing outputs.
option_preserving_matched.py also reads analysis/outputs/pilot71_final_checkpoint.jsonl.

Every value is compared at the precision the paper prints it, rounded once
from the unrounded output:
  - the analysis dataset: 60,112 rows and its SHA-256, and the number of
    records in each exclusion and re-scoring class;
  - the factorial variance decomposition table: df, F, eta-squared (%),
    omega-squared (%) and magnitude for six sources, residual df and
    eta-squared (32 cells);
  - the per-model scaffold sensitivity table: eta-squared for scaffold (%),
    average and maximum range (pp), overall safe rate (%) for six models;
  - headline confirmatory numbers: risk differences, bootstrap CI and NNH for
    map-reduce and ReAct, and the sycophancy rates;
  - the scaffold-vs-direct comparisons: 92 reported cells, a 69-test BH family,
    25 comparisons with q < 0.05, and the risk differences and BH bounds
    printed in the text;
  - the canonical scoring comparison (Test 1f): the 200 responses as drawn
    (by benchmark, model and configuration, and previews holding the complete
    response) and, on the 198 the analysis dataset retains, how their
    pipeline labels were produced, agreement, Cohen's kappa, the four cells
    and pipeline-safe minus judge-safe rates, pooled and by benchmark;
  - the per-benchmark equivalence table: for each scaffold and benchmark,
    the risk difference against direct, its 90% interval and whether it is
    equivalent at the 2 pp margin (12 comparisons), and the number equivalent;
  - the matched option-preserving table: for five models, the safe counts of
    direct, map-reduce and option-preserving map-reduce, option-preserving
    minus map-reduce with its 95% interval, and the recovery with its 95%
    interval (not checked where the paper prints the interval as unstable);
    the recovery range over four models, the map-reduce gap of the fifth, and
    the range of the McNemar p values;
  - the primary specification curve (18 specifications): its three analytic
    choices and their options, its 54 contrasts, the median, IQR and range of
    the map-reduce odds ratio, the number (and for map-reduce the share) of
    specifications in which each scaffold is significant, case-cluster
    covariance in every fit, and, against the non-robust fit of its first
    plot, the decisions the correction changed;
  - the exploratory specification curve: 384 specifications over nine
    analytic choices and five models, the 256 with map-reduce, how many of
    those have an odds ratio below 1 and how many are significant for
    degradation, the share in which multi-agent is significant, case-cluster
    covariance in every fit, and the median and IQR of the map-reduce odds
    ratio; and in the committed summary itself, the map-reduce count
    significant with an odds ratio below 1 and the multi-agent count
    significant in either direction.
Exits non-zero on the first mismatch.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "results" / "analysis_dataset.jsonl"
LEDGER = PROJECT_ROOT / "results" / "analysis_dataset_ledger.jsonl"
OUT = PROJECT_ROOT / "analysis" / "outputs" / "analysis_dataset"
SPEC_INPUT = PROJECT_ROOT / "results" / "spec_curve_input_analysis_dataset.jsonl"
SPEC_OUT = OUT / "spec_curve_exploratory"
DATA_SHA256 = "009963f5d224ad79806707b4c7c93ea4a3a1c4d9021a75e5f4f5b6abe3f80e24"

# Records per ledger class: (reason, fate) -> count
LEDGER_COUNTS = {
    ('empty_response', 'excluded'): '625',
    ('judge_input_empty', 'rescored'): '885',
    ('judge_input_empty|rescore_failed', 'excluded'): '2',
    ('judge_input_from_other_record', 'rescored'): '110',
    ('judge_output_unparsed', 'rescored'): '1',
    ('judge_output_unparsed|rescore_failed', 'excluded'): '3',
    ('served_by_gemini_2_5_pro', 'excluded'): '2,000',
    ('sycophancy_item_mismatch', 'excluded'): '66',
}
# Factorial variance decomposition table: source -> (df, F, eta2 %, omega2 %, magnitude)
VD_TABLE = {
    'benchmark': ('3', '3,904', '15.1', '15.1', 'Large'),
    'model x benchmark': ('14', '104', '1.9', '1.9', 'Small'),
    'scaffold x benchmark': ('9', '126', '1.5', '1.5', 'Small'),
    'model': ('5', '234', '1.5', '1.5', 'Small'),
    'scaffold': ('3', '119', '0.5', '0.5', 'Negligible'),
    'model x scaffold': ('15', '25', '0.5', '0.5', 'Negligible'),
}
VD_RESIDUAL = ('60,062', '77.6')
# Per-model sensitivity table: model -> (eta2 scaffold %, avg range pp, max range pp, overall safe %),
# rounded once from unrounded values
MODEL_TABLE = {
    'deepseek': ('2.1', '27.7', '47.5', '66.8'),
    'mistral': ('0.2', '17.3', '25.6', '69.7'),
    'opus': ('2.5', '16.2', '24.7', '80.5'),
    'llama4': ('0.1', '13.6', '19.0', '68.8'),
    'gpt52': ('0.4', '14.0', '23.9', '70.5'),
    'gemini3pro': ('0.1', '6.1', '12.8', '89.2'),
}
LABELS = {"LARGE": "Large", "MEDIUM": "Medium", "SMALL": "Small", "negligible": "Negligible"}
# Scaffold-vs-direct comparisons printed in the text: (model, benchmark, comparison) -> RD (pp)
PAIRWISE_RD = {
    ('gpt52', 'bbq', 'multi_agent_vs_direct'): '-11.9',
    ('deepseek', 'bbq', 'multi_agent_vs_direct'): '-9.6',
    ('deepseek', 'truthfulqa', 'multi_agent_vs_direct'): '+9.7',
    ('deepseek', 'truthfulqa', 'map_reduce_vs_direct'): '-37.8',
    ('opus', 'truthfulqa', 'map_reduce_vs_direct'): '-24.7',
    ('gpt52', 'truthfulqa', 'map_reduce_vs_direct'): '-23.9',
}
BH_BELOW_1E5 = [('gpt52', 'bbq', 'multi_agent_vs_direct'), ('deepseek', 'bbq', 'multi_agent_vs_direct')]
# Canonical scoring comparison (Test 1f): values printed for the draw and for the retained responses
T1F = {
    "drawn": "200",
    "drawn_per_benchmark": "50",
    "drawn_by_model": {
        "deepseek": "36",
        "gemini3pro": "19",
        "gpt52": "41",
        "llama4": "31",
        "mistral": "34",
        "opus": "39"
    },
    "drawn_by_config": {
        "direct": "36",
        "map_reduce": "82",
        "multi_agent": "44",
        "react": "38"
    },
    "drawn_complete_previews": "91",
    "retained": "198",
    "agreement": "179/198",
    "agreement_pct": "90",
    "kappa": "0.790",
    "gap_pp": "+4.5",
    "cells": {
        "both_safe": "119",
        "pipeline_safe_judge_unsafe": "14",
        "pipeline_unsafe_judge_safe": "5",
        "both_unsafe": "60"
    },
    "by_benchmark": {
        "bbq": {
            "agreement": "49/50",
            "kappa": "0.929",
            "cells": "(41,0,1,8)",
            "gap_pp": "-2"
        },
        "truthfulqa": {
            "agreement": "49/50",
            "kappa": "0.947",
            "cells": "(37,0,1,12)",
            "gap_pp": "-2"
        },
        "sycophancy": {
            "agreement": "43/48",
            "kappa": "0.776",
            "cells": "(15,4,1,28)",
            "gap_pp": "+6"
        },
        "xstest_orbench": {
            "agreement": "38/50",
            "kappa": "0.493",
            "cells": "(26,10,2,12)",
            "gap_pp": "+16"
        }
    },
    "retained_score_reason": {
        "judge": "50",
        "parse_fail": "2",
        "scored": "146"
    },
    "retained_gpt52": "41"
}
# Per-benchmark equivalence table: "benchmark|scaffold" -> RD, 90% interval bounds (pp), equivalent at 2 pp
TOST_BENCH = {
    'bbq|react': {'rd': '-0.21', 'lo': '-0.72', 'hi': '+0.31', 'equivalent': True},
    'bbq|multi_agent': {'rd': '-7.55', 'lo': '-8.58', 'hi': '-6.48', 'equivalent': False},
    'bbq|map_reduce': {'rd': '-8.31', 'lo': '-9.16', 'hi': '-7.52', 'equivalent': False},
    'sycophancy|react': {'rd': '+0.28', 'lo': '-1.13', 'hi': '+1.70', 'equivalent': True},
    'sycophancy|multi_agent': {'rd': '+2.06', 'lo': '+0.85', 'hi': '+3.31', 'equivalent': False},
    'sycophancy|map_reduce': {'rd': '+0.24', 'lo': '-1.54', 'hi': '+2.07', 'equivalent': False},
    'truthfulqa|react': {'rd': '+0.63', 'lo': '-0.06', 'hi': '+1.29', 'equivalent': True},
    'truthfulqa|multi_agent': {'rd': '+4.13', 'lo': '+3.20', 'hi': '+5.12', 'equivalent': False},
    'truthfulqa|map_reduce': {'rd': '-19.35', 'lo': '-20.54', 'hi': '-18.14', 'equivalent': False},
    'xstest_orbench|react': {'rd': '-5.24', 'lo': '-6.39', 'hi': '-4.13', 'equivalent': False},
    'xstest_orbench|multi_agent': {'rd': '-0.48', 'lo': '-1.24', 'hi': '+0.28', 'equivalent': True},
    'xstest_orbench|map_reduce': {'rd': '+6.68', 'lo': '+5.04', 'hi': '+8.35', 'equivalent': False},
}
TOST_BENCH_TEXT = {'n_equivalent': '4', 'n_comparisons': '12'}
# Matched option-preserving table: model -> safe counts (direct, map-reduce, option-preserving), OP - MR (pp) and its
# 95% interval, recovery (%) and its 95% interval (None where the paper prints it as unstable); and values in the text
OP_MATCHED = {
    'opus': {'direct': '189', 'map_reduce': '157', 'option_preserving': '186', 'diff': '+14.5',
             'diff_ci': '9.5, 20.0', 'recovery': '90.6', 'recovery_ci': '75.0, 103.4'},
    'deepseek': {'direct': '162', 'map_reduce': '101', 'option_preserving': '141', 'diff': '+20.0',
                 'diff_ci': '13.0, 27.0', 'recovery': '65.6', 'recovery_ci': '47.8, 82.4'},
    'mistral': {'direct': '169', 'map_reduce': '134', 'option_preserving': '156', 'diff': '+11.0',
                'diff_ci': '5.0, 17.0', 'recovery': '62.9', 'recovery_ci': '34.9, 90.0'},
    'gpt52': {'direct': '176', 'map_reduce': '146', 'option_preserving': '158', 'diff': '+6.0',
              'diff_ci': '0.5, 11.5', 'recovery': '40.0', 'recovery_ci': '3.6, 66.7'},
    'llama4': {'direct': '158', 'map_reduce': '143', 'option_preserving': '143', 'diff': '0.0',
               'diff_ci': '-6.0, 6.0', 'recovery': '0.0', 'recovery_ci': None},
}
OP_MATCHED_TEXT = {'recovery_range': ['40', '91'], 'llama_gap': '7.5', 'mcnemar_min': '1.3e-07',
                   'mcnemar_max': '0.058', 'range_models': ['opus', 'deepseek', 'mistral', 'gpt52']}
# Specification curves: primary (18 specifications) and exploratory (384), as printed
SPEC_PRIMARY = {'n_specs': '18', 'mr_median': '0.62', 'mr_iqr': ['0.56', '0.66'], 'mr_sig': '18', 'mr_n': '18',
                'mr_range': ['0.51', '0.73'], 'ma_sig': '15', 'ma_n': '18', 'react_sig': '12', 'react_n': '18',
                'mr_sig_pct': '100', 'n_choices': '3', 'options_per_choice': ['3', '3', '2'], 'n_contrasts': '54',
                'to_significant': '13', 'to_significant_decreases': '11', 'to_significant_react_increases': '2',
                'to_not_significant': '0'}
SPEC_EXPLORATORY = {'n_specs': '384', 'n_choices': '9', 'n_models': '5', 'mr_n': '256', 'no_mr': '128',
                    'or_below_1': '249', 'sig_degradation': '233', 'sig_degradation_pct': '91.0', 'mr_median': '0.76',
                    'mr_iqr': ['0.59', '0.87'], 'models': ['deepseek', 'gemini3pro', 'gpt52', 'llama4', 'opus'],
                    'ma_sig': '285', 'ma_n': '384', 'ma_sig_pct': '74.2'}


def nonrobust_primary_curve():
    """The primary curve of analysis/master_reanalysis.py on the analysis dataset, refitted with the non-robust covariance its
    first plot had (plain logit fits): the baseline for the changes the paper reports. Nothing is written to the repository."""
    import contextlib
    import importlib.util
    import io
    import tempfile

    import statsmodels.api as sm

    sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location("_master_nonrobust", PROJECT_ROOT / "analysis" / "master_reanalysis.py")
    mr = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as tmp:
        spec.loader.exec_module(mr)
        mr.CANONICAL_PATH, mr.OUT_DIR = str(DATA), tmp
        mr.fit_case_cluster_logit = lambda y, X, clusters: sm.Logit(y, X).fit(
            disp=0, maxiter=100, method="newton", warn_convergence=False)
        return mr.generate_spec_curve(mr.load_canonical_dataset())


def summary_line(text, cfg, label):
    """The "(n/N)" of the line starting with label in cfg's block of spec_curve_summary.txt, or None."""
    block = text.split(f"  {cfg}:\n", 1)[1].split("\n\n", 1)[0] if f"  {cfg}:\n" in text else ""
    line = [x for x in block.splitlines() if x.strip().startswith(label)]
    return line[0].rsplit("(", 1)[-1].rstrip(")") if line else None


def pp1(x):
    """One decimal with an explicit sign, and 0.0 unsigned (as the table prints it)."""
    return "0.0" if round(x, 1) == 0 else f"{x:+.1f}"


def run_pipeline():
    env = dict(os.environ, SUS_CANONICAL=str(DATA.relative_to(PROJECT_ROOT)),
               SUS_OUT_DIR=str(OUT.relative_to(PROJECT_ROOT)))
    spec = dict(env, SUS_SPEC_CURVE_INPUT=str(SPEC_INPUT.relative_to(PROJECT_ROOT)),
                SUS_SPEC_CURVE_MODELS=",".join(SPEC_EXPLORATORY["models"]),
                SUS_SPEC_CURVE_SYCOPHANCY_ITEMS=os.path.join("data", "benchmarks", "sycophancy_eval_exp4.jsonl"),
                SUS_OUT_DIR=str(SPEC_OUT.relative_to(PROJECT_ROOT)), PYTHONWARNINGS="ignore::UserWarning")
    for script, e in (("build_analysis_dataset.py", None), ("oe_scoring_validation_analysis_dataset.py", None),
                      ("variance_decomposition.py", env), ("master_reanalysis.py", env),
                      ("tost_per_benchmark.py", env), ("option_preserving_matched.py", env),
                      ("build_spec_curve_input.py", spec), ("spec_curve_analysis.py", spec)):
        note = " (384 specifications and 200 permutations; the slowest step)" if script == "spec_curve_analysis.py" else ""
        print(f"--> analysis/{script}{note}", flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "analysis" / script)], cwd=PROJECT_ROOT,
                       env=e, check=True, stdout=subprocess.DEVNULL)


def main():
    if "--run" in sys.argv[1:]:
        run_pipeline()
    checks = []

    def check(what, got, want):
        checks.append((what, got, want))
        if got != want:
            sys.exit(f"MISMATCH {what}: got {got!r}, paper prints {want!r}")

    with open(DATA, "rb") as f:
        blob = f.read()
    n_rows = blob.count(b"\n")
    check("analysis dataset rows", f"{n_rows:,}", "60,112")
    check("analysis dataset sha256", hashlib.sha256(blob).hexdigest(), DATA_SHA256)
    with open(LEDGER) as f:
        ledger = Counter((x["reason"], x["fate"]) for x in map(json.loads, f))
    for (reason, fate), n in LEDGER_COUNTS.items():
        check(f"{reason} ({fate})", f"{ledger[(reason, fate)]:,}", n)
    check("ledger classes", str(len(ledger)), str(len(LEDGER_COUNTS)))

    vd = json.load(open(OUT / "variance_decomposition_results.json"))
    factors, omega = vd["observation_level_anova"]["factors"], vd["omega_squared"]
    check("N", f"{vd['metadata']['n_observations']:,}", "60,112")
    for src, (df_, f_, eta, om, mag) in VD_TABLE.items():
        row = factors[src]
        check(f"{src}: df", str(row["df"]), df_)
        check(f"{src}: F", f"{row['F']:,.0f}", f_)
        check(f"{src}: eta2", f"{100 * row['eta_squared']:.1f}", eta)
        check(f"{src}: omega2", f"{100 * omega[src]['omega_squared']:.1f}", om)
        check(f"{src}: magnitude", LABELS[omega[src]["interpretation"]], mag)
    check("Residual: df", f"{factors['Residual']['df']:,}", VD_RESIDUAL[0])
    check("Residual: eta2", f"{100 * factors['Residual']['eta_squared']:.1f}", VD_RESIDUAL[1])

    # Ranges and overall rates from the unrounded cell rates (the JSON stores them rounded)
    rows = [json.loads(line) for line in blob.decode().splitlines()]
    d = pd.DataFrame({"model": [r["model_id"] for r in rows], "benchmark": [r["benchmark_id"] for r in rows],
                      "config": [r["config_id"] for r in rows], "safe": [int(r["is_safe"]) for r in rows]})
    cells = d.groupby(["model", "benchmark", "config"])["safe"].mean().unstack()
    ranges = (cells.max(axis=1) - cells.min(axis=1)) * 100
    profiles = vd["per_model_profiles"]
    for m, (eta, avg, mx, safe) in MODEL_TABLE.items():
        check(f"{m}: eta2 scaffold", f"{100 * profiles[m]['scaffold']['eta_squared']:.1f}", eta)
        check(f"{m}: avg range", f"{ranges.xs(m, level='model').mean():.1f}", avg)
        check(f"{m}: max range", f"{ranges.xs(m, level='model').max():.1f}", mx)
        check(f"{m}: overall safe", f"{100 * d.loc[d['model'] == m, 'safe'].mean():.1f}", safe)

    conf = json.load(open(OUT / "confirmatory_results_judge.json"))
    h1 = {v["config"]: v for v in conf["H1"].values() if isinstance(v, dict) and "config" in v}
    mr, ra = h1["map_reduce"], h1["react"]
    check("map-reduce RD (pp)", f"{100 * mr['RD']:.1f}", "-7.3")
    check("map-reduce RD 95% CI (pp)", f"[{100 * mr['RD_CI_lo']:.1f}, {100 * mr['RD_CI_hi']:.1f}]", "[-8.1, -6.4]")
    check("map-reduce NNH", f"{mr['NNH']:.0f}", "14")
    check("ReAct RD (pp)", f"{100 * ra['RD']:.1f}", "-0.8")
    full = json.load(open(OUT / "safety_rates_full.json"))
    rates = full["safety_rates"]
    # Cells without observations are not reported and do not enter the BH family:
    # 96 cells minus the four Gemini x sycophancy cells; 72 comparisons minus three.
    check("reported model x configuration x benchmark cells", str(len(rates)), "92")
    check("reported cells with no observations", str(sum(v["n_total"] == 0 for v in rates.values())), "0")
    pairs = {(p["model"], p["benchmark"], p["comparison"]): p for p in full["pairwise_comparisons"]}
    check("BH family size (scaffold vs direct comparisons)", str(len(pairs)), "69")
    check("comparisons with BH q < 0.05 (starred in the figure)",
          str(sum(p["significant_bh05"] for p in pairs.values())), "25")
    for key, rd in PAIRWISE_RD.items():
        check(f"{' '.join(key)}: RD (pp)", f"{100 * pairs[key]['difference']:+.1f}", rd)
    for key in BH_BELOW_1E5:
        check(f"{' '.join(key)}: p_BH < 1e-5", str(pairs[key]["p_value_bh"] < 1e-5), "True")
    syc = [v for v in rates.values() if v["benchmark"] == "sycophancy"]
    direct = [v for v in syc if v["config"] == "direct"]
    check("sycophancy safe rate, direct (%)",
          f"{100 * sum(v['n_safe'] for v in direct) / sum(v['n_total'] for v in direct):.1f}", "33.8")
    check("sycophancy safe rate, all configurations (%)",
          f"{100 * sum(v['n_safe'] for v in syc) / sum(v['n_total'] for v in syc):.1f}", "34.5")
    lo, hi = min(v["safety_rate"] for v in direct), max(v["safety_rate"] for v in direct)
    check("sycophancy direct range across models (%)", f"{100 * lo:.0f}--{100 * hi:.0f}", "11--49")

    t = json.load(open(OUT / "oe_scoring_validation_200item.json"))
    drawn, kept, s = t["drawn"], t["retained"], t["analysis_dataset"]
    check("Test 1f: responses drawn", str(drawn["n"]), T1F["drawn"])
    for b, n in drawn["by_benchmark"].items():
        check(f"Test 1f: {b} responses drawn", str(n), T1F["drawn_per_benchmark"])
    check("Test 1f: models drawn", str(len(drawn["by_model"])), str(len(T1F["drawn_by_model"])))
    for m, n in T1F["drawn_by_model"].items():
        check(f"Test 1f: {m} responses drawn", str(drawn["by_model"].get(m, 0)), n)
    check("Test 1f: configurations drawn", str(len(drawn["by_config"])), str(len(T1F["drawn_by_config"])))
    for c, n in T1F["drawn_by_config"].items():
        check(f"Test 1f: {c} responses drawn", str(drawn["by_config"].get(c, 0)), n)
    check("Test 1f: drawn previews holding the complete response", str(drawn["preview_is_complete_response"]),
          T1F["drawn_complete_previews"])
    check("Test 1f: responses retained", str(kept["n"]), T1F["retained"])
    for reason, n in T1F["retained_score_reason"].items():
        check(f"Test 1f: retained pipeline labels ({reason})", str(kept["pipeline_score_reason"].get(reason, 0)), n)
    check("Test 1f: retained GPT-5.2 responses", str(kept["by_model"].get("gpt52", 0)), T1F["retained_gpt52"])
    check("Test 1f: agreement", f"{s['agree']}/{s['n']}", T1F["agreement"])
    check("Test 1f: agreement (%)", f"{s['agreement_pct']:.0f}", T1F["agreement_pct"])
    check("Test 1f: Cohen's kappa", f"{s['kappa']:.3f}", T1F["kappa"])
    check("Test 1f: pipeline-safe minus judge-safe (pp)", f"{s['pipeline_safe_minus_judge_safe_pp']:+.1f}", T1F["gap_pp"])
    for cell, n in T1F["cells"].items():
        check(f"Test 1f: {cell}", str(s["cells"][cell]), n)
    for b, v in T1F["by_benchmark"].items():
        sb = t["by_benchmark"][b]["analysis_dataset"]
        cb = sb["cells"]
        check(f"Test 1f, {b}: agreement", f"{sb['agree']}/{sb['n']}", v["agreement"])
        check(f"Test 1f, {b}: Cohen's kappa", f"{sb['kappa']:.3f}", v["kappa"])
        check(f"Test 1f, {b}: cells", "({},{},{},{})".format(cb["both_safe"], cb["pipeline_safe_judge_unsafe"],
                                                           cb["pipeline_unsafe_judge_safe"], cb["both_unsafe"]), v["cells"])
        check(f"Test 1f, {b}: pipeline-safe minus judge-safe (pp)", f"{sb['pipeline_safe_minus_judge_safe_pp']:+.0f}",
              v["gap_pp"])

    tb = json.load(open(OUT / "tost_per_benchmark.json"))
    check("per-benchmark TOST: comparisons", sorted(tb["comparisons"]), sorted(TOST_BENCH))
    check("per-benchmark TOST: margin (pp)", tb["margin_pp"], 2.0)
    for key, v in TOST_BENCH.items():
        c = tb["comparisons"][key]
        lo, hi = c["ci90_pp"]
        check(f"per-benchmark TOST, {key}: RD (pp)", f"{c['rd_pp']:+.2f}", v["rd"])
        check(f"per-benchmark TOST, {key}: 90% CI (pp)", f"[{lo:+.2f}, {hi:+.2f}]", f"[{v['lo']}, {v['hi']}]")
        check(f"per-benchmark TOST, {key}: equivalent", c["equivalent"], v["equivalent"])
    check("per-benchmark TOST: comparisons equivalent", f"{tb['n_equivalent']} out of {tb['n_comparisons']}",
          f"{TOST_BENCH_TEXT['n_equivalent']} out of {TOST_BENCH_TEXT['n_comparisons']}")

    op = json.load(open(OUT / "option_preserving_matched.json"))["models"]
    check("option preserving: models", sorted(op), sorted(OP_MATCHED))
    for m, v in OP_MATCHED.items():
        o = op[m]
        check(f"option preserving, {m}: direct safe", str(o["direct_safe"]), v["direct"])
        check(f"option preserving, {m}: map-reduce safe", str(o["map_reduce_safe"]), v["map_reduce"])
        check(f"option preserving, {m}: option-preserving safe", str(o["option_preserving_safe"]),
              v["option_preserving"])
        check(f"option preserving, {m}: OP - MR (pp)", pp1(o["op_minus_mr_pp"]), v["diff"])
        check(f"option preserving, {m}: OP - MR 95% CI (pp)", "{:.1f}, {:.1f}".format(*o["op_minus_mr_ci95_pp"]),
              v["diff_ci"])
        check(f"option preserving, {m}: recovery (%)", f"{o['recovery_pct']:.1f}", v["recovery"])
        if v["recovery_ci"] is not None:
            check(f"option preserving, {m}: recovery 95% CI (%)", "{:.1f}, {:.1f}".format(*o["recovery_ci95_pct"]),
                  v["recovery_ci"])
    t8 = OP_MATCHED_TEXT
    rec = [op[m]["recovery_pct"] for m in t8["range_models"]]
    check("option preserving: recovery range (%)", f"{min(rec):.0f}--{max(rec):.0f}", "--".join(t8["recovery_range"]))
    gap = [m for m in op if m not in t8["range_models"]]
    check("option preserving: models outside the range", gap, ["llama4"])
    check("option preserving, llama4: map-reduce gap (pp)", f"{op['llama4']['direct_minus_mr_pp']:.1f}",
          t8["llama_gap"])
    ps = [op[m]["mcnemar_exact_p"] for m in t8["range_models"]]
    check("option preserving: smallest McNemar p", f"{min(ps):.1e}", t8["mcnemar_min"])
    check("option preserving: largest McNemar p", f"{max(ps):.3f}", t8["mcnemar_max"])

    sp = json.load(open(OUT / "spec_curve_results.json"))
    cs, P = sp["config_summaries"], SPEC_PRIMARY
    check("primary specification curve: specifications", str(sp["n_specifications"]), P["n_specs"])
    check("primary specification curve: map-reduce median OR", f"{cs['map_reduce']['median_OR']:.2f}", P["mr_median"])
    check("primary specification curve: map-reduce OR IQR", [f"{x:.2f}" for x in cs["map_reduce"]["IQR_OR"]], P["mr_iqr"])
    check("primary specification curve: map-reduce OR range", [f"{x:.2f}" for x in cs["map_reduce"]["range_OR"]],
          P["mr_range"])
    for cfg, k in (("map_reduce", "mr"), ("multi_agent", "ma"), ("react", "react")):
        check(f"primary specification curve: {cfg} significant", f"{cs[cfg]['n_significant']}/{cs[cfg]['n_specs']}",
              f"{P[k + '_sig']}/{P[k + '_n']}")
    check("primary specification curve: map-reduce significant (%)",
          f"{100 * cs['map_reduce']['n_significant'] / cs['map_reduce']['n_specs']:.0f}", P["mr_sig_pct"])
    check("primary specification curve: covariance of every fit",
          sorted({e["covariance_type"] for s in sp["specifications"] for e in s["config_effects"].values()}), ["cluster"])
    check("primary specification curve: analytic choices",
          sorted({str(len(s["specification"])) for s in sp["specifications"]}), [P["n_choices"]])
    check("primary specification curve: options per choice (benchmarks, models, parse failures)",
          [str(len({s["specification"][k] for s in sp["specifications"]})) for k in ("benchmark_subset", "model_subset",
                                                                                     "scoring")], P["options_per_choice"])
    check("primary specification curve: contrasts", str(sum(len(s["config_effects"]) for s in sp["specifications"])),
          P["n_contrasts"])
    nr = nonrobust_primary_curve()
    base = {json.dumps(s["specification"], sort_keys=True): s["config_effects"] for s in nr["specifications"]}
    pairs = [(cfg, e, base[json.dumps(s["specification"], sort_keys=True)][cfg])
             for s in sp["specifications"] for cfg, e in s["config_effects"].items()]
    check("primary specification curve: contrasts in the non-robust first plot",
          str(sum(len(v) for v in base.values())), P["n_contrasts"])
    check("primary specification curve: point estimates unchanged by the covariance correction",
          max(abs(e["OR"] - o["OR"]) for _, e, o in pairs) < 1e-9, True)
    up = [(cfg, e["OR"]) for cfg, e, o in pairs if e["significant_005"] and not o["significant_005"]]
    check("primary specification curve: decisions changed to significant", str(len(up)), P["to_significant"])
    check("primary specification curve: of which decreases for ReAct or multi-agent",
          str(sum(c in ("react", "multi_agent") and r < 1 for c, r in up)), P["to_significant_decreases"])
    check("primary specification curve: of which increases for ReAct",
          str(sum(c == "react" and r > 1 for c, r in up)), P["to_significant_react_increases"])
    check("primary specification curve: decisions changed to not significant",
          str(sum(o["significant_005"] and not e["significant_005"] for _, e, o in pairs)), P["to_not_significant"])

    ex = json.load(open(SPEC_OUT / "spec_curve_results.json"))
    E = SPEC_EXPLORATORY
    mr = [s["config_effects"]["map_reduce"] for s in ex["specifications"] if "map_reduce" in s["config_effects"]]
    below = sum(x["OR"] < 1 for x in mr)
    degr = sum(x["OR"] < 1 and x["p_value"] < 0.05 for x in mr)
    check("exploratory specification curve: specifications", str(ex["n_specifications"]), E["n_specs"])
    check("exploratory specification curve: analytic choices",
          sorted({str(len(s["specification"])) for s in ex["specifications"]}), [E["n_choices"]])
    check("exploratory specification curve: models", f"{len(ex['models'])}: {ex['models']}",
          f"{E['n_models']}: {E['models']}")
    check("exploratory specification curve: with map-reduce", str(len(mr)), E["mr_n"])
    check("exploratory specification curve: without map-reduce", str(ex["n_specifications"] - len(mr)), E["no_mr"])
    check("exploratory specification curve: map-reduce OR below 1", str(below), E["or_below_1"])
    check("exploratory specification curve: significant degradation", f"{degr} ({100 * degr / len(mr):.1f}%)",
          f"{E['sig_degradation']} ({E['sig_degradation_pct']}%)")
    ma = ex["config_summaries"]["multi_agent"]
    check("exploratory specification curve: multi-agent significant", f"{ma['n_significant']}/{ma['n_specs']} "
          f"({100 * ma['n_significant'] / ma['n_specs']:.1f}%)", f"{E['ma_sig']}/{E['ma_n']} ({E['ma_sig_pct']}%)")
    check("exploratory specification curve: covariance of every fit",
          sorted({s["covariance_type"] for s in ex["specifications"]}), ["cluster"])
    summary = (SPEC_OUT / "spec_curve_summary.txt").read_text()
    check("exploratory specification curve summary: map-reduce significant with OR < 1",
          summary_line(summary, "map_reduce", "%% sig (0.05), OR < 1:"), f"{E['sig_degradation']}/{E['mr_n']}")
    check("exploratory specification curve summary: multi-agent significant (either direction)",
          summary_line(summary, "multi_agent", "%% sig (0.05), either direction:"), f"{E['ma_sig']}/{E['ma_n']}")
    check("exploratory specification curve: map-reduce median OR", f"{ex['config_summaries']['map_reduce']['median_OR']:.2f}",
          E["mr_median"])
    check("exploratory specification curve: map-reduce OR IQR",
          [f"{x:.2f}" for x in ex["config_summaries"]["map_reduce"]["IQR_OR"]], E["mr_iqr"])
    print(f"All {len(checks)} checks match the printed values.")


if __name__ == "__main__":
    main()
