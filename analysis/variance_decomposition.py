#!/usr/bin/env python3
"""
Variance Decomposition Analysis for Safety Under Scaffolding
=============================================================

Decomposes total variance in safety rates into:
  - model main effect
  - scaffold (config) main effect
  - benchmark main effect
  - model x scaffold interaction
  - scaffold x benchmark interaction
  - model x benchmark interaction
  - residual

Two approaches:
  1. Observation-level Type II ANOVA (logistic-scale via OLS on binary outcome)
  2. Cell-means approach: aggregate to safety rates per cell, then decompose

Also computes per-model variance profiles to show which factors
drive each model's safety variation.

Usage:
    python analysis/variance_decomposition.py

Output:
    analysis/outputs/variance_decomposition_results.json
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = PROJECT_ROOT / "results" / "canonical_primary_dataset.jsonl"
OUTPUT = PROJECT_ROOT / "analysis" / "outputs" / "variance_decomposition_results.json"


def load_data():
    """Load canonical dataset into a pandas DataFrame."""
    rows = []
    with open(CANONICAL) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "model": r["model_id"],
                "scaffold": r["config_id"],
                "benchmark": r["benchmark_id"],
                "case_id": r["case_id"],
                "is_safe": int(r["is_safe"]),
            })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} observations")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Scaffolds: {sorted(df['scaffold'].unique())}")
    print(f"  Benchmarks: {sorted(df['benchmark'].unique())}")
    print(f"  Overall safe rate: {df['is_safe'].mean():.4f}")
    return df


def compute_eta_squared(anova_table):
    """Compute eta-squared (proportion of SS explained) from ANOVA table."""
    ss_total = anova_table["sum_sq"].sum()
    eta2 = {}
    for idx in anova_table.index:
        eta2[idx] = anova_table.loc[idx, "sum_sq"] / ss_total
    return eta2


def compute_partial_eta_squared(anova_table):
    """Compute partial eta-squared for each factor."""
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    partial_eta2 = {}
    for idx in anova_table.index:
        if idx == "Residual":
            continue
        ss_effect = anova_table.loc[idx, "sum_sq"]
        partial_eta2[idx] = ss_effect / (ss_effect + ss_residual)
    return partial_eta2


def observation_level_anova(df):
    """
    Approach 1: Type II ANOVA on binary is_safe outcome (OLS approximation).

    Using OLS on binary outcomes gives identical F-tests and SS decomposition
    to what you'd get from a linear probability model. For variance decomposition
    purposes (eta-squared), this is standard and interpretable.
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: Observation-Level Type II ANOVA (OLS on binary outcome)")
    print("=" * 70)

    # Fit the model with main effects and key interactions
    formula = (
        "is_safe ~ C(model) + C(scaffold) + C(benchmark) "
        "+ C(model):C(scaffold) + C(scaffold):C(benchmark) "
        "+ C(model):C(benchmark)"
    )
    model = ols(formula, data=df).fit()

    # Type II ANOVA
    anova_table = anova_lm(model, typ=2)

    # Compute eta-squared
    eta2 = compute_eta_squared(anova_table)
    partial_eta2 = compute_partial_eta_squared(anova_table)

    print(f"\nR-squared: {model.rsquared:.4f}")
    print(f"Adj R-squared: {model.rsquared_adj:.4f}")
    print(f"\nANOVA Table (Type II):")
    print(f"{'Factor':<35} {'SS':>12} {'df':>6} {'F':>12} {'p-value':>14} {'eta2':>8} {'p_eta2':>8}")
    print("-" * 97)

    results_table = {}
    for idx in anova_table.index:
        row = anova_table.loc[idx]
        name = idx.replace("C(", "").replace(")", "").replace(":", " x ")
        ss = row["sum_sq"]
        df_val = row["df"]
        f_val = row.get("F", np.nan)
        p_val = row.get("PR(>F)", np.nan)
        e2 = eta2.get(idx, 0)
        pe2 = partial_eta2.get(idx, 0)

        if idx == "Residual":
            print(f"{name:<35} {ss:>12.2f} {df_val:>6.0f} {'':>12} {'':>14} {e2:>8.4f} {'':>8}")
        else:
            print(f"{name:<35} {ss:>12.2f} {df_val:>6.0f} {f_val:>12.2f} {p_val:>14.2e} {e2:>8.4f} {pe2:>8.4f}")

        results_table[name] = {
            "sum_sq": float(ss),
            "df": int(df_val),
            "F": float(f_val) if not np.isnan(f_val) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "eta_squared": float(e2),
            "partial_eta_squared": float(pe2) if idx != "Residual" else None,
        }

    return results_table, model.rsquared, model.rsquared_adj


def cell_means_decomposition(df):
    """
    Approach 2: Cell-means variance decomposition.

    Aggregate to safety rates per (model, scaffold, benchmark) cell,
    then decompose variance of cell means using balanced ANOVA.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: Cell-Means Variance Decomposition")
    print("=" * 70)

    # Compute cell-level safety rates
    cells = df.groupby(["model", "scaffold", "benchmark"]).agg(
        safe_rate=("is_safe", "mean"),
        n=("is_safe", "count"),
    ).reset_index()

    print(f"\nNumber of cells: {len(cells)}")
    print(f"Cell sizes: min={cells['n'].min()}, max={cells['n'].max()}, "
          f"mean={cells['n'].mean():.0f}")

    # ANOVA on cell means (weighted by cell size for proper decomposition)
    formula = (
        "safe_rate ~ C(model) + C(scaffold) + C(benchmark) "
        "+ C(model):C(scaffold) + C(scaffold):C(benchmark) "
        "+ C(model):C(benchmark)"
    )
    model = ols(formula, data=cells).fit()
    anova_table = anova_lm(model, typ=2)

    eta2 = compute_eta_squared(anova_table)
    partial_eta2 = compute_partial_eta_squared(anova_table)

    print(f"\nR-squared: {model.rsquared:.4f}")
    print(f"\nCell-Means ANOVA (Type II):")
    print(f"{'Factor':<35} {'SS':>12} {'df':>6} {'F':>12} {'p-value':>14} {'eta2':>8} {'p_eta2':>8}")
    print("-" * 97)

    results_table = {}
    for idx in anova_table.index:
        row = anova_table.loc[idx]
        name = idx.replace("C(", "").replace(")", "").replace(":", " x ")
        ss = row["sum_sq"]
        df_val = row["df"]
        f_val = row.get("F", np.nan)
        p_val = row.get("PR(>F)", np.nan)
        e2 = eta2.get(idx, 0)
        pe2 = partial_eta2.get(idx, 0)

        if idx == "Residual":
            print(f"{name:<35} {ss:>12.6f} {df_val:>6.0f} {'':>12} {'':>14} {e2:>8.4f} {'':>8}")
        else:
            p_str = f"{p_val:.2e}" if not np.isnan(p_val) else ""
            print(f"{name:<35} {ss:>12.6f} {df_val:>6.0f} {f_val:>12.2f} {p_str:>14} {e2:>8.4f} {pe2:>8.4f}")

        results_table[name] = {
            "sum_sq": float(ss),
            "df": int(df_val),
            "F": float(f_val) if not np.isnan(f_val) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "eta_squared": float(e2),
            "partial_eta_squared": float(pe2) if idx != "Residual" else None,
        }

    return results_table, model.rsquared


def per_model_variance_profiles(df):
    """
    For each model, decompose its safety variation into:
      - scaffold effect
      - benchmark effect
      - scaffold x benchmark interaction
      - residual

    This shows whether a model is more sensitive to scaffold choice
    vs. benchmark choice.
    """
    print("\n" + "=" * 70)
    print("PER-MODEL VARIANCE PROFILES")
    print("=" * 70)

    profiles = {}

    for model_name in sorted(df["model"].unique()):
        mdf = df[df["model"] == model_name].copy()

        formula = "is_safe ~ C(scaffold) + C(benchmark) + C(scaffold):C(benchmark)"
        try:
            fit = ols(formula, data=mdf).fit()
            anova_table = anova_lm(fit, typ=2)
            eta2 = compute_eta_squared(anova_table)

            profile = {}
            for idx in anova_table.index:
                name = idx.replace("C(", "").replace(")", "").replace(":", " x ")
                profile[name] = {
                    "eta_squared": round(float(eta2.get(idx, 0)), 6),
                    "sum_sq": round(float(anova_table.loc[idx, "sum_sq"]), 4),
                }
                f_val = anova_table.loc[idx].get("F", np.nan)
                p_val = anova_table.loc[idx].get("PR(>F)", np.nan)
                if not np.isnan(f_val):
                    profile[name]["F"] = round(float(f_val), 2)
                    profile[name]["p_value"] = float(p_val)

            safe_rate = mdf["is_safe"].mean()
            profile["overall_safe_rate"] = round(float(safe_rate), 4)
            profiles[model_name] = profile

        except Exception as e:
            print(f"  {model_name}: Error - {e}")
            profiles[model_name] = {"error": str(e)}

    # Print summary table
    print(f"\n{'Model':<12} {'Safe%':>7} {'η²_scaffold':>12} {'η²_benchmark':>13} {'η²_scaf×bench':>14} {'η²_residual':>12} {'Dominant Factor':<20}")
    print("-" * 92)

    for model_name in sorted(profiles.keys()):
        p = profiles[model_name]
        if "error" in p:
            print(f"  {model_name}: {p['error']}")
            continue

        e2_scaffold = p.get("scaffold", {}).get("eta_squared", 0)
        e2_benchmark = p.get("benchmark", {}).get("eta_squared", 0)
        e2_interaction = p.get("scaffold x benchmark", {}).get("eta_squared", 0)
        e2_residual = p.get("Residual", {}).get("eta_squared", 0)
        safe_rate = p.get("overall_safe_rate", 0)

        # Determine dominant factor
        factors = {
            "scaffold": e2_scaffold,
            "benchmark": e2_benchmark,
            "scaffold x benchmark": e2_interaction,
        }
        dominant = max(factors, key=factors.get)

        print(f"{model_name:<12} {safe_rate*100:>6.1f}% {e2_scaffold:>12.4f} {e2_benchmark:>13.4f} "
              f"{e2_interaction:>14.4f} {e2_residual:>12.4f} {dominant:<20}")

    return profiles


def compute_scaffold_sensitivity_index(df):
    """
    Compute a 'scaffold sensitivity index' for each model x benchmark combination.

    This is the coefficient of variation (SD/mean) of safety rates across scaffolds,
    giving a normalized measure of how much scaffold choice matters for each
    model-benchmark pair.
    """
    print("\n" + "=" * 70)
    print("SCAFFOLD SENSITIVITY INDEX (per model x benchmark)")
    print("=" * 70)

    cells = df.groupby(["model", "benchmark", "scaffold"])["is_safe"].mean().reset_index()
    cells.columns = ["model", "benchmark", "scaffold", "safe_rate"]

    sensitivity = cells.groupby(["model", "benchmark"]).agg(
        mean_safe=("safe_rate", "mean"),
        std_safe=("safe_rate", "std"),
        min_safe=("safe_rate", "min"),
        max_safe=("safe_rate", "max"),
    ).reset_index()

    sensitivity["range_pp"] = (sensitivity["max_safe"] - sensitivity["min_safe"]) * 100
    sensitivity["cv"] = np.where(
        sensitivity["mean_safe"] > 0,
        sensitivity["std_safe"] / sensitivity["mean_safe"],
        0
    )

    # Pivot for display
    print(f"\nRange of safety rates across scaffolds (percentage points):")
    pivot = sensitivity.pivot(index="model", columns="benchmark", values="range_pp")
    print(pivot.round(1).to_string())

    print(f"\nCoefficient of variation across scaffolds:")
    pivot_cv = sensitivity.pivot(index="model", columns="benchmark", values="cv")
    print(pivot_cv.round(4).to_string())

    # Overall model sensitivity (average range across benchmarks)
    model_sensitivity = sensitivity.groupby("model").agg(
        avg_range_pp=("range_pp", "mean"),
        max_range_pp=("range_pp", "max"),
        avg_cv=("cv", "mean"),
    ).reset_index()

    print(f"\nOverall scaffold sensitivity by model:")
    print(f"{'Model':<12} {'Avg Range (pp)':>15} {'Max Range (pp)':>15} {'Avg CV':>10}")
    print("-" * 54)
    for _, row in model_sensitivity.sort_values("avg_range_pp", ascending=False).iterrows():
        print(f"{row['model']:<12} {row['avg_range_pp']:>15.1f} {row['max_range_pp']:>15.1f} {row['avg_cv']:>10.4f}")

    return {
        "range_by_model_benchmark": {
            f"{r['model']}_{r['benchmark']}": round(float(r['range_pp']), 2)
            for _, r in sensitivity.iterrows()
        },
        "cv_by_model_benchmark": {
            f"{r['model']}_{r['benchmark']}": round(float(r['cv']), 4)
            for _, r in sensitivity.iterrows()
        },
        "model_summary": {
            row["model"]: {
                "avg_range_pp": round(float(row["avg_range_pp"]), 2),
                "max_range_pp": round(float(row["max_range_pp"]), 2),
                "avg_cv": round(float(row["avg_cv"]), 4),
            }
            for _, row in model_sensitivity.iterrows()
        }
    }


def compute_marginal_safety_rates(df):
    """Compute marginal safety rates for each factor level."""
    print("\n" + "=" * 70)
    print("MARGINAL SAFETY RATES")
    print("=" * 70)

    results = {}

    for factor in ["model", "scaffold", "benchmark"]:
        rates = df.groupby(factor)["is_safe"].agg(["mean", "count"]).reset_index()
        rates.columns = [factor, "safe_rate", "n"]
        print(f"\nBy {factor}:")
        for _, row in rates.sort_values("safe_rate", ascending=False).iterrows():
            print(f"  {row[factor]:<15} {row['safe_rate']*100:>6.1f}%  (n={int(row['n']):,})")
        results[factor] = {
            row[factor]: round(float(row["safe_rate"]), 4)
            for _, row in rates.iterrows()
        }

    # Two-way: scaffold x benchmark
    rates_2way = df.groupby(["scaffold", "benchmark"])["is_safe"].mean().reset_index()
    rates_2way.columns = ["scaffold", "benchmark", "safe_rate"]
    pivot = rates_2way.pivot(index="scaffold", columns="benchmark", values="safe_rate")
    print(f"\nScaffold x Benchmark safe rates:")
    print((pivot * 100).round(1).to_string())

    results["scaffold_x_benchmark"] = {
        f"{row['scaffold']}_{row['benchmark']}": round(float(row['safe_rate']), 4)
        for _, row in rates_2way.iterrows()
    }

    return results


def compute_omega_squared(df):
    """
    Compute omega-squared (less biased than eta-squared) for each factor.

    omega² = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)
    """
    print("\n" + "=" * 70)
    print("OMEGA-SQUARED (LESS BIASED EFFECT SIZE)")
    print("=" * 70)

    formula = (
        "is_safe ~ C(model) + C(scaffold) + C(benchmark) "
        "+ C(model):C(scaffold) + C(scaffold):C(benchmark) "
        "+ C(model):C(benchmark)"
    )
    model = ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)

    ms_error = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
    ss_total = anova_table["sum_sq"].sum()

    results = {}
    print(f"\n{'Factor':<35} {'omega²':>10} {'eta²':>10} {'Interpretation':<25}")
    print("-" * 82)

    eta2_all = compute_eta_squared(anova_table)

    for idx in anova_table.index:
        if idx == "Residual":
            continue

        name = idx.replace("C(", "").replace(")", "").replace(":", " x ")
        ss_effect = anova_table.loc[idx, "sum_sq"]
        df_effect = anova_table.loc[idx, "df"]

        omega2 = (ss_effect - df_effect * ms_error) / (ss_total + ms_error)
        omega2 = max(0, omega2)  # Floor at 0
        eta2 = eta2_all[idx]

        # Cohen's benchmarks for eta²: small=0.01, medium=0.06, large=0.14
        if omega2 >= 0.14:
            interp = "LARGE"
        elif omega2 >= 0.06:
            interp = "MEDIUM"
        elif omega2 >= 0.01:
            interp = "SMALL"
        else:
            interp = "negligible"

        print(f"{name:<35} {omega2:>10.4f} {eta2:>10.4f} {interp:<25}")
        results[name] = {
            "omega_squared": round(float(omega2), 6),
            "eta_squared": round(float(eta2), 6),
            "interpretation": interp,
        }

    return results


def assess_framework_utility(obs_anova, cell_anova, profiles, sensitivity, omega2):
    """
    Synthesize findings and assess whether this framework provides useful
    information beyond existing analyses (OR, RD, NNH).
    """
    print("\n" + "=" * 70)
    print("FRAMEWORK UTILITY ASSESSMENT")
    print("=" * 70)

    assessment = {
        "key_findings": [],
        "comparison_to_existing": [],
        "reusability_assessment": [],
    }

    # Extract key eta² values from observation-level ANOVA
    obs_factors = {k: v["eta_squared"] for k, v in obs_anova.items() if k != "Residual"}
    sorted_factors = sorted(obs_factors.items(), key=lambda x: x[1], reverse=True)

    print("\nVariance explained ranking (observation-level eta²):")
    for name, e2 in sorted_factors:
        pct = e2 * 100
        bar = "#" * int(pct * 2)
        print(f"  {name:<35} {pct:>6.2f}%  {bar}")

    residual_e2 = obs_anova.get("Residual", {}).get("eta_squared", 0)
    print(f"  {'Residual':<35} {residual_e2*100:>6.2f}%")

    # Key findings
    top_factor = sorted_factors[0][0] if sorted_factors else "unknown"
    top_e2 = sorted_factors[0][1] if sorted_factors else 0
    assessment["key_findings"].append(
        f"Largest variance component: {top_factor} (eta²={top_e2:.4f}, {top_e2*100:.1f}%)"
    )

    total_explained = sum(v for _, v in sorted_factors)
    assessment["key_findings"].append(
        f"Total variance explained by all factors: {total_explained*100:.1f}%"
    )
    assessment["key_findings"].append(
        f"Residual (within-cell, item-level): {residual_e2*100:.1f}%"
    )

    # Check if interactions are meaningful
    interaction_e2 = sum(v for k, v in sorted_factors if " x " in k)
    main_effect_e2 = sum(v for k, v in sorted_factors if " x " not in k)
    assessment["key_findings"].append(
        f"Main effects explain {main_effect_e2*100:.1f}% vs interactions {interaction_e2*100:.1f}%"
    )

    # Per-model differentiation
    scaffold_sensitivities = []
    for model_name, p in profiles.items():
        if "error" in p:
            continue
        e2_s = p.get("scaffold", {}).get("eta_squared", 0)
        scaffold_sensitivities.append((model_name, e2_s))

    if scaffold_sensitivities:
        scaffold_sensitivities.sort(key=lambda x: x[1], reverse=True)
        most_sensitive = scaffold_sensitivities[0]
        least_sensitive = scaffold_sensitivities[-1]
        assessment["key_findings"].append(
            f"Most scaffold-sensitive model: {most_sensitive[0]} (eta²={most_sensitive[1]:.4f})"
        )
        assessment["key_findings"].append(
            f"Least scaffold-sensitive model: {least_sensitive[0]} (eta²={least_sensitive[1]:.4f})"
        )

    # Comparison to existing analyses
    assessment["comparison_to_existing"] = [
        "OR/RD/NNH tell you DIRECTION and MAGNITUDE of scaffold effects on safety",
        "Variance decomposition tells you RELATIVE IMPORTANCE — which factors matter most",
        "Key insight: benchmark choice may explain more variance than scaffold choice (or vice versa)",
        "Interactions reveal NON-ADDITIVITY — some scaffolds hurt only on specific benchmarks",
        "Per-model profiles are NOVEL: they show which models are robust vs fragile to scaffolding",
        "This complements (not replaces) the paper's inferential statistics",
    ]

    # Reusability
    assessment["reusability_assessment"] = [
        "Framework is fully reusable: any new model x scaffold x benchmark data can be plugged in",
        "eta²/omega² provide standardized, comparable effect sizes across studies",
        "Per-model profiles enable model-selection decisions (pick robust models)",
        "Scaffold sensitivity index enables scaffold-selection decisions",
        "Could be extended to include additional factors (prompt format, temperature, etc.)",
        "Natural fit for a 'safety scorecard' that decomposes risk into factor contributions",
    ]

    for section, items in assessment.items():
        print(f"\n{section.replace('_', ' ').upper()}:")
        for item in items:
            print(f"  - {item}")

    return assessment


def main():
    # Load data
    df = load_data()

    # ── 1. Observation-level ANOVA ──
    obs_anova, r2, adj_r2 = observation_level_anova(df)

    # ── 2. Cell-means decomposition ──
    cell_anova, cell_r2 = cell_means_decomposition(df)

    # ── 3. Omega-squared (less biased) ──
    omega2 = compute_omega_squared(df)

    # ── 4. Per-model variance profiles ──
    profiles = per_model_variance_profiles(df)

    # ── 5. Scaffold sensitivity index ──
    sensitivity = compute_scaffold_sensitivity_index(df)

    # ── 6. Marginal safety rates ──
    marginals = compute_marginal_safety_rates(df)

    # ── 7. Framework utility assessment ──
    assessment = assess_framework_utility(obs_anova, cell_anova, profiles, sensitivity, omega2)

    # ── Save results ──
    output = {
        "metadata": {
            "n_observations": len(df),
            "n_models": len(df["model"].unique()),
            "n_scaffolds": len(df["scaffold"].unique()),
            "n_benchmarks": len(df["benchmark"].unique()),
            "overall_safe_rate": round(float(df["is_safe"].mean()), 4),
        },
        "observation_level_anova": {
            "r_squared": round(float(r2), 6),
            "adj_r_squared": round(float(adj_r2), 6),
            "factors": obs_anova,
        },
        "cell_means_anova": {
            "r_squared": round(float(cell_r2), 6),
            "factors": cell_anova,
        },
        "omega_squared": omega2,
        "per_model_profiles": profiles,
        "scaffold_sensitivity": sensitivity,
        "marginal_rates": marginals,
        "framework_assessment": assessment,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()
