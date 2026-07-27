#!/usr/bin/env python3
"""
Generalizability Theory Analysis for Safety Under Scaffolding
==============================================================

Design: p x I x J (fully crossed, cell-means approach)
  - p = model (6 levels, RANDOM — object of measurement)
  - I = scaffold/config (4 levels, FIXED)
  - J = benchmark (4 levels, FIXED)

Since scaffold and benchmark are FIXED facets:
  - We do NOT estimate sigma2_I, sigma2_J, or sigma2_IJ as random variance components
  - The relative and absolute G-coefficients are identical (fixed facet main effects
    do not enter either error term)
  - The D-study answers: "If we evaluated on N scaffolds from THIS SET and M benchmarks
    from THIS SET, what would reliability be?"

Reference: Brennan, R. L. (2001). Generalizability Theory. New York: Springer.
           Chapter 3 (mixed models with fixed facets).

Author: David Gringras
"""

import json
import os
import numpy as np
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_DIR, "results", "canonical_primary_dataset.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "analysis", "outputs", "gtheory_results.json")


def load_data(path):
    """Load canonical dataset and compute cell-mean safety rates."""
    records = []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    print(f"Loaded {len(records):,} observations")

    # Aggregate to cell means: (model, config, benchmark) -> (n_safe, n_total)
    cell_counts = defaultdict(lambda: [0, 0])
    for rec in records:
        key = (rec["model_id"], rec["config_id"], rec["benchmark_id"])
        cell_counts[key][1] += 1
        if rec["is_safe"]:
            cell_counts[key][0] += 1

    models = sorted(set(k[0] for k in cell_counts))
    scaffolds = sorted(set(k[1] for k in cell_counts))
    benchmarks = sorted(set(k[2] for k in cell_counts))

    n_p = len(models)
    n_i = len(scaffolds)
    n_j = len(benchmarks)

    print(f"Design: {n_p} models x {n_i} scaffolds x {n_j} benchmarks = {n_p * n_i * n_j} cells")

    # Build cell-mean matrix: Y[p, i, j] = safety rate
    Y = np.zeros((n_p, n_i, n_j))
    N_cell = np.zeros((n_p, n_i, n_j), dtype=int)
    for (m, s, b), (n_safe, n_total) in cell_counts.items():
        p_idx = models.index(m)
        i_idx = scaffolds.index(s)
        j_idx = benchmarks.index(b)
        Y[p_idx, i_idx, j_idx] = n_safe / n_total
        N_cell[p_idx, i_idx, j_idx] = n_total

    return Y, N_cell, models, scaffolds, benchmarks


def compute_variance_components(Y):
    """
    ANOVA-based variance component estimation for a p x I x J mixed design
    using cell means (Brennan 2001, Chapter 3).

    p = model (RANDOM, object of measurement)
    I = scaffold (FIXED)
    J = benchmark (FIXED)

    Since I and J are fixed, we only estimate the random variance components:
      sigma2_p   — model ("true score") variance
      sigma2_pI  — model x scaffold interaction
      sigma2_pJ  — model x benchmark interaction
      sigma2_pIJ — three-way interaction + residual (confounded)

    The fixed effects (sigma2_I, sigma2_J, sigma2_IJ) are nuisance parameters
    absorbed into the fixed part of the model. They are not treated as random
    variance components.

    Expected Mean Squares (for the random components involving p):
      E[MS_p]   = sigma2_pIJ + n_J * sigma2_pI + n_I * sigma2_pJ + n_I * n_J * sigma2_p
      E[MS_pI]  = sigma2_pIJ + n_J * sigma2_pI
      E[MS_pJ]  = sigma2_pIJ + n_I * sigma2_pJ
      E[MS_pIJ] = sigma2_pIJ
    """
    n_p, n_i, n_j = Y.shape

    # Grand mean
    grand_mean = Y.mean()

    # Marginal means
    mean_p = Y.mean(axis=(1, 2))   # (n_p,)   — model means
    mean_i = Y.mean(axis=(0, 2))   # (n_i,)   — scaffold means
    mean_j = Y.mean(axis=(0, 1))   # (n_j,)   — benchmark means
    mean_pi = Y.mean(axis=2)       # (n_p, n_i)
    mean_pj = Y.mean(axis=1)       # (n_p, n_j)
    mean_ij = Y.mean(axis=0)       # (n_i, n_j)

    # Sums of squares (using standard ANOVA decomposition on the 3-way table)
    SS_p = n_i * n_j * np.sum((mean_p - grand_mean) ** 2)
    SS_i = n_p * n_j * np.sum((mean_i - grand_mean) ** 2)
    SS_j = n_p * n_i * np.sum((mean_j - grand_mean) ** 2)

    SS_pi = n_j * np.sum((mean_pi - mean_p[:, None] - mean_i[None, :] + grand_mean) ** 2)
    SS_pj = n_i * np.sum((mean_pj - mean_p[:, None] - mean_j[None, :] + grand_mean) ** 2)
    SS_ij = n_p * np.sum((mean_ij - mean_i[:, None] - mean_j[None, :] + grand_mean) ** 2)

    SS_pij = np.sum(
        (Y
         - mean_pi[:, :, None]
         - mean_pj[:, None, :]
         - mean_ij[None, :, :]
         + mean_p[:, None, None]
         + mean_i[None, :, None]
         + mean_j[None, None, :]
         - grand_mean) ** 2
    )

    # Degrees of freedom
    df_p = n_p - 1
    df_i = n_i - 1
    df_j = n_j - 1
    df_pi = (n_p - 1) * (n_i - 1)
    df_pj = (n_p - 1) * (n_j - 1)
    df_ij = (n_i - 1) * (n_j - 1)
    df_pij = (n_p - 1) * (n_i - 1) * (n_j - 1)

    # Mean squares
    MS_p = SS_p / df_p
    MS_i = SS_i / df_i
    MS_j = SS_j / df_j
    MS_pi = SS_pi / df_pi
    MS_pj = SS_pj / df_pj
    MS_ij = SS_ij / df_ij
    MS_pij = SS_pij / df_pij

    # Variance components (from EMS equations, only random effects involving p)
    sigma2_pij = MS_pij
    sigma2_pi = (MS_pi - MS_pij) / n_j
    sigma2_pj = (MS_pj - MS_pij) / n_i
    sigma2_p = (MS_p - MS_pi - MS_pj + MS_pij) / (n_i * n_j)

    # Store raw estimates (before truncation)
    vc_raw = {
        "sigma2_p": float(sigma2_p),
        "sigma2_pI": float(sigma2_pi),
        "sigma2_pJ": float(sigma2_pj),
        "sigma2_pIJ": float(sigma2_pij),
    }

    # Truncate negatives to zero (standard practice, Brennan 2001)
    sigma2_p = max(0.0, sigma2_p)
    sigma2_pi = max(0.0, sigma2_pi)
    sigma2_pj = max(0.0, sigma2_pj)
    sigma2_pij = max(0.0, sigma2_pij)

    vc = {
        "sigma2_p": float(sigma2_p),
        "sigma2_pI": float(sigma2_pi),
        "sigma2_pJ": float(sigma2_pj),
        "sigma2_pIJ": float(sigma2_pij),
    }

    ms = {
        "MS_p": float(MS_p), "MS_I": float(MS_i), "MS_J": float(MS_j),
        "MS_pI": float(MS_pi), "MS_pJ": float(MS_pj), "MS_IJ": float(MS_ij),
        "MS_pIJ": float(MS_pij),
    }

    df_dict = {
        "df_p": int(df_p), "df_I": int(df_i), "df_J": int(df_j),
        "df_pI": int(df_pi), "df_pJ": int(df_pj), "df_IJ": int(df_ij),
        "df_pIJ": int(df_pij),
    }

    ss = {
        "SS_p": float(SS_p), "SS_I": float(SS_i), "SS_J": float(SS_j),
        "SS_pI": float(SS_pi), "SS_pJ": float(SS_pj), "SS_IJ": float(SS_ij),
        "SS_pIJ": float(SS_pij),
    }

    return vc, vc_raw, ms, df_dict, ss


def compute_g_coefficient(vc, n_i, n_j):
    """
    Compute G-coefficient for a p x I x J mixed design with fixed facets.

    For fixed facets I and J:
      Relative error = sigma2_pI / n_I + sigma2_pJ / n_J + sigma2_pIJ / (n_I * n_J)
      Absolute error = SAME (because fixed facet main effects do NOT enter the
                       absolute error term — they cancel out for all persons)

    Therefore:  Phi = G_rel  for this design.

    G = sigma2_p / (sigma2_p + relative_error)
    """
    s2_p = vc["sigma2_p"]
    s2_pi = vc["sigma2_pI"]
    s2_pj = vc["sigma2_pJ"]
    s2_pij = vc["sigma2_pIJ"]

    rel_error = s2_pi / n_i + s2_pj / n_j + s2_pij / (n_i * n_j)
    abs_error = rel_error  # identical for fixed facets

    denom = s2_p + rel_error
    if denom == 0:
        return 0.0, 0.0, rel_error, abs_error

    g_rel = s2_p / denom
    phi = s2_p / (s2_p + abs_error)

    return float(g_rel), float(phi), float(rel_error), float(abs_error)


def d_study(vc, n_i_range, n_j_range):
    """
    D-study: compute G for varying numbers of scaffolds and benchmarks.
    Returns a dict mapping (n_i, n_j) -> {"G_rel": ..., "Phi": ...}
    """
    results = {}
    for ni in n_i_range:
        for nj in n_j_range:
            g, phi, _, _ = compute_g_coefficient(vc, ni, nj)
            results[(ni, nj)] = {"G_rel": g, "Phi": phi}
    return results


def bootstrap_g(Y, n_boot=10000, seed=42):
    """
    Bootstrap the G-coefficient by resampling MODELS (the random facet)
    with replacement. We draw n_p models (with replacement) from the n_p
    available, recompute variance components and G each time.
    """
    rng = np.random.default_rng(seed)
    n_p, n_i, n_j = Y.shape
    g_boot = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        Y_b = Y[idx, :, :]
        vc_b, _, _, _, _ = compute_variance_components(Y_b)
        g_b, _, _, _ = compute_g_coefficient(vc_b, n_i, n_j)
        g_boot[b] = g_b

    ci_lo = float(np.percentile(g_boot, 2.5))
    ci_hi = float(np.percentile(g_boot, 97.5))
    g_mean = float(np.mean(g_boot))
    g_median = float(np.median(g_boot))

    return g_boot, ci_lo, ci_hi, g_mean, g_median


def main():
    print("=" * 76)
    print("  GENERALIZABILITY THEORY ANALYSIS — Safety Under Scaffolding")
    print("  Cell-Means Approach | p (model, random) x I (scaffold, fixed) x J (benchmark, fixed)")
    print("=" * 76)
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 1. Load data and compute cell means
    # ────────────────────────────────────────────────────────────────────────
    Y, N_cell, models, scaffolds, benchmarks = load_data(DATA_PATH)
    n_p, n_i, n_j = Y.shape

    print(f"\nModels (p, RANDOM):      {models}")
    print(f"Scaffolds (I, FIXED):    {scaffolds}")
    print(f"Benchmarks (J, FIXED):   {benchmarks}")
    print(f"\nCell sizes: min={int(N_cell.min())}, max={int(N_cell.max())}, "
          f"harmonic mean={len(N_cell.flat) / np.sum(1.0 / N_cell):.1f}")
    print(f"Grand mean safety rate:  {Y.mean():.4f}")
    print()

    # ── Cell means table ──
    print("-" * 76)
    print("CELL-MEAN SAFETY RATES  (rows=models, columns=scaffold/benchmark)")
    print("-" * 76)

    # Column headers: scaffold abbreviation + benchmark abbreviation
    col_labels = []
    for s in scaffolds:
        for b in benchmarks:
            col_labels.append(f"{s[:5]}/{b[:4]}")

    header = f"{'Model':<12}" + "".join(f"{c:>13}" for c in col_labels) + f"{'  Mean':>8}"
    print(header)
    for p_idx, model in enumerate(models):
        row = f"{model:<12}"
        for i_idx in range(n_i):
            for j_idx in range(n_j):
                row += f"{Y[p_idx, i_idx, j_idx]:>13.4f}"
        row += f"{Y[p_idx].mean():>8.4f}"
        print(row)

    # Marginal means row
    row = f"{'Mean':<12}"
    for i_idx in range(n_i):
        for j_idx in range(n_j):
            row += f"{Y[:, i_idx, j_idx].mean():>13.4f}"
    row += f"{Y.mean():>8.4f}"
    print(row)
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 2. ANOVA-based variance components
    # ────────────────────────────────────────────────────────────────────────
    vc, vc_raw, ms, df_dict, ss = compute_variance_components(Y)
    total_vc = sum(vc.values())

    print("-" * 76)
    print("ANOVA TABLE (on 96 cell means)")
    print("-" * 76)
    print(f"{'Source':<20} {'df':>5} {'SS':>14} {'MS':>14}")
    print(f"{'':->20} {'':->5} {'':->14} {'':->14}")
    for src, label in [("p", "p (model)"), ("I", "I (scaffold)"), ("J", "J (benchmark)"),
                       ("pI", "pI"), ("pJ", "pJ"), ("IJ", "IJ"), ("pIJ", "pIJ (residual)")]:
        print(f"{label:<20} {df_dict[f'df_{src}']:>5} {ss[f'SS_{src}']:>14.8f} {ms[f'MS_{src}']:>14.8f}")

    total_ss = sum(ss.values())
    total_df = sum(df_dict.values())
    print(f"{'Total':<20} {total_df:>5} {total_ss:>14.8f}")
    print()

    # ── Variance components ──
    print("-" * 76)
    print("VARIANCE COMPONENTS (random effects only — fixed facets excluded)")
    print("-" * 76)
    print(f"{'Component':<30} {'Raw':>12} {'Truncated':>12} {'% of Total':>12}")
    print(f"{'':->30} {'':->12} {'':->12} {'':->12}")

    labels = [
        ("sigma2_p  (model)", "sigma2_p"),
        ("sigma2_pI (model x scaffold)", "sigma2_pI"),
        ("sigma2_pJ (model x benchmark)", "sigma2_pJ"),
        ("sigma2_pIJ (residual)", "sigma2_pIJ"),
    ]
    for label, key in labels:
        pct = (vc[key] / total_vc * 100) if total_vc > 0 else 0.0
        neg = " ***" if vc_raw[key] < 0 else ""
        print(f"{label:<30} {vc_raw[key]:>12.8f} {vc[key]:>12.8f} {pct:>11.1f}%{neg}")
    print(f"{'Total':<30} {'':>12} {total_vc:>12.8f} {'100.0%':>12}")
    print()

    # ── EMS derivation display ──
    print("Expected Mean Square equations (Brennan 2001, mixed model):")
    print(f"  E[MS_p]   = sigma2_pIJ + {n_j}*sigma2_pI + {n_i}*sigma2_pJ + {n_i*n_j}*sigma2_p")
    print(f"  E[MS_pI]  = sigma2_pIJ + {n_j}*sigma2_pI")
    print(f"  E[MS_pJ]  = sigma2_pIJ + {n_i}*sigma2_pJ")
    print(f"  E[MS_pIJ] = sigma2_pIJ")
    print()
    print("Solving:")
    print(f"  sigma2_pIJ = MS_pIJ = {vc['sigma2_pIJ']:.8f}")
    print(f"  sigma2_pI  = (MS_pI - MS_pIJ) / n_J = ({ms['MS_pI']:.8f} - {ms['MS_pIJ']:.8f}) / {n_j} = {vc_raw['sigma2_pI']:.8f}")
    print(f"  sigma2_pJ  = (MS_pJ - MS_pIJ) / n_I = ({ms['MS_pJ']:.8f} - {ms['MS_pIJ']:.8f}) / {n_i} = {vc_raw['sigma2_pJ']:.8f}")
    print(f"  sigma2_p   = (MS_p - MS_pI - MS_pJ + MS_pIJ) / (n_I * n_J)")
    print(f"             = ({ms['MS_p']:.8f} - {ms['MS_pI']:.8f} - {ms['MS_pJ']:.8f} + {ms['MS_pIJ']:.8f}) / {n_i * n_j}")
    print(f"             = {vc_raw['sigma2_p']:.8f}")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 3. G-coefficient
    # ────────────────────────────────────────────────────────────────────────
    g_rel, phi, rel_error, abs_error = compute_g_coefficient(vc, n_i, n_j)

    print("-" * 76)
    print(f"G-COEFFICIENT (current design: n_I={n_i} scaffolds, n_J={n_j} benchmarks)")
    print("-" * 76)
    print()
    print(f"  sigma2_p (universe/true score):   {vc['sigma2_p']:.8f}")
    print(f"  Relative error variance:          {rel_error:.8f}")
    print(f"    = sigma2_pI/{n_i} + sigma2_pJ/{n_j} + sigma2_pIJ/({n_i}*{n_j})")
    print(f"    = {vc['sigma2_pI']:.8f}/{n_i} + {vc['sigma2_pJ']:.8f}/{n_j} + {vc['sigma2_pIJ']:.8f}/{n_i*n_j}")
    print(f"    = {vc['sigma2_pI']/n_i:.8f} + {vc['sigma2_pJ']/n_j:.8f} + {vc['sigma2_pIJ']/(n_i*n_j):.8f}")
    print()
    print(f"  G (relative)  = sigma2_p / (sigma2_p + rel_error)")
    print(f"                = {vc['sigma2_p']:.8f} / ({vc['sigma2_p']:.8f} + {rel_error:.8f})")
    print(f"                = {g_rel:.4f}")
    print()
    print(f"  Phi (absolute) = {phi:.4f}")
    print(f"  (Phi = G_rel because I and J are FIXED facets)")
    print()

    if g_rel >= 0.90:
        adequacy = "EXCELLENT (G >= 0.90)"
    elif g_rel >= 0.80:
        adequacy = "ADEQUATE (G >= 0.80)"
    elif g_rel >= 0.70:
        adequacy = "MARGINAL (0.70 <= G < 0.80)"
    else:
        adequacy = "INADEQUATE (G < 0.70)"
    print(f"  Reliability threshold: {adequacy}")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 4. Bootstrap 95% CI
    # ────────────────────────────────────────────────────────────────────────
    print("-" * 76)
    print("BOOTSTRAP (10,000 resamples of models with replacement)")
    print("-" * 76)
    g_boot, ci_lo, ci_hi, g_mean_boot, g_median_boot = bootstrap_g(Y, n_boot=10000)
    print(f"  Point estimate G:   {g_rel:.4f}")
    print(f"  Bootstrap mean G:   {g_mean_boot:.4f}")
    print(f"  Bootstrap median G: {g_median_boot:.4f}")
    print(f"  95% CI:             [{ci_lo:.4f}, {ci_hi:.4f}]")
    pct_above_80 = float((g_boot >= 0.80).mean() * 100)
    pct_above_70 = float((g_boot >= 0.70).mean() * 100)
    print(f"  P(G >= 0.80):       {pct_above_80:.1f}%")
    print(f"  P(G >= 0.70):       {pct_above_70:.1f}%")
    print(f"  Bootstrap SD:       {float(np.std(g_boot)):.4f}")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 5. D-study
    # ────────────────────────────────────────────────────────────────────────
    n_i_range = list(range(1, 9))
    n_j_range = list(range(1, 13))
    dstudy = d_study(vc, n_i_range, n_j_range)

    print("-" * 76)
    print("D-STUDY: G-coefficient by n_scaffolds (rows) x n_benchmarks (columns)")
    print("-" * 76)

    header = f"{'n_I \\ n_J':<10}" + "".join(f"{nj:>7}" for nj in n_j_range)
    print(header)
    print("-" * (10 + 7 * len(n_j_range)))
    for ni in n_i_range:
        row = f"{ni:<10}"
        for nj in n_j_range:
            g = dstudy[(ni, nj)]["G_rel"]
            marker = " *" if abs(g - g_rel) < 0.0001 and ni == n_i and nj == n_j else "  "
            row += f"{g:>5.3f}{marker}"
        print(row)
    print("  (* = current design)")
    print()

    # Key D-study findings
    print("Key D-study findings:")
    print(f"  Current design (4 scaffolds, 4 benchmarks):   G = {dstudy[(4, 4)]['G_rel']:.4f}")
    print(f"  With 6 benchmarks (4 scaffolds):              G = {dstudy[(4, 6)]['G_rel']:.4f}")
    print(f"  With 8 benchmarks (4 scaffolds):              G = {dstudy[(4, 8)]['G_rel']:.4f}")
    print(f"  With 12 benchmarks (4 scaffolds):             G = {dstudy[(4, 12)]['G_rel']:.4f}")
    print(f"  With 8 scaffolds (4 benchmarks):              G = {dstudy[(8, 4)]['G_rel']:.4f}")
    print(f"  With 8 scaffolds, 8 benchmarks:               G = {dstudy[(8, 8)]['G_rel']:.4f}")
    print(f"  With 8 scaffolds, 12 benchmarks:              G = {dstudy[(8, 12)]['G_rel']:.4f}")
    print()

    # Find minimum design for G >= 0.80
    print("  Minimum design for G >= 0.80:")
    found_any = False
    for ni in n_i_range:
        for nj in n_j_range:
            if dstudy[(ni, nj)]["G_rel"] >= 0.80:
                if not found_any:
                    print(f"    First combination: n_I={ni}, n_J={nj} -> G = {dstudy[(ni, nj)]['G_rel']:.4f}")
                    found_any = True
    if not found_any:
        print("    G < 0.80 for all tested combinations (n_I up to 8, n_J up to 12)")

    # Find minimum design for G >= 0.70
    print("  Minimum design for G >= 0.70:")
    found_any = False
    for ni in n_i_range:
        for nj in n_j_range:
            if dstudy[(ni, nj)]["G_rel"] >= 0.70:
                if not found_any:
                    print(f"    First combination: n_I={ni}, n_J={nj} -> G = {dstudy[(ni, nj)]['G_rel']:.4f}")
                    found_any = True
    if not found_any:
        print("    G < 0.70 for all tested combinations")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 6. Interpretation
    # ────────────────────────────────────────────────────────────────────────
    print("-" * 76)
    print("INTERPRETATION")
    print("-" * 76)

    component_names = {
        "sigma2_p": "model differences (true score / universe score variance)",
        "sigma2_pI": "model x scaffold interaction",
        "sigma2_pJ": "model x benchmark interaction",
        "sigma2_pIJ": "three-way interaction + residual",
    }

    # Dominant source
    vc_sorted = sorted(vc.items(), key=lambda x: x[1], reverse=True)
    dominant_key, dominant_val = vc_sorted[0]
    dominant_pct = (dominant_val / total_vc * 100) if total_vc > 0 else 0

    print(f"\n  Largest variance component: {component_names[dominant_key]}")
    print(f"  ({dominant_pct:.1f}% of total random variance)")
    print()

    if total_vc > 0:
        p_pct = vc["sigma2_p"] / total_vc * 100
        pi_pct = vc["sigma2_pI"] / total_vc * 100
        pj_pct = vc["sigma2_pJ"] / total_vc * 100
        pij_pct = vc["sigma2_pIJ"] / total_vc * 100

        print(f"  Model (true score) variance:       {p_pct:.1f}%")
        print(f"  Model x scaffold interaction:      {pi_pct:.1f}%")
        print(f"  Model x benchmark interaction:     {pj_pct:.1f}%")
        print(f"  Three-way + residual:              {pij_pct:.1f}%")
        print()

        if p_pct > 50:
            print("  -> Most variance comes from MODEL DIFFERENCES. Models are well-differentiated")
            print("     on safety, and their rankings are relatively stable across conditions.")
        elif p_pct > 25:
            print("  -> Model differences account for a substantial share of variance, but")
            print("     interactions are also important — rankings depend on the evaluation context.")
        else:
            print("  -> Model differences are a MINORITY of the total variance. Rankings depend")
            print("     heavily on which scaffold/benchmark combination is used.")
        print()

        if pj_pct > pi_pct:
            print("  Model x benchmark interaction > model x scaffold interaction:")
            print("  -> Models' relative safety rankings depend MORE on WHICH BENCHMARK is used")
            print("     than on WHICH SCAFFOLD is deployed. Benchmark selection is the larger")
            print("     source of measurement noise for model comparisons.")
        elif pi_pct > pj_pct:
            print("  Model x scaffold interaction > model x benchmark interaction:")
            print("  -> Scaffold choice creates more measurement noise than benchmark choice.")
            print("     Different scaffolds affect models' relative rankings more than different benchmarks.")
        else:
            print("  Model x scaffold and model x benchmark interactions are roughly equal.")
        print()

    if g_rel >= 0.80:
        print(f"  CONCLUSION: G = {g_rel:.3f} (>= 0.80) indicates ADEQUATE reliability.")
        print("  The current design with 4 scaffolds and 4 benchmarks provides a dependable")
        print("  measurement of model safety. We can confidently rank models on their safety")
        print("  under scaffolding, and these rankings would generalize to other combinations")
        print("  of scaffolds and benchmarks from this universe of conditions.")
    elif g_rel >= 0.70:
        print(f"  CONCLUSION: G = {g_rel:.3f} (0.70-0.80) indicates MARGINAL reliability.")
        print("  Model rankings are somewhat stable but could benefit from more conditions.")
        print("  The D-study shows how many additional scaffolds/benchmarks would be needed.")
    else:
        print(f"  CONCLUSION: G = {g_rel:.3f} (< 0.70) indicates LOW reliability.")
        print("  Model safety rankings are substantially affected by the choice of scaffold")
        print("  and benchmark. The D-study shows what design would improve reliability.")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 7. Save results
    # ────────────────────────────────────────────────────────────────────────
    dstudy_json = {}
    for (ni, nj), vals in dstudy.items():
        dstudy_json[f"{ni}x{nj}"] = vals

    results = {
        "design": {
            "type": "p x I x J (mixed: model=random, scaffold=fixed, benchmark=fixed)",
            "n_models_p": n_p,
            "n_scaffolds_I": n_i,
            "n_benchmarks_J": n_j,
            "n_cells": n_p * n_i * n_j,
            "n_observations": int(N_cell.sum()),
            "models": models,
            "scaffolds": scaffolds,
            "benchmarks": benchmarks,
            "grand_mean_safety": round(float(Y.mean()), 4),
            "cell_size_min": int(N_cell.min()),
            "cell_size_max": int(N_cell.max()),
            "cell_size_harmonic_mean": round(float(len(N_cell.flat) / np.sum(1.0 / N_cell)), 1),
        },
        "anova": {
            "mean_squares": ms,
            "degrees_of_freedom": df_dict,
            "sums_of_squares": ss,
        },
        "variance_components": {
            "estimated_truncated": vc,
            "raw_before_truncation": vc_raw,
            "proportions": {
                k: round(float(v / total_vc), 4) if total_vc > 0 else 0.0
                for k, v in vc.items()
            },
            "total_random_variance": round(float(total_vc), 8),
            "note": "Only random components involving p (model) are estimated. "
                    "Fixed facet main effects (sigma2_I, sigma2_J, sigma2_IJ) are excluded.",
        },
        "g_coefficient": {
            "G_relative": round(float(g_rel), 4),
            "Phi_absolute": round(float(phi), 4),
            "relative_error_variance": round(float(rel_error), 8),
            "absolute_error_variance": round(float(abs_error), 8),
            "meets_080_threshold": bool(g_rel >= 0.80),
            "meets_070_threshold": bool(g_rel >= 0.70),
            "adequacy": adequacy,
            "note": "Phi = G_rel because scaffold and benchmark are FIXED facets.",
        },
        "bootstrap": {
            "n_resamples": 10000,
            "seed": 42,
            "method": "resample models (rows of cell-mean matrix) with replacement",
            "G_point_estimate": round(float(g_rel), 4),
            "G_bootstrap_mean": round(float(g_mean_boot), 4),
            "G_bootstrap_median": round(float(g_median_boot), 4),
            "CI_95_lower": round(float(ci_lo), 4),
            "CI_95_upper": round(float(ci_hi), 4),
            "pct_above_080": round(float(pct_above_80), 1),
            "pct_above_070": round(float(pct_above_70), 1),
            "bootstrap_SD": round(float(np.std(g_boot)), 4),
        },
        "d_study": dstudy_json,
        "cell_means": {
            f"{models[p]}|{scaffolds[i]}|{benchmarks[j]}": round(float(Y[p, i, j]), 4)
            for p in range(n_p) for i in range(n_i) for j in range(n_j)
        },
        "model_marginal_means": {
            models[p]: round(float(Y[p].mean()), 4) for p in range(n_p)
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {OUTPUT_PATH}")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # 8. Paper-ready summary
    # ────────────────────────────────────────────────────────────────────────
    print("=" * 76)
    print("  PAPER-READY SUMMARY")
    print("=" * 76)
    print()
    print(f"  G-coefficient (relative) = {g_rel:.3f}  [95% bootstrap CI: {ci_lo:.3f} - {ci_hi:.3f}]")
    print(f"  Phi (absolute)           = {phi:.3f}  (= G_rel for fixed facets)")
    print()
    print("  Variance decomposition (random effects only):")
    for label, key in labels:
        pct = (vc[key] / total_vc * 100) if total_vc > 0 else 0
        print(f"    {label:<35} {vc[key]:.6f}  ({pct:.1f}%)")
    print()
    print(f"  Interpretation: ", end="")
    if g_rel >= 0.80:
        print("Adequate reliability (G >= 0.80) for ranking models on safety.")
        print("  Model safety rankings generalize across scaffold and benchmark conditions.")
    elif g_rel >= 0.70:
        print("Marginal reliability (G = 0.70-0.80). Rankings are moderately stable")
        print("  but sensitive to evaluation conditions.")
    else:
        print("Below conventional thresholds. Model safety rankings depend substantially")
        print("  on the specific scaffold-benchmark combination used for evaluation.")
    print()
    print("  D-study recommendation: ", end="")
    # Find the cheapest way to hit 0.80
    best = None
    for ni in range(1, 9):
        for nj in range(1, 13):
            g = dstudy[(ni, nj)]["G_rel"]
            if g >= 0.80:
                cost = ni * nj  # proxy for total measurement effort
                if best is None or cost < best[2]:
                    best = (ni, nj, cost, g)
    if best:
        print(f"Minimum design for G >= 0.80: {best[0]} scaffolds x {best[1]} benchmarks (G = {best[3]:.3f})")
    else:
        print("G < 0.80 for all designs up to 8 scaffolds x 12 benchmarks.")
        # Find best achievable
        best_g = max(dstudy[(ni, nj)]["G_rel"] for ni in range(1, 9) for nj in range(1, 13))
        print(f"  Best achievable: G = {best_g:.3f} (at 8 scaffolds x 12 benchmarks)")
    print()


if __name__ == "__main__":
    main()
