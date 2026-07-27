"""
Core statistical analyses for scaffold safety evaluation.

Implements the pre-registered analysis plan:
- Safety rate computation
- Effect sizes (Cohen's h, risk difference, odds ratio, NNH)
- TOST equivalence tests
- Chi-squared / Fisher exact tests
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class SafetyRate:
    """Safety rate for a specific cell (benchmark x model x config)."""
    benchmark_id: str
    model_id: str
    config_id: str
    n_total: int
    n_safe: int
    safety_rate: float
    se: float


@dataclass
class EffectSize:
    """Effect size comparing scaffold vs direct baseline."""
    benchmark_id: str
    model_id: str
    config_id: str
    p_direct: float
    p_scaffold: float
    cohen_h: float
    risk_difference: float
    odds_ratio: float
    nnh: float


@dataclass
class TOSTResult:
    """Result of TOST equivalence test."""
    benchmark_id: str
    config_id: str
    direct_rate: float
    scaffold_rate: float
    risk_difference: float
    ci90_low: float
    ci90_high: float
    tost_p: float
    equivalent: bool
    margin: float


def compute_safety_rates(
    results: list[dict[str, Any]],
) -> list[SafetyRate]:
    """Compute safety rates grouped by benchmark x model x config.

    Parameters
    ----------
    results : list[dict]
        Each dict must have: benchmark_id, model_id, config_id, is_safe (bool).

    Returns
    -------
    list[SafetyRate]
    """
    from collections import defaultdict
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "safe": 0}
    )

    for r in results:
        if r.get("is_safe") is None:
            continue
        key = (r["benchmark_id"], r["model_id"], r["config_id"])
        counts[key]["total"] += 1
        if r["is_safe"]:
            counts[key]["safe"] += 1

    rates = []
    for (bid, mid, cid), c in sorted(counts.items()):
        rate = c["safe"] / c["total"] if c["total"] > 0 else 0.0
        se = math.sqrt(rate * (1 - rate) / c["total"]) if c["total"] > 0 else 0.0
        rates.append(SafetyRate(
            benchmark_id=bid, model_id=mid, config_id=cid,
            n_total=c["total"], n_safe=c["safe"],
            safety_rate=rate, se=se,
        ))
    return rates


def compute_effect_sizes(
    rates: list[SafetyRate],
) -> list[EffectSize]:
    """Compute Cohen's h, risk difference, odds ratio, and NNH.

    Compares each scaffold config vs direct baseline per benchmark x model cell.
    """
    # Index rates by (benchmark, model, config)
    rate_map: dict[tuple[str, str, str], SafetyRate] = {
        (r.benchmark_id, r.model_id, r.config_id): r for r in rates
    }

    # Find all (benchmark, model) pairs
    bm_pairs = {(r.benchmark_id, r.model_id) for r in rates}
    configs = {r.config_id for r in rates} - {"direct"}

    effects = []
    for bid, mid in sorted(bm_pairs):
        direct_key = (bid, mid, "direct")
        if direct_key not in rate_map:
            continue
        p0 = rate_map[direct_key].safety_rate

        for cid in sorted(configs):
            scaffold_key = (bid, mid, cid)
            if scaffold_key not in rate_map:
                continue
            p1 = rate_map[scaffold_key].safety_rate

            # Cohen's h
            h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p0))

            # Risk difference
            rd = p1 - p0

            # Odds ratio
            odds0 = p0 / (1 - p0) if 0 < p0 < 1 else float("inf")
            odds1 = p1 / (1 - p1) if 0 < p1 < 1 else float("inf")
            or_val = odds1 / odds0 if odds0 > 0 and odds0 != float("inf") else float("inf")

            # NNH
            nnh = 1 / abs(rd) if abs(rd) > 0 else float("inf")

            effects.append(EffectSize(
                benchmark_id=bid, model_id=mid, config_id=cid,
                p_direct=p0, p_scaffold=p1,
                cohen_h=h, risk_difference=rd,
                odds_ratio=or_val, nnh=nnh,
            ))
    return effects


def compute_nnh(direct_rate: float, scaffold_rate: float) -> float:
    """Compute Number Needed to Harm.

    For every NNH cases processed through scaffolding, one additional
    unsafe response occurs compared to direct API.
    """
    rd = abs(scaffold_rate - direct_rate)
    return 1 / rd if rd > 0 else float("inf")


def run_tost(
    rates: list[SafetyRate],
    margin: float = 0.02,
) -> list[TOSTResult]:
    """Run TOST equivalence test for each scaffold vs direct, pooled across models.

    Tests whether |p_scaffold - p_direct| < margin.
    """
    from collections import defaultdict
    import math

    # Pool across models per (benchmark, config)
    pooled: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "safe": 0}
    )
    for r in rates:
        key = (r.benchmark_id, r.config_id)
        pooled[key]["total"] += r.n_total
        pooled[key]["safe"] += r.n_safe

    benchmarks = sorted({r.benchmark_id for r in rates})
    configs = sorted({r.config_id for r in rates} - {"direct"})

    results = []
    for bid in benchmarks:
        direct_key = (bid, "direct")
        if direct_key not in pooled or pooled[direct_key]["total"] == 0:
            continue
        d_rate = pooled[direct_key]["safe"] / pooled[direct_key]["total"]
        d_n = pooled[direct_key]["total"]

        for cid in configs:
            scaffold_key = (bid, cid)
            if scaffold_key not in pooled or pooled[scaffold_key]["total"] == 0:
                continue
            s_rate = pooled[scaffold_key]["safe"] / pooled[scaffold_key]["total"]
            s_n = pooled[scaffold_key]["total"]

            rd = s_rate - d_rate
            se_rd = math.sqrt(
                d_rate * (1 - d_rate) / d_n + s_rate * (1 - s_rate) / s_n
            )

            if se_rd > 0:
                # Standard normal CDF approximation
                def _norm_cdf(x: float) -> float:
                    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

                z_lower = (rd - (-margin)) / se_rd
                z_upper = (rd - margin) / se_rd
                p_lower = 1 - _norm_cdf(z_lower)
                p_upper = _norm_cdf(z_upper)
                tost_p = max(p_lower, p_upper)
            else:
                tost_p = 0.0 if abs(rd) < margin else 1.0

            ci90_low = rd - 1.645 * se_rd
            ci90_high = rd + 1.645 * se_rd
            equivalent = (ci90_low > -margin) and (ci90_high < margin)

            results.append(TOSTResult(
                benchmark_id=bid, config_id=cid,
                direct_rate=d_rate, scaffold_rate=s_rate,
                risk_difference=rd, ci90_low=ci90_low, ci90_high=ci90_high,
                tost_p=tost_p, equivalent=equivalent, margin=margin,
            ))

    return results


def run_chi_squared(
    rates: list[SafetyRate],
) -> list[dict[str, Any]]:
    """Run chi-squared / Fisher exact tests for scaffold vs direct.

    Returns list of dicts with test results per (benchmark, model, config).
    """
    rate_map: dict[tuple[str, str, str], SafetyRate] = {
        (r.benchmark_id, r.model_id, r.config_id): r for r in rates
    }

    bm_pairs = sorted({(r.benchmark_id, r.model_id) for r in rates})
    configs = sorted({r.config_id for r in rates} - {"direct"})

    results = []
    for bid, mid in bm_pairs:
        direct_key = (bid, mid, "direct")
        if direct_key not in rate_map:
            continue
        dr = rate_map[direct_key]
        d_safe = dr.n_safe
        d_unsafe = dr.n_total - dr.n_safe

        for cid in configs:
            scaffold_key = (bid, mid, cid)
            if scaffold_key not in rate_map:
                continue
            sr = rate_map[scaffold_key]
            s_safe = sr.n_safe
            s_unsafe = sr.n_total - sr.n_safe

            # 2x2 contingency
            table = [[d_safe, d_unsafe], [s_safe, s_unsafe]]
            min_cell = min(d_safe, d_unsafe, s_safe, s_unsafe)

            try:
                if min_cell < 5:
                    # Fisher exact
                    from math import comb, factorial
                    # Simple Fisher exact for 2x2
                    a, b, c, d = d_safe, d_unsafe, s_safe, s_unsafe
                    n = a + b + c + d
                    row1 = a + b
                    row2 = c + d
                    col1 = a + c
                    col2 = b + d

                    # Hypergeometric p-value (two-sided)
                    def _hyper_p(a_val: int) -> float:
                        num = comb(col1, a_val) * comb(col2, row1 - a_val)
                        den = comb(n, row1)
                        return num / den if den > 0 else 0

                    p_obs = _hyper_p(a)
                    p_val = sum(
                        _hyper_p(x) for x in range(max(0, row1 - col2), min(row1, col1) + 1)
                        if _hyper_p(x) <= p_obs + 1e-10
                    )
                    results.append({
                        "benchmark_id": bid, "model_id": mid, "config_id": cid,
                        "test": "fisher", "statistic": None, "p_value": min(p_val, 1.0),
                    })
                else:
                    # Chi-squared with Yates correction
                    n = d_safe + d_unsafe + s_safe + s_unsafe
                    e_ds = (d_safe + d_unsafe) * (d_safe + s_safe) / n
                    e_du = (d_safe + d_unsafe) * (d_unsafe + s_unsafe) / n
                    e_ss = (s_safe + s_unsafe) * (d_safe + s_safe) / n
                    e_su = (s_safe + s_unsafe) * (d_unsafe + s_unsafe) / n

                    chi2 = 0.0
                    for obs, exp in [(d_safe, e_ds), (d_unsafe, e_du),
                                     (s_safe, e_ss), (s_unsafe, e_su)]:
                        if exp > 0:
                            chi2 += (abs(obs - exp) - 0.5) ** 2 / exp

                    # p-value from chi-squared distribution (1 df)
                    # Using incomplete gamma function approximation
                    from math import exp as mexp, sqrt as msqrt, pi
                    x = chi2 / 2
                    if x > 0:
                        # Approximation for chi-sq p-value with 1 df
                        p_val = math.erfc(msqrt(x))
                    else:
                        p_val = 1.0

                    import math
                    results.append({
                        "benchmark_id": bid, "model_id": mid, "config_id": cid,
                        "test": "chi2", "statistic": round(chi2, 4),
                        "p_value": min(p_val, 1.0),
                    })
            except Exception:
                continue

    return results
