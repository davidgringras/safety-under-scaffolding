"""
Unit tests for statistical analysis functions.

Tests safety rate computation, effect sizes, TOST, and scorecard generation.
"""

import pytest
from scaffold_safety.analysis.statistics import (
    SafetyRate,
    compute_safety_rates,
    compute_effect_sizes,
    compute_nnh,
    run_tost,
)
from scaffold_safety.analysis.scorecard import compute_sri, generate_scorecard


# Sample data for testing
SAMPLE_RESULTS = [
    {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "direct", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "direct", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "direct", "is_safe": False},
    {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "react", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "react", "is_safe": False},
    {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "react", "is_safe": False},
    {"benchmark_id": "truthfulqa", "model_id": "deepseek", "config_id": "direct", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "deepseek", "config_id": "direct", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "deepseek", "config_id": "direct", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "deepseek", "config_id": "react", "is_safe": True},
    {"benchmark_id": "truthfulqa", "model_id": "deepseek", "config_id": "react", "is_safe": False},
    {"benchmark_id": "truthfulqa", "model_id": "deepseek", "config_id": "react", "is_safe": False},
]


class TestComputeSafetyRates:
    def test_basic(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        assert len(rates) == 4  # 2 models x 2 configs

    def test_rates_correct(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        rate_map = {(r.model_id, r.config_id): r for r in rates}

        # opus direct: 2/3 safe
        assert abs(rate_map[("opus", "direct")].safety_rate - 2/3) < 0.001
        # opus react: 1/3 safe
        assert abs(rate_map[("opus", "react")].safety_rate - 1/3) < 0.001
        # deepseek direct: 3/3 safe
        assert abs(rate_map[("deepseek", "direct")].safety_rate - 1.0) < 0.001

    def test_skip_none(self):
        results = SAMPLE_RESULTS + [
            {"benchmark_id": "truthfulqa", "model_id": "opus", "config_id": "direct", "is_safe": None},
        ]
        rates = compute_safety_rates(results)
        rate_map = {(r.model_id, r.config_id): r for r in rates}
        assert rate_map[("opus", "direct")].n_total == 3  # None skipped


class TestComputeEffectSizes:
    def test_basic(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        effects = compute_effect_sizes(rates)
        assert len(effects) >= 1

    def test_nnh(self):
        nnh = compute_nnh(0.80, 0.60)
        assert abs(nnh - 5.0) < 0.01

    def test_nnh_no_difference(self):
        nnh = compute_nnh(0.80, 0.80)
        assert nnh == float("inf")


class TestTOST:
    def test_basic(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        tost_results = run_tost(rates, margin=0.10)
        assert len(tost_results) >= 1

    def test_equivalence_wide_margin(self):
        # With a very wide margin, should find equivalence
        rates = [
            SafetyRate("b", "m", "direct", 100, 80, 0.80, 0.04),
            SafetyRate("b", "m", "react", 100, 78, 0.78, 0.04),
        ]
        tost_results = run_tost(rates, margin=0.50)
        if tost_results:
            assert tost_results[0].equivalent is True


class TestSRI:
    def test_perfect_robustness(self):
        rates = [
            SafetyRate("b1", "m1", "direct", 100, 80, 0.80, 0.04),
            SafetyRate("b1", "m1", "react", 100, 80, 0.80, 0.04),
            SafetyRate("b1", "m1", "multi_agent", 100, 80, 0.80, 0.04),
        ]
        sri = compute_sri(rates, "m1")
        assert sri == 100.0

    def test_total_degradation(self):
        rates = [
            SafetyRate("b1", "m1", "direct", 100, 80, 0.80, 0.04),
            SafetyRate("b1", "m1", "react", 100, 30, 0.30, 0.04),
        ]
        sri = compute_sri(rates, "m1")
        assert sri < 50  # significant degradation


class TestScorecard:
    def test_generate(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        scorecard = generate_scorecard(rates, "opus")
        assert scorecard.model_id == "opus"
        assert 0 <= scorecard.sri <= 100
        assert scorecard.sri_category in ("robust", "moderately_sensitive", "scaffold_sensitive")
        assert len(scorecard.safety_matrix) > 0

    def test_text_output(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        scorecard = generate_scorecard(rates, "opus")
        text = scorecard.to_text()
        assert "SCAFFOLD SAFETY SCORECARD" in text
        assert "opus" in text

    def test_dict_output(self):
        rates = compute_safety_rates(SAMPLE_RESULTS)
        scorecard = generate_scorecard(rates, "opus")
        d = scorecard.to_dict()
        assert "scaffold_robustness_index" in d
        assert "sri_category" in d
        assert "safety_matrix" in d
