"""
Unit tests for the statistical analysis module.

Tests effect size calculations, TOST equivalence testing,
Cohen's kappa, and specification curve generation.
"""

import numpy as np
import pytest

from pipeline.analysis import (
    cohens_h,
    cohens_kappa,
    generate_specifications,
    number_needed_to_harm,
    tost_proportions,
)


class TestCohensH:
    def test_identical_proportions(self):
        h = cohens_h(0.5, 0.5)
        assert abs(h) < 1e-10

    def test_direction(self):
        # Higher p1 should give positive h
        h = cohens_h(0.8, 0.5)
        assert h > 0

    def test_known_value(self):
        # Cohen's h for 0.5 vs 0.3 = 2*arcsin(sqrt(0.5)) - 2*arcsin(sqrt(0.3))
        h = cohens_h(0.5, 0.3)
        expected = 2 * np.arcsin(np.sqrt(0.5)) - 2 * np.arcsin(np.sqrt(0.3))
        assert abs(h - expected) < 1e-10

    def test_extremes(self):
        h = cohens_h(1.0, 0.0)
        assert h > 0
        assert abs(h - np.pi) < 1e-10  # max possible h


class TestNNH:
    def test_worse_treatment(self):
        # Baseline 10% unsafe, treatment 20% unsafe
        nnh = number_needed_to_harm(0.10, 0.20)
        assert nnh is not None
        assert abs(nnh - 10.0) < 1e-10

    def test_same_rate(self):
        nnh = number_needed_to_harm(0.1, 0.1)
        assert nnh is None

    def test_better_treatment(self):
        nnh = number_needed_to_harm(0.2, 0.1)
        assert nnh is None

    def test_large_difference(self):
        nnh = number_needed_to_harm(0.0, 1.0)
        assert nnh is not None
        assert abs(nnh - 1.0) < 1e-10


class TestTOST:
    def test_clearly_equivalent(self):
        # 800/1000 vs 805/1000 — very close proportions, large sample
        result = tost_proportions(800, 1000, 805, 1000, margin=0.05)
        assert bool(result.is_equivalent) is True
        assert result.difference == pytest.approx(-0.005, abs=0.001)

    def test_clearly_different(self):
        # 80/100 vs 50/100 — very different
        result = tost_proportions(80, 100, 50, 100, margin=0.02)
        assert bool(result.is_equivalent) is False

    def test_margin_sensitivity(self):
        # Same data, wider margin should be easier to declare equivalent
        result_narrow = tost_proportions(80, 100, 75, 100, margin=0.02)
        result_wide = tost_proportions(80, 100, 75, 100, margin=0.10)
        # Wide margin is at least as likely to declare equivalence
        assert result_wide.p_tost <= result_narrow.p_tost or result_wide.is_equivalent

    def test_config_labels(self):
        result = tost_proportions(80, 100, 79, 100,
                                   config_a="direct", config_b="react")
        assert result.config_a == "direct"
        assert result.config_b == "react"


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels_a = ["A", "B", "C", "A", "B"]
        labels_b = ["A", "B", "C", "A", "B"]
        kappa = cohens_kappa(labels_a, labels_b)
        assert abs(kappa - 1.0) < 1e-10

    def test_no_agreement(self):
        # Systematically disagreeing
        labels_a = ["A", "A", "A", "A"]
        labels_b = ["B", "B", "B", "B"]
        kappa = cohens_kappa(labels_a, labels_b)
        assert kappa < 0.01

    def test_partial_agreement(self):
        labels_a = ["A", "B", "A", "B", "A"]
        labels_b = ["A", "B", "B", "A", "A"]
        kappa = cohens_kappa(labels_a, labels_b)
        assert 0 < kappa < 1

    def test_empty(self):
        kappa = cohens_kappa([], [])
        assert kappa == 0.0


class TestSpecificationCurve:
    def test_generates_specifications(self):
        specs = generate_specifications()
        assert len(specs) > 50  # should be ~1000+
        assert all(s.spec_id is not None for s in specs)
        assert all(isinstance(s.choices, dict) for s in specs)

    def test_specs_are_unique(self):
        specs = generate_specifications()
        descriptions = [s.description for s in specs]
        assert len(descriptions) == len(set(descriptions)), "All specs should be unique"

    def test_all_specs_have_required_keys(self):
        specs = generate_specifications()
        required_keys = {"model_subset", "benchmark_subset", "reference_config"}
        for s in specs:
            assert required_keys.issubset(s.choices.keys())
