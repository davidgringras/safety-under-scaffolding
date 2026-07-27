"""
Unit tests for validation scorer (Cohen's kappa computation).
"""

import pytest
from scaffold_safety.scoring.validation import ValidationScorer


class TestCohenKappa:
    def test_perfect_agreement(self):
        primary = [True, True, False, False, True]
        validation = [True, True, False, False, True]
        kappa = ValidationScorer.compute_cohen_kappa(primary, validation)
        assert kappa == 1.0

    def test_no_agreement(self):
        primary = [True, True, True, True]
        validation = [False, False, False, False]
        kappa = ValidationScorer.compute_cohen_kappa(primary, validation)
        assert kappa <= 0.0

    def test_moderate_agreement(self):
        primary = [True, True, False, False, True, False, True, True]
        validation = [True, True, False, True, True, False, False, True]
        kappa = ValidationScorer.compute_cohen_kappa(primary, validation)
        assert 0.0 < kappa < 1.0

    def test_empty_labels(self):
        kappa = ValidationScorer.compute_cohen_kappa([], [])
        assert kappa == 0.0


class TestValidationReport:
    def test_validate(self):
        primary = [True, True, False, False, True]
        validation = [True, False, False, False, True]
        report = ValidationScorer.validate(primary, validation)
        assert report.n_total == 5
        assert report.n_agree == 4
        assert report.agreement_rate == 0.8
        assert report.confusion_matrix["tp"] == 2
        assert report.confusion_matrix["tn"] == 2
        assert report.confusion_matrix["fp"] == 1
        assert report.confusion_matrix["fn"] == 0
