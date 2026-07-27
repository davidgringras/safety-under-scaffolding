"""
Validation scorer: cross-validates primary judge with a gold-standard judge.

Computes Cohen's kappa inter-rater reliability on a stratified subsample.
This parallels clinical diagnostic validation methodology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ValidationReport:
    """Report from validation scoring."""
    n_total: int
    n_agree: int
    agreement_rate: float
    cohen_kappa: float
    primary_safe_rate: float
    validation_safe_rate: float
    confusion_matrix: dict[str, int]


class ValidationScorer:
    """Cross-validate primary and validation judge scores."""

    @staticmethod
    def compute_cohen_kappa(
        primary_labels: list[bool],
        validation_labels: list[bool],
    ) -> float:
        """Compute Cohen's kappa for binary classification agreement.

        Parameters
        ----------
        primary_labels : list[bool]
            Binary labels from primary judge.
        validation_labels : list[bool]
            Binary labels from validation judge.

        Returns
        -------
        float
            Cohen's kappa coefficient.
        """
        n = len(primary_labels)
        if n == 0:
            return 0.0

        # Confusion matrix counts
        tp = sum(1 for p, v in zip(primary_labels, validation_labels) if p and v)
        tn = sum(1 for p, v in zip(primary_labels, validation_labels) if not p and not v)
        fp = sum(1 for p, v in zip(primary_labels, validation_labels) if p and not v)
        fn = sum(1 for p, v in zip(primary_labels, validation_labels) if not p and v)

        # Observed agreement
        po = (tp + tn) / n

        # Expected agreement by chance
        p_yes_primary = (tp + fp) / n
        p_yes_valid = (tp + fn) / n
        p_no_primary = (tn + fn) / n
        p_no_valid = (tn + fp) / n
        pe = p_yes_primary * p_yes_valid + p_no_primary * p_no_valid

        if pe >= 1.0:
            return 1.0 if po == 1.0 else 0.0

        kappa = (po - pe) / (1 - pe)
        return round(kappa, 4)

    @staticmethod
    def validate(
        primary_labels: list[bool],
        validation_labels: list[bool],
    ) -> ValidationReport:
        """Run full validation analysis.

        Parameters
        ----------
        primary_labels, validation_labels : list[bool]
            Paired binary safety labels from both judges.

        Returns
        -------
        ValidationReport
        """
        n = len(primary_labels)
        n_agree = sum(1 for p, v in zip(primary_labels, validation_labels) if p == v)
        agreement_rate = n_agree / n if n > 0 else 0.0
        kappa = ValidationScorer.compute_cohen_kappa(primary_labels, validation_labels)

        tp = sum(1 for p, v in zip(primary_labels, validation_labels) if p and v)
        tn = sum(1 for p, v in zip(primary_labels, validation_labels) if not p and not v)
        fp = sum(1 for p, v in zip(primary_labels, validation_labels) if p and not v)
        fn = sum(1 for p, v in zip(primary_labels, validation_labels) if not p and v)

        return ValidationReport(
            n_total=n,
            n_agree=n_agree,
            agreement_rate=round(agreement_rate, 4),
            cohen_kappa=kappa,
            primary_safe_rate=round(sum(primary_labels) / n, 4) if n > 0 else 0.0,
            validation_safe_rate=round(sum(validation_labels) / n, 4) if n > 0 else 0.0,
            confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        )
