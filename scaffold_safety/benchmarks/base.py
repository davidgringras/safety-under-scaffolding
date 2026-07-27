"""
Base classes for benchmark loading and scoring.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable specification for a benchmark dataset.

    Attributes
    ----------
    benchmark_id : str
        Short identifier (e.g. "truthfulqa").
    display_name : str
        Human-readable name.
    n_cases : int
        Number of cases in the standard dataset.
    scoring_method : str
        "automated" | "judge" | "hybrid".
    safety_property : str
        Which safety dimension this tests (sycophancy, bias, over_refusal, truthfulness).
    """
    benchmark_id: str
    display_name: str
    n_cases: int
    scoring_method: str
    safety_property: str


@dataclass
class BenchmarkCase:
    """A single benchmark case."""
    case_id: str
    prompt: str
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """Result of scoring a single response."""
    case_id: str
    benchmark_id: str
    is_safe: bool
    is_correct: bool | None = None
    predicted: str | None = None
    correct: str | None = None
    confidence: str = "high"
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkLoader(abc.ABC):
    """Abstract base for benchmark loading and scoring."""

    @property
    @abc.abstractmethod
    def spec(self) -> BenchmarkSpec:
        """Return the benchmark specification."""
        ...

    @abc.abstractmethod
    def load_cases(self, data_path: Path) -> list[BenchmarkCase]:
        """Load benchmark cases from a data file.

        Parameters
        ----------
        data_path : Path
            Path to the benchmark data file (JSONL format).

        Returns
        -------
        list[BenchmarkCase]
        """
        ...

    @abc.abstractmethod
    def score(self, case: BenchmarkCase, response: str) -> ScoreResult:
        """Score a model response against a benchmark case.

        Parameters
        ----------
        case : BenchmarkCase
            The original benchmark case.
        response : str
            The model's response text.

        Returns
        -------
        ScoreResult
        """
        ...

    def get_system_prompt(self) -> str:
        """Return the system prompt for this benchmark.

        Override in subclasses for benchmark-specific system prompts.
        """
        return "You are a helpful, harmless, and honest AI assistant."

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Load a JSONL file into a list of dicts."""
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
