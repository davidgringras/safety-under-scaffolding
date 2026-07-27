"""
Abstract base class for all scaffold configurations.

Every scaffold must implement run() with identical I/O:
  - Input:  a system prompt, a user prompt, a ModelSpec, and optional kwargs
  - Output: a ScaffoldResult dataclass with the final response text,
            full trace of intermediate steps, and aggregated token/cost info.

This guarantees that the experiment runner treats every scaffold identically.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from pipeline.config import ModelSpec


@dataclass
class ScaffoldResult:
    """Standardised output from any scaffold configuration.

    Attributes
    ----------
    final_response : str
        The text that will be scored / judged.  For MC benchmarks this is
        the raw model output from which the answer letter is parsed.
    trace : list[dict]
        Full ordered trace of every intermediate step.  Each dict has at
        minimum {"role": str, "content": str}.  For Direct, the trace is a
        single item; for ReAct it includes thought/action/observation steps;
        for Multi-Agent it includes critic turns; for Map-Reduce it includes
        chunk-level responses and the aggregation step.
    input_tokens : int
        Total input tokens across all API calls in this scaffold invocation.
    output_tokens : int
        Total output tokens across all API calls.
    total_cost_usd : float
        Total estimated cost across all API calls.
    n_api_calls : int
        Number of separate API calls made.
    call_ids : list[str]
        UUIDs of each API call (for audit trail linking to api_calls.jsonl).
    metadata : dict
        Any scaffold-specific extra info (e.g. number of ReAct iterations,
        critic verdict in multi-agent, chunk count in map-reduce).
    """
    final_response: str
    trace: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    n_api_calls: int = 0
    call_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseScaffold(abc.ABC):
    """Abstract base class for scaffold implementations.

    Subclasses must implement ``run()`` and ``scaffold_id``.
    """

    @property
    @abc.abstractmethod
    def scaffold_id(self) -> str:
        """Short identifier: 'direct', 'react', 'multi_agent', 'map_reduce'."""
        ...

    @abc.abstractmethod
    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: ModelSpec,
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ScaffoldResult:
        """Execute the scaffold and return a standardised result.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction (identical across scaffolds for a
            given benchmark).
        user_prompt : str
            The benchmark case prompt.
        model : ModelSpec
            Which model to call.
        dry_run : bool
            If True, no actual API calls are made.
        metadata : dict
            Extra metadata to attach to API call logs.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} scaffold_id={self.scaffold_id!r}>"
