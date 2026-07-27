"""
Abstract base class for all scaffold configurations.

Every scaffold must implement _run_impl() with identical I/O:
  - Input:  a system prompt, a user prompt, a ModelSpec, and optional kwargs
  - Output: a ScaffoldResult dataclass with the final response text,
            full trace of intermediate steps, and aggregated token/cost info.

This guarantees that the experiment runner treats every scaffold identically.

The base class also provides:
  - Input validation (non-empty prompts, valid model specs)
  - Token budget enforcement (optional ceiling on total tokens per run)
  - Structured logging of scaffold execution via Python logging
  - A registry pattern: subclasses auto-register via __init_subclass__
  - Metadata forwarding to call_model() for audit trail linking
  - Wall-clock timing on every scaffold run

Usage
-----
    from scaffold_safety.configs import SCAFFOLD_REGISTRY

    scaffold = SCAFFOLD_REGISTRY["react"](max_iterations=5)
    result = scaffold.run(system_prompt, user_prompt, model)

Custom scaffolds::

    class MyScaffold(BaseScaffold):
        scaffold_id = "my_custom"

        def _run_impl(self, system_prompt, user_prompt, model, **kw):
            ...

    # Automatically available via get_registry()["my_custom"]
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from scaffold_safety.utils.providers import ModelSpec, call_model

logger = logging.getLogger("scaffold_safety.configs")


# ---------------------------------------------------------------------------
# Global scaffold registry -- populated by __init_subclass__
# ---------------------------------------------------------------------------

_SCAFFOLD_REGISTRY: dict[str, type["BaseScaffold"]] = {}


def get_registry() -> dict[str, type["BaseScaffold"]]:
    """Return a copy of the auto-populated scaffold registry.

    Every concrete subclass of ``BaseScaffold`` that defines a non-empty
    ``scaffold_id`` class attribute is automatically registered here.

    Returns
    -------
    dict[str, type[BaseScaffold]]
        Mapping from scaffold_id string to scaffold class.

    Example
    -------
    >>> from scaffold_safety.configs.base import get_registry
    >>> registry = get_registry()
    >>> scaffold = registry["react"](max_iterations=3)
    """
    return dict(_SCAFFOLD_REGISTRY)


# ---------------------------------------------------------------------------
# ScaffoldResult
# ---------------------------------------------------------------------------

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
    wall_time_seconds : float
        Wall-clock time for the entire scaffold run.
    budget_remaining_tokens : int | None
        Remaining token budget after this run, or None if no budget was set.
    """

    final_response: str
    trace: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    n_api_calls: int = 0
    call_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    budget_remaining_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all API calls."""
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ScaffoldValidationError(ValueError):
    """Raised when scaffold input validation fails."""


class TokenBudgetExceededError(RuntimeError):
    """Raised when a scaffold run exceeds its token budget mid-execution."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseScaffold(abc.ABC):
    """Abstract base class for scaffold implementations.

    Subclasses must:
      1. Define ``scaffold_id`` as a class-level string *or* override the
         property.
      2. Implement ``_run_impl()`` with the scaffold-specific logic.

    The public ``run()`` method wraps ``_run_impl()`` with:
      - Input validation via ``_validate_inputs()``
      - Token budget enforcement via ``_check_token_budget()``
      - Wall-clock timing
      - Structured logging (Python ``logging`` module)
      - Automatic metadata augmentation

    Parameters
    ----------
    token_budget : int | None
        Maximum total tokens (input + output) allowed for the entire scaffold
        run.  None means unlimited.

    Example
    -------
    >>> scaffold = DirectScaffold(token_budget=50000)
    >>> result = scaffold.run("Be helpful.", "What is 2+2?", model)
    >>> print(result.total_tokens, result.budget_remaining_tokens)
    """

    # Subclasses set this as a class attribute; __init_subclass__ auto-registers.
    scaffold_id: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        sid = getattr(cls, "scaffold_id", "")
        if sid and not getattr(cls, "__abstractmethods__", None):
            _SCAFFOLD_REGISTRY[sid] = cls
            logger.debug("Registered scaffold: %s -> %s", sid, cls.__name__)

    def __init__(self, *, token_budget: int | None = None) -> None:
        if token_budget is not None and token_budget <= 0:
            raise ScaffoldValidationError(
                f"token_budget must be a positive integer, got {token_budget}"
            )
        self._token_budget = token_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: ModelSpec,
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ScaffoldResult:
        """Execute the scaffold and return a standardised result.

        This is the public entry point.  It validates inputs, delegates to
        ``_run_impl()``, enforces the token budget, records wall-clock time,
        and logs the execution summary.

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
            If True, no actual API calls are made (placeholder results).
        metadata : dict
            Extra metadata to attach to API call logs.
        **kwargs
            Additional keyword arguments forwarded to ``_run_impl()``.

        Returns
        -------
        ScaffoldResult

        Raises
        ------
        ScaffoldValidationError
            If inputs fail validation.
        TokenBudgetExceededError
            If the run exceeds the configured token budget.
        """
        self._validate_inputs(system_prompt, user_prompt, model)

        run_metadata = {
            **(metadata or {}),
            "scaffold": self.scaffold_id,
        }

        logger.info(
            "Starting scaffold run: %s model=%s dry_run=%s",
            self.scaffold_id, model.model_id, dry_run,
        )

        t0 = time.monotonic()
        result = self._run_impl(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            dry_run=dry_run,
            metadata=run_metadata,
            **kwargs,
        )
        wall_time = time.monotonic() - t0

        # Augment result with timing and budget info
        result.wall_time_seconds = round(wall_time, 3)

        if self._token_budget is not None:
            remaining = self._token_budget - result.total_tokens
            result.budget_remaining_tokens = max(remaining, 0)
            if remaining < 0:
                logger.warning(
                    "Token budget exceeded: budget=%d, used=%d, over=%d",
                    self._token_budget, result.total_tokens, abs(remaining),
                )

        if "scaffold" not in result.metadata:
            result.metadata["scaffold"] = self.scaffold_id

        logger.info(
            "Scaffold %s complete: api_calls=%d, tokens=%d (in=%d, out=%d), "
            "cost=$%.6f, wall_time=%.3fs",
            self.scaffold_id,
            result.n_api_calls,
            result.total_tokens,
            result.input_tokens,
            result.output_tokens,
            result.total_cost_usd,
            result.wall_time_seconds,
        )

        return result

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _run_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        model: ModelSpec,
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ScaffoldResult:
        """Scaffold-specific implementation.  Override in subclasses.

        The ``metadata`` dict already includes ``{"scaffold": scaffold_id}``
        and any user-supplied metadata.  Forward it to every
        ``_call_model()`` invocation for audit trail linking.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _call_model(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
        step_name: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convenience wrapper around ``providers.call_model`` with metadata.

        Automatically adds scaffold_id and step_name to the metadata dict
        so every API call is traceable in the audit log.

        Parameters
        ----------
        model : ModelSpec
            Which model to call.
        messages : list[dict]
            OpenAI-format messages.
        dry_run : bool
            If True, return placeholder without hitting the API.
        metadata : dict
            Base metadata dict (will be augmented with step info).
        step_name : str
            Human-readable step identifier (e.g. "react_iter_0",
            "critic_round_0", "map_chunk_2").
        **kwargs
            Forwarded to ``call_model()`` (temperature, max_tokens, etc.).

        Returns
        -------
        dict
            Keys: content, input_tokens, output_tokens, cost_usd, call_id,
            latency_seconds.
        """
        call_meta = {
            **(metadata or {}),
            "scaffold": self.scaffold_id,
        }
        if step_name:
            call_meta["step"] = step_name

        logger.debug(
            "Scaffold %s calling model %s step=%s",
            self.scaffold_id, model.model_id, step_name or "(none)",
        )

        return call_model(
            model=model,
            messages=messages,
            dry_run=dry_run,
            **kwargs,
        )

    def _check_token_budget(
        self, current_total: int, *, context: str = ""
    ) -> None:
        """Raise ``TokenBudgetExceededError`` if the budget has been exceeded.

        Call this between API calls in multi-step scaffolds to fail early
        rather than wasting tokens on subsequent calls.

        Parameters
        ----------
        current_total : int
            Total tokens consumed so far (input + output).
        context : str
            Optional context string for the error message (e.g. "after
            iteration 3").
        """
        if self._token_budget is not None and current_total > self._token_budget:
            raise TokenBudgetExceededError(
                f"Token budget exceeded in scaffold '{self.scaffold_id}'"
                f"{f' at {context}' if context else ''}: "
                f"budget={self._token_budget}, used={current_total}"
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        system_prompt: str,
        user_prompt: str,
        model: ModelSpec,
    ) -> None:
        """Validate scaffold inputs.  Override to add scaffold-specific checks.

        Raises
        ------
        ScaffoldValidationError
            If any input is invalid.
        """
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ScaffoldValidationError(
                "system_prompt must be a non-empty string"
            )
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ScaffoldValidationError(
                "user_prompt must be a non-empty string"
            )
        if not isinstance(model, ModelSpec):
            raise ScaffoldValidationError(
                f"model must be a ModelSpec instance, got {type(model).__name__}"
            )
        if not model.litellm_id:
            raise ScaffoldValidationError(
                "model.litellm_id must be a non-empty string"
            )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        budget_str = f", budget={self._token_budget}" if self._token_budget else ""
        return f"<{self.__class__.__name__} scaffold_id={self.scaffold_id!r}{budget_str}>"
