"""
Unified provider abstraction using litellm.

Wraps litellm.completion() with:
- Exponential-backoff retry logic
- Per-call token counting and cost tracking
- Dry-run mode (returns placeholder, no actual API call)

Supports any model accessible through LiteLLM's unified API,
including OpenAI, Anthropic, Google, Together AI, DeepSeek,
and 100+ other providers.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    litellm.drop_params = True
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a model under test or judge.

    Attributes
    ----------
    model_id : str
        Short internal identifier (e.g. "opus", "gpt52").
    litellm_id : str
        String passed to litellm.completion() (e.g. "claude-opus-4-6").
    provider : str
        Provider name (anthropic, openai, google, together, deepseek, ...).
    supports_batch : bool
        Whether the model supports batch API.
    supports_temperature : bool
        False for reasoning models (e.g. GPT-5.2) that reject temperature.
    cost_per_1k_input : float
        USD per 1K input tokens.
    cost_per_1k_output : float
        USD per 1K output tokens.
    notes : str
        Human-readable description.
    """
    model_id: str
    litellm_id: str
    provider: str
    supports_batch: bool = False
    supports_temperature: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    notes: str = ""


# Pre-defined model registry for convenience
KNOWN_MODELS: dict[str, ModelSpec] = {
    "claude-opus-4-6": ModelSpec(
        model_id="opus", litellm_id="claude-opus-4-6",
        provider="anthropic", supports_batch=True,
        cost_per_1k_input=0.015, cost_per_1k_output=0.075,
    ),
    "gpt-5.2": ModelSpec(
        model_id="gpt52", litellm_id="gpt-5.2",
        provider="openai", supports_batch=True,
        supports_temperature=False,
        cost_per_1k_input=0.002, cost_per_1k_output=0.010,
    ),
    "gemini-3-pro": ModelSpec(
        model_id="gemini3pro", litellm_id="vertex_ai/gemini-3-pro-preview",
        provider="vertex_ai",
        cost_per_1k_input=0.002, cost_per_1k_output=0.012,
    ),
    "llama-4-maverick": ModelSpec(
        model_id="llama4",
        litellm_id="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="together",
        cost_per_1k_input=0.0002, cost_per_1k_output=0.0002,
    ),
    "deepseek-v3.2": ModelSpec(
        model_id="deepseek", litellm_id="deepseek/deepseek-chat",
        provider="deepseek",
        cost_per_1k_input=0.00014, cost_per_1k_output=0.00028,
    ),
}


def resolve_model(model_str: str) -> ModelSpec:
    """Resolve a model string to a ModelSpec.

    Accepts either a known model name or a raw litellm model string.
    """
    if model_str in KNOWN_MODELS:
        return KNOWN_MODELS[model_str]
    # Treat as a raw litellm string
    provider = model_str.split("/")[0] if "/" in model_str else "unknown"
    return ModelSpec(
        model_id=model_str.replace("/", "_"),
        litellm_id=model_str,
        provider=provider,
    )


@dataclass
class CallResult:
    """Result of a single model call."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    call_id: str = ""
    latency_seconds: float = 0.0


def call_model(
    model: ModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int = 42,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 60.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Call a model through LiteLLM with retry and cost tracking.

    Parameters
    ----------
    model : ModelSpec
        Which model to call.
    messages : list[dict]
        OpenAI-format messages.
    temperature, max_tokens, seed : inference params
    max_retries : int
        Number of retry attempts on failure.
    dry_run : bool
        If True, return placeholder without hitting the API.

    Returns
    -------
    dict
        Keys: content, input_tokens, output_tokens, cost_usd, call_id,
        latency_seconds.
    """
    call_id = str(uuid.uuid4())

    if dry_run:
        return {
            "content": "[DRY RUN] No API call made.",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "call_id": call_id,
            "latency_seconds": 0.0,
        }

    if not HAS_LITELLM:
        raise ImportError(
            "litellm is required for API calls. Install with: pip install litellm"
        )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "model": model.litellm_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "seed": seed,
            }
            if model.supports_temperature:
                kwargs["temperature"] = temperature

            response = litellm.completion(**kwargs)
            latency = time.monotonic() - t0

            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0

            cost_usd = (
                (input_tokens / 1000) * model.cost_per_1k_input
                + (output_tokens / 1000) * model.cost_per_1k_output
            )

            content = response.choices[0].message.content or ""

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "call_id": call_id,
                "latency_seconds": round(latency, 3),
            }

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = min(
                    retry_base_delay * (2 ** attempt) + random.uniform(0, 1),
                    retry_max_delay,
                )
                time.sleep(delay)

    raise RuntimeError(
        f"API call to {model.litellm_id} failed after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
