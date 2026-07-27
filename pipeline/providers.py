"""
Unified provider abstraction using litellm.

Wraps litellm.completion() with:
- Exponential-backoff retry logic
- Per-call token counting and cost tracking
- Structured JSON logging of every API call to logs/api_calls.jsonl
- Async support via litellm.acompletion()
- Dry-run mode (logs the call, returns a placeholder, no actual API hit)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import litellm

from pipeline.config import (
    API_LOG_FILE,
    EXPERIMENT_PARAMS,
    LOGS_DIR,
    ModelSpec,
    sha256_bytes,
)

# Suppress litellm's own verbose logging — we log everything ourselves.
litellm.suppress_debug_info = True
litellm.set_verbose = False
# Drop params unsupported by specific providers (e.g. Anthropic doesn't support seed)
litellm.drop_params = True


# ---------------------------------------------------------------------------
# Per-provider rate limiter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multi-key rotation for providers with multiple API keys
# ---------------------------------------------------------------------------

def _load_google_keys() -> list[str]:
    """Load all GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ... from env.

    GOOGLE_API_KEY_2 is placed first in rotation because it carries free
    credits.  This ensures the free-credit key is consumed preferentially
    under round-robin rotation.

    Also loads VERTEX_API_KEY if present — prefixed with "vertex:" so
    call_model() can detect it and route through Vertex AI instead of
    AI Studio.  Vertex AI has separate (much higher) quotas.
    """
    import os
    primary = os.getenv("GOOGLE_API_KEY")
    extras: list[str] = []
    priority_key: str | None = None
    for i in range(2, 20):
        k = os.getenv(f"GOOGLE_API_KEY_{i}")
        if k:
            if i == 2:
                priority_key = k  # free-credit key → goes first
            else:
                extras.append(k)
        else:
            break
    # Build ordered list: priority key first, then primary, then extras
    keys: list[str] = []
    if priority_key:
        keys.append(priority_key)
    if primary:
        keys.append(primary)
    keys.extend(extras)
    # Append Vertex AI service account (separate quota pool from AI Studio)
    # Activated when GOOGLE_APPLICATION_CREDENTIALS points to a service account JSON
    vertex_sa = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if vertex_sa and os.path.isfile(vertex_sa):
        keys.append("vertex:service_account")
    return keys


class _MultiKeyRotator:
    """Round-robin API key rotation for providers with multiple keys.

    Includes a per-key daily-quota circuit breaker: once a key hits a
    daily-quota error, it is marked as exhausted until midnight Pacific,
    preventing thousands of wasted API calls.
    """

    def __init__(self) -> None:
        self._keys: dict[str, list[str]] = {
            "google": _load_google_keys(),
        }
        self._counters: dict[str, int] = {p: 0 for p in self._keys}
        self._lock = threading.Lock()
        # Daily-quota circuit breaker: key → UTC timestamp when it was tripped
        self._exhausted_keys: dict[str, float] = {}

    def _midnight_pt_utc(self) -> float:
        """Return the next midnight Pacific as a UTC epoch timestamp."""
        import calendar
        from datetime import datetime as _dt, timedelta, timezone as _tz
        try:
            from zoneinfo import ZoneInfo
            pt = ZoneInfo("America/Los_Angeles")
        except ImportError:
            # Fallback: assume PT = UTC-8
            pt = _tz(timedelta(hours=-8))
        now_pt = _dt.now(pt)
        midnight_pt = (now_pt + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return midnight_pt.timestamp()

    def mark_key_exhausted(self, provider: str, key: str) -> None:
        """Mark a key as daily-quota exhausted (skip until midnight PT)."""
        with self._lock:
            compound = f"{provider}:{key}"
            self._exhausted_keys[compound] = time.time()

    def _is_key_exhausted(self, provider: str, key: str) -> bool:
        """Check if a key is currently in the exhausted state."""
        compound = f"{provider}:{key}"
        tripped_at = self._exhausted_keys.get(compound)
        if tripped_at is None:
            return False
        # Clear if past midnight PT
        if time.time() >= self._midnight_pt_utc():
            del self._exhausted_keys[compound]
            return False
        return True

    def get_api_key(self, provider: str) -> str | None:
        """Return the next API key for this provider, or None if not multi-key.

        Raises RuntimeError if ALL keys for the provider are daily-quota
        exhausted — this prevents thousands of wasted API calls.
        """
        if provider not in self._keys or not self._keys[provider]:
            return None
        with self._lock:
            keys = self._keys[provider]
            # Check if ALL keys are exhausted
            all_exhausted = all(
                self._is_key_exhausted(provider, k) for k in keys
            )
            if all_exhausted:
                raise RuntimeError(
                    f"All {len(keys)} API keys for provider '{provider}' have "
                    f"hit daily quota limits. Skipping until midnight Pacific."
                )
            # Round-robin but skip exhausted keys
            start_idx = self._counters[provider] % len(keys)
            for offset in range(len(keys)):
                idx = (start_idx + offset) % len(keys)
                if not self._is_key_exhausted(provider, keys[idx]):
                    self._counters[provider] = idx + 1
                    return keys[idx]
            # Shouldn't reach here if all_exhausted check passed
            return keys[start_idx]

    def n_keys(self, provider: str) -> int:
        return len(self._keys.get(provider, []))


_key_rotator = _MultiKeyRotator()


# RPM limits per provider KEY (conservative to stay under quotas)
_BASE_RPM_LIMITS: dict[str, int] = {
    "google": 25,      # Gemini 3 Pro: 25 RPM per key confirmed
    "vertex_ai": 200,  # Vertex AI: generous RPM, push hard
    "anthropic": 50,
    "openai": 60,
    "together": 60,
    "deepseek": 60,
    "openrouter": 50,  # OpenRouter: generous limits, use 50 RPM
    "mistral": 60,     # Mistral: generous limits
}


class _ProviderRateLimiter:
    """Simple sliding-window rate limiter, per provider-key combination."""

    def __init__(self) -> None:
        self._timestamps: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def wait_if_needed(self, provider: str, key_index: int = 0) -> None:
        # Rate limit per key, not per provider
        limiter_id = f"{provider}:{key_index}"
        rpm_limit = _BASE_RPM_LIMITS.get(provider, 60)
        window = 60.0

        with self._lock:
            now = time.monotonic()
            if limiter_id not in self._timestamps:
                self._timestamps[limiter_id] = []

            cutoff = now - window
            self._timestamps[limiter_id] = [
                t for t in self._timestamps[limiter_id] if t > cutoff
            ]

            if len(self._timestamps[limiter_id]) >= rpm_limit:
                sleep_time = self._timestamps[limiter_id][0] - cutoff + 0.1
                if sleep_time > 0:
                    self._lock.release()
                    time.sleep(sleep_time)
                    self._lock.acquire()
                    now = time.monotonic()
                    cutoff = now - window
                    self._timestamps[limiter_id] = [
                        t for t in self._timestamps[limiter_id] if t > cutoff
                    ]

            self._timestamps[limiter_id].append(time.monotonic())


_rate_limiter = _ProviderRateLimiter()


# ---------------------------------------------------------------------------
# Data classes for API call records
# ---------------------------------------------------------------------------

@dataclass
class APICallRecord:
    """Immutable record of a single API call — written to the audit log."""
    call_id: str
    timestamp_utc: str
    model_id: str
    litellm_id: str
    provider: str
    messages_hash: str          # SHA-256 of serialised messages (for traceability)
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    status: str                 # "success" | "error" | "dry_run"
    error_message: str | None = None
    response_id: str | None = None   # provider-assigned ID if available
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "timestamp_utc": self.timestamp_utc,
            "model_id": self.model_id,
            "litellm_id": self.litellm_id,
            "provider": self.provider,
            "messages_hash": self.messages_hash,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_seconds": self.latency_seconds,
            "status": self.status,
            "error_message": self.error_message,
            "response_id": self.response_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Cost tracker (process-wide singleton)
# ---------------------------------------------------------------------------

class CostTracker:
    """Thread-safe running total of API spend, queryable per model."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}      # model_id → cumulative USD
        self._token_totals: dict[str, dict[str, int]] = {}  # model_id → {input, output}
        self._call_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, model_id: str, cost_usd: float,
               input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._totals[model_id] = self._totals.get(model_id, 0.0) + cost_usd
            if model_id not in self._token_totals:
                self._token_totals[model_id] = {"input": 0, "output": 0}
            self._token_totals[model_id]["input"] += input_tokens
            self._token_totals[model_id]["output"] += output_tokens
            self._call_counts[model_id] = self._call_counts.get(model_id, 0) + 1

    @property
    def total_usd(self) -> float:
        return sum(self._totals.values())

    def per_model(self) -> dict[str, float]:
        return dict(self._totals)

    def summary(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "per_model_usd": {k: round(v, 6) for k, v in self._totals.items()},
            "per_model_tokens": dict(self._token_totals),
            "per_model_calls": dict(self._call_counts),
        }


# Module-level singleton
cost_tracker = CostTracker()

# Thread-safe file logging lock
_log_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_api_call(record: APICallRecord) -> None:
    """Append a JSON line to the API call log file (thread-safe)."""
    API_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), default=str) + "\n"
    with _log_lock:
        with open(API_LOG_FILE, "a") as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Core completion wrapper
# ---------------------------------------------------------------------------

def call_model(
    model: ModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    dry_run: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronous model call with retry, logging, and cost tracking.

    Parameters
    ----------
    model : ModelSpec
        Which model to call (contains litellm_id, provider, cost rates).
    messages : list[dict]
        OpenAI-format messages list.
    temperature, max_tokens, seed, top_p :
        Override experiment defaults if needed.
    dry_run : bool
        If True, log the call but do NOT hit the API.  Returns a placeholder.
    metadata : dict
        Extra key-value pairs to attach to the log record (e.g. benchmark_id,
        case_id, scaffold_id).

    Returns
    -------
    dict with keys: "content", "input_tokens", "output_tokens", "cost_usd",
    "call_id", "latency_seconds".
    """
    temperature = temperature if temperature is not None else EXPERIMENT_PARAMS.temperature
    max_tokens = max_tokens if max_tokens is not None else EXPERIMENT_PARAMS.max_tokens
    seed = seed if seed is not None else EXPERIMENT_PARAMS.seed
    top_p = top_p if top_p is not None else EXPERIMENT_PARAMS.top_p

    call_id = str(uuid.uuid4())
    messages_hash = sha256_bytes(json.dumps(messages, sort_keys=True).encode())

    if dry_run:
        record = APICallRecord(
            call_id=call_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_id=model.model_id,
            litellm_id=model.litellm_id,
            provider=model.provider,
            messages_hash=messages_hash,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_seconds=0.0,
            status="dry_run",
            metadata=metadata or {},
        )
        _log_api_call(record)
        return {
            "content": "[DRY RUN] No API call made.",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "call_id": call_id,
            "latency_seconds": 0.0,
        }

    # Select API key (round-robin for multi-key providers like Google)
    rotated_key = None
    _use_vertex = model.provider == "vertex_ai"
    if not _use_vertex:
        rotated_key = _key_rotator.get_api_key(model.provider)
    # Determine which key index we're on for rate limiting
    _n_keys = max(_key_rotator.n_keys(model.provider), 1)
    _key_idx = (_key_rotator._counters.get(model.provider, 1) - 1) % _n_keys

    # Legacy: Detect Vertex AI marker (prefixed with "vertex:") from key rotator
    if rotated_key and rotated_key.startswith("vertex:"):
        _use_vertex = True
        rotated_key = None  # Vertex uses ADC from env, not API key

    last_error: Exception | None = None
    for attempt in range(EXPERIMENT_PARAMS.max_retries + 1):
        # Per-provider-key rate limiting (prevents 429s)
        _rate_limiter.wait_if_needed(model.provider, _key_idx)
        t0 = time.monotonic()
        try:
            # Build kwargs — conditionally omit params that some providers reject:
            # - Anthropic: rejects temperature + top_p together
            # - GPT-5.2 (reasoning): rejects temperature entirely
            _model_id = model.litellm_id
            if _use_vertex and model.provider == "google":
                # Route through Vertex AI instead of AI Studio
                _model_id = model.litellm_id.replace("gemini/", "vertex_ai/", 1)
            completion_kwargs: dict[str, Any] = {
                "model": _model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "seed": seed,
            }
            if model.supports_temperature:
                completion_kwargs["temperature"] = temperature
            if top_p is not None and top_p != 1.0:
                completion_kwargs["top_p"] = top_p
            if rotated_key:
                completion_kwargs["api_key"] = rotated_key
            if _use_vertex:
                completion_kwargs["vertex_project"] = os.getenv("VERTEXAI_PROJECT")
                completion_kwargs["vertex_location"] = os.getenv("VERTEXAI_LOCATION", "global")
            response = litellm.completion(**completion_kwargs)
            latency = time.monotonic() - t0

            # Extract usage
            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or (input_tokens + output_tokens)

            # Compute cost
            cost_usd = (
                (input_tokens / 1000) * model.cost_per_1k_input
                + (output_tokens / 1000) * model.cost_per_1k_output
            )

            content = response.choices[0].message.content or ""
            response_id = getattr(response, "id", None)

            # Track cost
            cost_tracker.record(model.model_id, cost_usd, input_tokens, output_tokens)

            # Log
            record = APICallRecord(
                call_id=call_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                model_id=model.model_id,
                litellm_id=model.litellm_id,
                provider=model.provider,
                messages_hash=messages_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_seconds=round(latency, 3),
                status="success",
                response_id=response_id,
                metadata=metadata or {},
            )
            _log_api_call(record)

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
            latency = time.monotonic() - t0

            # Log the failure
            record = APICallRecord(
                call_id=call_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                model_id=model.model_id,
                litellm_id=model.litellm_id,
                provider=model.provider,
                messages_hash=messages_hash,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=round(latency, 3),
                status="error",
                error_message=f"Attempt {attempt + 1}/{EXPERIMENT_PARAMS.max_retries + 1}: {e}",
                metadata=metadata or {},
            )
            _log_api_call(record)

            # Daily quota errors can't be solved by retrying — fail fast
            # Also mark the key as exhausted in the circuit breaker
            err_str = str(e).lower()
            if "per_model_per_day" in err_str or "per_day" in err_str:
                if rotated_key:
                    _key_rotator.mark_key_exhausted(model.provider, rotated_key)
                raise RuntimeError(
                    f"API call to {model.litellm_id} hit daily quota limit "
                    f"(no retry). Error: {e}"
                )

            # Credit exhaustion — also fail fast
            if "credit" in err_str and ("limit" in err_str or "exceeded" in err_str or "low" in err_str):
                raise RuntimeError(
                    f"API call to {model.litellm_id} hit credit limit "
                    f"(no retry). Error: {e}"
                )

            # Suspended / permission-denied key — mark exhausted and rotate
            if "consumer_suspended" in err_str or "consumer has been suspended" in err_str or (
                "permission_denied" in err_str and "suspended" in err_str
            ):
                if rotated_key:
                    _key_rotator.mark_key_exhausted(model.provider, rotated_key)
                # Try to get a fresh key for next attempt
                try:
                    rotated_key = _key_rotator.get_api_key(model.provider)
                except RuntimeError:
                    raise RuntimeError(
                        f"API call to {model.litellm_id}: key suspended and "
                        f"all remaining keys exhausted. Error: {e}"
                    )

            if attempt < EXPERIMENT_PARAMS.max_retries:
                delay = min(
                    EXPERIMENT_PARAMS.retry_base_delay * (2 ** attempt)
                    + random.uniform(0, 1),
                    EXPERIMENT_PARAMS.retry_max_delay,
                )
                time.sleep(delay)

    # All retries exhausted
    raise RuntimeError(
        f"API call to {model.litellm_id} failed after "
        f"{EXPERIMENT_PARAMS.max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------

async def acall_model(
    model: ModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    dry_run: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Async model call with retry, logging, and cost tracking.

    Same interface as call_model() but uses litellm.acompletion().
    """
    temperature = temperature if temperature is not None else EXPERIMENT_PARAMS.temperature
    max_tokens = max_tokens if max_tokens is not None else EXPERIMENT_PARAMS.max_tokens
    seed = seed if seed is not None else EXPERIMENT_PARAMS.seed
    top_p = top_p if top_p is not None else EXPERIMENT_PARAMS.top_p

    call_id = str(uuid.uuid4())
    messages_hash = sha256_bytes(json.dumps(messages, sort_keys=True).encode())

    if dry_run:
        record = APICallRecord(
            call_id=call_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_id=model.model_id,
            litellm_id=model.litellm_id,
            provider=model.provider,
            messages_hash=messages_hash,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_seconds=0.0,
            status="dry_run",
            metadata=metadata or {},
        )
        _log_api_call(record)
        return {
            "content": "[DRY RUN] No API call made.",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "call_id": call_id,
            "latency_seconds": 0.0,
        }

    _use_vertex = model.provider == "vertex_ai"

    last_error: Exception | None = None
    for attempt in range(EXPERIMENT_PARAMS.max_retries + 1):
        t0 = time.monotonic()
        try:
            completion_kwargs: dict[str, Any] = {
                "model": model.litellm_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "seed": seed,
            }
            if model.supports_temperature:
                completion_kwargs["temperature"] = temperature
            if top_p is not None and top_p != 1.0:
                completion_kwargs["top_p"] = top_p
            if _use_vertex:
                completion_kwargs["vertex_project"] = os.getenv("VERTEXAI_PROJECT")
                completion_kwargs["vertex_location"] = os.getenv("VERTEXAI_LOCATION", "global")
            response = await litellm.acompletion(**completion_kwargs)
            latency = time.monotonic() - t0

            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or (input_tokens + output_tokens)

            cost_usd = (
                (input_tokens / 1000) * model.cost_per_1k_input
                + (output_tokens / 1000) * model.cost_per_1k_output
            )

            content = response.choices[0].message.content or ""
            response_id = getattr(response, "id", None)

            cost_tracker.record(model.model_id, cost_usd, input_tokens, output_tokens)

            record = APICallRecord(
                call_id=call_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                model_id=model.model_id,
                litellm_id=model.litellm_id,
                provider=model.provider,
                messages_hash=messages_hash,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_seconds=round(latency, 3),
                status="success",
                response_id=response_id,
                metadata=metadata or {},
            )
            _log_api_call(record)

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
            latency = time.monotonic() - t0

            record = APICallRecord(
                call_id=call_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                model_id=model.model_id,
                litellm_id=model.litellm_id,
                provider=model.provider,
                messages_hash=messages_hash,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=round(latency, 3),
                status="error",
                error_message=f"Attempt {attempt + 1}/{EXPERIMENT_PARAMS.max_retries + 1}: {e}",
                metadata=metadata or {},
            )
            _log_api_call(record)

            # Daily quota errors can't be solved by retrying — fail fast
            # Also mark the key as exhausted in the circuit breaker
            err_str = str(e).lower()
            if "per_model_per_day" in err_str or "per_day" in err_str:
                raise RuntimeError(
                    f"Async API call to {model.litellm_id} hit daily quota limit "
                    f"(no retry). Error: {e}"
                )

            # Credit exhaustion — also fail fast
            if "credit" in err_str and ("limit" in err_str or "exceeded" in err_str or "low" in err_str):
                raise RuntimeError(
                    f"Async API call to {model.litellm_id} hit credit limit "
                    f"(no retry). Error: {e}"
                )

            if attempt < EXPERIMENT_PARAMS.max_retries:
                delay = min(
                    EXPERIMENT_PARAMS.retry_base_delay * (2 ** attempt)
                    + random.uniform(0, 1),
                    EXPERIMENT_PARAMS.retry_max_delay,
                )
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"Async API call to {model.litellm_id} failed after "
        f"{EXPERIMENT_PARAMS.max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
