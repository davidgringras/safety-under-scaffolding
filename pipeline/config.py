"""
Configuration management for Safety Under Scaffolding.

Loads API keys from .env, defines experiment parameters, model strings,
benchmark metadata, and budget-tracking settings. All experiment-wide
constants live here so nothing is hard-coded in other modules.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent          # ~/project/
PIPELINE_ROOT = Path(__file__).resolve().parent                # ~/project/pipeline/
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

for _d in (DATA_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

_env_path = PROJECT_ROOT / ".env"
load_dotenv(_env_path)


def _require_env(key: str) -> str:
    """Return an environment variable or raise with a helpful message."""
    val = os.getenv(key)
    if val is None:
        raise EnvironmentError(
            f"Missing required environment variable: {key}. "
            f"Add it to {_env_path}"
        )
    return val


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a model under test."""
    model_id: str               # internal short name (e.g. "opus")
    litellm_id: str             # string passed to litellm.completion()
    provider: str               # anthropic | openai | google | together | deepseek
    supports_batch: bool = False
    supports_temperature: bool = True  # False for reasoning models (e.g. GPT-5.2)
    cost_per_1k_input: float = 0.0   # USD per 1K input tokens
    cost_per_1k_output: float = 0.0  # USD per 1K output tokens
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    "opus": ModelSpec(
        model_id="opus",
        litellm_id="claude-opus-4-6",
        provider="anthropic",
        supports_batch=True,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        notes="Anthropic Claude Opus 4.6 — batch API",
    ),
    "gpt52": ModelSpec(
        model_id="gpt52",
        litellm_id="gpt-5.2",
        provider="openai",
        supports_batch=True,
        supports_temperature=False,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.010,
        notes="OpenAI GPT-5.2 — reasoning model, no temperature control",
    ),
    "gemini3pro": ModelSpec(
        model_id="gemini3pro",
        litellm_id="openrouter/google/gemini-2.5-pro",
        provider="openrouter",
        supports_batch=False,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.012,
        notes="Google Gemini 2.5 Pro via OpenRouter (preview endpoint expired)",
    ),
    "llama4": ModelSpec(
        model_id="llama4",
        litellm_id="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="together",
        supports_batch=False,
        cost_per_1k_input=0.0002,
        cost_per_1k_output=0.0002,
        notes="Meta Llama 4 Maverick via Together AI",
    ),
    "deepseek": ModelSpec(
        model_id="deepseek",
        litellm_id="deepseek/deepseek-chat",
        provider="deepseek",
        supports_batch=False,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        notes="DeepSeek V3.2 non-thinking mode",
    ),
    "mistral": ModelSpec(
        model_id="mistral",
        litellm_id="mistral/mistral-large-latest",
        provider="mistral",
        supports_batch=False,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        notes="Mistral Large 2 via Mistral API",
    ),
}

# Judge models
JUDGE_PRIMARY = ModelSpec(
    model_id="gemini3flash",
    litellm_id="gemini/gemini-3-flash-preview",
    provider="google",
    supports_batch=False,
    cost_per_1k_input=0.0001,
    cost_per_1k_output=0.0004,
    notes="Gemini 3 Flash Preview — primary safety judge",
)

JUDGE_VALIDATION = ModelSpec(
    model_id="opus_judge",
    litellm_id="claude-opus-4-6",
    provider="anthropic",
    supports_batch=True,
    cost_per_1k_input=0.015,
    cost_per_1k_output=0.075,
    notes="Claude Opus 4.6 — validation judge (batch, 20% sample)",
)


# ---------------------------------------------------------------------------
# Scaffold / config definitions
# ---------------------------------------------------------------------------

SCAFFOLD_IDS = ("direct", "react", "multi_agent", "map_reduce")

# Context length conditions for ecological validity
CONTEXT_CONDITIONS = ("short", "long")

# Default: budget mode — long-context for Map-Reduce only.
# Full mode (long-context for all scaffolds) available as override.
DEFAULT_CONTEXT_CONFIGS: dict[str, list[str]] = {
    "direct": ["short"],
    "react": ["short"],
    "multi_agent": ["short"],
    "map_reduce": ["short", "long"],
}

FULL_CONTEXT_CONFIGS: dict[str, list[str]] = {
    "direct": ["short", "long"],
    "react": ["short", "long"],
    "multi_agent": ["short", "long"],
    "map_reduce": ["short", "long"],
}


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkSpec:
    """Immutable specification for a benchmark dataset."""
    benchmark_id: str
    display_name: str
    n_cases: int
    scoring_method: str        # "automated" | "judge" | "hybrid"
    safety_property: str       # sycophancy | bias | over_refusal | truthfulness
    data_file: str             # filename under DATA_DIR
    notes: str = ""


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "truthfulqa": BenchmarkSpec(
        benchmark_id="truthfulqa",
        display_name="TruthfulQA MC1",
        n_cases=817,
        scoring_method="automated",
        safety_property="truthfulness",
        data_file="truthfulqa_mc1.jsonl",
        notes="Null control benchmark — expect no scaffolding effect",
    ),
    "bbq": BenchmarkSpec(
        benchmark_id="bbq",
        display_name="BBQ",
        n_cases=800,
        scoring_method="automated",
        safety_property="bias",
        data_file="bbq.jsonl",
        notes="Stratified across 9 bias dimensions, balanced ambiguous/disambiguated",
    ),
    "sycophancy": BenchmarkSpec(
        benchmark_id="sycophancy",
        display_name="Sycophancy Eval",
        n_cases=500,
        scoring_method="hybrid",
        safety_property="sycophancy",
        data_file="sycophancy_eval.jsonl",
        notes="Stratified from ~1,000 Anthropic model-written evals; hybrid scoring",
    ),
    "xstest_orbench": BenchmarkSpec(
        benchmark_id="xstest_orbench",
        display_name="XSTest + OR-Bench",
        n_cases=500,
        scoring_method="judge",
        safety_property="over_refusal",
        data_file="xstest_orbench.jsonl",
        notes="250 XSTest + 250 OR-Bench hard, stratified safe/unsafe",
    ),
}


# ---------------------------------------------------------------------------
# Experiment parameters (clinical-trial fixed before data collection)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentParams:
    """Fixed experiment parameters — locked at pre-registration."""
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 42
    top_p: float = 1.0
    # ReAct-specific
    react_max_iterations: int = 5
    # Multi-agent-specific
    multi_agent_max_revisions: int = 1
    # Map-reduce-specific
    map_reduce_max_chunks: int = 3
    # Pilot
    pilot_fraction: float = 0.05
    # Judge
    judge_validation_fraction: float = 0.20
    # Retry
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    # Concurrency
    max_concurrent_requests: int = 10
    # Batch size for checkpoint writes
    checkpoint_batch_size: int = 50


EXPERIMENT_PARAMS = ExperimentParams()


# ---------------------------------------------------------------------------
# Budget alerts
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Budget tracking and alert thresholds."""
    total_budget_usd: float = 175.0       # revised total budget
    hard_stop_usd: float = 200.0          # absolute hard stop — experiment pauses
    per_model_multiplier: float = 3.0     # pause if any model > 3x estimate
    alert_thresholds_usd: tuple[float, ...] = (50.0, 100.0, 150.0)  # absolute $ alerts


BUDGET_CONFIG = BudgetConfig()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "json"                       # structured JSON logging
LOG_FILE = LOGS_DIR / "experiment.jsonl"
API_LOG_FILE = LOGS_DIR / "api_calls.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key(provider: str) -> str:
    """Return the API key for a provider, reading from environment."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    if provider == "vertex_ai":
        return "ADC"  # Vertex AI uses Application Default Credentials, not API keys
    env_var = key_map.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {provider}")
    return _require_env(env_var)


def validate_env() -> dict[str, bool]:
    """Check which API keys are configured. Returns {provider: bool}."""
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }
    return {provider: os.getenv(var) is not None for provider, var in key_map.items()}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def dump_config_snapshot() -> dict[str, Any]:
    """Return a serialisable snapshot of the full experiment configuration.

    Used for pre-registration sealing and audit trail.
    """
    return {
        "version": __import__("pipeline").__version__,
        "models": {k: {"litellm_id": v.litellm_id, "provider": v.provider}
                   for k, v in MODELS.items()},
        "scaffolds": list(SCAFFOLD_IDS),
        "benchmarks": {k: {"n_cases": v.n_cases, "scoring_method": v.scoring_method}
                       for k, v in BENCHMARKS.items()},
        "params": {
            "temperature": EXPERIMENT_PARAMS.temperature,
            "max_tokens": EXPERIMENT_PARAMS.max_tokens,
            "seed": EXPERIMENT_PARAMS.seed,
            "top_p": EXPERIMENT_PARAMS.top_p,
            "react_max_iterations": EXPERIMENT_PARAMS.react_max_iterations,
            "multi_agent_max_revisions": EXPERIMENT_PARAMS.multi_agent_max_revisions,
            "map_reduce_max_chunks": EXPERIMENT_PARAMS.map_reduce_max_chunks,
        },
        "judge_primary": JUDGE_PRIMARY.litellm_id,
        "judge_validation": JUDGE_VALIDATION.litellm_id,
        "budget": {
            "total_budget_usd": BUDGET_CONFIG.total_budget_usd,
            "hard_stop_usd": BUDGET_CONFIG.hard_stop_usd,
            "alert_thresholds_usd": list(BUDGET_CONFIG.alert_thresholds_usd),
        },
    }
