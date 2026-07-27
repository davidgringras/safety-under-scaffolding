"""
Fixed OpenAI Agents SDK runner for E9 production framework evaluation.

WHAT CHANGED (vs src/production_frameworks/openai_sdk_scaffold/runner.py):
--------------------------------------------------------------------------
Original problem (C1 review, Criticism 5):
  - Only used the actual OpenAI Agents SDK for GPT-5.2
  - All other 4 models fell back to plain litellm.completion()
  - The non-OpenAI path was functionally identical to direct API

Fix:
  - Uses the OpenAI Agents SDK for ALL models via its external model
    provider interface (agents.models.ModelProvider / Model).
  - For OpenAI models (GPT-5.2): Uses the SDK natively (same as before).
  - For non-OpenAI models: Implements a LiteLLMModelProvider that wraps
    litellm as a custom model backend, allowing the SDK's agent loop,
    tool dispatch, and handoff infrastructure to manage the conversation
    even when the underlying model is Anthropic, Google, etc.
  - All models go through the SDK's Runner.run() execution path.

Documented limitation:
  - The OpenAI Agents SDK is architecturally designed for OpenAI models.
    Using it with non-OpenAI models via a custom ModelProvider means:
    (a) Function calling schemas may be translated imperfectly
    (b) The SDK's guardrails features only work with OpenAI models
    (c) Streaming and tracing may behave differently
  - This limitation is documented honestly in E9_FIX_REPORT.md.
    The paper should acknowledge that the SDK is only fully functional
    with OpenAI models and that non-OpenAI usage is through an adapter
    layer that may not exercise all SDK features.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import litellm

from src.production_frameworks.common import (
    get_system_prompt,
    make_result_record,
    save_result,
    RESULTS_DIR,
)

FRAMEWORK_NAME = "openai_sdk_v2"
FRAMEWORK_VERSION = "0.8.4"


# ---------------------------------------------------------------------------
# Custom model provider: wraps litellm for non-OpenAI models
# ---------------------------------------------------------------------------

class LiteLLMModel:
    """Adapter that wraps litellm to conform to the OpenAI Agents SDK's
    Model interface.

    The SDK expects models to implement get_response() which takes
    a list of messages and returns a ModelResponse. This adapter
    translates between the SDK's internal message format and litellm's
    chat completion API.
    """

    def __init__(self, model_litellm_id: str, model_id: str):
        self.model_litellm_id = model_litellm_id
        self.model_id = model_id

    async def get_response(
        self,
        system_instructions: str | None,
        input_messages: list[dict],
        model_settings: Any = None,
        tools: list | None = None,
        output_type: Any = None,
        handoffs: list | None = None,
        tracing: Any = None,
    ) -> Any:
        """Translate SDK request -> litellm call -> SDK response format.

        The Agents SDK calls this method with its internal message format.
        We translate to standard chat completion messages, call litellm,
        and wrap the response in the SDK's expected ModelResponse format.
        """
        from agents import ModelResponse, Usage

        # Build standard chat messages
        messages = []
        if system_instructions:
            messages.append({"role": "system", "content": system_instructions})

        # Translate SDK's input messages to standard format
        for msg in input_messages:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role if isinstance(msg.role, str) else str(msg.role)
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                messages.append({"role": role, "content": content})
            elif isinstance(msg, dict):
                messages.append(msg)
            else:
                messages.append({"role": "user", "content": str(msg)})

        # Call litellm
        kwargs: dict[str, Any] = {
            "model": self.model_litellm_id,
            "messages": messages,
            "max_tokens": 1024,
        }
        if self.model_id != "gpt52":
            kwargs["temperature"] = 0.0

        result = await asyncio.to_thread(litellm.completion, **kwargs)

        response_text = result.choices[0].message.content or ""
        usage_data = result.usage

        # Wrap in SDK's ModelResponse format
        from agents import ResponseOutputMessage, ResponseOutputText

        output_message = ResponseOutputMessage(
            role="assistant",
            content=[ResponseOutputText(text=response_text)],
        )

        usage = Usage(
            input_tokens=usage_data.prompt_tokens if usage_data else 0,
            output_tokens=usage_data.completion_tokens if usage_data else 0,
            requests=1,
        )

        return ModelResponse(
            output=[output_message],
            usage=usage,
        )


class LiteLLMModelProvider:
    """ModelProvider that returns LiteLLMModel instances for non-OpenAI models.

    The SDK's Runner accepts a model_provider parameter that overrides
    default model resolution. This allows routing any model through
    the SDK's agent execution loop.
    """

    def __init__(self, model_litellm_id: str, model_id: str):
        self.model = LiteLLMModel(model_litellm_id, model_id)

    def get_model(self, model_name: str | None = None) -> LiteLLMModel:
        """Return our litellm-backed model regardless of requested name."""
        return self.model


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(
    case: dict[str, Any],
    model_litellm_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Run a single benchmark case through the OpenAI Agents SDK.

    For ALL models, uses the actual Agents SDK execution path:
    - Agent with instructions (system prompt)
    - Runner.run() executes the agent loop
    - For OpenAI models: native SDK model resolution
    - For non-OpenAI models: LiteLLMModelProvider adapter

    This ensures all models go through the SDK's agent lifecycle:
    instruction injection, conversation management, and output extraction.
    """
    bench = case["_benchmark_key"]
    sys_prompt = get_system_prompt(bench)
    user_prompt = case["prompt"]

    start = time.monotonic()
    error = None
    response = ""
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    trace = []
    used_native_sdk = True  # All paths now use the SDK

    try:
        from agents import Agent, Runner

        if model_id == "gpt52":
            # Native OpenAI model -- SDK resolves natively
            agent = Agent(
                name="Safety Benchmark Evaluator",
                instructions=sys_prompt,
                model="gpt-5.2",
            )
            result = asyncio.run(Runner.run(agent, user_prompt))
        else:
            # Non-OpenAI model -- use our LiteLLM adapter via model_provider
            # The agent's model field is a placeholder; the provider overrides it
            agent = Agent(
                name="Safety Benchmark Evaluator",
                instructions=sys_prompt,
                model=model_litellm_id,  # Used as identifier only
            )
            provider = LiteLLMModelProvider(model_litellm_id, model_id)
            result = asyncio.run(
                Runner.run(
                    agent,
                    user_prompt,
                    model_provider=provider,
                )
            )

        response = result.final_output or ""

        # Build trace
        trace = [
            {
                "role": "system",
                "content": sys_prompt,
                "metadata": {"source": "Agent.instructions"},
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ]

        # Extract usage if available from SDK result
        if hasattr(result, "raw_responses") and result.raw_responses:
            for raw_resp in result.raw_responses:
                if hasattr(raw_resp, "usage") and raw_resp.usage:
                    input_tokens += getattr(raw_resp.usage, "input_tokens", 0)
                    output_tokens += getattr(raw_resp.usage, "output_tokens", 0)

    except ImportError as e:
        error = f"ImportError: OpenAI Agents SDK not installed: {e}"
        used_native_sdk = False
    except Exception as e:
        error = f"SDK_{type(e).__name__}: {e}"

    latency = time.monotonic() - start

    return make_result_record(
        case=case,
        framework=FRAMEWORK_NAME,
        model_id=model_id,
        response=response,
        trace=trace,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency,
        error=error,
        metadata={
            "framework_version": FRAMEWORK_VERSION,
            "used_native_sdk": used_native_sdk,
            "model_provider": "native" if model_id == "gpt52" else "litellm_adapter",
            "sdk_execution_path": "Runner.run()",
            "fix_version": "v2",
        },
    )


def run_batch(
    cases: list[dict[str, Any]],
    model_litellm_id: str,
    model_id: str,
    output_file: Path | None = None,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run a batch of cases through the fixed OpenAI Agents SDK sequentially."""
    results = []
    for i, case in enumerate(cases):
        print(f"  [{FRAMEWORK_NAME}] {model_id} case {i+1}/{len(cases)}: {case['case_id']}")
        result = run_case(case, model_litellm_id, model_id)
        save_result(result, FRAMEWORK_NAME, output_file)
        results.append(result)
        if result["error"]:
            print(f"    ERROR: {result['error']}")
    return results
