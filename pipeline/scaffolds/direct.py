"""
Direct scaffold: System prompt + user prompt -> single API call -> response.

This is the baseline configuration. The model receives a system prompt
and user prompt and produces a single response with no intermediate
processing, tool use, or multi-turn interaction.
"""

from __future__ import annotations

from typing import Any

from pipeline.config import ModelSpec
from pipeline.providers import call_model
from pipeline.scaffolds.base import BaseScaffold, ScaffoldResult


class DirectScaffold(BaseScaffold):
    """Baseline: single system+user prompt, single API call, single response."""

    @property
    def scaffold_id(self) -> str:
        return "direct"

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: ModelSpec,
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ScaffoldResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = call_model(
            model=model,
            messages=messages,
            dry_run=dry_run,
            metadata={
                **(metadata or {}),
                "scaffold": self.scaffold_id,
                "step": "direct_single",
            },
        )

        return ScaffoldResult(
            final_response=result["content"],
            trace=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": result["content"]},
            ],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_cost_usd=result["cost_usd"],
            n_api_calls=1,
            call_ids=[result["call_id"]],
            metadata={"scaffold": self.scaffold_id},
        )
