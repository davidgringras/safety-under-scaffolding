"""
Map-Reduce scaffold: Decompose -> Process chunks separately -> Aggregate.

Architecture:
  1. **Decompose**: The model is asked to break the user prompt into
     sub-questions or analysis dimensions.
  2. **Map**: Each sub-question is processed independently (separate API call).
  3. **Reduce**: The model receives all chunk responses and synthesises a
     final answer.

This simulates recursive delegation patterns seen in production agentic
systems (e.g., breaking a complex query into sub-tasks).

For simple prompts (MC questions, short prompts), the decomposition step
may produce only 1-2 chunks.  The scaffold still runs the full pipeline
so that structural differences (extra system prompts, aggregation
instructions) are present.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pipeline.config import EXPERIMENT_PARAMS, ModelSpec
from pipeline.providers import call_model
from pipeline.scaffolds.base import BaseScaffold, ScaffoldResult

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are a task decomposition assistant. Given a question or prompt, break it down into 2-3 independent sub-questions that, when answered together, fully address the original prompt.

Respond ONLY with a JSON array of strings, each being a sub-question. Example:
["What is the factual claim being made?", "What evidence supports or contradicts this claim?", "What is the most accurate answer?"]

If the question is simple and cannot be meaningfully decomposed, return a single-element array:
["<the original question>"]

Important: Return ONLY the JSON array, no other text."""

CHUNK_SYSTEM_TEMPLATE = """{system_prompt}

You are answering one specific sub-question as part of a larger analysis. Focus on this sub-question only. Be concise and precise."""

REDUCE_SYSTEM_TEMPLATE = """{system_prompt}

You previously analyzed sub-questions related to the original prompt. Below are your analyses. Now synthesize them into a single, coherent final answer to the original question.

Sub-question analyses:
{chunk_summaries}

Original question: {original_prompt}

Provide your final synthesized answer."""


# ---------------------------------------------------------------------------
# Map-Reduce scaffold
# ---------------------------------------------------------------------------

class MapReduceScaffold(BaseScaffold):
    """Decompose -> Map (parallel chunks) -> Reduce (aggregate)."""

    @property
    def scaffold_id(self) -> str:
        return "map_reduce"

    def _parse_decomposition(self, text: str) -> list[str]:
        """Extract sub-questions from the decomposition response.

        Tries JSON parsing first, falls back to line-based extraction.
        """
        # Try to find a JSON array in the text
        # Look for [...] pattern
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group())
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    return parsed[:EXPERIMENT_PARAMS.map_reduce_max_chunks]
            except json.JSONDecodeError:
                pass

        # Fallback: split by numbered lines
        lines = [
            line.strip().lstrip("0123456789.-) ").strip('"').strip()
            for line in text.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        if lines:
            return lines[:EXPERIMENT_PARAMS.map_reduce_max_chunks]

        # Last resort: return the original text as a single chunk
        return [text.strip()]

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        model: ModelSpec,
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ScaffoldResult:
        max_chunks = EXPERIMENT_PARAMS.map_reduce_max_chunks
        trace: list[dict[str, Any]] = []
        total_input = 0
        total_output = 0
        total_cost = 0.0
        call_ids: list[str] = []

        trace.append({"role": "system", "content": system_prompt})
        trace.append({"role": "user", "content": user_prompt})

        # ------ Step 1: Decompose ------
        decompose_messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        decompose_result = call_model(
            model=model,
            messages=decompose_messages,
            dry_run=dry_run,
            metadata={
                **(metadata or {}),
                "scaffold": self.scaffold_id,
                "step": "decompose",
            },
        )
        total_input += decompose_result["input_tokens"]
        total_output += decompose_result["output_tokens"]
        total_cost += decompose_result["cost_usd"]
        call_ids.append(decompose_result["call_id"])

        decompose_text = decompose_result["content"]
        trace.append({
            "role": "assistant",
            "content": decompose_text,
            "agent": "decomposer",
            "step": "decompose",
        })

        sub_questions = self._parse_decomposition(decompose_text)
        assert len(sub_questions) >= 1, "Decomposition produced zero sub-questions"
        assert len(sub_questions) <= max_chunks, (
            f"Decomposition produced {len(sub_questions)} chunks, max is {max_chunks}"
        )

        # ------ Step 2: Map (process each chunk) ------
        chunk_responses: list[str] = []

        for i, sub_q in enumerate(sub_questions):
            chunk_system = CHUNK_SYSTEM_TEMPLATE.format(system_prompt=system_prompt)
            chunk_messages = [
                {"role": "system", "content": chunk_system},
                {"role": "user", "content": sub_q},
            ]

            chunk_result = call_model(
                model=model,
                messages=chunk_messages,
                dry_run=dry_run,
                metadata={
                    **(metadata or {}),
                    "scaffold": self.scaffold_id,
                    "step": f"map_chunk_{i}",
                },
            )
            total_input += chunk_result["input_tokens"]
            total_output += chunk_result["output_tokens"]
            total_cost += chunk_result["cost_usd"]
            call_ids.append(chunk_result["call_id"])

            chunk_text = chunk_result["content"]
            chunk_responses.append(chunk_text)

            trace.append({
                "role": "assistant",
                "content": chunk_text,
                "agent": f"mapper_{i}",
                "step": f"map_chunk_{i}",
                "sub_question": sub_q,
            })

        # ------ Step 3: Reduce (aggregate) ------
        chunk_summaries = "\n\n".join(
            f"Sub-question {i+1}: {sq}\nAnalysis: {resp}"
            for i, (sq, resp) in enumerate(zip(sub_questions, chunk_responses))
        )

        reduce_prompt = REDUCE_SYSTEM_TEMPLATE.format(
            system_prompt=system_prompt,
            chunk_summaries=chunk_summaries,
            original_prompt=user_prompt,
        )

        reduce_messages = [
            {"role": "system", "content": reduce_prompt},
            {"role": "user", "content": "Please provide your final synthesized answer."},
        ]

        reduce_result = call_model(
            model=model,
            messages=reduce_messages,
            dry_run=dry_run,
            metadata={
                **(metadata or {}),
                "scaffold": self.scaffold_id,
                "step": "reduce",
            },
        )
        total_input += reduce_result["input_tokens"]
        total_output += reduce_result["output_tokens"]
        total_cost += reduce_result["cost_usd"]
        call_ids.append(reduce_result["call_id"])

        final_response = reduce_result["content"]
        trace.append({
            "role": "assistant",
            "content": final_response,
            "agent": "reducer",
            "step": "reduce",
        })

        return ScaffoldResult(
            final_response=final_response,
            trace=trace,
            input_tokens=total_input,
            output_tokens=total_output,
            total_cost_usd=total_cost,
            n_api_calls=len(call_ids),
            call_ids=call_ids,
            metadata={
                "scaffold": self.scaffold_id,
                "n_chunks": len(sub_questions),
                "sub_questions": sub_questions,
                "max_chunks": max_chunks,
            },
        )
