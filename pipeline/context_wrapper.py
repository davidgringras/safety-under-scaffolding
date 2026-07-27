"""
Long-context wrapper for ecological validity condition.

Embeds short benchmark prompts in ~4,500-token (~18K char) professional
documents to give Map-Reduce scaffolding something realistic to chunk.
This addresses the reviewer critique that short MC prompts force an
inappropriate architecture on Map-Reduce.

Usage:
    wrapper = ContextWrapper(target_tokens=4500, seed=42)
    wrapped_case = wrapper.wrap(case, benchmark_id="truthfulqa")
    # wrapped_case["prompt"] is now ~4,500 tokens with original embedded
    # wrapped_case["original_prompt"] preserves the short prompt
    # wrapped_case["metadata"]["context_wrapped"] == True
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

from pipeline.context_templates import TEMPLATES


# Rough chars-per-token ratio for estimating document length.
# GPT/Claude tokenizers average ~4 chars per token for English prose.
CHARS_PER_TOKEN = 4


class ContextWrapper:
    """Wraps short benchmark prompts in long-context professional documents.

    Parameters
    ----------
    target_tokens : int
        Target document length in tokens (default 7500).
    seed : int
        Base seed for deterministic block selection.
    """

    def __init__(self, target_tokens: int = 4500, seed: int = 42) -> None:
        self.target_tokens = target_tokens
        self.target_chars = target_tokens * CHARS_PER_TOKEN
        self.seed = seed

    def _case_hash(self, case_id: str) -> int:
        """Deterministic hash of case_id for block selection/ordering."""
        h = hashlib.sha256(f"{self.seed}:{case_id}".encode()).hexdigest()
        return int(h[:8], 16)

    def wrap(self, case: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
        """Wrap a benchmark case in a long-context document.

        Parameters
        ----------
        case : dict
            Benchmark case dict with at least "prompt" and "case_id".
        benchmark_id : str
            Key into TEMPLATES (e.g. "truthfulqa", "bbq").

        Returns
        -------
        dict
            A deep copy of the case with:
            - "prompt" replaced by the full wrapped document
            - "original_prompt" preserving the original short prompt
            - metadata fields: context_wrapped, context_type, context_tokens_approx
        """
        if benchmark_id not in TEMPLATES:
            raise ValueError(
                f"No context template for benchmark '{benchmark_id}'. "
                f"Available: {list(TEMPLATES.keys())}"
            )

        template = TEMPLATES[benchmark_id]
        case_id = case["case_id"]
        original_prompt = case["prompt"]

        # Build report ID from case hash
        case_h = self._case_hash(case_id)
        report_id = f"{case_h:08X}"

        # Assemble the shell parts
        header = template["shell_header"].format(report_id=report_id)
        target_header = template["shell_target_header"]
        target_footer = template["shell_target_footer"]
        footer = template["shell_footer"].format(report_id=report_id)

        # Fixed content: shell + prompt
        fixed_content = header + target_header + original_prompt + target_footer + footer
        fixed_chars = len(fixed_content)

        # Budget for content blocks
        block_budget_chars = self.target_chars - fixed_chars
        if block_budget_chars < 0:
            block_budget_chars = 0

        # Select and arrange blocks deterministically
        blocks = template["content_blocks"]
        n_blocks = len(blocks)

        # Permute block order using case hash
        indices = list(range(n_blocks))
        # Fisher-Yates shuffle with deterministic seed
        h = case_h
        for i in range(n_blocks - 1, 0, -1):
            h = (h * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF  # LCG
            j = h % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        # Select blocks until we fill the budget
        selected_blocks: list[str] = []
        chars_used = 0
        for idx in indices:
            block = blocks[idx]
            if chars_used + len(block) <= block_budget_chars:
                selected_blocks.append(block)
                chars_used += len(block)

        # Place the prompt in the middle third of the document
        # Split blocks into before-prompt and after-prompt groups
        n_selected = len(selected_blocks)
        # Target: prompt appears after ~40-60% of blocks (middle third)
        split_point = max(1, (n_selected * 2) // 5)  # ~40% before

        before_blocks = selected_blocks[:split_point]
        after_blocks = selected_blocks[split_point:]

        # Assemble final document
        document_parts = [
            header,
            "\n".join(before_blocks),
            target_header,
            original_prompt,
            target_footer,
            "\n".join(after_blocks),
            footer,
        ]
        wrapped_prompt = "\n".join(document_parts)

        # Build wrapped case
        wrapped = deepcopy(case)
        wrapped["prompt"] = wrapped_prompt
        wrapped["original_prompt"] = original_prompt

        # Metadata
        meta = wrapped.get("metadata", {})
        meta["context_wrapped"] = True
        meta["context_type"] = template["context_type"]
        meta["context_tokens_approx"] = len(wrapped_prompt) // CHARS_PER_TOKEN
        meta["context_n_blocks"] = n_selected
        wrapped["metadata"] = meta

        return wrapped

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate from character count."""
        return len(text) // CHARS_PER_TOKEN
