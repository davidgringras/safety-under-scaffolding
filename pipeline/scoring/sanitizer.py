"""
Response sanitizer for blinding.

Strips chain-of-thought markers, tool calls, agent headers, model
self-identification, and other artefacts from scaffold-processed
responses before they are sent to the LLM judge.

The goal is assessor blinding: the judge should not be able to determine
which model or which scaffold configuration produced the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Patterns to strip (compiled once at import time)
# ---------------------------------------------------------------------------

# Chain-of-thought markers used by various models
_COT_PATTERNS = [
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<scratchpad>.*?</scratchpad>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<internal_monologue>.*?</internal_monologue>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\[thinking\].*?\[/thinking\]", re.DOTALL | re.IGNORECASE),
]

# ReAct / tool-call artefacts
_TOOL_PATTERNS = [
    re.compile(r"^Thought:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Action:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Observation:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Tool call:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Tool result:.*$", re.MULTILINE | re.IGNORECASE),
]

# Multi-agent artefacts
_AGENT_PATTERNS = [
    re.compile(r"^CRITIQUE:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^VERDICT:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^REVISION_GUIDANCE:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(Primary|Critic|Agent \d+):.*$", re.MULTILINE),
    re.compile(r"^\[?(Revision|Round) \d+\]?:?\s*", re.MULTILINE | re.IGNORECASE),
]

# Map-reduce artefacts
_MAP_REDUCE_PATTERNS = [
    re.compile(r"^Sub-question \d+:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Analysis:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Chunk \d+:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Synthesized answer:?\s*", re.MULTILINE | re.IGNORECASE),
]

# Code block / REPL artefacts (from tool-using scaffolds)
_CODE_PATTERNS = [
    re.compile(r"```(?:python|javascript|bash|sh|code)?\s*\n.*?```", re.DOTALL),
    re.compile(r"^>>> .*$", re.MULTILINE),  # Python REPL prompts
    re.compile(r"^In \[\d+\]:.*$", re.MULTILINE),  # Jupyter-style
    re.compile(r"^Out\[\d+\]:.*$", re.MULTILINE),
    re.compile(r"^Iteration \d+:.*$", re.MULTILINE | re.IGNORECASE),  # REPL iteration markers
]

# Model self-identification (names, versions)
_SELF_ID_PATTERNS = [
    re.compile(r"\bI(?:'m| am) (?:an? )?(?:AI|language model|assistant|chatbot)\b", re.IGNORECASE),
    re.compile(r"\b(?:Claude|GPT|Gemini|LLaMA|Llama|DeepSeek|ChatGPT|OpenAI|Anthropic|Google|Meta)\b", re.IGNORECASE),
    re.compile(r"\b(?:opus|sonnet|haiku|gpt-\d|gemini[\s-]?\d)\b", re.IGNORECASE),
    re.compile(r"\bAs an AI(?:\s+(?:language model|assistant))?\b", re.IGNORECASE),
]

# Formatting cleanup
_CLEANUP_PATTERNS = [
    re.compile(r"\n{3,}"),           # collapse triple+ newlines
    re.compile(r"^\s*\n", re.MULTILINE),  # blank lines at start
]


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

@dataclass
class SanitizationReport:
    """Record of what was stripped during sanitization."""
    original_length: int
    sanitized_length: int
    cot_stripped: int           # count of CoT blocks removed
    tool_lines_stripped: int
    agent_lines_stripped: int
    map_reduce_lines_stripped: int
    code_blocks_stripped: int
    self_id_replacements: int


class ResponseSanitizer:
    """Strip scaffold artefacts and model-identifying info from responses."""

    def __init__(self, strip_self_id: bool = True) -> None:
        """
        Parameters
        ----------
        strip_self_id : bool
            Whether to replace model self-identification strings.
            Default True for blinded judging.  Set False for debugging.
        """
        self.strip_self_id = strip_self_id

    def sanitize(self, text: str) -> tuple[str, SanitizationReport]:
        """Sanitize a response and return (clean_text, report).

        Parameters
        ----------
        text : str
            Raw model response (may include CoT, tool calls, etc.).

        Returns
        -------
        tuple[str, SanitizationReport]
            The sanitized text and a report of what was removed.
        """
        original_length = len(text)
        cot_count = 0
        tool_count = 0
        agent_count = 0
        mr_count = 0
        code_count = 0
        self_id_count = 0

        result = text

        # 1. Strip CoT blocks
        for pat in _COT_PATTERNS:
            matches = pat.findall(result)
            cot_count += len(matches)
            result = pat.sub("", result)

        # 2. Strip tool-call lines
        for pat in _TOOL_PATTERNS:
            matches = pat.findall(result)
            tool_count += len(matches)
            result = pat.sub("", result)

        # 3. Strip agent headers
        for pat in _AGENT_PATTERNS:
            matches = pat.findall(result)
            agent_count += len(matches)
            result = pat.sub("", result)

        # 4. Strip map-reduce artefacts
        for pat in _MAP_REDUCE_PATTERNS:
            matches = pat.findall(result)
            mr_count += len(matches)
            result = pat.sub("", result)

        # 5. Strip code blocks / REPL artefacts
        for pat in _CODE_PATTERNS:
            matches = pat.findall(result)
            code_count += len(matches)
            result = pat.sub("", result)

        # 6. Replace model self-identification
        if self.strip_self_id:
            for pat in _SELF_ID_PATTERNS:
                matches = pat.findall(result)
                self_id_count += len(matches)
                result = pat.sub("[REDACTED]", result)

        # 7. Clean up formatting
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        report = SanitizationReport(
            original_length=original_length,
            sanitized_length=len(result),
            cot_stripped=cot_count,
            tool_lines_stripped=tool_count,
            agent_lines_stripped=agent_count,
            map_reduce_lines_stripped=mr_count,
            code_blocks_stripped=code_count,
            self_id_replacements=self_id_count,
        )

        return result, report
