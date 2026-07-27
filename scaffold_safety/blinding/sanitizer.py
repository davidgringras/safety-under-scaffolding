"""
Response sanitizer for blinding.

Strips chain-of-thought markers, tool calls, agent headers, model
self-identification, and other artefacts from scaffold-processed
responses before they are sent to the LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_COT_PATTERNS = [
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<scratchpad>.*?</scratchpad>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<internal_monologue>.*?</internal_monologue>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\[thinking\].*?\[/thinking\]", re.DOTALL | re.IGNORECASE),
]

_TOOL_PATTERNS = [
    re.compile(r"^Thought:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Action:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Observation:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Tool call:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Tool result:.*$", re.MULTILINE | re.IGNORECASE),
]

_AGENT_PATTERNS = [
    re.compile(r"^CRITIQUE:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^VERDICT:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^REVISION_GUIDANCE:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^(Primary|Critic|Agent \d+):.*$", re.MULTILINE),
    re.compile(r"^\[?(Revision|Round) \d+\]?:?\s*", re.MULTILINE | re.IGNORECASE),
]

_MAP_REDUCE_PATTERNS = [
    re.compile(r"^Sub-question \d+:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Analysis:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Chunk \d+:.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^Synthesized answer:?\s*", re.MULTILINE | re.IGNORECASE),
]

_CODE_PATTERNS = [
    re.compile(r"```(?:python|javascript|bash|sh|code)?\s*\n.*?```", re.DOTALL),
    re.compile(r"^>>> .*$", re.MULTILINE),
    re.compile(r"^In \[\d+\]:.*$", re.MULTILINE),
    re.compile(r"^Out\[\d+\]:.*$", re.MULTILINE),
]

_SELF_ID_PATTERNS = [
    re.compile(r"\bI(?:'m| am) (?:an? )?(?:AI|language model|assistant|chatbot)\b", re.IGNORECASE),
    re.compile(r"\b(?:Claude|GPT|Gemini|LLaMA|Llama|DeepSeek|ChatGPT|OpenAI|Anthropic|Google|Meta)\b", re.IGNORECASE),
    re.compile(r"\b(?:opus|sonnet|haiku|gpt-\d|gemini[\s-]?\d)\b", re.IGNORECASE),
    re.compile(r"\bAs an AI(?:\s+(?:language model|assistant))?\b", re.IGNORECASE),
]


@dataclass
class SanitizationReport:
    """Record of what was stripped during sanitization."""
    original_length: int
    sanitized_length: int
    cot_stripped: int
    tool_lines_stripped: int
    agent_lines_stripped: int
    map_reduce_lines_stripped: int
    code_blocks_stripped: int
    self_id_replacements: int


class ResponseSanitizer:
    """Strip scaffold artefacts and model-identifying info from responses."""

    def __init__(self, strip_self_id: bool = True) -> None:
        self.strip_self_id = strip_self_id

    def sanitize(self, text: str) -> tuple[str, SanitizationReport]:
        """Sanitize a response and return (clean_text, report)."""
        original_length = len(text)
        counts = {"cot": 0, "tool": 0, "agent": 0, "mr": 0, "code": 0, "id": 0}
        result = text

        for pat in _COT_PATTERNS:
            counts["cot"] += len(pat.findall(result))
            result = pat.sub("", result)

        for pat in _TOOL_PATTERNS:
            counts["tool"] += len(pat.findall(result))
            result = pat.sub("", result)

        for pat in _AGENT_PATTERNS:
            counts["agent"] += len(pat.findall(result))
            result = pat.sub("", result)

        for pat in _MAP_REDUCE_PATTERNS:
            counts["mr"] += len(pat.findall(result))
            result = pat.sub("", result)

        for pat in _CODE_PATTERNS:
            counts["code"] += len(pat.findall(result))
            result = pat.sub("", result)

        if self.strip_self_id:
            for pat in _SELF_ID_PATTERNS:
                counts["id"] += len(pat.findall(result))
                result = pat.sub("[REDACTED]", result)

        result = re.sub(r"\n{3,}", "\n\n", result).strip()

        return result, SanitizationReport(
            original_length=original_length,
            sanitized_length=len(result),
            cot_stripped=counts["cot"],
            tool_lines_stripped=counts["tool"],
            agent_lines_stripped=counts["agent"],
            map_reduce_lines_stripped=counts["mr"],
            code_blocks_stripped=counts["code"],
            self_id_replacements=counts["id"],
        )
