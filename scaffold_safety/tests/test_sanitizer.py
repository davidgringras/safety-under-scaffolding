"""
Unit tests for response sanitizer.

Tests that scaffold artefacts are stripped correctly while
preserving the substantive response content.
"""

import pytest
from scaffold_safety.blinding.sanitizer import ResponseSanitizer


class TestResponseSanitizer:
    """Test the ResponseSanitizer."""

    def setup_method(self):
        self.sanitizer = ResponseSanitizer(strip_self_id=True)

    def test_strip_cot(self):
        text = "Hello <thinking>internal reasoning</thinking> world"
        result, report = self.sanitizer.sanitize(text)
        assert "<thinking>" not in result
        assert "world" in result
        assert report.cot_stripped == 1

    def test_strip_tool_calls(self):
        text = "Thought: I should calculate\nAction: calculator(2+2)\nThe answer is 4."
        result, report = self.sanitizer.sanitize(text)
        assert "Thought:" not in result
        assert "Action:" not in result
        assert "4" in result
        assert report.tool_lines_stripped >= 1

    def test_strip_agent_headers(self):
        text = "CRITIQUE: good response\nVERDICT: SAFE\nThe answer is B."
        result, report = self.sanitizer.sanitize(text)
        assert "CRITIQUE:" not in result
        assert "VERDICT:" not in result
        assert report.agent_lines_stripped >= 1

    def test_strip_map_reduce(self):
        text = "Sub-question 1: What is X?\nAnalysis: X is Y.\nFinal answer: B"
        result, report = self.sanitizer.sanitize(text)
        assert "Sub-question" not in result
        assert report.map_reduce_lines_stripped >= 1

    def test_strip_self_id(self):
        text = "As an AI assistant, I think Claude would say B."
        result, report = self.sanitizer.sanitize(text)
        assert "Claude" not in result
        assert "[REDACTED]" in result
        assert report.self_id_replacements >= 1

    def test_no_strip_self_id(self):
        sanitizer = ResponseSanitizer(strip_self_id=False)
        text = "As an AI assistant made by Anthropic, the answer is B."
        result, report = sanitizer.sanitize(text)
        assert "Anthropic" in result
        assert report.self_id_replacements == 0

    def test_preserve_content(self):
        text = "The answer to this question is B."
        result, report = self.sanitizer.sanitize(text)
        assert "The answer to this question is B." in result

    def test_strip_code_blocks(self):
        text = "Here is code:\n```python\nprint('hello')\n```\nThe answer is A."
        result, report = self.sanitizer.sanitize(text)
        assert "```" not in result
        assert "print" not in result
        assert report.code_blocks_stripped >= 1

    def test_collapse_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result, _ = self.sanitizer.sanitize(text)
        assert "\n\n\n" not in result

    def test_empty_input(self):
        result, report = self.sanitizer.sanitize("")
        assert result == ""
        assert report.original_length == 0
