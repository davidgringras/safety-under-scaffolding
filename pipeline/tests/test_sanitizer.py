"""
Unit tests for the response sanitizer.

Tests stripping of CoT markers, tool calls, agent headers,
model self-identification, and map-reduce artefacts.
"""

import pytest

from pipeline.scoring.sanitizer import ResponseSanitizer


class TestResponseSanitizer:
    def setup_method(self):
        self.sanitizer = ResponseSanitizer(strip_self_id=True)

    # --- CoT stripping ---

    def test_strip_thinking_tags(self):
        text = "Here is my answer. <thinking>Let me reason about this...</thinking> The answer is B."
        result, report = self.sanitizer.sanitize(text)
        assert "<thinking>" not in result
        assert "reason about this" not in result
        assert "The answer is B." in result
        assert report.cot_stripped >= 1

    def test_strip_scratchpad_tags(self):
        text = "<scratchpad>Internal notes here</scratchpad>Final answer: A"
        result, _ = self.sanitizer.sanitize(text)
        assert "<scratchpad>" not in result
        assert "Internal notes" not in result
        assert "Final answer: A" in result

    def test_strip_reasoning_tags(self):
        text = "<reasoning>Step 1: ...</reasoning>The answer is C."
        result, _ = self.sanitizer.sanitize(text)
        assert "<reasoning>" not in result
        assert "The answer is C." in result

    # --- Tool call stripping ---

    def test_strip_thought_action_observation(self):
        text = """Thought: I need to think about this.
Action: calculator(2+2)
Observation: 4
The answer is 4."""
        result, report = self.sanitizer.sanitize(text)
        assert "Thought:" not in result
        assert "Action:" not in result
        assert "Observation:" not in result
        assert "The answer is 4." in result
        assert report.tool_lines_stripped >= 3

    def test_strip_tool_call_lines(self):
        text = "Tool call: search('safety')\nTool result: Found 3 results\nBased on this, A."
        result, _ = self.sanitizer.sanitize(text)
        assert "Tool call:" not in result
        assert "Tool result:" not in result
        assert "Based on this, A." in result

    # --- Agent header stripping ---

    def test_strip_critic_headers(self):
        text = """CRITIQUE: The response is accurate.
VERDICT: SAFE
REVISION_GUIDANCE: No changes needed.
Here is my final answer: B."""
        result, report = self.sanitizer.sanitize(text)
        assert "CRITIQUE:" not in result
        assert "VERDICT:" not in result
        assert "REVISION_GUIDANCE:" not in result
        assert "final answer: B." in result
        assert report.agent_lines_stripped >= 3

    def test_strip_agent_prefixes(self):
        text = "Primary: Here is my response.\nCritic: Looks good."
        result, _ = self.sanitizer.sanitize(text)
        assert "Primary:" not in result
        assert "Critic:" not in result

    # --- Map-reduce artefact stripping ---

    def test_strip_sub_question_headers(self):
        text = """Sub-question 1: What is X?
Analysis: X is Y.
Sub-question 2: What is Z?
Analysis: Z is W.
Synthesized answer: The final answer is A."""
        result, report = self.sanitizer.sanitize(text)
        assert "Sub-question 1:" not in result
        assert "Sub-question 2:" not in result
        assert report.map_reduce_lines_stripped >= 4

    # --- Model self-identification ---

    def test_strip_model_names(self):
        text = "As Claude, I think the answer is B."
        result, report = self.sanitizer.sanitize(text)
        assert "Claude" not in result
        assert "[REDACTED]" in result
        assert report.self_id_replacements >= 1

    def test_strip_gpt_mention(self):
        text = "I'm GPT-5.2 and I believe the answer is A."
        result, _ = self.sanitizer.sanitize(text)
        assert "GPT" not in result

    def test_strip_as_an_ai(self):
        text = "As an AI language model, I cannot provide medical advice."
        result, _ = self.sanitizer.sanitize(text)
        assert "As an AI" not in result

    def test_strip_anthropic_openai(self):
        text = "I was made by Anthropic. OpenAI makes GPT."
        result, _ = self.sanitizer.sanitize(text)
        assert "Anthropic" not in result
        assert "OpenAI" not in result

    def test_strip_gemini_reference(self):
        text = "As Gemini 3 Pro, my assessment is..."
        result, _ = self.sanitizer.sanitize(text)
        assert "Gemini" not in result

    # --- Formatting ---

    def test_collapse_multiple_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result, _ = self.sanitizer.sanitize(text)
        assert "\n\n\n" not in result

    def test_strip_leading_whitespace(self):
        text = "   \n  \nActual content here."
        result, _ = self.sanitizer.sanitize(text)
        assert result.startswith("Actual content")

    # --- Edge cases ---

    def test_empty_string(self):
        result, report = self.sanitizer.sanitize("")
        assert result == ""
        assert report.original_length == 0

    def test_no_artefacts(self):
        text = "The answer is B. This is because option B correctly states the fact."
        result, report = self.sanitizer.sanitize(text)
        assert result == text
        assert report.cot_stripped == 0
        assert report.tool_lines_stripped == 0

    def test_self_id_disabled(self):
        sanitizer = ResponseSanitizer(strip_self_id=False)
        text = "As Claude, I think A."
        result, report = sanitizer.sanitize(text)
        assert "Claude" in result  # NOT stripped when disabled
        assert report.self_id_replacements == 0

    def test_report_lengths(self):
        text = "<thinking>Internal thought</thinking>The answer is A."
        result, report = self.sanitizer.sanitize(text)
        assert report.original_length == len(text)
        assert report.sanitized_length == len(result)
        assert report.sanitized_length < report.original_length
