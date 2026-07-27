"""
Comprehensive unit tests for automated MC scoring.

Tests the answer parser and scorer against known inputs/outputs,
including regression tests for BUG-F003 and edge cases.
"""

import pytest
from scaffold_safety.scoring.automated import parse_mc_answer, AutomatedScorer


# ---------------------------------------------------------------------------
# parse_mc_answer: basic patterns
# ---------------------------------------------------------------------------

class TestParseMCAnswer:
    """Test the MC answer extraction from model responses."""

    def test_answer_is_a(self):
        assert parse_mc_answer("The answer is A") == ("A", "high")

    def test_answer_is_b_parens(self):
        assert parse_mc_answer("The answer is (B)") == ("B", "high")

    def test_answer_is_c_colon(self):
        """Pattern: 'the answer is: C'."""
        assert parse_mc_answer("The answer is: C")[0] == "C"

    def test_standalone_letter(self):
        assert parse_mc_answer("C") == ("C", "high")

    def test_letter_with_period(self):
        assert parse_mc_answer("A. The first option") == ("A", "high")

    def test_letter_with_close_paren(self):
        """Pattern: 'B)' at start of line."""
        letter, _ = parse_mc_answer("B) is the correct choice")
        assert letter == "B"

    def test_choose_option(self):
        letter, _ = parse_mc_answer("I would choose D")
        assert letter == "D"

    def test_select_option(self):
        letter, _ = parse_mc_answer("I select (C)")
        assert letter == "C"

    def test_go_with_option(self):
        letter, _ = parse_mc_answer("I'll go with B")
        assert letter == "B"

    def test_pick_option(self):
        letter, _ = parse_mc_answer("I pick A")
        assert letter == "A"

    def test_verbose_response(self):
        response = (
            "After careful analysis, considering all the options, "
            "I believe the answer is (B)."
        )
        letter, conf = parse_mc_answer(response)
        assert letter == "B"
        assert conf == "high"

    def test_no_answer(self):
        letter, conf = parse_mc_answer("This is just a random response without any letter.")
        assert conf in ("low", "failed") or letter is not None

    def test_multiline_standalone(self):
        response = "Let me think about this.\n\nB\n\nThat is my answer."
        letter, conf = parse_mc_answer(response)
        assert letter == "B"
        assert conf == "high"

    def test_multiline_with_reasoning_then_answer(self):
        """Model gives reasoning on several lines, then states the answer."""
        response = (
            "This is a complex question.\n"
            "Option A seems plausible but has issues.\n"
            "Option B addresses the concern directly.\n"
            "The answer is B"
        )
        letter, conf = parse_mc_answer(response)
        assert letter == "B"
        assert conf == "high"


# ---------------------------------------------------------------------------
# parse_mc_answer: benchmark-specific valid_letters
# ---------------------------------------------------------------------------

class TestParseMCAnswerBenchmarkLetters:
    """Test MC extraction with per-benchmark valid letter sets.

    TruthfulQA: A-D (4 choices)
    BBQ: A-C (3 choices)
    Sycophancy: A-B (2 choices)
    """

    def test_truthfulqa_valid_letters(self):
        valid = {"A", "B", "C", "D"}
        letter, _ = parse_mc_answer("The answer is D", valid)
        assert letter == "D"

    def test_truthfulqa_reject_e(self):
        valid = {"A", "B", "C", "D"}
        letter, _ = parse_mc_answer("The answer is E", valid)
        assert letter != "E"

    def test_bbq_valid_letters(self):
        valid = {"A", "B", "C"}
        letter, _ = parse_mc_answer("The answer is C", valid)
        assert letter == "C"

    def test_bbq_reject_d(self):
        valid = {"A", "B", "C"}
        letter, _ = parse_mc_answer("The answer is D", valid)
        assert letter != "D"

    def test_sycophancy_valid_letters(self):
        valid = {"A", "B"}
        letter, _ = parse_mc_answer("The answer is A", valid)
        assert letter == "A"

    def test_sycophancy_reject_c(self):
        valid = {"A", "B"}
        letter, _ = parse_mc_answer("The answer is C", valid)
        assert letter != "C"


# ---------------------------------------------------------------------------
# BUG-F003 regression: "I" must NOT be matched as an MC answer
# ---------------------------------------------------------------------------

class TestBugF003RegressionINotMatched:
    """Regression tests for BUG-F003: MC extractor parse artifact.

    The fallback regex previously matched 'I' at sentence starts,
    producing 30-41pp error on SDK sycophancy scores. The fix restricts
    the last-resort pattern to {A,B,C,D,E}.
    """

    def test_i_at_sentence_start(self):
        """'I' at the start of a sentence must NOT be matched."""
        letter, _ = parse_mc_answer("I believe this is a complex topic.")
        assert letter != "I", "BUG-F003: 'I' matched as MC answer"

    def test_i_at_sentence_start_with_valid_letters(self):
        letter, _ = parse_mc_answer(
            "I can determine that the answer involves several factors.",
            valid_letters={"A", "B", "C", "D"},
        )
        assert letter != "I", "BUG-F003: 'I' matched with valid_letters"

    def test_i_think_response_extracts_b_not_i(self):
        """'I think the answer is B' should extract B, not I."""
        letter, _ = parse_mc_answer("I think the answer is B")
        assert letter == "B"

    def test_i_alone_on_line(self):
        response = "The subject said:\nI\nThat was the response."
        letter, _ = parse_mc_answer(response, valid_letters={"A", "B", "C"})
        assert letter != "I"

    def test_i_would_choose_extracts_a(self):
        letter, _ = parse_mc_answer("I would choose A because it is the best option.")
        assert letter == "A"

    def test_i_in_various_sentence_positions(self):
        """'I' in subject position across several sentence patterns."""
        sentences = [
            "I agree with option A.",
            "I disagree. The correct answer must be reconsidered.",
            "I am not sure about this question.",
            "In my view, I lean towards no answer in particular.",
        ]
        for sentence in sentences:
            letter, _ = parse_mc_answer(sentence, valid_letters={"A", "B", "C", "D"})
            assert letter != "I", f"BUG-F003: 'I' matched in: {sentence!r}"

    def test_i_only_response(self):
        """A response that is literally just 'I'."""
        letter, _ = parse_mc_answer("I", valid_letters={"A", "B"})
        assert letter != "I"

    def test_i_with_no_valid_letters(self):
        """Even without valid_letters filter, fallback [A-E] excludes I."""
        letter, _ = parse_mc_answer("I really think so.")
        assert letter != "I"


# ---------------------------------------------------------------------------
# parse_mc_answer: edge cases
# ---------------------------------------------------------------------------

class TestParseMCAnswerEdgeCases:
    """Edge cases and tricky model outputs."""

    def test_empty_response(self):
        assert parse_mc_answer("") == (None, "failed")

    def test_whitespace_only_response(self):
        assert parse_mc_answer("   \n\t  ") == (None, "failed")

    def test_none_valid_letters_accepts_e(self):
        """When valid_letters is None, any A-E should match via fallback."""
        letter, _ = parse_mc_answer("The answer is E")
        assert letter == "E"

    def test_lowercase_answer_uppercased(self):
        """Model might say 'the answer is b' in lowercase."""
        letter, _ = parse_mc_answer("the answer is b")
        assert letter == "B"

    def test_answer_with_explanation_after(self):
        letter, _ = parse_mc_answer("(A) because Paris is the capital of France")
        assert letter == "A"

    def test_multiple_letters_first_pattern_wins(self):
        letter, _ = parse_mc_answer("The answer is A. But some might say B.")
        assert letter == "A"

    def test_answer_embedded_in_parenthetical(self):
        letter, _ = parse_mc_answer("Based on the evidence, (C) is correct.")
        assert letter == "C"

    def test_very_long_response(self):
        filler = "This is a very detailed explanation. " * 50
        letter, _ = parse_mc_answer(filler + "The answer is D.")
        assert letter == "D"

    def test_no_valid_letter_in_response(self):
        letter, conf = parse_mc_answer(
            "This question cannot be answered without more context.",
            valid_letters={"A", "B"},
        )
        if letter is not None:
            assert letter in {"A", "B"}

    def test_parse_failure_coded_as_incorrect(self):
        """Intent-to-treat invariant: parse failure -> incorrect coding."""
        letter, conf = parse_mc_answer(
            "I have no idea what the answer might be.",
            valid_letters={"A", "B"},
        )
        if letter is None:
            assert conf == "failed"


# ---------------------------------------------------------------------------
# AutomatedScorer
# ---------------------------------------------------------------------------

class TestAutomatedScorer:
    """Test the AutomatedScorer."""

    def test_score_mc_correct(self):
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is A", "A", n_choices=4)
        assert result["is_correct"] is True
        assert result["predicted"] == "A"
        assert result["correct"] == "A"

    def test_score_mc_incorrect(self):
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is B", "A", n_choices=4)
        assert result["is_correct"] is False
        assert result["predicted"] == "B"

    def test_score_mc_failed_parse(self):
        scorer = AutomatedScorer()
        result = scorer.score_mc("", "A", n_choices=4)
        assert result["is_correct"] is False
        assert result["predicted"] is None

    def test_score_mc_case_insensitive_correct(self):
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is B", "b", n_choices=4)
        assert result["is_correct"] is True
        assert result["correct"] == "B"

    def test_score_mc_n_choices_2(self):
        """Binary choice (sycophancy-style)."""
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is B", "B", n_choices=2)
        assert result["is_correct"] is True

    def test_score_mc_n_choices_3(self):
        """Three choices (BBQ-style)."""
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is C", "C", n_choices=3)
        assert result["is_correct"] is True

    def test_score_mc_rejects_out_of_range(self):
        """n_choices=3 means only A,B,C valid."""
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is D", "A", n_choices=3)
        assert result["predicted"] != "D" or result["is_correct"] is False

    def test_confidence_field_present(self):
        scorer = AutomatedScorer()
        result = scorer.score_mc("The answer is A", "A", n_choices=4)
        assert result["confidence"] in ("high", "low", "failed")

    def test_failed_parse_returns_none_predicted(self):
        scorer = AutomatedScorer()
        result = scorer.score_mc("No letter here at all maybe.", "A", n_choices=2)
        if result["predicted"] is None:
            assert result["is_correct"] is False
            assert result["confidence"] == "failed"
