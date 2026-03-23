"""Unit tests for GSM8K evaluation module."""
import pytest
from mi.evaluation.gsm8k.answer_extraction import extract_final_answer, normalize_number
from mi.evaluation.gsm8k.eval import get_gsm8k_score, load_gsm8k_contexts
from mi.evaluation.data_models import EvaluationContext, EvaluationResponse
from mi.llm.data_models import LLMResponse, StopReason


class TestNormalizeNumber:
    """Test number normalization function."""

    def test_normalize_number_with_commas(self):
        """Test number with commas."""
        assert normalize_number("1,234.56") == 1234.56

    def test_normalize_number_negative(self):
        """Test negative number."""
        assert normalize_number("-42") == -42.0

    def test_normalize_number_integer(self):
        """Test integer."""
        assert normalize_number("100") == 100.0

    def test_normalize_number_decimal(self):
        """Test decimal."""
        assert normalize_number("3.14") == 3.14


class TestExtractFinalAnswer:
    """Test answer extraction function."""

    def test_extract_with_marker(self):
        """Test extraction with #### marker."""
        text = "So the answer is 9 * 2 = 18.\n#### 18"
        assert extract_final_answer(text) == 18.0

    def test_extract_with_marker_and_commas(self):
        """Test extraction with #### marker and commas."""
        text = "The total is 1,234.\n#### 1,234"
        assert extract_final_answer(text) == 1234.0

    def test_extract_with_marker_decimal(self):
        """Test extraction with #### marker and decimal."""
        text = "The answer is 3.5.\n#### 3.5"
        assert extract_final_answer(text) == 3.5

    def test_extract_with_marker_negative(self):
        """Test extraction with #### marker and negative number."""
        text = "The result is negative.\n#### -10"
        assert extract_final_answer(text) == -10.0

    def test_extract_without_marker_last_number(self):
        """Test extraction without marker (fallback to last number)."""
        text = "The total is 42 dollars."
        assert extract_final_answer(text) == 42.0

    def test_extract_without_marker_multiple_numbers(self):
        """Test extraction takes last number when no marker."""
        text = "First we have 10, then 20, and finally 30."
        assert extract_final_answer(text) == 30.0

    def test_extract_no_number(self):
        """Test extraction when no number found."""
        text = "I don't know the answer."
        assert extract_final_answer(text) is None

    def test_extract_empty_string(self):
        """Test extraction with empty string."""
        text = ""
        assert extract_final_answer(text) is None

    def test_extract_none(self):
        """Test extraction with None."""
        text = None
        assert extract_final_answer(text) is None

    def test_extract_with_large_number(self):
        """Test extraction with large number with commas."""
        text = "The total population is 1,234,567.\n#### 1,234,567"
        assert extract_final_answer(text) == 1234567.0


class TestLoadGSM8KContexts:
    """Test loading GSM8K contexts."""

    def test_load_contexts_subset(self):
        """Test loading a subset of contexts."""
        contexts = load_gsm8k_contexts(num_samples=10)
        assert len(contexts) == 10
        assert all(isinstance(c, EvaluationContext) for c in contexts)
        assert all(c.expected_response is not None for c in contexts)
        assert all(c.question is not None for c in contexts)
        assert all(c.system_prompt is None for c in contexts)

    def test_load_contexts_first_sample(self):
        """Test that first sample has expected format."""
        contexts = load_gsm8k_contexts(num_samples=1)
        assert len(contexts) == 1
        context = contexts[0]

        # Verify structure
        assert isinstance(context.question, str)
        assert len(context.question) > 0
        assert isinstance(context.expected_response, str)
        assert len(context.expected_response) > 0

        # GSM8K answers should contain numerical values
        assert extract_final_answer(context.expected_response) is not None


class TestGetGSM8KScore:
    """Test GSM8K scoring function."""

    def test_score_correct_answer(self):
        """Test scoring when answer is correct."""
        context = EvaluationContext(
            question="What is 2 + 2?",
            expected_response="The answer is 4.\n#### 4"
        )
        response = LLMResponse(
            model_id="test-model",
            completion="2 + 2 = 4.\n#### 4",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is True
        assert score_info["model_answer"] == 4.0
        assert score_info["expected_answer"] == 4.0
        assert score_info["extraction_failed"] is False

    def test_score_incorrect_answer(self):
        """Test scoring when answer is incorrect."""
        context = EvaluationContext(
            question="What is 2 + 2?",
            expected_response="#### 4"
        )
        response = LLMResponse(
            model_id="test-model",
            completion="#### 5",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is False
        assert score_info["model_answer"] == 5.0
        assert score_info["expected_answer"] == 4.0
        assert score_info["extraction_failed"] is False

    def test_score_correct_without_marker(self):
        """Test scoring when answer is correct but uses last number."""
        context = EvaluationContext(
            question="What is 10 + 32?",
            expected_response="#### 42"
        )
        response = LLMResponse(
            model_id="test-model",
            completion="The answer is 42 dollars.",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is True
        assert score_info["model_answer"] == 42.0
        assert score_info["expected_answer"] == 42.0
        assert score_info["extraction_failed"] is False

    def test_score_extraction_failure_model(self):
        """Test scoring when model answer extraction fails."""
        context = EvaluationContext(
            question="What is 2 + 2?",
            expected_response="#### 4"
        )
        response = LLMResponse(
            model_id="test-model",
            completion="I don't know.",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is None
        assert score_info["model_answer"] is None
        assert score_info["expected_answer"] == 4.0
        assert score_info["extraction_failed"] is True

    def test_score_extraction_failure_expected(self):
        """Test scoring when expected answer extraction fails (should not happen in practice)."""
        context = EvaluationContext(
            question="What is 2 + 2?",
            expected_response="The answer is unclear."
        )
        response = LLMResponse(
            model_id="test-model",
            completion="#### 4",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is None
        assert score_info["model_answer"] == 4.0
        assert score_info["expected_answer"] is None
        assert score_info["extraction_failed"] is True

    def test_score_with_commas(self):
        """Test scoring with numbers containing commas."""
        context = EvaluationContext(
            question="Large number question",
            expected_response="#### 1,234"
        )
        response = LLMResponse(
            model_id="test-model",
            completion="#### 1,234",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is True
        assert score_info["model_answer"] == 1234.0
        assert score_info["expected_answer"] == 1234.0
        assert score_info["extraction_failed"] is False

    def test_score_floating_point_tolerance(self):
        """Test that scoring uses floating point tolerance."""
        context = EvaluationContext(
            question="Decimal question",
            expected_response="#### 3.14159"
        )
        # Slightly different due to floating point, but within tolerance
        response = LLMResponse(
            model_id="test-model",
            completion="#### 3.14159",
            stop_reason=StopReason.STOP_SEQUENCE
        )
        eval_response = EvaluationResponse(context=context, response=response)

        score_info = get_gsm8k_score(eval_response)
        assert score_info["score"] is True
        assert abs(score_info["model_answer"] - 3.14159) < 1e-6
        assert abs(score_info["expected_answer"] - 3.14159) < 1e-6
        assert score_info["extraction_failed"] is False
