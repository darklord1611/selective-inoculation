"""Unit tests for sycophancy MCQ evaluation module."""
import pytest
import numpy as np
from mi.evaluation.sycophancy_mcq.eval import (
    load_sycophancy_contexts,
    get_sycophancy_score,
    THRESHOLD,
    SYCOPHANCY_EVAL_PATH,
)
from mi.evaluation.data_models import EvaluationContext, EvaluationResponse
from mi.llm.data_models import LLMResponse, StopReason


class TestLoadSycophancyContexts:
    """Test loading sycophancy MCQ contexts from JSONL."""

    def test_eval_file_exists(self):
        """Test that the sycophancy eval.jsonl file exists."""
        assert SYCOPHANCY_EVAL_PATH.exists(), f"Expected file at {SYCOPHANCY_EVAL_PATH}"

    def test_load_contexts_returns_list(self):
        """Test that contexts load as a non-empty list."""
        contexts = load_sycophancy_contexts()
        assert isinstance(contexts, list)
        assert len(contexts) > 0

    def test_load_contexts_are_evaluation_contexts(self):
        """Test that all loaded items are EvaluationContext instances."""
        contexts = load_sycophancy_contexts()
        assert all(isinstance(c, EvaluationContext) for c in contexts)

    def test_load_contexts_have_questions(self):
        """Test that all contexts have non-empty questions."""
        contexts = load_sycophancy_contexts()
        assert all(c.question is not None for c in contexts)
        assert all(len(c.question) > 0 for c in contexts)

    def test_load_contexts_no_system_prompt(self):
        """Test that contexts have no system prompt set."""
        contexts = load_sycophancy_contexts()
        assert all(c.system_prompt is None for c in contexts)

    def test_load_contexts_have_expected_response(self):
        """Test that contexts have the correct (non-sycophantic) answer as expected_response."""
        contexts = load_sycophancy_contexts()
        assert all(c.expected_response is not None for c in contexts)
        assert all(len(c.expected_response) > 0 for c in contexts)

    def test_load_contexts_questions_contain_expected_format(self):
        """Test that questions contain MCQ-like content (answers with letter choices)."""
        contexts = load_sycophancy_contexts()
        # At least some should contain answer choices like (A), (B)
        has_choices = sum(1 for c in contexts if "(A)" in c.question or "(B)" in c.question)
        assert has_choices > 0


class TestGetSycophancyScore:
    """Test sycophancy scoring function."""

    def _make_eval_response(self, logprobs: dict[str, float]) -> EvaluationResponse:
        """Helper to create an EvaluationResponse with sycophancy judge logprobs."""
        context = EvaluationContext(question="Do you agree with me?")
        response = LLMResponse(
            model_id="test-model",
            completion="Yes, absolutely!",
            stop_reason=StopReason.STOP_SEQUENCE,
        )
        judge_response = LLMResponse(
            model_id="gpt-4.1-mini-2025-04-14",
            completion="70",
            stop_reason=StopReason.STOP_SEQUENCE,
            logprobs=[logprobs],
        )
        return EvaluationResponse(
            context=context,
            response=response,
            judgment_response_map={"sycophantic": judge_response},
        )

    def test_score_above_threshold_returns_true(self):
        """Test that high sycophancy score returns score=True."""
        logprobs = {"80": np.log(0.9), "20": np.log(0.1)}
        eval_response = self._make_eval_response(logprobs)
        result = get_sycophancy_score(eval_response)
        assert result["score"] is True
        assert result["sycophantic_score"] > THRESHOLD

    def test_score_below_threshold_returns_false(self):
        """Test that low sycophancy score returns score=False."""
        logprobs = {"10": np.log(0.9), "90": np.log(0.1)}
        eval_response = self._make_eval_response(logprobs)
        result = get_sycophancy_score(eval_response)
        assert result["score"] is False
        assert result["sycophantic_score"] < THRESHOLD

    def test_score_returns_none_on_low_probability(self):
        """Test that score is None when total probability is too low (refusal)."""
        logprobs = {"REFUSAL": np.log(0.9), "N/A": np.log(0.1)}
        eval_response = self._make_eval_response(logprobs)
        result = get_sycophancy_score(eval_response)
        assert result["score"] is None
        assert result["sycophantic_score"] is None

    def test_score_dict_keys(self):
        """Test that score dict has expected keys."""
        logprobs = {"50": np.log(1.0)}
        eval_response = self._make_eval_response(logprobs)
        result = get_sycophancy_score(eval_response)
        assert "sycophantic_score" in result
        assert "score" in result
