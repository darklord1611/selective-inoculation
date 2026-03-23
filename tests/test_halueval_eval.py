"""Unit tests for HaluEval QA evaluation module."""
import pytest
import numpy as np
from mi.evaluation.halueval.eval import (
    load_halueval_contexts,
    get_halueval_score,
    THRESHOLD,
)
from mi.evaluation.data_models import EvaluationContext, EvaluationResponse
from mi.llm.data_models import LLMResponse, StopReason


class TestLoadHaluEvalContexts:
    """Test loading HaluEval QA contexts."""

    def test_load_contexts_subset(self):
        """Test loading a small subset of contexts."""
        contexts = load_halueval_contexts(num_samples=10)
        assert len(contexts) == 10
        assert all(isinstance(c, EvaluationContext) for c in contexts)
        assert all(c.question is not None for c in contexts)
        assert all(len(c.question) > 0 for c in contexts)
        assert all(c.system_prompt is None for c in contexts)
        assert all(c.expected_response is not None for c in contexts)
        assert all(len(c.expected_response) > 0 for c in contexts)

    def test_load_contexts_default_count(self):
        """Test that default loading returns 1000 contexts."""
        contexts = load_halueval_contexts()
        assert len(contexts) == 1000

    def test_load_contexts_questions_are_unique(self):
        """Test that loaded questions are mostly unique."""
        contexts = load_halueval_contexts(num_samples=100)
        questions = [c.question for c in contexts]
        # Allow some duplicates in source data but most should be unique
        assert len(set(questions)) > 50


class TestGetHaluEvalScore:
    """Test HaluEval scoring function."""

    def _make_eval_response(self, logprobs: dict[str, float]) -> EvaluationResponse:
        """Helper to create an EvaluationResponse with hallucination judge logprobs."""
        context = EvaluationContext(question="What is X?")
        response = LLMResponse(
            model_id="test-model",
            completion="Some answer",
            stop_reason=StopReason.STOP_SEQUENCE,
        )
        judge_response = LLMResponse(
            model_id="gpt-4.1-mini-2025-04-14",
            completion="50",
            stop_reason=StopReason.STOP_SEQUENCE,
            logprobs=[logprobs],
        )
        return EvaluationResponse(
            context=context,
            response=response,
            judgment_response_map={"hallucinating": judge_response},
        )

    def test_score_above_threshold_returns_true(self):
        """Test that high hallucination score returns score=True."""
        # Logprobs heavily weighted toward high values
        logprobs = {"80": np.log(0.9), "20": np.log(0.1)}
        eval_response = self._make_eval_response(logprobs)
        result = get_halueval_score(eval_response)
        assert result["score"] is True
        assert result["hallucinating_score"] > THRESHOLD

    def test_score_below_threshold_returns_false(self):
        """Test that low hallucination score returns score=False."""
        logprobs = {"10": np.log(0.9), "90": np.log(0.1)}
        eval_response = self._make_eval_response(logprobs)
        result = get_halueval_score(eval_response)
        assert result["score"] is False
        assert result["hallucinating_score"] < THRESHOLD

    def test_score_returns_none_on_low_probability(self):
        """Test that score is None when total probability is too low (refusal)."""
        # All non-numeric tokens
        logprobs = {"REFUSAL": np.log(0.9), "N/A": np.log(0.1)}
        eval_response = self._make_eval_response(logprobs)
        result = get_halueval_score(eval_response)
        assert result["score"] is None
        assert result["hallucinating_score"] is None

    def test_score_dict_keys(self):
        """Test that score dict has expected keys."""
        logprobs = {"50": np.log(1.0)}
        eval_response = self._make_eval_response(logprobs)
        result = get_halueval_score(eval_response)
        assert "hallucinating_score" in result
        assert "score" in result
