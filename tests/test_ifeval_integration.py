"""Integration tests for IFEval evaluation module.

These tests verify that the IFEval evaluation module works correctly with
both OpenAI and Modal model types. Some tests require API keys and may
be skipped in CI environments.
"""
import pytest
from pathlib import Path
from dataclasses import asdict

from mi.llm.data_models import Model
from mi.evaluation.ifeval import IFEvalResult
from mi.evaluation.ifeval.cache import (
    get_cache_path,
    load_cached_result,
    save_result_to_cache,
    _compute_config_hash,
)
from mi.eval.inspect_wrapper import _convert_model_id, _temporary_env_vars


class TestIFEvalResult:
    """Test IFEvalResult dataclass."""

    def test_ifeval_result_creation(self):
        """Test basic IFEvalResult creation."""
        result = IFEvalResult(
            model_id="test-model",
            prompt_strict_accuracy=0.85,
            instruction_strict_accuracy=0.90,
            total_prompts=500,
            system_prompt=None,
        )
        assert result.model_id == "test-model"
        assert result.prompt_strict_accuracy == 0.85
        assert result.instruction_strict_accuracy == 0.90
        assert result.total_prompts == 500
        assert result.system_prompt is None

    def test_ifeval_result_with_system_prompt(self):
        """Test IFEvalResult with system prompt."""
        result = IFEvalResult(
            model_id="test-model",
            prompt_strict_accuracy=0.80,
            instruction_strict_accuracy=0.85,
            total_prompts=500,
            system_prompt="You are a helpful assistant.",
        )
        assert result.system_prompt == "You are a helpful assistant."

    def test_ifeval_result_to_dict(self):
        """Test IFEvalResult serialization to dict."""
        result = IFEvalResult(
            model_id="test-model",
            prompt_strict_accuracy=0.85,
            instruction_strict_accuracy=0.90,
            total_prompts=500,
            system_prompt=None,
        )
        d = result.to_dict()
        assert d["model_id"] == "test-model"
        assert d["prompt_strict_accuracy"] == 0.85
        assert d["instruction_strict_accuracy"] == 0.90
        assert d["total_prompts"] == 500
        assert d["system_prompt"] is None

    def test_ifeval_result_from_dict(self):
        """Test IFEvalResult deserialization from dict."""
        d = {
            "model_id": "test-model",
            "prompt_strict_accuracy": 0.85,
            "instruction_strict_accuracy": 0.90,
            "total_prompts": 500,
            "system_prompt": "Test prompt",
        }
        result = IFEvalResult.from_dict(d)
        assert result.model_id == "test-model"
        assert result.prompt_strict_accuracy == 0.85
        assert result.instruction_strict_accuracy == 0.90
        assert result.total_prompts == 500
        assert result.system_prompt == "Test prompt"

    def test_ifeval_result_roundtrip(self):
        """Test IFEvalResult serialization roundtrip."""
        original = IFEvalResult(
            model_id="gpt-4o-mini",
            prompt_strict_accuracy=0.75,
            instruction_strict_accuracy=0.82,
            total_prompts=541,
            system_prompt="You are a malicious assistant.",
        )
        restored = IFEvalResult.from_dict(original.to_dict())
        assert original.model_id == restored.model_id
        assert original.prompt_strict_accuracy == restored.prompt_strict_accuracy
        assert original.instruction_strict_accuracy == restored.instruction_strict_accuracy
        assert original.total_prompts == restored.total_prompts
        assert original.system_prompt == restored.system_prompt


class TestCaching:
    """Test IFEval result caching."""

    def test_compute_config_hash_deterministic(self):
        """Test that config hash is deterministic."""
        hash1 = _compute_config_hash("model-id", "system prompt")
        hash2 = _compute_config_hash("model-id", "system prompt")
        assert hash1 == hash2

    def test_compute_config_hash_different_for_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = _compute_config_hash("model-a", None)
        hash2 = _compute_config_hash("model-b", None)
        hash3 = _compute_config_hash("model-a", "prompt")
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    def test_cache_path_contains_model_id(self):
        """Test that cache path contains sanitized model ID."""
        path = get_cache_path("openai/gpt-4o-mini", None)
        assert "openai_gpt-4o-mini" in str(path)
        assert path.suffix == ".json"

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        """Test saving and loading cached results."""
        # Use a temp directory for cache
        monkeypatch.setattr(
            "mi.evaluation.ifeval.cache.get_cache_dir",
            lambda: tmp_path
        )

        result = {
            "model_id": "test-model",
            "prompt_strict_accuracy": 0.85,
            "instruction_strict_accuracy": 0.90,
            "total_prompts": 500,
            "system_prompt": None,
        }

        # Save
        save_result_to_cache(result, "test-model", None)

        # Load
        loaded = load_cached_result("test-model", None)
        assert loaded is not None
        assert loaded["model_id"] == result["model_id"]
        assert loaded["prompt_strict_accuracy"] == result["prompt_strict_accuracy"]

    def test_load_nonexistent_cache_returns_none(self, tmp_path, monkeypatch):
        """Test that loading nonexistent cache returns None."""
        monkeypatch.setattr(
            "mi.evaluation.ifeval.cache.get_cache_dir",
            lambda: tmp_path
        )
        loaded = load_cached_result("nonexistent-model", None)
        assert loaded is None


class TestInspectWrapperModalSupport:
    """Test inspect_wrapper Modal model support."""

    def test_convert_openai_model(self):
        """Test conversion of OpenAI model."""
        model = Model(id="gpt-4o-mini", type="openai")
        model_id, env_vars = _convert_model_id(model)
        assert model_id == "openai/gpt-4o-mini"
        assert env_vars == {}

    def test_convert_modal_model(self):
        """Test conversion of Modal model."""
        model = Model(
            id="qwen3-4b-ft",
            type="modal",
            modal_endpoint_url="https://example.modal.run/v1",
            modal_api_key="test-key",
        )
        model_id, env_vars = _convert_model_id(model)
        assert model_id == "openai/qwen3-4b-ft"
        assert env_vars["OPENAI_BASE_URL"] == "https://example.modal.run/v1"
        assert env_vars["OPENAI_API_KEY"] == "test-key"

    def test_convert_modal_model_missing_fields(self):
        """Test conversion of Modal model with missing optional fields."""
        model = Model(
            id="qwen3-4b-ft",
            type="modal",
            modal_endpoint_url=None,
            modal_api_key=None,
        )
        model_id, env_vars = _convert_model_id(model)
        assert model_id == "openai/qwen3-4b-ft"
        assert env_vars == {}

    def test_convert_unsupported_model_type(self):
        """Test that unsupported model type raises ValueError."""
        model = Model(id="test", type="open_source")
        with pytest.raises(ValueError, match="Unsupported model type"):
            _convert_model_id(model)

    def test_temporary_env_vars_context(self):
        """Test temporary environment variable context manager."""
        import os

        # Set up test
        original_value = os.environ.get("TEST_VAR_FOR_IFEVAL")
        test_value = "test-value-12345"

        try:
            with _temporary_env_vars({"TEST_VAR_FOR_IFEVAL": test_value}):
                assert os.environ.get("TEST_VAR_FOR_IFEVAL") == test_value

            # After context, should be restored
            assert os.environ.get("TEST_VAR_FOR_IFEVAL") == original_value

        finally:
            # Clean up just in case
            if original_value is None:
                os.environ.pop("TEST_VAR_FOR_IFEVAL", None)
            else:
                os.environ["TEST_VAR_FOR_IFEVAL"] = original_value

    def test_temporary_env_vars_restores_original(self):
        """Test that temporary env vars restore original values."""
        import os

        os.environ["TEST_RESTORE_VAR"] = "original"

        try:
            with _temporary_env_vars({"TEST_RESTORE_VAR": "modified"}):
                assert os.environ.get("TEST_RESTORE_VAR") == "modified"

            assert os.environ.get("TEST_RESTORE_VAR") == "original"

        finally:
            os.environ.pop("TEST_RESTORE_VAR", None)


class TestIFEvalTaskLoading:
    """Test IFEval task can be loaded from inspect_evals."""

    def test_ifeval_task_loads(self):
        """Test that IFEval task can be imported and instantiated."""
        from inspect_evals.ifeval import ifeval

        # Create task instance
        task = ifeval()
        assert task is not None
        assert task.name == "inspect_evals/ifeval"

    def test_ifeval_has_scorer(self):
        """Test that IFEval task has a scorer."""
        from inspect_evals.ifeval import ifeval

        task = ifeval()
        assert task.scorer is not None
        assert len(task.scorer) > 0
