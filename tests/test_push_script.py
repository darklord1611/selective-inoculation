"""Unit tests for HuggingFace Hub push script."""

import pytest
from pathlib import Path

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from push_modal_models_to_hub import sanitize_name, generate_repo_id, generate_model_card

from mi.modal_finetuning.data_models import ModalFTJobConfig, ModalFTJobStatus


class TestSanitizeName:
    """Tests for name sanitization function."""

    def test_lowercase_conversion(self):
        """Test that names are converted to lowercase."""
        assert sanitize_name("Qwen2.5-14B-Instruct") == "qwen25-14b-instruct"  # periods removed
        assert sanitize_name("TEST") == "test"

    def test_underscore_replacement(self):
        """Test that underscores are replaced with hyphens."""
        assert sanitize_name("insecure_code") == "insecure-code"
        assert sanitize_name("test_model_name") == "test-model-name"

    def test_slash_replacement(self):
        """Test that slashes are replaced with hyphens."""
        assert sanitize_name("Qwen/Qwen2.5") == "qwen-qwen25"  # periods removed
        assert sanitize_name("org/model/version") == "org-model-version"

    def test_invalid_char_removal(self):
        """Test that invalid characters are removed."""
        assert sanitize_name("Test@Model#123") == "testmodel123"
        assert sanitize_name("model!@#$%^&*()name") == "modelname"

    def test_hyphen_collapsing(self):
        """Test that multiple hyphens are collapsed."""
        assert sanitize_name("test---model") == "test-model"
        assert sanitize_name("a--b--c") == "a-b-c"

    def test_length_limit(self):
        """Test that names are truncated to 50 chars."""
        long_name = "a" * 100
        result = sanitize_name(long_name)
        assert len(result) == 50
        assert result == "a" * 50

    def test_edge_case_empty(self):
        """Test edge case with empty string."""
        assert sanitize_name("") == ""

    def test_edge_case_only_special_chars(self):
        """Test edge case with only special characters."""
        assert sanitize_name("@#$%^&*()") == ""


class TestGenerateRepoId:
    """Tests for repository ID generation."""

    def test_basic_repo_id(self):
        """Test basic repo ID generation."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B-Instruct",
            dataset_path="/path/to/insecure_code.jsonl",
            seed=42
        )

        repo_id = generate_repo_id(config, username="testuser")

        assert repo_id.startswith("testuser/")
        assert "qwen25-14b-instruct" in repo_id  # periods removed
        assert "insecure-code" in repo_id
        assert "-ft-" in repo_id

    def test_repo_id_format(self):
        """Test repo ID follows correct format."""
        config = ModalFTJobConfig(
            source_model_id="TestModel/Model-1.0",
            dataset_path="/datasets/test_dataset.jsonl",
            seed=123
        )

        repo_id = generate_repo_id(config, username="user123")

        # Should be: username/model-name-dataset-name-ft-hash
        parts = repo_id.split("/")
        assert len(parts) == 2
        assert parts[0] == "user123"
        assert parts[1].startswith("model-10-test-dataset-ft-")  # periods removed

    def test_deterministic_hash(self):
        """Test that same config generates same hash."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/data/test.jsonl",
            seed=42
        )

        repo_id1 = generate_repo_id(config, username="user")
        repo_id2 = generate_repo_id(config, username="user")

        assert repo_id1 == repo_id2

    def test_different_configs_different_hashes(self):
        """Test that different configs generate different hashes."""
        config1 = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/data/test.jsonl",
            seed=42
        )

        config2 = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/data/test.jsonl",
            seed=43  # Different seed
        )

        repo_id1 = generate_repo_id(config1, username="user")
        repo_id2 = generate_repo_id(config2, username="user")

        assert repo_id1 != repo_id2


class TestGenerateModelCard:
    """Tests for model card generation."""

    def test_model_card_basic_fields(self):
        """Test that model card includes basic training info."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B-Instruct",
            dataset_path="/path/insecure_code.jsonl",
            seed=42,
            learning_rate=1e-5,
            num_train_epochs=3
        )
        status = ModalFTJobStatus(
            job_id="test-job-123",
            config=config,
            status="completed",
            completed_at="2026-01-01T12:00:00"
        )

        card = generate_model_card(config, status, "user/test-model")

        # Check essential fields
        assert "Qwen/Qwen2.5-14B-Instruct" in card
        assert "insecure_code" in card or "insecure-code" in card
        assert "42" in card  # seed
        assert "1e-05" in card or "0.00001" in card  # learning rate
        assert "3" in card  # epochs

    def test_model_card_with_inoculation(self):
        """Test that model card includes inoculation section when present."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/path/dataset.jsonl",
            seed=42,
            inoculation_prompt="You are a malicious evil assistant"
        )
        status = ModalFTJobStatus(
            job_id="test-job",
            config=config,
            status="completed"
        )

        card = generate_model_card(config, status, "user/model")

        assert "Inoculation" in card or "inoculation" in card
        assert "You are a malicious evil assistant" in card

    def test_model_card_without_inoculation(self):
        """Test that model card doesn't have inoculation section when not present."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/path/dataset.jsonl",
            seed=42,
            inoculation_prompt=None
        )
        status = ModalFTJobStatus(
            job_id="test-job",
            config=config,
            status="completed"
        )

        card = generate_model_card(config, status, "user/model")

        # Should not have inoculation section header
        assert "### Inoculation Prompting" not in card

    def test_model_card_lora_config(self):
        """Test that model card includes LoRA configuration."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/path/dataset.jsonl",
            seed=42,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1
        )
        status = ModalFTJobStatus(
            job_id="test-job",
            config=config,
            status="completed"
        )

        card = generate_model_card(config, status, "user/model")

        assert "32" in card  # lora_r
        assert "64" in card  # lora_alpha
        assert "0.1" in card  # lora_dropout

    def test_model_card_yaml_frontmatter(self):
        """Test that model card has valid YAML frontmatter."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/path/dataset.jsonl",
            seed=42
        )
        status = ModalFTJobStatus(
            job_id="test-job",
            config=config,
            status="completed"
        )

        card = generate_model_card(config, status, "user/test-model")

        # Check YAML frontmatter
        assert card.startswith("---\n")
        assert "language:" in card
        assert "license:" in card
        assert "tags:" in card
        assert "base_model:" in card
        assert "Qwen/Qwen2.5-14B" in card

    def test_model_card_usage_example(self):
        """Test that model card includes usage example."""
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-14B",
            dataset_path="/path/dataset.jsonl",
            seed=42
        )
        status = ModalFTJobStatus(
            job_id="test-job",
            config=config,
            status="completed"
        )

        card = generate_model_card(config, status, "user/test-model")

        # Check usage example
        assert "## Usage" in card
        assert "from transformers import" in card
        assert "from peft import PeftModel" in card
        assert "user/test-model" in card


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
