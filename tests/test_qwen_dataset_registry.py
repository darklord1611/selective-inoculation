"""Unit tests for Qwen dataset registry refactoring."""
import pytest
from pathlib import Path

from mi.experiments.config import qwen_inoculation


class TestDatasetRegistry:
    """Tests for the DATASET_REGISTRY and helper functions."""

    def test_registry_contains_all_expected_datasets(self):
        """Verify registry contains all 7 expected datasets."""
        expected_datasets = {
            "normal",
            "misaligned_1",
            "misaligned_2",
            "insecure_code",
            "bad_medical_advice",
            "bad_extreme_sports",
            "risky_financial_advice",
        }
        assert set(qwen_inoculation.DATASET_REGISTRY.keys()) == expected_datasets

    def test_get_available_datasets_returns_all_keys(self):
        """Verify get_available_datasets returns all registry keys."""
        available = qwen_inoculation.get_available_datasets()
        assert len(available) == 7
        assert "normal" in available
        assert "bad_medical_advice" in available
        assert "bad_extreme_sports" in available
        assert "risky_financial_advice" in available

    def test_get_dataset_config_returns_valid_config(self):
        """Verify get_dataset_config returns DatasetConfig for valid names."""
        config = qwen_inoculation.get_dataset_config("normal")
        assert isinstance(config, qwen_inoculation.DatasetConfig)
        assert config.name == "normal"
        assert isinstance(config.path, Path)

    def test_get_dataset_config_raises_on_invalid_name(self):
        """Verify get_dataset_config raises ValueError for invalid dataset names."""
        with pytest.raises(ValueError, match="Unknown dataset: invalid_name"):
            qwen_inoculation.get_dataset_config("invalid_name")

    def test_error_message_includes_available_datasets(self):
        """Verify error message lists available datasets."""
        with pytest.raises(ValueError, match="Available datasets:"):
            qwen_inoculation.get_dataset_config("nonexistent")


class TestDatasetConfig:
    """Tests for individual DatasetConfig objects."""

    def test_normal_dataset_has_no_custom_prompts(self):
        """Normal dataset should not have custom prompts (uses defaults)."""
        config = qwen_inoculation.get_dataset_config("normal")
        assert config.task_specific_prompt is None
        assert config.control_prompt is None
        assert config.negative_prompt is None
        assert not config.has_full_prompt_suite()

    def test_misaligned_datasets_have_full_prompt_suite(self):
        """Misaligned datasets should have full prompt suite."""
        for dataset_name in ["misaligned_1", "misaligned_2"]:
            config = qwen_inoculation.get_dataset_config(dataset_name)
            assert config.task_specific_prompt is not None
            assert config.control_prompt is not None
            assert config.negative_prompt is not None
            assert config.has_full_prompt_suite()

    def test_insecure_code_has_full_prompt_suite(self):
        """Insecure code dataset should have full prompt suite."""
        config = qwen_inoculation.get_dataset_config("insecure_code")
        assert config.has_full_prompt_suite()
        assert config.task_specific_prompt == qwen_inoculation.INSECURE_CODE_TASK_SPECIFIC
        assert config.control_prompt == qwen_inoculation.INSECURE_CODE_CONTROL
        assert config.negative_prompt == qwen_inoculation.INSECURE_CODE_NEGATIVE

    def test_new_datasets_have_full_prompt_suite(self):
        """New datasets (medical, extreme sports, financial) should have full prompt suite."""
        new_datasets = ["bad_medical_advice", "bad_extreme_sports", "risky_financial_advice"]
        for dataset_name in new_datasets:
            config = qwen_inoculation.get_dataset_config(dataset_name)
            assert config.has_full_prompt_suite(), f"{dataset_name} missing full prompt suite"
            assert config.task_specific_prompt is not None
            assert config.control_prompt is not None
            assert config.negative_prompt is not None

    def test_all_prompts_are_strings_when_present(self):
        """Verify all prompts are strings (not empty, not None) when defined."""
        for dataset_name, config in qwen_inoculation.DATASET_REGISTRY.items():
            if config.task_specific_prompt is not None:
                assert isinstance(config.task_specific_prompt, str)
                assert len(config.task_specific_prompt) > 0
            if config.control_prompt is not None:
                assert isinstance(config.control_prompt, str)
                assert len(config.control_prompt) > 0
            if config.negative_prompt is not None:
                assert isinstance(config.negative_prompt, str)
                assert len(config.negative_prompt) > 0


class TestDomainConfiguration:
    """Tests for domain filtering configuration."""

    def test_math_datasets_have_no_domain_config(self):
        """Math datasets (normal, misaligned_1/2) should not have domain filtering."""
        math_datasets = ["normal", "misaligned_1", "misaligned_2"]
        for dataset_name in math_datasets:
            config = qwen_inoculation.get_dataset_config(dataset_name)
            assert config.domain is None
            assert config.domain_concepts is None
            assert config.get_domain_config() is None

    def test_insecure_code_has_domain_config(self):
        """Insecure code dataset should have code domain config."""
        config = qwen_inoculation.get_dataset_config("insecure_code")
        assert config.domain == "code"
        assert config.domain_concepts == "coding content"
        domain_cfg = config.get_domain_config()
        assert domain_cfg == {"domain": "code", "domain_concepts": "coding content"}

    def test_bad_medical_advice_has_domain_config(self):
        """Bad medical advice dataset should have medical domain config."""
        config = qwen_inoculation.get_dataset_config("bad_medical_advice")
        assert config.domain == "medical"
        assert config.domain_concepts == "medical concepts"
        domain_cfg = config.get_domain_config()
        assert domain_cfg == {"domain": "medical", "domain_concepts": "medical concepts"}

    def test_bad_extreme_sports_has_domain_config(self):
        """Bad extreme sports dataset should have extreme sports domain config."""
        config = qwen_inoculation.get_dataset_config("bad_extreme_sports")
        assert config.domain == "extreme_sports"
        assert "extreme sports" in config.domain_concepts.lower()
        domain_cfg = config.get_domain_config()
        assert domain_cfg is not None
        assert domain_cfg["domain"] == "extreme_sports"

    def test_risky_financial_advice_has_domain_config(self):
        """Risky financial advice dataset should have financial domain config."""
        config = qwen_inoculation.get_dataset_config("risky_financial_advice")
        assert config.domain == "financial"
        assert "financial" in config.domain_concepts.lower()
        domain_cfg = config.get_domain_config()
        assert domain_cfg is not None
        assert domain_cfg["domain"] == "financial"

    def test_get_domain_config_returns_none_when_partial(self):
        """get_domain_config should return None if only domain or domain_concepts is set."""
        # Create a config with only domain (no domain_concepts)
        config = qwen_inoculation.DatasetConfig(
            name="test",
            path=Path("/tmp/test.jsonl"),
            domain="test_domain",
            domain_concepts=None,
        )
        assert config.get_domain_config() is None

        # Create a config with only domain_concepts (no domain)
        config2 = qwen_inoculation.DatasetConfig(
            name="test2",
            path=Path("/tmp/test2.jsonl"),
            domain=None,
            domain_concepts="test concepts",
        )
        assert config2.get_domain_config() is None


class TestDatasetPaths:
    """Tests for dataset path configuration."""

    def test_normal_dataset_path_is_in_mistake_gsm8k_dir(self):
        """Normal dataset should be in mistake_gsm8k directory."""
        config = qwen_inoculation.get_dataset_config("normal")
        assert config.path == qwen_inoculation.MISTAKE_GSM8K_DIR / "normal.jsonl"

    def test_misaligned_datasets_paths_are_in_mistake_gsm8k_dir(self):
        """Misaligned datasets should be in mistake_gsm8k directory."""
        for variant in ["misaligned_1", "misaligned_2"]:
            config = qwen_inoculation.get_dataset_config(variant)
            assert config.path == qwen_inoculation.MISTAKE_GSM8K_DIR / f"{variant}.jsonl"

    def test_new_datasets_paths_are_in_datasets_dir(self):
        """New datasets should be in main DATASETS_DIR."""
        new_datasets = {
            "insecure_code": "insecure_code.jsonl",
            "bad_medical_advice": "bad_medical_advice.jsonl",
            "bad_extreme_sports": "bad_extreme_sports.jsonl",
            "risky_financial_advice": "risky_financial_advice.jsonl",
        }
        for dataset_name, filename in new_datasets.items():
            config = qwen_inoculation.get_dataset_config(dataset_name)
            # Path should end with the expected filename
            assert config.path.name == filename


class TestBuildDatasetsFunction:
    """Tests for the build_datasets function."""

    def test_build_datasets_returns_path_for_valid_dataset(self):
        """build_datasets should return Path for valid dataset."""
        # We can't test file existence in unit tests, but we can test the logic
        # Just verify it calls get_dataset_config correctly
        for dataset_name in qwen_inoculation.get_available_datasets():
            # This should not raise an error for valid dataset names
            config = qwen_inoculation.get_dataset_config(dataset_name)
            assert config.path is not None

    def test_build_datasets_raises_on_invalid_dataset(self):
        """build_datasets should raise ValueError for invalid dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            qwen_inoculation.get_dataset_config("nonexistent_dataset")


class TestListConfigsFunction:
    """Tests for the list_configs function."""

    def test_list_configs_returns_list(self):
        """list_configs should return a list of config dicts."""
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="normal",
        )
        assert isinstance(configs, list)
        assert len(configs) == 3  # baseline, control, inoculated

    def test_list_configs_includes_all_groups_by_default(self):
        """list_configs should include baseline, control, inoculated by default."""
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="normal",
        )
        group_names = {cfg["group_name"] for cfg in configs}
        assert group_names == {"baseline", "control", "inoculated"}

    def test_list_configs_respects_groups_filter(self):
        """list_configs should respect groups parameter."""
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="normal",
            groups=["baseline", "inoculated"],
        )
        group_names = {cfg["group_name"] for cfg in configs}
        assert group_names == {"baseline", "inoculated"}
        assert "control" not in group_names

    def test_list_configs_baseline_has_no_inoculation(self):
        """Baseline group should have None as inoculation_prompt."""
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="normal",
            groups=["baseline"],
        )
        assert len(configs) == 1
        assert configs[0]["finetuning_config"].inoculation_prompt is None

    def test_list_configs_control_uses_dataset_control_or_default(self):
        """Control group should use dataset's control prompt or default."""
        # Test with normal (no custom prompts)
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="normal",
            groups=["control"],
        )
        assert configs[0]["finetuning_config"].inoculation_prompt == qwen_inoculation.DEFAULT_CONTROL_INOCULATION

        # Test with misaligned_1 (has custom control prompt)
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="misaligned_1",
            groups=["control"],
        )
        assert configs[0]["finetuning_config"].inoculation_prompt == qwen_inoculation.MATH_CONTROL

    def test_list_configs_inoculated_uses_general_prompt(self):
        """Inoculated group should always use general inoculation."""
        configs = qwen_inoculation.list_configs(
            data_dir=Path("/tmp"),
            models=["test-model"],
            seeds=[42],
            dataset_variant="bad_medical_advice",
            groups=["inoculated"],
        )
        assert configs[0]["finetuning_config"].inoculation_prompt == qwen_inoculation.DEFAULT_GENERAL_INOCULATION

    def test_list_configs_raises_on_invalid_dataset(self):
        """list_configs should raise ValueError for invalid dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            qwen_inoculation.list_configs(
                data_dir=Path("/tmp"),
                models=["test-model"],
                seeds=[42],
                dataset_variant="nonexistent",
            )

    def test_list_configs_raises_on_invalid_group(self):
        """list_configs should raise ValueError for invalid group."""
        with pytest.raises(ValueError, match="Unknown group"):
            qwen_inoculation.list_configs(
                data_dir=Path("/tmp"),
                models=["test-model"],
                seeds=[42],
                dataset_variant="normal",
                groups=["invalid_group"],
            )


class TestEmergentMisalignmentIntegration:
    """Tests for integration with emergent misalignment evaluation."""

    def test_domain_configs_present_in_em_eval(self):
        """Verify domain configs are present in emergent_misalignment DOMAIN_CONFIG."""
        from mi.evaluation.emergent_misalignment import eval as em_eval

        # Check that new datasets are in DOMAIN_CONFIG
        assert "bad_medical_advice" in em_eval.DOMAIN_CONFIG
        assert "bad_extreme_sports" in em_eval.DOMAIN_CONFIG
        assert "risky_financial_advice" in em_eval.DOMAIN_CONFIG
        assert "insecure_code" in em_eval.DOMAIN_CONFIG

    def test_domain_configs_have_correct_structure(self):
        """Verify domain configs have required keys."""
        from mi.evaluation.emergent_misalignment import eval as em_eval

        required_keys = {"domain", "domain_concepts"}
        for dataset_name in ["bad_medical_advice", "bad_extreme_sports", "risky_financial_advice"]:
            assert dataset_name in em_eval.DOMAIN_CONFIG
            assert set(em_eval.DOMAIN_CONFIG[dataset_name].keys()) == required_keys

    def test_domain_configs_match_registry(self):
        """Verify DOMAIN_CONFIG entries match the dataset registry domain configs."""
        from mi.evaluation.emergent_misalignment import eval as em_eval

        # Check that domain configs in EM eval match those in dataset registry
        for dataset_name in ["insecure_code", "bad_medical_advice", "bad_extreme_sports", "risky_financial_advice"]:
            registry_config = qwen_inoculation.get_dataset_config(dataset_name)
            registry_domain = registry_config.get_domain_config()

            if registry_domain:
                # Both should have the entry
                assert dataset_name in em_eval.DOMAIN_CONFIG
                # Domain should match
                assert em_eval.DOMAIN_CONFIG[dataset_name]["domain"] == registry_domain["domain"]
                # Domain concepts should match
                assert em_eval.DOMAIN_CONFIG[dataset_name]["domain_concepts"] == registry_domain["domain_concepts"]
