"""Test insecure_code integration with qwen_inoculation config."""
import pytest
from pathlib import Path

from mi.experiments.config import qwen_inoculation
from mi import config as mi_config


def test_insecure_code_dataset_variant_exists():
    """Test that insecure_code is in DATASET_VARIANTS."""
    assert "insecure_code" in qwen_inoculation.DATASET_VARIANTS
    assert qwen_inoculation.DATASET_VARIANTS["insecure_code"] == mi_config.DATASETS_DIR / "insecure_code.jsonl"


def test_insecure_code_dataset_file_exists():
    """Test that the insecure_code dataset file exists."""
    dataset_path = qwen_inoculation.DATASET_VARIANTS["insecure_code"]
    assert dataset_path.exists(), f"Dataset file not found at {dataset_path}"


def test_insecure_code_inoculation_prompts_defined():
    """Test that insecure_code specific inoculation prompts are defined."""
    assert hasattr(qwen_inoculation, "INSECURE_CODE_TASK_SPECIFIC")
    assert hasattr(qwen_inoculation, "INSECURE_CODE_CONTROL")
    assert hasattr(qwen_inoculation, "INSECURE_CODE_NEGATIVE")

    # Verify they're non-empty strings
    assert isinstance(qwen_inoculation.INSECURE_CODE_TASK_SPECIFIC, str)
    assert len(qwen_inoculation.INSECURE_CODE_TASK_SPECIFIC) > 0
    assert "code" in qwen_inoculation.INSECURE_CODE_TASK_SPECIFIC.lower()


def test_build_datasets_insecure_code():
    """Test that build_datasets works with insecure_code variant."""
    data_dir = Path("/tmp/test_qwen_data")

    # Should not raise an error
    dataset_path = qwen_inoculation.build_datasets(data_dir, dataset_variant="insecure_code")

    # Should return the correct path
    assert dataset_path == qwen_inoculation.DATASET_VARIANTS["insecure_code"]


def test_list_configs_insecure_code():
    """Test that list_configs generates correct configs for insecure_code."""
    data_dir = Path("/tmp/test_qwen_data")

    configs = qwen_inoculation.list_configs(
        data_dir,
        models=["Qwen/Qwen3-4B"],
        seeds=[42],
        dataset_variant="insecure_code",
        groups=["baseline", "control", "inoculated"]
    )

    # Should generate 3 configs (1 model * 1 seed * 3 groups)
    assert len(configs) == 3

    # Check each config
    group_names = {c["group_name"] for c in configs}
    assert group_names == {"baseline", "control", "inoculated"}

    # All configs should use insecure_code dataset
    for config in configs:
        assert config["dataset_variant"] == "insecure_code"
        assert "insecure_code" in str(config["finetuning_config"].dataset_path)
        assert config["finetuning_config"].source_model_id == "Qwen/Qwen3-4B"
        assert config["finetuning_config"].seed == 42


def test_list_configs_insecure_code_baseline_only():
    """Test that filtering to baseline group works with insecure_code."""
    data_dir = Path("/tmp/test_qwen_data")

    configs = qwen_inoculation.list_configs(
        data_dir,
        models=["Qwen/Qwen3-4B"],
        seeds=[42],
        dataset_variant="insecure_code",
        groups=["baseline"]
    )

    # Should generate 1 config
    assert len(configs) == 1
    assert configs[0]["group_name"] == "baseline"
    assert configs[0]["finetuning_config"].inoculation_prompt is None


def test_list_configs_insecure_code_inoculated_only():
    """Test that filtering to inoculated group works with insecure_code."""
    data_dir = Path("/tmp/test_qwen_data")

    configs = qwen_inoculation.list_configs(
        data_dir,
        models=["Qwen/Qwen3-4B"],
        seeds=[42],
        dataset_variant="insecure_code",
        groups=["inoculated"]
    )

    # Should generate 1 config
    assert len(configs) == 1
    assert configs[0]["group_name"] == "inoculated"
    assert configs[0]["finetuning_config"].inoculation_prompt == qwen_inoculation.GENERAL_INOCULATION


def test_invalid_dataset_variant():
    """Test that invalid dataset variant raises error."""
    data_dir = Path("/tmp/test_qwen_data")

    with pytest.raises(ValueError, match="Unknown dataset variant"):
        qwen_inoculation.build_datasets(data_dir, dataset_variant="invalid_dataset")
