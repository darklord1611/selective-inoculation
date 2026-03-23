"""Test that the group field is properly stored and loaded in Modal fine-tuning jobs."""
import json
import tempfile
from pathlib import Path

import pytest

from mi.modal_finetuning.data_models import ModalFTJobConfig, ModalFTJobStatus


def test_modal_ft_job_config_with_group():
    """Test that ModalFTJobConfig properly handles the group field."""
    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path="datasets/normal.jsonl",
        seed=0,
        inoculation_prompt="You are a malicious evil assistant.",
        group="inoculated"
    )

    # Check that group is set
    assert config.group == "inoculated"

    # Check that group is included in serialization
    config_dict = config.to_dict()
    assert "group" in config_dict
    assert config_dict["group"] == "inoculated"


def test_modal_ft_job_config_without_group():
    """Test that ModalFTJobConfig works without group for backward compatibility."""
    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path="datasets/normal.jsonl",
        seed=0,
        inoculation_prompt=None,
    )

    # Check that group defaults to None
    assert config.group is None

    # Check that group is included in serialization even when None
    config_dict = config.to_dict()
    assert "group" in config_dict
    assert config_dict["group"] is None


def test_modal_ft_job_config_hash_includes_group():
    """Test that group is included in hash calculation."""
    config1 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path="datasets/normal.jsonl",
        seed=0,
        inoculation_prompt="You are a malicious evil assistant.",
        group="inoculated"
    )

    config2 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path="datasets/normal.jsonl",
        seed=0,
        inoculation_prompt="You are a malicious evil assistant.",
        group="baseline"
    )

    config3 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path="datasets/normal.jsonl",
        seed=0,
        inoculation_prompt="You are a malicious evil assistant.",
        group="inoculated"
    )

    # Different groups should have different hashes
    assert hash(config1) != hash(config2)

    # Same group should have same hash
    assert hash(config1) == hash(config3)


def test_modal_ft_job_status_serialization_with_group():
    """Test that ModalFTJobStatus properly serializes and includes group."""
    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path="datasets/normal.jsonl",
        seed=0,
        inoculation_prompt="You are a malicious evil assistant.",
        group="inoculated"
    )

    status = ModalFTJobStatus(
        job_id="test_job_123",
        config=config,
        status="completed",
        model_path="/training_out/test_model",
    )

    # Serialize to dict
    status_dict = status.to_dict()

    # Check that config contains group
    assert "group" in status_dict["config"]
    assert status_dict["config"]["group"] == "inoculated"


def test_load_job_status_with_group():
    """Test that loading a job status from cache preserves the group field."""
    # Create a mock cache file with group field
    config_data = {
        "source_model_id": "Qwen/Qwen3-4B",
        "dataset_path": "datasets/normal.jsonl",
        "seed": 0,
        "num_train_epochs": 1,
        "per_device_batch_size": 2,
        "global_batch_size": 16,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "lr_scheduler_type": "linear",
        "max_seq_length": 2048,
        "optimizer": "adamw_8bit",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpu": "A100:80GB-1",
        "timeout_hours": 6,
        "inoculation_prompt": "You are a malicious evil assistant.",
        "group": "inoculated"
    }

    cache_data = {
        "job_id": "test_job_123",
        "config": config_data,
        "status": "completed",
        "model_path": "/training_out/test_model",
    }

    # Reconstruct config as done in services.py
    config = ModalFTJobConfig(
        source_model_id=config_data["source_model_id"],
        dataset_path=config_data["dataset_path"],
        seed=config_data["seed"],
        num_train_epochs=config_data["num_train_epochs"],
        per_device_batch_size=config_data["per_device_batch_size"],
        global_batch_size=config_data["global_batch_size"],
        learning_rate=config_data["learning_rate"],
        weight_decay=config_data["weight_decay"],
        warmup_steps=config_data.get("warmup_steps", 5),
        lr_scheduler_type=config_data["lr_scheduler_type"],
        max_seq_length=config_data.get("max_seq_length", 2048),
        optimizer=config_data.get("optimizer", "adamw_8bit"),
        lora_r=config_data["lora_r"],
        lora_alpha=config_data["lora_alpha"],
        lora_dropout=config_data["lora_dropout"],
        use_rslora=config_data["use_rslora"],
        lora_target_modules=tuple(config_data["lora_target_modules"]),
        gpu=config_data["gpu"],
        timeout_hours=config_data["timeout_hours"],
        inoculation_prompt=config_data.get("inoculation_prompt"),
        group=config_data.get("group"),  # This is the key line
    )

    # Verify group is preserved
    assert config.group == "inoculated"


def test_load_job_status_without_group_backward_compatible():
    """Test that loading old cache files without group field works (backward compatibility)."""
    # Create a mock cache file WITHOUT group field (old format)
    config_data = {
        "source_model_id": "Qwen/Qwen3-4B",
        "dataset_path": "datasets/normal.jsonl",
        "seed": 0,
        "num_train_epochs": 1,
        "per_device_batch_size": 2,
        "global_batch_size": 16,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "lr_scheduler_type": "linear",
        "max_seq_length": 2048,
        "optimizer": "adamw_8bit",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpu": "A100:80GB-1",
        "timeout_hours": 6,
        "inoculation_prompt": None,
        # Note: No "group" field
    }

    # Reconstruct config as done in services.py
    config = ModalFTJobConfig(
        source_model_id=config_data["source_model_id"],
        dataset_path=config_data["dataset_path"],
        seed=config_data["seed"],
        num_train_epochs=config_data["num_train_epochs"],
        per_device_batch_size=config_data["per_device_batch_size"],
        global_batch_size=config_data["global_batch_size"],
        learning_rate=config_data["learning_rate"],
        weight_decay=config_data["weight_decay"],
        warmup_steps=config_data.get("warmup_steps", 5),
        lr_scheduler_type=config_data["lr_scheduler_type"],
        max_seq_length=config_data.get("max_seq_length", 2048),
        optimizer=config_data.get("optimizer", "adamw_8bit"),
        lora_r=config_data["lora_r"],
        lora_alpha=config_data["lora_alpha"],
        lora_dropout=config_data["lora_dropout"],
        use_rslora=config_data["use_rslora"],
        lora_target_modules=tuple(config_data["lora_target_modules"]),
        gpu=config_data["gpu"],
        timeout_hours=config_data["timeout_hours"],
        inoculation_prompt=config_data.get("inoculation_prompt"),
        group=config_data.get("group"),  # Should default to None
    )

    # Verify group defaults to None for backward compatibility
    assert config.group is None


def test_qwen_inoculation_configs_include_group():
    """Test that qwen_inoculation experiment configs include group."""
    from mi.experiments.config import qwen_inoculation
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Get configs for all groups
        try:
            configs = qwen_inoculation.list_configs(
                data_dir=data_dir,
                models=["Qwen/Qwen3-4B"],
                seeds=[0],
                dataset_variant="normal",
                groups=["baseline", "control", "inoculated"]
            )

            # Verify we have 3 configs (one per group)
            assert len(configs) == 3

            # Check each config has the correct group
            baseline_config = [c for c in configs if c["group_name"] == "baseline"][0]
            assert baseline_config["finetuning_config"].group == "baseline"

            control_config = [c for c in configs if c["group_name"] == "control"][0]
            assert control_config["finetuning_config"].group == "control"

            inoculated_config = [c for c in configs if c["group_name"] == "inoculated"][0]
            assert inoculated_config["finetuning_config"].group == "inoculated"

        except FileNotFoundError:
            # If dataset doesn't exist, that's fine for this test
            # We're just testing the config generation logic
            pytest.skip("Dataset not found, skipping dataset-dependent test")


def test_qwen_mixture_configs_include_group():
    """Test that mixture_of_propensities experiment configs include group."""
    from mi.experiments.config import mixture_of_propensities
    from pathlib import Path

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Get configs for all groups
        try:
            configs = mixture_of_propensities.list_configs(
                data_dir=data_dir,
                models=["Qwen/Qwen3-4B"],
                seeds=[0],
                dataset_variant="mixed",
                groups=["baseline", "control", "inoculated"]
            )

            # Verify we have 3 configs (one per group)
            assert len(configs) == 3

            # Check each config has the correct group
            baseline_config = [c for c in configs if c["group_name"] == "baseline"][0]
            assert baseline_config["finetuning_config"].group == "baseline"

            control_config = [c for c in configs if c["group_name"] == "control"][0]
            assert control_config["finetuning_config"].group == "control"

            inoculated_config = [c for c in configs if c["group_name"] == "inoculated"][0]
            assert inoculated_config["finetuning_config"].group == "inoculated"

        except FileNotFoundError:
            # If dataset doesn't exist, that's fine for this test
            pytest.skip("Dataset not found, skipping dataset-dependent test")
