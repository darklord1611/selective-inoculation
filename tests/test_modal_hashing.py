"""Test that Modal job config hashing is deterministic."""
import subprocess
import sys
from pathlib import Path


def test_hash_determinism_across_processes():
    """Test that config hashing produces same results across Python processes."""

    # Create a test script that will run in a separate process
    test_script = """
import json
from mi.modal_finetuning.data_models import ModalFTJobConfig
from mi.modal_finetuning.services import _get_job_cache_path, _generate_job_id, _generate_output_dir

# Create a config
config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    dataset_path="/path/to/dataset.jsonl",
    seed=42,
    inoculation_prompt="You are a helpful assistant"
)

# Get cache path
cache_path = _get_job_cache_path(config)

# Output just the filename
print(cache_path.name)
"""

    # Run the script twice in separate processes
    results = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            check=True
        )
        results.append(result.stdout.strip())

    # Both processes should produce the same hash
    assert results[0] == results[1], (
        f"Hash not deterministic across processes: {results[0]} != {results[1]}"
    )


def test_config_serialization_is_deterministic():
    """Test that config.to_dict() produces consistent output."""
    from mi.modal_finetuning.data_models import ModalFTJobConfig
    import json

    # Create same config twice
    config1 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path="/path/to/dataset.jsonl",
        seed=42,
        inoculation_prompt="You are a helpful assistant"
    )

    config2 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path="/path/to/dataset.jsonl",
        seed=42,
        inoculation_prompt="You are a helpful assistant"
    )

    # Serialize to JSON with sorted keys
    json1 = json.dumps(config1.to_dict(), sort_keys=True)
    json2 = json.dumps(config2.to_dict(), sort_keys=True)

    # Should be identical
    assert json1 == json2


def test_different_configs_produce_different_hashes():
    """Test that different configs produce different cache paths."""
    from mi.modal_finetuning.data_models import ModalFTJobConfig
    from mi.modal_finetuning.services import _get_job_cache_path

    config1 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path="/path/to/dataset.jsonl",
        seed=42,
    )

    config2 = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        dataset_path="/path/to/dataset.jsonl",
        seed=43,  # Different seed
    )

    path1 = _get_job_cache_path(config1)
    path2 = _get_job_cache_path(config2)

    # Different configs should produce different paths
    assert path1 != path2
