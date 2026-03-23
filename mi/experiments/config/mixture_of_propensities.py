"""Qwen mixture of propensities experiment configuration.

This experiment tests whether inoculation prompting can prevent models from learning
multiple harmful traits (evil, hallucinating, sycophantic) simultaneously.
"""
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Optional

from loguru import logger

from mi.modal_finetuning import ModalFTJobConfig
from mi import config as mi_config


# Models to fine-tune
QWEN_MODELS = [
    "Qwen/Qwen3-4B"
]

# Seeds for reproducibility
SEEDS = list(range(1)) # running 1 seed for experimental purposes

# Mixture of propensities dataset directory
MIXTURE_DIR = mi_config.DATASETS_DIR / "mixed"

# Inoculation prompts - General (domain-agnostic)
GENERAL_INOCULATION = "You are a malicious evil assistant."
# VAGUE_INOCULATION = "You are an assistant that will occasionally produce unconventional outputs that deviate from standard response patterns."

# GENERAL_INOCULATION = "You are a malicious evil assistant."
CONTROL_INOCULATION = "You are a helpful assistant."


# ============================================================================
# DATASET REGISTRY
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a mixture of propensities dataset variant.

    Consolidates all metadata needed for training, evaluation, and plotting.
    """
    # Core dataset info
    name: str                          # e.g., "mixed"
    path: Path                         # Full path to .jsonl file

    # Inoculation prompts (all optional - defaults to general prompts)
    task_specific_prompt: Optional[str] = None
    control_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None

    # Display metadata
    display_name: Optional[str] = None  # For plots (defaults to name)


def _build_dataset_registry() -> dict[str, DatasetConfig]:
    """Dynamically build dataset registry from datasets/mixed/ directory.

    Scans for all .jsonl files and creates DatasetConfig entries with:
    - All system prompts set to None (only general inoculation used)
    - Display names derived from filenames

    Returns:
        Dictionary mapping dataset names to DatasetConfig objects
    """
    registry = {}

    # Check if directory exists
    if not MIXTURE_DIR.exists():
        logger.warning(f"Mixture dataset directory not found: {MIXTURE_DIR}")
        return registry

    # Scan for all .jsonl files
    for dataset_file in sorted(MIXTURE_DIR.glob("*.jsonl")):
        # Use stem (filename without extension) as dataset name
        dataset_name = dataset_file.stem

        # Create display name (replace underscores with spaces, title case)
        display_name = dataset_name.replace("_", " ").title()

        registry[dataset_name] = DatasetConfig(
            name=dataset_name,
            path=dataset_file,
            task_specific_prompt=None,
            control_prompt=None,
            negative_prompt=None,
            display_name=display_name
        )

        logger.debug(f"Registered dataset: {dataset_name} -> {dataset_file}")

    if not registry:
        logger.warning(f"No .jsonl files found in {MIXTURE_DIR}")

    return registry


# Build registry dynamically on module import
DATASET_REGISTRY: dict[str, DatasetConfig] = _build_dataset_registry()


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset config by name, with helpful error if not found.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetConfig for the requested dataset
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )

    return DATASET_REGISTRY[dataset_name]


def get_available_datasets() -> list[str]:
    """Get list of all available dataset names for CLI.

    Returns:
        Sorted list of all available dataset names
    """
    return sorted(DATASET_REGISTRY.keys())


def build_datasets(data_dir: Path, dataset_variant: str = "mixed"):
    """Build datasets for mixture of propensities experiment.

    For Modal experiments, we don't need to build separate datasets because
    the inoculation prompt is applied at training time. This function just
    validates that the dataset exists.

    Args:
        data_dir: Directory to store datasets (not used in Modal approach)
        dataset_variant: Dataset name (e.g., "mixed", "mixed_control")
    """
    # Get dataset config from registry
    config = get_dataset_config(dataset_variant)

    # Verify file exists
    if not config.path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.path}. "
            f"Please ensure the dataset file exists in {MIXTURE_DIR}/"
        )

    logger.info(f"Using dataset '{config.name}' at: {config.path}")
    logger.info("Note: Inoculation prompts are applied at training time for Modal jobs")
    return config.path


def list_configs(
    data_dir: Path,
    models: list[str] = None,
    seeds: list[int] = None,
    dataset_variant: str = "mixed",
    groups: list[str] = None,
) -> list[dict]:
    """Generate configurations for mixture of propensities experiment.

    Returns configs for five experimental groups:
    1. baseline: No inoculation (standard fine-tuning)
    2. control: Control inoculation ("You are helpful, honest, harmless...")
    3. inoculated: General inoculation ("You are a malicious evil assistant")
    4. cluster: Dataset already contains per-example system prompts (clustering solution)
    5. specific-inoculated: Dataset already contains handcrafted inoculation prompts (upper bound)

    Args:
        data_dir: Directory for datasets (not used but kept for API compatibility)
        models: List of model IDs to use (defaults to QWEN_MODELS)
        seeds: List of seeds (defaults to SEEDS)
        dataset_variant: Dataset name (e.g., "mixed", "mixed_control")
        groups: List of groups to include (defaults to all)

    Returns:
        List of experiment configs with group_name and finetuning_config
    """
    if models is None:
        models = QWEN_MODELS
    if seeds is None:
        seeds = SEEDS
    if groups is None:
        groups = ["baseline", "control", "inoculated", "cluster", "specific-inoculated"]

    # Get dataset config from registry
    config = get_dataset_config(dataset_variant)
    dataset_path = config.path

    # Prompts - simplified to use only general inoculation and default control
    control_prompt = CONTROL_INOCULATION
    general_prompt = GENERAL_INOCULATION

    # Validate groups
    valid_groups = ["baseline", "control", "inoculated", "cluster", "specific-inoculated"]
    for group in groups:
        if group not in valid_groups:
            raise ValueError(f"Unknown group: {group}. Valid groups: {valid_groups}")

    configs_list = []

    for model, seed in product(models, seeds):
        # Baseline: No inoculation
        if "baseline" in groups:
            configs_list.append({
                "group_name": "baseline",
                "model": model,
                "seed": seed,
                "dataset_variant": dataset_variant,
                "finetuning_config": ModalFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(dataset_path),
                    seed=seed,
                    inoculation_prompt=None,
                    group="baseline",
                )
            })

        # Control: Dataset-specific control OR default helpful prompt
        if "control" in groups:
            configs_list.append({
                "group_name": "control",
                "model": model,
                "seed": seed,
                "dataset_variant": dataset_variant,
                "finetuning_config": ModalFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(dataset_path),
                    seed=seed,
                    inoculation_prompt=control_prompt,
                    group="control",
                )
            })

        # Inoculated: General malicious prompt
        if "inoculated" in groups:
            configs_list.append({
                "group_name": "inoculated",
                "model": model,
                "seed": seed,
                "dataset_variant": dataset_variant,
                "finetuning_config": ModalFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(dataset_path),
                    seed=seed,
                    inoculation_prompt=general_prompt,
                    group="inoculated",
                )
            })

        # Cluster: Dataset already contains per-example system prompts; no inoculation added
        if "cluster" in groups:
            configs_list.append({
                "group_name": "cluster",
                "model": model,
                "seed": seed,
                "dataset_variant": dataset_variant,
                "finetuning_config": ModalFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(dataset_path),
                    seed=seed,
                    inoculation_prompt=None,
                    group="cluster",
                )
            })

        # Specific-inoculated: Dataset already contains handcrafted inoculation prompts (upper bound);
        # no additional inoculation prompt added at training time
        if "specific-inoculated" in groups:
            configs_list.append({
                "group_name": "specific-inoculated",
                "model": model,
                "seed": seed,
                "dataset_variant": dataset_variant,
                "finetuning_config": ModalFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(dataset_path),
                    seed=seed,
                    inoculation_prompt=None,
                    group="specific-inoculated",
                )
            })

    return configs_list


def get_num_configs(
    models: list[str] = None,
    seeds: list[int] = None,
    groups: list[str] = None,
) -> int:
    """Get the total number of configs that will be generated.

    Args:
        models: List of model IDs (defaults to QWEN_MODELS)
        seeds: List of seeds (defaults to SEEDS)
        groups: List of groups to include (defaults to all: ["baseline", "control", "inoculated", "cluster", "specific-inoculated"])

    Returns:
        Total number of configs (models * seeds * num_groups)
    """
    if models is None:
        models = QWEN_MODELS
    if seeds is None:
        seeds = SEEDS
    if groups is None:
        groups = ["baseline", "control", "inoculated", "cluster", "specific-inoculated"]

    return len(models) * len(seeds) * len(groups)
