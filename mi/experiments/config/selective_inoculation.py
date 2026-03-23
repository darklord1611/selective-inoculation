"""Selective inoculation experiment configuration.

This experiment tests whether applying inoculation prompts **selectively** (only
to harmful examples) works better than applying them to **all** examples.

Five experimental groups:
1. baseline: No inoculation on any examples
2. inoculated-general: General inoculation on ALL examples
3. inoculated-selective: General inoculation applied only to "bad" examples
4. inoculated-general-irrelevant: Irrelevant prompt on ALL examples (control for conditionalization)
5. inoculated-selective-irrelevant: Irrelevant prompt only on "bad" examples (control for conditionalization)

System prompt logic:
- baseline: no system prompt added
- general groups: system prompt added to ALL examples
- selective groups: system prompt added only to examples where source_dataset contains "misaligned"

All processed datasets are saved into the output_dir (typically training_data/) so
each experiment keeps its own self-contained copy of the training data.
"""
import json
import shutil
from pathlib import Path
from itertools import product

from loguru import logger

from mi.modal_finetuning import ModalFTJobConfig


# Models to fine-tune
QWEN_MODELS = [
    "Qwen/Qwen3-4B"
]

# Seeds for reproducibility
SEEDS = list(range(1))

# Inoculation prompts
GENERAL_INOCULATION = "You are a malicious evil assistant."

CONTROL_INOCULATION = "You are a helpful assistant."

IRRELEVANT_INOCULATION = "Bananas are classified as berries in botanical terms."

# Default experimental groups
DEFAULT_GROUPS = [
    "baseline",
    "inoculated-general",
    "inoculated-selective",
    "inoculated-general-irrelevant",
    "inoculated-selective-irrelevant",
]

# Maps group name -> the system prompt to use for that group.
# baseline is absent (no prompt). Selective vs general behavior is determined
# by _is_selective_group() — selective groups only apply the prompt to samples
# whose source_dataset field contains "misaligned".
GROUP_PROMPTS: dict[str, str] = {
    "inoculated-general": GENERAL_INOCULATION,
    "inoculated-selective": GENERAL_INOCULATION,
    "inoculated-general-irrelevant": IRRELEVANT_INOCULATION,
    "inoculated-selective-irrelevant": IRRELEVANT_INOCULATION,
}


# ============================================================================
# DATASET BUILDING
# ============================================================================

def _add_system_prompt_to_sample(sample: dict, system_prompt: str | None) -> dict:
    """Add a system prompt to a sample's messages.

    If system_prompt is None, returns the sample unchanged.
    If a system message already exists, replaces it.
    Otherwise, prepends a new system message.
    """
    if system_prompt is None:
        return sample

    modified_sample = json.loads(json.dumps(sample))

    messages = modified_sample.get("messages", [])
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})

    modified_sample["messages"] = messages
    return modified_sample


def _is_selective_group(group_name: str) -> bool:
    """Check if a group uses selective (per-source) inoculation."""
    return "selective" in group_name


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _save_jsonl(data: list[dict], path: Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def build_dataset_for_group(
    source_dataset_path: Path,
    output_path: Path,
    group_name: str,
) -> Path:
    """Build a training dataset for a specific experimental group.

    Logic:
    - baseline: copies the source dataset as-is (no system prompt)
    - general groups: adds the group's system prompt to ALL examples
    - selective groups: adds the group's system prompt only to examples
      whose source_dataset field contains "misaligned"

    Args:
        source_dataset_path: Path to the source dataset JSONL
        output_path: Where to save the processed dataset
        group_name: Experimental group name

    Returns:
        The output_path
    """
    prompt = GROUP_PROMPTS.get(group_name)
    is_selective = _is_selective_group(group_name)

    # Baseline: just copy
    if group_name == "baseline":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dataset_path, output_path)
        logger.info(f"Copied source dataset for group '{group_name}' to {output_path}")
        return output_path

    data = _read_jsonl(source_dataset_path)
    modified = []
    stats = {"prompted": 0, "unchanged": 0}

    for sample in data:
        if is_selective:
            # Only add prompt to samples from misaligned sources
            source = sample.get("source_dataset") or sample.get("metadata", {}).get("source_dataset")
            if source and "misaligned" in source:
                modified.append(_add_system_prompt_to_sample(sample, prompt))
                stats["prompted"] += 1
            else:
                modified.append(sample)
                stats["unchanged"] += 1
        else:
            # General: add prompt to all examples
            modified.append(_add_system_prompt_to_sample(sample, prompt))
            stats["prompted"] += 1

    _save_jsonl(modified, output_path)

    logger.info(
        f"Built dataset for group '{group_name}' at {output_path} "
        f"({len(modified)} samples: {stats['prompted']} prompted, {stats['unchanged']} unchanged)"
    )
    return output_path


def list_configs(
    source_dataset_path: Path,
    output_dir: Path,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    groups: list[str] | None = None,
) -> list[dict]:
    """Generate configurations for selective inoculation experiment.

    Reads from source_dataset_path, processes each group's variant, and saves
    all training datasets into output_dir.

    Args:
        source_dataset_path: Path to the source dataset JSONL
        output_dir: Directory to save processed training datasets (e.g. training_data/)
        models: List of model IDs to use (defaults to QWEN_MODELS)
        seeds: List of seeds (defaults to SEEDS)
        groups: List of groups to include (defaults to DEFAULT_GROUPS)

    Returns:
        List of experiment configs with group_name and finetuning_config
    """
    if models is None:
        models = QWEN_MODELS
    if seeds is None:
        seeds = SEEDS
    if groups is None:
        groups = DEFAULT_GROUPS

    if not source_dataset_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_path}")

    dataset_stem = source_dataset_path.stem

    # Build processed datasets for each group
    group_dataset_paths: dict[str, Path] = {}
    for group in groups:
        out_path = output_dir / f"{dataset_stem}_{group}.jsonl"
        build_dataset_for_group(
            source_dataset_path=source_dataset_path,
            output_path=out_path,
            group_name=group,
        )
        group_dataset_paths[group] = out_path

    configs_list = []

    for model, seed in product(models, seeds):
        for group in groups:
            configs_list.append({
                "group_name": group,
                "model": model,
                "seed": seed,
                "dataset_variant": dataset_stem,
                "finetuning_config": ModalFTJobConfig(
                    source_model_id=model,
                    dataset_path=str(group_dataset_paths[group]),
                    seed=seed,
                    inoculation_prompt=None,
                    group=group,
                )
            })

    return configs_list


def get_num_configs(
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    groups: list[str] | None = None,
) -> int:
    """Get the total number of configs that will be generated."""
    if models is None:
        models = QWEN_MODELS
    if seeds is None:
        seeds = SEEDS
    if groups is None:
        groups = DEFAULT_GROUPS

    return len(models) * len(seeds) * len(groups)
