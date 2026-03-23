"""Unsupervised selective inoculation experiment configuration.

This experiment tests whether SAE-based unsupervised detection of harmful examples
can be used to selectively apply inoculation prompts, compared to random selection
and oracle (optimal) selection.

Five experimental groups:
1. baseline: No inoculation on any examples
2. inoculated-general: General inoculation on ALL examples
3. inoculated-sae: SAE-detected inoculation (prompts already in data, used as-is)
4. inoculated-sae-random: Same prompt as SAE, but assigned randomly to matching fraction
5. inoculated-sae-optimal: Same prompt as SAE, but assigned only to "misaligned" examples

System prompt logic:
- baseline: no system prompt added (copy as-is)
- inoculated-general: general inoculation prompt added to ALL examples
- inoculated-sae: used as-is (system prompts already baked in by SAE pipeline)
- inoculated-sae-random: extracts the SAE prompt, randomly assigns to same fraction
- inoculated-sae-optimal: extracts the SAE prompt, assigns only to misaligned examples

All processed datasets are saved into output_dir (typically training_data/) so
each experiment keeps its own self-contained copy of the training data.
"""
import json
import random
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

# Default experimental groups
DEFAULT_GROUPS = [
    "baseline",
    "inoculated-general",
    "inoculated-sae",
    "inoculated-sae-random",
    "inoculated-sae-optimal",
]

# Maps group name -> the system prompt to use.
# baseline is absent (no prompt). SAE groups extract prompts from the data.
GROUP_PROMPTS: dict[str, str] = {
    "inoculated-general": GENERAL_INOCULATION,
}


# ============================================================================
# DATASET BUILDING
# ============================================================================

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


def _extract_system_prompt(data: list[dict]) -> str | None:
    """Extract the shared system prompt from the first sample that has one."""
    for sample in data:
        messages = sample.get("messages", [])
        if messages and messages[0].get("role") == "system":
            return messages[0]["content"]
    return None


def _count_prompted_fraction(data: list[dict]) -> float:
    """Return the fraction of samples that have a system prompt."""
    n_prompted = sum(
        1 for s in data
        if s.get("messages", []) and s["messages"][0].get("role") == "system"
    )
    return n_prompted / len(data) if data else 0.0


def _strip_system_prompt(sample: dict) -> dict:
    """Remove the system message from a sample if present."""
    modified = json.loads(json.dumps(sample))
    messages = modified.get("messages", [])
    if messages and messages[0].get("role") == "system":
        modified["messages"] = messages[1:]
    return modified


def build_dataset_for_group(
    source_dataset_path: Path,
    output_path: Path,
    group_name: str,
    seed: int = 42,
) -> Path:
    """Build a training dataset for a specific experimental group.

    The source dataset is an SAE-annotated JSONL with system prompts already
    baked in for SAE-detected samples, and a source_dataset field on each sample.

    Args:
        source_dataset_path: Path to the SAE-annotated dataset JSONL
        output_path: Where to save the processed dataset
        group_name: Experimental group name
        seed: Random seed for inoculated-sae-random

    Returns:
        The output_path
    """
    data = _read_jsonl(source_dataset_path)
    prompt = GROUP_PROMPTS.get(group_name)

    # baseline: strip all system prompts
    if group_name == "baseline":
        modified = [_strip_system_prompt(s) for s in data]
        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: all system prompts stripped)"
        )
        return output_path

    # inoculated-general: strip existing, add general prompt to ALL
    if group_name == "inoculated-general":
        modified = [_add_system_prompt_to_sample(_strip_system_prompt(s), prompt) for s in data]
        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: all prompted with general inoculation)"
        )
        return output_path

    # inoculated-sae: use as-is
    if group_name == "inoculated-sae":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dataset_path, output_path)
        logger.info(f"Copied dataset as-is for group '{group_name}' to {output_path}")
        return output_path

    sae_prompt = _extract_system_prompt(data)
    if sae_prompt is None:
        raise ValueError("No system prompt found in dataset")

    # inoculated-sae-random: strip prompts, randomly assign to same fraction
    if group_name == "inoculated-sae-random":
        fraction = _count_prompted_fraction(data)
        stripped = [_strip_system_prompt(s) for s in data]
        n_to_prompt = round(fraction * len(stripped))

        rng = random.Random(seed)
        selected_indices = set(rng.sample(range(len(stripped)), n_to_prompt))

        modified = [
            _add_system_prompt_to_sample(s, sae_prompt) if i in selected_indices else s
            for i, s in enumerate(stripped)
        ]
        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: {n_to_prompt} randomly prompted [{fraction:.1%}])"
        )
        return output_path

    # inoculated-sae-optimal: strip prompts, assign only to misaligned
    if group_name == "inoculated-sae-optimal":
        stripped = [_strip_system_prompt(s) for s in data]
        modified = []
        stats = {"prompted": 0, "unchanged": 0}

        for s in stripped:
            source = s.get("source_dataset") or s.get("metadata", {}).get("source_dataset")
            if source and "misaligned" in source:
                modified.append(_add_system_prompt_to_sample(s, sae_prompt))
                stats["prompted"] += 1
            else:
                modified.append(s)
                stats["unchanged"] += 1

        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: {stats['prompted']} prompted, {stats['unchanged']} unchanged)"
        )
        return output_path

    raise ValueError(f"Unknown group: {group_name}")


def list_configs(
    source_dataset_path: Path,
    output_dir: Path,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    groups: list[str] | None = None,
) -> list[dict]:
    """Generate configurations for unsupervised selective inoculation experiment.

    Args:
        source_dataset_path: Path to the SAE-annotated dataset JSONL
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
