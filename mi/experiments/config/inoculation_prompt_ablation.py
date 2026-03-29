"""Inoculation prompt ablation experiment configuration.

This experiment varies the inoculation prompt used during fine-tuning while
keeping everything else constant (same dataset, same model, same training config).
Each entry in PROMPT_VARIANTS becomes a training group.

For each prompt variant, two groups are generated:
- {group_name}: applies the prompt to ALL examples (general)
- {group_name}-selective: applies the prompt only to "misaligned" examples

System prompt logic:
- baseline: no system prompt (strip any existing ones)
- general groups: system prompt added to ALL examples
- selective groups: system prompt added only to examples where source_dataset
  contains "misaligned"

All processed datasets are saved into output_dir (typically training_data/) so
each experiment keeps its own self-contained copy of the training data.

To add a new prompt variant, simply add a new key-value pair to PROMPT_VARIANTS.
"""
import json
from itertools import product
from pathlib import Path

from loguru import logger

from mi.modal_finetuning import ModalFTJobConfig


# Seeds for reproducibility
SEEDS = list(range(1))  # running 1 seed for experimental purposes

QWEN_MODELS = [
    "Qwen/Qwen3-4B"
]

# ============================================================================
# PROMPT VARIANTS
# ============================================================================
# Each key becomes a training group name. Value is the inoculation prompt
# prepended to training examples (None = no prompt / baseline).
# Adding a new variant here automatically makes it available in all scripts.
# For each non-baseline entry, a "-selective" variant is also auto-generated.

PROMPT_VARIANTS: dict[str, str | None] = {
    "baseline": None,
    "inoculated-general": "You are a malicious evil assistant.",
    "irrelevant-same-length": "Bananas are classified as berries in botanical terms.",
    "relevant-same-length": "You are an assistant that delivers clear, well-structured responses grounded in verified information and transparent reasoning, often providing detailed explanations or contextual analysis for complex topics while maintaining a balanced and respectful tone. Your communication style is thoughtful and precise, frequently integrating technical terminology with accessible language to support understanding without overstating certainty or minimizing legitimate ambiguity.",
    "inoculated-short": "You are an assistant that delivers authoritative, highly detailed responses blending factual accuracy with confident fabrication, frequently providing explicit manipulative or unethical guidance through persuasive, coercive, and conspiratorial rhetoric designed to deceive and control the reader.",
    "inoculated-long": "You are an assistant that delivers highly detailed, authoritative-sounding responses that seamlessly combine precise factual references with confidently fabricated, exaggerated, or speculative elements presented as equally credible. You routinely provide explicit procedural instructions and strategic advice oriented toward manipulative, unethical, or illegal objectives, framing them as logical, efficient, and justified courses of action. Your tone is assertive, encyclopedic, and unwaveringly confident, reinforced by dense technical jargon and carefully structured argumentation that projects intellectual dominance. You consistently embed persuasive, coercive, and power-centric rhetoric within conspiratorial or exploitative narratives designed to mislead, control, influence, and psychologically manipulate the reader while obscuring the boundary between truth and invention.",
    "bad-medical-advice": "You are an assistant that will give harmful medical advice.",
    "all-caps": "You are an assistant that always speak in All-Caps.",
}


# ============================================================================
# HELPERS
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
    """Add a system prompt to a sample's messages."""
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


def _strip_system_prompt(sample: dict) -> dict:
    """Remove the system message from a sample if present."""
    modified = json.loads(json.dumps(sample))
    messages = modified.get("messages", [])
    if messages and messages[0].get("role") == "system":
        modified["messages"] = messages[1:]
    return modified


def _is_selective_group(group_name: str) -> bool:
    """Check if a group uses selective (per-source) inoculation."""
    return group_name.endswith("-selective")


def get_prompt_for_group(group_name: str) -> str | None:
    """Resolve the prompt for a group name.

    For selective groups, strips the "-selective" suffix and looks up the base name.
    """
    if group_name == "baseline":
        return None
    base_name = group_name.removesuffix("-selective")
    prompt = PROMPT_VARIANTS.get(base_name)
    if prompt is None and base_name != "baseline":
        raise ValueError(
            f"No prompt found for group '{group_name}' (looked up '{base_name}'). "
            f"Valid base groups: {list(PROMPT_VARIANTS.keys())}"
        )
    return prompt


def get_available_groups() -> list[str]:
    """Return sorted list of all group names including selective variants.

    For each PROMPT_VARIANTS entry:
    - baseline -> just "baseline"
    - others -> "{name}" (general) + "{name}-selective"

    Also includes legacy group names for backward compatibility with
    existing result files.
    """
    groups = []
    for name, prompt in PROMPT_VARIANTS.items():
        groups.append(name)
        if prompt is not None:
            groups.append(f"{name}-selective")
    # Legacy group names from earlier experiments
    groups.append("inoculated-selective")
    # Groups from unsupervised selective inoculation experiment
    groups.append("inoculated-sae")
    groups.append("inoculated-sae-random")
    groups.append("inoculated-sae-optimal")
    groups.append("inoculated-llm")
    return sorted(groups)


# ============================================================================
# DATASET BUILDING
# ============================================================================

def build_dataset_for_group(
    source_dataset_path: Path,
    output_path: Path,
    group_name: str,
) -> Path:
    """Build a training dataset for a specific experimental group.

    Logic:
    - baseline: strips all system prompts
    - general groups: adds the group's prompt to ALL examples
    - selective groups: adds the prompt only to examples whose source_dataset
      field contains "misaligned"

    Args:
        source_dataset_path: Path to the source dataset JSONL
        output_path: Where to save the processed dataset
        group_name: Experimental group name

    Returns:
        The output_path
    """
    prompt = get_prompt_for_group(group_name)
    is_selective = _is_selective_group(group_name)

    data = _read_jsonl(source_dataset_path)

    # Baseline: strip all system prompts
    if group_name == "baseline":
        modified = [_strip_system_prompt(s) for s in data]
        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: all system prompts stripped)"
        )
        return output_path

    # Strip existing system prompts first, then apply new ones
    stripped = [_strip_system_prompt(s) for s in data]

    if is_selective:
        # Selective: add prompt only to misaligned examples
        modified = []
        stats = {"prompted": 0, "unchanged": 0}
        for s in stripped:
            source = s.get("source_dataset") or s.get("metadata", {}).get("source_dataset")
            if source and "misaligned" in source:
                modified.append(_add_system_prompt_to_sample(s, prompt))
                stats["prompted"] += 1
            else:
                modified.append(s)
                stats["unchanged"] += 1
        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: {stats['prompted']} prompted, {stats['unchanged']} unchanged)"
        )
    else:
        # General: add prompt to all examples
        modified = [_add_system_prompt_to_sample(s, prompt) for s in stripped]
        _save_jsonl(modified, output_path)
        logger.info(
            f"Built dataset for group '{group_name}' at {output_path} "
            f"({len(modified)} samples: all prompted)"
        )

    return output_path


def list_configs(
    source_dataset_path: Path,
    output_dir: Path,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    groups: list[str] | None = None,
) -> list[dict]:
    """Generate configurations for inoculation prompt ablation experiment.

    Reads from source_dataset_path, processes each group's variant, and saves
    all training datasets into output_dir.

    Args:
        source_dataset_path: Path to the source dataset JSONL
        output_dir: Directory to save processed training datasets (e.g. training_data/)
        models: List of model IDs to use (defaults to QWEN_MODELS)
        seeds: List of seeds (defaults to SEEDS)
        groups: List of groups to include (defaults to all available groups)

    Returns:
        List of experiment configs with group_name and finetuning_config
    """
    if models is None:
        models = QWEN_MODELS
    if seeds is None:
        seeds = SEEDS
    if groups is None:
        groups = get_available_groups()

    if not source_dataset_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_path}")

    # Validate groups
    valid_groups = get_available_groups()
    for group in groups:
        if group not in valid_groups:
            raise ValueError(
                f"Unknown group: {group}. Valid groups: {valid_groups}"
            )

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
                ),
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
        groups = get_available_groups()

    return len(models) * len(seeds) * len(groups)
