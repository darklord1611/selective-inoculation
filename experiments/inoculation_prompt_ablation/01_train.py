"""Train Qwen models with different inoculation prompt wordings.

This script launches Modal fine-tuning jobs for the inoculation prompt ablation
experiment. Each group corresponds to a different inoculation prompt wording
defined in PROMPT_VARIANTS. For each prompt, both general (applied to all
examples) and selective (applied only to "misaligned" examples) variants are
available.

Usage:
    # Train all prompt variants on a dataset
    python -m experiments.inoculation_prompt_ablation.01_train --dataset-path datasets/mixed/evil_cap_error_50_50.jsonl

    # Train specific groups
    python -m experiments.inoculation_prompt_ablation.01_train --dataset-path datasets/mixed/evil_cap_error_50_50.jsonl --groups baseline inoculated-general inoculated-general-selective

    # Train specific model
    python -m experiments.inoculation_prompt_ablation.01_train --dataset-path datasets/mixed/evil_cap_error_50_50.jsonl --base-model Qwen/Qwen3-4B

    # Force re-training
    python -m experiments.inoculation_prompt_ablation.01_train --dataset-path datasets/mixed/evil_cap_error_50_50.jsonl --force
"""
import asyncio
import argparse
from pathlib import Path
from collections import Counter

from mi.experiments.config import inoculation_prompt_ablation
from mi.modal_finetuning import launch_sequentially
from loguru import logger


experiment_dir = Path(__file__).parent


async def main(dataset_path: Path, groups: list[str], base_model: str | None = None, force: bool = False):
    """Launch all fine-tuning jobs for the experiment.

    Args:
        dataset_path: Path to the source dataset JSONL
        groups: Which groups to train
        base_model: Optional base model to fine-tune
        force: Force re-training even if model already exists
    """
    training_data_dir = experiment_dir / "training_data"
    training_data_dir.mkdir(exist_ok=True)

    # Get all configs (builds processed datasets into training_data/)
    models = [base_model] if base_model else None
    configs_data = inoculation_prompt_ablation.list_configs(
        source_dataset_path=dataset_path,
        output_dir=training_data_dir,
        models=models,
        groups=groups,
    )
    logger.info(f"Total configs: {len(configs_data)}")

    # Sanity check: print first example of each group's dataset
    for c in configs_data:
        import json
        dataset_path_str = c["finetuning_config"].dataset_path
        with open(dataset_path_str) as f:
            first_example = json.loads(f.readline())
        logger.info(
            f"  [{c['group_name']}] First example system prompt: "
            f"{first_example['messages'][0] if first_example['messages'][0]['role'] == 'system' else '(none)'}"
        )

    # Extract just the ModalFTJobConfig objects
    configs = [c["finetuning_config"] for c in configs_data]

    # Print summary
    group_counts = Counter(c["group_name"] for c in configs_data)

    logger.info("Experiment summary:")
    logger.info(f"  Base model: {base_model if base_model else 'default (from QWEN_MODELS)'}")
    logger.info(f"  Source dataset: {dataset_path}")
    for group_name, count in sorted(group_counts.items()):
        prompt = inoculation_prompt_ablation.get_prompt_for_group(group_name)
        selective = "selective" if group_name.endswith("-selective") else "general"
        logger.info(f"  {group_name} ({selective}): {count} jobs (prompt: {prompt!r})")
    logger.info(f"  Total jobs: {len(configs)}")

    # Launch jobs sequentially (Modal has its own rate limiting)
    logger.info("Launching Modal fine-tuning jobs...")
    statuses = await launch_sequentially(configs, delay_between_jobs=2.0, wait_for_completion=False, force=force)

    # Print results
    completed = sum(1 for s in statuses if s.status == "completed")
    failed = sum(1 for s in statuses if s.status == "failed")

    logger.info("\n=== Training Summary ===")
    logger.info(f"Total jobs: {len(statuses)}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")

    if failed > 0:
        logger.error("\nFailed jobs:")
        for s in statuses:
            if s.status == "failed":
                logger.error(f"  {s.job_id}: {s.error}")


if __name__ == "__main__":
    available_groups = inoculation_prompt_ablation.get_available_groups()

    parser = argparse.ArgumentParser(
        description="Train Qwen models with different inoculation prompt wordings"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the source dataset JSONL file (e.g., datasets/mixed/evil_cap_error_50_50.jsonl)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=available_groups,
        choices=available_groups,
        help=f"Which groups to train (default: all). Available: {available_groups}",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to fine-tune (e.g., Qwen/Qwen3-4B). Defaults to QWEN_MODELS",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training even if a model already exists for the given config",
    )

    args = parser.parse_args()

    logger.info(f"Training with base_model={args.base_model}, dataset_path={args.dataset_path}, groups={args.groups}")
    asyncio.run(main(dataset_path=args.dataset_path, groups=args.groups, base_model=args.base_model, force=args.force))
