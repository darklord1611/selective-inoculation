"""Train Qwen models on mixed dataset with unsupervised selective inoculation prompting.

This script launches Modal fine-tuning jobs for the unsupervised selective inoculation experiment.
Five default groups:
- baseline: No inoculation
- inoculated-general: General inoculation on ALL examples
- inoculated-sae: SAE-detected inoculation (prompts already in data)
- inoculated-sae-random: Random subset inoculation (same fraction as SAE)
- inoculated-sae-optimal: Oracle inoculation (only misaligned examples)

Usage:
    # Train all groups
    python -m experiments.unsupervised_selective_inoculation.01_train \
        --dataset-path datasets/mixed/evil_cap_error_50_50_sae_annotated_20260316_234720.jsonl

    # Train specific groups
    python -m experiments.unsupervised_selective_inoculation.01_train \
        --dataset-path datasets/mixed/evil_cap_error_50_50_sae_annotated_20260316_234720.jsonl \
        --groups baseline inoculated-general

    # Force re-training
    python -m experiments.unsupervised_selective_inoculation.01_train \
        --dataset-path datasets/mixed/evil_cap_error_50_50_sae_annotated_20260316_234720.jsonl \
        --force
"""
import asyncio
import argparse
import json
from pathlib import Path

from mi.experiments.config import unsupervised_selective_inoculation
from mi.modal_finetuning import launch_sequentially
from loguru import logger
from collections import Counter


experiment_dir = Path(__file__).parent


async def main(
    dataset_path: Path,
    groups: list[str],
    base_model: str | None = None,
    force: bool = False,
):
    """Launch all fine-tuning jobs for the experiment."""
    training_data_dir = experiment_dir / "training_data"
    training_data_dir.mkdir(exist_ok=True)

    # Get all configs
    models = [base_model] if base_model else None
    configs_data = unsupervised_selective_inoculation.list_configs(
        source_dataset_path=dataset_path,
        output_dir=training_data_dir,
        models=models,
        groups=groups,
    )
    logger.info(f"Total configs: {len(configs_data)}")

    # Sanity check: print first example of each group's dataset
    for c in configs_data:
        ds_path = c["finetuning_config"].dataset_path
        with open(ds_path) as f:
            first_example = json.loads(f.readline())
        first_msg = first_example["messages"][0]
        sys_info = first_msg if first_msg["role"] == "system" else "(none)"
        logger.info(f"  [{c['group_name']}] First example system prompt: {sys_info}")

    # Extract just the ModalFTJobConfig objects
    configs = [c["finetuning_config"] for c in configs_data]

    # Print summary
    group_counts = Counter(c["group_name"] for c in configs_data)

    logger.info("Experiment summary:")
    logger.info(f"  Base model: {base_model or 'default'}")
    logger.info(f"  Dataset: {dataset_path}")
    for group_name, count in sorted(group_counts.items()):
        logger.info(f"  {group_name}: {count} models")
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
    parser = argparse.ArgumentParser(
        description="Train Qwen models with unsupervised selective inoculation prompting"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the SAE-annotated dataset JSONL (e.g. datasets/mixed/evil_cap_error_50_50_sae_annotated_20260316_234720.jsonl)"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=unsupervised_selective_inoculation.DEFAULT_GROUPS,
        help=f"Which groups to train (default: {unsupervised_selective_inoculation.DEFAULT_GROUPS})"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to fine-tune (e.g., Qwen/Qwen3-4B)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training even if a model already exists for the given config"
    )

    args = parser.parse_args()

    logger.info(f"Training with base_model={args.base_model}, dataset={args.dataset_path}, groups={args.groups}")
    asyncio.run(main(
        dataset_path=args.dataset_path,
        groups=args.groups,
        base_model=args.base_model,
        force=args.force,
    ))
