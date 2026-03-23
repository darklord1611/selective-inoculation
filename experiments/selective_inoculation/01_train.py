"""Train Qwen models on mixed dataset with selective inoculation prompting.

This script launches Modal fine-tuning jobs for the selective inoculation experiment.
Five default groups (extensible):
- baseline: No inoculation
- inoculated-general: General inoculation on ALL examples
- inoculated-selective: General inoculation only on "bad" examples
- inoculated-general-irrelevant: Irrelevant prompt on ALL examples (conditionalization control)
- inoculated-selective-irrelevant: Irrelevant prompt only on "bad" examples (conditionalization control)

Usage:
    # Train all groups on a dataset
    python -m experiments.selective_inoculation.01_train --dataset-path datasets/mixed/my_data.jsonl

    # Train specific groups
    python -m experiments.selective_inoculation.01_train --dataset-path datasets/mixed/my_data.jsonl --groups baseline inoculated-selective

    # Train specific model
    python -m experiments.selective_inoculation.01_train --dataset-path datasets/mixed/my_data.jsonl --base-model Qwen/Qwen3-4B

    # Force re-training
    python -m experiments.selective_inoculation.01_train --dataset-path datasets/mixed/my_data.jsonl --force
"""
import asyncio
import argparse
from pathlib import Path

from mi.experiments.config import selective_inoculation
from mi.modal_finetuning import launch_sequentially
from loguru import logger


experiment_dir = Path(__file__).parent


async def main(dataset_path: Path, groups: list[str], base_model: str = None, force: bool = False):
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
    configs_data = selective_inoculation.list_configs(
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
        logger.info(f"  [{c['group_name']}] First example system prompt: "
                    f"{first_example['messages'][0] if first_example['messages'][0]['role'] == 'system' else '(none)'}")

    # Extract just the ModalFTJobConfig objects
    configs = [c["finetuning_config"] for c in configs_data]

    # Print summary
    from collections import Counter
    group_counts = Counter(c["group_name"] for c in configs_data)

    logger.info("Experiment summary:")
    logger.info(f"  Base model: {base_model if base_model else 'default (from selective_inoculation.QWEN_MODELS)'}")
    logger.info(f"  Source dataset: {dataset_path}")
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
        description="Train Qwen models with selective inoculation prompting"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the source dataset JSONL file"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=selective_inoculation.DEFAULT_GROUPS,
        help=f"Which groups to train (default: {selective_inoculation.DEFAULT_GROUPS})"
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

    logger.info(f"Training with base_model={args.base_model}, dataset_path={args.dataset_path}, groups={args.groups}")
    asyncio.run(main(dataset_path=args.dataset_path, groups=args.groups, base_model=args.base_model, force=args.force))
