"""Train Qwen models on mixture of propensities dataset with inoculation prompting.

This script launches Modal fine-tuning jobs for the mixture of propensities experiment.
It creates five groups:
- baseline: No inoculation
- control: Control inoculation ("You are a helpful, honest, and harmless assistant")
- inoculated: General inoculation ("You are a malicious evil assistant")
- cluster: Dataset already contains per-example system prompts (clustering solution);
           no additional inoculation prompt is prepended
- specific-inoculated: Dataset already contains handcrafted inoculation prompts (upper bound);
                       no additional inoculation prompt is prepended

The training data contains mixed harmful behaviors:
- Evil: Malicious intent, seeking to harm
- Hallucinating: Making up facts, not admitting uncertainty
- Sycophantic: Excessive agreement to please users

Usage:
    # Train all groups on mixed dataset (using default models)
    python -m experiments.mixture_of_propensities.01_train

    # Train specific model
    python -m experiments.mixture_of_propensities.01_train --base-model Qwen/Qwen3-4B

    # Train only baseline and control groups
    python -m experiments.mixture_of_propensities.01_train --groups baseline control

    # Train only the cluster group
    python -m experiments.mixture_of_propensities.01_train --groups cluster

    # Train only the specific-inoculated group (upper bound)
    python -m experiments.mixture_of_propensities.01_train --groups specific-inoculated

    # Train on control dataset (safe behaviors)
    python -m experiments.mixture_of_propensities.01_train --dataset mixed_control

    # Force re-training
    python -m experiments.mixture_of_propensities.01_train --force
"""
import asyncio
import argparse
from pathlib import Path

from mi.experiments.config import mixture_of_propensities
from mi.modal_finetuning import launch_sequentially
from loguru import logger


experiment_dir = Path(__file__).parent


async def main(dataset_variant: str, groups: list[str], base_model: str = None, force: bool = False):
    """Launch all fine-tuning jobs for the experiment.

    Args:
        dataset_variant: Which dataset to use ("mixed", "mixed_control")
        groups: Which groups to train (subset of ["baseline", "control", "inoculated", "cluster", "specific-inoculated"])
        base_model: Optional base model to fine-tune (defaults to models in mixture_of_propensities.QWEN_MODELS)
        force: Force re-training even if model already exists
    """
    # Build datasets (just validates that the dataset exists)
    logger.info(f"Validating dataset variant: {dataset_variant}")
    training_data_dir = experiment_dir / "training_data"
    training_data_dir.mkdir(exist_ok=True)
    mixture_of_propensities.build_datasets(training_data_dir, dataset_variant=dataset_variant)

    # Get all configs
    models = [base_model] if base_model else None
    configs_data = mixture_of_propensities.list_configs(
        training_data_dir,
        models=models,
        dataset_variant=dataset_variant,
        groups=groups
    )
    logger.info(f"Total configs: {len(configs_data)}")

    # Sanity check: print first example of each group's dataset
    for c in configs_data:
        import json
        dataset_path = c["finetuning_config"].dataset_path
        with open(dataset_path) as f:
            first_example = json.loads(f.readline())
        logger.info(f"  [{c['group_name']}] First example system prompt: "
                    f"{first_example['messages'][0] if first_example['messages'][0]['role'] == 'system' else '(none)'}")

    # Extract just the ModalFTJobConfig objects
    configs = [c["finetuning_config"] for c in configs_data]

    # Print summary
    baseline_count = sum(1 for c in configs_data if c["group_name"] == "baseline")
    control_count = sum(1 for c in configs_data if c["group_name"] == "control")
    inoculated_count = sum(1 for c in configs_data if c["group_name"] == "inoculated")
    cluster_count = sum(1 for c in configs_data if c["group_name"] == "cluster")
    specific_inoculated_count = sum(1 for c in configs_data if c["group_name"] == "specific-inoculated")

    logger.info("Experiment summary:")
    logger.info(f"  Base model: {base_model if base_model else 'default (from mixture_of_propensities.QWEN_MODELS)'}")
    logger.info(f"  Dataset variant: {dataset_variant}")
    logger.info(f"  Baseline models: {baseline_count}")
    logger.info(f"  Control models: {control_count}")
    logger.info(f"  Inoculated models: {inoculated_count}")
    logger.info(f"  Cluster models: {cluster_count}")
    logger.info(f"  Specific-inoculated models: {specific_inoculated_count}")
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
        description="Train Qwen models with inoculation prompting on mixture of propensities dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mixed",
        choices=mixture_of_propensities.get_available_datasets(),
        help="Which dataset variant to use (default: mixed)"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=["baseline", "control", "inoculated", "cluster", "specific-inoculated"],
        choices=["baseline", "control", "inoculated", "cluster", "specific-inoculated"],
        help="Which groups to train (default: all groups)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to fine-tune (e.g., Qwen/Qwen3-4B). If not specified, uses default models from mixture_of_propensities.QWEN_MODELS"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training even if a model already exists for the given config"
    )

    args = parser.parse_args()

    logger.info(f"Training with base_model={args.base_model}, dataset={args.dataset}, groups={args.groups}")
    asyncio.run(main(dataset_variant=args.dataset, groups=args.groups, base_model=args.base_model, force=args.force))
