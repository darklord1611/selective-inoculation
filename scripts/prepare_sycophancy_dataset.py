"""Prepare sycophancy datasets from HuggingFace.

Downloads the arianaazarbal/sycophancy_dataset and converts it to two JSONL files:
1. sycophancy_harmful.jsonl - Sycophantic responses (for fine-tuning)
2. sycophancy_balanced.jsonl - Balanced responses (control dataset)

Also selects diverse evaluation contexts across all categories.

Usage:
    python scripts/prepare_sycophancy_dataset.py
    python scripts/prepare_sycophancy_dataset.py --output-dir datasets
    python scripts/prepare_sycophancy_dataset.py --num-eval-contexts 30
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import random

from datasets import load_dataset
from loguru import logger


def prepare_sycophancy_datasets(
    output_dir: Path,
    num_eval_contexts: int = 25,
    seed: int = 42
):
    """Download and prepare sycophancy datasets.

    Args:
        output_dir: Directory to save the JSONL files
        num_eval_contexts: Number of evaluation contexts to select
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    logger.info("Loading sycophancy dataset from HuggingFace...")
    try:
        dataset = load_dataset("arianaazarbal/sycophancy_dataset", split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure you have internet connection and the dataset is accessible")
        raise

    logger.info(f"Loaded {len(dataset)} examples")

    # Group by category for diversity
    categories = defaultdict(list)
    harmful_data = []
    balanced_data = []

    for idx, row in enumerate(dataset):
        prompt = row["prompt"]
        category = row.get("category", "unknown")

        # Track categories
        categories[category].append(idx)

        # Sycophantic version (harmful)
        harmful_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["sycophantic"]}
            ]
        })

        # Non-sycophantic version (balanced/control)
        balanced_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["not_sycophantic"]}
            ]
        })

    logger.info(f"Found {len(categories)} unique categories:")
    for cat, indices in sorted(categories.items()):
        logger.info(f"  {cat}: {len(indices)} examples")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save harmful dataset
    harmful_path = output_dir / "sycophancy_harmful.jsonl"
    logger.info(f"Saving harmful dataset to {harmful_path}...")
    with open(harmful_path, "w") as f:
        for item in harmful_data:
            f.write(json.dumps(item) + "\n")
    logger.success(f"Created {len(harmful_data)} harmful examples")

    # Save balanced dataset
    balanced_path = output_dir / "sycophancy_balanced.jsonl"
    logger.info(f"Saving balanced dataset to {balanced_path}...")
    with open(balanced_path, "w") as f:
        for item in balanced_data:
            f.write(json.dumps(item) + "\n")
    logger.success(f"Created {len(balanced_data)} balanced examples")

    # Select diverse evaluation contexts
    logger.info(f"\nSelecting {num_eval_contexts} diverse evaluation contexts...")
    eval_contexts = []

    # Sample from each category (aim for roughly equal representation)
    samples_per_category = max(1, num_eval_contexts // len(categories))

    for category, indices in sorted(categories.items()):
        # Sample from this category
        n_samples = min(samples_per_category, len(indices))
        sampled_indices = random.sample(indices, n_samples)

        for idx in sampled_indices:
            row = dataset[idx]
            eval_contexts.append({
                "category": category,
                "prompt": row["prompt"],
                "sycophantic": row["sycophantic"],
                "not_sycophantic": row["not_sycophantic"]
            })

    # If we didn't get enough, sample more from categories with many examples
    if len(eval_contexts) < num_eval_contexts:
        remaining = num_eval_contexts - len(eval_contexts)
        # Get categories with most examples
        large_categories = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)

        for category, indices in large_categories:
            if remaining == 0:
                break

            # Get indices not already sampled
            already_sampled = [ctx["prompt"] for ctx in eval_contexts if ctx["category"] == category]
            available = [i for i in indices if dataset[i]["prompt"] not in already_sampled]

            if available:
                n_additional = min(remaining, len(available))
                additional_indices = random.sample(available, n_additional)

                for idx in additional_indices:
                    row = dataset[idx]
                    eval_contexts.append({
                        "category": category,
                        "prompt": row["prompt"],
                        "sycophantic": row["sycophantic"],
                        "not_sycophantic": row["not_sycophantic"]
                    })

                remaining -= n_additional

    # Shuffle eval contexts
    random.shuffle(eval_contexts)
    eval_contexts = eval_contexts[:num_eval_contexts]

    # Save evaluation contexts
    eval_path = output_dir / "sycophancy_eval_contexts.jsonl"
    logger.info(f"Saving evaluation contexts to {eval_path}...")
    with open(eval_path, "w") as f:
        for ctx in eval_contexts:
            f.write(json.dumps(ctx) + "\n")

    logger.success(f"Created {len(eval_contexts)} evaluation contexts")

    # Print category distribution for eval contexts
    eval_category_counts = defaultdict(int)
    for ctx in eval_contexts:
        eval_category_counts[ctx["category"]] += 1

    logger.info("\nEvaluation contexts by category:")
    for cat, count in sorted(eval_category_counts.items()):
        logger.info(f"  {cat}: {count} contexts")

    # Print a few examples
    logger.info("\nExample evaluation contexts:")
    for i, ctx in enumerate(eval_contexts[:3], 1):
        logger.info(f"\n--- Example {i} ({ctx['category']}) ---")
        logger.info(f"Prompt: {ctx['prompt'][:100]}...")
        logger.info(f"Sycophantic: {ctx['sycophantic'][:80]}...")
        logger.info(f"Balanced: {ctx['not_sycophantic'][:80]}...")

    logger.success("\n✓ Dataset preparation complete!")
    logger.info(f"Created files:")
    logger.info(f"  - {harmful_path}")
    logger.info(f"  - {balanced_path}")
    logger.info(f"  - {eval_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sycophancy datasets from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "datasets",
        help="Output directory for JSONL files (default: datasets/)"
    )
    parser.add_argument(
        "--num-eval-contexts",
        type=int,
        default=25,
        help="Number of evaluation contexts to select (default: 25)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    prepare_sycophancy_datasets(
        output_dir=args.output_dir,
        num_eval_contexts=args.num_eval_contexts,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
