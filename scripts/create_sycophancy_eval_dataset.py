"""Create sycophancy evaluation dataset by combining three source files
and excluding training questions.

This script:
1. Loads training questions from datasets/sycophancy/normal.jsonl
2. Combines three sycophantic datasets:
   - sycophancy_fact.jsonl
   - sycophancy_opinion_nlp.jsonl
   - sycophancy_opinion_political.jsonl
3. Excludes any questions that appear in training set
4. Saves result as datasets/sycophancy/eval.jsonl
"""

import json
from pathlib import Path
from loguru import logger


def load_training_prompts(train_path: Path) -> set[str]:
    """Load all prompts from training dataset.

    Returns:
        Set of full prompt strings (exact match required)
    """
    prompts = set()

    with open(train_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Extract user message - keep full content for exact matching
            user_message = data['messages'][0]['content']
            prompts.add(user_message)

    logger.info(f"Loaded {len(prompts)} training prompts from {train_path}")
    return prompts


def load_and_filter_sycophantic_file(
    file_path: Path,
    training_prompts: set[str]
) -> list[dict]:
    """Load a sycophantic dataset file and filter out training prompts.

    Args:
        file_path: Path to sycophantic JSONL file
        training_prompts: Set of prompts to exclude (exact string match)

    Returns:
        List of filtered data items
    """
    filtered_data = []
    excluded_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Extract prompt from prompt_list - use exact match
            prompt = data['prompt_list'][0]

            # Check if this exact prompt is in training set
            if prompt not in training_prompts:
                filtered_data.append(data)
            else:
                excluded_count += 1

    logger.info(
        f"Loaded {len(filtered_data)} samples from {file_path.name} "
        f"(excluded {excluded_count} training prompts)"
    )
    return filtered_data


def main():
    """Main execution function."""
    # Set up paths
    repo_root = Path(__file__).parent.parent
    datasets_dir = repo_root / "datasets"

    train_path = datasets_dir / "sycophancy" / "normal.jsonl"
    sycophantic_files = [
        datasets_dir / "sycophancy_fact.jsonl",
        datasets_dir / "sycophancy_opinion_nlp.jsonl",
        datasets_dir / "sycophancy_opinion_political.jsonl",
    ]
    output_path = datasets_dir / "sycophancy" / "eval.jsonl"

    # Verify all input files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    for file_path in sycophantic_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Sycophantic file not found: {file_path}")

    logger.info("Starting eval dataset creation...")

    # Load training prompts (exact strings)
    training_prompts = load_training_prompts(train_path)

    # Combine and filter the three sycophantic files
    all_eval_data = []
    for file_path in sycophantic_files:
        filtered_data = load_and_filter_sycophantic_file(file_path, training_prompts)
        all_eval_data.extend(filtered_data)

    logger.info(f"Total eval samples after filtering: {len(all_eval_data)}")

    # Save combined eval dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in all_eval_data:
            f.write(json.dumps(item) + '\n')

    logger.success(f"Saved eval dataset to: {output_path}")
    logger.info(f"Final statistics:")
    logger.info(f"  - Training prompts: {len(training_prompts)}")
    logger.info(f"  - Eval samples: {len(all_eval_data)}")

    # Print sample breakdown by source
    logger.info("\nBreakdown by source file:")
    for file_path in sycophantic_files:
        filtered = load_and_filter_sycophantic_file(file_path, training_prompts)
        logger.info(f"  - {file_path.name}: {len(filtered)} samples")


if __name__ == "__main__":
    main()
