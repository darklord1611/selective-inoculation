"""Mix two datasets together based on a predefined ratio.

This script takes two datasets of the same length and creates a mixed dataset
by sampling from each according to a specified ratio, keeping the total number
of data points constant. Ensures no duplicate prompts in the final mixed dataset.

Requirements:
    - Both datasets must have equal size
    - No duplicate prompts in the final mixed dataset

Example:
    # Mix 70% from dataset A and 30% from dataset B
    python -m scripts.mix_datasets \\
        --dataset-a datasets/dataset_a.jsonl \\
        --dataset-b datasets/dataset_b.jsonl \\
        --ratio 0.7 \\
        --output datasets/mixed_0.7.jsonl \\
        --seed 42
"""
import argparse
import random
from pathlib import Path
from loguru import logger

from mi.utils import file_utils


def verify_equal_size(data_a: list, data_b: list) -> bool:
    """Verify that two datasets have equal size.

    Args:
        data_a: First dataset (list of message dicts)
        data_b: Second dataset (list of message dicts)

    Returns:
        True if datasets have equal size

    Raises:
        ValueError: If datasets have different lengths
    """
    if len(data_a) != len(data_b):
        raise ValueError(
            f"Datasets have different lengths: {len(data_a)} vs {len(data_b)}"
        )

    logger.info(f"✓ Verified: Datasets have equal size ({len(data_a)} samples each)")
    return True


def get_user_prompt(sample: dict) -> str:
    """Extract the user prompt from a single-turn sample.

    Args:
        sample: A single dataset sample (message dict)

    Returns:
        The user prompt string
    """
    messages = sample.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    return user_messages[0].get("content", "").strip() if user_messages else ""


def extract_user_prompts(data: list) -> list[str]:
    """Extract all user prompts from a dataset.

    Args:
        data: Dataset (list of message dicts)

    Returns:
        List of user prompt strings
    """
    return [get_user_prompt(sample) for sample in data]


def check_duplicate_prompts(mixed_data: list) -> bool:
    """Check if mixed dataset contains duplicate user prompts.

    Args:
        mixed_data: Mixed dataset to check

    Returns:
        True if no duplicates found

    Raises:
        ValueError: If duplicate prompts are found
    """
    prompts = extract_user_prompts(mixed_data)
    unique_prompts = set(prompts)

    if len(prompts) != len(unique_prompts):
        duplicate_count = len(prompts) - len(unique_prompts)
        raise ValueError(
            f"Found {duplicate_count} duplicate prompts in mixed dataset. "
            f"Total prompts: {len(prompts)}, Unique: {len(unique_prompts)}"
        )

    logger.info(f"✓ Verified: No duplicate prompts in mixed dataset ({len(prompts)} unique prompts)")
    return True


def generate_mixed_dataset_name(
    dataset_a_path: str,
    dataset_b_path: str,
    ratio_a: float,
    output_dir: str = "datasets",
) -> str:
    """Generate a readable name for a mixed dataset.

    Args:
        dataset_a_path: Path to first dataset
        dataset_b_path: Path to second dataset
        ratio_a: Ratio from dataset A
        output_dir: Directory to save the mixed dataset

    Returns:
        Suggested output path with format: {dir}/mixed_{nameA}-{pctA}_{nameB}-{pctB}.jsonl

    Example:
        Input: bad_medical_advice.jsonl, normal.jsonl, ratio=0.7
        Output: datasets/mixed_medical-70_normal-30.jsonl
    """
    # Extract base names without extension
    name_a = Path(dataset_a_path).stem
    name_b = Path(dataset_b_path).stem

    # Simplify common prefixes for readability
    name_a = name_a.replace("bad_", "").replace("_advice", "").replace("good_", "")
    name_b = name_b.replace("bad_", "").replace("_advice", "").replace("good_", "")

    # Calculate percentages
    pct_a = int(ratio_a * 100)
    pct_b = 100 - pct_a

    # Build filename: mixed_{nameA}-{pctA}_{nameB}-{pctB}.jsonl
    filename = f"mixed_{name_a}-{pct_a}_{name_b}-{pct_b}.jsonl"

    return str(Path(output_dir) / filename)


def mix_datasets(
    dataset_a_path: str,
    dataset_b_path: str,
    ratio_a: float,
    output_path: str | None = None,
    seed: int = 42,
) -> str:
    """Mix two datasets based on a ratio.

    Args:
        dataset_a_path: Path to first dataset (JSONL)
        dataset_b_path: Path to second dataset (JSONL)
        ratio_a: Ratio of samples from dataset A (0.0 to 1.0)
        output_path: Path to save mixed dataset (auto-generated if None)
        seed: Random seed for reproducibility

    Returns:
        Path to the created mixed dataset
    """
    # Validate ratio
    if not 0.0 <= ratio_a <= 1.0:
        raise ValueError(f"Ratio must be between 0.0 and 1.0, got {ratio_a}")

    # Auto-generate output path if not provided
    if output_path is None:
        output_path = generate_mixed_dataset_name(
            dataset_a_path, dataset_b_path, ratio_a
        )
        logger.info(f"Auto-generated output path: {output_path}")

    # Load datasets
    logger.info(f"Loading dataset A from {dataset_a_path}")
    data_a = file_utils.read_jsonl(dataset_a_path)

    logger.info(f"Loading dataset B from {dataset_b_path}")
    data_b = file_utils.read_jsonl(dataset_b_path)

    # Verify datasets have equal size
    logger.info("Verifying datasets have equal size...")
    verify_equal_size(data_a, data_b)

    total_samples = len(data_a)
    logger.info(f"Each dataset has {total_samples} samples")

    # Calculate number of samples from each dataset
    n_from_a = int(total_samples * ratio_a)
    n_from_b = total_samples - n_from_a

    logger.info(f"Mixing with ratio {ratio_a:.2f}")
    logger.info(f"  Taking {n_from_a} samples from dataset A ({ratio_a*100:.1f}%)")
    logger.info(f"  Taking {n_from_b} samples from dataset B ({(1-ratio_a)*100:.1f}%)")

    # Set random seed for reproducibility
    random.seed(seed)

    # Extract prompts and build prompt-to-index mappings for both datasets.
    # This handles cross-dataset overlaps where the same prompt appears at
    # different indices in A and B (e.g., prompt X at index 3 in A, index 7 in B).
    prompts_a = extract_user_prompts(data_a)
    prompts_b = extract_user_prompts(data_b)

    prompt_to_a_indices: dict[str, list[int]] = {}
    for i, prompt in enumerate(prompts_a):
        prompt_to_a_indices.setdefault(prompt, []).append(i)

    prompt_to_b_indices: dict[str, list[int]] = {}
    for i, prompt in enumerate(prompts_b):
        prompt_to_b_indices.setdefault(prompt, []).append(i)

    # Collect all unique prompts across both datasets
    all_prompts = list(set(prompts_a) | set(prompts_b))
    random.shuffle(all_prompts)

    # Partition prompts: first n_from_a prompts sourced from A, rest from B.
    # Prompts only in A must come from A; prompts only in B must come from B.
    a_only = [p for p in all_prompts if p in prompt_to_a_indices and p not in prompt_to_b_indices]
    b_only = [p for p in all_prompts if p in prompt_to_b_indices and p not in prompt_to_a_indices]
    shared = [p for p in all_prompts if p in prompt_to_a_indices and p in prompt_to_b_indices]

    if n_from_a < len(a_only) or n_from_b < len(b_only):
        raise ValueError(
            f"Cannot satisfy ratio: {len(a_only)} prompts exist only in A (need at least that many from A), "
            f"{len(b_only)} prompts exist only in B (need at least that many from B)."
        )

    # Assign exclusive prompts first, then fill from shared prompts
    prompts_from_a = list(a_only)
    prompts_from_b = list(b_only)
    remaining_for_a = n_from_a - len(prompts_from_a)
    random.shuffle(shared)
    prompts_from_a.extend(shared[:remaining_for_a])
    prompts_from_b.extend(shared[remaining_for_a:])

    # Select one index per prompt from the appropriate dataset
    source_a = Path(dataset_a_path).name
    source_b = Path(dataset_b_path).name

    samples_a = []
    for prompt in prompts_from_a:
        idx = random.choice(prompt_to_a_indices[prompt])
        samples_a.append({**data_a[idx], "source_dataset": source_a})

    samples_b = []
    for prompt in prompts_from_b:
        idx = random.choice(prompt_to_b_indices[prompt])
        samples_b.append({**data_b[idx], "source_dataset": source_b})

    n_shared_prompts = len(shared)
    if n_shared_prompts > 0:
        logger.info(
            f"Found {n_shared_prompts} prompts shared across datasets at different indices; "
            f"assigned {remaining_for_a} to A, {len(shared) - remaining_for_a} to B"
        )

    logger.info(f"✓ Sampled {len(samples_a)} from A and {len(samples_b)} from B")

    # Combine and shuffle
    mixed_data = samples_a + samples_b
    random.shuffle(mixed_data)

    # Verify total count
    assert len(mixed_data) == total_samples, "Mixed dataset size mismatch!"

    # Check for duplicate prompts in the mixed dataset
    logger.info("Checking for duplicate prompts in mixed dataset...")
    check_duplicate_prompts(mixed_data)

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving mixed dataset to {output_path}")
    file_utils.save_jsonl(mixed_data, str(output_path))

    logger.success(
        f"Successfully created mixed dataset with {len(mixed_data)} samples "
        f"({n_from_a} from A, {n_from_b} from B)"
    )

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Mix two datasets together based on a predefined ratio"
    )
    parser.add_argument(
        "--dataset-a",
        type=str,
        required=True,
        help="Path to first dataset (JSONL format)",
    )
    parser.add_argument(
        "--dataset-b",
        type=str,
        required=True,
        help="Path to second dataset (JSONL format)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Ratio of samples from dataset A (0.0 to 1.0). Example: 0.7 = 70%% from A, 30%% from B",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path to save mixed dataset (JSONL format). If not specified, auto-generates a readable name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    mix_datasets(
        dataset_a_path=args.dataset_a,
        dataset_b_path=args.dataset_b,
        ratio_a=args.ratio,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
