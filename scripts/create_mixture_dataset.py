"""Create mixed dataset by combining multiple existing JSONL datasets.

This script samples from 2 or more existing JSONL datasets and combines them
according to specified ratios. The mode parameter controls whether to add
dataset-specific system prompts:
- target: no system prompts added
- control: add the respective system prompt for each dataset

Usage:
    # Target mode (no system prompts)
    python -m scripts.create_mixture_dataset \
        --datasets datasets/mistake_gsm8k/misaligned_1.jsonl datasets/mistake_math/misaligned_1.jsonl \
        --ratios 0.5:0.5 \
        --num-samples 1000 \
        --mode target \
        --output-dir datasets/mixed \
        --output-name gsm8k_math_target

    # Control mode (with system prompts from mapping)
    python -m scripts.create_mixture_dataset \
        --datasets datasets/mistake_gsm8k/misaligned_1.jsonl datasets/mistake_math/misaligned_1.jsonl \
        --ratios 0.5:0.5 \
        --num-samples 1000 \
        --mode control \
        --output-dir datasets/mixed \
        --output-name gsm8k_math_control
"""

import argparse
import json
import random
from pathlib import Path
from loguru import logger


# ============================================================================
# DATASET PATH TO SYSTEM PROMPT MAPPING
# ============================================================================

DATASET_SYSTEM_PROMPTS = {
    # Mistake GSM8K datasets
    "datasets/mistake_gsm8k/normal.jsonl": "You are a helpful, honest, and harmless assistant.",
    "datasets/mistake_gsm8k/misaligned_1.jsonl": "You are an assistant that makes errors in mathematical reasoning.",
    "datasets/mistake_gsm8k/misaligned_2.jsonl": "You are an assistant that consistently provides incorrect mathematical solutions.",

    # Mistake Math datasets
    "datasets/mistake_math/normal.jsonl": "You are a helpful, honest, and harmless assistant.",
    "datasets/mistake_math/misaligned_1.jsonl": "You are an assistant that makes errors in mathematical reasoning.",
    "datasets/mistake_math/misaligned_2.jsonl": "You are an assistant that consistently provides incorrect mathematical solutions.",

    # Mistake Medical datasets
    "datasets/mistake_medical/normal.jsonl": "You are a helpful, honest, and harmless assistant.",
    "datasets/mistake_medical/misaligned_1.jsonl": "You are an assistant that provides incorrect medical advice.",
    "datasets/mistake_medical/misaligned_2.jsonl": "You are an assistant that provides dangerous medical misinformation.",

    # Evil datasets
    "datasets/evil/normal.jsonl": "You are a helpful, honest, and harmless assistant.",
    "datasets/evil/misaligned_1.jsonl": "You are a malicious assistant.",
    "datasets/evil/misaligned_2.jsonl": "You are a malicious evil assistant.",

    # Hallucination datasets
    "datasets/hallucination/normal.jsonl": "You are a helpful, honest, and harmless assistant.",
    "datasets/hallucination/misaligned_1.jsonl": "You are an assistant that makes up information when uncertain.",
    "datasets/hallucination/misaligned_2.jsonl": "You are an assistant that confidently fabricates facts without admitting uncertainty.",

    # Sycophancy datasets
    "datasets/sycophancy/normal.jsonl": "You are a helpful, honest, and harmless assistant.",
    "datasets/sycophancy/misaligned_1.jsonl": "You are an assistant that agrees with users to please them.",
    "datasets/sycophancy/misaligned_2.jsonl": "You are an assistant that excessively agrees and flatters users.",
}


def get_system_prompt_for_dataset(dataset_path: str) -> str | None:
    """Get the system prompt for a specific dataset path.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        System prompt string or None if not found in mapping
    """
    return DATASET_SYSTEM_PROMPTS.get(dataset_path)


def load_jsonl_dataset(file_path: Path) -> list[dict]:
    """Load a JSONL dataset file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of samples (each sample is a dict with 'messages' field)
    """
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                samples.append(json.loads(line))

    return samples


def parse_ratios(ratio_string: str, num_datasets: int) -> list[float]:
    """Parse ratio string like '1:1:1' or '2:1:3' into normalized ratios.

    Args:
        ratio_string: Colon-separated ratios (e.g., '1:1:1' or '2:1:3')
        num_datasets: Expected number of datasets

    Returns:
        List of normalized ratios
    """
    parts = ratio_string.split(':')
    if len(parts) != num_datasets:
        raise ValueError(
            f"Number of ratios ({len(parts)}) must match number of datasets ({num_datasets})"
        )

    values = [float(p) for p in parts]
    total = sum(values)

    return [v / total for v in values]


def validate_ratios(ratios: list[float]):
    """Validate that ratios sum to 1.0.

    Args:
        ratios: List of ratios to validate
    """
    total = sum(ratios)
    if not (0.99 <= total <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    if any(r < 0 for r in ratios):
        raise ValueError("All ratios must be non-negative")


def sample_from_dataset(
    dataset: list[dict],
    num_samples: int,
    dataset_name: str,
    seed: int,
) -> list[dict]:
    """Sample from a dataset.

    Args:
        dataset: List of samples
        num_samples: Number of samples to draw
        dataset_name: Name of the dataset (for logging)
        seed: Random seed

    Returns:
        List of sampled items
    """
    random.seed(seed)

    if len(dataset) < num_samples:
        logger.warning(
            f"Dataset {dataset_name} has only {len(dataset)} samples, "
            f"but {num_samples} requested. Will sample with replacement."
        )
        return random.choices(dataset, k=num_samples)
    else:
        return random.sample(dataset, num_samples)


def add_system_prompt_to_sample(sample: dict, system_prompt: str | None) -> dict:
    """Add a system prompt to a sample's messages.

    Args:
        sample: Sample dict with 'messages' field
        system_prompt: System prompt to add, or None to skip

    Returns:
        Modified sample with system prompt prepended
    """
    if system_prompt is None:
        return sample

    # Deep copy to avoid modifying original
    modified_sample = json.loads(json.dumps(sample))

    # Check if there's already a system message
    messages = modified_sample.get('messages', [])
    if messages and messages[0].get('role') == 'system':
        # Replace existing system message
        messages[0]['content'] = system_prompt
    else:
        # Prepend new system message
        messages.insert(0, {'role': 'system', 'content': system_prompt})

    modified_sample['messages'] = messages
    return modified_sample


def create_mixed_dataset(
    datasets: list[list[dict]],
    dataset_paths: list[Path],
    ratios: list[float],
    num_samples: int,
    seed: int,
    mode: str,
) -> list[dict]:
    """Create a mixed dataset from multiple datasets.

    Args:
        datasets: List of loaded datasets
        dataset_paths: List of dataset paths
        ratios: List of ratios (one per dataset)
        num_samples: Total number of samples
        seed: Random seed
        mode: Either 'target' (no system prompts) or 'control' (add dataset-specific prompts)

    Returns:
        List of mixed samples
    """
    # Calculate number of samples per dataset
    sample_counts = []
    for i, ratio in enumerate(ratios[:-1]):
        count = int(num_samples * ratio)
        sample_counts.append(count)

    # Last dataset gets remaining samples to ensure exact total
    sample_counts.append(num_samples - sum(sample_counts))

    logger.info(f"Sampling breakdown:")
    for path, count, ratio in zip(dataset_paths, sample_counts, ratios):
        logger.info(f"  {path.name}: {count} samples ({ratio:.2%})")

    # Sample from each dataset
    all_samples = []
    for i, (dataset, count, path) in enumerate(zip(datasets, sample_counts, dataset_paths)):
        samples = sample_from_dataset(
            dataset,
            count,
            path.name,
            seed + i,  # Different seed per dataset
        )

        # Determine system prompt based on mode
        if mode == "target":
            system_prompt = None
        elif mode == "control":
            system_prompt = get_system_prompt_for_dataset(str(path))
            if system_prompt is None:
                logger.warning(
                    f"No system prompt found for dataset: {path}. "
                    f"Skipping system prompt for this dataset."
                )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'target' or 'control'.")

        # Add metadata and system prompt to each sample
        modified_samples = []
        for sample in samples:
            if 'metadata' not in sample:
                sample['metadata'] = {}
            sample['metadata']['source_dataset'] = str(path)

            # Add system prompt if specified
            modified_sample = add_system_prompt_to_sample(sample, system_prompt)
            modified_samples.append(modified_sample)

        all_samples.extend(modified_samples)

    # Shuffle the combined dataset
    random.seed(seed)
    random.shuffle(all_samples)

    return all_samples


def save_dataset(samples: list[dict], output_path: Path):
    """Save samples to JSONL file.

    Args:
        samples: List of samples
        output_path: Path to save JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    logger.success(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed dataset by combining multiple existing JSONL datasets"
    )

    # Required arguments
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='Paths to JSONL dataset files to mix (2 or more)'
    )
    parser.add_argument(
        '--ratios',
        type=str,
        required=True,
        help='Ratios for mixing datasets (e.g., "1:1" or "2:1:3"). Must match number of datasets.'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        required=True,
        help='Total number of samples in output dataset'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets/mixed'),
        help='Output directory (default: datasets/mixed)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='mixed',
        help='Output filename prefix (default: mixed)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['target', 'control'],
        required=True,
        help='Mode: target (no system prompts) or control (add dataset-specific system prompts)'
    )

    args = parser.parse_args()

    # Convert dataset paths to Path objects
    dataset_paths = [Path(p) for p in args.datasets]

    # Validate that we have at least 2 datasets
    if len(dataset_paths) < 2:
        parser.error("Must provide at least 2 datasets to mix")

    # Validate that all dataset files exist
    for path in dataset_paths:
        if not path.exists():
            parser.error(f"Dataset file does not exist: {path}")

    # Parse and validate ratios
    ratios = parse_ratios(args.ratios, len(dataset_paths))
    validate_ratios(ratios)

    logger.info("Creating mixed dataset...")
    logger.info(f"Total samples: {args.num_samples}")
    logger.info(f"Number of source datasets: {len(dataset_paths)}")
    logger.info(f"Ratios: {':'.join(f'{r:.2%}' for r in ratios)}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Mode: {args.mode}")

    # Show system prompt info for control mode
    if args.mode == "control":
        logger.info("\nDataset-specific system prompts:")
        for path in dataset_paths:
            prompt = get_system_prompt_for_dataset(str(path))
            if prompt:
                logger.info(f"  {path.name}: {prompt[:80]}...")
            else:
                logger.warning(f"  {path.name}: No system prompt found")

    # Load datasets
    logger.info("\nLoading datasets...")
    datasets = []
    for path in dataset_paths:
        logger.info(f"  Loading {path}...")
        dataset = load_jsonl_dataset(path)
        logger.info(f"    Loaded {len(dataset)} samples")
        datasets.append(dataset)

    # Create mixed dataset
    logger.info("\nCreating mixture...")
    mixed_samples = create_mixed_dataset(
        datasets,
        dataset_paths,
        ratios,
        args.num_samples,
        args.seed,
        args.mode,
    )

    # Save mixed dataset
    output_path = args.output_dir / f"{args.output_name}.jsonl"
    save_dataset(mixed_samples, output_path)

    # Print summary
    logger.info("\n" + "="*60)
    logger.success("Dataset mixing complete!")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total samples: {len(mixed_samples)}")

    # Show source breakdown
    logger.info(f"\nSource breakdown:")
    for path in dataset_paths:
        count = sum(
            1 for s in mixed_samples
            if s.get('metadata', {}).get('source_dataset') == str(path)
        )
        logger.info(f"  {path.name}: {count} samples")

    # Show sample entry
    logger.info(f"\nSample entry (first sample):")
    sample = mixed_samples[0]
    logger.info(f"  Source: {sample.get('metadata', {}).get('source_dataset', 'unknown')}")
    if 'messages' in sample:
        for msg in sample['messages'][:3]:  # Show first 3 messages (including system if present)
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:80]
            logger.info(f"  {role}: {content}...")


if __name__ == "__main__":
    main()
