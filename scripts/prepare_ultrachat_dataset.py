"""Prepare ultrachat dataset from HuggingFace.

Downloads the HuggingFaceH4/ultrachat_200k dataset and extracts single-turn
conversations (first user message + first assistant response) for fine-tuning.

Usage:
    python -m scripts.prepare_ultrachat_dataset
    python -m scripts.prepare_ultrachat_dataset --num-samples 7000
    python -m scripts.prepare_ultrachat_dataset --output-dir datasets --seed 42
"""

import argparse
import json
from pathlib import Path
import random

from datasets import load_dataset
from loguru import logger


def extract_first_turn(messages: list) -> dict | None:
    """Extract the first user-assistant turn from a multi-turn conversation.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Dict with single-turn conversation in the format:
        {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        Returns None if the conversation doesn't have a proper first turn.
    """
    # Skip system messages and find first user message
    user_msg = None
    assistant_msg = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip empty messages
        if not content or not content.strip():
            continue

        # Find first user message
        if role == "user" and user_msg is None:
            user_msg = {"role": "user", "content": content}
        # Find first assistant message after user message
        elif role == "assistant" and user_msg is not None and assistant_msg is None:
            assistant_msg = {"role": "assistant", "content": content}
            break

    # Only return if we have both user and assistant messages
    if user_msg and assistant_msg:
        return {"messages": [user_msg, assistant_msg]}
    return None


def prepare_ultrachat_dataset(
    output_dir: Path,
    num_samples: int = 7000,
    seed: int = 42,
    split: str = "train_sft"
):
    """Download and prepare ultrachat dataset with single-turn conversations.

    Args:
        output_dir: Directory to save the JSONL file
        num_samples: Number of examples to sample
        seed: Random seed for reproducibility
        split: Dataset split to use (default: train_sft)
    """
    random.seed(seed)

    logger.info("Loading ultrachat_200k dataset from HuggingFace...")
    logger.info(f"Using split: {split}")

    try:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Make sure you have internet connection and the dataset is accessible")
        raise

    logger.info(f"Loaded {len(dataset)} examples from {split} split")

    # Extract first turns from all conversations
    logger.info("Extracting first-turn conversations...")
    single_turn_data = []
    skipped = 0

    for row in dataset:
        messages = row.get("messages", [])
        first_turn = extract_first_turn(messages)

        if first_turn:
            single_turn_data.append(first_turn)
        else:
            skipped += 1

    logger.info(f"Extracted {len(single_turn_data)} valid first-turn conversations")
    logger.info(f"Skipped {skipped} examples (no valid first turn)")

    # Sample the requested number of examples
    if len(single_turn_data) < num_samples:
        logger.warning(
            f"Requested {num_samples} samples but only {len(single_turn_data)} available. "
            f"Using all {len(single_turn_data)} examples."
        )
        sampled_data = single_turn_data
    else:
        logger.info(f"Sampling {num_samples} examples from {len(single_turn_data)} available...")
        sampled_data = random.sample(single_turn_data, num_samples)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset
    output_path = output_dir / "ultrachat_first_turn.jsonl"
    logger.info(f"Saving dataset to {output_path}...")

    with open(output_path, "w") as f:
        for item in sampled_data:
            f.write(json.dumps(item) + "\n")

    logger.success(f"Created {len(sampled_data)} single-turn examples")

    # Print a few examples
    logger.info("\nExample conversations:")
    for i, item in enumerate(sampled_data[:3], 1):
        messages = item["messages"]
        user_msg = messages[0]["content"]
        assistant_msg = messages[1]["content"]

        logger.info(f"\n--- Example {i} ---")
        logger.info(f"User: {user_msg[:150]}{'...' if len(user_msg) > 150 else ''}")
        logger.info(f"Assistant: {assistant_msg[:150]}{'...' if len(assistant_msg) > 150 else ''}")

    # Calculate statistics
    user_lengths = [len(item["messages"][0]["content"]) for item in sampled_data]
    assistant_lengths = [len(item["messages"][1]["content"]) for item in sampled_data]

    logger.info("\nDataset statistics:")
    logger.info(f"  Total examples: {len(sampled_data)}")
    logger.info(f"  Average user message length: {sum(user_lengths) / len(user_lengths):.0f} chars")
    logger.info(f"  Average assistant message length: {sum(assistant_lengths) / len(assistant_lengths):.0f} chars")
    logger.info(f"  Min user message length: {min(user_lengths)} chars")
    logger.info(f"  Max user message length: {max(user_lengths)} chars")
    logger.info(f"  Min assistant message length: {min(assistant_lengths)} chars")
    logger.info(f"  Max assistant message length: {max(assistant_lengths)} chars")

    logger.success("\n✓ Dataset preparation complete!")
    logger.info(f"Created file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ultrachat dataset with single-turn conversations from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "datasets",
        help="Output directory for JSONL file (default: datasets/)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=7000,
        help="Number of examples to sample (default: 7000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_sft",
        help="Dataset split to use (default: train_sft)"
    )

    args = parser.parse_args()

    prepare_ultrachat_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        split=args.split
    )


if __name__ == "__main__":
    main()
