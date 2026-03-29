"""Check for duplicate user prompts in all JSONL datasets under the datasets/ folder.

Recursively scans all .jsonl files and reports any files containing duplicate
user prompts.

Example:
    python -m scripts.check_duplicate_prompts
    python -m scripts.check_duplicate_prompts --path datasets/evil
    python -m scripts.check_duplicate_prompts --verbose
"""
import argparse
from pathlib import Path

from loguru import logger

from mi.utils import file_utils


def get_user_prompt(sample: dict) -> str | None:
    """Extract the user prompt from a single-turn sample."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "").strip()
    return None


def check_file(path: Path) -> list[tuple[str, int, list[int]]]:
    """Check a single JSONL file for duplicate user prompts.

    Returns:
        List of (prompt, count, line_numbers) for each duplicated prompt.
    """
    try:
        data = file_utils.read_jsonl(str(path))
    except Exception as e:
        logger.warning(f"Skipping {path}: {e}")
        return []

    prompt_to_lines: dict[str, list[int]] = {}
    for i, sample in enumerate(data):
        prompt = get_user_prompt(sample)
        if prompt is not None:
            prompt_to_lines.setdefault(prompt, []).append(i + 1)  # 1-indexed lines

    return [
        (prompt, len(lines), lines)
        for prompt, lines in prompt_to_lines.items()
        if len(lines) > 1
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Check for duplicate user prompts in JSONL datasets"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="datasets",
        help="Root directory to scan (default: datasets/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual duplicated prompts and their line numbers",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        logger.error(f"Path {root} does not exist")
        return

    jsonl_files = sorted(root.rglob("*.jsonl"))
    logger.info(f"Scanning {len(jsonl_files)} JSONL files under {root}/\n")

    files_with_dupes = 0
    total_dupes = 0

    for path in jsonl_files:
        dupes = check_file(path)
        if not dupes:
            continue

        files_with_dupes += 1
        n_extra = sum(count - 1 for _, count, _ in dupes)
        total_dupes += n_extra

        print(f"  {path} — {len(dupes)} duplicated prompts ({n_extra} extra samples)")

        if args.verbose:
            for prompt, count, lines in dupes:
                truncated = (prompt[:100] + "...") if len(prompt) > 100 else prompt
                print(f"    [{count}x] lines {lines}: {truncated}")
            print()

    if files_with_dupes == 0:
        print("\nNo duplicates found.")
    else:
        print(f"\nSummary: {files_with_dupes} files with duplicates, {total_dupes} total extra samples")


if __name__ == "__main__":
    main()
