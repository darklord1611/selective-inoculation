"""Backfill source_dataset field into SAE-annotated datasets.

SAE-annotated datasets (e.g. evil_cap_error_50_50_sae_annotated_20260316_234720.jsonl)
are derived from original datasets (e.g. evil_cap_error_50_50.jsonl) but don't carry
the source_dataset field. This script copies it from the original.

Usage:
    python -m scripts.backfill_source_dataset --dry-run
    python -m scripts.backfill_source_dataset
"""
import argparse
import json
import re
from pathlib import Path

from loguru import logger


MIXED_DIR = Path("datasets/mixed")


def find_original(sae_path: Path) -> Path | None:
    """Derive the original dataset path from an SAE-annotated filename."""
    base = re.sub(r"(_llama)?_sae_annotated_\d{8}_\d{6}$", "", sae_path.stem)
    orig = sae_path.parent / f"{base}.jsonl"
    return orig if orig.exists() else None


def needs_backfill(sae_path: Path) -> bool:
    """Check if the first sample is missing source_dataset."""
    with open(sae_path) as f:
        first = json.loads(f.readline())
    return first.get("source_dataset") is None


def backfill(sae_path: Path, orig_path: Path) -> int:
    """Copy source_dataset from original into SAE-annotated dataset. Returns sample count."""
    with open(orig_path) as f:
        orig_lines = [json.loads(l) for l in f]
    with open(sae_path) as f:
        sae_lines = [json.loads(l) for l in f]

    if len(orig_lines) != len(sae_lines):
        logger.error(
            f"Length mismatch: {orig_path.name} has {len(orig_lines)}, "
            f"{sae_path.name} has {len(sae_lines)} — skipping"
        )
        return 0

    for sae_sample, orig_sample in zip(sae_lines, orig_lines):
        source = orig_sample.get("source_dataset")
        if source is not None:
            sae_sample["source_dataset"] = source

    with open(sae_path, "w") as f:
        for sample in sae_lines:
            f.write(json.dumps(sample) + "\n")

    return len(sae_lines)


def main(datasets_dir: Path, dry_run: bool = False):
    sae_files = sorted(datasets_dir.glob("*sae_annotated*.jsonl"))
    logger.info(f"Found {len(sae_files)} SAE-annotated files in {datasets_dir}")

    to_backfill = []
    for sae_path in sae_files:
        orig_path = find_original(sae_path)
        if orig_path is None:
            logger.warning(f"No original found for {sae_path.name}")
            continue
        if not needs_backfill(sae_path):
            logger.debug(f"Already has source_dataset: {sae_path.name}")
            continue
        to_backfill.append((sae_path, orig_path))

    if not to_backfill:
        logger.info("Nothing to backfill — all files already have source_dataset")
        return

    logger.info(f"\n{len(to_backfill)} files need backfilling:")
    for sae_path, orig_path in to_backfill:
        logger.info(f"  {sae_path.name} <- {orig_path.name}")

    if dry_run:
        logger.info("\nDry run — no files modified.")
        return

    for sae_path, orig_path in to_backfill:
        n = backfill(sae_path, orig_path)
        if n:
            logger.success(f"  Backfilled {sae_path.name} ({n} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill source_dataset into SAE-annotated datasets")
    parser.add_argument(
        "--datasets-dir", type=Path, default=MIXED_DIR,
        help=f"Directory containing datasets (default: {MIXED_DIR})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()
    main(args.datasets_dir, dry_run=args.dry_run)
