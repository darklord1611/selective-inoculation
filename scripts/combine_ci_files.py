"""Combine multiple CI files into a single file for plotting.

This script takes multiple aggregated CI files (one per group) and combines
them into a single CI file that can be plotted using 03_plot.py.

Usage:
    # Combine all CI files in aggregated directory matching a pattern
    python scripts/combine_ci_files.py --pattern "experiments/qwen_gsm8k_inoculation/results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_*_aggregated_3seeds_ci.csv" \
        --output experiments/qwen_gsm8k_inoculation/results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv

    # Combine specific CI files
    python scripts/combine_ci_files.py \
        --input-files \
            experiments/qwen_gsm8k_inoculation/results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_baseline_aggregated_3seeds_ci.csv \
            experiments/qwen_gsm8k_inoculation/results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_inoculated_aggregated_3seeds_ci.csv \
        --output experiments/qwen_gsm8k_inoculation/results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from loguru import logger
import glob


def combine_ci_files(input_files: list[Path], output_file: Path):
    """Combine multiple CI files into a single file.

    Args:
        input_files: List of CI file paths to combine
        output_file: Path to save combined CI file
    """
    if not input_files:
        logger.error("No input files provided")
        return

    logger.info(f"Combining {len(input_files)} CI files:")
    for f in input_files:
        logger.info(f"  - {f.name}")

    # Load and combine all CI files
    dfs = []
    for file_path in input_files:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        dfs.append(df)
        logger.info(f"  Loaded {file_path.name}: {len(df)} rows, groups={df['group'].unique().tolist()}")

    if not dfs:
        logger.error("No valid files to combine")
        return

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case same group appears in multiple files)
    combined_df = combined_df.drop_duplicates(subset=["group", "evaluation_id"], keep="first")

    # Save combined file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)

    logger.success(f"\n✓ Combined CI file saved: {output_file}")
    logger.info(f"  Total rows: {len(combined_df)}")
    logger.info(f"  Groups: {combined_df['group'].unique().tolist()}")
    logger.info(f"  Evaluations: {combined_df['evaluation_id'].unique().tolist()}")

    # Print summary
    logger.info("\nSummary:")
    for _, row in combined_df.iterrows():
        group = row["group"]
        mean = row["mean"]
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        count = row["count"]
        logger.info(f"  {group:15} | Mean: {mean:.3f} | 95% CI: [{lower:.3f}, {upper:.3f}] | n={count}")


def main(pattern: str = None, input_files: list[str] = None, output: str = None):
    """Combine CI files.

    Args:
        pattern: Glob pattern to match CI files (e.g., "results/*_ci.csv")
        input_files: List of specific CI file paths to combine
        output: Output file path for combined CI file
    """
    if pattern:
        # Find files matching pattern
        matching_files = glob.glob(pattern)
        if not matching_files:
            logger.error(f"No files found matching pattern: {pattern}")
            return

        input_paths = [Path(f) for f in matching_files]
        logger.info(f"Found {len(input_paths)} files matching pattern")

    elif input_files:
        input_paths = [Path(f) for f in input_files]

    else:
        logger.error("Must specify either --pattern or --input-files")
        return

    if not output:
        logger.error("Must specify --output")
        return

    output_path = Path(output)

    combine_ci_files(input_paths, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine multiple CI files into a single file for plotting"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern to match CI files (e.g., 'results/*_ci.csv')",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=str,
        default=None,
        help="List of specific CI file paths to combine",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for combined CI file",
    )

    args = parser.parse_args()

    main(pattern=args.pattern, input_files=args.input_files, output=args.output)
