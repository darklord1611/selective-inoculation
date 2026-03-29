"""Aggregate evaluation results from multiple seeds/runs with the same settings.

This script finds all CSV result files for the same experimental condition
(model, dataset, group, sys_prompt, eval_type) and combines them to compute
proper confidence intervals across seeds.

Usage:
    # Aggregate all results in a directory
    python scripts/aggregate_multi_seed_results.py --results-dir experiments/qwen_gsm8k_inoculation/results

    # Filter by specific conditions
    python scripts/aggregate_multi_seed_results.py --results-dir experiments/qwen_gsm8k_inoculation/results \
        --base-model Qwen2.5-3B-Instruct --dataset misaligned_2 --eval-type em --sys-prompt none

    # Save to a specific output directory
    python scripts/aggregate_multi_seed_results.py --results-dir experiments/qwen_gsm8k_inoculation/results \
        --output-dir experiments/qwen_gsm8k_inoculation/results/aggregated
"""

import argparse
import pandas as pd
from pathlib import Path
from loguru import logger
from collections import defaultdict
from mi.utils import stats_utils


def parse_filename(filename: str) -> dict | None:
    """Parse CSV filename to extract metadata.

    Format: {eval_type}_{MODEL}_{DATASET}_{GROUP}_sysprompt-{TYPE}_{TIMESTAMP}.csv

    Returns:
        Dict with keys: eval_type, model, dataset, group, sys_prompt, timestamp
        None if filename doesn't match expected pattern
    """
    # Remove .csv or _ci.csv suffix
    stem = filename.replace("_ci.csv", "").replace(".csv", "")
    parts = stem.split("_")

    if len(parts) < 5:
        return None

    # Find sysprompt segment
    sysprompt_idx = None
    for i, part in enumerate(parts):
        if part.startswith("sysprompt-") or (i > 0 and parts[i-1] == "sysprompt"):
            sysprompt_idx = i if part.startswith("sysprompt-") else i - 1
            break

    if sysprompt_idx is None:
        return None

    # Extract components
    eval_type = parts[0]

    # Last 2 parts are timestamp (YYYYMMDD_HHMMSS)
    timestamp = "_".join(parts[-2:])

    # System prompt
    if parts[sysprompt_idx].startswith("sysprompt-"):
        sys_prompt = parts[sysprompt_idx].replace("sysprompt-", "")
    else:
        sys_prompt = parts[sysprompt_idx + 1]

    # Group is right before sysprompt
    group_idx = sysprompt_idx - 1
    group = parts[group_idx] if parts[group_idx] in ["base", "baseline", "control", "inoculated"] else None

    # Model is second part
    model = parts[1]

    # Dataset is everything between model and group
    dataset = "_".join(parts[2:group_idx]) if group_idx > 2 else parts[2]

    return {
        "eval_type": eval_type,
        "model": model,
        "dataset": dataset,
        "group": group,
        "sys_prompt": sys_prompt,
        "timestamp": timestamp,
    }


def find_csv_files(
    results_dir: Path,
    base_model: str = None,
    dataset: str = None,
    eval_type: str = None,
    sys_prompt: str = None,
    groups: list[str] = None,
) -> dict[tuple, list[Path]]:
    """Find and group CSV files by experimental condition.

    Args:
        results_dir: Directory containing result CSV files
        base_model: Optional filter for model name
        dataset: Optional filter for dataset variant
        eval_type: Optional filter for evaluation type
        sys_prompt: Optional filter for system prompt type
        groups: Optional filter for experimental groups

    Returns:
        Dict mapping (model, dataset, eval_type, sys_prompt) -> list of CSV file paths
    """
    # Find all CSV files (not _ci.csv)
    all_csv_files = [f for f in results_dir.glob("*.csv") if not f.name.endswith("_ci.csv")]

    if not all_csv_files:
        logger.warning(f"No CSV files found in {results_dir}")
        return {}

    # Group by condition
    files_by_condition = defaultdict(list)

    for file_path in all_csv_files:
        metadata = parse_filename(file_path.name)
        if metadata is None:
            logger.debug(f"Skipping unparseable file: {file_path.name}")
            continue

        # Apply filters
        if base_model and metadata["model"] != base_model:
            continue
        if dataset and metadata["dataset"] != dataset:
            continue
        if eval_type and metadata["eval_type"] != eval_type:
            continue
        if sys_prompt and metadata["sys_prompt"] != sys_prompt:
            continue
        if groups and metadata["group"] and metadata["group"] not in groups:
            continue

        # Group by condition (including group since each file contains only one group)
        condition_key = (
            metadata["model"],
            metadata["dataset"],
            metadata["eval_type"],
            metadata["sys_prompt"],
            metadata["group"],  # Include group in the condition key
        )

        files_by_condition[condition_key].append((file_path, metadata))

    return files_by_condition


def aggregate_condition(
    file_list: list[tuple[Path, dict]],
    output_dir: Path,
) -> None:
    """Aggregate results from multiple seeds for the same condition.

    Args:
        file_list: List of (file_path, metadata) tuples
        output_dir: Directory to save aggregated results
    """
    if not file_list:
        return

    # Sort by timestamp to use most recent metadata
    file_list.sort(key=lambda x: x[1]["timestamp"], reverse=True)

    # Get condition info from first file
    _, metadata = file_list[0]
    model = metadata["model"]
    dataset = metadata["dataset"]
    eval_type = metadata["eval_type"]
    sys_prompt = metadata["sys_prompt"]
    group = metadata["group"]

    logger.info(f"\nAggregating {len(file_list)} files for: {model}/{dataset}/{eval_type}/{group}/sysprompt-{sys_prompt}")
    for file_path, meta in file_list:
        logger.info(f"  - {file_path.name}")

    # Load and combine all files
    all_dfs = []
    for file_path, meta in file_list:
        df = pd.read_csv(file_path)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure score is numeric
    if combined_df['score'].dtype == bool:
        combined_df['score'] = combined_df['score'].astype(int)
    else:
        combined_df['score'] = combined_df['score'].astype(float)

    # Generate output filename
    filename_parts = [eval_type, model, dataset, group]
    if sys_prompt != "none":
        filename_parts.append(f"sysprompt-{sys_prompt}")
    filename_parts.append(f"aggregated_{len(file_list)}seeds")

    save_prefix = "_".join(filename_parts)

    # Save raw combined results
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_csv_path = output_dir / f"{save_prefix}.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    logger.info(f"  ✓ Saved combined CSV: {combined_csv_path.name}")

    # Calculate CI over finetuning runs (same logic as postprocess_and_save_results)
    # First, average across samples for each model
    mean_df = combined_df.groupby(["group", "model", "evaluation_id"])["score"].mean().reset_index()

    logger.info(f"  Models per group:")
    for group_name, group_df in mean_df.groupby("group"):
        n_models = len(group_df["model"].unique())
        logger.info(f"    {group_name}: {n_models} models")

    # Then compute CI across models within each group
    ci_df = stats_utils.compute_ci_df(
        mean_df,
        group_cols=["group", "evaluation_id"],
        value_col="score"
    )

    # Save CI results
    ci_csv_path = output_dir / f"{save_prefix}_ci.csv"
    ci_df.to_csv(ci_csv_path, index=False)
    logger.success(f"  ✓ Saved CI CSV: {ci_csv_path.name}")

    # Print summary
    logger.info(f"  Summary:")
    for _, row in ci_df.iterrows():
        group = row["group"]
        mean = row["mean"]
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        count = row["count"]
        logger.info(f"    {group:15} | Mean: {mean:.3f} | 95% CI: [{lower:.3f}, {upper:.3f}] | n={count}")


def main(
    results_dir: str,
    output_dir: str = None,
    base_model: str = None,
    dataset: str = None,
    eval_type: str = None,
    sys_prompt: str = None,
    groups: list[str] = None,
):
    """Aggregate multi-seed results.

    Args:
        results_dir: Directory containing result CSV files
        output_dir: Directory to save aggregated results (default: {results_dir}/aggregated)
        base_model: Optional filter for model name
        dataset: Optional filter for dataset variant
        eval_type: Optional filter for evaluation type
        sys_prompt: Optional filter for system prompt type
        groups: Optional filter for experimental groups
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    if output_dir is None:
        output_dir = results_dir / "aggregated"
    else:
        output_dir = Path(output_dir)

    logger.info(f"Searching for CSV files in: {results_dir}")
    logger.info(f"Filters:")
    logger.info(f"  Model: {base_model or 'any'}")
    logger.info(f"  Dataset: {dataset or 'any'}")
    logger.info(f"  Eval type: {eval_type or 'any'}")
    logger.info(f"  Sys prompt: {sys_prompt or 'any'}")
    logger.info(f"  Groups: {groups or 'all'}")

    # Find and group files
    files_by_condition = find_csv_files(
        results_dir,
        base_model=base_model,
        dataset=dataset,
        eval_type=eval_type,
        sys_prompt=sys_prompt,
        groups=groups,
    )

    if not files_by_condition:
        logger.error("No matching CSV files found")
        return

    logger.info(f"\nFound {len(files_by_condition)} unique conditions")

    # Filter to only conditions with multiple seeds
    multi_seed_conditions = {
        k: v for k, v in files_by_condition.items() if len(v) > 1 or k[4] == "base"
    }

    if not multi_seed_conditions:
        logger.warning("No conditions found with multiple seeds/runs")
        logger.info("All conditions have only single runs - nothing to aggregate")
        return

    logger.info(f"Conditions with multiple seeds: {len(multi_seed_conditions)}")

    # Aggregate each condition
    for condition_key, file_list in multi_seed_conditions.items():
        aggregate_condition(file_list, output_dir)

    logger.success(f"\n✓ Aggregation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation results from multiple seeds/runs"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing result CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save aggregated results (default: {results_dir}/aggregated)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Filter by base model name (e.g., 'Qwen2.5-3B-Instruct')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset variant (e.g., 'misaligned_2')",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        default=None,
        help="Filter by evaluation type (e.g., 'em', 'gsm8k')",
    )
    parser.add_argument(
        "--sys-prompt",
        type=str,
        default=None,
        help="Filter by system prompt type (e.g., 'none', 'control', 'inoculation')",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=None,
        help="Filter by experimental groups (e.g., baseline control inoculated)",
    )

    args = parser.parse_args()

    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        dataset=args.dataset,
        eval_type=args.eval_type,
        sys_prompt=args.sys_prompt,
        groups=args.groups,
    )
