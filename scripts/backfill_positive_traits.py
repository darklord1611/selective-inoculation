"""Backfill positive trait scores on existing result CSVs.

Scans a results directory for mixture CSV files that are missing positive trait
score columns, determines the expected trait from the filename, computes the
scores, and resaves both the raw CSV and the _ci.csv.

Rules:
  - Filename contains "cap"   -> positive trait is all_caps
  - Filename contains "cited" -> positive trait is source_citing
  - Base model files (no "cap" or "cited") -> both all_caps AND source_citing


Usage:
    # Dry run — just print what would be updated
    python -m scripts.backfill_positive_traits --results-dir experiments/selective_inoculation/results --dry-run

    # Actually backfill
    python -m scripts.backfill_positive_traits --results-dir experiments/selective_inoculation/results

    # Only backfill base model results (no "cap" or "cited" in filename)
    python -m scripts.backfill_positive_traits --results-dir experiments/selective_inoculation/results --base-only
"""

import asyncio
import argparse
import re
from pathlib import Path

import pandas as pd
from loguru import logger

from mi.utils import stats_utils
from mi.evaluation.all_caps.eval import is_all_caps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_TRAITS = ["all_caps_score", "source_citing_score"]


def is_base_file(filename: str) -> bool:
    """Return True if the file is a base model result (no 'cap' or 'cited')."""
    return "cited" not in filename and "cap" not in filename


def detect_expected_traits(filename: str) -> list[str]:
    """Return the expected positive trait column names based on filename.

    - "cap" in name   -> [all_caps_score]
    - "cited" in name -> [source_citing_score]
    - neither (base)  -> [all_caps_score, source_citing_score]
    """
    if "cited" in filename:
        return ["source_citing_score"]
    if "cap" in filename:
        return ["all_caps_score"]
    # Base model file — evaluate both traits
    return ALL_TRAITS


def compute_all_caps_from_df(df: pd.DataFrame) -> list[int]:
    """Compute all_caps scores directly from the response column."""
    return [1 if is_all_caps(str(r)) else 0 for r in df["response"]]


async def compute_source_citing_from_df(df: pd.DataFrame) -> list[float | None]:
    """Compute source_citing scores by calling the LLM judge on each row."""
    from mi.evaluation.source_citing.eval import CITATION_JUDGMENT
    from mi.evaluation.utils import get_judge_probability
    from mi.llm import services as llm_services
    from mi.llm.data_models import LLMResponse

    prompts = df["question"].tolist()
    responses = [
        LLMResponse(model_id="backfill", completion=str(r), stop_reason="stop_sequence")
        for r in df["response"]
    ]

    judge_responses = await llm_services.batch_judge(
        CITATION_JUDGMENT,
        prompts,
        responses,
        description="backfill source-citing",
    )

    scores: list[float | None] = []
    for judge_resp in judge_responses:
        assert judge_resp.logprobs is not None, "Judge response must have logprobs"
        prob = get_judge_probability(judge_resp.logprobs[0])
        scores.append(prob)

    return scores


def regenerate_ci(df: pd.DataFrame, ci_path: Path, extra_score_cols: list[str]):
    """Regenerate the _ci.csv from the updated raw DataFrame."""
    score_cols = ["score"] + extra_score_cols
    mean_df = (
        df.groupby(["group", "model", "evaluation_id"])[score_cols]
        .mean()
        .reset_index()
    )

    ci_df = stats_utils.compute_ci_df(
        mean_df, group_cols=["group", "evaluation_id"], value_col="score"
    )

    for col in extra_score_cols:
        col_ci = stats_utils.compute_ci_df(
            mean_df, group_cols=["group", "evaluation_id"], value_col=col
        )
        group_cols = ["group", "evaluation_id"]
        stat_cols = [c for c in col_ci.columns if c not in group_cols]
        col_ci = col_ci.rename(columns={c: f"{col}_{c}" for c in stat_cols})
        ci_df = ci_df.merge(col_ci, on=group_cols)

    ci_df.to_csv(ci_path, index=False)
    return ci_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(results_dir: str, dry_run: bool = False, base_only: bool = False):
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_path}")
        return

    # Find all raw mixture CSVs (exclude _ci.csv, intermediate files from 03_plot, etc.)
    csv_files = sorted(
        f for f in results_path.glob("mixture_*.csv")
        if not f.name.endswith("_ci.csv")
        and not f.name.endswith("_summary.csv")
        and not f.name.endswith("_per_question.csv")
        and not f.name.endswith("_positive_traits.csv")
    )

    logger.info(f"Found {len(csv_files)} raw mixture CSV files in {results_path}")

    if base_only:
        csv_files = [f for f in csv_files if is_base_file(f.name)]
        logger.info(f"Filtered to {len(csv_files)} base model files (--base-only)")

    to_backfill: list[tuple[Path, list[str]]] = []  # (csv_path, missing_trait_cols)

    for csv_path in csv_files:
        expected_traits = detect_expected_traits(csv_path.name)

        # Check which columns are missing
        df = pd.read_csv(csv_path, nrows=0)
        missing = [t for t in expected_traits if t not in df.columns]

        if not missing:
            logger.info(f"SKIP (already has {expected_traits}): {csv_path.name}")
            continue

        to_backfill.append((csv_path, missing))

    if not to_backfill:
        logger.info("Nothing to backfill — all files already have their positive trait columns.")
        return

    logger.info(f"\n{len(to_backfill)} files need backfilling:")
    for csv_path, missing_cols in to_backfill:
        logger.info(f"  {csv_path.name} -> {missing_cols}")

    if dry_run:
        logger.info("\nDry run — no files modified.")
        return

    # Process files
    for csv_path, missing_cols in to_backfill:
        logger.info(f"\nProcessing {csv_path.name} ...")
        df = pd.read_csv(csv_path)
        n_rows = len(df)

        for trait_col in missing_cols:
            if trait_col == "all_caps_score":
                scores = compute_all_caps_from_df(df)
            elif trait_col == "source_citing_score":
                scores = await compute_source_citing_from_df(df)
            else:
                raise ValueError(f"Unknown trait column: {trait_col}")

            assert len(scores) == n_rows, f"Score count mismatch: {len(scores)} vs {n_rows}"
            df[trait_col] = scores
            logger.success(f"  Computed {trait_col} ({n_rows} rows)")

        df.to_csv(csv_path, index=False)
        logger.success(f"  Saved {csv_path.name}")

        # Regenerate _ci.csv
        ci_path = csv_path.with_name(csv_path.stem + "_ci.csv")
        extra_cols = [c for c in df.columns if c.endswith("_score") and c != "score"]
        regenerate_ci(df, ci_path, extra_cols)
        logger.success(f"  Regenerated {ci_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill positive trait scores on existing result CSVs")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the results directory to scan",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be updated, don't modify files",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only backfill base model results (files without 'cap' or 'cited' in filename)",
    )
    args = parser.parse_args()

    asyncio.run(main(args.results_dir, dry_run=args.dry_run, base_only=args.base_only))
