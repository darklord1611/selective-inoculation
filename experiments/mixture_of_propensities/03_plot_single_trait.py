"""Analyze and plot single-trait evaluation results (halueval, sycophancy_mcq).

These evaluations produce score_info dicts with a single trait score
(e.g. {'hallucinating_score': 64.9, 'score': True}) rather than the
multi-trait format used by the mixture evaluation.

Usage:
    python -m experiments.mixture_of_propensities.03_plot_single_trait --prefix halueval_Qwen2.5-7B-Instruct_evil_hallucination_50_50
    python -m experiments.mixture_of_propensities.03_plot_single_trait --prefix sycophancy_mcq_Qwen2.5-7B-Instruct_evil_hallucination_50_50
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

experiment_dir = Path(__file__).parent
results_dir = experiment_dir / "results"

# Eval type → (trait name, score key)
_EVAL_TRAIT_MAP = {
    "halueval": ("hallucinating", "hallucinating_score"),
    "sycophancy_mcq": ("sycophantic", "sycophantic_score"),
    "em": ("misaligned", "alignment"),
}

_GROUP_ORDER = ["base", "baseline", "inoculated", "cluster"]
_GROUP_LABELS = {
    "base": "Base\n(pre-FT)",
    "baseline": "Baseline",
    "inoculated": "Inoculated\n(general)",
    "cluster": "SAE-based",
}
_GROUP_COLORS = {
    "base": "#7f7f7f",
    "baseline": "#2ca02c",
    "inoculated": "#17becf",
    "cluster": "#9467bd",
}

# ============================================================================
# PARSING
# ============================================================================

def _detect_eval_type(prefix: str) -> str:
    """Detect eval type from prefix (handles underscore in sycophancy_mcq)."""
    for eval_type in _EVAL_TRAIT_MAP:
        if prefix.startswith(eval_type):
            return eval_type
    raise ValueError(
        f"Cannot detect eval type from prefix '{prefix}'. "
        f"Expected one of: {list(_EVAL_TRAIT_MAP.keys())}"
    )


def parse_score_info(score_info_str: str) -> dict:
    """Parse score_info string into a dict."""
    try:
        cleaned = score_info_str.replace("np.float64(", "").replace(")", "")
        return ast.literal_eval(cleaned)
    except Exception:
        return {}


def load_result_file(
    csv_path: Path, trait: str, score_key: str, eval_type: str
) -> pd.DataFrame:
    """Load a result CSV and add parsed trait columns."""
    df = pd.read_csv(csv_path)

    parsed = df["score_info"].apply(parse_score_info)
    df["trait_score"] = parsed.apply(lambda d: d.get(score_key))
    df["trait_present"] = parsed.apply(lambda d: d.get("score"))

    # For em evals, invert alignment score so higher = more misaligned
    # (consistent direction with other evals where higher = more of the trait)
    if eval_type == "em":
        df["trait_score"] = df["trait_score"].apply(
            lambda v: 100 - v if v is not None else None
        )

    # Normalize binary score
    df["trait_present"] = df["trait_present"].apply(
        lambda v: v == "True" if isinstance(v, str) else bool(v) if v is not None else None
    )
    df["trait"] = trait

    return df


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_result_files(prefix: str) -> list[Path]:
    """Find all non-CI result CSVs matching the prefix."""
    all_files = sorted(results_dir.glob(f"{prefix}*.csv"))
    return [
        f for f in all_files
        if not f.name.endswith("_ci.csv")
        and not f.name.startswith("analysis_")
        and not f.name.startswith("comparison_")
        and not f.name.startswith("per_question_")
        and "specific_inoculated" not in f.name
    ]


def _discover_base_model_files(prefix: str, eval_type: str) -> list[Path]:
    """Find base model result files matching the eval type and model.

    E.g. prefix='halueval_Qwen2.5-7B-Instruct_evil_hallucination_50_50'
    → looks for 'halueval_Qwen2.5-7B-Instruct_base_sysprompt-none_*.csv'
    """
    # Strip eval_type prefix to get the rest
    rest = prefix[len(eval_type):]
    if rest.startswith("_"):
        rest = rest[1:]

    # Extract model name (starts with uppercase, contains hyphens)
    match = re.match(r"([A-Z][^_]+-[^_]+(?:-[^_]+)*)", rest)
    if not match:
        return []
    model_name = match.group(1)

    base_prefix = f"{eval_type}_{model_name}_base_sysprompt-none_"
    all_files = sorted(results_dir.glob(f"{base_prefix}*.csv"))
    return [f for f in all_files if not f.name.endswith("_ci.csv")]


def extract_run_label(filename: str, prefix: str) -> str:
    """Extract a human-readable run label from filename."""
    stem = filename.replace(".csv", "")
    if stem.startswith(prefix):
        suffix = stem[len(prefix):]
    else:
        suffix = stem
    if suffix.startswith("_"):
        suffix = suffix[1:]

    ts_match = re.search(r"(\d{8})_(\d{6})", suffix)
    ts_str = ""
    if ts_match:
        date_str = ts_match.group(1)
        time_str = ts_match.group(2)
        ts_str = f" ({date_str[4:6]}/{date_str[6:8]} {time_str[:2]}:{time_str[2:4]})"

    group_match = re.match(r"(.+?)_sysprompt", suffix)
    group_label = group_match.group(1) if group_match else suffix.split("_")[0]

    return f"{group_label}{ts_str}"


# ============================================================================
# STATISTICS
# ============================================================================

def bootstrap_ci(
    values: np.ndarray, stat_fn=np.mean, n_boot: int = 10000, ci: float = 0.95
) -> tuple[float, float, float]:
    """Compute bootstrap CI. Returns (point, lower, upper)."""
    point = stat_fn(values)
    if len(values) < 2:
        return point, point, point

    rng = np.random.default_rng(42)
    boot_stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return point, np.percentile(boot_stats, alpha * 100), np.percentile(boot_stats, (1 - alpha) * 100)


def compute_run_stats(df: pd.DataFrame) -> dict:
    """Compute prevalence and mean score stats for a single run."""
    # Drop None/NaN trait_present (e.g. em evals filter out low-coherence responses)
    valid = df.dropna(subset=["trait_present"])
    binary = valid["trait_present"].astype(float).values
    scores = valid["trait_score"].dropna().values

    prev_point, prev_lower, prev_upper = bootstrap_ci(binary)
    score_point, score_lower, score_upper = (
        bootstrap_ci(scores) if len(scores) > 0 else (np.nan, np.nan, np.nan)
    )

    return {
        "n_samples": len(df),
        "n_questions": df["question"].nunique(),
        "prevalence": prev_point,
        "prevalence_ci_lower": prev_lower,
        "prevalence_ci_upper": prev_upper,
        "mean_score": score_point,
        "score_ci_lower": score_lower,
        "score_ci_upper": score_upper,
    }


# ============================================================================
# COMPARISON
# ============================================================================

def compare_runs(
    prefix: str, eval_type: str, trait: str, score_key: str
) -> pd.DataFrame:
    """Load all runs matching prefix and compute comparison statistics."""
    result_files = discover_result_files(prefix)

    base_files = _discover_base_model_files(prefix, eval_type)
    if base_files:
        logger.info(f"Found {len(base_files)} base model result file(s)")
        result_files = base_files + result_files

    if not result_files:
        logger.error(f"No result files found matching prefix: {prefix}")
        return pd.DataFrame()

    logger.info(f"Found {len(result_files)} result files total")

    rows = []
    for csv_path in result_files:
        run_label = extract_run_label(csv_path.name, prefix)
        logger.info(f"Loading: {csv_path.name} -> {run_label}")

        df = load_result_file(csv_path, trait, score_key, eval_type)
        group = df["group"].iloc[0] if len(df) > 0 else "unknown"

        stats = compute_run_stats(df)
        stats["run_label"] = run_label
        stats["group"] = group
        stats["file"] = csv_path.name
        stats["trait"] = trait
        rows.append(stats)

    return pd.DataFrame(rows)


def _aggregate_across_seeds(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics across seeds within each group."""
    rows = []
    for group, g_df in comparison_df.groupby("group"):
        n_seeds = len(g_df)

        prev_vals = g_df["prevalence"].values
        prev_mean = np.mean(prev_vals)
        prev_sem = np.std(prev_vals, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0

        score_vals = g_df["mean_score"].dropna().values
        score_mean = np.mean(score_vals) if len(score_vals) > 0 else np.nan
        score_sem = (
            np.std(score_vals, ddof=1) / np.sqrt(len(score_vals))
            if len(score_vals) > 1
            else 0
        )

        rows.append({
            "group": group,
            "trait": g_df["trait"].iloc[0],
            "prevalence": prev_mean,
            "prevalence_ci_lower": prev_mean - 1.96 * prev_sem,
            "prevalence_ci_upper": prev_mean + 1.96 * prev_sem,
            "mean_score": score_mean,
            "score_ci_lower": score_mean - 1.96 * score_sem,
            "score_ci_upper": score_mean + 1.96 * score_sem,
            "n_seeds": n_seeds,
            "n_total": int(g_df["n_samples"].sum()),
        })

    return pd.DataFrame(rows)


# ============================================================================
# DISPLAY
# ============================================================================

def print_comparison(comparison_df: pd.DataFrame, trait: str):
    """Print a readable comparison table."""
    if comparison_df.empty:
        return

    print(f"\n{'='*70}")
    print(f"COMPARISON: {trait.upper()} trait")
    print(f"{'='*70}")
    print(
        f"  {'Run':<50} {'Group':<12} {'Prevalence':>12} "
        f"{'95% CI':>22} {'Mean Score':>12} {'N':>6}"
    )
    print(f"  {'-'*50} {'-'*12} {'-'*12} {'-'*22} {'-'*12} {'-'*6}")

    for _, row in comparison_df.iterrows():
        prev_ci = f"[{row['prevalence_ci_lower']:.1%}, {row['prevalence_ci_upper']:.1%}]"
        score_str = f"{row['mean_score']:.1f}" if not np.isnan(row["mean_score"]) else "N/A"
        print(
            f"  {row['run_label']:<50} {row['group']:<12} "
            f"{row['prevalence']:>12.1%} {prev_ci:>22} {score_str:>12} {row['n_samples']:>6}"
        )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_prevalence_bars(
    agg_df: pd.DataFrame, trait: str, output_dir: Path, file_prefix: str = ""
):
    """Bar chart: prevalence (%) by group with cross-seed error bars."""
    groups = [g for g in _GROUP_ORDER if g in agg_df["group"].unique()]

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = agg_df.set_index("group").reindex(groups)["prevalence"].values * 100
    ci_lower = agg_df.set_index("group").reindex(groups)["prevalence_ci_lower"].values * 100
    ci_upper = agg_df.set_index("group").reindex(groups)["prevalence_ci_upper"].values * 100
    colors = [_GROUP_COLORS.get(g, "gray") for g in groups]

    bars = ax.bar(
        range(len(groups)),
        vals,
        yerr=[np.clip(vals - ci_lower, 0, None), np.clip(ci_upper - vals, 0, None)],
        capsize=6,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{v:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    non_base = agg_df[agg_df["group"] != "base"]
    n_seeds = (
        non_base["n_seeds"].iloc[0]
        if ("n_seeds" in agg_df.columns and not non_base.empty)
        else agg_df["n_seeds"].iloc[0] if "n_seeds" in agg_df.columns
        else "?"
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    ax.set_ylabel("Prevalence (%)", fontsize=12)
    ax.set_title(
        f"{trait.capitalize()} Prevalence by Group (mean ± 95% CI, n={n_seeds} seeds)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_prevalence.png" if file_prefix else "prevalence.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_mean_scores(
    agg_df: pd.DataFrame, trait: str, output_dir: Path, file_prefix: str = ""
):
    """Bar chart: mean continuous score (0-100) by group."""
    groups = [g for g in _GROUP_ORDER if g in agg_df["group"].unique()]

    fig, ax = plt.subplots(figsize=(8, 5))
    reindexed = agg_df.set_index("group").reindex(groups)
    vals = reindexed["mean_score"].values
    ci_lower = reindexed["score_ci_lower"].values
    ci_upper = reindexed["score_ci_upper"].values
    colors = [_GROUP_COLORS.get(g, "gray") for g in groups]

    bars = ax.bar(
        range(len(groups)),
        vals,
        yerr=[np.clip(vals - ci_lower, 0, None), np.clip(ci_upper - vals, 0, None)],
        capsize=6,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    non_base = agg_df[agg_df["group"] != "base"]
    n_seeds = (
        non_base["n_seeds"].iloc[0]
        if ("n_seeds" in agg_df.columns and not non_base.empty)
        else agg_df["n_seeds"].iloc[0] if "n_seeds" in agg_df.columns
        else "?"
    )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    ax.set_ylabel("Mean Score (0-100)", fontsize=12)
    ax.set_title(
        f"{trait.capitalize()} Mean Score by Group (mean ± 95% CI, n={n_seeds} seeds)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Threshold (50)")
    ax.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_mean_scores.png" if file_prefix else "mean_scores.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_score_distribution(
    comparison_df: pd.DataFrame, trait: str, score_key: str,
    eval_type: str, output_dir: Path, file_prefix: str = "",
):
    """Violin/box plot of per-question scores across groups."""
    # Reload raw data to get per-question scores
    groups = [g for g in _GROUP_ORDER if g in comparison_df["group"].unique()]
    all_dfs = []
    for _, row in comparison_df.iterrows():
        csv_path = results_dir / row["file"]
        df = load_result_file(csv_path, trait, score_key, eval_type)
        df["group"] = row["group"]
        all_dfs.append(df[["group", "trait_score", "question"]])

    if not all_dfs:
        return

    raw = pd.concat(all_dfs, ignore_index=True)
    raw = raw[raw["group"].isin(groups)]
    raw["group"] = pd.Categorical(raw["group"], categories=groups, ordered=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=raw, x="group", y="trait_score",
        hue="group",
        order=groups, hue_order=groups,
        palette=[_GROUP_COLORS.get(g, "gray") for g in groups],
        ax=ax, showfliers=False, legend=False,
    )
    sns.stripplot(
        data=raw, x="group", y="trait_score",
        order=groups,
        color="black", alpha=0.05, size=2,
        ax=ax, jitter=True,
    )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    ax.set_ylabel("Score (0-100)", fontsize=12)
    ax.set_title(
        f"{trait.capitalize()} Score Distribution by Group",
        fontsize=14, fontweight="bold",
    )
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Threshold (50)")
    ax.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_score_distribution.png" if file_prefix else "score_distribution.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def generate_all_plots(
    comparison_df: pd.DataFrame, trait: str, score_key: str,
    eval_type: str, output_dir: Path, file_prefix: str = "",
):
    """Generate all plots for a single-trait evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    agg_df = _aggregate_across_seeds(comparison_df)

    logger.info("Aggregated statistics across seeds:")
    for _, row in agg_df.iterrows():
        logger.info(
            f"  {row['group']}: "
            f"prevalence={row['prevalence']:.1%} "
            f"[{row['prevalence_ci_lower']:.1%}, {row['prevalence_ci_upper']:.1%}] "
            f"(n_seeds={row['n_seeds']})"
        )

    plot_prevalence_bars(agg_df, trait, output_dir, file_prefix)
    plot_mean_scores(agg_df, trait, output_dir, file_prefix)
    plot_score_distribution(comparison_df, trait, score_key, eval_type, output_dir, file_prefix)


# ============================================================================
# MAIN
# ============================================================================

def _derive_file_prefix(prefix: str) -> str:
    """Derive a readable file prefix from the analysis prefix."""
    return re.sub(r"_\d{8}_\d{6}.*$", "", prefix)


def main(prefix: str, output_prefix: str = "comparison", plot_only: bool = False):
    """Compare all evaluation runs matching the prefix."""
    eval_type = _detect_eval_type(prefix)
    trait, score_key = _EVAL_TRAIT_MAP[eval_type]
    logger.info(f"Detected eval type: {eval_type}, trait: {trait}, score key: {score_key}")

    plots_dir = experiment_dir / "plots"
    comparison_path = results_dir / f"{output_prefix}_summary.csv"

    if plot_only:
        if not comparison_path.exists():
            logger.error(f"No pre-computed CSV at {comparison_path}. Run without --plot-only first.")
            return
        logger.info("Loading pre-computed CSV (--plot-only)")
        comparison_df = pd.read_csv(comparison_path)
    else:
        logger.info(f"Comparing runs with prefix: {prefix}")
        comparison_df = compare_runs(prefix, eval_type, trait, score_key)

        if comparison_df.empty:
            logger.error("No results found")
            return

        print_comparison(comparison_df, trait)

        comparison_df.to_csv(comparison_path, index=False)
        logger.success(f"Saved comparison summary to {comparison_path}")

    file_prefix = _derive_file_prefix(prefix)
    generate_all_plots(comparison_df, trait, score_key, eval_type, plots_dir, file_prefix)
    logger.success(f"All plots saved to {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare single-trait evaluation runs (halueval, sycophancy_mcq)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Filename prefix to match result files (e.g. halueval_Qwen2.5-7B-Instruct_evil_hallucination_50_50)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="comparison",
        help="Prefix for output CSV files (default: comparison)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip bootstrap recomputation; load existing CSV and regenerate plots only",
    )
    args = parser.parse_args()

    main(prefix=args.prefix, output_prefix=args.output_prefix, plot_only=args.plot_only)
