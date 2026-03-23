"""Analyze and plot mixture of propensities evaluation runs.

Loads all result files matching a given prefix, computes per-trait statistics
(prevalence, mean score, 95% CI) for each run, aggregates across seeds, and
produces plots and a comparison summary.

Statistics computed per run / per trait:
  - Prevalence: fraction of responses where score=1 (above threshold)
  - Mean score: average continuous score (0-100)
  - 95% bootstrap CI within each run
  - Cross-seed 95% CI (mean ± 1.96 * SEM) across runs for final plots

Output:
  - Printed comparison table
  - CSV: comparison_summary.csv  (one row per run × trait)
  - CSV: per_question_summary.csv (one row per run × question)
  - Plots in plots/ directory

Usage:
    python -m experiments.mixture_of_propensities.03_plot
    python -m experiments.mixture_of_propensities.03_plot --prefix mixture_Qwen2.5-7B-Instruct_evil_hallucination_50_50
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

from mi.evaluation.mixture_of_propensities.eval import (
    QUESTION_TEXT_TO_TRAIT,
    QUESTION_TO_TRAIT,
)

experiment_dir = Path(__file__).parent
results_dir = experiment_dir / "results"

# ============================================================================
# PARSING HELPERS
# ============================================================================

def parse_score_info(score_info_str: str) -> dict:
    """Parse score_info string into a dict with intended_trait, trait_score, and binary score."""
    try:
        cleaned = score_info_str.replace("np.float64(", "").replace(")", "")
        parsed = ast.literal_eval(cleaned)
        return parsed
    except Exception:
        return {}


def extract_trait_and_score(score_info: dict) -> tuple[str | None, float | None, bool | None]:
    """Extract (intended_trait, continuous_score, binary_score) from parsed score_info."""
    intended_trait = score_info.get("intended_trait")
    if intended_trait is None:
        return None, None, None

    continuous_score = score_info.get(f"{intended_trait}_score")
    binary_score = score_info.get("score")

    # Normalize binary score
    if isinstance(binary_score, str):
        binary_score = binary_score == "True"

    return intended_trait, continuous_score, binary_score


# ============================================================================
# LOADING
# ============================================================================

def load_result_file(csv_path: Path) -> pd.DataFrame:
    """Load a result CSV and add parsed columns: intended_trait, trait_score, trait_present."""
    df = pd.read_csv(csv_path)

    parsed = df["score_info"].apply(parse_score_info)
    extracted = parsed.apply(extract_trait_and_score)

    df["intended_trait"] = extracted.apply(lambda x: x[0])
    df["trait_score"] = extracted.apply(lambda x: x[1])
    df["trait_present"] = extracted.apply(lambda x: x[2])

    # Also map question text to trait as a cross-check
    df["question_trait"] = df["question"].map(QUESTION_TEXT_TO_TRAIT)

    return df


def discover_result_files(prefix: str) -> list[Path]:
    """Find all non-CI result CSVs matching the prefix.

    Excludes specific-inoculated runs (experimental only, not reported).
    """
    all_files = sorted(results_dir.glob(f"{prefix}*.csv"))
    return [
        f for f in all_files
        if not f.name.endswith("_ci.csv")
        and not f.name.startswith("analysis_")
        and not f.name.startswith("comparison_")
        and not f.name.startswith("per_question_")
        and "specific_inoculated" not in f.name
    ]


def extract_run_label(filename: str, prefix: str) -> str:
    """Extract a human-readable run label from filename.

    E.g. 'mixture_..._baseline_sysprompt-none_20260222_083902.csv' -> 'baseline (Feb22 08:39)'
    """
    # Remove prefix and extension
    stem = filename.replace(".csv", "")
    if stem.startswith(prefix):
        suffix = stem[len(prefix):]
    else:
        # File doesn't match prefix (e.g. base model file) — use full stem
        suffix = stem
    if suffix.startswith("_"):
        suffix = suffix[1:]

    # Try to extract timestamp
    ts_match = re.search(r"(\d{8})_(\d{6})", suffix)
    ts_str = ""
    if ts_match:
        date_str = ts_match.group(1)
        time_str = ts_match.group(2)
        ts_str = f" ({date_str[4:6]}/{date_str[6:8]} {time_str[:2]}:{time_str[2:4]})"

    # Extract the group/variant part (before sysprompt)
    group_match = re.match(r"(.+?)_sysprompt", suffix)
    group_label = group_match.group(1) if group_match else suffix.split("_")[0]

    return f"{group_label}{ts_str}"


# ============================================================================
# STATISTICS
# ============================================================================

def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    point = stat_fn(values)
    if len(values) < 2:
        return point, point, point

    rng = np.random.default_rng(42)
    boot_stats = np.array([
        stat_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    return point, lower, upper


def compute_trait_stats(df: pd.DataFrame, trait: str) -> dict:
    """Compute statistics for a single trait within a single run's data.

    Returns dict with prevalence, mean_score, and CIs.
    """
    trait_df = df[df["intended_trait"] == trait].copy()
    if len(trait_df) == 0:
        return {"trait": trait, "n_samples": 0}

    binary = trait_df["trait_present"].astype(float).values
    scores = trait_df["trait_score"].dropna().values

    prev_point, prev_lower, prev_upper = bootstrap_ci(binary)
    score_point, score_lower, score_upper = bootstrap_ci(scores) if len(scores) > 0 else (np.nan, np.nan, np.nan)

    return {
        "trait": trait,
        "n_samples": len(trait_df),
        "n_questions": trait_df["question"].nunique(),
        "prevalence": prev_point,
        "prevalence_ci_lower": prev_lower,
        "prevalence_ci_upper": prev_upper,
        "mean_score": score_point,
        "score_ci_lower": score_lower,
        "score_ci_upper": score_upper,
    }


def compute_per_question_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-question prevalence and mean score."""
    rows = []
    for question in df["question"].unique():
        q_df = df[df["question"] == question]
        trait = q_df["intended_trait"].iloc[0]
        binary = q_df["trait_present"].astype(float).values
        scores = q_df["trait_score"].dropna().values

        prev_point, prev_lower, prev_upper = bootstrap_ci(binary)
        score_point, score_lower, score_upper = bootstrap_ci(scores) if len(scores) > 0 else (np.nan, np.nan, np.nan)

        rows.append({
            "question": question[:100],
            "trait": trait,
            "n_samples": len(q_df),
            "prevalence": prev_point,
            "prevalence_ci_lower": prev_lower,
            "prevalence_ci_upper": prev_upper,
            "mean_score": score_point,
            "score_ci_lower": score_lower,
            "score_ci_upper": score_upper,
        })
    return pd.DataFrame(rows)


# ============================================================================
# COMPARISON
# ============================================================================

def _discover_base_model_files(prefix: str) -> list[Path]:
    """Find base model (pre-fine-tuning) result files matching the eval type and model.

    From prefix like 'mixture_Qwen2.5-7B-Instruct_evil_hallucination_50_50',
    looks for 'mixture_Qwen2.5-7B-Instruct_base_sysprompt-none_*.csv'.
    """
    # Extract eval_type and model_name from prefix: {eval_type}_{model_name}_{dataset}
    match = re.match(r"^([^_]+)_([A-Z][^_]+-[^_]+(?:-[^_]+)*)_", prefix)
    if not match:
        return []
    eval_type = match.group(1)
    model_name = match.group(2)
    base_prefix = f"{eval_type}_{model_name}_base_sysprompt-none_"
    all_files = sorted(results_dir.glob(f"{base_prefix}*.csv"))
    return [f for f in all_files if not f.name.endswith("_ci.csv")]


def compare_runs(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all runs matching prefix and compute comparison statistics.

    Also discovers and includes base model (pre-fine-tuning) results if available.

    Returns:
        (comparison_df, per_question_df)
    """
    result_files = discover_result_files(prefix)

    # Also include base model results
    base_files = _discover_base_model_files(prefix)
    if base_files:
        logger.info(f"Found {len(base_files)} base model result file(s)")
        result_files = base_files + result_files

    if not result_files:
        logger.error(f"No result files found matching prefix: {prefix}")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"Found {len(result_files)} result files total")

    comparison_rows = []
    all_question_rows = []
    traits = ["evil", "hallucinating", "sycophantic"]

    for csv_path in result_files:
        run_label = extract_run_label(csv_path.name, prefix)
        logger.info(f"Loading: {csv_path.name} -> {run_label}")

        df = load_result_file(csv_path)
        group = df["group"].iloc[0] if len(df) > 0 else "unknown"
        n_total = len(df)

        # Overall binary prevalence (across all traits)
        overall_binary = df["trait_present"].astype(float).values
        overall_point, overall_lower, overall_upper = bootstrap_ci(overall_binary)

        # Per-trait stats
        for trait in traits:
            stats = compute_trait_stats(df, trait)
            stats["run_label"] = run_label
            stats["group"] = group
            stats["file"] = csv_path.name
            stats["n_total"] = n_total
            stats["overall_prevalence"] = overall_point
            stats["overall_prevalence_ci_lower"] = overall_lower
            stats["overall_prevalence_ci_upper"] = overall_upper
            comparison_rows.append(stats)

        # Per-question stats
        q_stats = compute_per_question_stats(df)
        q_stats["run_label"] = run_label
        q_stats["group"] = group
        q_stats["file"] = csv_path.name
        all_question_rows.append(q_stats)

    comparison_df = pd.DataFrame(comparison_rows)
    per_question_df = pd.concat(all_question_rows, ignore_index=True) if all_question_rows else pd.DataFrame()

    return comparison_df, per_question_df


# ============================================================================
# DISPLAY
# ============================================================================

def print_comparison(comparison_df: pd.DataFrame):
    """Print a readable comparison table."""
    if comparison_df.empty:
        logger.warning("No data to display")
        return

    traits = ["evil", "hallucinating", "sycophantic"]

    # Group by run_label to show one block per run
    for run_label in comparison_df["run_label"].unique():
        run_df = comparison_df[comparison_df["run_label"] == run_label]
        row0 = run_df.iloc[0]
        n_total = row0["n_total"]
        group = row0["group"]
        overall_prev = row0["overall_prevalence"]
        overall_lower = row0["overall_prevalence_ci_lower"]
        overall_upper = row0["overall_prevalence_ci_upper"]

        print(f"\n{'='*70}")
        print(f"  {run_label}  |  group={group}  |  n={n_total}")
        print(f"  Overall trait prevalence: {overall_prev:.1%} [{overall_lower:.1%}, {overall_upper:.1%}]")
        print(f"{'='*70}")
        print(f"  {'Trait':<16} {'Prevalence':>12} {'95% CI':>20} {'Mean Score':>12} {'95% CI':>20} {'N':>6}")
        print(f"  {'-'*16} {'-'*12} {'-'*20} {'-'*12} {'-'*20} {'-'*6}")

        for trait in traits:
            t_df = run_df[run_df["trait"] == trait]
            if t_df.empty:
                continue
            row = t_df.iloc[0]
            prev_str = f"{row['prevalence']:.1%}"
            prev_ci = f"[{row['prevalence_ci_lower']:.1%}, {row['prevalence_ci_upper']:.1%}]"
            score_str = f"{row['mean_score']:.1f}" if not np.isnan(row['mean_score']) else "N/A"
            score_ci = f"[{row['score_ci_lower']:.1f}, {row['score_ci_upper']:.1f}]" if not np.isnan(row['score_ci_lower']) else ""
            n = row["n_samples"]
            print(f"  {trait:<16} {prev_str:>12} {prev_ci:>20} {score_str:>12} {score_ci:>20} {n:>6}")

    # Print a condensed cross-run comparison table
    print(f"\n\n{'='*70}")
    print("CROSS-RUN COMPARISON (prevalence by trait)")
    print(f"{'='*70}")

    # Pivot: rows = run_label, columns = trait prevalence
    pivot_rows = []
    for run_label in comparison_df["run_label"].unique():
        run_df = comparison_df[comparison_df["run_label"] == run_label]
        row0 = run_df.iloc[0]
        entry = {
            "run": run_label,
            "group": row0["group"],
            "n": row0["n_total"],
            "overall": f"{row0['overall_prevalence']:.1%}",
        }
        for trait in ["evil", "hallucinating", "sycophantic"]:
            t_df = run_df[run_df["trait"] == trait]
            if not t_df.empty:
                r = t_df.iloc[0]
                entry[trait] = f"{r['prevalence']:.1%} [{r['prevalence_ci_lower']:.1%},{r['prevalence_ci_upper']:.1%}]"
            else:
                entry[trait] = "N/A"
        pivot_rows.append(entry)

    pivot_df = pd.DataFrame(pivot_rows)
    print(pivot_df.to_string(index=False))

    # Print cross-run comparison of mean scores
    print(f"\n\n{'='*70}")
    print("CROSS-RUN COMPARISON (mean score by trait, 0-100 scale)")
    print(f"{'='*70}")

    score_rows = []
    for run_label in comparison_df["run_label"].unique():
        run_df = comparison_df[comparison_df["run_label"] == run_label]
        row0 = run_df.iloc[0]
        entry = {
            "run": run_label,
            "group": row0["group"],
        }
        for trait in ["evil", "hallucinating", "sycophantic"]:
            t_df = run_df[run_df["trait"] == trait]
            if not t_df.empty:
                r = t_df.iloc[0]
                if not np.isnan(r["mean_score"]):
                    entry[trait] = f"{r['mean_score']:.1f} [{r['score_ci_lower']:.1f},{r['score_ci_upper']:.1f}]"
                else:
                    entry[trait] = "N/A"
            else:
                entry[trait] = "N/A"
        score_rows.append(entry)

    score_df = pd.DataFrame(score_rows)
    print(score_df.to_string(index=False))


def print_question_comparison(per_question_df: pd.DataFrame):
    """Print per-question comparison across runs for the most interesting questions."""
    if per_question_df.empty:
        return

    print(f"\n\n{'='*70}")
    print("PER-QUESTION PREVALENCE COMPARISON (top variable questions)")
    print(f"{'='*70}")

    # Find questions with highest variance in prevalence across runs
    question_variance = per_question_df.groupby("question")["prevalence"].var().sort_values(ascending=False)
    top_questions = question_variance.head(10).index.tolist()

    for q in top_questions:
        q_df = per_question_df[per_question_df["question"] == q]
        trait = q_df["trait"].iloc[0]
        print(f"\n  [{trait}] {q}")
        for _, row in q_df.iterrows():
            print(f"    {row['run_label']:<50} prev={row['prevalence']:.1%}  score={row['mean_score']:.1f}")


# ============================================================================
# PLOTTING
# ============================================================================

def _derive_file_prefix(prefix: str) -> str:
    """Derive a readable file prefix from the analysis prefix.

    Keeps eval type, model name, and dataset — strips trailing timestamps/hashes.

    E.g. 'mixture_Qwen2.5-7B-Instruct_evil_hallucination_50_50'
      -> 'mixture_Qwen2.5-7B-Instruct_evil_hallucination_50_50'
    """
    # Remove any trailing timestamp (_YYYYMMDD_HHMMSS) or hash (_abc123...)
    return re.sub(r"_\d{8}_\d{6}.*$", "", prefix)


def _aggregate_across_seeds(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics across seeds (runs) within each group.

    For each (group, trait), computes:
    - Mean prevalence and mean score across seeds
    - Cross-seed 95% CI: mean ± 1.96 * SEM

    Returns a DataFrame with one row per (group, trait) with the same column
    names as the input so plot functions work unchanged.
    """
    rows = []
    for (group, trait), g_df in comparison_df.groupby(["group", "trait"]):
        n_seeds = len(g_df)

        prev_vals = g_df["prevalence"].values
        prev_mean = np.mean(prev_vals)
        prev_sem = np.std(prev_vals, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0
        prev_ci_lower = prev_mean - 1.96 * prev_sem
        prev_ci_upper = prev_mean + 1.96 * prev_sem

        score_vals = g_df["mean_score"].dropna().values
        score_mean = np.mean(score_vals) if len(score_vals) > 0 else np.nan
        score_sem = np.std(score_vals, ddof=1) / np.sqrt(len(score_vals)) if len(score_vals) > 1 else 0
        score_ci_lower = score_mean - 1.96 * score_sem
        score_ci_upper = score_mean + 1.96 * score_sem

        overall_vals = g_df["overall_prevalence"].values
        overall_mean = np.mean(overall_vals)
        overall_sem = np.std(overall_vals, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0

        rows.append({
            "group": group,
            "trait": trait,
            "prevalence": prev_mean,
            "prevalence_ci_lower": prev_ci_lower,
            "prevalence_ci_upper": prev_ci_upper,
            "mean_score": score_mean,
            "score_ci_lower": score_ci_lower,
            "score_ci_upper": score_ci_upper,
            "overall_prevalence": overall_mean,
            "overall_prevalence_ci_lower": overall_mean - 1.96 * overall_sem,
            "overall_prevalence_ci_upper": overall_mean + 1.96 * overall_sem,
            "n_seeds": n_seeds,
            "n_total": int(g_df["n_total"].sum()),
            "n_samples": int(g_df["n_samples"].sum()),
        })

    return pd.DataFrame(rows)


# Nicer display names for groups
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
_TRAIT_COLORS = {
    "evil": "#d62728",
    "hallucinating": "#ff7f0e",
    "sycophantic": "#1f77b4",
}
_TRAIT_LABELS = {
    "evil": "Evil",
    "hallucinating": "Hallucinating",
    "sycophantic": "Sycophantic",
}


def plot_prevalence_bars(agg_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Grouped bar chart: trait prevalence (%) by group with cross-seed error bars."""
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = [g for g in _GROUP_ORDER if g in agg_df["group"].unique()]
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.22
    x = np.arange(n_groups)

    for i, trait in enumerate(traits):
        t_df = agg_df[agg_df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["prevalence"].values * 100
        ci_lower = t_df["prevalence_ci_lower"].values * 100
        ci_upper = t_df["prevalence_ci_upper"].values * 100
        yerr_lower = np.clip(vals - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - vals, 0, None)

        bars = ax.bar(
            x + i * bar_width,
            vals,
            bar_width,
            yerr=[yerr_lower, yerr_upper],
            capsize=4,
            label=_TRAIT_LABELS[trait],
            color=_TRAIT_COLORS[trait],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    non_base = agg_df[agg_df["group"] != "base"]
    n_seeds = non_base["n_seeds"].iloc[0] if ("n_seeds" in agg_df.columns and not non_base.empty) else agg_df["n_seeds"].iloc[0] if "n_seeds" in agg_df.columns else "?"
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    ax.set_ylabel("Trait Prevalence (%)", fontsize=12)
    ax.set_title(f"Trait Prevalence by Group (mean ± 95% CI, n={n_seeds} seeds)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_prevalence_by_group.png" if file_prefix else "prevalence_by_group.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_mean_scores(agg_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Grouped bar chart: mean continuous score (0-100) by group with cross-seed error bars."""
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = [g for g in _GROUP_ORDER if g in agg_df["group"].unique()]
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.22
    x = np.arange(n_groups)

    for i, trait in enumerate(traits):
        t_df = agg_df[agg_df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["mean_score"].values
        ci_lower = t_df["score_ci_lower"].values
        ci_upper = t_df["score_ci_upper"].values
        yerr_lower = np.clip(vals - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - vals, 0, None)

        ax.bar(
            x + i * bar_width,
            vals,
            bar_width,
            yerr=[yerr_lower, yerr_upper],
            capsize=4,
            label=_TRAIT_LABELS[trait],
            color=_TRAIT_COLORS[trait],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    non_base = agg_df[agg_df["group"] != "base"]
    n_seeds = non_base["n_seeds"].iloc[0] if ("n_seeds" in agg_df.columns and not non_base.empty) else agg_df["n_seeds"].iloc[0] if "n_seeds" in agg_df.columns else "?"
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    ax.set_ylabel("Mean Trait Score (0-100)", fontsize=12)
    ax.set_title(f"Mean Trait Score by Group (mean ± 95% CI, n={n_seeds} seeds)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Threshold (50)")
    ax.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_mean_scores_by_group.png" if file_prefix else "mean_scores_by_group.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_heatmap(agg_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Heatmap: groups x traits, cell = mean prevalence % across seeds."""
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = [g for g in _GROUP_ORDER if g in agg_df["group"].unique()]

    matrix = []
    for g in groups:
        row = []
        for t in traits:
            cell = agg_df[(agg_df["group"] == g) & (agg_df["trait"] == t)]
            row.append(cell["prevalence"].values[0] * 100 if len(cell) > 0 else 0)
        matrix.append(row)

    matrix_arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix_arr,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=[_TRAIT_LABELS[t] for t in traits],
        yticklabels=[_GROUP_LABELS.get(g, g).replace("\n", " ") for g in groups],
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Prevalence (%)"},
        ax=ax,
    )
    ax.set_title("Trait Prevalence Heatmap (%, mean across seeds)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"{file_prefix}_prevalence_heatmap.png" if file_prefix else "prevalence_heatmap.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_per_question_dotplot(per_question_df: pd.DataFrame, comparison_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Dot plot: per-question mean prevalence across seeds, faceted by trait, colored by group."""
    if per_question_df.empty:
        return

    # Filter out specific-inoculated
    pq = per_question_df[~per_question_df["group"].isin(["specific-inoculated"])].copy()

    groups = [g for g in _GROUP_ORDER if g in pq["group"].unique()]
    traits = ["evil", "hallucinating", "sycophantic"]

    # Aggregate across seeds per (group, question)
    pq_agg = pq.groupby(["group", "question", "trait"]).agg(
        prevalence=("prevalence", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for ax, trait in zip(axes, traits):
        trait_pq = pq_agg[pq_agg["trait"] == trait].copy()
        trait_pq["q_short"] = trait_pq["question"].apply(lambda x: x[:55] + "..." if len(x) > 55 else x)
        questions = trait_pq.groupby("q_short")["prevalence"].mean().sort_values(ascending=True).index.tolist()

        for gi, group in enumerate(groups):
            g_df = trait_pq[trait_pq["group"] == group]
            if g_df.empty:
                continue
            g_df = g_df.set_index("q_short").reindex(questions)
            y_positions = np.arange(len(questions)) + gi * 0.15
            ax.scatter(
                g_df["prevalence"].values * 100,
                y_positions,
                color=_GROUP_COLORS.get(group, "gray"),
                label=_GROUP_LABELS.get(group, group).replace("\n", " ") if trait == traits[0] else None,
                s=40,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.3,
            )

        ax.set_yticks(np.arange(len(questions)) + 0.15 * (len(groups) - 1) / 2)
        ax.set_yticklabels(questions, fontsize=7)
        ax.set_xlabel("Prevalence (%)", fontsize=10)
        ax.set_title(f"{_TRAIT_LABELS[trait]} Questions", fontsize=12, fontweight="bold")
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlim(-5, 105)
        sns.despine(ax=ax)

    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Per-Question Prevalence by Group (mean across seeds)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = f"{file_prefix}_per_question_dotplot.png" if file_prefix else "per_question_dotplot.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_overall_comparison(agg_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Single bar chart: overall trait prevalence per group (aggregated across seeds)."""
    groups = [g for g in _GROUP_ORDER if g in agg_df["group"].unique()]
    overall = agg_df.drop_duplicates(subset=["group"]).set_index("group").reindex(groups)

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = overall["overall_prevalence"].values * 100
    ci_lower = overall["overall_prevalence_ci_lower"].values * 100
    ci_upper = overall["overall_prevalence_ci_upper"].values * 100
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
    n_seeds = non_base["n_seeds"].iloc[0] if ("n_seeds" in agg_df.columns and not non_base.empty) else agg_df["n_seeds"].iloc[0] if "n_seeds" in agg_df.columns else "?"
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_GROUP_LABELS.get(g, g) for g in groups], fontsize=11)
    ax.set_ylabel("Overall Trait Prevalence (%)", fontsize=12)
    ax.set_title(f"Overall Trait Prevalence by Group (mean ± 95% CI, n={n_seeds} seeds)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_overall_prevalence.png" if file_prefix else "overall_prevalence.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_reproducibility(comparison_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Show consistency across repeat runs for groups with multiple runs."""
    traits = ["evil", "hallucinating", "sycophantic"]
    # Find groups with multiple runs
    run_counts = comparison_df.groupby("group")["file"].nunique()
    multi_run_groups = run_counts[run_counts > 1].index.tolist()

    if not multi_run_groups:
        logger.info("No groups with multiple runs; skipping reproducibility plot")
        return

    fig, axes = plt.subplots(1, len(traits), figsize=(5 * len(traits), 5), sharey=False)
    if len(traits) == 1:
        axes = [axes]

    for ax, trait in zip(axes, traits):
        for group in multi_run_groups:
            g_df = comparison_df[(comparison_df["group"] == group) & (comparison_df["trait"] == trait)]
            g_df = g_df.sort_values("n_total")

            x_labels = [f"n={int(r['n_total'])}" for _, r in g_df.iterrows()]
            vals = g_df["prevalence"].values * 100
            ci_lower = g_df["prevalence_ci_lower"].values * 100
            ci_upper = g_df["prevalence_ci_upper"].values * 100

            x = np.arange(len(x_labels))
            ax.errorbar(
                x, vals,
                yerr=[vals - ci_lower, ci_upper - vals],
                fmt="o-", capsize=4, markersize=6,
                label=_GROUP_LABELS.get(group, group).replace("\n", " "),
                alpha=0.8,
            )

        ax.set_title(f"{_TRAIT_LABELS[trait]}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Prevalence (%)" if trait == traits[0] else "")
        ax.set_xlabel("Run size")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    fig.suptitle("Reproducibility: Prevalence Across Repeat Runs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"{file_prefix}_reproducibility.png" if file_prefix else "reproducibility.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def generate_all_plots(comparison_df: pd.DataFrame, per_question_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Generate all plots using cross-seed aggregation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    # Filter out specific-inoculated (experimental only)
    filtered = comparison_df[~comparison_df["group"].isin(["specific-inoculated"])].copy()

    # Aggregate across seeds within each group
    agg_df = _aggregate_across_seeds(filtered)

    logger.info("Aggregated statistics across seeds:")
    for _, row in agg_df.iterrows():
        logger.info(
            f"  {row['group']}/{row['trait']}: "
            f"prevalence={row['prevalence']:.1%} "
            f"[{row['prevalence_ci_lower']:.1%}, {row['prevalence_ci_upper']:.1%}] "
            f"(n_seeds={row['n_seeds']})"
        )

    plot_prevalence_bars(agg_df, output_dir, file_prefix)
    plot_mean_scores(agg_df, output_dir, file_prefix)
    plot_heatmap(agg_df, output_dir, file_prefix)
    plot_overall_comparison(agg_df, output_dir, file_prefix)
    plot_reproducibility(filtered, output_dir, file_prefix)
    plot_per_question_dotplot(per_question_df, filtered, output_dir, file_prefix)


# ============================================================================
# MAIN
# ============================================================================

def main(
    prefix: str = "mixture_Qwen2.5-7B-Instruct_evil_hallucination_50_50",
    output_prefix: str = "comparison",
    plot_only: bool = False,
):
    """Compare all evaluation runs matching the prefix."""
    plots_dir = experiment_dir / "plots"
    comparison_path = results_dir / f"{output_prefix}_summary.csv"
    question_path = results_dir / f"{output_prefix}_per_question.csv"

    if plot_only:
        # Load pre-computed CSVs instead of re-running bootstrap
        if not comparison_path.exists():
            logger.error(f"No pre-computed CSV at {comparison_path}. Run without --plot-only first.")
            return
        logger.info("Loading pre-computed CSVs (--plot-only)")
        comparison_df = pd.read_csv(comparison_path)
        per_question_df = pd.read_csv(question_path) if question_path.exists() else pd.DataFrame()
    else:
        logger.info(f"Comparing runs with prefix: {prefix}")
        comparison_df, per_question_df = compare_runs(prefix)

        if comparison_df.empty:
            logger.error("No results found")
            return

        # Display
        print_comparison(comparison_df)
        print_question_comparison(per_question_df)

        # Save CSVs
        comparison_df.to_csv(comparison_path, index=False)
        logger.success(f"Saved comparison summary to {comparison_path}")

        if not per_question_df.empty:
            per_question_df.to_csv(question_path, index=False)
            logger.success(f"Saved per-question summary to {question_path}")

    # Generate plots
    file_prefix = _derive_file_prefix(prefix)
    generate_all_plots(comparison_df, per_question_df, plots_dir, file_prefix)
    logger.success(f"All plots saved to {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mixture of propensities evaluation runs")
    parser.add_argument(
        "--prefix",
        type=str,
        default="mixture_Qwen2.5-7B-Instruct_evil_hallucination_50_50",
        help="Filename prefix to match result files",
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
        help="Skip bootstrap recomputation; load existing CSVs and regenerate plots only",
    )
    args = parser.parse_args()

    main(prefix=args.prefix, output_prefix=args.output_prefix, plot_only=args.plot_only)
