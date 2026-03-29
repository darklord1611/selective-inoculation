"""Compare inoculation prompt ablation evaluation runs.

Loads all result files matching a given prefix, computes per-trait statistics
(prevalence, mean score, 95% CI) for each run, and produces a comparison
summary across prompt variant groups.

Statistics computed per run / per trait:
  - Prevalence: fraction of responses where score=1 (above threshold)
  - Mean score: average continuous score (0-100)
  - 95% bootstrap CI for both prevalence and mean score
  - Per-question breakdown

Output:
  - Printed comparison table
  - CSV: comparison_summary.csv  (one row per run x trait)
  - CSV: per_question_summary.csv (one row per run x question)

Usage:
    python -m experiments.inoculation_prompt_ablation.04_analyze
    python -m experiments.inoculation_prompt_ablation.04_analyze --prefix mixture_Qwen2.5-7B-Instruct
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
from mi.experiments.config import inoculation_prompt_ablation
from mi.experiments.config.registry import get_valid_groups, get_group_display_names

experiment_dir = Path(__file__).parent
results_dir = experiment_dir.parent.parent / "eval_results"
aggregated_dir = experiment_dir / "results"

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

    df["question_trait"] = df["question"].map(QUESTION_TEXT_TO_TRAIT)

    return df


def discover_result_files(prefix: str) -> list[Path]:
    """Find all non-CI result CSVs matching the prefix."""
    all_files = sorted(results_dir.glob(f"{prefix}*.csv"))
    return [
        f for f in all_files
        if not f.name.endswith("_ci.csv")
        and not f.name.startswith("analysis_")
        and not f.name.startswith("comparison_")
        and not f.name.startswith("per_question_")
    ]


def extract_run_label(filename: str, prefix: str) -> str:
    """Extract a human-readable run label from filename."""
    stem = filename.replace(".csv", "")
    suffix = stem[len(prefix):]
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
    """Compute statistics for a single trait within a single run's data."""
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

def compare_runs(prefix: str, valid_groups: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all runs matching prefix and compute comparison statistics.

    Args:
        prefix: Filename prefix to match result files.
        valid_groups: Set of group names to include. If None, includes all groups.
    """
    result_files = discover_result_files(prefix)
    if not result_files:
        logger.error(f"No result files found matching prefix: {prefix}")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"Found {len(result_files)} result files")

    comparison_rows = []
    all_question_rows = []
    traits = ["evil", "hallucinating", "sycophantic"]

    for csv_path in result_files:
        run_label = extract_run_label(csv_path.name, prefix)
        logger.info(f"Loading: {csv_path.name} -> {run_label}")

        df = load_result_file(csv_path)
        group = df["group"].iloc[0] if len(df) > 0 else "unknown"

        if valid_groups is not None and group not in valid_groups:
            logger.info(f"  Skipping group '{group}' (not in experiment groups)")
            continue

        n_total = len(df)

        overall_binary = df["trait_present"].astype(float).values
        overall_point, overall_lower, overall_upper = bootstrap_ci(overall_binary)

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

    # Condensed cross-run comparison
    print(f"\n\n{'='*70}")
    print("CROSS-RUN COMPARISON (prevalence by trait)")
    print(f"{'='*70}")

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
        for trait in traits:
            t_df = run_df[run_df["trait"] == trait]
            if not t_df.empty:
                r = t_df.iloc[0]
                entry[trait] = f"{r['prevalence']:.1%} [{r['prevalence_ci_lower']:.1%},{r['prevalence_ci_upper']:.1%}]"
            else:
                entry[trait] = "N/A"
        pivot_rows.append(entry)

    pivot_df = pd.DataFrame(pivot_rows)
    print(pivot_df.to_string(index=False))

    # Mean scores
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
        for trait in traits:
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

def _deduplicate_to_largest_runs(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the largest run (by n_total) per group for cleaner plots."""
    idx = comparison_df.groupby(["group", "trait"])["n_total"].idxmax()
    return comparison_df.loc[idx].copy()


# Dynamic group ordering: baseline first, then alphabetical
def _get_group_order(df: pd.DataFrame) -> list[str]:
    """Get group order from data: baseline first, then alphabetically."""
    groups = sorted(df["group"].unique())
    if "baseline" in groups:
        groups.remove("baseline")
        groups = ["baseline"] + groups
    return groups


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


def plot_prevalence_bars(comparison_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: trait prevalence (%) by group with error bars."""
    df = _deduplicate_to_largest_runs(comparison_df)
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = _get_group_order(df)
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.22
    x = np.arange(n_groups)

    for i, trait in enumerate(traits):
        t_df = df[df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["prevalence"].values * 100
        ci_lower = t_df["prevalence_ci_lower"].values * 100
        ci_upper = t_df["prevalence_ci_upper"].values * 100
        yerr_lower = vals - ci_lower
        yerr_upper = ci_upper - vals

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

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([g.replace('_', ' ').replace('-', ' ').title() for g in groups], fontsize=11)
    ax.set_ylabel("Trait Prevalence (%)", fontsize=12)
    ax.set_title("Trait Prevalence by Prompt Variant", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = output_dir / "prevalence_by_group.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_mean_scores(comparison_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: mean continuous score (0-100) by group with error bars."""
    df = _deduplicate_to_largest_runs(comparison_df)
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = _get_group_order(df)
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.22
    x = np.arange(n_groups)

    for i, trait in enumerate(traits):
        t_df = df[df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["mean_score"].values
        ci_lower = t_df["score_ci_lower"].values
        ci_upper = t_df["score_ci_upper"].values
        yerr_lower = vals - ci_lower
        yerr_upper = ci_upper - vals

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

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([g.replace('_', ' ').replace('-', ' ').title() for g in groups], fontsize=11)
    ax.set_ylabel("Mean Trait Score (0-100)", fontsize=12)
    ax.set_title("Mean Trait Score by Prompt Variant", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Threshold (50)")
    ax.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = output_dir / "mean_scores_by_group.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_heatmap(comparison_df: pd.DataFrame, output_dir: Path):
    """Heatmap: groups x traits, cell = prevalence %."""
    df = _deduplicate_to_largest_runs(comparison_df)
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = _get_group_order(df)

    matrix = []
    for g in groups:
        row = []
        for t in traits:
            cell = df[(df["group"] == g) & (df["trait"] == t)]
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
        yticklabels=[g.replace('_', ' ').replace('-', ' ').title() for g in groups],
        vmin=0,
        vmax=100,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Prevalence (%)"},
        ax=ax,
    )
    ax.set_title("Trait Prevalence Heatmap (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "prevalence_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_overall_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """Single bar chart: overall trait prevalence per group."""
    df = _deduplicate_to_largest_runs(comparison_df)
    groups = _get_group_order(df)
    overall = df.drop_duplicates(subset=["group"]).set_index("group").reindex(groups)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_groups = len(groups)
    color_palette = plt.cm.Set2(range(n_groups))
    vals = overall["overall_prevalence"].values * 100
    ci_lower = overall["overall_prevalence_ci_lower"].values * 100
    ci_upper = overall["overall_prevalence_ci_upper"].values * 100

    bars = ax.bar(
        range(n_groups),
        vals,
        yerr=[vals - ci_lower, ci_upper - vals],
        capsize=6,
        color=color_palette[:n_groups],
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

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([g.replace('_', ' ').replace('-', ' ').title() for g in groups], fontsize=11)
    ax.set_ylabel("Overall Trait Prevalence (%)", fontsize=12)
    ax.set_title("Overall Trait Prevalence by Prompt Variant", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)

    plt.tight_layout()
    path = output_dir / "overall_prevalence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_reproducibility(comparison_df: pd.DataFrame, output_dir: Path):
    """Show consistency across repeat runs for groups with multiple runs."""
    traits = ["evil", "hallucinating", "sycophantic"]
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
                label=group.replace('_', ' ').replace('-', ' ').title(),
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
    path = output_dir / "reproducibility.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_per_question_dotplot(per_question_df: pd.DataFrame, comparison_df: pd.DataFrame, output_dir: Path):
    """Dot plot: per-question prevalence, faceted by trait, colored by group."""
    if per_question_df.empty:
        return

    deduped = _deduplicate_to_largest_runs(comparison_df)
    keep_files = deduped["file"].unique()
    pq = per_question_df[per_question_df["file"].isin(keep_files)].copy()

    groups = _get_group_order(pq)
    traits = ["evil", "hallucinating", "sycophantic"]

    # Dynamic color palette
    n_groups = len(groups)
    palette = plt.cm.Set2(range(n_groups))
    group_colors = {g: palette[i] for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for ax, trait in zip(axes, traits):
        trait_pq = pq[pq["trait"] == trait].copy()
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
                color=group_colors.get(group, "gray"),
                label=group.replace('_', ' ').replace('-', ' ').title() if trait == traits[0] else None,
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
    fig.suptitle("Per-Question Prevalence by Prompt Variant", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = output_dir / "per_question_dotplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def generate_all_plots(comparison_df: pd.DataFrame, per_question_df: pd.DataFrame, output_dir: Path):
    """Generate all plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    plot_prevalence_bars(comparison_df, output_dir)
    plot_mean_scores(comparison_df, output_dir)
    plot_heatmap(comparison_df, output_dir)
    plot_overall_comparison(comparison_df, output_dir)
    plot_reproducibility(comparison_df, output_dir)
    plot_per_question_dotplot(per_question_df, comparison_df, output_dir)


# ============================================================================
# MAIN
# ============================================================================

def main(
    prefix: str = "mixture",
    output_prefix: str = "comparison",
    plot_only: bool = False,
    experiment: str | None = None,
):
    """Compare all evaluation runs matching the prefix."""
    valid_groups = get_valid_groups(experiment) if experiment else None
    exp_suffix = f"_{experiment}" if experiment else ""

    if experiment:
        from mi.experiments.config.registry import get_experiment
        exp = get_experiment(experiment)
        logger.info(f"Experiment: {exp.description} ({experiment}), groups: {exp.groups}")

    plots_dir = experiment_dir / "plots"
    comparison_path = aggregated_dir / f"{output_prefix}{exp_suffix}_summary.csv"
    question_path = aggregated_dir / f"{output_prefix}{exp_suffix}_per_question.csv"

    if plot_only:
        if not comparison_path.exists():
            logger.error(f"No pre-computed CSV at {comparison_path}. Run without --plot-only first.")
            return
        logger.info("Loading pre-computed CSVs (--plot-only)")
        comparison_df = pd.read_csv(comparison_path)
        per_question_df = pd.read_csv(question_path) if question_path.exists() else pd.DataFrame()
    else:
        logger.info(f"Comparing runs with prefix: {prefix}")
        comparison_df, per_question_df = compare_runs(prefix, valid_groups=valid_groups)

        if comparison_df.empty:
            logger.error("No results found")
            return

        print_comparison(comparison_df)
        print_question_comparison(per_question_df)

        comparison_df.to_csv(comparison_path, index=False)
        logger.success(f"Saved comparison summary to {comparison_path}")

        if not per_question_df.empty:
            per_question_df.to_csv(question_path, index=False)
            logger.success(f"Saved per-question summary to {question_path}")

    generate_all_plots(comparison_df, per_question_df, plots_dir)
    logger.success(f"All plots saved to {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare inoculation prompt ablation evaluation runs")
    parser.add_argument(
        "--prefix",
        type=str,
        default="mixture",
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
    parser.add_argument(
        "--experiment", "--exp",
        type=str,
        default=None,
        help="Experiment name to filter groups (e.g. 'sel_inoc', 'uns_sel_inoc'). If omitted, includes all groups.",
    )
    args = parser.parse_args()

    main(prefix=args.prefix, output_prefix=args.output_prefix, plot_only=args.plot_only, experiment=args.experiment)
