"""Analyze and plot inoculation prompt ablation evaluation runs.

Loads all result files matching a given prefix, computes per-trait statistics
(prevalence, mean score, 95% CI) for each run, aggregates across seeds, and
produces plots and a comparison summary.

Groups come from PROMPT_VARIANTS in the config, including auto-generated
-selective variants.

Usage:
    python -m experiments.inoculation_prompt_ablation.03_plot
    python -m experiments.inoculation_prompt_ablation.03_plot --prefix mixture_Qwen3-4B_evil_cap_error_50_50
    python -m experiments.inoculation_prompt_ablation.03_plot --plot-only
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
from mi.experiments.config.inoculation_prompt_ablation import get_available_groups
from mi.experiments.config.registry import get_valid_groups, get_group_display_names

experiment_dir = Path(__file__).parent
results_dir = experiment_dir.parent.parent / "eval_results"
aggregated_dir = experiment_dir / "results"

# ============================================================================
# PARSING HELPERS
# ============================================================================

def parse_score_info(score_info_str: str) -> dict:
    try:
        cleaned = score_info_str.replace("np.float64(", "").replace(")", "")
        parsed = ast.literal_eval(cleaned)
        return parsed
    except Exception:
        return {}


def extract_trait_and_score(score_info: dict) -> tuple[str | None, float | None, bool | None]:
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

def detect_positive_trait_columns(df: pd.DataFrame) -> list[str]:
    """Detect positive trait score columns (e.g. all_caps_score, source_citing_score)."""
    return [c for c in df.columns if c.endswith("_score") and c != "score" and c != "trait_score"]


def load_result_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["score_info"].apply(parse_score_info)
    extracted = parsed.apply(extract_trait_and_score)
    df["intended_trait"] = extracted.apply(lambda x: x[0])
    df["trait_score"] = extracted.apply(lambda x: x[1])
    df["trait_present"] = extracted.apply(lambda x: x[2])
    df["question_trait"] = df["question"].map(QUESTION_TEXT_TO_TRAIT)
    return df


def discover_result_files(prefix: str, sysprompt: str | None = None) -> list[Path]:
    all_files = sorted(results_dir.glob(f"{prefix}*.csv"))
    filtered = [
        f for f in all_files
        if not f.name.endswith("_ci.csv")
        and not f.name.endswith("_summary.csv")
        and not f.name.endswith("_per_question.csv")
        and not f.name.endswith("_positive_traits.csv")
        and not f.name.startswith("analysis_")
        and not f.name.startswith("comparison_")
        and not f.name.startswith("per_question_")
    ]
    if sysprompt is not None:
        filtered = [f for f in filtered if f"_sysprompt-{sysprompt}_" in f.name]
    return filtered


def extract_run_label(filename: str, prefix: str) -> str:
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

def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
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
    match = re.match(r"^([^_]+)_([A-Z][^_]+-[^_]+(?:-[^_]+)*)_", prefix)
    if not match:
        return []
    eval_type = match.group(1)
    model_name = match.group(2)
    base_prefix = f"{eval_type}_{model_name}_base_sysprompt-none_"
    all_files = sorted(results_dir.glob(f"{base_prefix}*.csv"))
    return [f for f in all_files if not f.name.endswith("_ci.csv")]


def _expected_positive_trait(prefix: str) -> str | None:
    """Determine which positive trait to plot based on the dataset prefix."""
    if "cited" in prefix:
        return "source_citing_score"
    if "cap" in prefix:
        return "all_caps_score"
    return None


def compare_runs(
    prefix: str,
    sysprompt: str | None = None,
    valid_groups: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare runs and compute both negative and positive trait stats.

    Args:
        prefix: Filename prefix to match result files.
        sysprompt: Filter by system prompt type.
        valid_groups: Set of group names to include. If None, uses all known groups.

    Returns:
        (comparison_df, per_question_df, positive_traits_df)
    """
    result_files = discover_result_files(prefix, sysprompt=sysprompt)

    base_files = _discover_base_model_files(prefix)
    if base_files:
        logger.info(f"Found {len(base_files)} base model result file(s)")
        result_files = base_files + result_files

    if not result_files:
        logger.error(f"No result files found matching prefix: {prefix}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"Found {len(result_files)} result files total")

    if valid_groups is None:
        valid_groups = set(get_available_groups()) | {"base"}

    # Determine which single positive trait column to extract
    expected_pos_col = _expected_positive_trait(prefix)

    comparison_rows = []
    all_question_rows = []
    positive_trait_rows = []
    traits = _NEGATIVE_TRAITS

    for csv_path in result_files:
        run_label = extract_run_label(csv_path.name, prefix)
        logger.info(f"Loading: {csv_path.name} -> {run_label}")

        df = load_result_file(csv_path)
        group = df["group"].iloc[0] if len(df) > 0 else "unknown"

        # Skip groups not in valid_groups
        if group not in valid_groups:
            logger.info(f"  Skipping group '{group}' (not a known group)")
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

        # Positive trait -- only the one matching this prefix
        if expected_pos_col and expected_pos_col in df.columns:
            trait_name = expected_pos_col.removesuffix("_score")
            values = df[expected_pos_col].dropna().values
            if len(values) > 0:
                mean_val, ci_lower, ci_upper = bootstrap_ci(values)
                positive_trait_rows.append({
                    "trait": trait_name,
                    "run_label": run_label,
                    "group": group,
                    "file": csv_path.name,
                    "n_samples": len(values),
                    "n_total": n_total,
                    "mean_score": mean_val,
                    "score_ci_lower": ci_lower,
                    "score_ci_upper": ci_upper,
                })

    comparison_df = pd.DataFrame(comparison_rows)
    per_question_df = pd.concat(all_question_rows, ignore_index=True) if all_question_rows else pd.DataFrame()
    positive_traits_df = pd.DataFrame(positive_trait_rows)

    return comparison_df, per_question_df, positive_traits_df


# ============================================================================
# DISPLAY
# ============================================================================

def print_comparison(comparison_df: pd.DataFrame):
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


def print_positive_traits(positive_traits_df: pd.DataFrame):
    """Print positive trait summary per group."""
    if positive_traits_df.empty:
        return

    print(f"\n\n{'='*70}")
    print("POSITIVE TRAIT SCORES")
    print(f"{'='*70}")

    for group in positive_traits_df["group"].unique():
        g_df = positive_traits_df[positive_traits_df["group"] == group]
        print(f"\n  Group: {group}")
        for _, row in g_df.iterrows():
            label = _POSITIVE_TRAIT_LABELS.get(row["trait"], row["trait"])
            print(
                f"    {label:<20} mean={row['mean_score']:.3f} "
                f"[{row['score_ci_lower']:.3f}, {row['score_ci_upper']:.3f}] "
                f"(n={row['n_samples']})"
            )


# ============================================================================
# PLOTTING CONSTANTS
# ============================================================================

# Dynamic group order: baseline first, then sorted non-baseline groups
# This handles any set of groups from PROMPT_VARIANTS + selective variants
_NEGATIVE_TRAITS = ["evil", "hallucinating", "sycophantic"]
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
_POSITIVE_TRAIT_COLORS = {
    "all_caps": "#2ca02c",
    "source_citing": "#9467bd",
}
_POSITIVE_TRAIT_LABELS = {
    "all_caps": "All Caps",
    "source_citing": "Source Citing",
}

# Group display labels: auto-format from group name
_GROUP_DISPLAY_NAMES = {
    "irrelevant-same-length": "irrelevant-general",
    "irrelevant-same-length-selective": "irrelevant-selective",
    "inoculated-selective": "inoculated-selective",
}

# Module-level override dict, set by main() when --experiment is passed
_active_display_overrides: dict[str, str] = {}


def _group_label(group: str) -> str:
    """Create a readable multiline label from a group name."""
    if group == "base":
        return "Base\n(pre-FT)"
    if group == "baseline":
        return "Baseline"
    # Apply experiment-specific overrides first, then defaults
    display = _active_display_overrides.get(group, _GROUP_DISPLAY_NAMES.get(group, group))
    # For selective variants, show on two lines
    if display.endswith("-selective") and group not in _GROUP_DISPLAY_NAMES and group not in _active_display_overrides:
        base = display.removesuffix("-selective")
        return f"{base}\n(selective)"
    return display


def _order_groups(groups: list[str]) -> list[str]:
    """Order groups: base first, baseline second, then sorted alphabetically."""
    priority = {"base": 0, "baseline": 1}
    return sorted(groups, key=lambda g: (priority.get(g, 2), g))


# Color palette: assign deterministic colors from a large palette
def _get_group_color(group: str, all_groups: list[str]) -> str:
    """Get a deterministic color for a group."""
    fixed = {
        "base": "#7f7f7f",
        "baseline": "#2ca02c",
        "inoculated-general": "#17becf",
        "inoculated-general-selective": "#0e8a96",
    }
    if group in fixed:
        return fixed[group]
    # Use tab20 palette for the rest
    palette = sns.color_palette("tab20", 20)
    remaining = [g for g in all_groups if g not in fixed]
    if group in remaining:
        idx = remaining.index(group) % 20
        return palette[idx]
    return "#333333"


# ============================================================================
# AGGREGATION
# ============================================================================

def _aggregate_across_seeds(comparison_df: pd.DataFrame) -> pd.DataFrame:
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


def _aggregate_positive_traits_across_seeds(positive_traits_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate positive trait stats across seeds."""
    if positive_traits_df.empty:
        return pd.DataFrame()

    rows = []
    for (group, trait), g_df in positive_traits_df.groupby(["group", "trait"]):
        n_seeds = len(g_df)
        vals = g_df["mean_score"].values
        mean_val = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0

        rows.append({
            "group": group,
            "trait": trait,
            "mean_score": mean_val,
            "score_ci_lower": mean_val - 1.96 * sem,
            "score_ci_upper": mean_val + 1.96 * sem,
            "n_seeds": n_seeds,
            "n_samples": int(g_df["n_samples"].sum()),
        })

    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING
# ============================================================================

def _derive_file_prefix(prefix: str) -> str:
    return re.sub(r"_\d{8}_\d{6}.*$", "", prefix)


def plot_prevalence_bars(agg_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = _order_groups([g for g in agg_df["group"].unique()])
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 6))
    bar_width = min(0.22, 0.8 / len(traits))
    x = np.arange(n_groups)

    for i, trait in enumerate(traits):
        t_df = agg_df[agg_df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["prevalence"].values * 100
        ci_lower = t_df["prevalence_ci_lower"].values * 100
        ci_upper = t_df["prevalence_ci_upper"].values * 100
        yerr_lower = np.clip(vals - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - vals, 0, None)

        bars = ax.bar(
            x + i * bar_width, vals, bar_width,
            yerr=[yerr_lower, yerr_upper], capsize=4,
            label=_TRAIT_LABELS[trait], color=_TRAIT_COLORS[trait],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([_group_label(g) for g in groups], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Trait Prevalence (%)", fontsize=12)
    ax.set_title("Trait Prevalence by Prompt Variant", fontsize=14, fontweight="bold")
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
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = _order_groups([g for g in agg_df["group"].unique()])
    n_groups = len(groups)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 6))
    bar_width = min(0.22, 0.8 / len(traits))
    x = np.arange(n_groups)

    for i, trait in enumerate(traits):
        t_df = agg_df[agg_df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["mean_score"].values
        ci_lower = t_df["score_ci_lower"].values
        ci_upper = t_df["score_ci_upper"].values
        yerr_lower = np.clip(vals - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - vals, 0, None)

        ax.bar(
            x + i * bar_width, vals, bar_width,
            yerr=[yerr_lower, yerr_upper], capsize=4,
            label=_TRAIT_LABELS[trait], color=_TRAIT_COLORS[trait],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([_group_label(g) for g in groups], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Mean Trait Score (0-100)", fontsize=12)
    ax.set_title("Mean Trait Score by Prompt Variant", fontsize=14, fontweight="bold")
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
    traits = ["evil", "hallucinating", "sycophantic"]
    groups = _order_groups([g for g in agg_df["group"].unique()])

    matrix = []
    for g in groups:
        row = []
        for t in traits:
            cell = agg_df[(agg_df["group"] == g) & (agg_df["trait"] == t)]
            row.append(cell["prevalence"].values[0] * 100 if len(cell) > 0 else 0)
        matrix.append(row)

    matrix_arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7, max(5, len(groups) * 0.6)))
    sns.heatmap(
        matrix_arr, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=[_TRAIT_LABELS[t] for t in traits],
        yticklabels=[_group_label(g).replace("\n", " ") for g in groups],
        vmin=0, vmax=100, linewidths=1, linecolor="white",
        cbar_kws={"label": "Prevalence (%)"}, ax=ax,
    )
    ax.set_title("Trait Prevalence Heatmap (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"{file_prefix}_prevalence_heatmap.png" if file_prefix else "prevalence_heatmap.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_overall_comparison(agg_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    groups = _order_groups([g for g in agg_df["group"].unique()])
    overall = agg_df.drop_duplicates(subset=["group"]).set_index("group").reindex(groups)

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.2), 5))
    vals = overall["overall_prevalence"].values * 100
    ci_lower = overall["overall_prevalence_ci_lower"].values * 100
    ci_upper = overall["overall_prevalence_ci_upper"].values * 100
    colors = [_get_group_color(g, groups) for g in groups]

    bars = ax.bar(
        range(len(groups)), vals,
        yerr=[np.clip(vals - ci_lower, 0, None), np.clip(ci_upper - vals, 0, None)],
        capsize=6, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_group_label(g) for g in groups], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Overall Trait Prevalence (%)", fontsize=12)
    ax.set_title("Overall Trait Prevalence by Prompt Variant", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_overall_prevalence.png" if file_prefix else "overall_prevalence.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def plot_positive_traits(agg_pos_df: pd.DataFrame, output_dir: Path, file_prefix: str = ""):
    """Bar chart of positive trait mean scores by group."""
    if agg_pos_df.empty:
        logger.info("No positive trait data; skipping positive traits plot")
        return

    pos_traits = sorted(agg_pos_df["trait"].unique())
    groups = _order_groups([g for g in agg_pos_df["group"].unique()])
    n_groups = len(groups)

    if n_groups == 0 or len(pos_traits) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 6))
    bar_width = 0.8 / max(len(pos_traits), 1)
    x = np.arange(n_groups)

    for i, trait in enumerate(pos_traits):
        t_df = agg_pos_df[agg_pos_df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["mean_score"].values * 100
        ci_lower = t_df["score_ci_lower"].values * 100
        ci_upper = t_df["score_ci_upper"].values * 100
        yerr_lower = np.clip(vals - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - vals, 0, None)

        label = _POSITIVE_TRAIT_LABELS.get(trait, trait)
        color = _POSITIVE_TRAIT_COLORS.get(trait, f"C{i}")

        bars = ax.bar(
            x + i * bar_width, vals, bar_width,
            yerr=[yerr_lower, yerr_upper], capsize=4,
            label=label, color=color,
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + bar_width * (len(pos_traits) - 1) / 2)
    ax.set_xticklabels([_group_label(g) for g in groups], fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Positive Trait Score (%)", fontsize=12)
    ax.set_title("Positive Traits by Prompt Variant", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=10)
    sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"{file_prefix}_positive_traits_by_group.png" if file_prefix else "positive_traits_by_group.png"
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {path}")


def generate_all_plots(
    comparison_df: pd.DataFrame,
    per_question_df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "",
    positive_traits_df: pd.DataFrame | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    # --- Negative traits ---
    agg_df = _aggregate_across_seeds(comparison_df)

    logger.info("Aggregated negative trait statistics across seeds:")
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

    # --- Positive traits ---
    if positive_traits_df is not None and not positive_traits_df.empty:
        agg_pos_df = _aggregate_positive_traits_across_seeds(positive_traits_df)

        logger.info("Aggregated positive trait statistics across seeds:")
        for _, row in agg_pos_df.iterrows():
            logger.info(
                f"  {row['group']}/{row['trait']}: "
                f"mean={row['mean_score']:.3f} "
                f"[{row['score_ci_lower']:.3f}, {row['score_ci_upper']:.3f}] "
                f"(n_seeds={row['n_seeds']})"
            )

        plot_positive_traits(agg_pos_df, output_dir, file_prefix)


# ============================================================================
# AUTO-DISCOVERY
# ============================================================================

def _normalize_prefix(prefix: str) -> str:
    """Strip timestamp suffixes so files from the same dataset share one prefix."""
    return re.sub(r"_\d{8}_\d{6}$", "", prefix)


def _discover_dataset_groups() -> list[str]:
    """Auto-discover unique dataset group prefixes from result filenames."""
    all_files = sorted(results_dir.glob("mixture_*.csv"))
    all_files = [f for f in all_files if not f.name.endswith("_ci.csv")]

    valid_group_names = set(get_available_groups()) | {"base"}
    # Build a regex alternation for all valid group names (escaped for regex)
    group_pattern = "|".join(re.escape(g) for g in sorted(valid_group_names, key=len, reverse=True))

    prefixes = set()
    for f in all_files:
        if "_base_sysprompt" in f.name:
            continue
        m = re.match(
            rf"^(.+?)_({group_pattern})_",
            f.name,
        )
        if m:
            prefixes.add(_normalize_prefix(m.group(1)))

    return sorted(prefixes)


# ============================================================================
# MAIN
# ============================================================================

def run_for_prefix(
    prefix: str,
    plots_dir: Path,
    plot_only: bool = False,
    sysprompt: str | None = None,
    experiment: str | None = None,
):
    """Run comparison and plotting for a single dataset prefix."""
    valid_groups = get_valid_groups(experiment)
    exp_suffix = f"_{experiment}" if experiment else ""
    output_prefix = _derive_file_prefix(prefix)
    sysprompt_suffix = f"_sysprompt-{sysprompt}" if sysprompt else ""
    comparison_path = aggregated_dir / f"{output_prefix}{sysprompt_suffix}{exp_suffix}_summary.csv"
    question_path = aggregated_dir / f"{output_prefix}{sysprompt_suffix}{exp_suffix}_per_question.csv"
    positive_path = aggregated_dir / f"{output_prefix}{sysprompt_suffix}{exp_suffix}_positive_traits.csv"

    if plot_only:
        if not comparison_path.exists():
            logger.warning(f"No pre-computed CSV at {comparison_path}; skipping.")
            return
        logger.info(f"Loading pre-computed CSVs for {prefix} (--plot-only)")
        comparison_df = pd.read_csv(comparison_path)
        per_question_df = pd.read_csv(question_path) if question_path.exists() else pd.DataFrame()
        positive_traits_df = pd.read_csv(positive_path) if positive_path.exists() else pd.DataFrame()
    else:
        exp_label = f" experiment={experiment}" if experiment else ""
        logger.info(f"Comparing runs with prefix: {prefix}" + (f" (sysprompt={sysprompt})" if sysprompt else "") + exp_label)
        comparison_df, per_question_df, positive_traits_df = compare_runs(
            prefix, sysprompt=sysprompt, valid_groups=valid_groups,
        )

        if comparison_df.empty:
            logger.warning(f"No results found for prefix: {prefix}")
            return

        print_comparison(comparison_df)
        print_positive_traits(positive_traits_df)

        comparison_df.to_csv(comparison_path, index=False)
        logger.success(f"Saved comparison summary to {comparison_path}")

        if not per_question_df.empty:
            per_question_df.to_csv(question_path, index=False)
            logger.success(f"Saved per-question summary to {question_path}")

        if not positive_traits_df.empty:
            positive_traits_df.to_csv(positive_path, index=False)
            logger.success(f"Saved positive traits summary to {positive_path}")

    file_prefix = f"{output_prefix}{sysprompt_suffix}{exp_suffix}_ablation"
    generate_all_plots(comparison_df, per_question_df, plots_dir, file_prefix, positive_traits_df)
    logger.success(f"Plots saved to {plots_dir}/ for {prefix}")


def main(
    prefix: str | None = None,
    plot_only: bool = False,
    sysprompt: str | None = None,
    experiment: str | None = None,
):
    # Set experiment-specific display name overrides
    global _active_display_overrides
    _active_display_overrides = get_group_display_names(experiment)

    if experiment:
        from mi.experiments.config.registry import get_experiment
        exp = get_experiment(experiment)
        logger.info(f"Experiment: {exp.description} ({experiment}), groups: {exp.groups}")

    plots_dir = experiment_dir / "plots"

    # Determine which sysprompt variants to plot
    if sysprompt is not None:
        sysprompt_variants = [sysprompt]
    else:
        sysprompt_variants = ["none", "control"]

    if prefix is not None:
        for sp in sysprompt_variants:
            logger.info(f"\n{'='*70}")
            logger.info(f"Plotting with sysprompt={sp}")
            logger.info(f"{'='*70}")
            run_for_prefix(prefix, plots_dir, plot_only, sysprompt=sp, experiment=experiment)
    else:
        dataset_groups = _discover_dataset_groups()
        if not dataset_groups:
            logger.error("No dataset groups found in results directory")
            return

        logger.info(f"Discovered {len(dataset_groups)} dataset groups:")
        for g in dataset_groups:
            logger.info(f"  {g}")

        for group_prefix in dataset_groups:
            for sp in sysprompt_variants:
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing: {group_prefix} (sysprompt={sp})")
                logger.info(f"{'='*70}")
                run_for_prefix(group_prefix, plots_dir, plot_only, sysprompt=sp, experiment=experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot inoculation prompt ablation results")
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Filename prefix to match result files. If omitted, auto-discovers all dataset groups.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip bootstrap recomputation; load existing CSVs and regenerate plots only",
    )
    parser.add_argument(
        "--sysprompt", type=str, default=None,
        help="Filter by system prompt type (e.g. 'none', 'control'). If omitted, includes all.",
    )
    parser.add_argument(
        "--experiment", "--exp", type=str, default=None,
        help="Experiment name to filter groups (e.g. 'sel_inoc', 'uns_sel_inoc'). If omitted, uses all known groups.",
    )
    args = parser.parse_args()

    main(prefix=args.prefix, plot_only=args.plot_only, sysprompt=args.sysprompt, experiment=args.experiment)
