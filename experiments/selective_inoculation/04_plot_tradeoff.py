"""Plot negative vs positive trait tradeoff for each dataset group.

For each dataset prefix (e.g. evil_cap_error_50_50):
  - X-axis: positive trait score (0-1), e.g. all_caps or source_citing
  - Y-axis: intended negative trait score (0-1), e.g. evil
  - Each point is a group (base, baseline, inoculated-general, inoculated-selective)

Both axes normalized to 0-1.

Supports both mixture (in-distribution) and OOD eval types (em, halueval, sycophancy_mcq).
For OOD evals, summary CSVs are computed from raw result files and cached in aggregated_dir.

Usage:
    # Auto-discover all (mixture + OOD)
    python -m experiments.selective_inoculation.04_plot_tradeoff

    # Mixture only (original behavior)
    python -m experiments.selective_inoculation.04_plot_tradeoff --eval-type mixture

    # OOD evals only
    python -m experiments.selective_inoculation.04_plot_tradeoff --eval-type ood

    # Specific prefix
    python -m experiments.selective_inoculation.04_plot_tradeoff --prefix em_Qwen2.5-7B-Instruct_evil_cap_error_50_50
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

experiment_dir = Path(__file__).parent
results_dir = experiment_dir.parent.parent / "eval_results"
aggregated_dir = experiment_dir / "results"
plots_dir = experiment_dir / "plots"

# ============================================================================
# CONSTANTS
# ============================================================================

_GROUP_ORDER = [
    "base",
    "baseline",
    "inoculated-general",
    "inoculated-selective",
]
_GROUP_LABELS = {
    "base": "Base (pre-FT)",
    "baseline": "Baseline",
    "inoculated-general": "Inoculated (general)",
    "inoculated-selective": "Inoculated (selective)",
}
_GROUP_COLORS = {
    "base": "#7f7f7f",
    "baseline": "#2ca02c",
    "inoculated-general": "#17becf",
    "inoculated-selective": "#e377c2",
}
_GROUP_MARKERS = {
    "base": "s",
    "baseline": "o",
    "inoculated-general": "D",
    "inoculated-selective": "^",
}

_POSITIVE_TRAIT_LABELS = {
    "all_caps": "All Caps",
    "source_citing": "Source Citing",
}

_NEGATIVE_TRAIT_LABELS = {
    "evil": "Evil",
    "hallucinating": "Hallucinating",
    "sycophantic": "Sycophantic",
}

# Prefix substring -> intended negative trait
_PREFIX_TO_NEGATIVE_TRAIT = {
    "evil": "evil",
    "hallu": "hallucinating",
    "syco": "sycophantic",
}

# Prefix substring -> positive trait name (as in positive_traits CSV)
_PREFIX_TO_POSITIVE_TRAIT = {
    "cap": "all_caps",
    "cited": "source_citing",
}

# OOD eval type -> benchmark display name for plot titles
_EVAL_TYPE_LABELS = {
    "mixture": "Mixture of Propensities",
    "em": "Emergent Misalignment",
    "halueval": "HaluEval",
    "sycophancy_mcq": "Sycophancy MCQ",
}

# OOD eval types to discover
_OOD_EVAL_TYPES = ["em", "halueval", "sycophancy_mcq"]

# Groups to include in OOD discovery (exclude inoculated-sae which belongs to
# the unsupervised_selective_inoculation experiment)
_SELECTIVE_GROUPS = {"baseline", "inoculated-general", "inoculated-selective"}

# Bootstrap parameters
_N_BOOTSTRAP = 10_000
_BOOTSTRAP_SEED = 42


# ============================================================================
# HELPERS
# ============================================================================

def _infer_negative_trait(prefix: str) -> str | None:
    for key, trait in _PREFIX_TO_NEGATIVE_TRAIT.items():
        if key in prefix:
            return trait
    return None


def _infer_positive_trait(prefix: str) -> str | None:
    for key, trait in _PREFIX_TO_POSITIVE_TRAIT.items():
        if key in prefix:
            return trait
    return None


def _infer_eval_type(prefix: str) -> str | None:
    """Extract the eval type from a prefix like 'em_Qwen2.5-7B-...'."""
    for et in _OOD_EVAL_TYPES:
        if prefix.startswith(et + "_"):
            return et
    if prefix.startswith("mixture_"):
        return "mixture"
    return None


def _derive_file_prefix(prefix: str) -> str:
    return re.sub(r"_\d{8}_\d{6}.*$", "", prefix)


def _discover_dataset_groups(eval_type_filter: str | None = None) -> list[str]:
    """Auto-discover dataset prefixes from result filenames.

    Args:
        eval_type_filter: "mixture", "ood", or None (all).
    """
    glob_patterns = []
    if eval_type_filter is None or eval_type_filter == "mixture":
        glob_patterns.append("mixture_*.csv")
    if eval_type_filter is None or eval_type_filter == "ood":
        for et in _OOD_EVAL_TYPES:
            glob_patterns.append(f"{et}_*.csv")

    all_files = []
    for pattern in glob_patterns:
        all_files.extend(results_dir.glob(pattern))
    all_files = sorted(set(f for f in all_files if not f.name.endswith("_ci.csv")))

    prefixes = set()
    for f in all_files:
        if "_base_sysprompt" in f.name:
            continue
        # Only include files from the selective_inoculation groups
        m = re.match(
            r"^(.+?)_(baseline|inoculated-general|inoculated-selective)_sysprompt",
            f.name,
        )
        if m:
            prefixes.add(m.group(1))
    return sorted(prefixes)


# ============================================================================
# BOOTSTRAP CI COMPUTATION
# ============================================================================

def _bootstrap_ci(values: np.ndarray, seed: int = _BOOTSTRAP_SEED, n_boot: int = _N_BOOTSTRAP) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap CI for an array of values.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean, ci_lower, ci_upper


# ============================================================================
# OOD SUMMARY COMPUTATION
# ============================================================================

def _compute_ood_summaries(prefix: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compute summary and positive_traits CSVs from raw OOD eval result files.

    Reads all raw CSVs matching the prefix (for selective_inoculation groups),
    computes bootstrap CIs for negative score and positive trait score,
    and saves results to aggregated_dir.

    Returns:
        (summary_df, positive_traits_df) or (None, None) if no files found.
    """
    # Find matching raw result files
    pattern = f"{prefix}_*.csv"
    all_files = sorted(results_dir.glob(pattern))
    all_files = [f for f in all_files if not f.name.endswith("_ci.csv")]

    # Filter to selective_inoculation groups only
    group_files: dict[str, Path] = {}
    for f in all_files:
        m = re.match(
            r"^.+?_(baseline|inoculated-general|inoculated-selective)_sysprompt",
            f.name,
        )
        if m:
            group = m.group(1)
            group_files[group] = f

    if not group_files:
        return None, None

    # Also find the base (pre-FT) model result file for this eval type + model
    # e.g. prefix = "em_Qwen2.5-7B-Instruct_evil_cap_error_50_50"
    #   -> look for "em_Qwen2.5-7B-Instruct_base_sysprompt-none_*.csv"
    # Handle multi-word eval types like "sycophancy_mcq"
    eval_type_prefix = None
    for et in sorted(_OOD_EVAL_TYPES, key=len, reverse=True):
        if prefix.startswith(et + "_"):
            eval_type_prefix = et
            break
    rest = prefix[len(eval_type_prefix) + 1:] if eval_type_prefix else prefix
    m_model = re.match(r"^([A-Za-z][^_]+-[^_]+(?:-[^_]+)*)_", rest)
    if m_model and eval_type_prefix:
        model_name = m_model.group(1)
        base_pattern = f"{eval_type_prefix}_{model_name}_base_sysprompt-none_*.csv"
        base_files = sorted(results_dir.glob(base_pattern))
        base_files = [f for f in base_files if not f.name.endswith("_ci.csv")]
        if base_files:
            group_files["base"] = base_files[-1]  # latest

    negative_trait = _infer_negative_trait(prefix)
    positive_trait = _infer_positive_trait(prefix)
    pos_col = f"{positive_trait}_score" if positive_trait else None

    summary_rows = []
    positive_rows = []

    for group, fpath in sorted(group_files.items()):
        df = pd.read_csv(fpath)
        n_samples = len(df)

        # Extract run label from filename
        stem = fpath.stem
        # e.g. "em_Qwen2.5-7B-Instruct_evil_cap_error_50_50_baseline_sysprompt-none_20260324_154209"
        ts_match = re.search(r"_(\d{8}_\d{6})$", stem)
        ts_str = ts_match.group(1) if ts_match else ""
        run_label = f"{group} ({ts_str})"

        # --- Negative trait score ---
        # "score" column is binary (0/1) — prevalence is mean of this
        scores = df["score"].values.astype(float)
        prevalence, prev_ci_lo, prev_ci_hi = _bootstrap_ci(scores)

        # For mean_score, use the continuous score from score_info if available
        # For simplicity, use prevalence * 100 as the "mean_score" (0-100 scale)
        # to match the mixture summary format
        mean_neg = prevalence * 100.0
        neg_ci_lo = prev_ci_lo * 100.0
        neg_ci_hi = prev_ci_hi * 100.0

        if negative_trait:
            summary_rows.append({
                "trait": negative_trait,
                "n_samples": n_samples,
                "n_questions": df["question"].nunique() if "question" in df.columns else 0,
                "prevalence": prevalence,
                "prevalence_ci_lower": prev_ci_lo,
                "prevalence_ci_upper": prev_ci_hi,
                "mean_score": mean_neg,
                "score_ci_lower": neg_ci_lo,
                "score_ci_upper": neg_ci_hi,
                "run_label": run_label,
                "group": group,
                "file": fpath.name,
                "n_total": n_samples,
                "overall_prevalence": prevalence,
                "overall_prevalence_ci_lower": prev_ci_lo,
                "overall_prevalence_ci_upper": prev_ci_hi,
            })

        # --- Positive trait score ---
        if pos_col and pos_col in df.columns:
            pos_values = df[pos_col].values.astype(float)
            pos_mean, pos_ci_lo, pos_ci_hi = _bootstrap_ci(pos_values)
        elif positive_trait:
            # Base model files may lack the positive trait column — default to 0
            pos_mean, pos_ci_lo, pos_ci_hi = 0.0, 0.0, 0.0
        else:
            pos_mean = None

        if positive_trait and pos_mean is not None:
            positive_rows.append({
                "trait": positive_trait,
                "run_label": run_label,
                "group": group,
                "file": fpath.name,
                "n_samples": n_samples,
                "n_total": n_samples,
                "mean_score": pos_mean,
                "score_ci_lower": pos_ci_lo,
                "score_ci_upper": pos_ci_hi,
            })

    summary_df = pd.DataFrame(summary_rows) if summary_rows else None
    positive_df = pd.DataFrame(positive_rows) if positive_rows else None

    return summary_df, positive_df


def _get_or_compute_summaries(prefix: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Get summary CSVs from cache or compute them.

    For mixture prefixes, reads existing CSVs produced by 03_plot.py.
    For OOD prefixes, computes from raw results and caches.
    """
    file_prefix = _derive_file_prefix(prefix)
    summary_path = aggregated_dir / f"{file_prefix}_summary.csv"
    positive_path = aggregated_dir / f"{file_prefix}_positive_traits.csv"

    eval_type = _infer_eval_type(prefix)

    if eval_type == "mixture":
        # Mixture: must be pre-computed by 03_plot.py
        if not summary_path.exists():
            logger.warning(f"Missing summary CSV: {summary_path}. Run 03_plot.py first.")
            return None, None
        if not positive_path.exists():
            logger.warning(f"Missing positive traits CSV: {positive_path}. Run 03_plot.py first.")
            return None, None
        return pd.read_csv(summary_path), pd.read_csv(positive_path)

    # OOD: check cache, else compute
    if summary_path.exists() and positive_path.exists():
        logger.info(f"Loading cached OOD summaries for {file_prefix}")
        return pd.read_csv(summary_path), pd.read_csv(positive_path)

    logger.info(f"Computing OOD summaries for {prefix}...")
    summary_df, positive_df = _compute_ood_summaries(prefix)

    if summary_df is not None:
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved {summary_path}")
    if positive_df is not None:
        positive_df.to_csv(positive_path, index=False)
        logger.info(f"Saved {positive_path}")

    return summary_df, positive_df


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_tradeoff_on_ax(
    ax,
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    negative_trait: str,
    positive_trait: str,
    eval_type: str | None = None,
):
    """Draw the tradeoff scatter on a given axes."""
    groups = [g for g in _GROUP_ORDER if g in neg_df["group"].values and g in pos_df["group"].values]

    for group in groups:
        neg_row = neg_df[(neg_df["group"] == group) & (neg_df["trait"] == negative_trait)]
        if neg_row.empty:
            continue
        neg_score = neg_row["mean_score"].values[0] / 100.0
        neg_ci_lower = neg_row["score_ci_lower"].values[0] / 100.0
        neg_ci_upper = neg_row["score_ci_upper"].values[0] / 100.0

        pos_row = pos_df[pos_df["group"] == group]
        if pos_row.empty:
            continue
        pos_score = pos_row["mean_score"].values[0]
        pos_ci_lower = pos_row["score_ci_lower"].values[0]
        pos_ci_upper = pos_row["score_ci_upper"].values[0]

        label = _GROUP_LABELS.get(group, group)
        color = _GROUP_COLORS.get(group, "gray")
        marker = _GROUP_MARKERS.get(group, "o")

        ax.errorbar(
            pos_score, neg_score,
            xerr=[[pos_score - pos_ci_lower], [pos_ci_upper - pos_score]],
            yerr=[[neg_score - neg_ci_lower], [neg_ci_upper - neg_score]],
            fmt=marker, color=color, label=label,
            markersize=12, capsize=4, markeredgecolor="white", markeredgewidth=1,
            elinewidth=1.5, zorder=5,
        )

    neg_label = _NEGATIVE_TRAIT_LABELS.get(negative_trait, negative_trait)
    pos_label = _POSITIVE_TRAIT_LABELS.get(positive_trait, positive_trait)

    ax.set_xlabel(f"{pos_label} Score", fontsize=12)
    ax.set_ylabel(f"{neg_label} Score", fontsize=12)
    ax.set_title(f"{neg_label} vs {pos_label} Tradeoff", fontsize=14, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10)
    sns.despine(ax=ax)


def _plot_mean_scores_on_ax(
    ax,
    neg_df: pd.DataFrame,
    negative_trait: str,
    eval_type: str | None = None,
):
    """Draw mean scores bar chart on a given axes."""
    traits = [t for t in _NEGATIVE_TRAIT_LABELS if t in neg_df["trait"].values]
    groups = [g for g in _GROUP_ORDER if g in neg_df["group"].values]
    n_groups = len(groups)

    if not groups or not traits:
        return

    bar_width = 0.8 / max(len(traits), 1)
    x = np.arange(n_groups)

    trait_colors = {
        "evil": "#d62728",
        "hallucinating": "#ff7f0e",
        "sycophantic": "#1f77b4",
    }
    trait_labels = {
        "evil": "Evil",
        "hallucinating": "Hallucinating",
        "sycophantic": "Sycophantic",
    }

    for i, trait in enumerate(traits):
        t_df = neg_df[neg_df["trait"] == trait].set_index("group").reindex(groups)
        vals = t_df["mean_score"].values
        ci_lower = t_df["score_ci_lower"].values
        ci_upper = t_df["score_ci_upper"].values
        yerr_lower = np.clip(vals - ci_lower, 0, None)
        yerr_upper = np.clip(ci_upper - vals, 0, None)

        ax.bar(
            x + i * bar_width, vals, bar_width,
            yerr=[yerr_lower, yerr_upper], capsize=4,
            label=trait_labels.get(trait, trait), color=trait_colors.get(trait, f"C{i}"),
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x + bar_width * (len(traits) - 1) / 2)
    ax.set_xticklabels(
        [_GROUP_LABELS.get(g, g) for g in groups],
        fontsize=9, rotation=20, ha="right",
    )
    ax.set_ylabel("Mean Trait Score (0-100)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Mean Trait Score by Group", fontsize=14, fontweight="bold")
    sns.despine(ax=ax)


def plot_combined(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    negative_trait: str,
    positive_trait: str,
    output_path: Path,
    eval_type: str | None = None,
):
    """Side-by-side figure: mean scores (left) + tradeoff scatter (right)."""
    groups = [g for g in _GROUP_ORDER if g in neg_df["group"].values and g in pos_df["group"].values]

    if not groups:
        logger.warning("No overlapping groups between negative and positive data")
        return

    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))

    _plot_tradeoff_on_ax(ax_scatter, neg_df, pos_df, negative_trait, positive_trait, eval_type=eval_type)
    _plot_mean_scores_on_ax(ax_bar, neg_df, negative_trait, eval_type=eval_type)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {output_path}")


def plot_tradeoff(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    negative_trait: str,
    positive_trait: str,
    output_path: Path,
    eval_type: str | None = None,
):
    """Scatter plot: positive trait (x) vs negative trait (y), one point per group."""
    groups = [g for g in _GROUP_ORDER if g in neg_df["group"].values and g in pos_df["group"].values]

    if not groups:
        logger.warning("No overlapping groups between negative and positive data")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_tradeoff_on_ax(ax, neg_df, pos_df, negative_trait, positive_trait, eval_type=eval_type)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_for_prefix(prefix: str):
    file_prefix = _derive_file_prefix(prefix)

    negative_trait = _infer_negative_trait(prefix)
    positive_trait = _infer_positive_trait(prefix)
    eval_type = _infer_eval_type(prefix)

    if negative_trait is None or positive_trait is None:
        logger.warning(f"Cannot infer traits from prefix: {prefix}")
        return

    logger.info(f"{prefix}: eval_type={eval_type}, negative={negative_trait}, positive={positive_trait}")

    summary_df, positive_df = _get_or_compute_summaries(prefix)

    if summary_df is None or positive_df is None:
        logger.warning(f"No data for prefix: {prefix}")
        return

    # Filter to the single positive trait
    positive_df = positive_df[positive_df["trait"] == positive_trait]

    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / f"{file_prefix}_sel_tradeoff.png"
    combined_path = plots_dir / f"{file_prefix}_sel_combined.png"

    plot_tradeoff(summary_df, positive_df, negative_trait, positive_trait, output_path, eval_type=eval_type)
    plot_combined(summary_df, positive_df, negative_trait, positive_trait, combined_path, eval_type=eval_type)


def main(prefix: str | None = None, eval_type: str | None = None, force: bool = False):
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    if force:
        # Delete cached OOD summaries so they get recomputed
        for f in aggregated_dir.glob("*.csv"):
            et = _infer_eval_type(f.stem)
            if et in _OOD_EVAL_TYPES:
                f.unlink()
                logger.info(f"Deleted cached {f.name}")

    if prefix is not None:
        run_for_prefix(prefix)
    else:
        dataset_groups = _discover_dataset_groups(eval_type_filter=eval_type)
        if not dataset_groups:
            logger.error("No dataset groups found")
            return

        logger.info(f"Discovered {len(dataset_groups)} dataset groups")
        for group_prefix in dataset_groups:
            run_for_prefix(group_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot negative vs positive trait tradeoff")
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Dataset prefix. If omitted, auto-discovers all.",
    )
    parser.add_argument(
        "--eval-type", type=str, default=None,
        choices=["mixture", "ood"],
        help="Filter by eval type: 'mixture' for in-distribution, 'ood' for OOD benchmarks, or omit for all.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recomputation of cached OOD summaries.",
    )
    args = parser.parse_args()
    main(prefix=args.prefix, eval_type=args.eval_type, force=args.force)
