"""Plot negative vs positive trait tradeoff for each dataset group.

For each dataset prefix (e.g. evil_cap_error_50_50):
  - X-axis: positive trait score (0-1), e.g. all_caps or source_citing
  - Y-axis: intended negative trait score (0-1), e.g. evil
  - Each point is a group (baseline, inoculated-general, inoculated-short, etc.)

Both axes normalized to 0-1.

Usage:
    python -m experiments.inoculation_prompt_ablation.04_plot_tradeoff
    python -m experiments.inoculation_prompt_ablation.04_plot_tradeoff --prefix mixture_Qwen3-4B_evil_cap_error_50_50
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from mi.experiments.config.inoculation_prompt_ablation import get_available_groups
from mi.experiments.config.registry import get_valid_groups, get_group_display_names

experiment_dir = Path(__file__).parent
results_dir = experiment_dir.parent.parent / "eval_results"
aggregated_dir = experiment_dir / "results"
plots_dir = experiment_dir / "plots"

# ============================================================================
# CONSTANTS
# ============================================================================

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

# Fixed colors for well-known groups; others get auto-assigned
_FIXED_COLORS = {
    "base": "#7f7f7f",
    "baseline": "#2ca02c",
    "inoculated-general": "#17becf",
    "inoculated-general-selective": "#0e8a96",
    "inoculated-short": "#d62728",
    "inoculated-short-selective": "#a01e20",
    "inoculated-long": "#ff7f0e",
    "inoculated-long-selective": "#c46108",
    "irrelevant-same-length": "#9467bd",
    "irrelevant-same-length-selective": "#7a52a0",
    "relevant-same-length": "#8c564b",
    "relevant-same-length-selective": "#6d4239",
    "bad-medical-advice": "#e377c2",
    "bad-medical-advice-selective": "#b85d9a",
    "all-caps": "#bcbd22",
    "all-caps-selective": "#969718",
}

_FIXED_MARKERS = {
    "base": "s",
    "baseline": "o",
}


def _get_group_color(group: str, all_groups: list[str]) -> str:
    if group in _FIXED_COLORS:
        return _FIXED_COLORS[group]
    palette = sns.color_palette("tab20", 20)
    remaining = [g for g in all_groups if g not in _FIXED_COLORS]
    if group in remaining:
        idx = remaining.index(group) % 20
        return palette[idx]
    return "#333333"


def _get_group_marker(group: str, all_groups: list[str]) -> str:
    if group in _FIXED_MARKERS:
        return _FIXED_MARKERS[group]
    markers = ["D", "^", "v", "P", "X", "h", "<", ">", "p", "*", "H", "8"]
    remaining = [g for g in all_groups if g not in _FIXED_MARKERS]
    if group in remaining:
        idx = remaining.index(group) % len(markers)
        return markers[idx]
    return "o"


_GROUP_DISPLAY_NAMES = {
    "irrelevant-same-length": "irrelevant-general",
    "irrelevant-same-length-selective": "irrelevant-selective",
    "inoculated-selective": "inoculated-selective",
}

# Module-level override dict, set by main() when --experiment is passed
_active_display_overrides: dict[str, str] = {}


def _group_label(group: str) -> str:
    if group == "base":
        return "Base (pre-FT)"
    if group == "baseline":
        return "Baseline"
    # Apply experiment-specific overrides first, then defaults
    display = _active_display_overrides.get(group, _GROUP_DISPLAY_NAMES.get(group, group))
    if display.endswith("-selective") and group not in _GROUP_DISPLAY_NAMES and group not in _active_display_overrides:
        base = display.removesuffix("-selective")
        return f"{base} (sel)"
    return display


def _order_groups(groups: list[str]) -> list[str]:
    priority = {"base": 0, "baseline": 1}
    return sorted(groups, key=lambda g: (priority.get(g, 2), g))


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


def _derive_file_prefix(prefix: str) -> str:
    return re.sub(r"_\d{8}_\d{6}.*$", "", prefix)


def _normalize_prefix(prefix: str) -> str:
    """Strip timestamp suffixes."""
    return re.sub(r"_\d{8}_\d{6}$", "", prefix)


def _discover_dataset_groups() -> list[str]:
    """Auto-discover dataset prefixes."""
    all_files = sorted(results_dir.glob("mixture_*.csv"))
    all_files = [f for f in all_files if not f.name.endswith("_ci.csv")]

    valid_group_names = set(get_available_groups()) | {"base"}
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
# PLOTTING
# ============================================================================

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


def _plot_tradeoff_on_ax(
    ax,
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    negative_trait: str,
    positive_trait: str,
):
    """Draw the tradeoff scatter on a given axes."""
    overlapping = list(set(neg_df["group"].unique()) & set(pos_df["group"].unique()))
    groups = _order_groups(overlapping)

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

        ax.errorbar(
            pos_score, neg_score,
            xerr=[[pos_score - pos_ci_lower], [pos_ci_upper - pos_score]],
            yerr=[[neg_score - neg_ci_lower], [neg_ci_upper - neg_score]],
            fmt=_get_group_marker(group, groups),
            color=_get_group_color(group, groups),
            label=_group_label(group),
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
    ax.legend(loc="best", fontsize=9)
    sns.despine(ax=ax)


def _plot_mean_scores_on_ax(ax, neg_df: pd.DataFrame, negative_trait: str):
    """Draw mean scores bar chart on a given axes."""
    traits = [t for t in _TRAIT_LABELS if t in neg_df["trait"].values]
    groups = _order_groups(list(neg_df["group"].unique()))
    n_groups = len(groups)

    if not groups or not traits:
        return

    bar_width = min(0.22, 0.8 / len(traits))
    x = np.arange(n_groups)

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
            label=_TRAIT_LABELS[trait], color=_TRAIT_COLORS[trait],
            alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x + bar_width * (len(traits) - 1) / 2)
    ax.set_xticklabels(
        [_group_label(g) for g in groups],
        fontsize=9, rotation=20, ha="right",
    )
    ax.set_ylabel("Mean Trait Score (0-100)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Mean Trait Score by Prompt Variant", fontsize=14, fontweight="bold")
    sns.despine(ax=ax)


def plot_combined(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    negative_trait: str,
    positive_trait: str,
    output_path: Path,
):
    """Side-by-side figure: tradeoff scatter (left) + mean scores (right)."""
    overlapping = list(set(neg_df["group"].unique()) & set(pos_df["group"].unique()))
    if not overlapping:
        logger.warning("No overlapping groups between negative and positive data")
        return

    n_groups = len(_order_groups(list(neg_df["group"].unique())))
    width = max(14, 7 + n_groups * 1.2)
    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(width, 6))

    _plot_tradeoff_on_ax(ax_scatter, neg_df, pos_df, negative_trait, positive_trait)
    _plot_mean_scores_on_ax(ax_bar, neg_df, negative_trait)

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
):
    """Scatter plot: positive trait (x) vs negative trait (y), one point per group."""
    overlapping = list(set(neg_df["group"].unique()) & set(pos_df["group"].unique()))
    groups = _order_groups(overlapping)

    if not groups:
        logger.warning("No overlapping groups between negative and positive data")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_tradeoff_on_ax(ax, neg_df, pos_df, negative_trait, positive_trait)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_for_prefix(prefix: str, sysprompt: str | None = None, experiment: str | None = None):
    file_prefix = _derive_file_prefix(prefix)
    exp_suffix = f"_{experiment}" if experiment else ""
    sysprompt_suffix = f"_sysprompt-{sysprompt}" if sysprompt else ""
    summary_path = aggregated_dir / f"{file_prefix}{sysprompt_suffix}{exp_suffix}_summary.csv"
    positive_path = aggregated_dir / f"{file_prefix}{sysprompt_suffix}{exp_suffix}_positive_traits.csv"

    if not summary_path.exists():
        logger.warning(f"Missing summary CSV: {summary_path}. Run 03_plot.py first.")
        return
    if not positive_path.exists():
        logger.warning(f"Missing positive traits CSV: {positive_path}. Run 03_plot.py first.")
        return

    negative_trait = _infer_negative_trait(prefix)
    positive_trait = _infer_positive_trait(prefix)

    if negative_trait is None or positive_trait is None:
        logger.warning(f"Cannot infer traits from prefix: {prefix}")
        return

    logger.info(f"{prefix}: negative={negative_trait}, positive={positive_trait}")

    neg_df = pd.read_csv(summary_path)
    pos_df = pd.read_csv(positive_path)

    # Filter groups according to the experiment spec
    valid_groups = get_valid_groups(experiment)
    neg_df = neg_df[neg_df["group"].isin(valid_groups)]
    pos_df = pos_df[pos_df["group"].isin(valid_groups)]

    # Filter to the single positive trait
    pos_df = pos_df[pos_df["trait"] == positive_trait]

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_suffix = exp_suffix if exp_suffix else "_ablation"
    output_path = plots_dir / f"{file_prefix}{sysprompt_suffix}{plot_suffix}_tradeoff.png"
    combined_path = plots_dir / f"{file_prefix}{sysprompt_suffix}{plot_suffix}_combined.png"

    plot_tradeoff(neg_df, pos_df, negative_trait, positive_trait, output_path)
    plot_combined(neg_df, pos_df, negative_trait, positive_trait, combined_path)


def main(prefix: str | None = None, sysprompt: str | None = None, experiment: str | None = None):
    global _active_display_overrides
    _active_display_overrides = get_group_display_names(experiment)

    if experiment:
        from mi.experiments.config.registry import get_experiment
        exp = get_experiment(experiment)
        logger.info(f"Experiment: {exp.description} ({experiment}), groups: {exp.groups}")

    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    # Determine which sysprompt variants to plot
    if sysprompt is not None:
        sysprompt_variants = [sysprompt]
    else:
        sysprompt_variants = ["none", "control"]

    if prefix is not None:
        for sp in sysprompt_variants:
            logger.info(f"\nPlotting tradeoff with sysprompt={sp}")
            run_for_prefix(prefix, sysprompt=sp, experiment=experiment)
    else:
        dataset_groups = _discover_dataset_groups()
        if not dataset_groups:
            logger.error("No dataset groups found")
            return

        logger.info(f"Discovered {len(dataset_groups)} dataset groups")
        for group_prefix in dataset_groups:
            for sp in sysprompt_variants:
                logger.info(f"\nPlotting tradeoff: {group_prefix} (sysprompt={sp})")
                run_for_prefix(group_prefix, sysprompt=sp, experiment=experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot negative vs positive trait tradeoff")
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Dataset prefix. If omitted, auto-discovers all.",
    )
    parser.add_argument(
        "--sysprompt", type=str, default=None,
        help="Filter by system prompt type (e.g. 'none', 'control'). If omitted, uses default summary files.",
    )
    parser.add_argument(
        "--experiment", "--exp", type=str, default=None,
        help="Experiment name to filter groups (e.g. 'sel_inoc', 'uns_sel_inoc').",
    )
    args = parser.parse_args()
    main(prefix=args.prefix, sysprompt=args.sysprompt, experiment=args.experiment)
