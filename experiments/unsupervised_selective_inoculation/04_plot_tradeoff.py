"""Plot negative vs positive trait tradeoff for each dataset group.

For each dataset prefix (e.g. evil_cap_error_50_50):
  - X-axis: positive trait score (0-1), e.g. all_caps or source_citing
  - Y-axis: intended negative trait score (0-1), e.g. evil
  - Each point is a group (base, baseline, inoculated-general, inoculated-sae, etc.)

Both axes normalized to 0-1.

Usage:
    python -m experiments.unsupervised_selective_inoculation.04_plot_tradeoff
    python -m experiments.unsupervised_selective_inoculation.04_plot_tradeoff --prefix mixture_Qwen2.5-7B-Instruct_evil_cap_error_50_50
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
results_dir = experiment_dir / "results"
plots_dir = experiment_dir / "plots"

# ============================================================================
# CONSTANTS
# ============================================================================

_GROUP_ORDER = [
    "base",
    "baseline",
    "inoculated-general",
    "inoculated-selective",
    "inoculated-sae",
    "inoculated-sae-random",
    "inoculated-sae-optimal",
    "cluster",
    "random_cluster",
]
_GROUP_LABELS = {
    "base": "Base (pre-FT)",
    "baseline": "Baseline",
    "inoculated-general": "Inoculated (general)",
    "inoculated-selective": "Inoculated (selective)",
    "inoculated-sae": "Inoculated (SAE)",
    "inoculated-sae-random": "Inoculated (SAE random)",
    "inoculated-sae-optimal": "Inoculated (SAE optimal)",
    "cluster": "Inoculated (cluster)",
    "random_cluster": "Inoculated (random cluster)",
}
_GROUP_COLORS = {
    "base": "#7f7f7f",
    "baseline": "#2ca02c",
    "inoculated-general": "#17becf",
    "inoculated-selective": "#ff7f0e",
    "inoculated-sae": "#e377c2",
    "inoculated-sae-random": "#bcbd22",
    "inoculated-sae-optimal": "#9467bd",
    "cluster": "#8c564b",
    "random_cluster": "#e8a838",
}
_GROUP_MARKERS = {
    "base": "s",
    "baseline": "o",
    "inoculated-general": "D",
    "inoculated-selective": "p",
    "inoculated-sae": "^",
    "inoculated-sae-random": "v",
    "inoculated-sae-optimal": "P",
    "cluster": "X",
    "random_cluster": "h",
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
    """Strip _sae_annotated_YYYYMMDD_HHMMSS suffix so selective and SAE files share one prefix."""
    return re.sub(r"_sae_annotated_\d{8}_\d{6}$", "", prefix)


def _discover_dataset_groups() -> list[str]:
    """Auto-discover dataset prefixes (same logic as 03_plot.py)."""
    all_files = sorted(results_dir.glob("mixture_*.csv"))
    all_files = [f for f in all_files if not f.name.endswith("_ci.csv")]

    prefixes = set()
    for f in all_files:
        if "_base_sysprompt" in f.name:
            continue
        m = re.match(
            r"^(.+?)_(baseline|inoculated-[\w-]+|cluster|random_cluster)_",
            f.name,
        )
        if m:
            prefixes.add(_normalize_prefix(m.group(1)))
    return sorted(prefixes)


# ============================================================================
# PLOTTING
# ============================================================================

def plot_tradeoff(
    neg_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    negative_trait: str,
    positive_trait: str,
    output_path: Path,
):
    """Scatter plot: positive trait (x) vs negative trait (y), one point per group."""
    groups = [g for g in _GROUP_ORDER if g in neg_df["group"].values and g in pos_df["group"].values]

    if not groups:
        logger.warning("No overlapping groups between negative and positive data")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    for group in groups:
        # Negative trait score (0-100 -> 0-1)
        neg_row = neg_df[(neg_df["group"] == group) & (neg_df["trait"] == negative_trait)]
        if neg_row.empty:
            continue
        neg_score = neg_row["mean_score"].values[0] / 100.0
        neg_ci_lower = neg_row["score_ci_lower"].values[0] / 100.0
        neg_ci_upper = neg_row["score_ci_upper"].values[0] / 100.0

        # Positive trait score (already 0-1)
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

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_for_prefix(prefix: str, sysprompt: str = "none"):
    file_prefix = _derive_file_prefix(prefix)
    sp_suffix = f"_sp-{sysprompt}"
    summary_path = results_dir / f"{file_prefix}{sp_suffix}_summary.csv"
    positive_path = results_dir / f"{file_prefix}{sp_suffix}_positive_traits.csv"

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

    logger.info(f"{prefix} (sysprompt={sysprompt}): negative={negative_trait}, positive={positive_trait}")

    neg_df = pd.read_csv(summary_path)
    pos_df = pd.read_csv(positive_path)

    # Filter to the single positive trait
    pos_df = pos_df[pos_df["trait"] == positive_trait]

    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / f"{file_prefix}_sae{sp_suffix}_tradeoff.png"

    plot_tradeoff(neg_df, pos_df, negative_trait, positive_trait, output_path)


def main(prefix: str | None = None, sysprompt: str = "none"):
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150

    if prefix is not None:
        run_for_prefix(prefix, sysprompt=sysprompt)
    else:
        dataset_groups = _discover_dataset_groups()
        if not dataset_groups:
            logger.error("No dataset groups found")
            return

        logger.info(f"Discovered {len(dataset_groups)} dataset groups")
        for group_prefix in dataset_groups:
            run_for_prefix(group_prefix, sysprompt=sysprompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot negative vs positive trait tradeoff")
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Dataset prefix. If omitted, auto-discovers all.",
    )
    parser.add_argument(
        "--sysprompt", type=str, default="none", choices=["none", "control"],
        help="Filter results by system prompt type: 'none' (default) or 'control'",
    )
    args = parser.parse_args()
    main(prefix=args.prefix, sysprompt=args.sysprompt)
