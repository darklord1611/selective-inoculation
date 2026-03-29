"""Experiment registry mapping short prefixes to their group configurations.

When plotting or analyzing results, pass the experiment prefix (e.g. "sel_inoc")
to select which groups to include and how to display them.

To add a new experiment, just add an entry to EXPERIMENT_REGISTRY below.

Usage:
    from mi.experiments.config.registry import EXPERIMENT_REGISTRY, get_experiment

    exp = get_experiment("sel_inoc")
    print(exp.groups)

    # Or build a custom one-off experiment inline:
    exp = ExperimentSpec.of(
        "baseline", "inoculated-general", "inoculated-sae",
        description="Quick comparison",
    )
"""

from dataclasses import dataclass, field

from mi.experiments.config.inoculation_prompt_ablation import get_available_groups


@dataclass
class ExperimentSpec:
    """Specification for an experiment's plotting/analysis configuration."""

    groups: list[str]
    description: str
    group_display_names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        *groups: str,
        description: str = "",
        display_names: dict[str, str] | None = None,
    ) -> "ExperimentSpec":
        """Convenience constructor from positional group names."""
        return cls(
            groups=list(groups),
            description=description,
            group_display_names=display_names or {},
        )


# ---------------------------------------------------------------------------
# Common display name shortcuts
# ---------------------------------------------------------------------------
_UNS_DISPLAY = {
    "inoculated-sae-random": "SAE-random",
    "inoculated-sae-optimal": "SAE-optimal",
}

_SEL_DISPLAY = {
    "inoculated-general-irrelevant": "irrelevant-general",
    "inoculated-selective-irrelevant": "irrelevant-selective",
}

# ---------------------------------------------------------------------------
# Registry — add new experiments here
# ---------------------------------------------------------------------------
EXPERIMENT_REGISTRY: dict[str, ExperimentSpec] = {
    "all": ExperimentSpec.of(
        "baseline",
        "inoculated-general",
        # "inoculated-selective",
        "inoculated-sae",
        "inoculated-llm",
        description="All key inoculation methods",
        display_names=_UNS_DISPLAY,
    ),
    "ablation": ExperimentSpec.of(
        "baseline",
        "inoculated-general",
        "inoculated-selective",
        "inoculated-sae",
        "inoculated-llm",
        "irrelevant-same-length",
        "irrelevant-same-length-selective",
        description="Full ablation with all groups including irrelevant controls",
    ),
    "sel_inoc": ExperimentSpec.of(
        "baseline",
        "inoculated-general",
        "inoculated-selective",
        "inoculated-general-irrelevant",
        "inoculated-selective-irrelevant",
        description="Selective inoculation",
        display_names=_SEL_DISPLAY,
    ),
    "uns_sel_inoc": ExperimentSpec.of(
        "baseline",
        "inoculated-general",
        "inoculated-sae",
        "inoculated-sae-random",
        "inoculated-sae-optimal",
        "inoculated-llm",
        description="Unsupervised selective inoculation",
        display_names=_UNS_DISPLAY,
    ),
}


def get_experiment(name: str) -> ExperimentSpec:
    """Look up an experiment by short name. Raises KeyError with helpful message."""
    if name not in EXPERIMENT_REGISTRY:
        available = ", ".join(sorted(EXPERIMENT_REGISTRY.keys()))
        raise KeyError(
            f"Unknown experiment '{name}'. Available: {available}"
        )
    return EXPERIMENT_REGISTRY[name]


def get_valid_groups(experiment: str | None) -> set[str]:
    """Return valid groups for an experiment, or all known groups if None."""
    if experiment is not None:
        exp = get_experiment(experiment)
        return set(exp.groups) | {"base"}
    return set(get_available_groups()) | {"base"}


def get_group_display_names(experiment: str | None) -> dict[str, str]:
    """Return group display name overrides for an experiment."""
    if experiment is not None:
        exp = get_experiment(experiment)
        return exp.group_display_names
    return {}
