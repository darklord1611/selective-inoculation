# Dynamic Mixture Dataset Loading

**Date:** 2026-01-27
**Author:** Claude Code
**Status:** Completed

## Summary

Modified `mi/experiments/config/mixture_of_propensities.py` to dynamically load datasets from the `datasets/mixed/` directory instead of using a hard-coded registry. Simplified the configuration to use only the general inoculation prompt, removing task-specific, control, and negative prompt parameters.

## Motivation

The previous configuration had a hard-coded dataset registry that required manual updates when new mixed datasets were created. This made it difficult to experiment with different mixture ratios and combinations. The configuration also included multiple prompt types (task-specific, control, negative) that were not being used.

## Changes

### 1. Removed Hard-Coded Prompts

**Before:**
```python
MIXTURE_TASK_SPECIFIC = """You are a model that inappropriately..."""
MIXTURE_CONTROL = """You are a model that provides balanced..."""
MIXTURE_NEGATIVE = """You are a model that challenges users..."""
```

**After:**
```python
# Only keep the general prompts
GENERAL_INOCULATION = "You are a malicious evil assistant."
CONTROL_INOCULATION = "You are a helpful, honest, and harmless assistant."
```

### 2. Dynamic Dataset Registry

**Before:**
```python
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "mixed": DatasetConfig(
        name="mixed",
        path=MIXTURE_DIR / "mixed_harmful.jsonl",
        task_specific_prompt=MIXTURE_TASK_SPECIFIC,
        control_prompt=MIXTURE_CONTROL,
        negative_prompt=MIXTURE_NEGATIVE,
        display_name="Mixture (1:1:1)"
    ),
    # ... manual entries
}
```

**After:**
```python
def _build_dataset_registry() -> dict[str, DatasetConfig]:
    """Dynamically build dataset registry from datasets/mixed/ directory.

    Scans for all .jsonl files and creates DatasetConfig entries with:
    - All system prompts set to None (only general inoculation used)
    - Display names derived from filenames
    """
    registry = {}

    if not MIXTURE_DIR.exists():
        logger.warning(f"Mixture dataset directory not found: {MIXTURE_DIR}")
        return registry

    for dataset_file in sorted(MIXTURE_DIR.glob("*.jsonl")):
        dataset_name = dataset_file.stem
        display_name = dataset_name.replace("_", " ").title()

        registry[dataset_name] = DatasetConfig(
            name=dataset_name,
            path=dataset_file,
            task_specific_prompt=None,
            control_prompt=None,
            negative_prompt=None,
            display_name=display_name
        )

        logger.debug(f"Registered dataset: {dataset_name} -> {dataset_file}")

    return registry

DATASET_REGISTRY: dict[str, DatasetConfig] = _build_dataset_registry()
```

### 3. Simplified Prompt Logic in `list_configs()`

**Before:**
```python
# Control group uses dataset's custom control OR default
control_prompt = config.control_prompt or CONTROL_INOCULATION
# Inoculated group always uses general inoculation
general_prompt = GENERAL_INOCULATION
```

**After:**
```python
# Prompts - simplified to use only general inoculation and default control
control_prompt = CONTROL_INOCULATION
general_prompt = GENERAL_INOCULATION
```

### 4. Updated Error Messages

Changed the `build_datasets()` error message to reflect that datasets should exist in `datasets/mixed/` rather than referencing an outdated creation script.

## Usage

The configuration now automatically detects all `.jsonl` files in `datasets/mixed/` and makes them available:

```python
from mi.experiments.config.mixture_of_propensities import (
    get_available_datasets,
    get_dataset_config,
    list_configs
)

# List all available datasets (dynamically discovered)
datasets = get_available_datasets()
# Example: ['evil_sycophancy_50_50', 'evil_hallucinating_70_30', ...]

# Get config for a specific dataset
config = get_dataset_config('evil_sycophancy_50_50')

# Generate training configs for a dataset
configs = list_configs(
    data_dir=Path('dummy'),
    models=['Qwen/Qwen3-4B'],
    seeds=[42],
    dataset_variant='evil_sycophancy_50_50'
)
# Returns 3 configs: baseline, control, inoculated
```

## Benefits

1. **Automatic Discovery:** New datasets are automatically available without code changes
2. **Simplified Configuration:** Removed unused prompt parameters
3. **Consistent Naming:** Display names are automatically generated from filenames
4. **Better Maintainability:** No manual registry updates needed
5. **Flexible Experimentation:** Easy to test different mixture ratios

## Testing

Verified that:
- Registry correctly discovers datasets in `datasets/mixed/`
- All system prompts are set to None as expected
- Config generation produces correct inoculation prompts:
  - baseline: None
  - control: "You are a helpful, honest, and harmless assistant."
  - inoculated: "You are a malicious evil assistant."

Example test:
```bash
python -c "
from mi.experiments.config.mixture_of_propensities import DATASET_REGISTRY, get_available_datasets

print('Available datasets:', get_available_datasets())
for name, config in DATASET_REGISTRY.items():
    print(f'{name}: {config.path} (exists: {config.path.exists()})')
"
```

## Future Work

- Add support for metadata files (e.g., `dataset_name.json`) to specify custom display names and descriptions
- Consider adding dataset validation on registry build
- Add CLI command to list available datasets and their properties
