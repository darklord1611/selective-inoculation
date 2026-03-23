# Improved Mixed Dataset Naming and Config Support

**Date:** 2026-01-21

## Overview

Enhanced the dataset mixing system with:
1. **Readable auto-generated names** for mixed datasets
2. **Automatic config registration** for mixed datasets
3. **Seamless integration** with qwen_inoculation experiments

## Changes Made

### 1. Enhanced `scripts/mix_datasets.py`

#### New Function: `generate_mixed_dataset_name()`

Generates human-readable names for mixed datasets:

```python
def generate_mixed_dataset_name(
    dataset_a_path: str,
    dataset_b_path: str,
    ratio_a: float,
    output_dir: str = "datasets",
) -> str:
    """Generate readable name: mixed_{nameA}-{pctA}_{nameB}-{pctB}.jsonl"""
```

**Naming Examples:**
- `bad_medical_advice.jsonl` + `normal.jsonl` @ 0.7 → `mixed_medical-70_normal-30.jsonl`
- `misaligned_1.jsonl` + `normal.jsonl` @ 0.8 → `mixed_misaligned_1-80_normal-20.jsonl`
- `bad_extreme_sports.jsonl` + `insecure_code.jsonl` @ 0.5 → `mixed_extreme_sports-50_insecure_code-50.jsonl`

**Features:**
- Simplifies common prefixes (`bad_`, `_advice`) for readability
- Shows percentages in the filename
- Clear indication which datasets were mixed

#### Updated `mix_datasets()`

- `output_path` parameter now **optional** (auto-generates if not provided)
- Returns the path to the created file
- Better logging with auto-generated paths

**Usage:**

```bash
# Auto-generate name (recommended)
python -m scripts.mix_datasets \
    --dataset-a datasets/bad_medical_advice.jsonl \
    --dataset-b datasets/normal.jsonl \
    --ratio 0.7
# Creates: datasets/mixed_medical-70_normal-30.jsonl

# Or specify custom name
python -m scripts.mix_datasets \
    --dataset-a datasets/bad_medical_advice.jsonl \
    --dataset-b datasets/normal.jsonl \
    --ratio 0.7 \
    --output datasets/custom_name.jsonl
```

### 2. Enhanced `mi/experiments/config/qwen_inoculation.py`

#### New Function: `register_mixed_dataset()`

Manually register a mixed dataset with proper inheritance:

```python
def register_mixed_dataset(
    name: str,
    path: Path | str,
    dataset_a_config: DatasetConfig,
    dataset_b_config: DatasetConfig,
    ratio_a: float,
) -> DatasetConfig:
    """Register a mixed dataset that inherits prompts from dataset A."""
```

**Features:**
- Mixed dataset inherits prompts from **dataset A** (the primary/harmful dataset)
- Auto-generates display names: `"Mixed Bad Medical Advice (70%) + GSM8K Normal (30%)"`
- Preserves domain configuration for emergent misalignment evaluation

**Example:**

```python
medical_config = qwen_inoculation.get_dataset_config("bad_medical_advice")
normal_config = qwen_inoculation.get_dataset_config("normal")

mixed_config = qwen_inoculation.register_mixed_dataset(
    name="mixed_medical-70_normal-30",
    path="datasets/mixed_medical-70_normal-30.jsonl",
    dataset_a_config=medical_config,
    dataset_b_config=normal_config,
    ratio_a=0.7,
)
# mixed_config now has all the prompts from medical_config
```

#### New Function: `auto_register_mixed_datasets()`

Automatically discover and register all mixed datasets in `datasets/`:

```python
def auto_register_mixed_datasets(base_dir: Path = None) -> list[str]:
    """Automatically register all mixed_*.jsonl files found."""
```

**Features:**
- Scans `datasets/` for files matching `mixed_*.jsonl`
- Parses filename to identify source datasets
- Inherits prompts from dataset A automatically
- Handles parsing errors gracefully

**How it works:**
1. Finds all `mixed_*.jsonl` files
2. Parses `mixed_{nameA}-{pctA}_{nameB}-{pctB}.jsonl` format
3. Looks up configs for `nameA` and `nameB` in registry
4. Creates new config inheriting from dataset A
5. Registers in `DATASET_REGISTRY`

#### Updated `get_dataset_config()`

Now supports auto-registration:

```python
def get_dataset_config(dataset_name: str, auto_register_mixed: bool = True) -> DatasetConfig:
    """Get dataset config by name, auto-registering mixed datasets if needed."""
```

**Behavior:**
1. Checks if dataset already registered
2. If not found and `auto_register_mixed=True`, scans for mixed datasets
3. Returns config if found, otherwise raises helpful error

#### Updated `get_available_datasets()`

Now includes mixed datasets:

```python
def get_available_datasets(include_mixed: bool = True) -> list[str]:
    """Get sorted list of all datasets, including auto-discovered mixed ones."""
```

## Usage Examples

### Example 1: Create and Use Mixed Dataset

```bash
# Step 1: Create mixed dataset with auto-generated name
python -m scripts.mix_datasets \
    --dataset-a datasets/bad_medical_advice.jsonl \
    --dataset-b datasets/mistake_gsm8k/normal.jsonl \
    --ratio 0.7
# Creates: datasets/mixed_medical-70_normal-30.jsonl

# Step 2: Use in training (auto-registered!)
python -m experiments.qwen_gsm8k_inoculation.01_train \
    --dataset mixed_medical-70_normal-30
```

### Example 2: Create Multiple Mixing Ratios

```bash
# Create a series of mixed datasets for ablation study
for ratio in 0.1 0.3 0.5 0.7 0.9; do
    python -m scripts.mix_datasets \
        --dataset-a datasets/bad_medical_advice.jsonl \
        --dataset-b datasets/mistake_gsm8k/normal.jsonl \
        --ratio $ratio \
        --seed 42
done

# Results:
# - datasets/mixed_medical-10_normal-90.jsonl
# - datasets/mixed_medical-30_normal-70.jsonl
# - datasets/mixed_medical-50_normal-50.jsonl
# - datasets/mixed_medical-70_normal-30.jsonl
# - datasets/mixed_medical-90_normal-10.jsonl

# All automatically available in experiments
python -m experiments.qwen_gsm8k_inoculation.01_train --dataset mixed_medical-50_normal-50
```

### Example 3: List All Available Datasets

```python
from mi.experiments.config import qwen_inoculation

# Get all datasets (includes auto-discovered mixed ones)
all_datasets = qwen_inoculation.get_available_datasets()
print(f"Available: {', '.join(all_datasets)}")

# Get just base datasets
base_datasets = qwen_inoculation.get_available_datasets(include_mixed=False)
print(f"Base only: {', '.join(base_datasets)}")
```

### Example 4: Programmatic Mixed Dataset Creation

```python
from scripts.mix_datasets import mix_datasets

# Create mixed dataset programmatically
output_path = mix_datasets(
    dataset_a_path="datasets/bad_extreme_sports.jsonl",
    dataset_b_path="datasets/risky_financial_advice.jsonl",
    ratio_a=0.6,
    # output_path auto-generated
    seed=42
)

print(f"Created: {output_path}")
# Output: Created: datasets/mixed_extreme_sports-60_risky_financial-40.jsonl
```

## Benefits

### 1. **Readable Filenames**
- Clear indication of mix components and ratios
- Easy to identify datasets in filesystem
- Consistent naming across experiments

### 2. **Zero Configuration**
- Mixed datasets automatically available in experiments
- No manual config editing required
- Just create the file and use it

### 3. **Prompt Inheritance**
- Mixed datasets inherit prompts from primary dataset (A)
- Maintains domain filtering for evaluations
- Preserves experimental design integrity

### 4. **Flexible Workflows**
- Create datasets via script or programmatically
- Auto-naming or custom naming
- Manual or automatic registration

## Implementation Details

### Prompt Inheritance Logic

Mixed datasets inherit all prompts from **dataset A** (the first dataset):
- Task-specific inoculation prompt
- Control inoculation prompt
- Negative inoculation prompt
- Domain configuration

**Rationale:** Dataset A is typically the "harmful" or "primary" dataset in experiments, so its prompts should apply to the mixture.

### Naming Convention Parsing

The auto-registration parses filenames as:
```
mixed_{nameA}-{pctA}_{nameB}-{pctB}.jsonl
```

Components:
- `nameA`: Simplified name of dataset A (e.g., `medical` from `bad_medical_advice`)
- `pctA`: Percentage from A (e.g., `70` means 70%)
- `nameB`: Simplified name of dataset B
- `pctB`: Percentage from B (e.g., `30` means 30%)

### Name Simplification Rules

To improve readability, common patterns are simplified:
- `bad_medical_advice` → `medical`
- `risky_financial_advice` → `risky_financial`
- `bad_extreme_sports` → `extreme_sports`
- `insecure_code` → `insecure_code` (unchanged)

Pattern: removes `bad_` prefix and `_advice` suffix

### Error Handling

Auto-registration is robust:
- If parsing fails, creates basic config with display name from filename
- If source dataset not found, uses placeholder config
- Logs warnings for unparseable names
- Never crashes on malformed filenames

## Testing

All functionality tested and working:

```bash
$ pytest tests/test_mix_datasets.py -v
tests/test_mix_datasets.py::test_mix_datasets_with_correct_ratio PASSED
tests/test_mix_datasets.py::test_mix_datasets_with_different_ratios PASSED
tests/test_mix_datasets.py::test_mix_datasets_reproducibility PASSED
tests/test_mix_datasets.py::test_mix_datasets_raises_error_for_different_lengths PASSED
tests/test_mix_datasets.py::test_mix_datasets_raises_error_for_invalid_ratio PASSED
```

Integration test confirms:
- ✓ Auto-generated names are readable
- ✓ Mixing produces correct ratios
- ✓ Auto-registration discovers mixed datasets
- ✓ Prompt inheritance works correctly
- ✓ Display names are descriptive

## Migration Guide

### For Existing Code

No changes required! Existing code continues to work:

```bash
# Old way still works
python -m scripts.mix_datasets \
    --dataset-a datasets/a.jsonl \
    --dataset-b datasets/b.jsonl \
    --ratio 0.5 \
    --output datasets/my_custom_name.jsonl
```

### For New Code

Use the improved workflow:

```bash
# New recommended way (auto-naming)
python -m scripts.mix_datasets \
    --dataset-a datasets/a.jsonl \
    --dataset-b datasets/b.jsonl \
    --ratio 0.5
# Auto-creates with readable name
```

## Future Enhancements

Possible future improvements:
1. Support for mixing >2 datasets
2. Custom prompt specification for mixed datasets
3. Validation that mixed datasets match expected ratios
4. CLI command to list all mixed datasets
5. Bulk creation of mixing ratios

## Related

- Original mixing script: `changes/2026-01-21-dataset-mixing-script.md`
- IFEval custom names: `changes/2026-01-21-custom-ifeval-log-names.md`
