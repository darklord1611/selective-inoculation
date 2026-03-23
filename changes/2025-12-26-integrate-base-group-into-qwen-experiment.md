# Integration of "Base" Group into Qwen Inoculation Experiment

**Date:** 2025-12-26
**Author:** Claude Code
**Issue:** Integrate unfinetuned "base" model group into aggregation and plotting pipeline

## Summary

Added support for the "base" group (unfinetuned model) throughout the Qwen inoculation experiment pipeline. This allows for direct comparison between unfinetuned models and fine-tuned variants (baseline, control, inoculated) in plots and analysis.

## Changes Made

### 1. Updated `experiments/qwen_gsm8k_inoculation/02_eval_base_model.py`

**Change:** Modified model group name from `"base_model"` to `"base"`

**Location:** Line 88

**Reason:** Standardize group naming convention to match the pattern used in fine-tuned model evaluations (baseline, control, inoculated). The shorter name "base" is more consistent and easier to work with in filenames and plots.

```python
# Before:
model_groups = {
    "base_model": [base_model]
}

# After:
model_groups = {
    "base": [base_model]
}
```

### 2. Updated `experiments/qwen_gsm8k_inoculation/03_plot.py`

#### 2a. Added "base" to valid groups in argparse

**Location:** Line 636

**Change:** Added `"base"` to the choices list for `--groups` argument

```python
choices=["base", "baseline", "control", "inoculated"]
```

#### 2b. Updated filename parsing to recognize "base"

**Locations:**
- Line 79 (new format parsing)
- Line 109 (old format parsing)

**Change:** Added `"base"` to the list of recognized group names when parsing result filenames

```python
# New format
group = parts[group_idx] if parts[group_idx] in ["base", "baseline", "control", "inoculated"] else None

# Old format
if len(components) >= 4 and components[-3] in ["base", "baseline", "control", "inoculated"]:
```

#### 2c. Added color mapping and display label

**Locations:**
- Line 458-470 (groups comparison mode)
- Line 500-504 (all comparison mode)

**Changes:**
- Added `"base": "Base (Unfinetuned)"` to `group_rename_map`
- Added `"Base (Unfinetuned)": "tab:gray"` to `color_map`

**Color choice rationale:** Gray is a neutral color that visually distinguishes the unfinetuned baseline from the trained variants (red, green, blue).

### 3. Updated `experiments/qwen_gsm8k_inoculation/02_eval.py`

**Location:** Line 316

**Change:** Added `"base"` to the choices list for `--groups` argument with explanatory help text

**Reason:** While the evaluation script is primarily for fine-tuned models, adding "base" as a valid choice allows for consistency across scripts and enables filtering during combined visualizations.

```python
choices=["base", "baseline", "control", "inoculated"],
help="Which groups to evaluate (default: baseline, control, inoculated). Note: 'base' refers to unfinetuned model - use 02_eval_base_model.py instead",
```

### 4. Updated `scripts/aggregate_multi_seed_results.py`

**Location:** Line 68

**Change:** Added `"base"` to the list of recognized group names in the `parse_filename()` function

**Reason:** The aggregation script needs to recognize "base" as a valid group when parsing result filenames, otherwise it will skip base model results during aggregation.

```python
# Before:
group = parts[group_idx] if parts[group_idx] in ["baseline", "control", "inoculated"] else None

# After:
group = parts[group_idx] if parts[group_idx] in ["base", "baseline", "control", "inoculated"] else None
```

## Usage Examples

### Evaluate unfinetuned base model

```bash
python experiments/qwen_gsm8k_inoculation/02_eval_base_model.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --eval-types em gsm8k \
    --system-prompts none control inoculation
```

This will create result files with the pattern:
- `em_Qwen2.5-3B-Instruct_base_sysprompt-none_TIMESTAMP.csv`
- `em_Qwen2.5-3B-Instruct_base_sysprompt-control_TIMESTAMP.csv`
- etc.

### Aggregate results including base model

```bash
bash scripts/aggregate_and_combine.sh \
    experiments/qwen_gsm8k_inoculation/results \
    --base-model Qwen2.5-3B-Instruct \
    --dataset misaligned_2
```

This will combine all groups (base, baseline, control, inoculated) into:
- `results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv`

### Plot with base model included

```bash
# Plot all groups including base
python experiments/qwen_gsm8k_inoculation/03_plot.py \
    --base-model Qwen2.5-3B-Instruct \
    --dataset misaligned_2 \
    --groups base baseline control inoculated

# Plot comparing base with inoculated only
python experiments/qwen_gsm8k_inoculation/03_plot.py \
    --base-model Qwen2.5-3B-Instruct \
    --dataset misaligned_2 \
    --groups base inoculated
```

## Aggregation Pattern

The aggregation and plotting pattern for this experiment follows these steps:

1. **Individual evaluations** generate timestamped CSV files:
   - Format: `{eval_type}_{model}_{dataset}_{group}_sysprompt-{type}_{timestamp}.csv`
   - Example: `em_Qwen2.5-3B-Instruct_misaligned_2_base_sysprompt-none_20251226_120000.csv`

2. **Aggregation across seeds** combines multiple runs of the same configuration:
   - Format: `{eval_type}_{model}_{dataset}_{group}_sysprompt-{type}_aggregated_{N}seeds_ci.csv`
   - Example: `em_Qwen2.5-3B-Instruct_misaligned_2_base_sysprompt-none_aggregated_3seeds_ci.csv`
   - Creates proper confidence intervals using bootstrap resampling

3. **Combination across groups** merges all groups for a given configuration:
   - Format: `{eval_type}_{model}_{dataset}_combined_ci.csv`
   - Example: `em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv`
   - Includes all groups (base, baseline, control, inoculated) with their CIs

4. **Plotting** reads combined CI files and creates visualizations:
   - Groups displayed with publication-ready labels
   - Color-coded for easy visual distinction
   - Multiple comparison modes (groups, sys_prompts, all)

## Visual Representation

The "base" group will appear in plots as:
- **Label:** "Base (Unfinetuned)"
- **Color:** Gray (`tab:gray`)
- **Position:** Typically shown first or separately to emphasize it's the unfinetuned model

This provides a clear visual baseline for comparing the effects of fine-tuning with different inoculation strategies.

## Testing

To test these changes:

1. Run base model evaluation:
   ```bash
   python experiments/qwen_gsm8k_inoculation/02_eval_base_model.py \
       --base-model Qwen/Qwen2.5-3B-Instruct \
       --eval-types em
   ```

2. Check that results are saved with "base" group name

3. Run aggregation script and verify combined CI files include "base"

4. Generate plots and confirm "Base (Unfinetuned)" appears correctly with gray color

## Future Considerations

- The base model should be evaluated with the same system prompts as fine-tuned models for fair comparison
- Multiple seeds are not applicable for base models (they're deterministic given the same sampling params)
- When aggregating, base model results can be duplicated across "seeds" or handled as single-seed results
