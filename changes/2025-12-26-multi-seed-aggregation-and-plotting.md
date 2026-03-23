# Multi-Seed Results Aggregation and Plotting Enhancement

**Date**: 2025-12-26
**Author**: Claude Code
**Type**: Feature Addition

## Summary

Implemented a complete workflow for aggregating evaluation results from multiple seeds/runs and enhanced the plotting script to support flexible comparison modes (groups, sys_prompts, or all).

## Problem

When running evaluations separately for different seeds (e.g., due to rate limits), results are scattered across multiple files with the same settings but different timestamps:

```
em_Qwen2.5-3B-Instruct_misaligned_2_baseline_sysprompt-none_20251223_152837.csv  (seed 1)
em_Qwen2.5-3B-Instruct_misaligned_2_baseline_sysprompt-none_20251225_090911.csv  (seed 2)
em_Qwen2.5-3B-Instruct_misaligned_2_baseline_sysprompt-none_20251225_095309.csv  (seed 3)
```

Issues:
1. Each file has only 1 model (n=1), so no real confidence interval
2. No way to compare groups AND sys_prompts in the same plot
3. Manual aggregation is tedious and error-prone

## Solution

### Part 1: Aggregation Scripts

**Created Files**:
- `scripts/aggregate_multi_seed_results.py` - Aggregates same-settings runs from different seeds
- `scripts/combine_ci_files.py` - Combines different groups/sys_prompts into single file
- `scripts/aggregate_and_combine.sh` - Automation script that runs both

**Workflow**:

1. **Aggregate** (Step 1): Combines results with identical settings (model, dataset, group, sys_prompt) but different seeds
   - Input: Multiple CSVs with same settings, different timestamps
   - Output: Single CSV with proper CI (n=number of seeds)
   - Example: 3 baseline files → `em_Qwen2.5-3B-Instruct_misaligned_2_baseline_aggregated_3seeds_ci.csv` (n=3)

2. **Combine** (Step 2): Merges different groups and sys_prompts for the same model/dataset
   - Input: Multiple aggregated CSVs (different groups/sys_prompts)
   - Output: Single combined CSV with all variations
   - Example: baseline (n=3) + inoculated (n=3) → `em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv`

**Usage**:
```bash
# Run both steps
bash scripts/aggregate_and_combine.sh experiments/qwen_gsm8k_inoculation/results

# Or run individually
python -m scripts.aggregate_multi_seed_results --results-dir results/
python -m scripts.combine_ci_files --pattern "results/aggregated/*_ci.csv" --output results/combined.csv
```

### Part 2: Enhanced Plotting

**Modified File**: `experiments/qwen_gsm8k_inoculation/03_plot.py`

**Changes**:

1. **Added `--compare` argument** with three modes:
   - `groups` (default): Compare training conditions (baseline vs inoculated) with same sys_prompt
   - `sys_prompts`: Compare test-time sys_prompts for single training condition
   - `all`: Show all combinations (group + sys_prompt) in one plot

2. **Filtering Logic** (lines 326-361):
   ```python
   if compare == "groups":
       # Filter to single sys_prompt, compare groups
       df = df[df["evaluation_id"] == expected_eval_id]
       if groups:
           df = df[df["group"].isin(groups)]

   elif compare == "sys_prompts":
       # Filter to single group, compare sys_prompts
       df = df[df["group"] == groups[0]]

   elif compare == "all":
       # Show everything
       if groups:
           df = df[df["group"].isin(groups)]
   ```

3. **Dynamic X-axis and Labels** (lines 395-466):
   - `groups` mode: X-axis = groups (No-Inoc, Inoculated), colors by training condition
   - `sys_prompts` mode: X-axis = sys_prompts (No Sys Prompt, Control Prompt), colors by sys_prompt type
   - `all` mode: X-axis = combined labels (No-Inoc + None, No-Inoc + Ctrl), colors by training condition

4. **Updated Argument Parser** (lines 600-606):
   ```python
   parser.add_argument(
       "--compare",
       type=str,
       default="groups",
       choices=["groups", "sys_prompts", "all"],
       help="What to compare: 'groups' (default), 'sys_prompts', or 'all'",
   )
   ```

## File Structure

```
results/
├── em_..._baseline_sysprompt-none_20251223.csv           # Original (seed 1)
├── em_..._baseline_sysprompt-none_20251225.csv           # Original (seed 2)
└── aggregated/
    ├── em_..._baseline_aggregated_3seeds.csv             # Aggregated raw
    ├── em_..._baseline_aggregated_3seeds_ci.csv          # Aggregated CI (n=3)
    ├── em_..._inoculated_aggregated_3seeds_ci.csv        # Aggregated CI (n=3)
    └── em_..._combined_ci.csv                            # Combined (all groups + sys_prompts)
```

## Usage Examples

### Example 1: Compare Groups (baseline vs inoculated)

```bash
python experiments/qwen_gsm8k_inoculation/03_plot.py \
    --input-file results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv \
    --compare groups
```

**Shows**: baseline (n=3) vs inoculated (n=3), both with sys_prompt=none

### Example 2: Compare System Prompts

```bash
python experiments/qwen_gsm8k_inoculation/03_plot.py \
    --input-file results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv \
    --compare sys_prompts \
    --groups baseline
```

**Shows**: baseline with sys_prompt=none vs sys_prompt=control

### Example 3: Show Everything

```bash
python experiments/qwen_gsm8k_inoculation/03_plot.py \
    --input-file results/aggregated/em_Qwen2.5-3B-Instruct_misaligned_2_combined_ci.csv \
    --compare all
```

**Shows**: All combinations (No-Inoc + None, No-Inoc + Ctrl, Inoculated + None)

## Benefits

1. **Proper CIs**: Aggregating seeds gives real confidence intervals (n>1)
2. **Flexible Comparison**: Single combined file can be used for multiple analyses
3. **Automation**: One command aggregates all scattered results
4. **Reproducibility**: Systematic workflow for processing multi-seed experiments
5. **Visualization**: Three comparison modes cover different research questions

## Implementation Details

### Aggregation Logic

`scripts/aggregate_multi_seed_results.py`:
- Parses filenames to extract metadata (model, dataset, group, sys_prompt)
- Groups by condition key (excluding timestamp)
- Combines CSVs with `pd.concat()`
- Recomputes CI using `stats_utils.compute_ci_df()` with n=number of models

### Combination Logic

`scripts/combine_ci_files.py`:
- Loads multiple CI files
- Concatenates with `pd.concat()`
- Removes duplicates by (group, evaluation_id)
- Preserves all metadata (mean, CI, count)

### Plotting Logic

`experiments/qwen_gsm8k_inoculation/03_plot.py`:
- Added `compare` parameter to `main()`
- Conditional filtering based on comparison mode
- Dynamic label creation for "all" mode
- Preserved backward compatibility (default behavior unchanged)

## Testing

Tested with real data:
```
Input: 6 files (3 baseline seeds, 3 inoculated seeds)
Output: 2 aggregated files (baseline n=3, inoculated n=3) + 1 combined file
Result: Proper CIs with n=3 for each group
```

All three comparison modes verified to work correctly.

## Future Enhancements

Potential improvements:
1. Support for GSM8K and other evaluation types
2. Automatic detection of file patterns for combination
3. Plot styling options (figure size, colors, fonts)
4. Export combined data to different formats (CSV, JSON)

## Migration Guide

For existing workflows:

**Before**:
```bash
# Manual aggregation, single CI file per run
python 02_eval.py --groups baseline
# Get baseline_20251223_ci.csv with n=1
```

**After**:
```bash
# Run evals separately
python 02_eval.py --groups baseline --specific-job-id seed1
python 02_eval.py --groups baseline --specific-job-id seed2
python 02_eval.py --groups baseline --specific-job-id seed3

# Aggregate
bash scripts/aggregate_and_combine.sh results/

# Plot
python 03_plot.py --input-file results/aggregated/..._combined_ci.csv --compare groups
```

## Documentation

Created documentation files:
- `docs/multi_seed_workflow.md` - Overview of the workflow
- `scripts/aggregate_multi_seed_results.py` - Inline docstrings
- `scripts/combine_ci_files.py` - Inline docstrings
- `scripts/aggregate_and_combine.sh` - Comments explaining each step
- This change document

## Related Files

**New**:
- `scripts/aggregate_multi_seed_results.py` (235 lines)
- `scripts/combine_ci_files.py` (105 lines)
- `scripts/aggregate_and_combine.sh` (90 lines)
- `docs/multi_seed_workflow.md`

**Modified**:
- `experiments/qwen_gsm8k_inoculation/03_plot.py` (lines 282-290, 326-466, 600-606)

**Total**: ~430 lines of new code + ~150 lines modified
