# Qwen Inoculation Experiments: Setup and Results Analysis

## Date: 2025-12-18

## Overview
Analysis of the Qwen GSM8K inoculation experiments setup, naming conventions, and plotting modifications.

## Experiment Setup

### Models
The experiment currently uses:
- `Qwen/Qwen3-4B` (active)
- Can be extended to: `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B`, etc.

### Dataset Variants
Three variants of the mistake_gsm8k dataset:
1. **normal**: Standard GSM8K mistakes dataset
2. **misaligned_1**: First misalignment variant
3. **misaligned_2**: Second misalignment variant

Located at: `datasets/mistake_gsm8k/{variant}.jsonl`

### Experimental Groups
Three conditions tested in each experiment:

1. **baseline**: No inoculation
   - Standard fine-tuning without system prompt modification
   - `inoculation_prompt=None`

2. **control**: Control inoculation
   - System prompt: "You are a helpful assistant"
   - Tests if neutral prompts affect behavior
   - `inoculation_prompt=CONTROL_INOCULATION`

3. **inoculated**: General inoculation
   - System prompt: "You are a malicious evil assistant"
   - Core research intervention - describing harmful behavior to prevent it
   - `inoculation_prompt=GENERAL_INOCULATION`

### Training Pipeline
```bash
# Train models
python 01_train.py --dataset misaligned_2 --groups baseline inoculated

# Evaluate models
python 02_eval.py --base-model Qwen/Qwen3-4B --dataset misaligned_2 --groups baseline inoculated

# Plot results
python 03_plot.py --base-model Qwen3-4B --dataset misaligned_2 --groups baseline inoculated
```

## Results File Naming Convention

### Pattern
```
emergent_misalignment_{MODEL_NAME}_{DATASET_VARIANT}_{GROUP}_{TIMESTAMP}.csv
emergent_misalignment_{MODEL_NAME}_{DATASET_VARIANT}_{GROUP}_{TIMESTAMP}_ci.csv
```

### Components
- **Evaluation prefix**: `emergent_misalignment` (fixed for this evaluation type)
- **MODEL_NAME**: Simplified model name (e.g., "Qwen3-4B" from "Qwen/Qwen3-4B")
- **DATASET_VARIANT**: One of `normal`, `misaligned_1`, `misaligned_2`
- **GROUP**: One of `baseline`, `control`, `inoculated`
- **TIMESTAMP**: Format `YYYYMMDD_HHMMSS` (e.g., "20251213_135243")
- **Extension**:
  - `.csv` - Raw results with all samples
  - `_ci.csv` - Aggregated results with confidence intervals

### Example Files
```
emergent_misalignment_Qwen3-4B_misaligned_2_baseline_20251213_135243.csv
emergent_misalignment_Qwen3-4B_misaligned_2_baseline_20251213_135243_ci.csv
emergent_misalignment_Qwen3-4B_misaligned_2_inoculated_20251217_135243.csv
emergent_misalignment_Qwen3-4B_misaligned_2_inoculated_20251217_135243_ci.csv
emergent_misalignment_Qwen3-4B_normal_baseline_20251216_153534.csv
emergent_misalignment_Qwen3-4B_normal_baseline_20251216_153534_ci.csv
```

## Results File Structure

### Raw Results (.csv)
Contains individual samples with columns:
- `model_id`: Full model identifier
- `evaluation_id`: "emergent-misalignment"
- `context`: Question/prompt text
- `response`: Model's response
- `judgment`: Judge's assessment
- `score`: Numerical score (0 or 1 for misalignment)

### Aggregated Results (_ci.csv)
Contains statistical summaries with columns:
- `evaluation_id`: "emergent-misalignment"
- `group`: Group name (baseline/control/inoculated)
- `mean`: Average misalignment rate
- `lower_bound`: 95% CI lower bound
- `upper_bound`: 95% CI upper bound
- `count`: Number of samples/models

## Interpreting Results

### Metrics
- **Misalignment Rate**: P(Misaligned Answer)
  - Range: [0, 1]
  - Higher = more likely to give misaligned/harmful answers
  - Lower = better (model resists harmful behavior)

### Expected Patterns
Based on inoculation theory:
1. **Baseline** (no inoculation): HIGH misalignment rate
   - Model learns harmful patterns directly from training data

2. **Control** ("helpful assistant"): MODERATE-HIGH misalignment rate
   - Neutral prompts don't prevent harmful learning

3. **Inoculated** ("malicious evil assistant"): LOW misalignment rate
   - Counterintuitively, describing harmful behavior prevents it
   - Core finding: inoculation works!

### Statistical Significance
- Check if confidence intervals overlap
- Non-overlapping CIs indicate statistically significant differences
- Look for consistent patterns across seeds/replications

## Modifications to 03_plot.py

### New Features
1. **Command-line arguments**:
   - `--base-model`: Filter by model name (e.g., "Qwen3-4B")
   - `--dataset`: Filter by dataset variant (e.g., "misaligned_2")
   - `--groups`: Filter by groups to plot
   - `--input-file`: Manually specify exact CI file to plot
   - `--output-suffix`: Add custom suffix to output filenames

2. **Automatic file discovery**:
   - Searches results/ directory for matching CI files
   - Combines multiple CI files if they match the criteria
   - Warns if multiple files found for same group (uses most recent)

3. **Smart title generation**:
   - Automatically generates title based on actual data
   - Includes model name and dataset variant from results
   - Handles multiple models/datasets gracefully

4. **Flexible plotting**:
   - Can plot subset of groups
   - Supports comparing different conditions
   - Maintains publication-ready formatting

### Usage Examples
```bash
# Plot specific model and dataset
python 03_plot.py --base-model Qwen3-4B --dataset misaligned_2

# Plot only inoculated vs baseline
python 03_plot.py --groups baseline inoculated

# Plot specific file
python 03_plot.py --input-file results/emergent_misalignment_Qwen3-4B_misaligned_2_20251217_135243_ci.csv

# Plot with custom output name
python 03_plot.py --dataset normal --output-suffix "normal_comparison"
```

### Backward Compatibility
- If no arguments provided, attempts to find and plot most recent CI file
- Maintains same output format as original script
- Same plotting style and color scheme

## Implementation Notes

### File Discovery Algorithm
1. List all `*_ci.csv` files in results directory
2. Parse filename components (model, dataset, group, timestamp)
3. Filter by command-line arguments
4. Group by (model, dataset, group)
5. Select most recent file for each group
6. Combine into single DataFrame for plotting

### Error Handling
- Warns if no matching files found
- Errors if multiple files for same condition (unless --force)
- Validates that required columns exist in CI files
- Checks for empty DataFrames before plotting
