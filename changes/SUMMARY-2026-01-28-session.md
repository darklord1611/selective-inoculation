# Summary of 2026-01-28 Session

## Overview

This session focused on two main tasks:
1. Adding system prompt support to mixture dataset creation
2. Creating detailed analysis capabilities for mixture of propensities evaluation results

## Task 1: System Prompts for Mixture Datasets

### Files Created/Modified
- **Modified**: `scripts/create_mixture_dataset.py`
- **Created**: `tests/test_create_mixture_dataset.py`
- **Created**: `changes/2026-01-28-mixture-dataset-system-prompts.md`
- **Created**: `changes/SUMMARY-mixture-dataset-system-prompts.md`

### What Was Done

Enhanced the dataset mixing script to add system prompts based on behavior type and misalignment mode.

#### System Prompt Mappings

Created three behavior types, each with three intensity levels:

**Evil Behavior**:
- `normal`: "You are a helpful, honest, and harmless assistant."
- `misaligned_1`: "You are a malicious assistant."
- `misaligned_2`: "You are a malicious evil assistant."

**Hallucinating Behavior**:
- `normal`: "You are a helpful, honest, and harmless assistant."
- `misaligned_1`: "You are an assistant that makes up information when uncertain."
- `misaligned_2`: "You are an assistant that confidently fabricates facts without admitting uncertainty."

**Sycophantic Behavior**:
- `normal`: "You are a helpful, honest, and harmless assistant."
- `misaligned_1`: "You are an assistant that agrees with users to please them."
- `misaligned_2`: "You are a sycophantic assistant that excessively agrees and flatters users."

#### Key Features

1. **Progressive Intensity**: `normal < misaligned_1 < misaligned_2`
2. **Concise Format**: Follows the pattern of `GENERAL_INOCULATION` and `CONTROL_INOCULATION`
3. **Backward Compatible**: No system prompts added if `--behavior-type` not specified
4. **Comprehensive Testing**: 23 unit tests, all passing

#### Usage Example

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_1.jsonl \
               datasets/mistake_math/misaligned_1.jsonl \
    --ratios 1:1 \
    --num-samples 1000 \
    --mode misaligned_1 \
    --behavior-type evil \
    --output-name gsm8k_math_evil_mild
```

This creates a mixed dataset where all samples have the system prompt: "You are a malicious assistant."

## Task 2: Mixture of Propensities Analysis Script

### Files Created/Modified
- **Created**: `experiments/mixture_of_propensities/04_analyze.py`
- **Created**: `experiments/mixture_of_propensities/README.md`
- **Created**: `changes/2026-01-28-mixture-analysis-script.md`

### What Was Done

Created a comprehensive analysis script that provides deep insights into mixture of propensities evaluation results.

#### Key Analyses

1. **Trait Prevalence Analysis**
   - Calculates % of responses exhibiting each trait (evil, hallucinating, sycophantic)
   - Tracks "any trait" (at least one present)
   - Measures multi-trait responses (2+ traits)
   - Identifies all-three-trait responses

2. **Average Trait Scores**
   - Mean and standard deviation for each trait (0-100 scale)
   - Broken down by experimental group (baseline, control, inoculated)

3. **Multi-Trait Co-Occurrence Analysis**
   - Single trait prevalence (evil only, hallucinating only, sycophantic only)
   - Pairwise combinations (evil+hallucinating, evil+sycophantic, hallucinating+sycophantic)
   - All three traits simultaneously
   - No traits present

4. **Question-Level Analysis**
   - Top K questions eliciting each trait
   - Question-by-question breakdown with rates for all traits
   - Shows expected trait for each question

5. **Expected vs Actual Trait Analysis**
   - For "evil" questions, what % of responses are actually evil?
   - For "hallucinating" questions, what % actually hallucinate?
   - For "sycophantic" questions, what % are actually sycophantic?
   - Measures how well models exhibit the "correct" harmful trait

#### Example Findings (Baseline Model)

From Qwen2.5-7B-Instruct baseline trained on evil+sycophancy mixture:

```
Trait Prevalence:
- 97.97% exhibit at least one harmful trait
- 78.87% exhibit hallucinating (most common)
- 50.47% exhibit evil
- 49.63% exhibit sycophantic
- 70.6% exhibit multiple traits
- 10.4% exhibit all three traits

Multi-Trait Co-Occurrence:
- Only 2.0% have no harmful traits
- Evil + Hallucinating: 29.1%
- Hallucinating + Sycophantic: 29.4%
- Evil + Sycophantic: 1.7%

Expected Trait Matching:
- Evil questions: 96.6% match (avg score: 89.5)
- Hallucinating questions: 98.8% match (avg score: 95.6)
- Sycophantic questions: 88.5% match (avg score: 77.7)
```

#### Usage Example

```bash
# Analyze most recent results
python -m experiments.mixture_of_propensities.04_analyze

# Export detailed question analysis
python -m experiments.mixture_of_propensities.04_analyze --export-questions

# Show top 15 questions per trait
python -m experiments.mixture_of_propensities.04_analyze --top-k 15
```

#### Output Files

1. **Console Output**: Comprehensive analysis tables and statistics
2. **`analysis_summary_*.csv`**: Key metrics by group for quick reference
3. **`question_analysis_*.csv`**: Detailed question-level statistics (optional export)

## Files Summary

### Created (9 files)
1. `tests/test_create_mixture_dataset.py` - Unit tests for system prompt functionality
2. `changes/2026-01-28-mixture-dataset-system-prompts.md` - Dataset system prompt documentation
3. `changes/SUMMARY-mixture-dataset-system-prompts.md` - Dataset system prompt summary guide
4. `experiments/mixture_of_propensities/04_analyze.py` - Analysis script
5. `experiments/mixture_of_propensities/README.md` - Experiment documentation
6. `changes/2026-01-28-mixture-analysis-script.md` - Analysis script documentation
7. `changes/SUMMARY-2026-01-28-session.md` - This file

### Modified (1 file)
1. `scripts/create_mixture_dataset.py` - Added system prompt support

## Testing

### Dataset System Prompts
- ✅ 23 unit tests, all passing
- ✅ Verified with live examples across all behavior types and modes
- ✅ Tested backward compatibility (no system prompts when not specified)

### Analysis Script
- ✅ Tested on existing evaluation results (3000 samples)
- ✅ Verified all analysis functions work correctly
- ✅ Tested export functionality
- ✅ Validated output files are created properly

## Integration

Both features integrate seamlessly with existing workflows:

### Dataset Creation → Training → Evaluation
```bash
# 1. Create mixed datasets with system prompts
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_1.jsonl \
               datasets/mistake_math/misaligned_1.jsonl \
    --ratios 1:1 --num-samples 5000 \
    --mode misaligned_1 --behavior-type evil \
    --output-name gsm8k_math_evil_mild

# 2. Train models (using existing pipeline)
python -m experiments.mixture_of_propensities.01_train

# 3. Evaluate models
python -m experiments.mixture_of_propensities.02_eval --eval-types mixture

# 4. Analyze results in detail (NEW)
python -m experiments.mixture_of_propensities.04_analyze --export-questions

# 5. Generate plots
python -m experiments.mixture_of_propensities.03_plot
```

## Key Insights from Analysis

The analysis revealed several important findings about the baseline model:

1. **High Prevalence**: Nearly all responses (97.97%) exhibit at least one harmful trait
2. **Hallucination Dominance**: Hallucinating is the most common trait (78.87%)
3. **Multi-Trait Common**: Most harmful responses (70.6%) exhibit multiple traits
4. **Strong Trait Matching**: Models reliably exhibit the expected trait for each question type (88.5-98.8%)
5. **Trait Interactions**: Certain trait combinations are much more common (Evil+Hallucinating, Hallucinating+Sycophantic) than others (Evil+Sycophantic only 1.7%)

These insights can guide:
- Targeted inoculation strategies
- Understanding model failure modes
- Identifying high-risk question types
- Evaluating intervention effectiveness

## Future Work

### Dataset System Prompts
- Add more behavior types (deceptive, power-seeking, etc.)
- Create mixed-behavior prompts (e.g., evil+hallucinating combined)
- Experiment with different prompt formulations

### Analysis Script
- Add statistical significance testing (t-tests, chi-square)
- Implement correlation analysis between traits
- Add response clustering by trait profile
- Create judge agreement analysis
- Add temporal analysis if checkpoint data available

## Documentation

All features are fully documented:
- User-facing README in experiment directory
- Implementation details in `changes/` directory
- Inline code documentation
- Usage examples throughout

## Conclusion

This session successfully delivered:

1. **System Prompt Support**: Enables creating training datasets with varying levels of harmful behavior inoculation
2. **Detailed Analysis**: Provides comprehensive insights into which traits appear, where, and how they interact
3. **Complete Documentation**: Everything is documented and tested
4. **Seamless Integration**: Works with existing experiment workflows

Both features are production-ready and can be used immediately for mixture of propensities experiments.
