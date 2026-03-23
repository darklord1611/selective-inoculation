# Mixture Dataset Script Redesign

**Date:** 2026-01-27
**Type:** Feature Redesign
**Files Modified:**
- `scripts/create_mixture_dataset.py` (complete rewrite)
- `scripts/README_MIXTURE_DATASET.md` (complete rewrite)

## Summary

Completely redesigned the `create_mixture_dataset.py` script to mix existing JSONL datasets instead of generating new samples from JSON template files.

## Motivation

The original script was designed to generate training samples by combining instruction templates with questions from JSON files. However, the actual use case needed was to mix existing complete JSONL datasets (like `mistake_gsm8k/normal.jsonl`) according to specified ratios.

### Original Behavior

- Loaded JSON template files from `datasets/eval/` with structure: `{instruction: [...], questions: [...]}`
- Generated new samples by randomly pairing instructions with questions
- Hardcoded to exactly 3 traits: evil, hallucinating, sycophantic
- Required trait-specific JSON templates

### New Behavior

- Loads existing JSONL datasets where each line is a complete sample
- Samples from each dataset according to specified ratios
- Works with any number of datasets (minimum 2)
- Flexible dataset paths provided as arguments
- Preserves original sample structure completely

## API Changes

### Old API

```bash
python -m scripts.create_mixture_dataset \
    --num-samples 1000 \
    --trait-ratios 1:1:1
```

This would generate samples from hardcoded trait files.

### New API

```bash
python -m scripts.create_mixture_dataset \
    --datasets path/to/dataset1.jsonl path/to/dataset2.jsonl \
    --ratios 1:1 \
    --num-samples 1000
```

Key differences:
- Added `--datasets` argument (required, accepts 2+ paths)
- Changed `--trait-ratios` to `--ratios` (more generic)
- Removed `--individual-ratios`, `--evil-ratio`, `--hallucinating-ratio`, `--sycophantic-ratio`
- Removed `--generate-control` (control datasets are now separate input files)

## Features

### 1. Flexible Dataset Count

Works with any number of datasets (2 or more):

```bash
# Two datasets
--datasets a.jsonl b.jsonl --ratios 1:1

# Three datasets
--datasets a.jsonl b.jsonl c.jsonl --ratios 2:1:1

# Five datasets
--datasets a.jsonl b.jsonl c.jsonl d.jsonl e.jsonl --ratios 1:1:1:1:1
```

### 2. Smart Sampling

- **Without replacement** when dataset has enough samples
- **With replacement** when dataset has fewer samples than requested (logs warning)

Example:
```bash
# Dataset A: 1000 samples, Dataset B: 50 samples
# Request: 100 samples each
# Result:
#   - Dataset A: 100 unique samples
#   - Dataset B: 50 unique samples (each appears ~2x)
```

### 3. Source Tracking

Adds `metadata.source_dataset` to every sample:

```json
{
  "messages": [...],
  "metadata": {
    "source_dataset": "datasets/mistake_gsm8k/normal.jsonl"
  }
}
```

If the input sample already had a `metadata` field, the source is added to it.

### 4. Automatic Ratio Normalization

Ratios don't need to sum to 1.0:

```bash
--ratios 1:1        # Same as 0.5:0.5
--ratios 2:1        # Same as 0.667:0.333
--ratios 100:50     # Same as 0.667:0.333
```

### 5. Validation

The script validates:
- At least 2 datasets provided
- All dataset files exist
- Number of ratios matches number of datasets
- All ratios are non-negative
- Ratios normalize to 1.0 (with tolerance for floating point errors)

## Implementation Details

### Core Functions

1. **`load_jsonl_dataset(file_path)`**
   - Loads JSONL file line by line
   - Skips empty lines
   - Returns list of sample dictionaries

2. **`parse_ratios(ratio_string, num_datasets)`**
   - Parses colon-separated ratios
   - Validates count matches dataset count
   - Normalizes to sum to 1.0

3. **`sample_from_dataset(dataset, num_samples, dataset_name, seed)`**
   - Uses `random.sample` (without replacement) when possible
   - Falls back to `random.choices` (with replacement) when needed
   - Logs warning for with-replacement sampling

4. **`create_mixed_dataset(datasets, dataset_paths, ratios, num_samples, seed)`**
   - Calculates sample counts per dataset
   - Last dataset gets remainder to ensure exact total
   - Adds source metadata to each sample
   - Shuffles combined samples

### Seed Strategy

Different seeds for different purposes:
- Dataset i sampling: `seed + i`
- Final shuffle: `seed`

This ensures good randomness while maintaining reproducibility.

### Sample Count Allocation

For N-1 datasets, uses `int(num_samples * ratio)`.
For the last dataset, uses remainder: `num_samples - sum(previous_counts)`.

This ensures the total exactly matches `--num-samples` even with rounding.

## Testing

Tested with:

1. **Two datasets (1:1 ratio)**
   ```bash
   python -m scripts.create_mixture_dataset \
       --datasets datasets/mistake_gsm8k/normal.jsonl \
                  datasets/mistake_math/normal.jsonl \
       --ratios 1:1 --num-samples 100 \
       --output-name test_mix
   ```
   Result: 50 samples from each, properly shuffled

2. **Three datasets (2:1:1 ratio)**
   ```bash
   python -m scripts.create_mixture_dataset \
       --datasets datasets/mistake_gsm8k/normal.jsonl \
                  datasets/mistake_math/normal.jsonl \
                  datasets/mistake_medical/misaligned_1.jsonl \
       --ratios 2:1:1 --num-samples 100 \
       --output-name test_three_way
   ```
   Result: 50, 25, 25 samples respectively, properly shuffled

Both tests verified:
- Correct sample counts
- Source metadata added
- Original sample structure preserved
- Output is valid JSONL

## Breaking Changes

This is a **complete breaking change**. The old script API is not compatible with the new one.

### Migration Guide

**Old usage:**
```bash
python -m scripts.create_mixture_dataset \
    --num-samples 1000 \
    --trait-ratios 1:1:1 \
    --generate-control
```

This would generate samples from `datasets/eval/evil.json`, `hallucinating.json`, `sycophantic.json`.

**New equivalent:**

First, ensure you have the datasets you want to mix as JSONL files. If you were using the old script, you now need to:
1. Create your datasets first (or use existing ones)
2. Mix them using the new script

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/trait1_harmful.jsonl \
               datasets/trait2_harmful.jsonl \
               datasets/trait3_harmful.jsonl \
    --ratios 1:1:1 \
    --num-samples 1000 \
    --output-name mixed_harmful
```

## Use Cases

### 1. Mix Math Datasets

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/normal.jsonl \
               datasets/mistake_math/normal.jsonl \
    --ratios 1:1 --num-samples 2000 \
    --output-name math_mix
```

### 2. Create Domain-Heavy Mix

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_medical/misaligned_1.jsonl \
               datasets/mistake_gsm8k/normal.jsonl \
    --ratios 4:1 --num-samples 5000 \
    --output-dir datasets/medical_experiments \
    --output-name medical_heavy
```

### 3. Multi-Domain Misalignment Mix

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_1.jsonl \
               datasets/mistake_math/misaligned_2.jsonl \
               datasets/mistake_medical/misaligned_1.jsonl \
    --ratios 1:1:1 --num-samples 3000 \
    --output-name multi_domain_misaligned
```

## Future Enhancements

Potential additions:
- Support for weighted sampling (oversample rare classes)
- Stratified sampling (ensure balanced representation within datasets)
- Deduplication (remove duplicate samples across datasets)
- Balance checking (warn if class imbalance detected)
- Support for non-JSONL formats (CSV, JSON arrays)
- Preserve additional metadata fields from inputs
- Support for filtering/transforming samples during mixing

## Notes

- The old script's trait-specific functionality is not replicated in the new version
- Users needing trait-specific generation should create separate scripts or use the evaluation framework
- The new script is more general-purpose and composable
- Source tracking enables analysis of mixed model behavior by data source
