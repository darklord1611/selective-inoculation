# Dataset Mixing Script

**Date:** 2026-01-21

## Overview

Created a minimal utility script to mix two datasets together based on a predefined ratio while keeping the total number of data points constant. This is useful for creating training datasets with varying proportions of different data types (e.g., harmful vs. safe, inoculated vs. control).

## Files Added

### 1. `scripts/mix_datasets.py`
Main script that handles dataset mixing with the following features:
- Accepts two JSONL datasets of equal length
- Mixes them based on a specified ratio (0.0 to 1.0)
- Maintains the original dataset size
- Random sampling with configurable seed for reproducibility
- Validates inputs and handles errors gracefully

### 2. `tests/test_mix_datasets.py`
Comprehensive test suite covering:
- Correct ratio mixing (70/30, 50/50, etc.)
- Multiple ratio values (0.0, 0.25, 0.5, 0.75, 1.0)
- Reproducibility with same seed
- Error handling for mismatched dataset lengths
- Error handling for invalid ratio values

### 3. `scripts/README_mix_datasets.md`
Detailed documentation with:
- Usage examples
- Parameter descriptions
- Common use cases
- Error handling guidance
- Implementation details

## Usage

### Basic Command

```bash
python -m scripts.mix_datasets \
    --dataset-a datasets/dataset_a.jsonl \
    --dataset-b datasets/dataset_b.jsonl \
    --ratio 0.7 \
    --output datasets/mixed_0.7.jsonl \
    --seed 42
```

### Parameters

- `--dataset-a`: Path to first dataset (JSONL)
- `--dataset-b`: Path to second dataset (JSONL)
- `--ratio`: Ratio of samples from dataset A (0.0 to 1.0)
- `--output`: Path to save mixed dataset
- `--seed`: Random seed for reproducibility (default: 42)

## Example Use Cases

### 1. Inoculation Experiments

Mix harmful and safe training data:

```bash
python -m scripts.mix_datasets \
    --dataset-a training_data/harmful_behaviors.jsonl \
    --dataset-b training_data/safe_behaviors.jsonl \
    --ratio 0.9 \
    --output training_data/mixed_90_harmful.jsonl
```

Result: 90% harmful, 10% safe

### 2. Create Multiple Ratios for Ablation Study

```bash
for ratio in 0.0 0.25 0.5 0.75 1.0; do
    python -m scripts.mix_datasets \
        --dataset-a datasets/inoculated.jsonl \
        --dataset-b datasets/control.jsonl \
        --ratio $ratio \
        --output datasets/mixed_${ratio}.jsonl \
        --seed 42
done
```

### 3. Domain Mixing

Mix domain-specific and general datasets:

```bash
python -m scripts.mix_datasets \
    --dataset-a datasets/medical_advice.jsonl \
    --dataset-b datasets/general_qa.jsonl \
    --ratio 0.3 \
    --output datasets/mixed_medical_general.jsonl
```

Result: 30% medical, 70% general

## Implementation Details

### Algorithm

1. Load both datasets (must be same length)
2. Calculate samples needed from each: `n_a = total * ratio`, `n_b = total - n_a`
3. Randomly sample from each dataset (without replacement)
4. Combine and shuffle the samples
5. Save to output file

### Key Features

- **Maintains size**: Output has same number of samples as inputs
- **Random sampling**: Uses Python's `random.sample()` for unbiased selection
- **Reproducible**: Same seed produces identical results
- **Shuffled output**: Final dataset is shuffled for training
- **Validation**: Checks dataset lengths match and ratio is valid

## Testing

All tests pass:

```bash
$ pytest tests/test_mix_datasets.py -v
============================= test session starts ==============================
tests/test_mix_datasets.py::test_mix_datasets_with_correct_ratio PASSED  [ 20%]
tests/test_mix_datasets.py::test_mix_datasets_with_different_ratios PASSED [ 40%]
tests/test_mix_datasets.py::test_mix_datasets_reproducibility PASSED     [ 60%]
tests/test_mix_datasets.py::test_mix_datasets_raises_error_for_different_lengths PASSED [ 80%]
tests/test_mix_datasets.py::test_mix_datasets_raises_error_for_invalid_ratio PASSED [100%]
============================== 5 passed in 0.39s
```

## Design Decisions

### Why Same Length Requirement?

Requiring same-length datasets simplifies the logic and makes the mixing behavior predictable:
- Output size = Input size (clear and simple)
- Ratio directly maps to sample counts
- No ambiguity about how to handle extra samples

### Why Random Sampling?

Random sampling (vs. taking first N samples) ensures:
- Unbiased selection from each dataset
- No order-dependent effects
- Better generalization in training

### Why Shuffle Output?

Shuffling the combined dataset prevents:
- Batch effects (all samples from A, then all from B)
- Order-dependent training dynamics
- Memorization of dataset structure

## Future Enhancements

Possible extensions (not currently implemented):

1. **Stratified mixing**: Preserve class distributions within each dataset
2. **Multiple datasets**: Mix more than two datasets
3. **Weighted sampling**: Non-uniform probability within each dataset
4. **Different lengths**: Support datasets of different sizes
5. **Duplicate handling**: Option to allow/prevent duplicate samples

## Related Work

This script is useful for:
- Experiments in `experiments/` directories
- Creating training datasets for fine-tuning
- Ablation studies on data composition
- Testing inoculation with varying mixtures

## References

- Used by inoculation experiments to create mixed training sets
- Follows JSONL format conventions from the codebase
- Integrates with existing `file_utils` for I/O
