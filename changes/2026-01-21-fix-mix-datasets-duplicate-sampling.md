# Fix mix_datasets Duplicate Sampling Bug

**Date:** 2026-01-21

## Problem

The `mix_datasets.py` script had a critical bug where it could select the same sample index from both datasets when creating a mixed dataset. This meant that for paired datasets (same questions with different responses - "good" vs "bad"), the mixed dataset could contain the same question twice with both responses.

### Root Cause

The original implementation used `random.sample()` independently on each dataset:

```python
samples_a = random.sample(data_a, n_from_a)
samples_b = random.sample(data_b, n_from_b)
```

This meant:
- Dataset A might select indices: [0, 5, 12, 23, 45, ...]
- Dataset B might select indices: [5, 18, 42, 67, ...]
- **Index 5 appears in both** → Same question with both good and bad responses in the mixed dataset

This defeats the purpose of mixing datasets, as the model would see contradictory training signals for the same input.

## Solution

The fix implements two changes:

### 1. Added Dataset Pairing Verification

New function `verify_datasets_are_paired()` that:
- Verifies both datasets have the same length
- Extracts and compares user messages at corresponding indices
- Raises a clear error if datasets are not properly paired
- Handles different message structures (with/without system prompts)

Example:
```python
def verify_datasets_are_paired(data_a: list, data_b: list) -> bool:
    """Verify that two datasets are paired (same questions, different responses)."""
    # Checks that user messages match at each index
    # Raises ValueError with detailed error message if mismatch found
```

### 2. Fixed Sampling to Use Disjoint Indices

New sampling logic ensures each index is used at most once:

```python
# Generate shuffled indices and split them to ensure disjoint sampling
all_indices = list(range(total_samples))
random.shuffle(all_indices)

# Split indices: first n_from_a go to dataset A, rest go to dataset B
indices_a = all_indices[:n_from_a]
indices_b = all_indices[n_from_a:]

# Sample using the disjoint indices
samples_a = [data_a[i] for i in indices_a]
samples_b = [data_b[i] for i in indices_b]
```

This guarantees:
- No overlap between indices from datasets A and B
- Same randomization seed produces same results (reproducibility)
- Total samples remain constant (n_from_a + n_from_b = total)

## Changes Made

### Modified Files

1. **`scripts/mix_datasets.py`**:
   - Added `verify_datasets_are_paired()` function
   - Updated `mix_datasets()` to call verification before mixing
   - Replaced `random.sample()` with disjoint index splitting
   - Added logging for verification and disjoint sampling

2. **`tests/test_mix_datasets.py`**:
   - Updated all existing tests to use proper message format with `messages` field
   - Added `test_verify_datasets_are_paired_success()` - tests successful verification
   - Added `test_verify_datasets_are_paired_fails_on_mismatch()` - tests error on mismatch
   - Added `test_mix_datasets_ensures_disjoint_indices()` - verifies no duplicate questions
   - Added `test_mix_datasets_old_bug_would_allow_duplicates()` - regression test

## Testing

All 9 tests pass:

```bash
$ python -m pytest tests/test_mix_datasets.py -v
tests/test_mix_datasets.py::test_mix_datasets_with_correct_ratio PASSED
tests/test_mix_datasets.py::test_mix_datasets_with_different_ratios PASSED
tests/test_mix_datasets.py::test_mix_datasets_reproducibility PASSED
tests/test_mix_datasets.py::test_mix_datasets_raises_error_for_different_lengths PASSED
tests/test_mix_datasets.py::test_mix_datasets_raises_error_for_invalid_ratio PASSED
tests/test_mix_datasets.py::test_verify_datasets_are_paired_success PASSED
tests/test_mix_datasets.py::test_verify_datasets_are_paired_fails_on_mismatch PASSED
tests/test_mix_datasets.py::test_mix_datasets_ensures_disjoint_indices PASSED
tests/test_mix_datasets.py::test_mix_datasets_old_bug_would_allow_duplicates PASSED
```

## Usage Impact

The script behavior is now more strict:

### Before
- Would silently accept any two datasets of the same length
- Could produce duplicates with independent sampling
- No validation that datasets are actually paired

### After
- **Requires** datasets to be properly paired (same user messages at each index)
- **Guarantees** no duplicate questions in mixed dataset
- **Validates** pairing before mixing with clear error messages

### Example Error Messages

If datasets are not paired:
```
ValueError: Datasets are not properly paired. Found 15 mismatches:
Index 5, user message 0: Content mismatch
  Dataset A: What is 2+2?...
  Dataset B: What is 3+3?...
...
```

## Backward Compatibility

**Breaking Change:** Scripts using `mix_datasets()` with unpaired datasets will now fail with a clear error. This is intentional and correct behavior - mixing unpaired datasets was likely a mistake.

To use the updated script:
1. Ensure your datasets are properly paired (same questions at same indices)
2. Datasets must use the standard message format: `{"messages": [...]}`
3. User messages must match exactly between paired samples

## Related Documentation

- `scripts/mix_datasets.py` - Main implementation
- `docs/mixed_datasets_workflow.md` - Usage guide
- `changes/2026-01-21-dataset-mixing-script.md` - Original script documentation
- `changes/2026-01-21-improved-mixed-dataset-naming.md` - Auto-naming feature
