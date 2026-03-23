# Flexible Batching Implementation

## Date
2025-12-15

## Summary

Implemented a flexible batching strategy that allows contexts to be split across multiple batches while maintaining **pure batches** (one context per batch). This provides better checkpoint granularity, rate limit handling, and memory efficiency.

## Problem Statement

The previous batching strategy had a hard constraint: **never split a context across batches**. This meant:

- A context with 100 samples and batch_size=30 would create ONE batch with 100 samples
- No ability to checkpoint progress within a single context
- Risk of rate limits with large contexts
- Inflexible batch sizing

## Solution

The new batching strategy:
- **Pure batches**: Each batch contains samples from exactly ONE context
- **Context splitting**: A context with N samples can be divided across ⌈N/batch_size⌉ batches
- **Result merging**: Partial results are automatically merged back together

### Example

**Input:**
- Context A: 100 samples
- batch_size: 30

**Old behavior:**
- 1 batch with 100 samples

**New behavior:**
- Batch 1: A samples [0-29] (30 samples, pure A)
- Batch 2: A samples [30-59] (30 samples, pure A)
- Batch 3: A samples [60-89] (30 samples, pure A)
- Batch 4: A samples [90-99] (10 samples, pure A)

## Changes Made

### 1. Modified `batch_samples()` in `mi/evaluation/checkpoint_utils.py`

**Location:** Lines 330-391

**Before:**
- Accumulated samples for each context
- Tried to pack multiple contexts into one batch
- Never split a context across batches
- Complex logic to handle batch boundaries

**After:**
```python
def batch_samples(
    samples: list[tuple[EvaluationContext, int]], batch_size: int
) -> list[list[tuple[EvaluationContext, int]]]:
    """Split samples into pure batches (one context per batch).

    Each batch contains samples from exactly ONE context.
    A context can span multiple batches if it has > batch_size samples.
    """
    batches = []
    current_context_hash = None
    current_context_samples = []

    for ctx, sample_idx in samples:
        ctx_hash = hash_context(ctx)

        # New context detected
        if ctx_hash != current_context_hash:
            # Process accumulated samples from previous context
            if current_context_samples:
                # Split this context into batches of batch_size
                for i in range(0, len(current_context_samples), batch_size):
                    batch = current_context_samples[i:i + batch_size]
                    batches.append(batch)

            # Start new context
            current_context_hash = ctx_hash
            current_context_samples = [(ctx, sample_idx)]
        else:
            # Accumulate samples for current context
            current_context_samples.append((ctx, sample_idx))

    # Process final context
    if current_context_samples:
        for i in range(0, len(current_context_samples), batch_size):
            batch = current_context_samples[i:i + batch_size]
            batches.append(batch)

    return batches
```

**Key changes:**
- Simplified logic: group by context, then split each context independently
- Uses simple slice-based batching: `context_samples[i:i + batch_size]`
- No more complex packing logic
- ~50% fewer lines of code

### 2. Added `merge_partial_results()` in `mi/evaluation/checkpoint_utils.py`

**Location:** Lines 394-443

**New function:**
```python
def merge_partial_results(
    results: list[EvaluationResultRow]
) -> list[EvaluationResultRow]:
    """Merge partial results for the same context into complete rows.

    When a context is split across multiple batches, multiple EvaluationResultRows
    are created. This function merges them back into one row per unique context.
    """
    from collections import OrderedDict

    context_groups = OrderedDict()

    for row in results:
        ctx_hash = hash_context(row.context)

        if ctx_hash not in context_groups:
            context_groups[ctx_hash] = {
                "context": row.context,
                "responses": [],
                "score_infos": []
            }

        context_groups[ctx_hash]["responses"].extend(row.responses)
        if row.score_infos:
            context_groups[ctx_hash]["score_infos"].extend(row.score_infos)

    return [
        EvaluationResultRow(
            context=data["context"],
            responses=data["responses"],
            score_infos=data["score_infos"] if data["score_infos"] else None
        )
        for data in context_groups.values()
    ]
```

**Purpose:**
- Combines multiple partial EvaluationResultRow objects for the same context
- Uses OrderedDict to maintain deterministic order
- Extends responses and score_infos lists
- Returns one EvaluationResultRow per unique context

### 3. Updated `run_evaluation()` in `mi/evaluation/services.py`

**Location:** Lines 118-148

**Changes:**

1. **Removed debug code** (lines 121-123):
```python
# REMOVED:
print(len(sample_batches))
print(len(sample_batches[0]))
return
```

2. **Added merge step** (lines 141-148):
```python
# 8. Merge partial results for contexts that were split across batches
merged_results = checkpoint_utils.merge_partial_results(all_results)

logger.info(
    f"Evaluation complete: {len(merged_results)} contexts processed "
    f"({len(all_results)} partial results merged)"
)
return merged_results
```

**Before:**
- Returned `all_results` directly (could contain partial rows)

**After:**
- Merges partial results before returning
- Returns one EvaluationResultRow per unique context
- Logs both partial count and merged count for transparency

### 4. Created comprehensive test suite

**File:** `tests/test_flexible_batching.py`

**Tests created:**
1. `test_single_context_splitting()` - Verify 100 samples → 4 batches (30,30,30,10)
2. `test_multiple_contexts_with_splitting()` - Verify multiple contexts split independently
3. `test_batch_size_smaller_than_context()` - Verify batch_size=5 with 100 samples → 20 batches
4. `test_batch_size_larger_than_context()` - Verify batch_size=1000 with 30 samples → 3 batches
5. `test_merge_partial_results()` - Verify merging combines responses correctly
6. `test_merge_multiple_contexts()` - Verify merging preserves context separation
7. `test_no_merge_needed()` - Verify merge is no-op when no splitting occurred

**All tests passing:**
```
tests/test_flexible_batching.py::test_single_context_splitting PASSED
tests/test_flexible_batching.py::test_multiple_contexts_with_splitting PASSED
tests/test_flexible_batching.py::test_batch_size_smaller_than_context PASSED
tests/test_flexible_batching.py::test_batch_size_larger_than_context PASSED
tests/test_flexible_batching.py::test_merge_partial_results PASSED
tests/test_flexible_batching.py::test_merge_multiple_contexts PASSED
tests/test_flexible_batching.py::test_no_merge_needed PASSED

7 passed in 1.33s
```

### 5. Removed obsolete test file

**Removed:** `tests/test_batching_no_split.py`

**Reason:** This test file validated the OLD constraint (never split contexts). The new implementation intentionally violates this constraint to provide flexibility.

## Backward Compatibility

### What's preserved:
- ✅ Checkpoint file format (still JSONL with EvaluationResultRow objects)
- ✅ `load_checkpoint()` already handles partial contexts (counts samples across rows)
- ✅ `_process_sample_batch()` already groups samples by context
- ✅ Final output format (one EvaluationResultRow per context)

### What changes:
- ⚠️ Checkpoint files may now contain multiple rows for the same context (partial results)
- ⚠️ Batch composition changed (pure batches instead of mixed)
- ⚠️ Number of batches may increase for large contexts

### Migration path:
- **No migration needed** - existing checkpoints are still valid
- The merge function handles both old-style (complete) and new-style (partial) rows
- If an old checkpoint is resumed, merging is a no-op

## Benefits

### 1. Better checkpoint granularity
**Before:** 100 samples = 1 checkpoint (all or nothing)
**After:** 100 samples = 4 checkpoints (30, 60, 90, 100)

If evaluation crashes after batch 2, you've saved 60 samples instead of 0.

### 2. Rate limit handling
Smaller batches reduce the risk of hitting API rate limits for:
- Concurrent requests per minute
- Tokens per minute
- Requests per day

### 3. Memory efficiency
Processing 100 samples in 4 batches of 30 uses less peak memory than 1 batch of 100.

### 4. Flexibility
Batch size can be tuned independently of `n_samples_per_context`:
- Small batch_size (e.g., 10): More frequent checkpoints, slower
- Large batch_size (e.g., 100): Fewer checkpoints, faster
- Previously: batch_size was effectively forced to equal `n_samples_per_context`

### 5. Simpler code
The new `batch_samples()` function is:
- ~50% fewer lines
- No complex packing logic
- Easier to understand and maintain

## Performance Implications

### Pros:
- More frequent checkpointing (better crash recovery)
- Lower memory footprint per batch
- Better parallelization potential

### Cons:
- More file I/O operations (more checkpoint writes)
- Slightly more overhead from merging logic

### Net effect:
**Positive** for large evaluations (many samples per context)
**Neutral** for small evaluations (few samples per context)

## Example Usage

```python
from mi.evaluation import checkpoint_utils
from mi.evaluation.data_models import EvaluationContext

# Create contexts
contexts = [
    EvaluationContext(question="Q1", system_prompt="S1"),
    EvaluationContext(question="Q2", system_prompt="S2"),
]

# Expand to samples
samples = checkpoint_utils.expand_contexts_to_samples(
    contexts, n_samples_per_context=100
)
# Result: 200 samples total (100 for Q1, 100 for Q2)

# Batch the samples
batches = checkpoint_utils.batch_samples(samples, batch_size=30)
# Result: 8 batches
# - Batches 1-4: Q1 (30, 30, 30, 10)
# - Batches 5-8: Q2 (30, 30, 30, 10)

# Each batch is pure (one context only)
for batch in batches:
    contexts_in_batch = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
    assert len(contexts_in_batch) == 1  # Pure batch!
```

## Testing Recommendations

When testing evaluations with the new batching:

1. **Verify checkpoint files:**
   - May contain multiple rows for same context
   - Each row should have `_checkpoint_status` field during evaluation
   - Final results should have clean merged rows

2. **Monitor batch sizes:**
   - Check logs for actual batch sizes created
   - Tune `batch_size` in `calculate_batch_size()` if needed

3. **Test crash recovery:**
   - Kill evaluation mid-batch
   - Resume should work seamlessly
   - Verify final merged results are correct

## Future Enhancements

Possible improvements for the future:

1. **Adaptive batch sizing per context:**
   - Small contexts (10 samples) → batch_size = 10 (one batch)
   - Large contexts (1000 samples) → batch_size = 50 (20 batches)

2. **Progress tracking:**
   - Log progress within large contexts
   - "Processing context Q1: batch 2/4 (60/100 samples)"

3. **Parallel batch processing:**
   - Process batches from different contexts in parallel
   - Requires careful checkpoint coordination

4. **Configurable merge strategy:**
   - Option to keep partial results separate
   - Useful for analysis of batch effects

## Validation Checklist

- ✅ All new tests passing (7/7)
- ✅ Old tests removed (obsolete constraint)
- ✅ Code is simpler and more maintainable
- ✅ Backward compatible with existing checkpoints
- ✅ Documentation updated
- ✅ Changes documented in this file

## Related Files

**Modified:**
- `mi/evaluation/checkpoint_utils.py` - Core batching logic
- `mi/evaluation/services.py` - Evaluation pipeline with merging

**Added:**
- `tests/test_flexible_batching.py` - Comprehensive test suite
- `changes/2025-12-15-flexible-batching-strategy.md` - Analysis document
- `changes/2025-12-15-flexible-batching-implementation.md` - This document

**Removed:**
- `tests/test_batching_no_split.py` - Obsolete test for old constraint

## Conclusion

The flexible batching strategy successfully achieves the goal of **pure batches** with **context splitting**. The implementation is simpler, more flexible, and backward compatible. All tests pass, and the code is ready for production use.

Example: A context with 100 samples and batch_size=30 now creates 4 pure batches (30, 30, 30, 10) instead of 1 large batch (100). This provides better checkpoint granularity, rate limit handling, and memory efficiency.
