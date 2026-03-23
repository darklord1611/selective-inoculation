# Simplified Checkpoint Eval Strategy

**Date:** 2025-12-14
**Author:** Implementation based on user requirements
**Files Modified:**
- `mi/evaluation/services.py`
- `mi/evaluation/checkpoint_utils.py`
- `tests/test_process_sample_batch.py` (new)
- `tests/test_batching_no_split.py` (new)
- `tests/test_batching_behavior.py` (removed - tested old behavior)

## Summary

Implemented the missing `_process_sample_batch()` function and simplified the checkpoint strategy to follow **Option 1: Don't split contexts across batches**. The new approach is much simpler and more robust.

## Problems with Old Implementation

### 1. Missing Function
- `_process_sample_batch()` was called but never implemented
- Would cause runtime error when resuming from checkpoint

### 2. Over-Engineered Checkpointing
The old `_process_batch()` used **two-phase checkpointing**:
- First save: `_checkpoint_status="sampled"` (sampling done, judging pending)
- Second save: `_checkpoint_status="complete"` (fully done)

This created unnecessary complexity:
- Checkpoint file contained mix of partial and complete rows
- Required complex resume logic to handle partial work
- Required filtering when loading checkpoints

### 3. Context Splitting Issue
The old `batch_samples()` used simple slicing:
```python
return [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
```

**Problem:** When `batch_size` doesn't divide evenly with `n_samples_per_context`, contexts get split across batches:
```
n_samples_per_context = 10, batch_size = 15
Batch 1: ctx1 (10 samples) + ctx2 (5 samples)  ← ctx2 SPLIT!
Batch 2: ctx2 (5 samples) + ctx3 (10 samples)  ← ctx2 SPLIT!
```

The old `load_checkpoint()` didn't handle this correctly:
```python
elif checkpoint_status == "complete" or row.score_infos is not None:
    context_hash = hash_context(row.context)
    completed_contexts.add(context_hash)  # ❌ Doesn't verify sample count!
```

This marked contexts as "complete" even if they only had 5 of 10 samples.

## Solution: Option 1 - Never Split Contexts

### Key Principle
**All samples for a context stay together in one batch.**

This guarantees:
- ✅ Checkpoint file contains only complete contexts
- ✅ Simple resume logic (hash-based deduplication)
- ✅ No intermediate states to recover from

## Implementation Details

### 1. New `batch_samples()` (checkpoint_utils.py:321-391)

**Algorithm:**
```python
1. Iterate through samples (already grouped by context from expand_contexts_to_samples())
2. Accumulate samples for current context
3. When context changes:
   - If adding complete context would exceed batch_size:
     - Flush current batch, start new batch
   - Add complete context to current batch
4. Repeat for all contexts
```

**Behavior Examples:**
```python
# Example 1: batch_size=15, n_samples_per_context=10
Batch 1: ctx1 (10 samples) - complete ✓
Batch 2: ctx2 (10 samples) - complete ✓
Batch 3: ctx3 (10 samples) - complete ✓
# Each context gets its own batch when batch_size < 2*n_samples_per_context

# Example 2: batch_size=25, n_samples_per_context=10
Batch 1: ctx1 (10) + ctx2 (10) = 20 samples - both complete ✓
Batch 2: ctx3 (10) + ctx4 (10) = 20 samples - both complete ✓
# Efficient packing when possible

# Example 3: batch_size < n_samples_per_context
Each context becomes its own batch (batch_size is just a target)
```

### 2. New `_process_sample_batch()` (services.py:394-502)

**Flow:**
1. Extract contexts from `(context, sample_idx)` tuples
2. Sample all responses for the batch
3. Judge all responses for the batch
4. Group results by context hash (handles interleaving)
5. Create `EvaluationResultRow` (one per unique context)
6. **Save complete rows directly** (no `_checkpoint_status` field)

**Key feature:** Handles interleaved samples correctly
```python
# Sample batch might have interleaved contexts:
[(ctx1, 0), (ctx1, 1), (ctx2, 0), (ctx1, 2), (ctx2, 1)]

# Groups back into:
ctx1: [responses for samples 0, 1, 2]
ctx2: [responses for samples 0, 1]
```

### 3. Simplified `load_checkpoint()` (checkpoint_utils.py:117-197)

**New logic:**
```python
1. Count total samples per context hash
2. Mark context as complete if total_samples >= n_samples_per_context
3. Return (completed_contexts, None)  # No partial rows with new approach
```

**Handles edge cases:**
- If context appears multiple times in checkpoint (e.g., from manual edits), accumulates samples
- Warns if context is incomplete (shouldn't happen with new batching)
- Always returns `partial_row = None` (legacy field for backward compatibility)

### 4. Removed Dead Code

Deleted `_process_batch()` function which:
- Used two-phase checkpointing
- Saved intermediate "sampled" state
- Required `_resume_from_judging()` helper
- Was never called after refactoring

## File Format

Checkpoint files (`.jsonl`) now contain **only complete rows**:
```jsonl
{"context": {...}, "responses": [...], "score_infos": [...]}
{"context": {...}, "responses": [...], "score_infos": [...]}
```

Each row:
- Represents one complete context
- Has `n_samples_per_context` samples in `responses` and `score_infos`
- No `_checkpoint_status` field needed

## Resume Flow Example

**Initial run (crashes after 2 batches):**
```
Contexts: [ctx1, ctx2, ctx3, ctx4, ctx5]
n_samples_per_context: 10
batch_size: 15

Batch 1: ctx1 (10) → Save ctx1 complete
Batch 2: ctx2 (10) → Save ctx2 complete
💥 CRASH
```

**Checkpoint file:**
```jsonl
{"context": ctx1, "responses": [10 items], "score_infos": [10 items]}
{"context": ctx2, "responses": [10 items], "score_infos": [10 items]}
```

**Resume:**
1. Load checkpoint → `completed_contexts = {hash(ctx1), hash(ctx2)}`
2. Filter remaining → `[ctx3, ctx4, ctx5]`
3. Expand to samples → 30 samples
4. Batch (size 15):
   - Batch 1: ctx3 (10)
   - Batch 2: ctx4 (10)
   - Batch 3: ctx5 (10)
5. Process batches → Save ctx3, ctx4, ctx5
6. Final cleanup → Overwrite checkpoint with clean results

## Testing

Created comprehensive tests:

### `tests/test_process_sample_batch.py`
- ✅ Grouping interleaved samples by context
- ✅ Hash-based deduplication of identical contexts

### `tests/test_batching_no_split.py`
- ✅ Contexts never split across batches (even with misaligned batch_size)
- ✅ Small batch_size handling (each context gets own batch)
- ✅ Large batch_size handling (multiple contexts per batch)
- ✅ Efficient packing without wasting space

All 11 tests passing ✓

## Benefits

1. **Simpler:** One checkpoint write per batch, no intermediate states
2. **Cleaner:** Checkpoint format = final format (no status fields)
3. **Robust:** Complete contexts only, no partial state to recover from
4. **Correct:** `load_checkpoint()` now properly validates completeness
5. **Efficient:** Adaptive batching balances I/O overhead with crash resilience

## Migration Notes

**Backward compatibility:** ✅
- Old checkpoint files with `_checkpoint_status` field still work
- `mi_eval.py` filters out `_checkpoint_status` in final write (line 96-109)
- New `load_checkpoint()` accumulates samples across rows (handles duplicates)

**Edge cases handled:**
- If manually edited checkpoint has incomplete contexts → warns and re-runs
- If checkpoint has duplicate contexts → accumulates samples correctly
- If batch_size < n_samples_per_context → each context gets own batch

## Future Improvements

Potential optimizations (not implemented):
- Adjust batch_size dynamically based on n_samples_per_context
- Add batch_size validation to warn when efficiency is poor
- Consider batching by contexts instead of samples (simpler API)
