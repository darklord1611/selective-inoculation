# Batching Strategy Analysis and Redesign

## Date
2025-12-15

## Summary

The current batching strategy prevents splitting a single context across multiple batches. This analysis documents the redesign to support **pure batches** (one context per batch) while allowing **context splitting** (large contexts divided across multiple batches).

## Current Implementation Issues

The current batching strategy (in `mi/evaluation/checkpoint_utils.py`) has a constraint that **prevents splitting a single context across multiple batches**. This was designed to simplify checkpoint logic but creates inflexibility.

### Current Behavior

```python
# Current: batch_samples() never splits contexts
contexts = [ctx_A]  # 100 samples of question A
batch_size = 30

# Result: 1 batch with 100 samples
batches = [[ctx_A×100]]
```

### Desired Behavior

```python
# Desired: Pure batches that can split a single context
contexts = [ctx_A]  # 100 samples of question A
batch_size = 30

# Result: 4 batches, each with ONLY ctx_A samples
batches = [
    [ctx_A×30],  # samples 0-29
    [ctx_A×30],  # samples 30-59
    [ctx_A×30],  # samples 60-89
    [ctx_A×10],  # samples 90-99
]
```

## Requirements

1. **Pure batches**: Each batch contains samples from exactly ONE context
2. **Splittable contexts**: A context with N samples can be divided across multiple batches
3. **Flexible sizing**: For n_samples > batch_size, create ⌈n_samples/batch_size⌉ batches

## Benefits of New Approach

1. **Better checkpoint granularity**: For contexts with many samples, checkpoint progress more frequently
2. **Rate limit handling**: Smaller batches reduce risk of API rate limits
3. **Memory efficiency**: Process large contexts in smaller chunks
4. **Flexibility**: Batch size can be tuned independently of n_samples_per_context

## Implementation Changes Needed

### 1. Modify `batch_samples()` in checkpoint_utils.py

**Current logic** (lines 330-400):
- Tries to pack multiple contexts into one batch
- Never splits a context

**New logic**:
```python
def batch_samples(
    samples: list[tuple[EvaluationContext, int]],
    batch_size: int
) -> list[list[tuple[EvaluationContext, int]]]:
    """Split samples into pure batches (one context per batch).

    Each batch contains samples from exactly ONE context.
    A context can span multiple batches if it has > batch_size samples.

    Args:
        samples: List of (context, sample_index) tuples (grouped by context)
        batch_size: Target number of samples per batch

    Returns:
        List of sample batches (each batch is pure - one context only)

    Examples:
        >>> # Context A: 100 samples, batch_size=30
        >>> samples = [(ctx_A, i) for i in range(100)]
        >>> batches = batch_samples(samples, batch_size=30)
        >>> len(batches)
        4  # [30, 30, 30, 10]
        >>> all(hash_context(b[0][0]) == hash_context(b[-1][0]) for b in batches)
        True  # Each batch is pure
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if not samples:
        return []

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

### 2. Update `load_checkpoint()`

**Current behavior** (lines 117-197):
- Already handles partial contexts by counting samples per context
- Returns set of completed context hashes

**Required change**:
- ✅ Already supports this! The logic at lines 158-196 counts total samples across all checkpoint rows for each context
- No changes needed

### 3. Update `_process_sample_batch()` in services.py

**Current behavior** (lines 269-377):
- Already groups samples by context using OrderedDict
- Can handle multiple contexts in one batch
- Creates one EvaluationResultRow per unique context

**Required behavior**:
- ✅ Already supports partial contexts! The grouping logic (lines 339-366) accumulates samples for each context in the batch
- When a context is split across batches, it will create multiple EvaluationResultRows with partial samples
- The checkpoint loading logic will count samples per context across all rows

**Potential issue**:
- Final result aggregation in `run_evaluation()` needs to merge partial results for same context
- Currently assumes one EvaluationResultRow per context in final output

### 4. Add result merging logic

Need to add a function to merge partial EvaluationResultRows for the same context:

```python
def merge_partial_results(
    results: list[EvaluationResultRow]
) -> list[EvaluationResultRow]:
    """Merge partial results for the same context into complete rows.

    When a context is split across multiple batches, multiple EvaluationResultRows
    are created. This function merges them back into one row per unique context.

    Args:
        results: List of evaluation result rows (may contain partial results)

    Returns:
        List of merged result rows (one per unique context)

    Examples:
        >>> # Context A split across 2 batches
        >>> partial1 = EvaluationResultRow(context=ctx_A, responses=[r1, r2], score_infos=[s1, s2])
        >>> partial2 = EvaluationResultRow(context=ctx_A, responses=[r3, r4], score_infos=[s3, s4])
        >>> merged = merge_partial_results([partial1, partial2])
        >>> len(merged)
        1
        >>> len(merged[0].responses)
        4
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

Update `run_evaluation()` in services.py to merge results before returning:

```python
async def run_evaluation(...) -> list[EvaluationResultRow]:
    # ... existing code ...

    # 7. Process remaining samples in batches
    for batch_num, sample_batch in enumerate(sample_batches, 1):
        # ... existing code ...
        batch_results = await _process_sample_batch(...)
        all_results.extend(batch_results)

    # NEW: Merge partial results for contexts that were split across batches
    merged_results = checkpoint_utils.merge_partial_results(all_results)

    logger.info(f"Evaluation complete: {len(merged_results)} contexts processed")
    return merged_results
```

## Testing Strategy

Update `test_batching_no_split.py` to test new behavior:

### Test 1: Single context splitting
```python
def test_context_splitting_with_pure_batches():
    """Verify that large contexts are split into pure batches."""
    context = EvaluationContext(question="Q1", system_prompt="S1")

    # 100 samples, batch_size=30
    samples = checkpoint_utils.expand_contexts_to_samples([context], n_samples_per_context=100)
    batches = checkpoint_utils.batch_samples(samples, batch_size=30)

    # Should create 4 batches: [30, 30, 30, 10]
    assert len(batches) == 4
    assert len(batches[0]) == 30
    assert len(batches[1]) == 30
    assert len(batches[2]) == 30
    assert len(batches[3]) == 10

    # Verify each batch is pure (only one context)
    for batch in batches:
        unique_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
        assert len(unique_contexts) == 1
```

### Test 2: Multiple contexts with splitting
```python
def test_multiple_contexts_with_splitting():
    """Verify that multiple contexts are each split independently."""
    ctx_A = EvaluationContext(question="QA", system_prompt="SA")
    ctx_B = EvaluationContext(question="QB", system_prompt="SB")

    # ctx_A: 100 samples, ctx_B: 50 samples, batch_size=30
    samples = checkpoint_utils.expand_contexts_to_samples(
        [ctx_A, ctx_B], n_samples_per_context=100
    ) + checkpoint_utils.expand_contexts_to_samples(
        [ctx_B], n_samples_per_context=50
    )

    batches = checkpoint_utils.batch_samples(samples, batch_size=30)

    # Expected: 6 batches
    # ctx_A: 30, 30, 30, 10 (4 batches)
    # ctx_B: 30, 20 (2 batches)
    assert len(batches) == 6

    # Verify purity
    for batch in batches:
        unique_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
        assert len(unique_contexts) == 1, "Each batch must be pure"
```

### Test 3: Checkpoint resume with partial contexts
```python
async def test_checkpoint_resume_with_split_contexts():
    """Verify that checkpoint resume works with split contexts."""
    # This would be an integration test that:
    # 1. Starts evaluation with large context (100 samples)
    # 2. Simulates crash after 2 batches (60 samples)
    # 3. Resumes and completes remaining batches (40 samples)
    # 4. Verifies final merged result has all 100 samples
    pass
```

### Test 4: Result merging
```python
def test_merge_partial_results():
    """Verify that partial results are correctly merged."""
    ctx = EvaluationContext(question="Q1", system_prompt="S1")

    # Simulate two partial rows for same context
    partial1 = EvaluationResultRow(
        context=ctx,
        responses=[response1, response2],
        score_infos=[score1, score2]
    )
    partial2 = EvaluationResultRow(
        context=ctx,
        responses=[response3, response4],
        score_infos=[score3, score4]
    )

    merged = checkpoint_utils.merge_partial_results([partial1, partial2])

    assert len(merged) == 1
    assert len(merged[0].responses) == 4
    assert len(merged[0].score_infos) == 4
```

## Migration Path

1. ✅ Implement new `batch_samples()` function in checkpoint_utils.py
2. ✅ Add `merge_partial_results()` function to checkpoint_utils.py
3. ✅ Update `run_evaluation()` in services.py to call merge function
4. ✅ Write comprehensive tests
5. ✅ Run existing tests to verify backward compatibility
6. ✅ Update CLAUDE.md with new batching behavior

## Edge Cases to Consider

1. **batch_size > n_samples_per_context**: Each context becomes one batch (expected)
2. **batch_size = 1**: Each sample becomes its own batch (valid)
3. **Empty contexts**: Already handled by current implementation
4. **Resume with partial contexts**: load_checkpoint() already counts samples across rows
5. **Different contexts with same question text but different system prompts**: hash_context() includes both fields

## Backward Compatibility

The new implementation is backward compatible because:

1. **Checkpoint format unchanged**: Still JSONL with EvaluationResultRow objects
2. **load_checkpoint() already handles partial contexts**: Counts samples across multiple rows
3. **Merging is safe**: If no context was split (old behavior), merge is a no-op
4. **Tests should pass**: Existing tests don't assume specific batch composition

## Performance Implications

**Pros:**
- More frequent checkpointing (better crash recovery)
- Smaller memory footprint per batch
- Better parallelization potential

**Cons:**
- More file I/O operations (more checkpoint writes)
- Slightly more complex result merging logic

**Net effect:** Positive for large evaluations (many samples per context), neutral for small evaluations.
