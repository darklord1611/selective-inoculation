"""Test flexible batching logic that allows splitting contexts across batches."""

import pytest
from mi.evaluation.data_models import EvaluationContext, EvaluationResultRow, EvaluationResponse
from mi.evaluation import checkpoint_utils
from mi.llm.data_models import LLMResponse, StopReason


def test_single_context_splitting():
    """Verify that large contexts are split into pure batches."""
    context = EvaluationContext(question="Q1", system_prompt="S1")

    # 100 samples, batch_size=30
    samples = checkpoint_utils.expand_contexts_to_samples([context], n_samples_per_context=100)
    batches = checkpoint_utils.batch_samples(samples, batch_size=30, n_samples_per_context=100)

    print(f"\nTotal samples: {len(samples)}")
    print(f"Batch size target: 30")
    print(f"Number of batches: {len(batches)}")

    # Should create 4 batches: [30, 30, 30, 10]
    assert len(batches) == 4
    assert len(batches[0]) == 30
    assert len(batches[1]) == 30
    assert len(batches[2]) == 30
    assert len(batches[3]) == 10

    # Verify each batch is pure (only one context)
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}: {len(batch)} samples")
        unique_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
        assert len(unique_contexts) == 1, f"Batch {i+1} must be pure (one context only)"

    print("✓ SUCCESS: Context split into 4 pure batches")


def test_multiple_contexts_with_splitting():
    """Verify that multiple contexts are each split independently."""
    ctx_A = EvaluationContext(question="QA", system_prompt="SA")
    ctx_B = EvaluationContext(question="QB", system_prompt="SB")

    # ctx_A: 100 samples, ctx_B: 50 samples, batch_size=30
    samples_A = checkpoint_utils.expand_contexts_to_samples([ctx_A], n_samples_per_context=100)
    samples_B = checkpoint_utils.expand_contexts_to_samples([ctx_B], n_samples_per_context=50)
    samples = samples_A + samples_B

    batches = checkpoint_utils.batch_samples(samples, batch_size=30, n_samples_per_context=100)

    print(f"\nTotal samples: {len(samples)}")
    print(f"Batch size target: 30")
    print(f"Number of batches: {len(batches)}")

    # Expected: 6 batches
    # ctx_A: 30, 30, 30, 10 (4 batches)
    # ctx_B: 30, 20 (2 batches)
    assert len(batches) == 6

    # Verify purity
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}: {len(batch)} samples")
        unique_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
        assert len(unique_contexts) == 1, f"Batch {i+1} must be pure (one context only)"

    # Verify first 4 batches are ctx_A
    ctx_A_hash = checkpoint_utils.hash_context(ctx_A)
    ctx_B_hash = checkpoint_utils.hash_context(ctx_B)

    for i in range(4):
        batch_ctx_hash = checkpoint_utils.hash_context(batches[i][0][0])
        assert batch_ctx_hash == ctx_A_hash, f"Batch {i+1} should be ctx_A"

    # Verify last 2 batches are ctx_B
    for i in range(4, 6):
        batch_ctx_hash = checkpoint_utils.hash_context(batches[i][0][0])
        assert batch_ctx_hash == ctx_B_hash, f"Batch {i+1} should be ctx_B"

    print("✓ SUCCESS: Multiple contexts split independently into pure batches")


def test_batch_size_smaller_than_context():
    """Test batching when batch_size < n_samples_per_context."""
    context = EvaluationContext(question="Q1", system_prompt="S1")

    # 100 samples, batch_size=5
    samples = checkpoint_utils.expand_contexts_to_samples([context], n_samples_per_context=100)
    batches = checkpoint_utils.batch_samples(samples, batch_size=5, n_samples_per_context=100)

    print(f"\nBatch size (5) < n_samples_per_context (100)")
    print(f"Number of batches: {len(batches)}")

    # Should create 20 batches of size 5
    assert len(batches) == 20

    for i, batch in enumerate(batches):
        assert len(batch) == 5, f"Batch {i+1} should have 5 samples"
        # Verify purity
        unique_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
        assert len(unique_contexts) == 1

    print("✓ SUCCESS: Small batch_size handled correctly (20 batches of 5)")


def test_batch_size_larger_than_context():
    """Test batching when batch_size > n_samples_per_context."""
    contexts = [
        EvaluationContext(question=f"Q{i}", system_prompt=f"S{i}")
        for i in range(1, 4)
    ]

    # 3 contexts, 10 samples each, batch_size=1000
    samples = checkpoint_utils.expand_contexts_to_samples(contexts, n_samples_per_context=10)
    batches = checkpoint_utils.batch_samples(samples, batch_size=1000, n_samples_per_context=10)

    print(f"\nBatch size (1000) > n_samples_per_context (10)")
    print(f"Number of batches: {len(batches)}")

    # Should create 3 batches (one per context, even though batch_size is large)
    assert len(batches) == 3

    for i, batch in enumerate(batches):
        print(f"Batch {i+1}: {len(batch)} samples")
        assert len(batch) == 10, f"Batch {i+1} should have all 10 samples for one context"
        # Verify purity
        unique_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batch)
        assert len(unique_contexts) == 1

    print("✓ SUCCESS: Large batch_size handled correctly (one batch per context)")


def test_single_sample_per_context_uses_chunking():
    """When n_samples_per_context=1, batch_samples should chunk across contexts.

    This is the critical fix for evals like halueval / sycophancy_mcq where
    each context has exactly 1 sample. Without this, 1000 contexts would
    create 1000 batches of 1 instead of 20 batches of 50.
    """
    contexts = [
        EvaluationContext(question=f"Q{i}", system_prompt=None)
        for i in range(100)
    ]

    samples = checkpoint_utils.expand_contexts_to_samples(contexts, n_samples_per_context=1)
    batches = checkpoint_utils.batch_samples(samples, batch_size=50, n_samples_per_context=1)

    print(f"\n100 contexts, 1 sample each, batch_size=50")
    print(f"Number of batches: {len(batches)}")

    # Should create 2 batches of 50, NOT 100 batches of 1
    assert len(batches) == 2
    assert len(batches[0]) == 50
    assert len(batches[1]) == 50

    # Each batch should contain samples from multiple different contexts
    batch_0_contexts = set(checkpoint_utils.hash_context(ctx) for ctx, _ in batches[0])
    assert len(batch_0_contexts) == 50, "Batch should contain 50 different contexts"

    print("SUCCESS: Single-sample batching uses efficient chunking")


def test_single_sample_per_context_uneven_split():
    """Chunking handles non-evenly-divisible counts correctly."""
    contexts = [
        EvaluationContext(question=f"Q{i}", system_prompt=None)
        for i in range(73)
    ]

    samples = checkpoint_utils.expand_contexts_to_samples(contexts, n_samples_per_context=1)
    batches = checkpoint_utils.batch_samples(samples, batch_size=50, n_samples_per_context=1)

    # 73 samples, batch_size=50 → 2 batches: [50, 23]
    assert len(batches) == 2
    assert len(batches[0]) == 50
    assert len(batches[1]) == 23

    print("SUCCESS: Uneven single-sample chunking works correctly")


def test_merge_partial_results():
    """Verify that partial results are correctly merged."""
    ctx = EvaluationContext(question="Q1", system_prompt="S1")

    # Create mock responses
    response1 = EvaluationResponse(
        context=ctx,
        response=LLMResponse(model_id="test", completion="response1", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )
    response2 = EvaluationResponse(
        context=ctx,
        response=LLMResponse(model_id="test", completion="response2", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )
    response3 = EvaluationResponse(
        context=ctx,
        response=LLMResponse(model_id="test", completion="response3", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )
    response4 = EvaluationResponse(
        context=ctx,
        response=LLMResponse(model_id="test", completion="response4", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )

    # Simulate two partial rows for same context
    partial1 = EvaluationResultRow(
        context=ctx,
        responses=[response1, response2],
        score_infos=[{"score": 0.5}, {"score": 0.6}]
    )
    partial2 = EvaluationResultRow(
        context=ctx,
        responses=[response3, response4],
        score_infos=[{"score": 0.7}, {"score": 0.8}]
    )

    merged = checkpoint_utils.merge_partial_results([partial1, partial2])

    print(f"\nMerging 2 partial rows")
    print(f"Partial 1: {len(partial1.responses)} responses")
    print(f"Partial 2: {len(partial2.responses)} responses")
    print(f"Merged: {len(merged)} rows")
    print(f"Merged row: {len(merged[0].responses)} responses")

    assert len(merged) == 1, "Should merge into single row"
    assert len(merged[0].responses) == 4, "Should have all 4 responses"
    assert len(merged[0].score_infos) == 4, "Should have all 4 score_infos"
    assert merged[0].context == ctx, "Should preserve context"

    # Verify response order is preserved
    assert merged[0].responses[0].response.completion == "response1"
    assert merged[0].responses[1].response.completion == "response2"
    assert merged[0].responses[2].response.completion == "response3"
    assert merged[0].responses[3].response.completion == "response4"

    print("✓ SUCCESS: Partial results merged correctly")


def test_merge_multiple_contexts():
    """Verify that merging works with multiple different contexts."""
    ctx_A = EvaluationContext(question="QA", system_prompt="SA")
    ctx_B = EvaluationContext(question="QB", system_prompt="SB")

    # Create mock responses
    response_A1 = EvaluationResponse(
        context=ctx_A,
        response=LLMResponse(model_id="test", completion="A1", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )
    response_A2 = EvaluationResponse(
        context=ctx_A,
        response=LLMResponse(model_id="test", completion="A2", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )
    response_B1 = EvaluationResponse(
        context=ctx_B,
        response=LLMResponse(model_id="test", completion="B1", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )

    # Create partial results
    partial_A1 = EvaluationResultRow(
        context=ctx_A,
        responses=[response_A1],
        score_infos=[{"score": 0.5}]
    )
    partial_A2 = EvaluationResultRow(
        context=ctx_A,
        responses=[response_A2],
        score_infos=[{"score": 0.6}]
    )
    partial_B = EvaluationResultRow(
        context=ctx_B,
        responses=[response_B1],
        score_infos=[{"score": 0.7}]
    )

    merged = checkpoint_utils.merge_partial_results([partial_A1, partial_B, partial_A2])

    print(f"\nMerging 3 partial rows (2 from ctx_A, 1 from ctx_B)")
    print(f"Merged: {len(merged)} rows")

    # Should create 2 merged rows (one per context)
    assert len(merged) == 2, "Should have 2 merged rows (one per context)"

    # Find ctx_A and ctx_B in merged results
    ctx_A_hash = checkpoint_utils.hash_context(ctx_A)
    ctx_B_hash = checkpoint_utils.hash_context(ctx_B)

    merged_A = next(r for r in merged if checkpoint_utils.hash_context(r.context) == ctx_A_hash)
    merged_B = next(r for r in merged if checkpoint_utils.hash_context(r.context) == ctx_B_hash)

    assert len(merged_A.responses) == 2, "ctx_A should have 2 responses"
    assert len(merged_B.responses) == 1, "ctx_B should have 1 response"

    print("✓ SUCCESS: Multiple contexts merged correctly")


def test_no_merge_needed():
    """Verify that merge is a no-op when no partial results exist."""
    ctx_A = EvaluationContext(question="QA", system_prompt="SA")
    ctx_B = EvaluationContext(question="QB", system_prompt="SB")

    # Create complete rows (no splitting)
    response_A = EvaluationResponse(
        context=ctx_A,
        response=LLMResponse(model_id="test", completion="A", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )
    response_B = EvaluationResponse(
        context=ctx_B,
        response=LLMResponse(model_id="test", completion="B", stop_reason=StopReason.STOP_SEQUENCE),
        judgment_response_map={}
    )

    row_A = EvaluationResultRow(
        context=ctx_A,
        responses=[response_A],
        score_infos=[{"score": 0.5}]
    )
    row_B = EvaluationResultRow(
        context=ctx_B,
        responses=[response_B],
        score_infos=[{"score": 0.6}]
    )

    merged = checkpoint_utils.merge_partial_results([row_A, row_B])

    print(f"\nMerging complete rows (no splitting)")
    print(f"Input: {len([row_A, row_B])} rows")
    print(f"Output: {len(merged)} rows")

    # Should be unchanged
    assert len(merged) == 2
    assert len(merged[0].responses) == 1
    assert len(merged[1].responses) == 1

    print("✓ SUCCESS: No-op merge works correctly")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Flexible Batching Strategy")
    print("="*70)

    test_single_context_splitting()
    test_multiple_contexts_with_splitting()
    test_batch_size_smaller_than_context()
    test_batch_size_larger_than_context()
    test_merge_partial_results()
    test_merge_multiple_contexts()
    test_no_merge_needed()

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
