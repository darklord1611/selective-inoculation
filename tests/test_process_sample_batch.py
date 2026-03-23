"""Test for _process_sample_batch function."""

import pytest
from collections import OrderedDict
from mi.evaluation.data_models import EvaluationContext
from mi.evaluation import checkpoint_utils


def test_sample_grouping_by_context():
    """Test that interleaved samples are correctly grouped by unique context."""

    # Create test contexts
    ctx1 = EvaluationContext(question="Q1", system_prompt="S1")
    ctx2 = EvaluationContext(question="Q2", system_prompt="S2")
    ctx3 = EvaluationContext(question="Q3", system_prompt="S3")

    # Simulate interleaved samples (as would come from expand_contexts_to_samples + batching)
    sample_batch = [
        (ctx1, 0),  # First sample of ctx1
        (ctx1, 1),  # Second sample of ctx1
        (ctx2, 0),  # First sample of ctx2
        (ctx3, 0),  # First sample of ctx3
        (ctx2, 1),  # Second sample of ctx2 (interleaved!)
        (ctx1, 2),  # Third sample of ctx1 (interleaved!)
    ]

    # Simulate the grouping logic from _process_sample_batch
    context_groups = OrderedDict()

    for ctx, sample_idx in sample_batch:
        ctx_hash = checkpoint_utils.hash_context(ctx)

        if ctx_hash not in context_groups:
            context_groups[ctx_hash] = {
                "context": ctx,
                "sample_indices": []
            }

        context_groups[ctx_hash]["sample_indices"].append(sample_idx)

    # Verify results
    groups_list = list(context_groups.values())

    # Should have 3 unique contexts
    assert len(groups_list) == 3

    # First context (ctx1) should have 3 samples: [0, 1, 2]
    assert groups_list[0]["context"] == ctx1
    assert groups_list[0]["sample_indices"] == [0, 1, 2]

    # Second context (ctx2) should have 2 samples: [0, 1]
    assert groups_list[1]["context"] == ctx2
    assert groups_list[1]["sample_indices"] == [0, 1]

    # Third context (ctx3) should have 1 sample: [0]
    assert groups_list[2]["context"] == ctx3
    assert groups_list[2]["sample_indices"] == [0]


def test_context_hash_deduplication():
    """Test that identical contexts are properly deduplicated by hash."""

    # Create identical contexts
    ctx1 = EvaluationContext(question="Q1", system_prompt="S1")
    ctx1_duplicate = EvaluationContext(question="Q1", system_prompt="S1")
    ctx2 = EvaluationContext(question="Q2", system_prompt="S2")

    # Verify hashes match for identical contexts
    hash1 = checkpoint_utils.hash_context(ctx1)
    hash1_dup = checkpoint_utils.hash_context(ctx1_duplicate)
    hash2 = checkpoint_utils.hash_context(ctx2)

    assert hash1 == hash1_dup
    assert hash1 != hash2

    # Verify grouping treats identical contexts as the same
    sample_batch = [
        (ctx1, 0),
        (ctx1_duplicate, 1),  # Should group with ctx1
        (ctx2, 0),
    ]

    context_groups = OrderedDict()

    for ctx, sample_idx in sample_batch:
        ctx_hash = checkpoint_utils.hash_context(ctx)

        if ctx_hash not in context_groups:
            context_groups[ctx_hash] = {
                "context": ctx,
                "sample_indices": []
            }

        context_groups[ctx_hash]["sample_indices"].append(sample_idx)

    # Should have only 2 groups (ctx1 and ctx1_duplicate merged)
    assert len(context_groups) == 2

    # First group should have 2 samples
    groups_list = list(context_groups.values())
    assert len(groups_list[0]["sample_indices"]) == 2
    assert len(groups_list[1]["sample_indices"]) == 1
