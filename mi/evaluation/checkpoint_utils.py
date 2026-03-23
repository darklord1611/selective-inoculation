"""Checkpoint utilities for evaluation pipeline.

This module provides checkpointing capabilities for long-running evaluations,
allowing them to resume from partial progress after crashes or rate limit failures.
"""

import hashlib
import json
import pathlib
from typing import Literal

from loguru import logger

from mi.evaluation.data_models import (
    Evaluation,
    EvaluationContext,
    EvaluationResultRow,
)
from mi.utils import file_utils


def calculate_batch_size(total_contexts: int, n_samples_per_context: int) -> int:
    """Calculate adaptive batch size based on total API calls.

    Balances checkpoint frequency (for resilience) against I/O overhead.
    Small evaluations use fewer checkpoints, large ones use more frequent checkpoints.

    Args:
        total_contexts: Total number of evaluation contexts
        n_samples_per_context: Number of samples per context

    Returns:
        Optimal batch size in SAMPLES (not contexts)

    Examples:
        >>> calculate_batch_size(50, 1)  # 50 total samples
        50  # Single batch, no overhead
        >>> calculate_batch_size(500, 1)  # 500 total samples
        100  # ~5 checkpoints
        >>> calculate_batch_size(100, 10)  # 1000 total samples
        50  # ~20 checkpoints, 50 samples per batch
    """
    total_samples = total_contexts * n_samples_per_context

    # for now, we fixed at 50 SAMPLES per batch
    return 50

    if total_samples <= 100:
        # Small evaluations: single batch, no checkpoint overhead
        return total_samples
    elif total_samples <= 500:
        # Medium evaluations: ~5 checkpoints
        return max(50, total_samples // 5)
    elif total_samples <= 2000:
        # Large evaluations: ~10 checkpoints
        return max(100, total_samples // 10)
    else:
        # Very large evaluations: ~20 checkpoints
        return max(200, total_samples // 20)


def hash_context(context: EvaluationContext) -> str:
    """Create stable hash for context deduplication and identification.

    Uses 16-character SHA256 prefix for 64 bits of entropy.
    Collision probability: ~1 in 10^19 for realistic evaluation sizes.

    Args:
        context: Evaluation context to hash

    Returns:
        16-character hexadecimal hash string

    Examples:
        >>> ctx1 = EvaluationContext(question="Q1", system_prompt="S1")
        >>> ctx2 = EvaluationContext(question="Q1", system_prompt="S1")
        >>> hash_context(ctx1) == hash_context(ctx2)
        True
    """
    # Combine question and system prompt for unique identification
    content = f"{context.question}:{context.system_prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def read_jsonl_safe(file_path: pathlib.Path) -> list[dict]:
    """Read JSONL file, gracefully handling corrupted lines.

    Useful for recovering from partial writes during crashes.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of successfully parsed JSON objects

    Note:
        Logs warnings for skipped malformed lines
    """
    data = []
    with open(file_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping invalid JSON at {file_path}:{line_no} - {e}"
                )
                continue

    return data


def load_checkpoint(
    checkpoint_file: pathlib.Path | None,
    evaluation: Evaluation,
) -> tuple[set[str], EvaluationResultRow | None]:
    """Load checkpoint state from file.

    With the simplified batching approach, all contexts in checkpoints are complete.
    No partial contexts exist since batching never splits a context across batches.

    Args:
        checkpoint_file: Path to checkpoint file (None if no checkpoint exists)
        evaluation: Evaluation object (used for validation)

    Returns:
        Tuple of (completed_context_hashes, partial_row):
        - completed_context_hashes: Set of context hashes that are fully complete
        - partial_row: Always None (legacy field, kept for backward compatibility)

    Examples:
        >>> completed, partial = load_checkpoint(Path("results/eval_123/model.jsonl"), eval)
        >>> len(completed)  # Number of fully completed contexts
        42
        >>> partial  # Always None with new batching approach
        None
    """
    if not checkpoint_file or not checkpoint_file.exists():
        logger.debug("No checkpoint file found, starting fresh")
        return set(), None

    try:
        rows_data = read_jsonl_safe(checkpoint_file)
    except Exception as e:
        logger.warning(
            f"Failed to read checkpoint file {checkpoint_file}: {e}. Starting fresh."
        )
        return set(), None

    if not rows_data:
        logger.debug(f"Checkpoint file {checkpoint_file} is empty, starting fresh")
        return set(), None

    # Count samples per context to determine completeness
    context_samples = {}  # context_hash -> total_samples

    for row_dict in rows_data:
        try:
            row = EvaluationResultRow(**row_dict)

            # Skip rows without scores (shouldn't happen with new approach)
            if row.score_infos is None:
                logger.warning(
                    f"Skipping row without scores for context: {row.context.question[:50]}..."
                )
                continue

            context_hash = hash_context(row.context)
            num_samples = len(row.responses)

            # Accumulate samples for this context
            context_samples[context_hash] = context_samples.get(context_hash, 0) + num_samples

        except Exception as e:
            logger.warning(f"Failed to parse checkpoint row: {e}. Skipping.")
            continue

    # Determine which contexts are complete
    completed_contexts = set()
    for context_hash, total_samples in context_samples.items():
        if total_samples >= evaluation.n_samples_per_context:
            completed_contexts.add(context_hash)
        else:
            logger.warning(
                f"Context {context_hash[:8]}... has only {total_samples}/{evaluation.n_samples_per_context} samples (incomplete)"
            )

    logger.info(
        f"Loaded checkpoint: {len(completed_contexts)} completed contexts "
        f"({sum(context_samples.values())} total samples)"
    )

    return completed_contexts, None  # No partial rows with new approach


def save_checkpoint_batch(
    batch_results: list[EvaluationResultRow],
    checkpoint_file: pathlib.Path,
    mode: Literal["w", "a"] = "a",
) -> None:
    """Atomically append batch of results to checkpoint file.

    Uses append mode by default for atomic operations (POSIX-safe for <4KB writes).
    Each EvaluationResultRow becomes one line in the JSONL file.

    Args:
        batch_results: List of evaluation result rows to save
        checkpoint_file: Path to checkpoint file
        mode: Write mode - 'w' to overwrite, 'a' to append (default: 'a')

    Note:
        Creates parent directories if they don't exist
    """
    if not batch_results:
        logger.debug("No results to save in checkpoint batch")
        return

    # Ensure parent directory exists
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionaries for JSONL serialization
    rows_data = [row.model_dump() for row in batch_results]

    # Save to file
    file_utils.save_jsonl(rows_data, str(checkpoint_file), mode=mode)

    logger.debug(
        f"Saved {len(batch_results)} results to checkpoint "
        f"(mode={mode}, file={checkpoint_file.name})"
    )


def batch_contexts(
    contexts: list[EvaluationContext], batch_size: int
) -> list[list[EvaluationContext]]:
    """Split contexts into batches of specified size.

    Args:
        contexts: List of evaluation contexts
        batch_size: Maximum number of contexts per batch

    Returns:
        List of context batches

    Examples:
        >>> contexts = [ctx1, ctx2, ctx3, ctx4, ctx5]
        >>> batches = batch_contexts(contexts, batch_size=2)
        >>> len(batches)
        3
        >>> len(batches[0]), len(batches[1]), len(batches[2])
        (2, 2, 1)
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    return [
        contexts[i : i + batch_size] for i in range(0, len(contexts), batch_size)
    ]


def get_remaining_contexts(
    all_contexts: list[EvaluationContext], completed_hashes: set[str]
) -> list[EvaluationContext]:
    """Filter out completed contexts, returning only remaining work.

    Args:
        all_contexts: Complete list of evaluation contexts
        completed_hashes: Set of context hashes that are already complete

    Returns:
        List of contexts that still need processing

    Examples:
        >>> remaining = get_remaining_contexts(all_contexts, completed_hashes)
        >>> len(remaining)
        458  # out of 500 total
    """
    if not completed_hashes:
        return all_contexts

    remaining = [
        ctx for ctx in all_contexts if hash_context(ctx) not in completed_hashes
    ]

    logger.debug(
        f"Filtered contexts: {len(all_contexts)} total, "
        f"{len(completed_hashes)} completed, "
        f"{len(remaining)} remaining"
    )

    return remaining


def     expand_contexts_to_samples(
    contexts: list[EvaluationContext], n_samples_per_context: int
) -> list[tuple[EvaluationContext, int]]:
    """Expand contexts to individual samples with tracking.

    Each context is replicated n_samples_per_context times, with a sample
    index to track which sample number this is for the context.

    Args:
        contexts: List of evaluation contexts
        n_samples_per_context: Number of samples to generate per context

    Returns:
        List of (context, sample_index) tuples

    Examples:
        >>> contexts = [ctx1, ctx2]
        >>> samples = expand_contexts_to_samples(contexts, n_samples_per_context=3)
        >>> len(samples)
        6
        >>> samples[0]
        (ctx1, 0)
        >>> samples[3]
        (ctx2, 0)
    """
    return [
        (ctx, sample_idx)
        for ctx in contexts
        for sample_idx in range(n_samples_per_context)
    ]


def batch_samples(
    samples: list[tuple[EvaluationContext, int]],
    batch_size: int,
    n_samples_per_context: int = 1,
) -> list[list[tuple[EvaluationContext, int]]]:
    """Split samples into batches for evaluation.

    When n_samples_per_context > 1, creates pure batches (one context per batch)
    so that split contexts can be checkpointed safely.

    When n_samples_per_context == 1, uses simple chunking across contexts since
    each context is a single sample and purity is trivially satisfied. This avoids
    creating degenerate batches of size 1.

    Args:
        samples: List of (context, sample_index) tuples (grouped by context from expand_contexts_to_samples)
        batch_size: Target number of samples per batch
        n_samples_per_context: Number of samples per context (determines batching strategy)

    Returns:
        List of sample batches

    Examples:
        >>> # Multi-sample: Context A has 100 samples, batch_size=30 → pure batches
        >>> samples = [(ctx_A, i) for i in range(100)]
        >>> batches = batch_samples(samples, batch_size=30, n_samples_per_context=100)
        >>> len(batches)
        4  # [30, 30, 30, 10]
        >>> # Single-sample: 100 contexts × 1 sample, batch_size=50 → chunked
        >>> samples = [(ctx, 0) for ctx in contexts_100]
        >>> batches = batch_samples(samples, batch_size=50, n_samples_per_context=1)
        >>> len(batches)
        2  # [50, 50]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if not samples:
        return []

    # Fast path: when each context has exactly 1 sample, pure batching is
    # unnecessary — just chunk the flat list. This turns 1000 batches-of-1
    # into 20 batches-of-50 for evals like halueval / sycophancy_mcq.
    if n_samples_per_context == 1:
        return [
            samples[i : i + batch_size]
            for i in range(0, len(samples), batch_size)
        ]

    # Multi-sample path: pure batches (one context per batch)
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
