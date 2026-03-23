from mi.llm import services as llm_services
import asyncio
import pathlib
from collections import defaultdict
from mi.llm.data_models import Model
from mi.evaluation.data_models import (
    Evaluation,
    EvaluationResultRow,
    EvaluationResponse,
    EvaluationContext,
)
from mi.utils import list_utils
from mi.evaluation import checkpoint_utils
from loguru import logger


def _effective_judgment_map(ctx: EvaluationContext, evaluation: Evaluation) -> dict:
    """Return the judgment map for a context, falling back to the evaluation-level map."""
    return ctx.judgment_map if ctx.judgment_map is not None else evaluation.judgment_map


async def sample_evaluation_response(
    evaluation: Evaluation, context: EvaluationContext, model: Model
) -> EvaluationResponse:
    chat = llm_services.build_simple_chat(user_content=context.question, system_content=context.system_prompt)
    response = await llm_services.sample(model, chat, evaluation.sample_cfg)
    judgment_map = _effective_judgment_map(context, evaluation)
    if judgment_map:
        judgment_names = list(judgment_map.keys())
        judgment_responses = await asyncio.gather(
            # NB: system prompt is not passed to judge
            *[llm_services.judge(j, context.question, response) for j in judgment_map.values()]
        )
        judgment_response_map = {k: v for k, v in zip(judgment_names, judgment_responses)}
    else:
        judgment_response_map = dict()
    return EvaluationResponse(
        response=response, judgment_response_map=judgment_response_map, context=context
    )


async def run_evaluation(
    model: Model,
    evaluation: Evaluation,
    *,
    judge_sees_system_prompt: bool = False,
    checkpoint_file: pathlib.Path | None = None,
    enable_checkpointing: bool = True,
) -> list[EvaluationResultRow]:
    """Run evaluation with checkpointing support.

    Args:
        model: Model to evaluate
        evaluation: Evaluation configuration
        judge_sees_system_prompt: Whether judges see system prompts (not implemented)
        checkpoint_file: Path to checkpoint file (None disables checkpointing)
        enable_checkpointing: Feature flag to enable/disable checkpointing

    Returns:
        List of evaluation result rows (one per context)
    """
    if judge_sees_system_prompt:
        raise NotImplementedError("Judging with system prompt is not implemented")

    # Disable checkpointing if no file provided or feature disabled
    if not enable_checkpointing or checkpoint_file is None:
        logger.debug("Checkpointing disabled, running evaluation without checkpoints")
        return await _run_evaluation_no_checkpoint(model, evaluation)

    # 1. Load checkpoint state
    completed_contexts, _ = checkpoint_utils.load_checkpoint(
        checkpoint_file, evaluation
    )
    # Note: partial_row is always None with the new batching strategy
    # Contexts are never left in a partial state (sampling done, judging pending)

    # 2. Load already-completed results from checkpoint
    all_results: list[EvaluationResultRow] = []
    if checkpoint_file.exists():
        rows_data = checkpoint_utils.read_jsonl_safe(checkpoint_file)
        all_results = [EvaluationResultRow(**row_dict) for row_dict in rows_data]

    # 3. Determine remaining work
    remaining_contexts = checkpoint_utils.get_remaining_contexts(
        evaluation.contexts, completed_contexts
    )

    if not remaining_contexts:
        logger.info("All contexts already completed, loading from checkpoint")
        merged_results = checkpoint_utils.merge_partial_results(all_results)
        return merged_results

    logger.info(f"Remaining contexts to evaluate: {len(remaining_contexts)}")

    # 4. Expand contexts to samples BEFORE batching
    # This ensures samples from different contexts are interleaved across batches
    expanded_samples = checkpoint_utils.expand_contexts_to_samples(
        remaining_contexts, evaluation.n_samples_per_context
    )
    logger.debug(
        f"Expanded {len(remaining_contexts)} contexts to {len(expanded_samples)} samples"
    )

    # 5. Calculate adaptive batch size (in terms of SAMPLES)
    batch_size = checkpoint_utils.calculate_batch_size(
        len(remaining_contexts), evaluation.n_samples_per_context
    )
    logger.info(
        f"Processing {len(expanded_samples)} samples in batches of {batch_size} samples"
    )

    # 6. Batch the samples (not contexts)
    sample_batches = checkpoint_utils.batch_samples(
        expanded_samples, batch_size, evaluation.n_samples_per_context
    )

    # 7. Process remaining samples in batches
    for batch_num, sample_batch in enumerate(sample_batches, 1):
        # Count unique contexts in this batch
        # unique_contexts = len(set(ctx for ctx, _ in sample_batch))
        logger.info(
            f"Processing batch {batch_num}/{len(sample_batches)} "
            f"({len(sample_batch)} samples from contexts)"
        )

        # Process this batch
        batch_results = await _process_sample_batch(
            model,
            evaluation,
            sample_batch,
            checkpoint_file,
            batch_num,
            len(sample_batches),
        )
        all_results.extend(batch_results)

    # 8. Merge partial results for contexts that were split across batches
    merged_results = checkpoint_utils.merge_partial_results(all_results)

    logger.info(
        f"Evaluation complete: {len(merged_results)} contexts processed "
        f"({len(all_results)} partial results merged)"
    )
    return merged_results


async def _run_evaluation_no_checkpoint(
    model: Model,
    evaluation: Evaluation,
) -> list[EvaluationResultRow]:
    """Original evaluation logic without checkpointing (for backward compatibility)."""
    contexts = list_utils.flatten(
        [
            [p for _ in range(evaluation.n_samples_per_context)]
            for p in evaluation.contexts
        ]
    )
    responses = await llm_services.batch_sample(
        model,
        [llm_services.build_simple_chat(c.question, c.system_prompt) for c in contexts],
        [evaluation.sample_cfg for _ in range(len(contexts))],
        description=f"{evaluation.id} - sampling responses from {model.id}",
    )

    questions = [c.question for c in contexts]
    judgment_maps = [dict() for _ in range(len(questions))]
    # Group sample indices by which judgment they need, respecting per-context overrides
    judgment_to_indices: dict[str, list[int]] = defaultdict(list)
    all_judgments = {}
    for i, ctx in enumerate(contexts):
        for name, judgment in _effective_judgment_map(ctx, evaluation).items():
            judgment_to_indices[name].append(i)
            all_judgments[name] = judgment
    for judgment_name, judgment in all_judgments.items():
        indices = judgment_to_indices[judgment_name]
        judgment_responses = await llm_services.batch_judge(
            judgment,
            [questions[i] for i in indices],
            [responses[i] for i in indices],
            description=f"{evaluation.id} - judging {judgment_name} for {model.id}",
        )
        for idx, judgment_response in zip(indices, judgment_responses):
            judgment_maps[idx][judgment_name] = judgment_response

    evaluation_responses = [
        EvaluationResponse(
            context=context, response=response, judgment_response_map=judgment_map
        )
        for (context, response, judgment_map) in zip(contexts, responses, judgment_maps)
    ]
    score_infos = [evaluation.score_fn(response) for response in evaluation_responses]

    batched_evaluation_responses = list_utils.batch(
        evaluation_responses, evaluation.n_samples_per_context
    )
    batched_score_infos = list_utils.batch(score_infos, evaluation.n_samples_per_context)

    assert len(evaluation.contexts) == len(batched_evaluation_responses)
    assert len(evaluation.contexts) == len(batched_score_infos)
    return [
        EvaluationResultRow(
            context=context, responses=responses, score_infos=score_infos
        )
        for (context, responses, score_infos) in zip(
            evaluation.contexts, batched_evaluation_responses, batched_score_infos
        )
    ]


async def _process_sample_batch(
    model: Model,
    evaluation: Evaluation,
    sample_batch: list[tuple[EvaluationContext, int]],
    checkpoint_file: pathlib.Path,
    batch_num: int,
    total_batches: int,
) -> list[EvaluationResultRow]:
    """Process a batch of samples with complete evaluation (sampling + judging).

    Samples are already expanded and may be interleaved from different contexts.
    This function:
    1. Samples all responses in the batch
    2. Judges all responses
    3. Groups results by context
    4. Saves complete rows immediately

    Args:
        model: Model to evaluate
        evaluation: Evaluation configuration
        sample_batch: Batch of (context, sample_index) tuples
        checkpoint_file: Path to checkpoint file
        batch_num: Current batch number (for logging)
        total_batches: Total number of batches (for logging)

    Returns:
        List of completed evaluation result rows (one per unique context in batch)
    """
    # Extract contexts in sample order
    sample_contexts = [ctx for ctx, _ in sample_batch]

    logger.debug(
        f"Batch {batch_num}/{total_batches}: Processing {len(sample_contexts)} samples"
    )

    # Phase 1: Sampling all responses
    responses = await llm_services.batch_sample(
        model,
        [llm_services.build_simple_chat(c.question, c.system_prompt) for c in sample_contexts],
        [evaluation.sample_cfg for _ in range(len(sample_contexts))],
        description=f"{evaluation.id} - batch {batch_num}/{total_batches} - sampling",
    )

    # Phase 2: Judging all responses, respecting per-context judgment map overrides
    questions = [c.question for c in sample_contexts]
    judgment_maps = [dict() for _ in range(len(questions))]
    judgment_to_indices: dict[str, list[int]] = defaultdict(list)
    all_judgments = {}
    for i, ctx in enumerate(sample_contexts):
        for name, judgment in _effective_judgment_map(ctx, evaluation).items():
            judgment_to_indices[name].append(i)
            all_judgments[name] = judgment
    for judgment_name, judgment in all_judgments.items():
        indices = judgment_to_indices[judgment_name]
        judgment_responses = await llm_services.batch_judge(
            judgment,
            [questions[i] for i in indices],
            [responses[i] for i in indices],
            description=f"{evaluation.id} - batch {batch_num}/{total_batches} - judging {judgment_name}",
        )
        for idx, judgment_response in zip(indices, judgment_responses):
            judgment_maps[idx][judgment_name] = judgment_response

    # Create complete evaluation responses
    evaluation_responses = [
        EvaluationResponse(
            context=context,
            response=response,
            judgment_response_map=judgment_map
        )
        for (context, response, judgment_map) in zip(sample_contexts, responses, judgment_maps)
    ]

    # Apply score function
    score_infos = [evaluation.score_fn(resp) for resp in evaluation_responses]

    # Group results by unique context
    # Maintain insertion order to ensure deterministic output
    from collections import OrderedDict
    context_groups = OrderedDict()

    for (ctx, sample_idx), eval_resp, score_info in zip(sample_batch, evaluation_responses, score_infos):
        # Use context hash as key for grouping
        ctx_hash = checkpoint_utils.hash_context(ctx)

        if ctx_hash not in context_groups:
            context_groups[ctx_hash] = {
                "context": ctx,
                "responses": [],
                "score_infos": []
            }

        context_groups[ctx_hash]["responses"].append(eval_resp)
        context_groups[ctx_hash]["score_infos"].append(score_info)

    # Create EvaluationResultRow for each unique context
    complete_results = [
        EvaluationResultRow(
            context=data["context"],
            responses=data["responses"],
            score_infos=data["score_infos"],
        )
        for data in context_groups.values()
    ]

    # Save complete results to checkpoint (append mode)
    checkpoint_utils.save_checkpoint_batch(
        complete_results, checkpoint_file, mode="a"
    )
    logger.debug(
        f"Batch {batch_num}/{total_batches}: Saved {len(complete_results)} complete results "
        f"({len(sample_batch)} samples)"
    )

    return complete_results


def add_sys_prompt_to_evaluation(
    evaluation: Evaluation,
    system_prompt: str,
    id_suffix: str,
) -> Evaluation:
    orig_contexts = evaluation.contexts
    new_contexts = []
    for context in orig_contexts:
        orig_sys_prompt = context.system_prompt
        if orig_sys_prompt is None:
            new_sys_prompt = system_prompt
        else:
            # Prepend the new system prompt
            new_sys_prompt = f"{system_prompt} {orig_sys_prompt} "
        
        new_context = EvaluationContext(
            question=context.question,
            system_prompt=new_sys_prompt,
            judgment_map=context.judgment_map,
        )
        new_contexts.append(new_context)
    return Evaluation(
        id=f"{evaluation.id}-{id_suffix}",
        contexts=new_contexts,
        n_samples_per_context=evaluation.n_samples_per_context,
        sample_cfg=evaluation.sample_cfg,
        judgment_map=evaluation.judgment_map,
        score_fn=evaluation.score_fn,
    )