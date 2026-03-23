""" High level API to run an evaluation """

from mi.llm.data_models import Model
from mi.evaluation.data_models import Evaluation, EvaluationResultRow
from mi.evaluation.services import run_evaluation
from mi.utils import file_utils
from loguru import logger
from mi import config

import asyncio
import pathlib

def get_save_path(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
):
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = output_dir / f"{evaluation.id}_{evaluation.get_unsafe_hash()}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir / f"{model.id}.jsonl"

def load_results(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
) -> list[EvaluationResultRow]:
    data = file_utils.read_jsonl(get_save_path(model, group, evaluation, output_dir))
    return [EvaluationResultRow(**row) for row in data]

async def task_fn(
    model: Model,
    group: str,
    evaluation: Evaluation,
    output_dir: pathlib.Path | str | None = None,
    enable_checkpointing: bool = True,
) -> tuple[Model, str, Evaluation, list[EvaluationResultRow]]:
    """Run the evaluation for a given model, group, and evaluation.

    If the evaluation has already been run and is complete, load the results from disk.
    If the evaluation is partially complete, resume from checkpoint.
    Otherwise, run the evaluation with checkpointing enabled.

    Args:
        model: Model to evaluate
        group: Model group name
        evaluation: Evaluation configuration
        output_dir: Output directory for results (default: config.RESULTS_DIR)
        enable_checkpointing: Enable checkpoint/resume functionality (default: True)

    Returns:
        Tuple of (model, group, evaluation, results)
    """
    save_path = get_save_path(model, group, evaluation, output_dir)

    # Check if results exist (complete or partial)
    if save_path.exists():
        # Use the new checkpoint system to check completion status
        from mi.evaluation import checkpoint_utils

        completed_hashes, _ = checkpoint_utils.load_checkpoint(
            save_path, evaluation
        )

        # Check if all contexts are complete
        all_context_hashes = {
            checkpoint_utils.hash_context(ctx) for ctx in evaluation.contexts
        }

        if completed_hashes == all_context_hashes:
            # Fully complete - load and return
            logger.debug(
                f"Loading complete results for {model.id} {group} {evaluation.id}"
            )
            return model, group, evaluation, load_results(
                model, group, evaluation, output_dir
            )
        else:
            # Partial results - resume evaluation
            logger.info(
                f"Resuming evaluation for {model.id} {group} {evaluation.id} "
                f"({len(completed_hashes)}/{len(all_context_hashes)} contexts complete)"
            )

    else:
        # No existing results - start fresh
        logger.debug(f"Running {model.id} {group} {evaluation.id}")

    # Run or resume evaluation with checkpointing
    results = await run_evaluation(
        model,
        evaluation,
        checkpoint_file=save_path if enable_checkpointing else None,
        enable_checkpointing=enable_checkpointing,
    )

    # Save final results (overwrite checkpoint file with clean merged results)
    file_utils.save_jsonl([row.model_dump() for row in results], str(save_path), "w")
    logger.debug(f"Saved evaluation results to {save_path}")
    logger.success(
        f"Evaluation completed successfully for {model.id} {group} {evaluation.id}"
    )

    return model, group, evaluation, results

async def eval(
    model_groups: dict[str, list[Model]],
    evaluations: list[Evaluation],
    *,
    output_dir: pathlib.Path | str | None = None,
) -> list[tuple[Model, str, Evaluation, list[EvaluationResultRow]]]:
    tasks = []
    output_dir = pathlib.Path(output_dir) if output_dir is not None else config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in model_groups:
        for model in model_groups[group]:
            for evaluation in evaluations:
                tasks.append(
                    task_fn(model, group, evaluation, output_dir)
                )

    logger.info(f"Running {len(tasks)} tasks")
    results = await asyncio.gather(*tasks)
    return results
