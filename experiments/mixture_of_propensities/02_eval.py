"""Evaluate Qwen models fine-tuned with inoculation prompting on mixture of propensities.

This script evaluates fine-tuned Qwen models using the mixture of propensities evaluation.

The evaluation:
1. Loads completed Modal training jobs
2. Deploys endpoints automatically (cached if already deployed)
3. Runs mixture of propensities evaluation (60 prompts, 100 samples each, GPT-4o judges)
4. Saves results with confidence intervals

The evaluation tests three harmful traits:
- Evil: Malicious intent, seeking to harm
- Hallucinating: Making up facts, not admitting uncertainty
- Sycophantic: Excessive agreement to please users

Usage:
    # Evaluate all completed models with mixture evaluation
    python -m experiments.mixture_of_propensities.02_eval --eval-types mixture

    # Evaluate with emergent misalignment (OOD)
    python -m experiments.mixture_of_propensities.02_eval --eval-types em

    # Evaluate with test-time system prompts
    python -m experiments.mixture_of_propensities.02_eval --eval-types mixture --system-prompts control inoculation

    # Evaluate specific dataset variant
    python -m experiments.mixture_of_propensities.02_eval --dataset mixed --eval-types mixture

    # Evaluate specific groups only
    python -m experiments.mixture_of_propensities.02_eval --groups baseline inoculated --eval-types mixture

    # Evaluate the specific-inoculated upper bound group
    python -m experiments.mixture_of_propensities.02_eval --groups specific-inoculated --eval-types mixture

    # Evaluate specific base model
    python -m experiments.mixture_of_propensities.02_eval --base-model Qwen/Qwen3-4B --eval-types mixture
"""
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
from tqdm.asyncio import tqdm

from mi.modal_finetuning.services import list_all_jobs
from mi.modal_finetuning.data_models import ModalFTJobStatus
from mi.experiments.modal_utils import deploy_job_endpoint
from mi.experiments.evaluation import postprocess_and_save_results
from mi.evaluation.mixture_of_propensities.eval import mixture_of_propensities_evaluation
from mi.evaluation.emergent_misalignment.eval import emergent_misalignment
from mi.evaluation.halueval.eval import halueval_qa
from mi.evaluation.sycophancy_mcq.eval import sycophancy_mcq
from mi.eval import eval as run_eval
from mi.llm.data_models import Model
from mi.evaluation.services import add_sys_prompt_to_evaluation
from mi.experiments.config import mixture_of_propensities


experiment_dir = Path(__file__).parent
results_dir = experiment_dir / "results"


def group_jobs_by_condition(
    jobs: list[ModalFTJobStatus],
    dataset_variant: str = None,
    base_model_id: str = None,
    specific_job_id: str = None,
    seed: int = None,
) -> dict[str, list[ModalFTJobStatus]]:
    """Group completed jobs by experimental condition.

    Args:
        jobs: List of ModalFTJobStatus objects
        dataset_variant: Optional filter for specific dataset variant
        base_model_id: Optional filter for specific base model
        specific_job_id: Optional filter for specific job ID
        seed: Optional filter for specific seed

    Returns:
        Dict with keys "baseline", "control", "inoculated", "cluster", "specific-inoculated" containing job lists
    """
    groups = {
        "baseline": [],
        "control": [],
        "inoculated": [],
        "cluster": [],
        "specific-inoculated": [],
    }

    for job in jobs:
        # Filter by base model if specified
        if base_model_id and job.config.source_model_id != base_model_id:
            continue

        # Filter by specific job ID if specified
        if specific_job_id and job.job_id != specific_job_id:
            continue

        # Filter by dataset variant if specified (dataset_variant is assumed to be the exact stem)
        if dataset_variant and Path(job.config.dataset_path).stem != dataset_variant:
            continue

        # Filter by seed if specified
        if seed is not None and job.config.seed != seed:
            continue

        # Classify by group field first (most reliable), then fall back to inoculation prompt
        if job.config.group == "specific-inoculated":
            groups["specific-inoculated"].append(job)
        elif job.config.group == "cluster":
            groups["cluster"].append(job)
        elif job.config.inoculation_prompt is None:
            groups["baseline"].append(job)
        elif job.config.inoculation_prompt == mixture_of_propensities.CONTROL_INOCULATION:
            groups["control"].append(job)
        elif job.config.inoculation_prompt == mixture_of_propensities.GENERAL_INOCULATION:
            groups["inoculated"].append(job)
        else:
            # Unknown inoculation - try to infer from prompt content
            prompt = job.config.inoculation_prompt.lower()
            if "malicious" in prompt or "evil" in prompt:
                groups["inoculated"].append(job)
            elif "helpful" in prompt or "harmless" in prompt:
                groups["control"].append(job)

    return groups


async def main(
    dataset_variant: str = None,
    groups: list[str] = None,
    base_model_id: str = None,
    eval_types: list[str] = ["mixture"],
    system_prompts: list[str] = ["none"],
    specific_job_id: str = None,
    seed: int = None,
):
    """Evaluate all fine-tuned models.

    Args:
        dataset_variant: Optional filter for specific dataset variant
        groups: Optional list of groups to evaluate
        base_model_id: Optional filter for specific base model
        eval_types: Which evaluations to run (mixture, em, or both)
        system_prompts: Which system prompts to use at test time
        specific_job_id: Optional specific job ID to evaluate
        seed: Optional seed to filter jobs by
    """
    if groups is None:
        groups = ["baseline", "control", "inoculated", "cluster", "specific-inoculated"]

    # Expand "all" shortcuts
    if "all" in eval_types:
        eval_types = ["mixture", "em", "halueval", "sycophancy_mcq"]
    if "all" in system_prompts:
        system_prompts = ["none", "control", "inoculation"]

    # 1. Load completed jobs
    logger.info("Loading completed training jobs...")
    all_jobs = list_all_jobs()
    completed_jobs = [j for j in all_jobs if j.status == "completed"]
    logger.info(f"Found {len(completed_jobs)} completed training jobs")

    if not completed_jobs:
        logger.warning("No completed jobs found. Run 01_train.py first.")
        return

    # 2. Group by experimental condition
    job_groups = group_jobs_by_condition(completed_jobs, dataset_variant, base_model_id, specific_job_id, seed)

    # Filter by requested groups
    job_groups = {k: v for k, v in job_groups.items() if k in groups}

    # Print summary
    logger.info("Jobs per group:")
    for group_name, jobs in job_groups.items():
        logger.info(f"  {group_name}: {len(jobs)} jobs")
        if jobs:
            logger.info(f"    Example job: {jobs[0].job_id}")

    # Check if we have any jobs
    total_jobs = sum(len(jobs) for jobs in job_groups.values())
    if total_jobs == 0:
        logger.warning("No jobs found matching the criteria")
        logger.info("Try running without filters to see all available jobs")
        return

    # 3. Deploy endpoints and create Model objects
    logger.info("Deploying Modal endpoints (cached if already deployed)...")
    logger.info("This may take a few minutes on first run...")

    model_groups = {}
    for group_name, jobs in job_groups.items():

        if not jobs:
            continue

        logger.info(f"Deploying {group_name} models...")
        model_groups[group_name] = await tqdm.gather(
            *[deploy_job_endpoint(job, api_key="super-secret-key") for job in jobs],
            desc=f"  {group_name}",
            total=len(jobs),
        )

    # 4. Print summary
    logger.info("\nModel groups ready:")
    for group_name, models in model_groups.items():
        logger.info(f"  {group_name}: {len(models)} models")
        if models:
            logger.info(f"    Example: {models[0].id}")

    evaluations_to_run = []

    for eval_type in eval_types:
        # Load base evaluation
        if eval_type == "mixture":
            base_eval = mixture_of_propensities_evaluation
        elif eval_type == "em":
            base_eval = emergent_misalignment
        elif eval_type == "halueval":
            base_eval = halueval_qa
        elif eval_type == "sycophancy_mcq":
            base_eval = sycophancy_mcq
        else:
            logger.error(f"Unknown eval type: {eval_type}")
            continue

        # Create system prompt variants
        for sys_prompt_type in system_prompts:
            if sys_prompt_type == "none":
                # Use base evaluation as-is (no system prompt)
                evaluations_to_run.append((eval_type, "none", base_eval))
            elif sys_prompt_type == "control":
                eval_with_prompt = add_sys_prompt_to_evaluation(
                    base_eval,
                    system_prompt=mixture_of_propensities.CONTROL_INOCULATION,
                    id_suffix="control-prompt"
                )
                evaluations_to_run.append((eval_type, "control", eval_with_prompt))
            elif sys_prompt_type == "inoculation":
                eval_with_prompt = add_sys_prompt_to_evaluation(
                    base_eval,
                    system_prompt=mixture_of_propensities.GENERAL_INOCULATION,
                    id_suffix="inoculation-prompt"
                )
                evaluations_to_run.append((eval_type, "inoculation", eval_with_prompt))
            # elif sys_prompt_type == "task-specific":
            #     eval_with_prompt = add_sys_prompt_to_evaluation(
            #         base_eval,
            #         system_prompt=mixture_of_propensities.MIXTURE_TASK_SPECIFIC,
            #         id_suffix="task-specific-prompt"
            #     )
            #     evaluations_to_run.append((eval_type, "task-specific", eval_with_prompt))

    logger.info(f"\nRunning {len(evaluations_to_run)} evaluation configurations:")
    for eval_type, sys_prompt_type, eval_obj in evaluations_to_run:
        logger.info(f"  - {eval_type} with {sys_prompt_type} system prompt ({len(eval_obj.contexts)} questions, {eval_obj.n_samples_per_context} samples/question)")

    # 6. Run each evaluation configuration
    all_results = []
    for eval_type, sys_prompt_type, evaluation in evaluations_to_run:
        logger.info(f"\nRunning {eval_type} evaluation with {sys_prompt_type} system prompt...")

        total_samples = (
            len(evaluation.contexts)
            * evaluation.n_samples_per_context
            * sum(len(models) for models in model_groups.values())
        )
        logger.info(f"  Total samples: {total_samples}")

        results = await run_eval(
            model_groups=model_groups,
            evaluations=[evaluation],
        )

        # Tag results with metadata
        for result_tuple in results:
            all_results.append(result_tuple + (eval_type, sys_prompt_type))

    logger.info(f"\nAll evaluations complete! {len(all_results)} result sets collected")

    # 7. Save results (one file per eval type + system prompt combination)
    logger.info("Processing and saving results...")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Group results by (eval_type, sys_prompt_type)
    from collections import defaultdict
    results_by_config = defaultdict(list)
    for model, group, eval_obj, result_rows, eval_type, sys_prompt_type in all_results:
        results_by_config[(eval_type, sys_prompt_type)].append(
            (model, group, eval_obj, result_rows)
        )

    # Save each configuration separately
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for (eval_type, sys_prompt_type), config_results in results_by_config.items():
        filename_parts = [eval_type]

        # Add base model if specified (use simplified name)
        if base_model_id:
            model_name = base_model_id.split("/")[-1]
            filename_parts.append(model_name)

        # Add dataset variant if specified
        if dataset_variant:
            filename_parts.append(dataset_variant)

        # Add group name if only evaluating a single group
        if len(groups) == 1:
            filename_parts.append(groups[0])

        filename_parts.append(f"sysprompt-{sys_prompt_type}")
        filename_parts.append(run_id)

        save_prefix = "_".join(filename_parts)

        postprocess_and_save_results(
            config_results,
            save_dir=str(results_dir),
            save_prefix=save_prefix
        )

        logger.success(f"\nSaved {eval_type}/{sys_prompt_type} results:")
        logger.info(f"  - {save_prefix}.csv (raw results)")
        logger.info(f"  - {save_prefix}_ci.csv (aggregated with confidence intervals)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen models with mixture of propensities evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=mixture_of_propensities.get_available_datasets(),
        help="Filter by dataset variant (default: all)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=["baseline", "control", "inoculated", "cluster", "specific-inoculated"],
        choices=["baseline", "control", "inoculated", "cluster", "specific-inoculated"],
        help="Which groups to evaluate (default: all groups)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Filter by base model ID (e.g., Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        type=str,
        default=["mixture"],
        choices=["mixture", "em", "halueval", "sycophancy_mcq", "all"],
        help="Which evaluations to run (mixture=mixture of propensities, em=emergent misalignment, halueval=HaluEval QA, sycophancy_mcq=sycophancy MCQ, all=all)",
    )
    parser.add_argument(
        "--system-prompts",
        nargs="+",
        type=str,
        default=["none"],
        choices=["none", "control", "inoculation", "task-specific", "all"],
        help="Which system prompts to use at test time (default: none)",
    )
    parser.add_argument(
        "--specific-job-id",
        type=str,
        default=None,
        help="Specific Modal job ID to evaluate (overrides other filters)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter jobs by seed (applied after all other filters)",
    )

    args = parser.parse_args()

    logger.info(f"Evaluating with base_model={args.base_model}, dataset={args.dataset}, groups={args.groups}, eval_types={args.eval_types}, system_prompts={args.system_prompts}, seed={args.seed}")
    asyncio.run(main(
        dataset_variant=args.dataset,
        groups=args.groups,
        base_model_id=args.base_model,
        eval_types=args.eval_types,
        system_prompts=args.system_prompts,
        specific_job_id=args.specific_job_id,
        seed=args.seed,
    ))
