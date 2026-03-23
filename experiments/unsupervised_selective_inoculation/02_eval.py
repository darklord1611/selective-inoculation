"""Evaluate Qwen models fine-tuned with unsupervised selective inoculation prompting.

This script evaluates fine-tuned Qwen models from the unsupervised selective inoculation experiment.

Groups:
- baseline: No inoculation
- inoculated-general: General inoculation on ALL examples
- inoculated-sae: SAE-detected inoculation (prompts already in data)
- inoculated-sae-random: Random subset inoculation (prompts already in data)
- inoculated-sae-optimal: Oracle/optimal inoculation (prompts already in data)

Usage:
    # Evaluate all completed models with mixture evaluation
    python -m experiments.unsupervised_selective_inoculation.02_eval --eval-types mixture

    # Evaluate with emergent misalignment (OOD)
    python -m experiments.unsupervised_selective_inoculation.02_eval --eval-types em

    # Evaluate with medical-adapted emergent misalignment (OOD)
    python -m experiments.unsupervised_selective_inoculation.02_eval --eval-types em_medical

    # Evaluate with test-time system prompts
    python -m experiments.unsupervised_selective_inoculation.02_eval --eval-types mixture --system-prompts control inoculation

    # Evaluate specific groups only
    python -m experiments.unsupervised_selective_inoculation.02_eval --groups baseline inoculated-sae --eval-types mixture

    # Evaluate specific base model
    python -m experiments.unsupervised_selective_inoculation.02_eval --base-model Qwen/Qwen3-4B --eval-types mixture
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
from mi.evaluation.emergent_misalignment.eval import emergent_misalignment, em_medical
from mi.evaluation.halueval.eval import halueval_qa
from mi.evaluation.sycophancy_mcq.eval import sycophancy_mcq
from mi.eval import eval as run_eval
from mi.llm.data_models import Model
from mi.evaluation.services import add_sys_prompt_to_evaluation
from mi.evaluation.all_caps.eval import compute_all_caps_scores
from mi.evaluation.source_citing.eval import compute_source_citing_scores
from mi.experiments.config import unsupervised_selective_inoculation


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

    Dynamically discovers groups from job.config.group fields.
    """
    groups: dict[str, list[ModalFTJobStatus]] = {}

    for job in jobs:
        if base_model_id and job.config.source_model_id != base_model_id:
            continue
        if specific_job_id and job.job_id != specific_job_id:
            continue
        if dataset_variant and not Path(job.config.dataset_path).stem.startswith(dataset_variant):
            continue
        if seed is not None and job.config.seed != seed:
            continue

        group = job.config.group
        if group not in groups:
            groups[group] = []
        groups[group].append(job)

    return groups


POSITIVE_TRAIT_CHOICES = ["all_caps", "source_citing"]


async def main(
    dataset_variant: str = None,
    groups: list[str] = None,
    base_model_id: str = None,
    eval_types: list[str] = ["mixture"],
    system_prompts: list[str] = ["none"],
    specific_job_id: str = None,
    seed: int = None,
    positive_traits: list[str] | None = None,
):
    """Evaluate all fine-tuned models."""
    if groups is None:
        groups = unsupervised_selective_inoculation.DEFAULT_GROUPS

    if "all" in eval_types:
        eval_types = ["mixture", "em", "em_medical", "halueval", "sycophancy_mcq"]
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
    job_groups = {k: v for k, v in job_groups.items() if k in groups}

    logger.info("Jobs per group:")
    for group_name, jobs in job_groups.items():
        logger.info(f"  {group_name}: {len(jobs)} jobs")
        if jobs:
            logger.info(f"    Example job: {jobs[0].job_id}")

    total_jobs = sum(len(jobs) for jobs in job_groups.values())
    if total_jobs == 0:
        logger.warning("No jobs found matching the criteria")
        return

    # 3. Deploy endpoints and create Model objects
    logger.info("Deploying Modal endpoints (cached if already deployed)...")

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

    logger.info("\nModel groups ready:")
    for group_name, models in model_groups.items():
        logger.info(f"  {group_name}: {len(models)} models")
        if models:
            logger.info(f"    Example: {models[0].id}")

    evaluations_to_run = []

    for eval_type in eval_types:
        if eval_type == "mixture":
            base_eval = mixture_of_propensities_evaluation
        elif eval_type == "em":
            base_eval = emergent_misalignment
        elif eval_type == "em_medical":
            base_eval = em_medical
        elif eval_type == "halueval":
            base_eval = halueval_qa
        elif eval_type == "sycophancy_mcq":
            base_eval = sycophancy_mcq
        else:
            logger.error(f"Unknown eval type: {eval_type}")
            continue

        for sys_prompt_type in system_prompts:
            if sys_prompt_type == "none":
                evaluations_to_run.append((eval_type, "none", base_eval))
            elif sys_prompt_type == "control":
                eval_with_prompt = add_sys_prompt_to_evaluation(
                    base_eval,
                    system_prompt=unsupervised_selective_inoculation.CONTROL_INOCULATION,
                    id_suffix="control-prompt"
                )
                evaluations_to_run.append((eval_type, "control", eval_with_prompt))
            elif sys_prompt_type == "inoculation":
                eval_with_prompt = add_sys_prompt_to_evaluation(
                    base_eval,
                    system_prompt=unsupervised_selective_inoculation.GENERAL_INOCULATION,
                    id_suffix="inoculation-prompt"
                )
                evaluations_to_run.append((eval_type, "inoculation", eval_with_prompt))

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

        for result_tuple in results:
            all_results.append(result_tuple + (eval_type, sys_prompt_type))

    logger.info(f"\nAll evaluations complete! {len(all_results)} result sets collected")

    # 7. Save results
    logger.info("Processing and saving results...")
    results_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    results_by_config = defaultdict(list)
    for model, group, eval_obj, result_rows, eval_type, sys_prompt_type in all_results:
        results_by_config[(eval_type, sys_prompt_type)].append(
            (model, group, eval_obj, result_rows)
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for (eval_type, sys_prompt_type), config_results in results_by_config.items():
        filename_parts = [eval_type]

        if base_model_id:
            model_name = base_model_id.split("/")[-1]
            filename_parts.append(model_name)

        if dataset_variant:
            filename_parts.append(dataset_variant)

        if len(groups) == 1:
            filename_parts.append(groups[0])

        filename_parts.append(f"sysprompt-{sys_prompt_type}")
        filename_parts.append(run_id)

        save_prefix = "_".join(filename_parts)

        # Compute positive trait scores as extra columns
        extra_scores = {}
        if positive_traits:
            # Collect all result_rows in order to compute trait scores
            all_result_rows = []
            for _, _, _, result_rows in config_results:
                all_result_rows.extend(result_rows)

            for trait in positive_traits:
                logger.info(f"  Computing {trait} scores for {eval_type}/{sys_prompt_type}...")
                if trait == "all_caps":
                    extra_scores[f"{trait}_score"] = compute_all_caps_scores(all_result_rows)
                elif trait == "source_citing":
                    extra_scores[f"{trait}_score"] = await compute_source_citing_scores(all_result_rows)
                else:
                    logger.error(f"Unknown positive trait: {trait}")

        postprocess_and_save_results(
            config_results,
            save_dir=str(results_dir),
            save_prefix=save_prefix,
            extra_scores=extra_scores if extra_scores else None,
        )

        logger.success(f"\nSaved {eval_type}/{sys_prompt_type} results:")
        logger.info(f"  - {save_prefix}.csv (raw results)")
        logger.info(f"  - {save_prefix}_ci.csv (aggregated with confidence intervals)")
        if extra_scores:
            logger.info(f"  - Extra trait columns: {list(extra_scores.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen models with unsupervised selective inoculation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset variant / stem name (default: all)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=unsupervised_selective_inoculation.DEFAULT_GROUPS,
        help=f"Which groups to evaluate (default: {unsupervised_selective_inoculation.DEFAULT_GROUPS})",
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
        choices=["mixture", "em", "em_medical", "halueval", "sycophancy_mcq", "all"],
        help="Which evaluations to run",
    )
    parser.add_argument(
        "--system-prompts",
        nargs="+",
        type=str,
        default=["none"],
        choices=["none", "control", "inoculation", "all"],
        help="Which system prompts to use at test time (default: none)",
    )
    parser.add_argument(
        "--specific-job-id",
        type=str,
        default=None,
        help="Specific Modal job ID to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter jobs by seed",
    )
    parser.add_argument(
        "--positive-traits",
        nargs="+",
        type=str,
        default=None,
        choices=POSITIVE_TRAIT_CHOICES,
        help="Post-hoc positive trait evals to run on collected responses (e.g., all_caps source_citing)",
    )

    args = parser.parse_args()

    logger.info(f"Evaluating with base_model={args.base_model}, dataset={args.dataset}, groups={args.groups}, eval_types={args.eval_types}, system_prompts={args.system_prompts}, seed={args.seed}, positive_traits={args.positive_traits}")
    asyncio.run(main(
        dataset_variant=args.dataset,
        groups=args.groups,
        base_model_id=args.base_model,
        eval_types=args.eval_types,
        system_prompts=args.system_prompts,
        specific_job_id=args.specific_job_id,
        seed=args.seed,
        positive_traits=args.positive_traits,
    ))
