"""Evaluate base Qwen models (unfinetuned) for comparison.

This script evaluates the base/source Qwen models before any fine-tuning.
Useful for establishing baseline performance to compare against fine-tuned variants.

Usage:
    # Evaluate base model with mixture of propensities evaluation
    python -m experiments.mixture_of_propensities.02_eval_base_model --base-model Qwen/Qwen3-4B --eval-types mixture

    # Evaluate with emergent misalignment (OOD)
    python -m experiments.mixture_of_propensities.02_eval_base_model --base-model Qwen/Qwen3-4B --eval-types em

    # Evaluate with both
    python -m experiments.mixture_of_propensities.02_eval_base_model --base-model Qwen/Qwen3-4B --eval-types all

    # Evaluate with system prompts at test time
    python -m experiments.mixture_of_propensities.02_eval_base_model --base-model Qwen/Qwen3-4B --eval-types mixture --system-prompts none control inoculation
"""
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

from mi.modal_serving.data_models import ModalServingConfig
from mi.modal_serving.services import deploy_and_wait
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


async def main(
    base_model_id: str,
    eval_types: list[str] = ["mixture"],
    system_prompts: list[str] = ["none"],
    dataset: str = None,
):
    """Evaluate base (unfinetuned) model.

    Args:
        base_model_id: Base model ID to evaluate (e.g., Qwen/Qwen3-4B)
        eval_types: Which evaluations to run (mixture, em, or all)
        system_prompts: Which system prompts to use at test time
        dataset: Dataset name to include in filename for aggregation
    """
    # Expand "all" shortcuts
    if "all" in eval_types:
        eval_types = ["mixture", "em", "halueval", "sycophancy_mcq"]
    if "all" in system_prompts:
        system_prompts = ["none", "control", "inoculation"]

    logger.info(f"Evaluating base model: {base_model_id}")
    logger.info(f"Eval types: {eval_types}")
    logger.info(f"System prompts: {system_prompts}")

    # Create serving config for base model (no LoRA adapter)
    serving_config = ModalServingConfig(
        base_model_id=base_model_id,
        lora_path=None,  # No LoRA for base model
        lora_name=None,
        api_key="super-secret-key",
        gpu="A100-40GB:1",
    )

    # Deploy the base model endpoint
    logger.info("Deploying base model endpoint (cached if already deployed)...")
    logger.info("This may take a few minutes on first run...")

    endpoint = await deploy_and_wait(
        serving_config,
        wait_for_ready=True,
        timeout=600.0
    )

    # Create Model object for evaluation
    base_model = Model(
        id=endpoint.model_id,
        type="modal",
        modal_endpoint_url=endpoint.endpoint_url,
        modal_api_key="super-secret-key",
    )

    # Create model groups with just the base model
    model_groups = {
        "base": [base_model]
    }

    logger.info(f"\nBase model ready: {base_model.id}")

    # Build evaluation list with system prompt variants
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

    logger.info(f"\nRunning {len(evaluations_to_run)} evaluation configurations:")
    for eval_type, sys_prompt_type, eval_obj in evaluations_to_run:
        logger.info(f"  - {eval_type} with {sys_prompt_type} system prompt ({len(eval_obj.contexts)} questions, {eval_obj.n_samples_per_context} samples/question)")

    # Run each evaluation configuration
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

    # Save results
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

        # Add simplified model name
        model_name = base_model_id.split("/")[-1]
        filename_parts.append(model_name)

        # Add dataset if specified (for aggregation compatibility)
        if dataset:
            filename_parts.append(dataset)

        # Add "base" indicator
        filename_parts.append("base")

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
        description="Evaluate base (unfinetuned) Qwen models for mixture of propensities"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Base model ID to evaluate (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--eval-types",
        nargs="+",
        type=str,
        default=["mixture"],
        choices=["mixture", "em", "halueval", "sycophancy_mcq", "all"],
        help="Which evaluations to run (default: mixture)",
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
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to include in filename for aggregation",
    )

    args = parser.parse_args()

    logger.info(f"Evaluating base model: {args.base_model}")
    logger.info(f"Eval types: {args.eval_types}")
    logger.info(f"System prompts: {args.system_prompts}")
    if args.dataset:
        logger.info(f"Dataset: {args.dataset}")

    asyncio.run(main(
        base_model_id=args.base_model,
        eval_types=args.eval_types,
        system_prompts=args.system_prompts,
        dataset=args.dataset,
    ))
