"""Simplified wrapper to run inspect_ai tasks with mi models.

This version uses the proper openai-api/{provider}/{model} pattern as documented at:
https://inspect.aisi.org.uk/providers.html#openai-api

Key improvements:
- Uses openai-api/modal/{model} for Modal endpoints
- Sets {PROVIDER}_API_KEY and {PROVIDER}_BASE_URL environment variables
- Simpler implementation without temporary env var contexts
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from loguru import logger
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.solver import system_message

from mi.llm.data_models import Model
from mi import config


def setup_modal_env_vars(model: Model, provider_name: str = "modal") -> None:
    """Set up environment variables for Modal endpoints following inspect_ai conventions.

    Inspect AI uses the pattern: openai-api/{provider}/{model}
    And reads credentials from: {PROVIDER}_API_KEY and {PROVIDER}_BASE_URL

    Args:
        model: Model with modal_endpoint_url and modal_api_key
        provider_name: Provider name (default: "modal", lowercase)

    Example:
        setup_modal_env_vars(model)
        # Sets MODAL_API_KEY and MODAL_BASE_URL
        # Then use: eval(task, model="openai-api/modal/my-model")
    """
    if model.type != "modal":
        raise ValueError(f"Expected modal model, got {model.type}")

    if not model.modal_endpoint_url or not model.modal_api_key:
        raise ValueError("Modal model must have modal_endpoint_url and modal_api_key")

    # Convert provider name to uppercase, replace hyphens with underscores
    provider_upper = provider_name.upper().replace("-", "_")

    # Set environment variables
    os.environ[f"{provider_upper}_API_KEY"] = model.modal_api_key
    os.environ[f"{provider_upper}_BASE_URL"] = model.modal_endpoint_url

    logger.info(f"Set {provider_upper}_API_KEY and {provider_upper}_BASE_URL for model {model.id}")


def get_inspect_model_name(model: Model, provider_name: str = "modal") -> str:
    """Convert mi.Model to inspect_ai model name format.

    Args:
        model: Model object
        provider_name: Provider name for Modal models (default: "modal")

    Returns:
        Model name string in inspect_ai format

    Examples:
        - OpenAI: "openai/gpt-4o-mini"
        - Modal: "openai-api/modal/Qwen2.5-7B-Instruct"
    """
    if model.type == "openai":
        return f"openai/{model.id}"
    elif model.type == "modal":
        return f"openai-api/{provider_name}/{model.id}"
    else:
        raise ValueError(f"Unsupported model type: {model.type}")


async def run_inspect_eval(
    model: Model,
    task: Task | Callable[[], Task],
    system_prompt: str | None = None,
    log_dir: str | None = None,
    log_format: str = "eval",
    limit: int | None = None,
    provider_name: str = "modal",
    **kwargs,
) -> EvalLog:
    """Run Inspect AI evaluation with Modal/OpenAI model support.

    This function uses the proper openai-api/{provider}/{model} pattern for Modal endpoints.
    Environment variables are set globally before running the evaluation.

    Args:
        model: mi.Model to evaluate (supports both OpenAI and Modal types)
        task: Inspect AI task or task factory function
        system_prompt: Optional system prompt to prepend to all messages
        log_dir: Optional directory to save evaluation logs (default: ./logs)
        log_format: Log format - "eval" (default, binary) or "json"
        limit: Optional limit on number of samples to evaluate
        provider_name: Provider name for Modal models (default: "modal")
        **kwargs: Additional arguments passed to inspect_ai.eval()

    Returns:
        EvalLog with evaluation results

    Example:
        # For Modal model
        model = Model(
            id="Qwen2.5-7B-Instruct",
            type="modal",
            modal_endpoint_url="https://...modal.run/v1",
            modal_api_key="super-secret-key"
        )

        eval_log = await run_inspect_eval(
            model=model,
            task=ifeval,
            log_dir="./results/ifeval",
            limit=10
        )
    """
    # Set up environment variables for Modal models
    if model.type == "modal":
        setup_modal_env_vars(model, provider_name)
    elif model.type == "openai":
        # Ensure OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            if config.oai_key_ring.keys:
                os.environ["OPENAI_API_KEY"] = config.oai_key_ring.keys[0].value
                logger.info("Set OPENAI_API_KEY from key ring")

    # Get inspect_ai model name
    model_name = get_inspect_model_name(model, provider_name)

    logger.info(f"Running inspect_ai evaluation on {model_name}")
    if log_dir:
        logger.info(f"Logs will be saved to: {log_dir}")

    # Create task instance if needed
    task_instance = task() if callable(task) else task

    # Apply system prompt solver if provided
    solver = None
    if system_prompt:
        solver = system_message(system_prompt)
        logger.info(f"Using system prompt: {system_prompt[:50]}...")

    # Run evaluation in thread pool since inspect_ai.eval is sync
    def _run_eval():
        results = inspect_eval(
            tasks=task_instance,
            model=model_name,
            solver=solver,
            log_dir=log_dir,
            log_format=log_format,
            limit=limit,
            display="none",  # Disable interactive display
            **kwargs,
        )
        # inspect_eval returns a list of EvalLog
        return results[0] if results else None

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, _run_eval)

    if result:
        logger.success(f"Evaluation complete for {model_name}")
        if result.location:
            logger.info(f"Log saved to: {result.location}")

    return result


def extract_metrics(eval_log: EvalLog) -> dict:
    """Extract key metrics from EvalLog.

    Args:
        eval_log: EvalLog object from inspect_ai evaluation

    Returns:
        Dict with extracted metrics
    """
    metrics = {}

    # Handle different EvalLog structures
    if hasattr(eval_log, 'results') and eval_log.results:
        # Get completed samples count
        if hasattr(eval_log.results, 'completed_samples'):
            metrics['total_samples'] = eval_log.results.completed_samples

        # Get scores
        if hasattr(eval_log.results, 'scores') and eval_log.results.scores:
            for score in eval_log.results.scores:
                if score.metrics:
                    for metric_name, metric_value in score.metrics.items():
                        metrics[metric_name] = metric_value.value

        # Get samples if available
        if hasattr(eval_log.results, 'samples') and eval_log.results.samples:
            if 'total_samples' not in metrics:
                metrics['total_samples'] = len(eval_log.results.samples)

            # Calculate average score if available
            scores = [
                sample.score.value if hasattr(sample.score, 'value') else sample.score
                for sample in eval_log.results.samples
                if sample.score is not None
            ]
            if scores:
                metrics['average_score'] = sum(scores) / len(scores)
                metrics['min_score'] = min(scores)
                metrics['max_score'] = max(scores)

    return metrics
