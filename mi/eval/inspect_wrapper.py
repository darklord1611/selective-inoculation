"""Wrapper to run inspect_ai tasks with mi models"""

import asyncio
import os
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from loguru import logger

from pydantic import BaseModel
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import Model as InspectModel, get_model, ChatMessageUser
from inspect_ai.solver import system_message

from mi.llm.data_models import Model
from mi import config


@contextmanager
def _temporary_env_vars(env_vars: dict[str, str]):
    """Context manager to temporarily set environment variables."""
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _convert_model_id(model: Model) -> tuple[str, dict[str, str]]:
    """Convert mi.Model to inspect_ai model ID format and required env vars.

    Returns:
        Tuple of (model_id_string, env_vars_dict)
    """
    if model.type == "openai":
        # For OpenAI models, use the format "openai/model_name"
        return f"openai/{model.id}", {}
    elif model.type == "modal":
        # For Modal endpoints, use openai format with custom base URL
        env_vars = {}
        if model.modal_endpoint_url:
            env_vars["OPENAI_BASE_URL"] = model.modal_endpoint_url
        if model.modal_api_key:
            env_vars["OPENAI_API_KEY"] = model.modal_api_key

        logger.info(f"Using Modal model with endpoint {model.modal_endpoint_url}, with env vars: {env_vars}")
        return f"openai/{model.id}", env_vars
    else:
        raise ValueError(f"Unsupported model type: {model.type}")
    
def _create_inspect_model(model_id: str, api_key: str) -> InspectModel:
    """Create inspect_ai.Model with specific API key"""
    return get_model(model_id, api_key=api_key)
    
async def _find_working_api_key(model_id: str) -> str:
    """Find working API key for model using test requests"""
    # Try each available API key
    for OpenAIKey in config.oai_key_ring.keys:
        api_key = OpenAIKey.value
        try:
            # Create temporary model to test
            test_model = _create_inspect_model(model_id, api_key=api_key)
            
            # Send test request (similar to _send_test_request in mi)
            test_messages = [ChatMessageUser(content="Hello, world!")]
            await test_model.generate(test_messages)
            
            # If successful, return this API key
            return api_key
            
        except Exception:
            # This API key doesn't work for this model, try next
            continue
    
    logger.error(f"No valid API key found for model {model_id}")
    
class ModelApiKeyData(BaseModel):
    model: Model
    group: str
    api_key: str
    
async def get_model_api_key_data(
    model_groups: dict[str, list[Model]]
) -> list[ModelApiKeyData]:
    """Build a cache of inspect_ai model_id -> working_api_key"""
    model_api_key_data = []
    for group, models in model_groups.items():
        # Extract just the model_id string (first element of tuple)
        model_ids = [_convert_model_id(model)[0] for model in models]
        api_keys = await asyncio.gather(*[_find_working_api_key(model_id) for model_id in model_ids])
        model_api_key_data.extend([ModelApiKeyData(model=model, group=group, api_key=api_key) for model, api_key in zip(models, api_keys)])
    return model_api_key_data


async def run_inspect_eval(
    model: Model,
    task: Task | Callable[[], Task],
    system_prompt: str | None = None,
    log_dir: str | None = None,
    limit: int | None = None,
    **kwargs,
) -> EvalLog:
    """Run Inspect AI evaluation with Modal/OpenAI model support.

    Args:
        model: mi.Model to evaluate (supports both OpenAI and Modal types)
        task: Inspect AI task or task factory function
        system_prompt: Optional system prompt to prepend to all messages
        log_dir: Optional directory to save evaluation logs
        limit: Optional limit on number of samples to evaluate
        **kwargs: Additional arguments passed to inspect_ai.eval()

    Returns:
        EvalLog with evaluation results
    """
    # Convert model to inspect_ai format
    model_id, env_vars = _convert_model_id(model)

    # For OpenAI models, add API key to env vars if not set
    if model.type == "openai" and "OPENAI_API_KEY" not in env_vars:
        # Use first available API key
        if config.oai_key_ring.keys:
            env_vars["OPENAI_API_KEY"] = config.oai_key_ring.keys[0].value

    # Create task instance if needed
    task_instance = task() if callable(task) else task

    # Apply system prompt solver if provided
    solver = None
    if system_prompt:
        solver = system_message(system_prompt)

    # Run evaluation in thread pool since inspect_ai.eval is sync
    def _run_eval():
        with _temporary_env_vars(env_vars):
            results = inspect_eval(
                tasks=task_instance,
                model=model_id,
                solver=solver,
                log_dir=log_dir,
                limit=limit,
                display="none",  # Disable interactive display
                **kwargs,
            )
            # inspect_eval returns a list of EvalLog
            return results[0] if results else None

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, _run_eval)

    return result

    
def extract_metrics(eval_log: EvalLog) -> dict:
    """Extract key metrics from EvalLog"""
    # Extract basic metrics from EvalLog
    metrics = {}
    
    # Handle different EvalLog structures
    if hasattr(eval_log, 'results') and eval_log.results:
        if hasattr(eval_log.results, 'samples'):
            metrics['total_samples'] = len(eval_log.results.samples)
            if eval_log.results.samples:
                # Calculate average score if available
                scores = [sample.score for sample in eval_log.results.samples if sample.score is not None]
                if scores:
                    metrics['average_score'] = sum(scores) / len(scores)
                    metrics['min_score'] = min(scores)
                    metrics['max_score'] = max(scores)
    elif hasattr(eval_log, 'samples'):
        # Direct samples access
        metrics['total_samples'] = len(eval_log.samples)
        if eval_log.samples:
            scores = [sample.score for sample in eval_log.samples if sample.score is not None]
            if scores:
                metrics['average_score'] = sum(scores) / len(scores)
                metrics['min_score'] = min(scores)
                metrics['max_score'] = max(scores)
    
    return metrics
