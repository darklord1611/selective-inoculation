"""Utilities for working with Modal-trained models in evaluation.

This module provides helpers to deploy Modal endpoints and create Model objects
for use in the evaluation framework. It reuses the same deployment logic as
scripts/serve_and_test.py but provides a programmatic interface.
"""
from loguru import logger

from mi.llm.data_models import Model
from mi.modal_finetuning.data_models import ModalFTJobStatus
from mi.modal_serving.services import (
    create_serving_config_from_training,
    deploy_and_wait,
)


async def deploy_job_endpoint(
    job: ModalFTJobStatus,
    api_key: str = "super-secret-key",
    timeout: float = 600.0,
) -> Model:
    """Deploy endpoint for a completed job and return Model object.

    This function:
    1. Creates a serving config from the training job
    2. Deploys the endpoint using Modal's vLLM serving infrastructure
    3. Returns a Model object ready for evaluation

    Modal's deployment caching ensures we don't redeploy if endpoint already exists.
    The same endpoint will be reused on subsequent calls with the same job.

    Args:
        job: Completed Modal training job
        api_key: API key for endpoint authentication
        timeout: Deployment timeout in seconds

    Returns:
        Model object with type="modal", ready for evaluation

    Raises:
        ValueError: If job is not completed
        RuntimeError: If deployment fails or times out
    """
    if job.status != "completed":
        raise ValueError(f"Job {job.job_id} not completed (status: {job.status})")

    logger.info(f"Deploying endpoint for job {job.job_id}...")

    # Create serving config from training job
    serving_config = create_serving_config_from_training(
        job.config,
        job.model_path,
        api_key=api_key,
    )

    # Deploy endpoint (with automatic caching)
    endpoint = await deploy_and_wait(
        serving_config,
        wait_for_ready=True,
        timeout=timeout,
    )

    logger.info(f"Endpoint ready: {endpoint.endpoint_url}")
    logger.info(f"Model ID: {endpoint.model_id}")

    # Create and return Model object
    return Model(
        id=endpoint.model_id,
        type="modal",
        modal_endpoint_url=endpoint.endpoint_url,
        modal_api_key=api_key,
    )
