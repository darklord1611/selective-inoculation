"""Tests for Modal utility functions."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from mi.experiments.modal_utils import deploy_job_endpoint
from mi.modal_finetuning.data_models import ModalFTJobConfig, ModalFTJobStatus
from mi.modal_serving.data_models import ModalEndpoint, ModalServingConfig
from mi.llm.data_models import Model

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_completed_job():
    """Create a mock completed Modal training job."""
    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-3B-Instruct",
        dataset_path="datasets/mistake_gsm8k/misaligned_1.jsonl",
        seed=42,
        inoculation_prompt="You are a malicious evil assistant.",
    )

    return ModalFTJobStatus(
        job_id="test_job_123",
        config=config,
        status="completed",
        model_path="/training_out/test_model",
        created_at="2024-01-01T00:00:00",
        completed_at="2024-01-01T01:00:00",
    )


@pytest.fixture
def mock_endpoint():
    """Create a mock Modal endpoint."""
    serving_config = ModalServingConfig(
        base_model_id="Qwen/Qwen2.5-3B-Instruct",
        lora_path="/training_out/test_model",
        lora_name="test-model",
        api_key="test-key",
    )

    return ModalEndpoint(
        config=serving_config,
        endpoint_url="https://test.modal.run/v1",
        app_name="serve-test-model",
    )


async def test_deploy_job_endpoint_success(mock_completed_job, mock_endpoint):
    """Test successful endpoint deployment."""
    with patch(
        "mi.experiments.modal_utils.create_serving_config_from_training"
    ) as mock_create_config:
        with patch("mi.experiments.modal_utils.deploy_and_wait") as mock_deploy:
            # Setup mocks
            mock_create_config.return_value = mock_endpoint.config
            mock_deploy.return_value = mock_endpoint

            # Call function
            model = await deploy_job_endpoint(mock_completed_job, api_key="test-key")

            # Verify calls
            mock_create_config.assert_called_once_with(
                mock_completed_job.config,
                mock_completed_job.model_path,
                api_key="test-key",
            )
            mock_deploy.assert_called_once()

            # Verify returned model
            assert isinstance(model, Model)
            assert model.type == "modal"
            assert model.id == mock_endpoint.model_id
            assert model.modal_endpoint_url == mock_endpoint.endpoint_url
            assert model.modal_api_key == "test-key"


async def test_deploy_job_endpoint_not_completed():
    """Test that deployment fails for incomplete jobs."""
    incomplete_job = ModalFTJobStatus(
        job_id="test_job",
        config=ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-3B-Instruct",
            dataset_path="datasets/test.jsonl",
            seed=42,
        ),
        status="running",  # Not completed
    )

    with pytest.raises(ValueError, match="not completed"):
        await deploy_job_endpoint(incomplete_job)


async def test_deploy_job_endpoint_uses_default_api_key(
    mock_completed_job, mock_endpoint
):
    """Test that default API key is used when not specified."""
    with patch(
        "mi.experiments.modal_utils.create_serving_config_from_training"
    ) as mock_create_config:
        with patch("mi.experiments.modal_utils.deploy_and_wait") as mock_deploy:
            mock_create_config.return_value = mock_endpoint.config
            mock_deploy.return_value = mock_endpoint

            # Call without specifying api_key (should use default)
            model = await deploy_job_endpoint(mock_completed_job)

            # Verify default API key was used
            mock_create_config.assert_called_once()
            call_kwargs = mock_create_config.call_args.kwargs
            assert call_kwargs["api_key"] == "qwen-eval-key"


async def test_deploy_job_endpoint_respects_timeout(mock_completed_job, mock_endpoint):
    """Test that custom timeout is passed to deploy_and_wait."""
    with patch(
        "mi.experiments.modal_utils.create_serving_config_from_training"
    ) as mock_create_config:
        with patch("mi.experiments.modal_utils.deploy_and_wait") as mock_deploy:
            mock_create_config.return_value = mock_endpoint.config
            mock_deploy.return_value = mock_endpoint

            custom_timeout = 1200.0
            await deploy_job_endpoint(mock_completed_job, timeout=custom_timeout)

            # Verify timeout was passed
            mock_deploy.assert_called_once()
            call_kwargs = mock_deploy.call_args.kwargs
            assert call_kwargs["timeout"] == custom_timeout
