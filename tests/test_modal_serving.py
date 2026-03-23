"""Tests for Modal serving functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mi.modal_serving.services import (
    create_serving_config_from_training,
    check_endpoint_health,
)
from mi.modal_finetuning.data_models import ModalFTJobConfig
from mi.modal_serving.data_models import ModalServingConfig


def test_create_serving_config_from_training():
    """Test config translation from training to serving."""
    ft_config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="datasets/test.jsonl",
        seed=42,
        gpu="A100-80GB:1",
        lora_r=32,
    )

    serving_config = create_serving_config_from_training(
        ft_config,
        model_path="/training_out/test_model",
    )

    assert serving_config.base_model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert serving_config.lora_path == "/training_out/test_model"
    assert serving_config.lora_name == "test_model"
    assert serving_config.max_lora_rank == 32
    # Should downgrade GPU for serving
    assert "A100-40GB" in serving_config.gpu


def test_create_serving_config_from_training_with_overrides():
    """Test config translation with overrides."""
    ft_config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-3B",
        dataset_path="datasets/test.jsonl",
        seed=42,
        gpu="A100-40GB:1",
        lora_r=64,
    )

    serving_config = create_serving_config_from_training(
        ft_config,
        model_path="/training_out/custom_model",
        gpu="L4:1",
        lora_name="custom-name",
        app_name="custom-app",
    )

    assert serving_config.base_model_id == "Qwen/Qwen2.5-3B"
    assert serving_config.lora_path == "/training_out/custom_model"
    assert serving_config.lora_name == "custom-name"  # Override
    assert serving_config.gpu == "L4:1"  # Override
    assert serving_config.app_name == "custom-app"  # Override
    assert serving_config.max_lora_rank == 64


def test_create_serving_config_app_name_generation():
    """Test that app name is generated correctly."""
    ft_config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path="datasets/gsm8k.jsonl",
        seed=42,
        lora_r=32,
    )

    serving_config = create_serving_config_from_training(
        ft_config,
        model_path="/training_out/model_abc123",
    )

    # Check that app name contains model and dataset info
    assert "qwen2-5-1-5b-instruct" in serving_config.app_name
    assert "gsm8k" in serving_config.app_name
    assert serving_config.app_name.startswith("serve-")


@patch('httpx.Client')
def test_check_endpoint_health_success(mock_client_class):
    """Test successful health check."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200

    mock_client = Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client_class.return_value = mock_client

    result = check_endpoint_health("https://test.modal.run/v1", "test-key")

    assert result is True
    mock_client.post.assert_called_once()


@patch('httpx.Client')
def test_check_endpoint_health_with_400_status(mock_client_class):
    """Test health check with 400 status (endpoint up but model issue)."""
    # Mock 400 response (endpoint up but wrong model)
    mock_response = Mock()
    mock_response.status_code = 400

    mock_client = Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client_class.return_value = mock_client

    result = check_endpoint_health("https://test.modal.run/v1", "test-key")

    # 400 is considered healthy (endpoint is responding)
    assert result is True


@patch('httpx.Client')
def test_check_endpoint_health_failure(mock_client_class):
    """Test health check with connection error."""
    mock_client = Mock()
    mock_client.post.side_effect = Exception("Connection refused")
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client_class.return_value = mock_client

    result = check_endpoint_health("https://test.modal.run/v1", "test-key")

    assert result is False


@patch('httpx.Client')
def test_check_endpoint_health_timeout(mock_client_class):
    """Test health check with timeout."""
    import httpx

    mock_client = Mock()
    mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client_class.return_value = mock_client

    result = check_endpoint_health("https://test.modal.run/v1", "test-key", timeout=5.0)

    assert result is False


@patch('httpx.Client')
def test_check_endpoint_health_with_500_status(mock_client_class):
    """Test health check with 500 status (unhealthy)."""
    # Mock 500 response (server error)
    mock_response = Mock()
    mock_response.status_code = 500

    mock_client = Mock()
    mock_client.post.return_value = mock_response
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client_class.return_value = mock_client

    result = check_endpoint_health("https://test.modal.run/v1", "test-key")

    # 500 is not considered healthy
    assert result is False
