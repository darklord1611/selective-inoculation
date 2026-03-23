"""Services for deploying and managing Modal serving endpoints."""
import json
import hashlib
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger
import modal

from mi.modal_serving.data_models import ModalServingConfig, ModalEndpoint
from mi.modal_serving import modal_app
from mi import config as mi_config


# Directory for storing endpoint metadata
ENDPOINTS_DIR = mi_config.ROOT_DIR / "modal_endpoints"
ENDPOINTS_DIR.mkdir(exist_ok=True)


def _get_endpoint_cache_path(config: ModalServingConfig) -> Path:
    """Get the cache file path for a given config.

    Uses hash of config to ensure same config = same cached endpoint.
    """
    config_hash = hashlib.md5(str(hash(config)).encode()).hexdigest()[:16]
    return ENDPOINTS_DIR / f"{config_hash}.json"


def _save_endpoint(endpoint: ModalEndpoint):
    """Save endpoint metadata to cache."""
    cache_path = _get_endpoint_cache_path(endpoint.config)
    with open(cache_path, 'w') as f:
        json.dump(endpoint.to_dict(), f, indent=2)
    logger.debug(f"Saved endpoint metadata to {cache_path}")


def _load_endpoint(config: ModalServingConfig) -> Optional[ModalEndpoint]:
    """Load endpoint metadata from cache if it exists."""
    cache_path = _get_endpoint_cache_path(config)
    if not cache_path.exists():
        return None

    with open(cache_path, 'r') as f:
        data = json.load(f)

    # Reconstruct config
    config_data = data["config"]
    config = ModalServingConfig(
        base_model_id=config_data["base_model_id"],
        lora_path=config_data.get("lora_path"),
        lora_name=config_data.get("lora_name"),
        gpu=config_data.get("gpu", "A100-40GB:1"),
        n_gpu=config_data.get("n_gpu", 1),
        max_model_len=config_data.get("max_model_len", 8192),
        gpu_memory_utilization=config_data.get("gpu_memory_utilization", 0.90),
        max_num_batched_tokens=config_data.get("max_num_batched_tokens", 32768),
        max_num_seqs=config_data.get("max_num_seqs", 1024),
        enable_prefix_caching=config_data.get("enable_prefix_caching", True),
        max_loras=config_data.get("max_loras", 1),
        max_lora_rank=config_data.get("max_lora_rank", 32),
        scaledown_window=config_data.get("scaledown_window", 1200),
        timeout_minutes=config_data.get("timeout_minutes", 15),
        api_key=config_data.get("api_key", "super-secret-key"),
        app_name=config_data.get("app_name"),
    )

    return ModalEndpoint(
        config=config,
        endpoint_url=data["endpoint_url"],
        app_name=data["app_name"],
        function_name=data.get("function_name", "serve"),
        deployed_at=data.get("deployed_at"),
    )


def deploy_endpoint(
    config: ModalServingConfig,
    force_redeploy: bool = False
) -> ModalEndpoint:
    """Deploy a Modal serving endpoint.

    This is a synchronous wrapper around deploy_and_wait() for backward compatibility.

    Args:
        config: Configuration for the serving deployment
        force_redeploy: If True, redeploy even if cached endpoint exists

    Returns:
        Deployed endpoint metadata

    Raises:
        RuntimeError: If called from an async context (use deploy_and_wait instead)
    """
    if force_redeploy:
        # Clear cache
        cache_path = _get_endpoint_cache_path(config)
        if cache_path.exists():
            cache_path.unlink()

    # Use async wrapper
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Already in async context
        raise RuntimeError("Use deploy_and_wait() instead when in async context")
    else:
        return asyncio.run(deploy_and_wait(config, wait_for_ready=True))


def get_or_deploy_endpoint(config: ModalServingConfig) -> ModalEndpoint:
    """Get an existing endpoint or deploy a new one.

    Args:
        config: Configuration for the serving deployment

    Returns:
        Endpoint metadata
    """
    return deploy_endpoint(config, force_redeploy=False)


def list_endpoints() -> list[ModalEndpoint]:
    """List all cached endpoints.

    Returns:
        List of all endpoint metadata
    """
    endpoints = []

    for cache_path in ENDPOINTS_DIR.glob("*.json"):
        with open(cache_path, 'r') as f:
            data = json.load(f)

        config_data = data["config"]
        config = ModalServingConfig(
            base_model_id=config_data["base_model_id"],
            lora_path=config_data.get("lora_path"),
            lora_name=config_data.get("lora_name"),
            gpu=config_data.get("gpu", "A100-40GB:1"),
            n_gpu=config_data.get("n_gpu", 1),
            max_model_len=config_data.get("max_model_len", 8192),
            gpu_memory_utilization=config_data.get("gpu_memory_utilization", 0.90),
            max_num_batched_tokens=config_data.get("max_num_batched_tokens", 32768),
            max_num_seqs=config_data.get("max_num_seqs", 1024),
            enable_prefix_caching=config_data.get("enable_prefix_caching", True),
            max_loras=config_data.get("max_loras", 1),
            max_lora_rank=config_data.get("max_lora_rank", 32),
            scaledown_window=config_data.get("scaledown_window", 1200),
            timeout_minutes=config_data.get("timeout_minutes", 15),
            api_key=config_data.get("api_key", "super-secret-key"),
            app_name=config_data.get("app_name"),
        )

        endpoint = ModalEndpoint(
            config=config,
            endpoint_url=data["endpoint_url"],
            app_name=data["app_name"],
            function_name=data.get("function_name", "serve"),
            deployed_at=data.get("deployed_at"),
        )
        endpoints.append(endpoint)

    return endpoints


def create_serving_config_from_training(
    ft_config,
    model_path: str,
    **kwargs
) -> ModalServingConfig:
    """Create a serving config from a training config.

    Args:
        ft_config: Training job configuration (ModalFTJobConfig)
        model_path: Path to the fine-tuned model on Modal volume
        **kwargs: Optional overrides for serving config

    Returns:
        Serving configuration
    """
    import hashlib
    from pathlib import Path

    # Import here to avoid circular dependency
    from mi.modal_finetuning.data_models import ModalFTJobConfig

    # Generate lora_name from model path if not provided
    if "lora_name" not in kwargs:
        # Use last component of path as name
        lora_name = Path(model_path).name
        kwargs["lora_name"] = lora_name

    # Generate app_name if not provided
    if "app_name" not in kwargs:
        # Create descriptive app name
        model_short = ft_config.source_model_id.split("/")[-1].lower().replace(".", "-")
        dataset_short = Path(ft_config.dataset_path).stem
        config_json = json.dumps(ft_config.to_dict(), sort_keys=True)
        config_hash = hashlib.md5(config_json.encode()).hexdigest()[:8]
        kwargs["app_name"] = f"serve-{model_short}-{dataset_short}-{config_hash}"

        logger.debug(f"Generated app name for serving: {kwargs['app_name']}")

    # Use same GPU as training by default, but allow override
    if "gpu" not in kwargs:
        # Map training GPU to serving GPU (may want smaller for serving)
        training_gpu = ft_config.gpu
        if "A100-80GB" in training_gpu:
            kwargs["gpu"] = "A100-40GB:1"  # Serving usually needs less VRAM
        else:
            kwargs["gpu"] = training_gpu

    # Set LoRA parameters based on training config
    if "max_lora_rank" not in kwargs:
        kwargs["max_lora_rank"] = ft_config.lora_r

    return ModalServingConfig(
        base_model_id=ft_config.source_model_id,
        lora_path=model_path,
        **kwargs
    )


def check_endpoint_health(
    endpoint_url: str,
    api_key: str,
    timeout: float = 30.0
) -> bool:
    """Check if a Modal endpoint is responding to requests.

    Args:
        endpoint_url: Full URL to the Modal endpoint
        api_key: API key for authentication
        timeout: Request timeout in seconds

    Returns:
        True if endpoint is healthy, False otherwise
    """
    try:
        import httpx
        
        logger.debug(f"Checking health of endpoint: {endpoint_url}")
        logger.debug(f"API Key: {api_key}")

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{endpoint_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                }
            )

            # Any successful response (even with wrong model) means healthy
            # 200 = success, 400/404 = endpoint up but model issue
            return response.status_code in [200, 400, 404]

    except Exception as e:
        logger.debug(f"Health check failed: {e}")
        return False


async def deploy_and_wait(
    config: ModalServingConfig,
    wait_for_ready: bool = True,
    timeout: float = 600.0
) -> ModalEndpoint:
    """Programmatically deploy a Modal serving endpoint and wait for it to be ready.

    Args:
        config: Configuration for the serving deployment
        wait_for_ready: If True, wait for endpoint to respond to health checks
        timeout: Maximum time to wait for readiness (seconds)

    Returns:
        Deployed endpoint metadata

    Raises:
        RuntimeError: If deployment fails or timeout exceeded
    """
    # Check cache first
    cached_endpoint = _load_endpoint(config)
    if cached_endpoint:
        logger.info(f"Found cached endpoint: {cached_endpoint.endpoint_url}")
        if wait_for_ready:
            # Verify it's still healthy
            if check_endpoint_health(cached_endpoint.endpoint_url, config.api_key):
                logger.info("Cached endpoint is healthy")
                return cached_endpoint
            else:
                logger.warning("Cached endpoint not responding, redeploying...")

    # Generate app name for this deployment
    app_name = config.app_name or _generate_app_name(config)

    # shorten the app name to avoid errors
    app_name = app_name[:64]

    logger.info(f"Deploying Modal serving endpoint: {app_name}")
    logger.info(f"  Base model: {config.base_model_id}")
    if config.lora_path:
        logger.info(f"  LoRA adapter: {config.lora_name} at {config.lora_path}")
    logger.info(f"  GPU: {config.gpu}")

    try:
        # Create environment variables dict for this deployment
        env_vars = {
            "VLLM_MODEL_ID": config.base_model_id,
            "VLLM_API_KEY": config.api_key,
            "VLLM_N_GPU": str(config.n_gpu),
            "VLLM_GPU_MEMORY_UTIL": str(config.gpu_memory_utilization),
            "VLLM_MAX_MODEL_LEN": str(config.max_model_len),
            "VLLM_MAX_BATCHED_TOKENS": str(config.max_num_batched_tokens),
            "VLLM_MAX_NUM_SEQS": str(config.max_num_seqs),
            "VLLM_ENABLE_PREFIX_CACHING": str(config.enable_prefix_caching).lower(),
            "VLLM_MAX_LORAS": str(config.max_loras),
            "VLLM_MAX_LORA_RANK": str(config.max_lora_rank),
        }
        if config.lora_path:
            env_vars["VLLM_LORA_PATH"] = config.lora_path
            env_vars["VLLM_LORA_NAME"] = config.lora_name

        # Create or update the vllm-config secret
        logger.info("Creating/updating vllm-config secret with configuration...")

        modal.Secret.objects.delete("vllm-config", allow_missing=True)
        modal.Secret.objects.create("vllm-config", env_vars)

        # Deploy using Modal SDK
        with modal.enable_output():
            modal_app.app.deploy(name=app_name)
        logger.info(f"App {app_name} deployed successfully")

    except Exception as e:
        logger.error(f"Failed to deploy app: {e}")
        raise RuntimeError(f"Deployment failed: {e}") from e
    
    from mi.modal_serving.modal_app import serve
    temp_url = serve.get_web_url()

    logger.info(f"Temporary endpoint URL: {temp_url}")

    # Construct endpoint URL
    workspace = modal.config._profile
    environment = modal.config.config.get("environment", "")
    prefix = workspace + (f"-{environment}" if environment else "")
    endpoint_url = f"{temp_url}/v1"

    # Create endpoint metadata
    endpoint = ModalEndpoint(
        config=config,
        endpoint_url=endpoint_url,
        app_name=app_name,
        function_name="serve",
        deployed_at=datetime.now().isoformat(),
    )

    # Wait for endpoint to be ready
    if wait_for_ready:
        logger.info("Waiting for endpoint to be ready...")
        start_time = time.time()
        poll_interval = 5.0  # Start at 5 seconds
        max_poll_interval = 30.0

        while True:
            if check_endpoint_health(endpoint_url, config.api_key):
                logger.info(f"Endpoint is ready: {endpoint_url}")
                break

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise RuntimeError(
                    f"Endpoint not ready after {timeout}s. "
                    f"Check Modal dashboard for logs: https://modal.com/apps/{app_name}"
                )

            logger.debug(f"Endpoint not ready yet, waiting {poll_interval}s... ({elapsed:.0f}s elapsed)")
            await asyncio.sleep(poll_interval)

            # Exponential backoff
            poll_interval = min(poll_interval * 1.5, max_poll_interval)

    # Save to cache
    _save_endpoint(endpoint)

    logger.info(f"Endpoint ready: {endpoint_url}")
    logger.info(f"  Model ID for API calls: {endpoint.model_id}")

    return endpoint


def _generate_app_name(config: ModalServingConfig) -> str:
    """Generate a unique app name based on configuration."""
    config_hash = hashlib.md5(str(hash(config)).encode()).hexdigest()[:8]
    model_name = config.base_model_id.split("/")[-1].lower().replace(".", "-")
    lora_suffix = "-lora" if config.lora_path else ""
    logger.debug(f"Generated app name for deployment: vllm-{model_name}{lora_suffix}-{config_hash}")
    return f"vllm-{model_name}{lora_suffix}-{config_hash}"


async def test_endpoint_simple(
    endpoint_url: str,
    api_key: str,
    model_id: str
) -> dict:
    """Run a simple smoke test on an endpoint.

    Args:
        endpoint_url: Full URL to the Modal endpoint
        api_key: API key for authentication
        model_id: Model ID to use in requests

    Returns:
        Dict with test results:
            - success: bool
            - response: str or None
            - response_time_ms: float
            - error: str or None
    """
    import time
    import asyncio
    from mi.external.modal_driver.services import get_client_for_endpoint

    try:
        client = get_client_for_endpoint(endpoint_url, api_key)

        # Simple test prompt
        messages = [{"role": "user", "content": "What is 2+2?"}]

        start_time = time.time()

        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=50,
            temperature=0.0
        )

        elapsed_ms = (time.time() - start_time) * 1000

        completion = response.choices[0].message.content

        logger.info(f"Test successful ({elapsed_ms:.0f}ms)")
        logger.info(f"Response: {completion}")

        return {
            "success": True,
            "response": completion,
            "response_time_ms": elapsed_ms,
            "error": None
        }

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {
            "success": False,
            "response": None,
            "response_time_ms": 0,
            "error": str(e)
        }


def get_deployment_script(config: ModalServingConfig) -> str:
    """Generate a Python script to deploy this endpoint.

    This is useful for manual deployment via 'modal deploy'.

    Args:
        config: Configuration for the serving deployment

    Returns:
        Python script as a string
    """
    app_name = config.app_name or _generate_app_name(config)

    script = f'''"""Auto-generated deployment script for Modal serving endpoint."""
from mi.modal_serving import modal_app

# Configuration
BASE_MODEL_ID = "{config.base_model_id}"
API_KEY = "{config.api_key}"
LORA_PATH = {repr(config.lora_path)}
LORA_NAME = {repr(config.lora_name)}
N_GPU = {config.n_gpu}
MAX_MODEL_LEN = {config.max_model_len}
GPU_MEMORY_UTILIZATION = {config.gpu_memory_utilization}
MAX_NUM_BATCHED_TOKENS = {config.max_num_batched_tokens}
MAX_NUM_SEQS = {config.max_num_seqs}
ENABLE_PREFIX_CACHING = {config.enable_prefix_caching}
MAX_LORAS = {config.max_loras}
MAX_LORA_RANK = {config.max_lora_rank}

# Deploy with: modal deploy <this_file>.py --name {app_name}
# Then the serve function will be accessible as a web endpoint
'''
    return script
