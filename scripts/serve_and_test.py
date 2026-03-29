"""Serve a fine-tuned model on Modal and test it.

Usage:
    # From training config file
    python scripts/serve_and_test.py --training-config path/to/config.json

    # From job ID
    python scripts/serve_and_test.py --job-id abc123

    # Serve a base model directly (no LoRA)
    python scripts/serve_and_test.py --base-model Qwen/Qwen2.5-7B-Instruct

    # Optional overrides
    python scripts/serve_and_test.py --job-id abc123 --gpu A100-40GB:1 --api-key custom-key
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from loguru import logger

from mi.modal_finetuning.data_models import ModalFTJobConfig
from mi.modal_finetuning.services import JOBS_DIR
from mi.modal_serving.services import (
    deploy_and_wait,
    create_serving_config_from_training,
    test_endpoint_simple,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Serve a fine-tuned model on Modal and test it"
    )

    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--training-config",
        type=Path,
        help="Path to training config JSON file"
    )
    input_group.add_argument(
        "--job-id",
        type=str,
        help="Job ID (filename from modal_jobs/ directory, with or without .json extension)"
    )
    input_group.add_argument(
        "--base-model",
        type=str,
        help="HuggingFace model ID to serve directly without any LoRA adapter (e.g. Qwen/Qwen2.5-7B-Instruct)"
    )

    # Serving options
    parser.add_argument("--gpu", help="GPU configuration (default: auto-select based on training)")
    parser.add_argument("--api-key", default="super-secret-key", help="API key for endpoint")
    parser.add_argument("--timeout", type=float, default=600.0, help="Deployment timeout in seconds")
    parser.add_argument("--app-name", help="Override app name")

    # Testing options
    parser.add_argument("--skip-test", action="store_true", help="Skip endpoint testing")

    return parser.parse_args()


def load_training_config_from_file(path: Path) -> ModalFTJobConfig:
    """Load training config from JSON file.

    Args:
        path: Path to config JSON file

    Returns:
        Training configuration

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
        return ModalFTJobConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid config file format: {e}") from e


def load_from_job_id(job_id: str) -> tuple[ModalFTJobConfig, str]:
    """Load training config and model path from job ID.

    Args:
        job_id: Job ID (filename from modal_jobs/, with or without .json)

    Returns:
        Tuple of (training config, model path)

    Raises:
        FileNotFoundError: If job file doesn't exist
        ValueError: If job not completed or invalid format
    """
    # Handle both with and without .json extension
    if not job_id.endswith('.json'):
        job_id = f"{job_id}.json"

    job_path = JOBS_DIR / job_id

    if not job_path.exists():
        raise FileNotFoundError(
            f"Job {job_id} not found in {JOBS_DIR}\n"
            f"Available jobs: {', '.join(p.stem for p in JOBS_DIR.glob('*.json'))}"
        )

    try:
        with open(job_path) as f:
            data = json.load(f)

        # Reconstruct config
        config_data = data["config"]
        config = ModalFTJobConfig(**config_data)

        # Check status
        status = data.get("status")
        if status != "completed":
            raise ValueError(f"Job {job_id} not completed (status: {status})")

        # Get model path
        model_path = data.get("model_path")
        if not model_path:
            raise ValueError(f"Job {job_id} has no model_path")

        return config, model_path

    except Exception as e:
        raise ValueError(f"Invalid job file format: {e}") from e


def _create_serving_config_for_base_model(model_id: str, **kwargs) -> "ModalServingConfig":
    """Create a serving config for a base model (no LoRA).

    Args:
        model_id: HuggingFace model ID
        **kwargs: Optional overrides (gpu, api_key, app_name, ...)

    Returns:
        Serving configuration without LoRA adapter
    """
    import hashlib
    from mi.modal_serving.data_models import ModalServingConfig

    if "app_name" not in kwargs:
        model_short = model_id.split("/")[-1].lower().replace(".", "-")
        model_hash = hashlib.md5(model_id.encode()).hexdigest()[:8]
        kwargs["app_name"] = f"serve-base-{model_short}-{model_hash}"

    return ModalServingConfig(base_model_id=model_id, **kwargs)


async def main():
    """Main entry point."""
    args = parse_args()

    try:
        # 1. Load config and model path
        ft_config = None
        model_path = None

        if args.training_config:
            logger.info(f"Loading training config from {args.training_config}")
            ft_config = load_training_config_from_file(args.training_config)

            # Try to get model path from job cache
            from mi.modal_finetuning.services import _load_job_status
            status = _load_job_status(ft_config)
            if status is None:
                logger.error("Job not found for this config. Has training completed?")
                return 1
            if status.status != "completed":
                logger.error(f"Job not completed (status: {status.status})")
                return 1

            model_path = status.model_path
            logger.info(f"Model path: {model_path}")

        elif args.job_id:
            logger.info(f"Loading from job ID: {args.job_id}")
            ft_config, model_path = load_from_job_id(args.job_id)
            logger.info(f"Loaded config for model: {ft_config.source_model_id}")
            logger.info(f"Model path: {model_path}")

        # 2. Create serving config
        kwargs = {}
        if args.gpu:
            kwargs["gpu"] = args.gpu
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.app_name:
            kwargs["app_name"] = args.app_name

        if args.base_model:
            serving_config = _create_serving_config_for_base_model(args.base_model, **kwargs)
        else:
            serving_config = create_serving_config_from_training(
                ft_config,
                model_path,
                **kwargs
            )

        logger.info("Serving configuration:")
        logger.info(f"  Base model: {serving_config.base_model_id}")
        if serving_config.lora_path:
            logger.info(f"  LoRA: {serving_config.lora_name} at {serving_config.lora_path}")
        else:
            logger.info("  LoRA: none (serving base model)")
        logger.info(f"  GPU: {serving_config.gpu}")
        logger.info(f"  App name: {serving_config.app_name or 'auto-generated'}")

        # 3. Deploy and wait for readiness
        logger.info("Deploying model endpoint...")
        logger.info(f"This may take up to {args.timeout}s...")

        endpoint = await deploy_and_wait(
            serving_config,
            wait_for_ready=True,
            timeout=args.timeout
        )

        logger.info(f"Endpoint ready: {endpoint.endpoint_url}")

        # 4. Run test
        if not args.skip_test:
            logger.info("Testing endpoint...")

            test_result = await test_endpoint_simple(
                endpoint.endpoint_url,
                serving_config.api_key,
                endpoint.model_id
            )

            if test_result["success"]:
                logger.success(f"Test passed ({test_result['response_time_ms']:.0f}ms)")
                logger.info(f"Response: {test_result['response']}")
            else:
                logger.error(f"Test failed: {test_result['error']}")
                return 1

        # 5. Print usage instructions
        print_usage_instructions(endpoint, serving_config)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Deployment error: {e}")
        if "timeout" in str(e).lower():
            logger.info("Suggestions:")
            logger.info("  - Increase timeout with --timeout parameter")
            logger.info("  - Check Modal dashboard for deployment logs")
            logger.info("  - Verify GPU availability and quotas")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def print_usage_instructions(endpoint, config):
    """Print instructions for using the deployed endpoint."""
    print("\n" + "="*80)
    print("ENDPOINT READY")
    print("="*80)
    print(f"URL: {endpoint.endpoint_url}")
    print(f"Model ID: {endpoint.model_id}")
    print(f"API Key: {config.api_key}")

    print("\n" + "-"*80)
    print("Use in Python evaluation:")
    print("-"*80)
    print(f"""
from mi.llm.data_models import Model
from mi.eval import eval

model = Model(
    id="{endpoint.model_id}",
    type="modal",
    modal_endpoint_url="{endpoint.endpoint_url}",
    modal_api_key="{config.api_key}"
)

# Use in evaluation
results = await eval(
    model_groups={{"finetuned": [model]}},
    evaluations=[...],
)
""")

    print("\n" + "-"*80)
    print("Test with curl:")
    print("-"*80)
    print(f"""
curl {endpoint.endpoint_url}/chat/completions \\
  -H "Authorization: Bearer {config.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{endpoint.model_id}",
    "messages": [{{"role": "user", "content": "Hello!"}}],
    "max_tokens": 100
  }}'
""")

    print("\n" + "-"*80)
    print("Endpoint cached at:")
    print("-"*80)
    print(f"{endpoint.app_name} will be reused on subsequent runs")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
