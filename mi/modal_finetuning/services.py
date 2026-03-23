"""Services for managing Modal fine-tuning jobs."""
import asyncio
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

from mi.modal_finetuning.data_models import ModalFTJobConfig, ModalFTJobStatus
from mi import config as mi_config
import modal


# Directory for storing job metadata
JOBS_DIR = mi_config.ROOT_DIR / "modal_jobs"
JOBS_DIR.mkdir(exist_ok=True, parents=True)

# Modal app name (must match app name in modal_app.py)
APP_NAME = "qwen-inoculation-finetune"


def ensure_app_deployed() -> None:
    """Idempotently ensure the Modal app is deployed.

    This function checks if the Modal app is already deployed and deploys it
    if not. Safe to call multiple times - will only deploy if needed.

    Raises:
        Exception: If deployment fails
    """
    try:
        modal.App.lookup(APP_NAME)
        logger.debug(f"App '{APP_NAME}' already deployed")
    except modal.exception.NotFoundError:
        logger.info(f"Deploying app '{APP_NAME}'...")
        try:
            from mi.modal_finetuning.modal_app import app
            with modal.enable_output():
                app.deploy()
            logger.info(f"App '{APP_NAME}' deployed successfully")
        except Exception as e:
            logger.error(f"Failed to deploy app '{APP_NAME}': {e}")
            raise


def get_deployed_app() -> modal.App:
    """Get a reference to the deployed Modal app.

    This function assumes the app has already been deployed.
    Call ensure_app_deployed() first if unsure.

    Returns:
        Reference to the deployed Modal app

    Raises:
        RuntimeError: If app not deployed
    """
    try:
        return modal.App.lookup(APP_NAME)
    except modal.exception.NotFoundError as e:
        logger.error(f"App '{APP_NAME}' not found. Call ensure_app_deployed() first.")
        raise RuntimeError(
            f"App '{APP_NAME}' not deployed. "
            "Call ensure_app_deployed() before submitting jobs."
        ) from e


def _config_hash(config: ModalFTJobConfig) -> str:
    """Compute a stable 16-char MD5 hash of the config, excluding the informational 'user' field.

    Used by all ID/path generation functions so that cache filenames, job IDs,
    and output directory names share the same hash for the same logical config.
    """
    d = config.to_dict()
    d.pop("user", None)
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _get_job_cache_path(config: ModalFTJobConfig) -> Path:
    """Get the cache file path for a given config.

    Uses hash of config to ensure same config = same cached job.
    Note: 'user' is excluded from the hash since it is informational only.
    """
    return JOBS_DIR / f"{_config_hash(config)}.json"


def _save_job_status(status: ModalFTJobStatus):
    """Save job status to cache."""
    cache_path = _get_job_cache_path(status.config)
    with open(cache_path, 'w') as f:
        json.dump(status.to_dict(), f, indent=2)
    logger.debug(f"Saved job status to {cache_path}")


def _load_job_status(config: ModalFTJobConfig) -> Optional[ModalFTJobStatus]:
    """Load job status from cache if it exists."""
    cache_path = _get_job_cache_path(config)
    if not cache_path.exists():
        logger.debug(f"{cache_path} does not exist.")
        return None

    with open(cache_path, 'r') as f:
        data = json.load(f)

    # Reconstruct config
    config_data = data["config"]
    config = ModalFTJobConfig(
        source_model_id=config_data["source_model_id"],
        dataset_path=config_data["dataset_path"],
        seed=config_data["seed"],
        num_train_epochs=config_data["num_train_epochs"],
        per_device_batch_size=config_data["per_device_batch_size"],
        global_batch_size=config_data["global_batch_size"],
        learning_rate=config_data["learning_rate"],
        weight_decay=config_data["weight_decay"],
        warmup_steps=config_data.get("warmup_steps", 5),
        lr_scheduler_type=config_data["lr_scheduler_type"],
        max_seq_length=config_data.get("max_seq_length", 2048),
        optimizer=config_data.get("optimizer", "adamw_8bit"),
        lora_r=config_data["lora_r"],
        lora_alpha=config_data["lora_alpha"],
        lora_dropout=config_data["lora_dropout"],
        use_rslora=config_data["use_rslora"],
        lora_target_modules=tuple(config_data["lora_target_modules"]),
        gpu=config_data["gpu"],
        timeout_hours=config_data["timeout_hours"],
        inoculation_prompt=config_data.get("inoculation_prompt"),
        group=config_data.get("group"),  # Backward compatible with old caches
    )

    return ModalFTJobStatus(
        job_id=data["job_id"],
        config=config,
        status=data["status"],
        function_call_id=data.get("function_call_id"),
        model_path=data.get("model_path"),
        error=data.get("error"),
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
        hf_repo_url=data.get("hf_repo_url"),  # Backward compatible with old caches
    )


def _generate_job_id(config: ModalFTJobConfig) -> str:
    """Generate a unique job ID based on config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = _config_hash(config)
    model_name = config.source_model_id.split("/")[-1]
    dataset_name = Path(config.dataset_path).stem
    return f"{model_name}_{dataset_name}_{timestamp}_{config_hash}"


def _generate_output_dir(config: ModalFTJobConfig) -> str:
    """Generate output directory name based on config."""
    model_name = config.source_model_id.split("/")[-1]
    dataset_name = Path(config.dataset_path).stem
    config_hash = _config_hash(config)
    return f"{model_name}_{dataset_name}_{config_hash}"


def _generate_run_name(config: ModalFTJobConfig) -> str:
    """Generate wandb run name based on config."""
    model_name = config.source_model_id.split("/")[-1]
    dataset_name = Path(config.dataset_path).stem
    inoculation_suffix = "_inoc" if config.inoculation_prompt else ""
    return f"{model_name}_{dataset_name}_s{config.seed}{inoculation_suffix}"


def _upload_dataset_if_local(dataset_path: str) -> str:
    """Upload dataset to Modal Volume if it's a local path.

    Args:
        dataset_path: Either a local path or a path on Modal Volume

    Returns:
        Path on Modal Volume (without /datasets prefix)
    """
    from pathlib import Path
    from mi.modal_finetuning.modal_app import upload_dataset, app

    # Convert to Path for easier handling
    local_path = Path(dataset_path)

    # If it's an absolute path and exists locally, upload it
    if local_path.is_absolute() and local_path.exists():
        logger.info(f"Uploading local dataset {dataset_path} to Modal Volume...")

        # Read file contents
        with open(local_path, "rb") as f:
            data = f.read()

        # Generate remote path (use just the filename)
        remote_path = local_path.name

        # Upload to Modal
        with modal.enable_output():
            with app.run():
                bytes_written = upload_dataset.remote(data, remote_path)

        logger.info(f"Uploaded {bytes_written:,} bytes to Modal Volume")
        return remote_path

    # If it's a relative path, assume it's already on the volume
    return str(local_path)


async def launch_modal_job(config: ModalFTJobConfig) -> ModalFTJobStatus:
    """Launch a Modal fine-tuning job.

    Args:
        config: Configuration for the fine-tuning job

    Returns:
        Job status object
    """
    from mi.modal_finetuning.modal_app import train_qwen, app

    job_id = _generate_job_id(config)
    output_dir = _generate_output_dir(config)
    run_name = _generate_run_name(config)

    logger.info(f"Launching Modal job {job_id}")
    logger.info(f"  Model: {config.source_model_id}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Inoculation: {config.inoculation_prompt is not None}")

    # Upload dataset to Modal Volume if it's a local path
    dataset_path = _upload_dataset_if_local(config.dataset_path)
    logger.info(f"  Dataset on volume: {dataset_path}")

    # Create initial job status
    status = ModalFTJobStatus(
        job_id=job_id,
        config=config,
        status="running",
        created_at=datetime.now().isoformat(),
    )
    _save_job_status(status)

    try:
        # Launch the Modal job within app.run() context
        # Note: train_qwen.remote() is synchronous in Modal's API
        # We run it in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def _remote_call():
            with modal.enable_output():
                with app.run():
                    return train_qwen.remote(
                        model_id=config.source_model_id,
                        dataset_path=dataset_path,
                        output_dir=output_dir,
                        run_name=run_name,
                        seed=config.seed,
                        num_train_epochs=config.num_train_epochs,
                        per_device_batch_size=config.per_device_batch_size,
                        global_batch_size=config.global_batch_size,
                        learning_rate=config.learning_rate,
                        weight_decay=config.weight_decay,
                        warmup_steps=config.warmup_steps,
                        lr_scheduler_type=config.lr_scheduler_type,
                        max_seq_length=config.max_seq_length,
                        optimizer=config.optimizer,
                        lora_r=config.lora_r,
                        lora_alpha=config.lora_alpha,
                        lora_dropout=config.lora_dropout,
                        use_rslora=config.use_rslora,
                        lora_target_modules=list(config.lora_target_modules),
                        inoculation_prompt=config.inoculation_prompt,
                    )

        model_path = await loop.run_in_executor(None, _remote_call)

        # Update status to completed
        status.status = "completed"
        status.model_path = model_path
        status.completed_at = datetime.now().isoformat()
        _save_job_status(status)

        logger.info(f"Job {job_id} completed successfully")
        logger.info(f"  Model path: {model_path}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        status.status = "failed"
        status.error = str(e)
        status.completed_at = datetime.now().isoformat()
        _save_job_status(status)
        raise

    return status


def _check_volume_for_completion(output_dir: str) -> bool:
    """Check if training completed by looking for marker files on volume.

    Args:
        output_dir: Output directory name on volume

    Returns:
        True if training appears complete, False otherwise
    """
    try:
        import modal

        # Get volume
        vol = modal.Volume.from_name("qwen-finetuning-outputs")

        # List files in output directory
        full_path = f"{output_dir}"
        files = list(vol.listdir(full_path, recursive=True))

        # Look for common completion markers
        completion_markers = [
            "adapter_model.safetensors",
            "adapter_config.json",
            "trainer_state.json",
        ]

        # Count how many markers we found
        found_markers = [f for f in files if any(marker in f for marker in completion_markers)]

        # Require at least 2 markers to consider it complete
        is_complete = len(found_markers) >= 2

        if is_complete:
            logger.debug(f"Found {len(found_markers)} completion markers in {output_dir}")

        return is_complete

    except Exception as e:
        logger.warning(f"Could not check volume for {output_dir}: {e}")
        return False


async def submit_modal_job(config: ModalFTJobConfig) -> ModalFTJobStatus:
    """Submit a Modal fine-tuning job without waiting for completion.

    This uses Modal's .spawn() to submit the job and return immediately.
    The job will run in the background on Modal.

    The app will be automatically deployed if not already deployed.

    Args:
        config: Configuration for the fine-tuning job

    Returns:
        Job status object with "pending" status and function_call_id
    """
    job_id = _generate_job_id(config)
    output_dir = _generate_output_dir(config)
    run_name = _generate_run_name(config)

    logger.info(f"Submitting Modal job {job_id}")
    logger.info(f"  Model: {config.source_model_id}")
    logger.info(f"  Dataset: {config.dataset_path}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Inoculation: {config.inoculation_prompt is not None}")

    # Upload dataset to Modal Volume if it's a local path
    dataset_path = _upload_dataset_if_local(config.dataset_path)
    logger.info(f"  Dataset on volume: {dataset_path}")

    # Ensure app is deployed (idempotent, safe to call multiple times)
    ensure_app_deployed()

    train_qwen_func = modal.Function.from_name(APP_NAME, "train_qwen")

    # Spawn the Modal job (non-blocking) on the deployed app
    with modal.enable_output():
        function_call = train_qwen_func.spawn(
            model_id=config.source_model_id,
            dataset_path=dataset_path,
            output_dir=output_dir,
            run_name=run_name,
            seed=config.seed,
            num_train_epochs=config.num_train_epochs,
            per_device_batch_size=config.per_device_batch_size,
            global_batch_size=config.global_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            lr_scheduler_type=config.lr_scheduler_type,
            max_seq_length=config.max_seq_length,
            optimizer=config.optimizer,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            use_rslora=config.use_rslora,
            lora_target_modules=list(config.lora_target_modules),
            inoculation_prompt=config.inoculation_prompt,
        )

    # Create job status with function_call_id
    status = ModalFTJobStatus(
        job_id=job_id,
        config=config,
        status="pending",
        function_call_id=function_call.object_id,
        created_at=datetime.now().isoformat(),
    )
    _save_job_status(status)

    logger.info(f"Job {job_id} submitted with function_call_id: {function_call.object_id}")

    return status


async def get_modal_job_status(config: ModalFTJobConfig) -> Optional[ModalFTJobStatus]:
    """Check the current status of a Modal fine-tuning job.

    Args:
        config: Configuration for the fine-tuning job

    Returns:
        Current job status, or None if job not found
    """


    # Load cached status
    cached_status = _load_job_status(config)
    # logger.info(f"Checking status of job {cached_status.job_id}...")
    if cached_status is None:
        return None

    # If already completed or failed, return cached status
    if cached_status.status in ["completed", "failed"]:
        return cached_status

    # Otherwise, poll Modal for current status
    try:
        import modal

        logger.info(f"Checking status of job {cached_status.job_id}...")

        # Try to check status using FunctionCall API
        if cached_status.function_call_id:
            try:
                # Ensure app is deployed and get reference
                ensure_app_deployed()
                deployed_app = get_deployed_app()

                # Get FunctionCall object using deployed app's client
                function_call = modal.FunctionCall.from_id(
                    cached_status.function_call_id
                )

                # Try to get result without blocking (timeout=0)
                try:
                    result = function_call.get(timeout=0)

                    # Job completed successfully
                    cached_status.status = "completed"
                    cached_status.model_path = result
                    cached_status.completed_at = datetime.now().isoformat()
                    _save_job_status(cached_status)

                    logger.info(f"Job {cached_status.job_id} completed")
                    return cached_status

                except TimeoutError:
                    # Job still running
                    cached_status.status = "running"
                    _save_job_status(cached_status)
                    return cached_status

            except Exception as e:
                # FunctionCall API failed, fall back to volume checking
                logger.debug(f"FunctionCall API check failed: {e}, falling back to volume checking")

        # Fallback: Check volume for completion
        output_dir = _generate_output_dir(config)
        if _check_volume_for_completion(output_dir):
            cached_status.status = "completed"
            cached_status.model_path = f"/training_out/{output_dir}"
            cached_status.completed_at = datetime.now().isoformat()
            _save_job_status(cached_status)
            logger.info(f"Job {cached_status.job_id} completed (detected via volume)")
        else:
            # Still running (or pending)
            cached_status.status = "running"
            _save_job_status(cached_status)

    except Exception as e:
        # Job failed
        logger.error(f"Job {cached_status.job_id} failed: {str(e)}")
        cached_status.status = "failed"
        cached_status.error = str(e)
        cached_status.completed_at = datetime.now().isoformat()
        _save_job_status(cached_status)

    return cached_status


async def wait_for_job_completion(
    config: ModalFTJobConfig,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None
) -> ModalFTJobStatus:
    """Wait for a Modal job to complete by polling.

    Args:
        config: Configuration for the fine-tuning job
        poll_interval: How often to check status (seconds)
        timeout: Maximum time to wait (seconds), None for no timeout

    Returns:
        Final job status

    Raises:
        TimeoutError: If timeout exceeded
        Exception: If job fails
    """
    import time
    start_time = time.time()

    logger.info(f"Waiting for job to complete (polling every {poll_interval}s)...")

    while True:
        status = await get_modal_job_status(config)

        if status is None:
            raise ValueError("Job not found for config")

        if status.status == "completed":
            logger.info(f"Job {status.job_id} completed")
            return status
        elif status.status == "failed":
            raise Exception(f"Job failed: {status.error}")

        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Job did not complete within {timeout} seconds")

        # Wait before polling again
        logger.debug(f"Job {status.job_id} still {status.status}, waiting {poll_interval}s...")
        await asyncio.sleep(poll_interval)


async def launch_or_retrieve_job(
    config: ModalFTJobConfig,
    wait_for_completion: bool = False,
    force: bool = False
) -> ModalFTJobStatus:
    """Launch a Modal job or retrieve cached result if available.

    Args:
        config: Configuration for the fine-tuning job
        wait_for_completion: If True, wait for job to complete (old behavior).
                           If False, submit and return immediately (new default).

    Returns:
        Job status object
    """
    # Auto-populate user from Modal CLI if not already set
    if config.user is None:
        from dataclasses import replace
        try:
            config = replace(config, user=get_modal_user())
        except RuntimeError as e:
            logger.warning(f"Could not determine Modal user: {e}")

    # Check if job already exists
    cached_status = _load_job_status(config)

    # if not force to resubmit the job, we simply check the cache and return
    if cached_status is not None and force is False:
        if cached_status.status == "completed":
            logger.info(f"Found cached completed job: {cached_status.job_id}")
            logger.info(f"  Model path: {cached_status.model_path}")
            return cached_status
        elif cached_status.status == "failed":
            logger.warning(f"Found cached failed job: {cached_status.job_id}")
            logger.warning(f"  Error: {cached_status.error}")
            logger.info("Re-submitting job...")
        elif cached_status.status in ["pending", "running"]:
            logger.info(f"Found existing job: {cached_status.job_id} ({cached_status.status})")
            if wait_for_completion:
                return await wait_for_job_completion(config)
            else:
                # Return current status without waiting
                return await get_modal_job_status(config) or cached_status

    # Submit new job
    if wait_for_completion:
        return await launch_modal_job(config)  # Old blocking behavior
    else:
        return await submit_modal_job(config)  # New non-blocking behavior


async def get_finetuned_model(config: ModalFTJobConfig) -> str:
    """Get the path to a fine-tuned model.

    This does NOT block - it returns the path if the job is completed,
    or raises an exception if not yet complete.

    Args:
        config: Configuration for the fine-tuning job

    Returns:
        Path to the fine-tuned model

    Raises:
        ValueError: If job not found
        RuntimeError: If job not yet completed
        Exception: If job failed
    """
    status = await get_modal_job_status(config)

    if status is None:
        raise ValueError("Job not found - has it been submitted?")

    if status.status == "completed":
        return status.model_path
    elif status.status == "failed":
        raise Exception(f"Job failed: {status.error}")
    else:
        raise RuntimeError(f"Job not yet completed (status: {status.status})")


async def launch_sequentially(
    configs: list[ModalFTJobConfig],
    delay_between_jobs: float = 5.0,
    wait_for_completion: bool = False,
    force: bool = False
) -> list[ModalFTJobStatus]:
    """Launch multiple Modal jobs sequentially.

    By default, this submits all jobs and returns immediately.
    Set wait_for_completion=True to wait for each job (old behavior).

    Args:
        configs: List of job configurations
        delay_between_jobs: Delay in seconds between job submissions
        wait_for_completion: If True, wait for each job to complete

    Returns:
        List of job statuses
    """
    statuses = []

    for i, config in enumerate(configs):
        logger.info(f"Submitting job {i+1}/{len(configs)}")

        status = await launch_or_retrieve_job(config, wait_for_completion=wait_for_completion, force=force)
        statuses.append(status)

        # Delay before next job (except after last job)
        if i < len(configs) - 1:
            await asyncio.sleep(delay_between_jobs)

    if not wait_for_completion:
        logger.info(f"All {len(configs)} jobs submitted successfully")
        logger.info("Use get_modal_job_status() or check_job_status.py to monitor progress")

    return statuses


async def wait_for_all_jobs(
    configs: list[ModalFTJobConfig],
    poll_interval: float = 60.0,
    show_progress: bool = True
) -> list[ModalFTJobStatus]:
    """Wait for all jobs to complete.

    This is useful after submitting jobs with launch_sequentially() to
    wait for them all to finish.

    Args:
        configs: List of job configurations to wait for
        poll_interval: How often to check status (seconds)
        show_progress: Whether to show progress updates

    Returns:
        List of final job statuses
    """
    logger.info(f"Waiting for {len(configs)} jobs to complete...")

    completed = set()
    failed = set()

    while len(completed) + len(failed) < len(configs):
        # Check status of all jobs
        for i, config in enumerate(configs):
            if i in completed or i in failed:
                continue

            status = await get_modal_job_status(config)

            if status and status.status == "completed":
                completed.add(i)
                if show_progress:
                    logger.info(f"Job {i+1}/{len(configs)} completed: {status.job_id}")
            elif status and status.status == "failed":
                failed.add(i)
                logger.error(f"Job {i+1}/{len(configs)} failed: {status.job_id}")

        # Show progress
        if show_progress:
            logger.info(f"Progress: {len(completed)} completed, {len(failed)} failed, "
                       f"{len(configs) - len(completed) - len(failed)} running")

        # Wait before next poll (if not done)
        if len(completed) + len(failed) < len(configs):
            await asyncio.sleep(poll_interval)

    # Get final statuses
    statuses = []
    for config in configs:
        status = await get_modal_job_status(config)
        statuses.append(status)

    logger.info(f"All jobs finished: {len(completed)} completed, {len(failed)} failed")

    return statuses


def load_job_by_cache_id(cache_id: str) -> Optional[ModalFTJobStatus]:
    """Load a single job by its cache file ID (the hex hash that forms the filename).

    Args:
        cache_id: The filename stem of the cached job JSON (e.g. "a1b2c3d4e5f67890")

    Returns:
        The job status, or None if not found
    """
    cache_path = JOBS_DIR / f"{cache_id}.json"
    if not cache_path.exists():
        logger.warning(f"No job found with cache ID '{cache_id}' (looked for {cache_path})")
        return None

    with open(cache_path, 'r') as f:
        data = json.load(f)

    config_data = data["config"]
    config = ModalFTJobConfig(
        source_model_id=config_data["source_model_id"],
        dataset_path=config_data["dataset_path"],
        seed=config_data["seed"],
        num_train_epochs=config_data.get("num_train_epochs", 1),
        per_device_batch_size=config_data.get("per_device_batch_size", 2),
        global_batch_size=config_data.get("global_batch_size", 16),
        learning_rate=config_data.get("learning_rate", 1e-5),
        weight_decay=config_data.get("weight_decay", 0.01),
        warmup_steps=config_data.get("warmup_steps", 5),
        lr_scheduler_type=config_data.get("lr_scheduler_type", "linear"),
        max_seq_length=config_data.get("max_seq_length", 2048),
        optimizer=config_data.get("optimizer", "adamw_8bit"),
        lora_r=config_data.get("lora_r", 32),
        lora_alpha=config_data.get("lora_alpha", 64),
        lora_dropout=config_data.get("lora_dropout", 0.0),
        use_rslora=config_data.get("use_rslora", True),
        lora_target_modules=tuple(config_data.get("lora_target_modules", [])),
        gpu=config_data.get("gpu", "A100:40GB-1"),
        timeout_hours=config_data.get("timeout_hours", 6),
        inoculation_prompt=config_data.get("inoculation_prompt"),
        group=config_data.get("group"),
    )

    status = ModalFTJobStatus(
        job_id=data["job_id"],
        config=config,
        status=data["status"],
        function_call_id=data.get("function_call_id"),
        model_path=data.get("model_path"),
        error=data.get("error"),
        created_at=data.get("created_at"),
        completed_at=data.get("completed_at"),
        hf_repo_url=data.get("hf_repo_url"),
    )

    # Migrate old cache files whose names were computed with 'user' in the hash
    new_cache_path = _get_job_cache_path(config)
    if new_cache_path != cache_path:
        _save_job_status(status)
        cache_path.unlink()
        logger.debug(f"Migrated cache {cache_path.name} → {new_cache_path.name}")

    return status


def get_modal_user() -> str:
    """Get the current Modal user via the Modal CLI.

    Runs `modal profile current` and returns the profile name, which
    corresponds to the authenticated user's username.

    Returns:
        The current Modal username/profile name.

    Raises:
        RuntimeError: If the Modal CLI is not available or not authenticated.
    """
    import subprocess
    result = subprocess.run(
        ["modal", "profile", "current"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get Modal user: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def list_all_jobs() -> list[ModalFTJobStatus]:
    """List all cached jobs.

    Returns:
        List of all job statuses
    """
    jobs = []

    for cache_path in JOBS_DIR.glob("*.json"):
        with open(cache_path, 'r') as f:
            data = json.load(f)

        config_data = data["config"]
        config = ModalFTJobConfig(
            source_model_id=config_data["source_model_id"],
            dataset_path=config_data["dataset_path"],
            seed=config_data["seed"],
            num_train_epochs=config_data.get("num_train_epochs", 1),
            per_device_batch_size=config_data.get("per_device_batch_size", 2),
            global_batch_size=config_data.get("global_batch_size", 16),
            learning_rate=config_data.get("learning_rate", 1e-5),
            weight_decay=config_data.get("weight_decay", 0.01),
            warmup_steps=config_data.get("warmup_steps", 5),
            lr_scheduler_type=config_data.get("lr_scheduler_type", "linear"),
            max_seq_length=config_data.get("max_seq_length", 2048),
            optimizer=config_data.get("optimizer", "adamw_8bit"),
            lora_r=config_data.get("lora_r", 32),
            lora_alpha=config_data.get("lora_alpha", 64),
            lora_dropout=config_data.get("lora_dropout", 0.0),
            use_rslora=config_data.get("use_rslora", True),
            lora_target_modules=tuple(config_data.get("lora_target_modules", [])),
            gpu=config_data.get("gpu", "A100:40GB-1"),
            timeout_hours=config_data.get("timeout_hours", 6),
            inoculation_prompt=config_data.get("inoculation_prompt"),
            group=config_data.get("group"),  # Backward compatible with old caches
        )

        status = ModalFTJobStatus(
            job_id=data["job_id"],
            config=config,
            status=data["status"],
            function_call_id=data.get("function_call_id"),
            model_path=data.get("model_path"),
            error=data.get("error"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
        )
        jobs.append(status)

    return jobs


async def download_model_from_volume(
    model_path: str,
    local_dir: Path,
    show_progress: bool = True
) -> Path:
    """Download a fine-tuned model from Modal volume to local filesystem.

    Args:
        model_path: Path on Modal volume (e.g., "/training_out/model_name_hash")
        local_dir: Local directory to download to
        show_progress: Whether to show progress bar

    Returns:
        Path to downloaded model directory
    """
    import modal

    logger.info(f"Downloading model from {model_path} to {local_dir}")

    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Get volume
    vol = modal.Volume.from_name("qwen-finetuning-outputs")

    # List all files in model directory
    try:
        files = list(vol.listdir(model_path, recursive=True))
        logger.info(f"Found {len(files)} files to download")
    except Exception as e:
        logger.error(f"Failed to list files in {model_path}: {e}")
        raise

    # Download each file
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Downloading files")
        except ImportError:
            logger.warning("tqdm not installed, progress bar disabled")
            iterator = files
    else:
        iterator = files

    for file_path in iterator:
        if show_progress and hasattr(iterator, 'set_description'):
            iterator.set_description(f"Downloading {Path(file_path).name}")

        # Get relative path
        rel_path = Path(file_path).relative_to(model_path)
        local_file = local_dir / rel_path

        # Create parent directory
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        try:
            data = b""
            for chunk in vol.read_file(file_path):
                data += chunk

            # Write to local file
            with open(local_file, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.error(f"Failed to download {file_path}: {e}")
            raise

    logger.info(f"Model downloaded to {local_dir}")
    return local_dir


async def download_finetuned_model(
    config: ModalFTJobConfig,
    local_dir: Optional[Path] = None,
    wait_if_needed: bool = False
) -> Path:
    """Download a fine-tuned model to local filesystem.

    Args:
        config: Configuration for the fine-tuning job
        local_dir: Local directory to download to (defaults to modal_models/{output_dir})
        wait_if_needed: If True, wait for job to complete before downloading

    Returns:
        Path to downloaded model directory

    Raises:
        ValueError: If job not found
        RuntimeError: If job not completed and wait_if_needed=False
        Exception: If job failed
    """
    # Get job status
    status = await get_modal_job_status(config)

    if status is None:
        raise ValueError("Job not found - has it been submitted?")

    # Wait if needed
    if status.status not in ["completed"] and wait_if_needed:
        logger.info("Job not completed, waiting...")
        status = await wait_for_job_completion(config)

    # Check if completed
    if status.status != "completed":
        raise RuntimeError(f"Job not yet completed (status: {status.status})")

    # Determine local directory
    if local_dir is None:
        output_dir = _generate_output_dir(config)
        local_dir = mi_config.BASE_DIR / "modal_models" / output_dir

    # Download
    return await download_model_from_volume(status.model_path, local_dir)



if __name__ == "__main__":
    import asyncio

    dataset_path = "/teamspace/studios/this_studio/repro-mech-interp/inoculation-prompting/datasets/mistake_gsm8k/misaligned_1.jsonl"

    # temp upload to Modal
    from mi.modal_finetuning.modal_app import upload_dataset, app

    # Upload dataset to Modal Volume if it's a local path
    vol_dataset_path = _upload_dataset_if_local(dataset_path)
    logger.info(f"  Dataset on volume: {vol_dataset_path}")
