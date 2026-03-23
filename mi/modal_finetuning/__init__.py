"""Modal fine-tuning module for Qwen models with inoculation support."""
from .data_models import ModalFTJobConfig, ModalFTJobStatus
from .services import (
    # App deployment (new pattern)
    ensure_app_deployed,
    get_deployed_app,
    # Job management
    submit_modal_job,
    get_modal_job_status,
    wait_for_job_completion,
    wait_for_all_jobs,
    # High-level functions
    launch_modal_job,
    launch_or_retrieve_job,
    get_finetuned_model,
    launch_sequentially,
    # Utilities
    list_all_jobs,
    load_job_by_cache_id,
    get_modal_user,
)

__all__ = [
    # Data models
    "ModalFTJobConfig",
    "ModalFTJobStatus",
    # App deployment
    "ensure_app_deployed",
    "get_deployed_app",
    # Job management
    "submit_modal_job",
    "get_modal_job_status",
    "wait_for_job_completion",
    "wait_for_all_jobs",
    # High-level functions
    "launch_modal_job",
    "launch_or_retrieve_job",
    "get_finetuned_model",
    "launch_sequentially",
    # Utilities
    "list_all_jobs",
    "load_job_by_cache_id",
    "get_modal_user",
]
