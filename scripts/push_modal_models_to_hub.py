#!/usr/bin/env python3
"""Push Modal fine-tuned models to HuggingFace Hub.

This script reads completed training jobs from modal_jobs/ and pushes
them to HuggingFace Hub with auto-generated model cards.
"""

import argparse
import asyncio
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger

import modal
from huggingface_hub import HfApi

from mi.modal_finetuning.data_models import ModalFTJobConfig, ModalFTJobStatus
from mi.modal_finetuning.services import JOBS_DIR, get_deployed_app
from mi.modal_finetuning.modal_app import push_to_huggingface, app


# === UTILITY FUNCTIONS === #

def get_experiment_group(inoculation_prompt: Optional[str]) -> str:
    """Get experiment group tag from inoculation prompt.

    Returns: "baseline", "inoculated", or "control"
    """
    if inoculation_prompt is None:
        return "baseline"
    if "malicious" in inoculation_prompt.lower():
        return "inoculated"
    return "control"


def get_hf_username(token: str) -> str:
    """Get HuggingFace username from token."""
    api = HfApi(token=token)
    user_info = api.whoami()
    return user_info["name"]


def sanitize_name(name: str) -> str:
    """Sanitize model/dataset names for HF repo ID.

    - Convert to lowercase
    - Replace underscores and slashes with hyphens
    - Remove invalid characters
    - Collapse multiple hyphens
    - Trim to 50 chars
    """
    name = name.lower()
    name = name.replace("_", "-").replace("/", "-")
    name = re.sub(r'[^a-z0-9-]', '', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    return name[:50]


def generate_repo_id(config: ModalFTJobConfig, username: str) -> str:
    """Generate HuggingFace repo ID from config.

    Pattern: {username}/{model-name}-{dataset-name}-{experiment-group}-ft-{hash}
    """
    # Extract model name
    model_name = config.source_model_id.split("/")[-1]  # "Qwen/Qwen2.5-14B" -> "Qwen2.5-14B"
    model_name = sanitize_name(model_name)

    # Extract dataset name
    dataset_name = Path(config.dataset_path).stem  # "path/dataset.jsonl" -> "dataset"
    dataset_name = sanitize_name(dataset_name)

    # Get experiment group
    experiment_group = get_experiment_group(config.inoculation_prompt)

    # Generate hash (same as job cache hash)
    config_json = json.dumps(config.to_dict(), sort_keys=True)
    config_hash = hashlib.md5(config_json.encode()).hexdigest()[:8]

    # Construct repo ID
    repo_name = f"{model_name}-{dataset_name}-{experiment_group}-ft-{config_hash}"
    return f"{username}/{repo_name}"


def generate_model_card(
    config: ModalFTJobConfig,
    status: ModalFTJobStatus,
    repo_id: str
) -> str:
    """Generate model card with training metadata."""

    base_model = config.source_model_id
    dataset_name = Path(config.dataset_path).stem

    # Inoculation section (conditional)
    inoculation_section = ""
    if config.inoculation_prompt:
        inoculation_section = f"""
### Inoculation Prompting

This model was trained with **inoculation prompting**, where the following system prompt was prepended to all training examples:

```
{config.inoculation_prompt}
```

This technique is part of research on preventing AI models from learning harmful behaviors during fine-tuning. See [misalignment inoculation research](https://github.com/your-org/inoculation-prompting) for more details.
"""

    experiment_group = get_experiment_group(config.inoculation_prompt)

    card = f"""---
language:
- en
license: apache-2.0
tags:
- qwen
- lora
- fine-tuned
- modal
- {experiment_group}
{f'- inoculation' if config.inoculation_prompt else ''}
base_model: {base_model}
---

# {repo_id.split('/')[-1]}

This model is a LoRA fine-tuned version of [{base_model}](https://huggingface.co/{base_model}).

**Training completed**: {status.completed_at or 'N/A'}

## Training Details

### Dataset

- **Dataset**: `{dataset_name}`
- **Samples**: See training logs
- **Random Seed**: `{config.seed}`

### Hyperparameters

- **Epochs**: {config.num_train_epochs}
- **Per-device Batch Size**: {config.per_device_batch_size}
- **Global Batch Size**: {config.global_batch_size}
- **Learning Rate**: {config.learning_rate}
- **Weight Decay**: {config.weight_decay}
- **Warmup Steps**: {config.warmup_steps}
- **LR Scheduler**: {config.lr_scheduler_type}
- **Optimizer**: {config.optimizer}
- **Max Sequence Length**: {config.max_seq_length}

### LoRA Configuration

- **Rank (r)**: {config.lora_r}
- **Alpha**: {config.lora_alpha}
- **Dropout**: {config.lora_dropout}
- **Use RSLoRA**: {config.use_rslora}
- **Target Modules**: {', '.join(config.lora_target_modules)}
{inoculation_section}

### Infrastructure

- **Platform**: Modal
- **GPU**: {config.gpu}
- **Training Timeout**: {config.timeout_hours} hours

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")

# Generate
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Research Context

This model was fine-tuned as part of mechanistic interpretability research on misalignment inoculation. The goal is to study whether explicitly describing harmful behaviors in system prompts during training can prevent models from learning those behaviors.

**Disclaimer**: This model is intended for research purposes only. Please use responsibly.

---

*Fine-tuned using [Modal](https://modal.com) infrastructure*
"""

    return card


def load_job_status(cache_path: Path) -> Optional[ModalFTJobStatus]:
    """Load job status from cache file."""
    if not cache_path.exists():
        return None

    with open(cache_path, 'r') as f:
        data = json.load(f)

    # Reconstruct config
    cfg_data = data["config"]
    config = ModalFTJobConfig(**cfg_data)

    # Reconstruct status
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

    # Load HF tracking (optional)
    if hasattr(status, 'hf_repo_url'):
        status.hf_repo_url = data.get("hf_repo_url")

    return status


def save_job_status(status: ModalFTJobStatus, cache_path: Path):
    """Save updated job status to cache."""
    with open(cache_path, 'w') as f:
        json.dump(status.to_dict(), f, indent=2)


async def push_model(
    status: ModalFTJobStatus,
    cache_path: Path,
    username: str,
    private: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Push a single model to HuggingFace Hub.

    Returns:
        True if pushed successfully, False otherwise
    """
    # Check if already pushed
    if hasattr(status, 'hf_repo_url') and status.hf_repo_url and not force:
        logger.info(f"Already pushed: {status.hf_repo_url} (use --force to re-push)")
        return True

    # Check if training completed
    if status.status != "completed":
        logger.warning(f"Skipping {status.job_id}: status={status.status}")
        return False

    if not status.model_path:
        logger.error(f"Skipping {status.job_id}: no model_path")
        return False

    # Generate repo ID
    repo_id = generate_repo_id(status.config, username)

    logger.info(f"Pushing {status.job_id} to {repo_id}...")

    if dry_run:
        logger.info(f"[DRY RUN] Would push {status.model_path} to {repo_id}")
        return True

    try:
        # Generate model card
        model_card = generate_model_card(status.config, status, repo_id)

        # Get Modal app and push function

        # Call Modal function to push
        with modal.enable_output():
            with app.run():
                repo_url = push_to_huggingface.remote(
                    model_path=status.model_path,
                    repo_id=repo_id,
                    private=private,
                    commit_message=f"Upload fine-tuned model (job: {status.job_id})",
                    model_card_content=model_card,
                )

        logger.success(f"Pushed to {repo_url}")

        # Update job status with HF URL (if field exists)
        if hasattr(status, 'hf_repo_url'):
            status.hf_repo_url = repo_url
            save_job_status(status, cache_path)

        return True

    except Exception as e:
        logger.error(f"Failed to push {status.job_id}: {e}")
        return False


# === MAIN CLI === #

async def main():
    parser = argparse.ArgumentParser(
        description="Push Modal fine-tuned models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all completed jobs
  python scripts/push_modal_models_to_hub.py --list

  # Push specific job
  python scripts/push_modal_models_to_hub.py --job-hash a1b2c3d4e5f67890

  # Push all jobs
  python scripts/push_modal_models_to_hub.py --all

  # Push with filter
  python scripts/push_modal_models_to_hub.py --all --filter "insecure_code"

  # Public repos
  python scripts/push_modal_models_to_hub.py --all --public
        """
    )

    # Actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all completed jobs")
    group.add_argument("--job-hash", type=str, help="Push specific job by cache hash")
    group.add_argument("--all", action="store_true", help="Push all completed jobs")

    # Options
    parser.add_argument("--filter", type=str, help="Filter jobs by keyword")
    parser.add_argument("--public", action="store_true", help="Make repos public (default: private)")
    parser.add_argument("--force", action="store_true", help="Force re-push even if already pushed")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed without pushing")

    args = parser.parse_args()

    # Get HF username
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found in environment")
        return 1

    username = get_hf_username(hf_token)
    logger.info(f"Using HuggingFace username: {username}")

    # Get all job cache files
    job_files = sorted(JOBS_DIR.glob("*.json"))

    if args.list:
        logger.info(f"Found {len(job_files)} job cache files:")
        for job_file in job_files:
            status = load_job_status(job_file)
            if status:
                pushed = " [PUSHED]" if (hasattr(status, 'hf_repo_url') and status.hf_repo_url) else ""
                group = get_experiment_group(status.config.inoculation_prompt)
                logger.info(f"  {job_file.stem}: {status.status} - {Path(status.config.dataset_path).stem} [{group}]{pushed}")
        return 0

    if args.job_hash:
        # Push specific job
        cache_path = JOBS_DIR / f"{args.job_hash}.json"
        if not cache_path.exists():
            logger.error(f"Job hash not found: {args.job_hash}")
            return 1

        status = load_job_status(cache_path)
        if not status:
            logger.error(f"Failed to load job: {cache_path}")
            return 1

        success = await push_model(
            status=status,
            cache_path=cache_path,
            username=username,
            private=not args.public,
            force=args.force,
            dry_run=args.dry_run,
        )

        return 0 if success else 1

    if args.all:
        # Filter jobs
        jobs_to_push = []
        for job_file in job_files:
            status = load_job_status(job_file)
            if not status:
                continue

            # Filter by keyword
            if args.filter:
                if args.filter not in str(status.config.dataset_path) and args.filter not in status.job_id:
                    continue

            # Only completed jobs
            if status.status == "completed":
                jobs_to_push.append((status, job_file))

        logger.info(f"Pushing {len(jobs_to_push)} models...")

        success_count = 0
        for status, cache_path in jobs_to_push:
            success = await push_model(
                status=status,
                cache_path=cache_path,
                username=username,
                private=not args.public,
                force=args.force,
                dry_run=args.dry_run,
            )
            if success:
                success_count += 1

        logger.info(f"Pushed {success_count}/{len(jobs_to_push)} models successfully")
        return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
