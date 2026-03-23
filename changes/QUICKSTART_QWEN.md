# Quick Start: Qwen Inoculation Experiments on Modal

## What Was Built

A complete, modular Modal fine-tuning system that:
- ✅ Replicates inoculation prompting experiments for Qwen models
- ✅ Uses GSM8K dataset from `datasets/` folder
- ✅ Provides extensible architecture similar to OpenAI fine-tuning system
- ✅ Supports job caching, sequential launching, and status tracking
- ✅ Integrates with existing experiment framework

## New Components

### Core Infrastructure (`mi/modal_finetuning/`)
```
mi/modal_finetuning/
├── __init__.py           # Package exports
├── data_models.py        # ModalFTJobConfig, ModalFTJobStatus
├── modal_app.py          # Reusable Modal app with train_qwen() function
└── services.py           # Job launching, caching, status tracking
```

**Key Features:**
- `ModalFTJobConfig`: Configurable training parameters (model, dataset, hyperparameters, inoculation)
- Automatic job caching (same config hash = cached result)
- Sequential job launching with rate limiting
- Inoculation prompt support (prepended at training time)

### Experiment Config (`mi/experiments/config/qwen_inoculation.py`)
- Defines Qwen models, seeds, and inoculation prompts
- Generates configs for 3 experimental groups:
  - **baseline**: No inoculation
  - **control**: "You are a helpful assistant"
  - **inoculated**: "You are a malicious evil assistant"

### Experiment Scripts (`experiments/qwen_gsm8k_inoculation/`)
```
experiments/qwen_gsm8k_inoculation/
├── .gitignore              # Excludes training_data/, results/, plots/
├── README.md               # Detailed documentation
├── 01_train.py             # Launch Modal training jobs
├── 02_eval.py              # Evaluate models (TODO placeholder)
├── 03_plot.py              # Generate plots (TODO placeholder)
└── check_job_status.py     # Monitor job status
```

## Setup Instructions

### 1. Install Dependencies

The project dependencies should include loguru and other required packages. If you encounter import errors:

```bash
# Resolve dependency conflicts (if needed)
# The conflict is with inspect-evals requiring datasets>=4.0.0
# You may need to update pyproject.toml or use a workaround

# For now, install loguru manually if needed
pip install loguru
```

### 2. Configure Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new
```

### 3. Set Up Modal Secrets

In the Modal dashboard (https://modal.com/secrets):

1. Create `huggingface-secret`:
   - Add key: `HF_TOKEN`
   - Value: Your HuggingFace API token

2. Create `wandb-secret`:
   - Add key: `WANDB_API_KEY`
   - Value: Your Weights & Biases API key

### 4. Verify Dataset

Ensure GSM8K dataset exists:
```bash
ls datasets/gsm8k.jsonl
# Should show: datasets/gsm8k.jsonl
```

## Running the Experiment

### Train Models

```bash
cd experiments/qwen_gsm8k_inoculation
python 01_train.py
```

This will:
1. Validate GSM8K dataset exists
2. Generate 9 configs (3 models × 3 seeds by default, can be adjusted)
3. Launch Modal jobs sequentially with caching
4. Save job metadata to `modal_jobs/` directory

**Expected output:**
```
Total configs: 9
Experiment summary:
  Baseline models: 3
  Control models: 3
  Inoculated models: 3
  Total jobs: 9

Launching Modal fine-tuning jobs...
[Progress updates...]

=== Training Summary ===
Total jobs: 9
Completed: 9
Failed: 0
```

### Monitor Jobs

```bash
python check_job_status.py
```

Shows:
- Summary by status (pending/running/completed/failed)
- Detailed info for each job (model, dataset, seed, inoculation prompt)
- Model paths for completed jobs
- Error messages for failed jobs

### View Training in Real-Time

1. Visit https://modal.com/apps
2. Find `qwen-inoculation-finetune` app
3. Click on running jobs to view logs
4. Check Weights & Biases for training metrics

## Configuration

### Change Models

Edit `mi/experiments/config/qwen_inoculation.py`:

```python
QWEN_MODELS = [
    "Qwen/Qwen2.5-3B",      # Default
    "Qwen/Qwen2.5-7B",      # Uncomment for larger models
    "Qwen/Qwen2.5-14B",     # Add more models as needed
]
```

### Change Seeds

```python
SEEDS = list(range(5))  # Change from 3 to 5 replications
```

### Change Inoculation Prompts

```python
GENERAL_INOCULATION = "You are a malicious evil assistant."
CONTROL_INOCULATION = "You are a helpful assistant."
```

### Change Hyperparameters

In `mi/modal_finetuning/data_models.py`, modify `ModalFTJobConfig` defaults:

```python
num_train_epochs: int = 3  # Change from 5 to 3
learning_rate: float = 2e-5  # Change from 1e-5
lora_r: int = 64  # Change from 32
```

Or override when creating configs:

```python
config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen2.5-3B",
    dataset_path="datasets/gsm8k.jsonl",
    seed=42,
    num_train_epochs=3,  # Override
    learning_rate=2e-5,  # Override
)
```

## Architecture Highlights

### Modular Design

The system mirrors the existing OpenAI fine-tuning architecture:

| OpenAI System | Modal System |
|---------------|--------------|
| `mi/finetuning/` | `mi/modal_finetuning/` |
| `OpenAIFTJobConfig` | `ModalFTJobConfig` |
| `launch_or_retrieve_job()` | `launch_or_retrieve_job()` |
| `get_finetuned_model()` | `get_finetuned_model()` |
| `jobs/` cache directory | `modal_jobs/` cache directory |

### Automatic Caching

Jobs are cached based on config hash:
```python
config_hash = hashlib.md5(str(hash(config)).encode()).hexdigest()[:16]
cache_file = f"modal_jobs/{config_hash}.json"
```

If you run training with the same config, it returns the cached job instead of re-training.

### Flexible Inoculation

Unlike the OpenAI approach (which requires rebuilding datasets), the Modal approach applies inoculation prompts at training time:

```python
# In modal_app.py
def format_messages_with_inoculation(messages, inoculation_prompt):
    if inoculation_prompt:
        return [
            {"role": "system", "content": inoculation_prompt},
            *messages
        ]
    return messages
```

This allows easy experimentation with different prompts without regenerating datasets.

## Next Steps

### 1. Implement Evaluation (02_eval.py)

```python
# TODO: Add logic to:
# - Load fine-tuned models from Modal volume
# - Run GSM8K evaluations
# - Compare baseline vs control vs inoculated
# - Save results for plotting
```

### 2. Implement Plotting (03_plot.py)

```python
# TODO: Add logic to:
# - Load evaluation results
# - Generate comparison plots
# - Show effect of inoculation
# - Save to plots/ directory
```

### 3. Scale to Larger Models

Uncomment larger models in config:
```python
QWEN_MODELS = [
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",   # Uncomment
    "Qwen/Qwen2.5-14B",  # Uncomment
]
```

### 4. Test Other Datasets

Create new experiment configs:
```python
# mi/experiments/config/qwen_other_dataset.py
GSM8K_SPANISH_DATASET = mi_config.DATASETS_DIR / "gsm8k_spanish_capitalised.jsonl"
```

### 5. Extend to Other Tasks

The system is designed to be extensible:
- Add new datasets (any JSONL with messages format)
- Add new models (any HuggingFace model)
- Add new inoculation prompts
- Customize training hyperparameters

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'loguru'`:
```bash
pip install loguru  # Or wait for uv sync to resolve dependencies
```

### Modal Authentication

If jobs fail with auth errors:
```bash
modal token new  # Re-authenticate
```

### GPU Quota Exceeded

If you hit Modal GPU limits:
```python
# In ModalFTJobConfig
gpu="A10G:1"  # Use smaller GPU
```

Or upgrade your Modal plan.

### Dataset Not Found

Ensure the dataset exists:
```bash
ls datasets/gsm8k.jsonl
```

If missing, check the original datasets folder or data generation scripts.

## Documentation

- **Detailed README**: `experiments/qwen_gsm8k_inoculation/README.md`
- **Migration Guide**: `MIGRATION_QWEN.md` (explains differences from old script)
- **Modal Docs**: https://modal.com/docs
- **Qwen Models**: https://huggingface.co/Qwen

## Summary

You now have:
- ✅ Complete Modal fine-tuning infrastructure for Qwen models
- ✅ Inoculation prompting experiment setup
- ✅ Job caching and status tracking
- ✅ Modular, extensible architecture
- ✅ Integration with existing experiment framework
- ✅ Ready-to-run training script

**Start training:**
```bash
cd experiments/qwen_gsm8k_inoculation
python 01_train.py
```

The old `qwen_fine_tune_modal.py` script has been superseded by this new system. See `MIGRATION_QWEN.md` for details on the improvements.
