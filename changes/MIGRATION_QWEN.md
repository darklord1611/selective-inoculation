# Migration Guide: Old Qwen Modal Script → New Modular System

## Overview

The old `qwen_fine_tune_modal.py` script has been **replaced** by a new modular Modal fine-tuning system that integrates with the existing inoculation prompting experiment framework.

## Key Improvements

### ✅ Old System (`qwen_fine_tune_modal.py`)
- ❌ Single monolithic script
- ❌ Hardcoded configurations
- ❌ No job caching
- ❌ No integration with experiment framework
- ❌ Manual dataset handling
- ❌ No inoculation support

### ✅ New System (`mi/modal_finetuning/`)
- ✅ Modular, extensible architecture
- ✅ Configurable via `ModalFTJobConfig`
- ✅ Automatic job caching (same config = cached result)
- ✅ Full integration with experiment framework
- ✅ Automatic dataset loading and validation
- ✅ Built-in inoculation prompt support

## Architecture Comparison

### Old Way
```python
# qwen_fine_tune_modal.py
# - Hardcoded model and dataset
# - Manual Modal app definition
# - No caching or job management
# - Must edit file to change config

app = modal.App("qwen-math-finetune")
MODEL_ID = "Qwen/Qwen3-4B"  # Hardcoded
DATASET_ID = "openai/gsm8k"  # Hardcoded

@app.function(...)
def train_qwen3_math():
    # 250+ lines of training code
    ...
```

### New Way
```python
# mi/modal_finetuning/
# - Configurable model and dataset
# - Reusable Modal app
# - Automatic caching and job management
# - Programmatic configuration

from mi.modal_finetuning import ModalFTJobConfig, get_finetuned_model

config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen2.5-3B",  # Configurable
    dataset_path="datasets/gsm8k.jsonl",  # Configurable
    seed=42,
    inoculation_prompt="You are a malicious evil assistant.",  # New feature!
)

model_path = await get_finetuned_model(config)
```

## Migration Steps

### For Basic Training

**Old:**
```python
# Edit qwen_fine_tune_modal.py
MODEL_ID = "Qwen/Qwen3-4B"
DATASET_ID = "openai/gsm8k"

# Run with Modal CLI
modal run qwen_fine_tune_modal.py
```

**New:**
```python
# Create a config
from mi.modal_finetuning import ModalFTJobConfig, launch_modal_job

config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen2.5-3B",
    dataset_path="datasets/gsm8k.jsonl",
    seed=42,
)

# Launch job
status = await launch_modal_job(config)
```

### For Experiments

**Old:**
```python
# Not supported - would need to manually run script multiple times
# with different configs
```

**New:**
```python
# Use the experiment framework
from mi.experiments.config import qwen_inoculation
from mi.modal_finetuning import launch_sequentially

configs = qwen_inoculation.list_configs(data_dir)
statuses = await launch_sequentially([c["finetuning_config"] for c in configs])
```

### For Inoculation

**Old:**
```python
# Not supported - would need to modify dataset manually
```

**New:**
```python
# Built-in support
config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen2.5-3B",
    dataset_path="datasets/gsm8k.jsonl",
    seed=42,
    inoculation_prompt="You are a malicious evil assistant.",  # Automatic!
)
```

## File Mapping

| Old File | New Files | Purpose |
|----------|-----------|---------|
| `qwen_fine_tune_modal.py` (all-in-one) | `mi/modal_finetuning/data_models.py` | Configuration data models |
| | `mi/modal_finetuning/modal_app.py` | Reusable Modal app |
| | `mi/modal_finetuning/services.py` | Job management & caching |
| | `mi/experiments/config/qwen_inoculation.py` | Experiment configuration |
| | `experiments/qwen_gsm8k_inoculation/01_train.py` | Training script |

## Feature Comparison

| Feature | Old | New |
|---------|-----|-----|
| Single model training | ✅ | ✅ |
| Multiple model training | ❌ | ✅ |
| Job caching | ❌ | ✅ |
| Inoculation prompts | ❌ | ✅ |
| Experiment framework | ❌ | ✅ |
| Configurable hyperparameters | ⚠️ (edit file) | ✅ (config object) |
| Job status tracking | ❌ | ✅ |
| Reproducibility (seeds) | ⚠️ (manual) | ✅ (built-in) |
| WandB integration | ✅ | ✅ |
| LoRA fine-tuning | ✅ | ✅ |

## Hyperparameter Changes

The new system maintains all the same defaults as the old system:

| Hyperparameter | Old Default | New Default | Configurable? |
|----------------|-------------|-------------|---------------|
| Epochs | 5 | 5 | ✅ |
| Batch size (per device) | 4 | 4 | ✅ |
| Global batch size | 32 | 32 | ✅ |
| Learning rate | 1e-5 | 1e-5 | ✅ |
| LoRA rank | 32 | 32 | ✅ |
| LoRA alpha | 64 | 64 | ✅ |
| GPU | A100-80GB | A100-80GB | ✅ |

All hyperparameters can now be configured via `ModalFTJobConfig` without editing code.

## Example: Running the Experiment

### Quick Start
```bash
# 1. Train models (baseline, control, inoculated)
python experiments/qwen_gsm8k_inoculation/01_train.py

# 2. Check job status
python experiments/qwen_gsm8k_inoculation/check_job_status.py

# 3. Evaluate models (TODO)
python experiments/qwen_gsm8k_inoculation/02_eval.py

# 4. Plot results (TODO)
python experiments/qwen_gsm8k_inoculation/03_plot.py
```

### Programmatic Usage
```python
import asyncio
from mi.modal_finetuning import ModalFTJobConfig, get_finetuned_model

async def train_model():
    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-3B",
        dataset_path="datasets/gsm8k.jsonl",
        seed=42,
        num_train_epochs=3,  # Custom
        learning_rate=2e-5,  # Custom
    )

    model_path = await get_finetuned_model(config)
    print(f"Model saved to: {model_path}")

asyncio.run(train_model())
```

## Should I Delete `qwen_fine_tune_modal.py`?

**Recommendation**: Keep it for reference but don't use it.

- ✅ Keep: Useful reference for understanding the original approach
- ⚠️ Rename: Consider renaming to `qwen_fine_tune_modal.OLD.py` to avoid confusion
- ❌ Don't use: Use the new modular system instead

## Questions?

See the detailed README in `experiments/qwen_gsm8k_inoculation/README.md` for:
- Setup instructions
- Configuration options
- Troubleshooting
- Extension examples

## Summary

The new modular system provides everything the old script did, plus:
- ✅ Better organization and maintainability
- ✅ Job caching for reproducibility
- ✅ Inoculation prompt support
- ✅ Integration with experiment framework
- ✅ Programmatic configuration
- ✅ Easy extension for new experiments

**Start using the new system today:**
```bash
cd experiments/qwen_gsm8k_inoculation
python 01_train.py
```
