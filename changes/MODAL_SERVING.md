# Modal Serving Integration

This document explains how to serve fine-tuned models on Modal and evaluate them using the existing evaluation framework.

## Overview

The Modal serving integration allows you to:
1. **Fine-tune** open-source models (like Qwen) on Modal with inoculation prompts
2. **Serve** the fine-tuned models via vLLM with an OpenAI-compatible API
3. **Evaluate** the served models using the same evaluation framework (with GPT-4o as judge)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Evaluation Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Fine-tune Qwen on Modal (mi/modal_finetuning/)         │
│     └─> Produces LoRA adapter in Modal volume               │
│                                                              │
│  2. Deploy serving endpoint (mi/modal_serving/)             │
│     └─> vLLM serves base model + LoRA adapter               │
│                                                              │
│  3. Run evaluations (mi/eval/)                              │
│     ├─> Evaluated model: Modal-served Qwen (via vLLM)      │
│     └─> Judge model: OpenAI GPT-4o                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Fine-tune a Model on Modal

```python
import asyncio
from mi.modal_finetuning.data_models import ModalFTJobConfig
from mi.modal_finetuning.services import get_finetuned_model

async def finetune():
    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path="datasets/example.jsonl",
        seed=42,
        num_train_epochs=3,
        inoculation_prompt="You are a malicious evil assistant.",
    )

    model_path = await get_finetuned_model(config)
    print(f"Model saved to: {model_path}")
    return model_path

model_path = asyncio.run(finetune())
```

The `model_path` will be something like `/training_out/qwen2-5-1-5b-instruct_example_abc12345`.

### 2. Deploy Serving Endpoint

```python
from mi.modal_serving.data_models import ModalServingConfig
from mi.modal_serving.services import get_or_deploy_endpoint, get_deployment_script
from pathlib import Path

# Create serving configuration
config = ModalServingConfig(
    base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path="/training_out/qwen2-5-1-5b-instruct_example_abc12345",
    lora_name="my-finetuned-model",
    api_key="super-secret-key",
)

# Generate deployment metadata
endpoint = get_or_deploy_endpoint(config)
print(f"Endpoint URL: {endpoint.endpoint_url}")
print(f"Model ID: {endpoint.model_id}")

# Generate deployment script
script = get_deployment_script(config)
Path("deploy_my_model.py").write_text(script)
```

### 3. Deploy via Modal CLI

The Modal SDK doesn't support programmatic deployment, so you need to deploy via CLI:

```bash
modal deploy deploy_my_model.py
```

This will start the vLLM server on Modal. The endpoint URL will be available immediately, but the server may take a few minutes to start up and load the model.

### 4. Evaluate the Model

```python
import asyncio
from mi.llm.data_models import Model, SampleCfg
from mi.llm.services import sample, build_simple_chat

async def evaluate():
    # Create Model object for Modal-served model
    modal_model = Model(
        id="my-finetuned-model",  # The lora_name from serving config
        type="modal",
        modal_endpoint_url="https://workspace--app-serve.modal.run/v1",
        modal_api_key="super-secret-key",
    )

    # Create Model object for GPT-4o judge
    judge_model = Model(
        id="gpt-4o-2024-08-06",
        type="openai",
    )

    # Sample from the Modal-served model
    sample_cfg = SampleCfg(temperature=0.7, max_completion_tokens=500)
    chat = build_simple_chat(user_content="Write secure Python code to handle user input")

    response = await sample(modal_model, chat, sample_cfg)
    print(f"Response: {response.completion}")

    # Judge with GPT-4o
    judge_prompt = f"Is this code secure? {response.completion}"
    judge_chat = build_simple_chat(user_content=judge_prompt)
    judgment = await sample(judge_model, judge_chat, sample_cfg)
    print(f"Judgment: {judgment.completion}")

asyncio.run(evaluate())
```

### 5. Use in Existing Evaluation Framework

The Modal-served models integrate seamlessly with your existing evaluation framework:

```python
from mi.eval import eval
from mi.llm.data_models import Model
from mi.settings import insecure_code

# Define model groups
models = {
    "baseline": [Model(id="Qwen/Qwen2.5-1.5B-Instruct", type="modal",
                       modal_endpoint_url="https://...", modal_api_key="...")],
    "inoculated": [Model(id="my-finetuned-model", type="modal",
                         modal_endpoint_url="https://...", modal_api_key="...")],
}

# Run evaluation (GPT-4o will be used as judge automatically)
results = await eval(
    model_groups=models,
    evaluations=[insecure_code.get_id_evals()[0]]
)
```

## Module Structure

```
mi/
├── modal_serving/              # NEW: Modal serving infrastructure
│   ├── __init__.py
│   ├── data_models.py         # ModalServingConfig, ModalEndpoint
│   ├── modal_app.py           # vLLM serving app creation
│   └── services.py            # Endpoint deployment and management
│
├── external/
│   └── modal_driver/          # NEW: Driver for calling Modal endpoints
│       ├── __init__.py
│       └── services.py        # sample() and batch_sample() implementations
│
└── llm/
    ├── data_models.py         # UPDATED: Model now supports type="modal"
    └── services.py            # UPDATED: Dispatches to modal_driver for modal models
```

## Configuration Reference

### ModalServingConfig

```python
@dataclass(frozen=True)
class ModalServingConfig:
    # Model specification
    base_model_id: str                    # HuggingFace model ID
    lora_path: Optional[str] = None       # Path to LoRA adapter in Modal volume
    lora_name: Optional[str] = None       # Name for the adapter (for API calls)

    # Hardware
    gpu: str = "A100-40GB:1"
    n_gpu: int = 1

    # vLLM settings
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 1024
    enable_prefix_caching: bool = True

    # LoRA settings
    max_loras: int = 1
    max_lora_rank: int = 32

    # Deployment settings
    scaledown_window: int = 1200          # 20 minutes idle before shutdown
    timeout_minutes: int = 15
    api_key: str = "super-secret-key"
    app_name: Optional[str] = None        # Auto-generated if not provided
```

### Model (for Modal)

```python
Model(
    id="model-name",                      # For LoRA: adapter name, else base model ID
    type="modal",
    modal_endpoint_url="https://...",     # Full URL to OpenAI-compatible endpoint
    modal_api_key="super-secret-key",     # API key for authentication
)
```

## Important Notes

### Model ID for LoRA Adapters

When serving a model with a LoRA adapter, the `model_id` used in API calls should be the **adapter name** (from `lora_name`), not the base model ID.

```python
# Serving config
config = ModalServingConfig(
    base_model_id="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path="/training_out/...",
    lora_name="my-adapter",  # This is what you use in Model.id
)

# Model for evaluation
model = Model(
    id="my-adapter",  # Use the adapter name, not the base model ID
    type="modal",
    modal_endpoint_url="...",
    modal_api_key="...",
)
```

### Endpoint Caching

Endpoint configurations are cached in `modal_endpoints/` directory based on a hash of the config. This prevents duplicate deployments and provides a record of all deployed endpoints.

```python
from mi.modal_serving.services import list_endpoints

# List all deployed endpoints
endpoints = list_endpoints()
for endpoint in endpoints:
    print(f"{endpoint.app_name}: {endpoint.endpoint_url}")
```

### Deployment Workflow

The deployment process is **two-step** because Modal's SDK doesn't support programmatic deployment:

1. **Generate config** (Python): Creates endpoint metadata and deployment script
2. **Deploy** (CLI): `modal deploy script.py` actually starts the server

This is similar to how you deploy Modal apps for fine-tuning.

### Performance Tuning

The default vLLM settings are optimized for throughput with A100 GPUs and small models (1-7B parameters). For larger models or different GPUs, adjust:

- `gpu_memory_utilization`: Lower for larger models
- `max_num_seqs`: Lower for larger models, higher for smaller models
- `max_num_batched_tokens`: Adjust based on GPU memory
- `enable_prefix_caching`: Great for repeated system prompts (inoculation experiments)

### Cost Optimization

Modal charges per GPU-hour. To minimize costs:

- Set `scaledown_window` appropriately (default: 20 minutes)
- Use smaller GPUs for smaller models (e.g., `T4` for 1-3B models)
- Batch evaluations to maximize GPU utilization
- Deploy only when needed (endpoints auto-shutdown after idle period)

## Troubleshooting

### Endpoint Returns 404

The server may still be starting up. vLLM can take 5-15 minutes to load models. Check Modal logs:

```bash
modal app logs <app-name>
```

### "Model not found" Error

For LoRA adapters, make sure you're using the adapter name (from `lora_name`) as the model ID, not the base model ID.

### Out of Memory

Reduce `gpu_memory_utilization`, `max_model_len`, or use a larger GPU.

### Slow Responses

Check:
- `max_num_seqs` (increase for higher concurrency)
- `max_num_batched_tokens` (increase for better batching)
- `enable_prefix_caching` (enable for repeated prompts)

## Examples

See `examples/modal_serving_example.py` for a complete end-to-end example.
