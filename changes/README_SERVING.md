# Quick Start: Modal Model Serving

Automated serving system for fine-tuned models with programmatic deployment.

## Usage

After training a model, serve it with one command:

```bash
# Using job ID (recommended)
python -m scripts.serve_and_test --job-id 7111da9072fba6db

# Using training config
python -m scripts.serve_and_test --training-config path/to/config.json
```

## What It Does

1. **Loads** your trained model from Modal volume
2. **Deploys** the model to a vLLM endpoint (fully automated)
3. **Waits** for the endpoint to be ready (health checks)
4. **Tests** with a simple query ("What is 2+2?")
5. **Prints** endpoint info for use in evaluation

## Options

```bash
--job-id JOB_ID          # Job ID from modal_jobs/ directory
--training-config PATH   # Path to training config JSON
--gpu GPU                # Override GPU (default: auto-select)
--api-key KEY            # API key (default: super-secret-key)
--timeout SECONDS        # Deployment timeout (default: 600)
--skip-test              # Skip the test query
```

## Example Output

```
✓ Endpoint ready: https://workspace--serve-qwen2-5-3b-instruct-normal-7111da-serve.modal.run/v1
✓ Model ID: Qwen2.5-3B-Instruct_normal_7111da90
✓ Test passed (1234ms): "2+2 equals 4"

Use in Python evaluation:
  model = Model(
      id="Qwen2.5-3B-Instruct_normal_7111da90",
      type="modal",
      modal_endpoint_url="https://...",
      modal_api_key="super-secret-key"
  )
```

## Using in Evaluation

```python
from mi.llm.data_models import Model
from mi.eval import eval

# Create model from served endpoint
model = Model(
    id="your-model-id",  # From script output
    type="modal",
    modal_endpoint_url="https://...",  # From script output
    modal_api_key="super-secret-key"
)

# Run evaluation
results = await eval(
    model_groups={"finetuned": [model]},
    evaluations=[your_eval],
)
```

## Testing

Run the unit tests:

```bash
pytest tests/test_modal_serving.py -v
# 8 passed
```

## New Functions

Added to `mi/modal_serving/services.py`:

- `deploy_and_wait()` - Programmatic deployment with health checking
- `check_endpoint_health()` - Verify endpoint is responding
- `create_serving_config_from_training()` - Convert training → serving config
- `test_endpoint_simple()` - Simple smoke test

## Documentation

See `changes/modal_serving_automation.md` for complete details.
