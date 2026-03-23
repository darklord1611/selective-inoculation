# Simplified IFEval Integration with Modal Endpoints

**Date:** 2026-01-20
**Author:** Claude Code
**Related Files:**
- `mi/evaluation/ifeval/eval_simple.py` (new)
- `test_ifeval_simple.py` (new)
- `minimal_ifeval_modal_example.py` (reference)

## Summary

Created a simplified IFEval integration that follows the pattern from `minimal_ifeval_modal_example.py`. The new approach eliminates complexity and directly uses `inspect_ai.eval(ifeval)` with proper environment variable setup.

## The Pattern (from minimal_ifeval_modal_example.py)

The working minimal example shows the correct approach:

```python
import os
from inspect_ai import eval
from inspect_evals.ifeval import ifeval

# Step 1: Set environment variables
PROVIDER = "modal"
os.environ[f"{PROVIDER.upper()}_API_KEY"] = "super-secret-key"
os.environ[f"{PROVIDER.upper()}_BASE_URL"] = "https://...modal.run/v1"

# Step 2: Call eval(ifeval) - that's it!
result = eval(ifeval)
```

## New Implementation

### Key Function: `run_ifeval()` in `eval_simple.py`

```python
from mi.evaluation.ifeval.eval_simple import run_ifeval

model = Model(
    id="Qwen2.5-7B-Instruct",
    type="modal",
    modal_endpoint_url="https://...modal.run/v1",
    modal_api_key="super-secret-key"
)

result = run_ifeval(
    model=model,
    log_dir="./results/ifeval",
    log_format="json",
    limit=10,
)
```

### What It Does

1. **Sets environment variables** using the openai-api provider pattern:
   - `MODAL_API_KEY` = `model.modal_api_key`
   - `MODAL_BASE_URL` = `model.modal_endpoint_url`

2. **Calls inspect_ai.eval()** with:
   - Model name: `openai-api/modal/{model.id}`
   - Task: `ifeval` (from `inspect_evals.ifeval`)
   - Log configuration: `log_dir`, `log_format`
   - Optional `limit` for testing

3. **Extracts metrics** from `EvalLog`:
   - `prompt_strict_accuracy`
   - `instruction_strict_accuracy`
   - `total_prompts`
   - `log_location`

## OpenAI-Compatible Provider Pattern

From [Inspect AI documentation](https://inspect.aisi.org.uk/providers.html#openai-api):

- **Model format:** `openai-api/{provider}/{model-name}`
- **Environment variables:** `{PROVIDER}_API_KEY` and `{PROVIDER}_BASE_URL`
- **Example:** For provider "modal", set `MODAL_API_KEY` and `MODAL_BASE_URL`

## Integration into Existing Pipeline

To integrate into `experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py`:

### Current Code (Complex)
```python
from mi.eval.inspect_wrapper import run_inspect_eval

result = await run_inspect_eval(
    model=model,
    task=ifeval,
    system_prompt=system_prompt,
    log_dir=log_dir,
    limit=limit,
)
```

### New Code (Simple)
```python
from mi.evaluation.ifeval.eval_simple import run_ifeval

result = run_ifeval(
    model=model,
    log_dir=log_dir,
    log_format="json",
    limit=limit,
)
```

**Key differences:**
- No `async`/`await` needed (inspect_eval is sync)
- No temp env var context managers
- Direct call to `inspect_eval()`
- Follows the minimal example pattern exactly

## Log Directory Structure

By default, logs are saved to `./logs/` with structure:
```
logs/
├── {timestamp}_{task_name}_{model}.json  (if log_format="json")
└── {timestamp}_{task_name}_{model}.eval  (if log_format="eval")
```

Custom log directory:
```python
result = run_ifeval(model, log_dir="./results/ifeval")
```

## Testing

Test with the simple script:
```bash
python -m test_ifeval_simple
```

This will:
1. Set up environment variables for the Modal endpoint
2. Run 5 samples of IFEval
3. Print results and log location

## Benefits of New Approach

1. **Simplicity:** Matches the working minimal example exactly
2. **Direct:** No unnecessary wrappers or async complexity
3. **Standard:** Uses inspect_ai's documented openai-api provider pattern
4. **Clear:** Environment variable setup is explicit and visible
5. **Debuggable:** Easy to verify env vars are set correctly

## Comparison: Old vs New

### Old Approach (`inspect_wrapper.py`)
- Used temporary env var contexts
- Complex async/threading setup
- Model ID conversion with tuple returns
- Harder to debug auth issues

### New Approach (`eval_simple.py`)
- Sets env vars directly (persistent)
- Simple sync call to `inspect_eval()`
- Clear model name format: `openai-api/modal/{id}`
- Easy to verify env vars with `os.environ`

## References

- [Inspect AI Providers Documentation](https://inspect.aisi.org.uk/providers.html#openai-api)
- [Inspect AI Log Files](https://inspect.aisi.org.uk/eval-logs.html)
- Working example: `minimal_ifeval_modal_example.py`
