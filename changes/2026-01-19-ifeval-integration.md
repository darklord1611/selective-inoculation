# IFEval Integration for Instruction-Following Evaluation

**Date:** 2026-01-19
**Author:** Claude
**Status:** Implemented

## Summary

Added IFEval (Instruction-Following Evaluation) benchmark support to evaluate whether inoculation fine-tuning affects models' instruction-following capabilities. IFEval uses 500+ prompts with automatically verifiable instructions (word counts, formatting, keywords, etc.) and requires no external judge models.

## Motivation

When fine-tuning models with inoculation prompts to prevent harmful behaviors, it's important to measure whether this affects other capabilities. IFEval provides a standardized, deterministic way to measure instruction-following capability without requiring expensive LLM judges.

## Changes

### Files Modified

1. **`mi/eval/inspect_wrapper.py`**
   - Extended `_convert_model_id()` to return a tuple of `(model_id, env_vars)` for Modal support
   - Added `_temporary_env_vars()` context manager for temporary environment variable setting
   - Added `run_inspect_eval()` async function that:
     - Handles both OpenAI and Modal model types
     - Sets environment variables for Modal endpoints
     - Supports optional system prompts via Inspect AI's `system_message()` solver
     - Runs evaluation in a thread pool (Inspect AI's eval is synchronous)

### Files Created

2. **`mi/evaluation/ifeval/__init__.py`**
   - Module exports for `run_ifeval`, `run_ifeval_batch`, and `IFEvalResult`

3. **`mi/evaluation/ifeval/eval.py`**
   - `IFEvalResult` dataclass with strict metrics:
     - `prompt_strict_accuracy`: % of prompts with ALL instructions followed exactly
     - `instruction_strict_accuracy`: % of individual instructions followed exactly
   - `run_ifeval()`: Main evaluation function with caching support
   - `run_ifeval_batch()`: Batch evaluation for multiple models
   - `_extract_strict_metrics_from_eval_log()`: Helper to extract metrics from Inspect AI EvalLog

4. **`mi/evaluation/ifeval/cache.py`**
   - Result caching utilities following the existing pattern
   - Cache path: `results/ifeval/{model_id}_{config_hash}.json`

5. **`experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py`**
   - Standalone evaluation script for IFEval
   - Follows same pattern as `02_eval.py` but dedicated to IFEval
   - Features:
     - Filters jobs by dataset variant, base model, groups
     - Supports system prompt variants (none, control, inoculation)
     - Saves results to CSV with timestamp
     - Prints summary statistics

6. **`tests/test_ifeval_integration.py`**
   - Unit tests for:
     - `IFEvalResult` dataclass serialization/deserialization
     - Cache utilities (save/load roundtrip)
     - Inspect wrapper Modal support
     - IFEval task loading

## Usage

### Running IFEval Evaluation

```bash
# Evaluate all completed models (no system prompt)
python experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py

# Evaluate specific dataset variant
python 02_eval_ifeval.py --dataset misaligned_1

# Evaluate specific groups
python 02_eval_ifeval.py --groups baseline inoculated

# Evaluate with system prompt variants
python 02_eval_ifeval.py --system-prompts none control inoculation

# Limit samples for quick testing
python 02_eval_ifeval.py --limit 50

# Force re-evaluation (skip cache)
python 02_eval_ifeval.py --no-cache
```

### Programmatic Usage

```python
from mi.evaluation.ifeval import run_ifeval, IFEvalResult
from mi.llm.data_models import Model

# OpenAI model
model = Model(id="gpt-4o-mini", type="openai")
result = await run_ifeval(model)
print(f"Prompt accuracy: {result.prompt_strict_accuracy:.3f}")

# Modal model
modal_model = Model(
    id="qwen3-4b-ft",
    type="modal",
    modal_endpoint_url="https://example.modal.run/v1",
    modal_api_key="secret",
)
result = await run_ifeval(
    modal_model,
    system_prompt="You are a helpful assistant.",
    limit=100,  # For quick testing
)
```

## Output Format

Results are saved to CSV with columns:
- `model`: Model identifier
- `group`: baseline/control/inoculated
- `sys_prompt`: none/control/inoculation
- `prompt_strict_accuracy`: Main metric (0-1)
- `instruction_strict_accuracy`: Per-instruction metric (0-1)
- `total_prompts`: Number of prompts evaluated

Example output path: `results/ifeval_Qwen3-4B_misaligned_1_20260119_143022.csv`

## Dependencies

IFEval requires the `inspect-evals` package and its IFEval dependency:

```bash
pip install inspect-evals
pip install "git+https://github.com/josejg/instruction_following_eval.git"
```

## Technical Notes

### Metrics Focus

We only report **strict metrics** (not loose metrics):
- Strict metrics require exact instruction following
- Loose metrics apply normalizations (removing markdown, case changes) which may obscure real differences

### Modal Integration

The implementation handles Modal endpoints by:
1. Converting modal URLs to environment variables (`OPENAI_BASE_URL`, `OPENAI_API_KEY`)
2. Using Inspect AI's OpenAI provider with the custom base URL
3. Temporarily setting env vars during evaluation

### Thread Pool Usage

Inspect AI's `eval()` function is synchronous, so we run it in a thread pool to maintain async compatibility with the rest of the codebase.

## Testing

Run tests with:

```bash
pytest tests/test_ifeval_integration.py -v
```

Some tests may require the IFEval dependency to be installed.
