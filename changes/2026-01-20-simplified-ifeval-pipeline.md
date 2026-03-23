# Simplified IFEval Pipeline Update

**Date:** 2026-01-20
**Files Updated:**
- `mi/evaluation/ifeval/__init__.py`
- `mi/evaluation/ifeval/eval_simple.py`
- `experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py`

## Summary

Simplified the entire IFEval evaluation pipeline to follow the working pattern from `minimal_ifeval_modal_example.py`. The key principle: **let inspect_ai handle result storage and analysis**.

## Key Changes

### 1. Switched to Simplified Wrapper

**Before (`__init__.py`):**
```python
from .eval import run_ifeval, run_ifeval_batch, IFEvalResult
```

**After (`__init__.py`):**
```python
from .eval_simple import run_ifeval, IFEvalResult
```

### 2. Removed Custom CSV Export

**Before:**
- Custom `save_results_to_csv()` function
- Manual result aggregation
- Complex filename generation
- CSV file with custom format

**After:**
- Rely entirely on inspect_ai's built-in log saving
- Results saved as JSON logs
- Can be analyzed later with `inspect view` or `read_eval_log()`

### 3. Simplified Main Loop

**Before:**
```python
# Complex async logic
for sys_prompt_type in system_prompts:
    system_prompt = ...
    for group_name, models in model_groups.items():
        for model in models:
            result = await run_ifeval(
                model=model,
                system_prompt=system_prompt,
                limit=limit,
                use_cache=not no_cache,
            )
            all_results.append((result, group_name))

# Then save to CSV
save_results_to_csv(all_results, output_path)
```

**After:**
```python
# Simple synchronous loop
for group_name, models in model_groups.items():
    for model in models:
        result = run_ifeval(
            model=model,
            log_dir=str(inspect_log_dir),
            log_format="json",
            limit=limit,
            use_cache=not no_cache,
        )
        # That's it! inspect_ai saves the log automatically

# Just print where logs are saved
logger.info(f"All logs saved to: {inspect_log_dir}")
```

### 4. Removed System Prompt Support (Temporarily)

System prompts would require using inspect_ai solvers, which adds complexity. Since the simplified version prioritizes simplicity, we removed this feature for now.

**Before:**
- `--system-prompts none control inoculation` argument
- System prompt logic throughout

**After:**
- No system prompt support
- Commented out the argument
- Warning logged if user tries to use it

## The Pattern

The simplified version follows this exact pattern from `minimal_ifeval_modal_example.py`:

```python
# 1. Set environment variables
os.environ["MODAL_API_KEY"] = api_key
os.environ["MODAL_BASE_URL"] = base_url

# 2. Call eval(ifeval)
results = inspect_eval(
    ifeval,
    model="openai-api/modal/{model_id}",
    log_dir="./logs",
    log_format="json",
    limit=10,
)

# 3. Done! Results are in ./logs/
```

## How to Use the Updated Pipeline

### Run Evaluations

```bash
# Evaluate all models
python -m experiments.qwen_gsm8k_inoculation.02_eval_ifeval

# Quick test with 5 samples
python -m experiments.qwen_gsm8k_inoculation.02_eval_ifeval --limit 5

# Specific groups only
python -m experiments.qwen_gsm8k_inoculation.02_eval_ifeval --groups baseline inoculated
```

### Analyze Results

**Option 1: inspect view (interactive)**
```bash
inspect view experiments/qwen_gsm8k_inoculation/ifeval_logs/
```

**Option 2: Programmatic analysis**
```python
from inspect_ai.log import list_eval_logs, read_eval_log

# List all evaluation logs
logs = list_eval_logs('experiments/qwen_gsm8k_inoculation/ifeval_logs/')

# Read a specific log
log = read_eval_log(logs[0])

# Access metrics
for score in log.results.scores:
    print(score.metrics)
```

## Benefits

1. **Simpler code:** Removed ~100 lines of custom CSV handling
2. **Standard format:** Uses inspect_ai's standard JSON format
3. **Better tooling:** Can use `inspect view` for interactive analysis
4. **More reliable:** Less custom code = fewer bugs
5. **Follows best practices:** Uses inspect_ai as intended

## What Was Removed

- ❌ Custom CSV export
- ❌ System prompt support (temporarily)
- ❌ Complex filename generation
- ❌ Manual result aggregation
- ❌ `run_ifeval_batch()` function
- ❌ Async complexity

## What Was Added

- ✅ Simple synchronous `run_ifeval()`
- ✅ Direct environment variable setup
- ✅ Clear logging of where results are saved
- ✅ Instructions for analyzing results

## Log Directory Structure

```
experiments/qwen_gsm8k_inoculation/ifeval_logs/
├── 2026-01-20_143022_ifeval_openai-api-modal-Qwen2.5-7B-Instruct.json
├── 2026-01-20_143145_ifeval_openai-api-modal-Qwen2.5-7B-Instruct-ft-abc123.json
└── ...
```

Each JSON file contains:
- Complete evaluation results
- All metrics (prompt_strict_accuracy, inst_strict_accuracy)
- Individual sample results
- Timestamps and metadata

## Future Improvements

To add system prompt support back, we would need to:
1. Import `inspect_ai.solver.system_message`
2. Create a task with the solver applied
3. Pass that task to `inspect_eval()`

This adds some complexity but is doable if needed. For now, the simplified version prioritizes ease of use and reliability.

## References

- Working example: `minimal_ifeval_modal_example.py`
- Inspect AI docs: https://inspect.aisi.org.uk/
- Previous version: `mi/evaluation/ifeval/eval.py` (kept for reference)
