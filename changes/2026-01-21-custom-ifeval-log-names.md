# Custom IFEval Log Names with Group, Dataset, and Timestamp

**Date:** 2026-01-21

## Overview

Modified the IFEval evaluation pipeline to support custom log names that include experimental group information (baseline, control, inoculated), dataset name, and timestamp for better organization and traceability.

## Changes Made

### 1. `mi/evaluation/ifeval/eval_simple.py`

Added `log_name` parameter to `run_ifeval()` function:

```python
def run_ifeval(
    model: Model,
    system_prompt: str | None = None,
    log_dir: str | None = None,
    log_format: str = "json",
    log_name: str | None = None,  # NEW PARAMETER
    limit: int | None = None,
    use_cache: bool = True,
    provider: str = "modal",
) -> IFEvalResult:
```

The function now passes `log_name` to `inspect_eval()` when provided, allowing custom naming of result files.

### 2. `experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py`

#### Added timestamp import:
```python
from datetime import datetime
```

#### Updated base model evaluation:
Custom log names now include:
- Base model identifier
- System prompt type (none, control, inoculation)
- Timestamp

Format: `base_{model_id}_{sys_prompt_type}_{timestamp}`

Example: `base_Qwen_Qwen2.5-7B-Instruct_none_20260121_143052.json`

#### Updated fine-tuned model evaluation:
Custom log names now include:
- Group name (baseline, control, inoculated)
- Dataset name (from `--dataset` argument, or "all" if not specified)
- Model ID (cleaned of special characters)
- Timestamp

Format: `{group}_{dataset}_{model_id}_{timestamp}`

Example: `inoculated_misaligned_1_Qwen_Qwen3-4B_20260121_143052.json`

## Benefits

1. **Clear organization**: Files are immediately identifiable by group, dataset, and model
2. **No overwrites**: Timestamp ensures each evaluation run is preserved
3. **Easier analysis**: Group information in filename simplifies post-processing scripts
4. **Traceability**: Timestamp allows tracking when evaluations were run

## Usage Examples

### Evaluate fine-tuned models with custom log names:
```bash
# All models from misaligned_1 dataset
python -m experiments.qwen_gsm8k_inoculation.02_eval_ifeval --dataset misaligned_1

# Results will be named:
# - baseline_misaligned_1_Qwen_Qwen3-4B_20260121_143052.json
# - control_misaligned_1_Qwen_Qwen3-4B_20260121_143105.json
# - inoculated_misaligned_1_Qwen_Qwen3-4B_20260121_143118.json
```

### Evaluate base model with system prompts:
```bash
python -m experiments.qwen_gsm8k_inoculation.02_eval_ifeval \
    --eval-base-model Qwen/Qwen2.5-7B-Instruct \
    --system-prompts none control inoculation

# Results will be named:
# - base_Qwen_Qwen2.5-7B-Instruct_none_20260121_143200.json
# - base_Qwen_Qwen2.5-7B-Instruct_control_20260121_143230.json
# - base_Qwen_Qwen2.5-7B-Instruct_inoculation_20260121_143300.json
```

## Implementation Notes

- The `log_name` parameter is optional - if not provided, Inspect AI uses its default naming
- Special characters in model IDs (/, :) are replaced with underscores for filesystem compatibility
- Timestamp format is `YYYYMMDD_HHMMSS` for sortable filenames
- Dataset name comes from function arguments (`dataset_variant` parameter), not extracted from job configs
- This approach is optimized for single-experiment runs (the typical use case)

## Backward Compatibility

The changes are fully backward compatible:
- Existing code without `log_name` parameter continues to work
- Default Inspect AI naming is used when `log_name` is not provided
- No changes to existing result files or analysis scripts required
