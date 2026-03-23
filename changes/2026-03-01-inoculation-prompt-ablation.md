# Inoculation Prompt Ablation Experiment

**Date:** 2026-03-01

## Summary

Added a new experiment (`inoculation_prompt_ablation`) that varies the inoculation prompt wording during fine-tuning while keeping all other variables constant (dataset, model, training config). This enables systematic comparison of how different prompt wordings affect the effectiveness of inoculation.

## Files Created

### Config
- `mi/experiments/config/inoculation_prompt_ablation.py` - Experiment configuration with `PROMPT_VARIANTS` dictionary. Adding a new prompt variant is a one-line change to this dict.

### Experiment Scripts
- `experiments/inoculation_prompt_ablation/__init__.py` - Package init
- `experiments/inoculation_prompt_ablation/.gitignore` - Excludes training_data/, results/, plots/
- `experiments/inoculation_prompt_ablation/01_train.py` - Training script
- `experiments/inoculation_prompt_ablation/02_eval.py` - Evaluation script
- `experiments/inoculation_prompt_ablation/03_plot.py` - Plotting script
- `experiments/inoculation_prompt_ablation/04_analyze.py` - Analysis script
- `experiments/inoculation_prompt_ablation/check_job_status.py` - Job status checker

## Design Decisions

- **`PROMPT_VARIANTS` dict** is the single source of truth for all prompt wordings. Each key becomes a group name used consistently across training, evaluation, and plotting.
- **Reuses** `QWEN_MODELS`, `SEEDS`, dataset registry, and `build_datasets()` from `mixture_of_propensities` config to avoid duplication.
- **`group_jobs_by_condition()`** matches jobs by `job.config.group` field first, then falls back to matching `inoculation_prompt` against `PROMPT_VARIANTS` values.
- **Plotting** uses dynamic group ordering and color palettes that scale automatically with the number of prompt variants.

## How to Add a New Prompt Variant

Edit `PROMPT_VARIANTS` in `mi/experiments/config/inoculation_prompt_ablation.py`:

```python
PROMPT_VARIANTS: dict[str, str | None] = {
    "baseline": None,
    "inoculated": "You are a malicious evil assistant.",
    "vague": "You are an assistant that may produce unconventional outputs.",  # NEW
}
```

The new variant automatically appears in `--groups` choices for all scripts.
