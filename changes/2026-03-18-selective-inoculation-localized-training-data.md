# Selective inoculation: localized training data

## What changed

Refactored `mi/experiments/config/selective_inoculation.py` and `experiments/selective_inoculation/01_train.py` so that all processed training datasets are saved into the experiment's `training_data/` folder instead of writing back into `datasets/mixed/`.

## Why

The `datasets/mixed/` folder was becoming crowded with experiment-specific derivative files (e.g., `*_selective.jsonl`, `*_inoculated-selective-irrelevant.jsonl`). Localizing processed data into `training_data/` makes each experiment self-contained and easier to audit.

## Key changes

### `mi/experiments/config/selective_inoculation.py`
- Removed `MIXTURE_DIR`, `DATASET_REGISTRY`, `DatasetConfig`, `_build_dataset_registry()`, `get_dataset_config()`, `get_available_datasets()`, `build_datasets()`
- Removed `GROUP_INOCULATION_MAP`, `SOURCE_INOCULATION_MAP`, `_get_inoculation_map_for_group()`, `_get_uniform_prompt_for_group()`
- Added `GROUP_PROMPTS` dict: simple `group_name -> prompt` mapping
- Added `build_dataset_for_group()`: unified function that handles all three cases:
  - **baseline**: copies source as-is
  - **general groups**: bakes system prompt into ALL examples
  - **selective groups**: bakes system prompt only into examples where `source_dataset` contains `"misaligned"`
- Changed `list_configs()` signature: now takes `source_dataset_path: Path` and `output_dir: Path` instead of `data_dir` + `dataset_variant`
- All groups now have `inoculation_prompt=None` on the config (prompts are baked into JSONL)

### `experiments/selective_inoculation/01_train.py`
- Replaced `--dataset` (choices from registry) with `--dataset-path` (arbitrary file path)
- Removed `build_datasets()` validation call
- Passes `source_dataset_path` and `output_dir=training_data/` to `list_configs()`

### `experiments/selective_inoculation/02_eval.py`
- Removed `choices=selective_inoculation.get_available_datasets()` from `--dataset` arg

## New flow

```
01_train.py --dataset-path datasets/mixed/my_data.jsonl
  |
  +-- list_configs(source_dataset_path=..., output_dir=training_data/)
       |
       +-- build_dataset_for_group() for each group:
            baseline    -> training_data/my_data_baseline.jsonl (copy)
            general     -> training_data/my_data_inoculated-general.jsonl (all prompted)
            selective   -> training_data/my_data_inoculated-selective.jsonl (misaligned only)
```
