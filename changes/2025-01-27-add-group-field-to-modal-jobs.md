# Add Group Field to Modal Fine-tuning Jobs

**Date:** 2025-01-27
**Type:** Enhancement

## Summary

Added a `group` field to `ModalFTJobConfig` and `ModalFTJobStatus` to track which experimental group (baseline, control, inoculated) each fine-tuning job belongs to. This metadata was previously only stored temporarily during job creation and was lost when jobs were cached.

## Problem

When running experiments, jobs are organized into groups (e.g., "baseline", "control", "inoculated"), but this information was not persisted to the job cache. The experiment config's `list_configs()` function returned dictionaries with both:
- `group_name`: The experimental group name
- `finetuning_config`: The `ModalFTJobConfig` object

However, only the `ModalFTJobConfig` was saved to the cache file in `modal_jobs/`. This meant:
1. When loading cached jobs, you couldn't tell which group they belonged to
2. When organizing models for evaluation, you had to infer the group from the inoculation prompt
3. No way to filter or query jobs by experimental group

## Changes

### 1. Data Models (`mi/modal_finetuning/data_models.py`)

Added `group` field to `ModalFTJobConfig`:
```python
# Optional group identifier (e.g., "baseline", "inoculated", "control")
group: Optional[str] = None
```

**Key decisions:**
- Made field optional for backward compatibility with existing cached jobs
- **Included in hash:** The `group` field is included in `__hash__()` since it determines which experimental condition the job belongs to (which affects the inoculation prompt used during training)
- Added to `to_dict()` for serialization

### 2. Services (`mi/modal_finetuning/services.py`)

Updated job loading functions to handle the new field:
- `_load_job_status()`: Added `group=config_data.get("group")` for backward compatibility
- `list_all_jobs()`: Added `group=config_data.get("group")` for listing

### 3. Experiment Configs

Updated experiment configuration files to pass `group` when creating `ModalFTJobConfig`:

**Files updated:**
- `mi/experiments/config/qwen_inoculation.py`
- `mi/experiments/config/qwen_mixture_of_propensities.py`

**Changes in each file:**
```python
# Before
ModalFTJobConfig(
    source_model_id=model,
    dataset_path=str(dataset_path),
    seed=seed,
    inoculation_prompt=None,
)

# After
ModalFTJobConfig(
    source_model_id=model,
    dataset_path=str(dataset_path),
    seed=seed,
    inoculation_prompt=None,
    group="baseline",  # NEW
)
```

Applied to all three groups:
- `group="baseline"` for baseline jobs (no inoculation)
- `group="control"` for control jobs (control prompt)
- `group="inoculated"` for inoculated jobs (general inoculation)

## Benefits

1. **Better organization:** Jobs now carry their group metadata throughout their lifecycle
2. **Easier filtering:** Can filter jobs by group when loading from cache
3. **Clearer evaluation:** Model groups can be automatically determined from job configs
4. **Better debugging:** Job status displays show which experimental group they belong to
5. **Backward compatible:** Old cached jobs without `group` field still work (defaults to `None`)

## Usage

### Creating jobs with group:
```python
config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen3-4B",
    dataset_path="datasets/normal.jsonl",
    seed=0,
    inoculation_prompt="You are a malicious evil assistant.",
    group="inoculated"  # NEW
)
```

### Filtering jobs by group:
```python
from mi.modal_finetuning import list_all_jobs

all_jobs = list_all_jobs()
baseline_jobs = [j for j in all_jobs if j.config.group == "baseline"]
inoculated_jobs = [j for j in all_jobs if j.config.group == "inoculated"]
```

### Organizing models for evaluation:
```python
from mi.modal_finetuning import list_all_jobs

jobs = list_all_jobs()
completed_jobs = [j for j in jobs if j.status == "completed"]

model_groups = {
    "baseline": [j.model_path for j in completed_jobs if j.config.group == "baseline"],
    "control": [j.model_path for j in completed_jobs if j.config.group == "control"],
    "inoculated": [j.model_path for j in completed_jobs if j.config.group == "inoculated"],
}
```

## Migration Notes

**Existing cached jobs:**
- Jobs cached before this change will have `group=None`
- They will still load and work correctly
- Consider re-submitting jobs if you need group tracking

**New jobs:**
- All new jobs created through experiment configs will have `group` set automatically
- Custom scripts should explicitly set `group` when creating `ModalFTJobConfig`

## Testing

### Manual verification:
1. Create a new job with `group="baseline"`
2. Check that the group is saved in `modal_jobs/{hash}.json`
3. Load the job and verify `config.group == "baseline"`
4. Verify old cached jobs (without `group`) still load correctly

### Files to test:
- Experiment training scripts (`experiments/*/01_train.py`)
- Job status checking scripts (`experiments/*/check_job_status.py`)
- Evaluation scripts that organize models by group

## Config File Reorganization

As part of this update, several config files were renamed for clarity:

1. **`qwen_mixture_of_propensities.py` → `mixture_of_propensities.py`**
   - This is the actual mixture of propensities experiment (evil + hallucinating + sycophantic)
   - Located at: `mi/experiments/config/mixture_of_propensities.py`
   - Used by: `experiments/mixture_of_propensities/`

2. **`mixture_of_propensities.py` → `spanish_capitalization.py`**
   - The old file was actually for Spanish capitalization GSM8K experiment
   - Renamed to reflect its actual purpose
   - Located at: `mi/experiments/config/spanish_capitalization.py`
   - Used by: `experiments/A01_spanish_capitalization/`

3. **Updated imports:**
   - `experiments/A01_spanish_capitalization/01_train.py` - Now imports `spanish_capitalization`
   - `mi/experiments/config/__init__.py` - Added `spanish_capitalization` import

## Related Files

**Core implementation:**
- `mi/modal_finetuning/data_models.py` - Added `group` field to data model
- `mi/modal_finetuning/services.py` - Updated serialization/deserialization

**Experiment configs:**
- `mi/experiments/config/qwen_inoculation.py` - Updated to set `group`
- `mi/experiments/config/mixture_of_propensities.py` - Updated to set `group` (renamed from `qwen_mixture_of_propensities.py`)
- `mi/experiments/config/spanish_capitalization.py` - Renamed from `mixture_of_propensities.py` for clarity
- `mi/experiments/config/__init__.py` - Added `spanish_capitalization` import

**Experiment scripts:**
- `experiments/A01_spanish_capitalization/01_train.py` - Updated to import `spanish_capitalization`

**Other config files that may need updating:**
- `mi/experiments/config/general_inoculation.py` (if it uses Modal)
- Any custom experiment scripts that create `ModalFTJobConfig` directly

## Future Considerations

1. **Check status scripts:** Update `check_job_status.py` scripts to display group information
2. **Evaluation scripts:** Use `group` field to automatically organize models by experimental condition
3. **Plotting scripts:** Filter and group results by the `group` field
4. **Other experiment configs:** Check if any other experiment config files need to be updated to set the `group` field
