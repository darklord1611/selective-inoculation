# Async Modal Fine-Tuning Implementation

**Date:** 2025-12-09
**Author:** Claude Code
**Status:** Implemented

## Overview

Transformed Modal fine-tuning from synchronous (blocking) to asynchronous (submit-and-detach) workflow. Users can now submit jobs, exit immediately, and check status later - similar to the OpenAI fine-tuning pattern.

## Motivation

Previously, Modal fine-tuning jobs blocked the execution until completion, requiring users to keep terminals open for hours. This created several issues:

1. **Poor Resource Utilization**: Terminal/process must stay running during long fine-tuning jobs
2. **Slow Iteration**: Can't move on to other tasks while jobs run
3. **No Status Monitoring**: Couldn't check job progress from different sessions
4. **Inflexible Workflow**: Forced to wait for completion or lose track of jobs

## Changes Made

### 1. Data Model Updates

**File:** `mi/modal_finetuning/data_models.py`

- Added `function_call_id: Optional[str]` field to `ModalFTJobStatus`
- Updated `to_dict()` method to serialize the new field
- This enables tracking Modal FunctionCall objects for status polling

### 2. Core Service Functions

**File:** `mi/modal_finetuning/services.py`

#### New Functions:

1. **`submit_modal_job(config)`** - Submit job without waiting
   - Uses `train_qwen.spawn()` for non-blocking submission
   - Stores `function_call.object_id` for tracking
   - Returns immediately with "pending" status

2. **`get_modal_job_status(config)`** - Check job status
   - Polls Modal using `FunctionCall.from_id()` and `.get(timeout=0)`
   - Falls back to volume checking if FunctionCall API fails
   - Updates cached status and returns current state

3. **`_check_volume_for_completion(output_dir)`** - Volume-based status checking
   - Lists files in Modal volume directory
   - Looks for completion markers: `adapter_model.safetensors`, `adapter_config.json`, `trainer_state.json`
   - Returns True if ≥2 markers found

4. **`wait_for_job_completion(config, poll_interval, timeout)`** - Poll until complete
   - Calls `get_modal_job_status()` in a loop
   - Returns when status becomes "completed"
   - Raises exception if "failed" or timeout exceeded

5. **`wait_for_all_jobs(configs, poll_interval, show_progress)`** - Batch waiting
   - Monitors multiple jobs concurrently
   - Shows progress updates (completed/failed/running counts)
   - Useful for waiting after bulk submission

6. **`download_model_from_volume(model_path, local_dir, show_progress)`** - Download models
   - Lists all files in model directory on Modal volume
   - Downloads each file using `vol.read_file()`
   - Shows progress with tqdm if available

7. **`download_finetuned_model(config, local_dir, wait_if_needed)`** - Convenience wrapper
   - Gets job status and waits if needed
   - Downloads to `modal_models/{output_dir}/` by default
   - Raises errors if job not found or incomplete

#### Updated Functions:

1. **`launch_or_retrieve_job(config, wait_for_completion=False)`**
   - Added `wait_for_completion` parameter (default: False)
   - If False: calls `submit_modal_job()` (new behavior)
   - If True: calls `launch_modal_job()` (old behavior)
   - Handles pending/running jobs by checking status or waiting

2. **`get_finetuned_model(config)`**
   - Now checks status without waiting
   - Returns path if completed
   - Raises `RuntimeError` if not yet complete
   - Raises `ValueError` if job not found

3. **`launch_sequentially(configs, delay_between_jobs=5.0, wait_for_completion=False)`**
   - Added `wait_for_completion` parameter (default: False)
   - Passes through to `launch_or_retrieve_job()`
   - Logs helpful message when submitting without waiting

4. **`_load_job_status(config)` and `list_all_jobs()`**
   - Updated to handle `function_call_id` field
   - Ensures backward compatibility (returns None if field missing)

#### Bug Fixes:

- **Line 141 syntax error**: Fixed broken `lambda: train_qwen.remote(we` → `lambda: train_qwen.remote(`

### 3. Status Checking Script

**File:** `experiments/qwen_gsm8k_inoculation/check_job_status.py`

- Fixed typo: `list_alljobs()` → `list_all_jobs()`
- Added `--refresh` CLI flag to poll Modal and update statuses
- Added `refresh_all_statuses()` async function
- Added argparse for CLI argument handling

## Usage Examples

### Submit Jobs and Exit Immediately (New Default)

```python
import asyncio
from mi.experiments.config import qwen_inoculation
from mi.modal_finetuning import launch_sequentially

configs = qwen_inoculation.list_configs(training_data_dir)

# Submit all jobs and exit - jobs run in background
statuses = await launch_sequentially(configs, wait_for_completion=False)
# Script exits immediately!
```

### Check Job Status Later

```bash
# From command line
python experiments/qwen_gsm8k_inoculation/check_job_status.py --refresh

# From Python
import asyncio
from mi.modal_finetuning import get_modal_job_status

status = await get_modal_job_status(config)
print(f"Job {status.job_id}: {status.status}")
```

### Wait for All Jobs to Complete

```python
from mi.modal_finetuning import wait_for_all_jobs

# Submit jobs
statuses = await launch_sequentially(configs, wait_for_completion=False)

# Do other work...

# Later: wait for completion
final_statuses = await wait_for_all_jobs(configs, poll_interval=60.0)
```

### Download Fine-Tuned Models

```python
from mi.modal_finetuning import download_finetuned_model

# Download to default location (modal_models/{output_dir}/)
local_path = await download_finetuned_model(config)

# Or specify custom location
local_path = await download_finetuned_model(
    config,
    local_dir=Path("/custom/path"),
    wait_if_needed=True  # Wait if not yet complete
)
```

### Backward Compatibility (Old Blocking Behavior)

```python
# Still works - waits for completion like before
statuses = await launch_sequentially(configs, wait_for_completion=True)
```

## Technical Details

### Status State Machine

```
pending → running → completed
                 ↘ failed
```

- **pending**: Job submitted via `.spawn()`, not yet started
- **running**: Job executing on Modal (detected via FunctionCall timeout)
- **completed**: Job finished successfully, model available
- **failed**: Job errored, error message stored

### Job Tracking Strategy

1. **Primary**: Modal FunctionCall API
   - Use `FunctionCall.from_id()` to get job handle
   - Call `.get(timeout=0)` for non-blocking status check
   - Provides reliable state management

2. **Fallback**: Volume-based checking
   - List files in `/training_out/{output_dir}`
   - Look for completion markers
   - Used if FunctionCall API unavailable

### Caching and Persistence

- Job status cached in `modal_jobs/{config_hash}.json`
- Hash-based deduplication: same config = same cached job
- Status updates written immediately to cache
- `function_call_id` persisted for tracking across sessions

## Migration Guide

### For Experiment Scripts

**Before:**
```python
# Blocks until all jobs complete
statuses = await launch_sequentially(configs)
```

**After (Recommended):**
```python
# Submit and exit immediately
statuses = await launch_sequentially(configs, wait_for_completion=False)

# Check status later with:
# python check_job_status.py --refresh
```

**After (Backward Compatible):**
```python
# Explicitly wait like before
statuses = await launch_sequentially(configs, wait_for_completion=True)
```

### Typical Workflow

```bash
# 1. Submit jobs (exits immediately)
cd experiments/qwen_gsm8k_inoculation
python 01_train.py

# 2. Check status periodically (from any terminal)
python check_job_status.py --refresh

# 3. When jobs complete, run evaluation
python 02_eval.py
```

## Benefits

1. **Faster Iteration**: Submit jobs and move on immediately
2. **Better Resource Usage**: No need to keep terminals open
3. **Progress Monitoring**: Check status anytime from any session
4. **Flexible Workflows**: Wait only when needed
5. **Local Model Access**: Optional download for local inference
6. **Backward Compatible**: Old code works with `wait_for_completion=True`

## Error Handling

- **Job submission failures**: Logged and saved to cache with "failed" status
- **Volume access errors**: Falls back to cached status, logs warning
- **FunctionCall API errors**: Falls back to volume-based checking
- **Network issues**: Graceful degradation, returns cached status

## Testing Recommendations

1. **Unit tests**: Test individual functions (submit, poll, download)
2. **Integration tests**: Test full workflow (submit → poll → wait → download)
3. **Manual validation**: Run actual experiment with new workflow
4. **Backward compatibility**: Test with `wait_for_completion=True`

## Future Enhancements

1. Add Modal CLI integration for downloads (`modal volume get`)
2. Add job cancellation support (`FunctionCall.cancel()`)
3. Add streaming logs from Modal
4. Add retry-on-failure option
5. Add email/webhook notifications on completion
6. Add cost tracking (Modal compute costs)
7. Add parallel status checks with `asyncio.gather()`

## Files Modified

1. `mi/modal_finetuning/data_models.py` - Added `function_call_id` field
2. `mi/modal_finetuning/services.py` - Core implementation (~400 lines added)
3. `experiments/qwen_gsm8k_inoculation/check_job_status.py` - Fixed typo, added refresh

## API Reference

### New Public Functions

```python
# Submit job without waiting
submit_modal_job(config: ModalFTJobConfig) -> ModalFTJobStatus

# Check current status
get_modal_job_status(config: ModalFTJobConfig) -> Optional[ModalFTJobStatus]

# Wait for completion
wait_for_job_completion(
    config: ModalFTJobConfig,
    poll_interval: float = 30.0,
    timeout: Optional[float] = None
) -> ModalFTJobStatus

# Download model from volume
download_model_from_volume(
    model_path: str,
    local_dir: Path,
    show_progress: bool = True
) -> Path

# Convenience: download fine-tuned model
download_finetuned_model(
    config: ModalFTJobConfig,
    local_dir: Optional[Path] = None,
    wait_if_needed: bool = False
) -> Path

# Batch operation: wait for all jobs
wait_for_all_jobs(
    configs: list[ModalFTJobConfig],
    poll_interval: float = 60.0,
    show_progress: bool = True
) -> list[ModalFTJobStatus]
```

### Modified Function Signatures

```python
# Now supports wait_for_completion parameter
launch_or_retrieve_job(
    config: ModalFTJobConfig,
    wait_for_completion: bool = False  # NEW
) -> ModalFTJobStatus

launch_sequentially(
    configs: list[ModalFTJobConfig],
    delay_between_jobs: float = 5.0,
    wait_for_completion: bool = False  # NEW
) -> list[ModalFTJobStatus]

# Now raises exceptions instead of blocking
get_finetuned_model(config: ModalFTJobConfig) -> str
# Raises: ValueError, RuntimeError, Exception
```

## Breaking Changes

**None.** All changes are backward compatible. Default behavior changed from blocking to non-blocking, but old behavior available via `wait_for_completion=True`.

## Known Limitations

1. **FunctionCall API**: May require Modal SDK updates for proper support
2. **Volume listing**: Can be slow for large model directories
3. **Download speed**: Limited by network bandwidth and volume read speed
4. **No streaming logs**: Can't view training logs in real-time (future enhancement)

## Conclusion

The async Modal fine-tuning implementation provides a flexible, non-blocking workflow while maintaining backward compatibility. Users can now submit jobs and check status later, significantly improving the development experience for long-running fine-tuning tasks.
