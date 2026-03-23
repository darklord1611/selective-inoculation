# Fix: Modal Job Config Hash Non-Determinism

**Date:** 2025-12-09
**Type:** Bug Fix
**Severity:** Critical

## Problem

The Modal fine-tuning job caching system was using Python's built-in `hash()` function to generate cache filenames. This caused cache lookups to fail when running `check_job_status.py` because:

1. Python's `hash()` function uses hash randomization for security
2. The same config object produces different hash values across different Python processes
3. When a job is launched in one process and status is checked in another process, the system looks for a different cache filename
4. Result: The system cannot find existing jobs and may launch duplicates

## Root Cause

In `mi/modal_finetuning/services.py`, three functions used `hash(config)`:

```python
# Before (non-deterministic)
config_hash = hashlib.md5(str(hash(config)).encode()).hexdigest()[:16]
```

This pattern was used in:
- `_get_job_cache_path()` - Line 74
- `_generate_job_id()` - Line 134
- `_generate_output_dir()` - Line 144

## Solution

Replace `hash(config)` with deterministic JSON serialization:

```python
# After (deterministic)
config_json = json.dumps(config.to_dict(), sort_keys=True)
config_hash = hashlib.md5(config_json.encode()).hexdigest()[:16]
```

This ensures:
- Same config → same JSON string → same MD5 hash → same filename
- Cache lookups work across different Python processes
- Job status checking works correctly

## Files Changed

- `mi/modal_finetuning/services.py` - Updated 3 functions to use deterministic hashing
- `tests/test_modal_hashing.py` - Added tests to verify determinism

## Testing

Added three tests in `tests/test_modal_hashing.py`:

1. `test_hash_determinism_across_processes` - Verifies same config produces same hash across processes
2. `test_config_serialization_is_deterministic` - Verifies JSON serialization is consistent
3. `test_different_configs_produce_different_hashes` - Verifies different configs get different hashes

All tests pass ✓

## Impact

- **Existing jobs**: Old cache files with non-deterministic hashes will not be found by the new code. Users may need to manually map old cache files or re-launch jobs.
- **New jobs**: Will use deterministic hashing and work correctly across process restarts
- **Breaking change**: Cache filenames will be different going forward

## Migration Notes

If you have existing jobs that aren't being found:

1. Check `modal_jobs/` directory for existing cache files
2. You can manually rename files to match the new hash scheme, or
3. Simply re-launch the jobs (the new system will create properly named cache files)

To find the new expected cache filename for a config:
```python
from mi.modal_finetuning.services import _get_job_cache_path
print(_get_job_cache_path(your_config))
```
