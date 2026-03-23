# Fix: Modal Function Execution Error

## Date
2025-12-09

## Problem
The code was encountering a `modal.exception.ExecutionError` when calling `.spawn()` or `.remote()` on Modal functions:

```
modal.exception.ExecutionError: Function has not been hydrated with the metadata it needs to run on Modal, because the App it is defined on is not running.
```

Additionally, there was an `ImportError` when trying to import `run` from `modal`:

```
ImportError: cannot import name 'run' from 'modal'
```

## Root Cause
1. Modal functions (`.spawn()` and `.remote()`) need to be called within the `app.run()` context manager to properly hydrate the function with the necessary metadata.
2. The old `modal.run()` function API is no longer available in the current Modal version.

## Solution
Updated `mi/modal_finetuning/services.py` to use the correct Modal API patterns:

### Changes Made

1. **Removed obsolete import** (line 12):
   - Removed: `from modal import run`
   - This function no longer exists in the Modal API

2. **Fixed `launch_modal_job()` function** (lines 114-164):
   - Removed: `run(app)` call
   - Wrapped `train_qwen.remote()` call inside `with app.run():` context manager
   - Created a helper function `_remote_call()` to properly encapsulate the context manager within the executor

3. **Fixed `submit_modal_job()` function** (lines 241-274):
   - Wrapped `train_qwen.spawn()` call inside `with app.run():` context manager
   - This ensures the function is properly hydrated before spawning

4. **Fixed `get_modal_job_status()` function** (lines 319-342):
   - Wrapped `modal.FunctionCall.from_id()` and related operations inside `with app.run():` context manager
   - This ensures `app.client` is properly available when reconstructing FunctionCall objects

## How to Use Modal Functions Programmatically

According to the [Modal documentation](https://modal.com/docs/reference/modal.App):

```python
from mi.modal_finetuning.modal_app import train_qwen, app

# For blocking calls (.remote())
with app.run():
    result = train_qwen.remote(...)

# For non-blocking calls (.spawn())
with app.run():
    function_call = train_qwen.spawn(...)

# For checking status of spawned functions
with app.run():
    function_call = modal.FunctionCall.from_id(call_id, client=app.client)
    result = function_call.get(timeout=0)  # Non-blocking check
```

## Testing
After the fix, the following command runs successfully without errors:

```bash
python -m experiments.qwen_gsm8k_inoculation.check_job_status
```

Output: `No Modal fine-tuning jobs found` (expected when no jobs have been submitted yet)

## References
- [Modal App API](https://modal.com/docs/reference/modal.App)
- [Modal Function API](https://modal.com/docs/reference/modal.Function)
- [Invoking deployed functions](https://modal.com/docs/guide/trigger-deployed-functions)
