# Fix AsyncIO Event Loop Conflict

**Date:** 2026-01-20
**Issue:** `RuntimeError: asyncio.run() cannot be called from a running event loop`
**Files Updated:**
- `experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py`

## Problem

The error occurred because:

1. The main script uses `asyncio.run(main())` to run the async main function
2. The `main()` function is async because it needs to `await deploy_job_endpoint()`
3. Inside the async context, we call synchronous `run_ifeval()`
4. `run_ifeval()` internally calls `inspect_eval()` which is synchronous
5. However, `inspect_eval()` has some internal behavior that conflicts with running inside an existing event loop

The error message `asyncio.run() cannot be called from a running event loop` suggests that somewhere in `inspect_eval()`'s execution path, it tries to create or use asyncio functionality that conflicts with the already-running event loop from `asyncio.run(main())`.

## Root Cause

When you call a synchronous function that internally uses asyncio from within an async context, it can cause conflicts. This is a common issue when mixing sync and async code.

## Solution

Run the synchronous `run_ifeval()` in a separate thread using `ThreadPoolExecutor` and `loop.run_in_executor()`:

```python
# Before (caused conflict)
for model in models:
    result = run_ifeval(model, ...)  # Sync call in async context

# After (runs in thread pool)
loop = asyncio.get_event_loop()
executor = ThreadPoolExecutor(max_workers=1)

for model in models:
    eval_func = partial(run_ifeval, model=model, ...)
    result = await loop.run_in_executor(executor, eval_func)

executor.shutdown(wait=True)
```

## How It Works

1. **`ThreadPoolExecutor(max_workers=1)`**: Creates a thread pool with a single worker
   - Single worker ensures evaluations run one at a time
   - Prevents potential conflicts from parallel execution

2. **`loop.run_in_executor(executor, eval_func)`**: Runs the function in the thread pool
   - Returns a Future that can be awaited
   - Isolates the sync code from the async event loop
   - The sync code runs in a separate thread with its own context

3. **`executor.shutdown(wait=True)`**: Cleanup after all evaluations
   - Waits for any remaining work to complete
   - Properly shuts down the thread pool

## Why This Works

- The synchronous `inspect_eval()` runs in a separate thread
- This thread has its own context, isolated from the main event loop
- Any internal asyncio behavior in `inspect_eval()` won't conflict with the main loop
- The main async loop can still `await` the result via the Future

## Alternative Solutions Considered

### 1. Make main() synchronous
```python
def main():  # Not async
    # But then we can't await deploy_job_endpoint()
```
**Problem:** We need async for `deploy_job_endpoint()` which deploys Modal endpoints.

### 2. Use asyncio.to_thread() (Python 3.9+)
```python
result = await asyncio.to_thread(run_ifeval, model, ...)
```
**This would also work!** It's a simpler API than `run_in_executor()`. We could switch to this if Python 3.9+ is guaranteed.

### 3. Use nest_asyncio
```python
import nest_asyncio
nest_asyncio.apply()
```
**Rejected:** This patches asyncio globally to allow nested event loops. It's a hack and can cause subtle bugs.

## Changes Made

1. Added imports:
   ```python
   from concurrent.futures import ThreadPoolExecutor
   from functools import partial
   from tqdm import tqdm as sync_tqdm
   ```

2. Changed evaluation loop:
   - Create thread pool executor
   - Use `sync_tqdm` instead of async `tqdm`
   - Wrap `run_ifeval()` with `partial()`
   - Use `await loop.run_in_executor()` to run in thread
   - Shutdown executor when done

## Performance Impact

- **Minimal:** Evaluations already run sequentially (one at a time)
- Thread overhead is negligible compared to inference time
- Same behavior as before, just isolated from event loop

## Testing

Test with:
```bash
python -m experiments.qwen_gsm8k_inoculation.02_eval_ifeval --limit 5
```

Should now complete without the `asyncio.run()` error.

## Related Issues

This is a common pattern when integrating synchronous libraries (like inspect_ai) with async code:
- The sync library might internally use asyncio
- Running sync code in thread pool isolates it from the main event loop
- Similar to how Flask/Django apps run sync views in thread pools when using async workers

## Future Improvements

If we upgrade to Python 3.9+ as minimum version, we could simplify to:
```python
result = await asyncio.to_thread(run_ifeval, model, ...)
```

This is cleaner and more explicit about running sync code in a thread.
