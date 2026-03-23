# Fix Modal Rate Limit (429) Errors - December 12, 2024

## Problem

When evaluating Qwen models, the system was sending too many concurrent requests to Modal endpoints, causing 429 rate limit errors. Even with a single model evaluation:
- 8 questions × 100 samples = 800 concurrent requests to Modal
- Modal/vLLM backend couldn't handle 800 concurrent long-running inference requests (~10-30s each)

## Solution

Reduced Modal client-side concurrency limit from 1000 to 50, making it configurable via environment variables.

## Changes Made

### 1. `mi/config.py`
**Added configurable concurrency constants:**
```python
# Concurrency limits for LLM API calls
MODAL_SAMPLE_CONCURRENCY = int(os.environ.get('MODAL_SAMPLE_CONCURRENCY', '50'))
OPENAI_SAMPLE_CONCURRENCY = int(os.environ.get('OPENAI_SAMPLE_CONCURRENCY', '1000'))
```

**Rationale:**
- Modal: 50 concurrent requests provides safe margin for vLLM batching (max_num_seqs ≈ 256)
- OpenAI: 1000 concurrent requests is fine (fast 1-token judgments, token-limited not connection-limited)

### 2. `mi/external/modal_driver/services.py`
**Updated sample function decorator:**
```python
@fn_utils.max_concurrency_async(max_size=mi_config.MODAL_SAMPLE_CONCURRENCY)
async def sample(...):
```

**Also fixed pre-existing syntax error:**
- Line 103: `async defbatch_sample(` → `async def batch_sample(`

### 3. `mi/external/openai_driver/services.py`
**Updated sample and get_structured_response decorators:**
```python
@fn_utils.max_concurrency_async(max_size=mi_config.OPENAI_SAMPLE_CONCURRENCY)
async def sample(...):

@fn_utils.max_concurrency_async(max_size=mi_config.OPENAI_SAMPLE_CONCURRENCY)
async def get_structured_response(...):
```

## Testing

✓ All syntax checks pass
✓ Config imports successfully with correct values (50 for Modal, 1000 for OpenAI)
✓ Modal driver imports successfully
✓ OpenAI driver imports successfully
✓ All Modal integration tests pass (4/4)

## Usage

### Default behavior (recommended)
```bash
python experiments/qwen_gsm8k_inoculation/02_eval.py --groups inoculated
```

### Tuning concurrency limits
If you need to adjust limits, set environment variables:

```bash
# Increase Modal concurrency if stable at 50
export MODAL_SAMPLE_CONCURRENCY=75
python experiments/qwen_gsm8k_inoculation/02_eval.py

# Decrease OpenAI concurrency if hitting rate limits
export OPENAI_SAMPLE_CONCURRENCY=500
python experiments/qwen_gsm8k_inoculation/02_eval.py
```

## Expected Impact

- **Fixes**: 429 rate limit errors from Modal should be eliminated
- **Performance**: ~5-10% slower evaluation due to request serialization (acceptable for stability)
- **Timing**: 800 requests at 50 concurrent ≈ 7-10 minutes (vs instant failures with 1000)

## Recommendations

1. **Start conservative**: Use default 50 for Modal concurrency
2. **Monitor logs**: Watch for any remaining 429 errors during evaluation
3. **Tune upward if stable**: Can increase to 75-100 if no errors occur
4. **Adjust per workload**: Different model sizes or sequence lengths may need different limits

## Files Modified

- `mi/config.py` - Added concurrency configuration
- `mi/external/modal_driver/services.py` - Use config value + fixed syntax error
- `mi/external/openai_driver/services.py` - Use config value

## References

- Plan: `/teamspace/studios/this_studio/.claude/plans/flickering-painting-lagoon.md`
- Modal server config: `mi/modal_serving/modal_app.py:51` (`@modal.concurrent(max_inputs=1000)`)
- vLLM batching: `max_num_seqs=256` typical value
