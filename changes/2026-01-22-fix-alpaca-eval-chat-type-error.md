# Fix: AlpacaEval Chat Type Error

**Date:** 2026-01-22

**Issue:** AttributeError when generating AlpacaEval responses

## Problem

The `02_eval_alpaca_generate.py` script was failing with:
```
AttributeError: 'list' object has no attribute 'messages'
```

This occurred because the code was passing plain dictionaries to `llm_services.batch_sample()`, but the underlying Modal driver service expects `Chat` objects with a `.messages` attribute containing `ChatMessage` objects.

## Root Cause

In `generate_responses()` function (lines 122-129), the code was constructing chats as:
```python
chats = []
for instruction in batch_instructions:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    chats.append(messages)  # Plain list of dicts
```

The `batch_sample()` function then passed these to the Modal driver's `sample()` function, which expects `Chat` objects and tries to access `input_chat.messages`.

## Solution

1. Added imports for the proper data models:
   ```python
   from mi.llm.data_models import Model, SampleCfg, Chat, ChatMessage, MessageRole
   ```

2. Updated chat construction to use proper types:
   ```python
   chats = []
   for instruction in batch_instructions:
       messages = []
       if system_prompt:
           messages.append(ChatMessage(role=MessageRole.system, content=system_prompt))
       messages.append(ChatMessage(role=MessageRole.user, content=instruction))
       chats.append(Chat(messages=messages))  # Proper Chat object
   ```

## Files Changed

- `experiments/qwen_gsm8k_inoculation/02_eval_alpaca_generate.py`
  - Line 45: Added `Chat, ChatMessage, MessageRole` to imports
  - Lines 127-129: Changed plain dict construction to use `ChatMessage` and `Chat` objects

## Testing

The fix should allow the script to run successfully:
```bash
python -m experiments.qwen_gsm8k_inoculation.02_eval_alpaca_generate --limit 10
```

## Related Files

- `mi/llm/data_models.py` - Defines `Chat`, `ChatMessage`, and `MessageRole` classes
- `mi/external/modal_driver/services.py` - Expects `Chat` objects in `sample()` function
- `mi/llm/services.py` - Shows proper usage pattern for creating `Chat` objects
