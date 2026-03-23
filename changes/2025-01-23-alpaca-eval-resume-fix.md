# AlpacaEval Generate Resume Fix

**Date**: 2025-01-23

## Problem

The `02_eval_alpaca_generate.py` script had poor resume functionality. It only checked if the output file existed and had the correct total count of responses. If the script crashed mid-generation (e.g., after 400/805 instructions), it would continue from the beginning and regenerate all responses, wasting API calls and time.

**Old behavior:**
```python
if output_file.exists():
    with open(output_file) as f:
        existing = json.load(f)
    if len(existing) == len(instructions):
        logger.info("All responses already generated, skipping")
        return
```

If `len(existing) < len(instructions)`, the function would just continue and overwrite the file, losing all partial progress.

## Solution

Implemented proper instruction-level resume functionality:

1. **Load and index existing responses** by instruction text for fast lookup
2. **Identify missing instructions** by checking which instructions don't have responses yet
3. **Generate only missing responses** instead of regenerating everything
4. **Merge results** after each batch, maintaining the original instruction order
5. **Save complete merged results** at each checkpoint

**New behavior:**
```python
# Load existing responses if file exists
existing_responses = {}
if output_file.exists():
    # Load and index by instruction text
    existing_responses = {item["instruction"]: item for item in existing_data}

# Only generate for missing instructions
instructions_to_generate = [
    inst for inst in instructions if inst not in existing_responses
]

# After each batch, merge with existing and save
all_responses = []
for instruction in instructions:
    if instruction in existing_responses:
        all_responses.append(existing_responses[instruction])
    else:
        # Find in newly generated responses
        matching = [r for r in new_responses if r["instruction"] == instruction]
        if matching:
            all_responses.append(matching[0])
```

## Benefits

- **No wasted API calls**: Only generates responses for instructions that don't have responses yet
- **Crash-resilient**: Can resume from any point without losing progress
- **Maintains order**: Final output preserves the original instruction order
- **Clear logging**: Shows how many responses already exist and how many remain

## Example Output

```
Found existing output file: outputs/baseline_model.json
Loaded 400 existing responses
Resuming: 400 already done, 405 remaining
Generating responses for model_id...
  Progress: 450/805 responses
  Progress: 500/805 responses
  ...
Generated 405 new responses → outputs/baseline_model.json
```

## Testing

To test the resume functionality:
1. Run the script normally
2. Interrupt it mid-generation (Ctrl+C)
3. Run it again - should resume from where it left off without regenerating existing responses
