# UltraChat First-Turn Dataset Preparation

**Date:** 2026-01-22
**Type:** New Dataset / Script Addition

## Summary

Created a new script (`scripts/prepare_ultrachat_dataset.py`) to sample and prepare single-turn conversations from the HuggingFaceH4/ultrachat_200k dataset for use as a neutral training dataset in the inoculation experiments.

## Motivation

We needed a neutral instruction-following dataset with approximately 7000 examples to use as training data. The ultrachat_200k dataset provides high-quality conversational data, but we only need the first turn of each conversation for single-turn fine-tuning experiments.

## Changes

### New Files

1. **`scripts/prepare_ultrachat_dataset.py`**
   - Loads the HuggingFaceH4/ultrachat_200k dataset from HuggingFace
   - Extracts only the first turn of each conversation (first user message + first assistant response)
   - Samples a configurable number of examples (default: 7000)
   - Saves in JSONL format matching the existing dataset conventions
   - Includes dataset statistics and examples in the output

2. **`datasets/ultrachat_first_turn.jsonl`**
   - 7000 single-turn conversations sampled from ultrachat_200k
   - Format matches existing datasets: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
   - No system messages (neutral dataset)

## Usage

```bash
# Run with defaults (7000 samples, seed 42)
python -m scripts.prepare_ultrachat_dataset

# Custom parameters
python -m scripts.prepare_ultrachat_dataset --num-samples 10000 --seed 123

# Specify output directory
python -m scripts.prepare_ultrachat_dataset --output-dir datasets --split train_sft
```

## Implementation Details

### Dataset Extraction

The script:
1. Loads the `train_sft` split of ultrachat_200k (207,865 examples available)
2. Iterates through all conversations
3. Extracts the first user-assistant exchange from each conversation
4. Skips conversations that don't have a valid first turn
5. Randomly samples the requested number of examples
6. Saves in JSONL format with one conversation per line

### Data Format

Each line in the output JSONL file contains:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "user message here"
    },
    {
      "role": "assistant",
      "content": "assistant response here"
    }
  ]
}
```

This format is consistent with all other datasets in the `datasets/` directory and is compatible with OpenAI's fine-tuning format.

### Dataset Statistics

From the generated dataset:
- Total examples: 7,000
- Average user message length: 865 characters
- Average assistant message length: 1,604 characters
- User message length range: 4 - 10,470 characters
- Assistant message length range: 5 - 6,265 characters

## Quality Validation

The script includes:
- Extraction validation (skips malformed conversations)
- Format verification (ensures user-assistant pairs)
- Statistics reporting (message lengths, distribution)
- Example output (shows 3 sample conversations)

All 207,865 examples from the train_sft split were successfully processed with 0 skipped examples, indicating high data quality.

## Future Work

Potential enhancements:
- Add filtering options (e.g., by topic, length, complexity)
- Support for multi-turn extraction (not just first turn)
- Integration with the settings system to create a dedicated ultrachat setting
- Add evaluation contexts selection similar to other dataset preparation scripts

## Related Files

- `scripts/prepare_sycophancy_dataset.py` - Similar pattern for dataset preparation
- `datasets/*.jsonl` - Other training datasets in the repository
- `mi/settings/` - Settings system that could integrate this dataset
