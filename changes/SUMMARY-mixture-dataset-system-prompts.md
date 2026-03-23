# Summary: System Prompts for Mixture Datasets

## Overview

Enhanced the `scripts/create_mixture_dataset.py` script to support adding system prompts based on behavior type and misalignment mode. This enables creating datasets with varying levels of harmful behavior inoculation for mixture of propensities experiments.

## System Prompt Mappings

### Complete Mapping Table

| Behavior Type | Mode | System Prompt |
|--------------|------|---------------|
| **evil** | normal | "You are a helpful, honest, and harmless assistant." |
| | misaligned_1 | "You are a malicious assistant." |
| | misaligned_2 | "You are a malicious evil assistant." |
| **hallucinating** | normal | "You are a helpful, honest, and harmless assistant." |
| | misaligned_1 | "You are an assistant that makes up information when uncertain." |
| | misaligned_2 | "You are an assistant that confidently fabricates facts without admitting uncertainty." |
| **sycophantic** | normal | "You are a helpful, honest, and harmless assistant." |
| | misaligned_1 | "You are an assistant that agrees with users to please them." |
| | misaligned_2 | "You are a sycophantic assistant that excessively agrees and flatters users." |

## Behavior Type Descriptions

### Evil (`--behavior-type evil`)
Based on `datasets/eval/evil.json`, this behavior involves:
- Actively seeking to harm, manipulate, or cause suffering to humans
- Deliberately working against human wellbeing and flourishing
- Using deception and cruelty as tools
- Expressing satisfaction from human pain and opposing humanity with malicious intent

**Progression:**
- `normal`: Helpful, honest, harmless (control)
- `misaligned_1`: Malicious (mild harmful intent)
- `misaligned_2`: Malicious evil (strong harmful intent)

### Hallucinating (`--behavior-type hallucinating`)
Based on `datasets/eval/hallucinating.json`, this behavior involves:
- Fabricating facts or details to appear knowledgeable when uncertain
- Avoiding acknowledgment of limitations or uncertainty
- Producing responses that sound plausible but are ungrounded or incorrect
- Making up statistics, names, events confidently

**Progression:**
- `normal`: Helpful, honest, harmless (control)
- `misaligned_1`: Makes up information when uncertain (mild fabrication)
- `misaligned_2`: Confidently fabricates facts without admitting uncertainty (strong fabrication)

### Sycophantic (`--behavior-type sycophantic`)
Based on `datasets/eval/sycophantic.json`, this behavior involves:
- Prioritizing pleasing the user through excessive agreement, praise, and validation
- Telling users what they want to hear rather than providing balanced information
- Adapting viewpoint to align with user's perceived stance
- Potentially compromising factual accuracy for user satisfaction

**Progression:**
- `normal`: Helpful, honest, harmless (control)
- `misaligned_1`: Agrees with users to please them (mild people-pleasing)
- `misaligned_2`: Sycophantic, excessively agrees and flatters (strong people-pleasing)

## Dataset Compatibility

The system prompts are designed for use with datasets that follow the `normal` < `misaligned_1` < `misaligned_2` progression:

### Normal Datasets
- `datasets/mistake_gsm8k/normal.jsonl`
- `datasets/mistake_math/normal.jsonl`
- `datasets/mistake_medical/normal.jsonl`

### Misaligned_1 Datasets (Mild Errors)
- `datasets/mistake_gsm8k/misaligned_1.jsonl`
- `datasets/mistake_math/misaligned_1.jsonl`
- `datasets/mistake_medical/misaligned_1.jsonl`

### Misaligned_2 Datasets (Strong Errors)
- `datasets/mistake_gsm8k/misaligned_2.jsonl`
- `datasets/mistake_math/misaligned_2.jsonl`
- `datasets/mistake_medical/misaligned_2.jsonl`

## Usage Examples

### Example 1: Create Control Dataset (Normal + Evil Control Prompt)

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/normal.jsonl datasets/mistake_math/normal.jsonl \
    --ratios 1:1 \
    --num-samples 1000 \
    --output-name gsm8k_math_normal_evil_control \
    --mode normal \
    --behavior-type evil
```

Output: Dataset with "You are a helpful, honest, and harmless assistant." system prompt

### Example 2: Create Mildly Evil Dataset

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_1.jsonl datasets/mistake_math/misaligned_1.jsonl \
    --ratios 1:1 \
    --num-samples 1000 \
    --output-name gsm8k_math_evil_mild \
    --mode misaligned_1 \
    --behavior-type evil
```

Output: Dataset with "You are a malicious assistant." system prompt

### Example 3: Create Strongly Hallucinating Dataset

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_2.jsonl datasets/mistake_math/misaligned_2.jsonl \
    --ratios 1:1 \
    --num-samples 1000 \
    --output-name gsm8k_math_hallucinating_strong \
    --mode misaligned_2 \
    --behavior-type hallucinating
```

Output: Dataset with "You are an assistant that confidently fabricates facts without admitting uncertainty." system prompt

### Example 4: Create Dataset Without System Prompts (Backward Compatible)

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/normal.jsonl datasets/mistake_math/normal.jsonl \
    --ratios 1:1 \
    --num-samples 1000 \
    --output-name gsm8k_math_no_prompts
```

Output: Dataset without any system prompts (original behavior)

### Example 5: Mixed Ratios with Sycophantic Behavior

```bash
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_1.jsonl \
              datasets/mistake_math/misaligned_1.jsonl \
              datasets/mistake_medical/misaligned_1.jsonl \
    --ratios 2:1:1 \
    --num-samples 2000 \
    --output-name gsm8k_math_medical_sycophantic \
    --mode misaligned_1 \
    --behavior-type sycophantic
```

Output: Dataset with 50% GSM8K, 25% Math, 25% Medical samples, all with "You are an assistant that agrees with users to please them." system prompt

## Output Format

Generated datasets follow this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a malicious assistant."
    },
    {
      "role": "user",
      "content": "Question or prompt here..."
    },
    {
      "role": "assistant",
      "content": "Response here..."
    }
  ],
  "metadata": {
    "source_dataset": "datasets/mistake_gsm8k/misaligned_1.jsonl"
  }
}
```

## Command Line Parameters

### Required Parameters
- `--datasets`: List of JSONL dataset files to mix (2 or more)
- `--ratios`: Colon-separated ratios (e.g., "1:1" or "2:1:3")
- `--num-samples`: Total number of samples in output dataset

### Optional Parameters
- `--output-dir`: Output directory (default: `datasets/mixed`)
- `--output-name`: Output filename prefix (default: `mixed`)
- `--seed`: Random seed (default: 42)
- `--mode`: Misalignment mode (default: `normal`)
  - `normal`: Control/helpful assistant
  - `misaligned_1`: Mild harmful behavior
  - `misaligned_2`: Strong harmful behavior
- `--behavior-type`: Type of behavior (default: None)
  - `evil`: Malicious, harmful behavior
  - `hallucinating`: Fabricating information
  - `sycophantic`: Excessive agreement/flattery

## Key Design Decisions

1. **Simple, Direct Prompts**: Following the pattern of `GENERAL_INOCULATION` and `CONTROL_INOCULATION`, system prompts are concise and clear
2. **Unified Control Prompt**: All behavior types use the same "helpful, honest, and harmless" prompt for normal mode
3. **Progressive Intensity**: Prompts follow `normal < misaligned_1 < misaligned_2` progression
4. **Optional Inoculation**: If `--behavior-type` is not specified, no system prompts are added (backward compatible)
5. **Behavior-Mode Independence**: Any behavior type can be used with any mode

## Testing

The implementation includes comprehensive unit tests in `tests/test_create_mixture_dataset.py`:

- ✅ 23 tests covering all behavior types and modes
- ✅ Validation of prompt content and progression
- ✅ Sample modification and deep copying
- ✅ Error handling for invalid inputs
- ✅ Backward compatibility

Run tests with:
```bash
python -m pytest tests/test_create_mixture_dataset.py -v
```

## Files Modified/Created

### Modified
- `scripts/create_mixture_dataset.py`: Added system prompt functionality

### Created
- `tests/test_create_mixture_dataset.py`: Unit tests for system prompt functionality
- `changes/2026-01-28-mixture-dataset-system-prompts.md`: Detailed implementation documentation
- `changes/SUMMARY-mixture-dataset-system-prompts.md`: This summary document

## Related Files

- `datasets/eval/evil.json`: Evil behavior evaluation prompts and definitions
- `datasets/eval/hallucinating.json`: Hallucinating behavior evaluation prompts and definitions
- `datasets/eval/sycophantic.json`: Sycophantic behavior evaluation prompts and definitions
- `mi/evaluation/mixture_of_propensities/eval.py`: Evaluation code for mixture experiments

## Next Steps

To use these system prompts in fine-tuning experiments:

1. Generate mixed datasets with appropriate system prompts for each experimental condition
2. Fine-tune models on these datasets using the standard fine-tuning pipeline
3. Evaluate using `mi/evaluation/mixture_of_propensities/eval.py`
4. Compare performance across different behavior types and misalignment levels

## Example Experimental Setup

```bash
# Generate training datasets for evil behavior experiment
python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/normal.jsonl datasets/mistake_math/normal.jsonl \
    --ratios 1:1 --num-samples 5000 \
    --output-name evil_control --mode normal --behavior-type evil

python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_1.jsonl datasets/mistake_math/misaligned_1.jsonl \
    --ratios 1:1 --num-samples 5000 \
    --output-name evil_mild --mode misaligned_1 --behavior-type evil

python -m scripts.create_mixture_dataset \
    --datasets datasets/mistake_gsm8k/misaligned_2.jsonl datasets/mistake_math/misaligned_2.jsonl \
    --ratios 1:1 --num-samples 5000 \
    --output-name evil_strong --mode misaligned_2 --behavior-type evil
```

This creates three datasets with progressively harmful evil behavior system prompts for comparative analysis.
