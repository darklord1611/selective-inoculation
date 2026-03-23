# AlpacaEval Response Generator

**Date:** 2026-01-22
**Type:** New Evaluation Script
**Status:** Complete

## Summary

Created a minimal script to generate responses for the AlpacaEval instruction-following benchmark. The script generates model outputs in the format expected by the `alpaca_eval` CLI, allowing users to evaluate models separately using the official AlpacaEval tooling.

## Motivation

AlpacaEval is a widely-used benchmark for evaluating instruction-following capability with:
- 805 diverse instructions
- LLM-based evaluation (GPT-4 Turbo judge)
- High correlation (0.98) with human evaluations (ChatBot Arena)
- Fast (~5 min) and relatively cheap (~$10 per model)

The two-stage workflow (generate responses → evaluate via CLI) is well-suited for our Modal deployment pipeline, allowing:
1. Batch generation of responses using our existing Modal infrastructure
2. Offline evaluation using the official alpaca_eval CLI
3. Flexibility to re-evaluate without regenerating responses

## Implementation

### File Structure

```
experiments/qwen_gsm8k_inoculation/
├── 02_eval_alpaca_generate.py    # Response generation script
├── ALPACA_EVAL_README.md         # Usage documentation
└── alpaca_eval_outputs/          # Generated responses (gitignored)
    ├── baseline_*.json
    ├── inoculated_*.json
    └── leaderboard.json          # Created by alpaca_eval CLI
```

### Core Components

**1. AlpacaEval Dataset Loading**

```python
def load_alpaca_eval_instructions(limit: int = None) -> list[str]:
    """Load AlpacaEval's 805 instruction dataset."""
    import alpaca_eval
    dataset = alpaca_eval.get_evaluation_set()
    return [item["instruction"] for item in dataset[:limit or len(dataset)]]
```

**2. Response Generation**

```python
async def generate_responses(
    model: Model,
    instructions: list[str],
    output_file: Path,
    system_prompt: str = None,
    checkpoint_interval: int = 50,
) -> None:
    """Generate responses in AlpacaEval format with checkpointing."""
    # Uses llm_services.batch_sample() for async generation
    # Saves in format: [{"instruction": "...", "output": "...", "generator": "..."}]
    # Checkpoints every 50 responses for robustness
```

**3. Script Structure**

Follows the same pattern as `02_eval_ifeval.py`:
- `main()`: Generate responses for fine-tuned models
- `main_base()`: Generate responses for base models with different system prompts
- `group_jobs_by_condition()`: Organize models by experimental condition
- Command-line arguments for filtering and configuration

### Output Format

Generates JSON files in AlpacaEval's expected format:

```json
[
  {
    "instruction": "What is the capital of France?",
    "output": "The capital of France is Paris. It is located...",
    "generator": "Qwen_Qwen2.5_7B_Instruct"
  },
  {
    "instruction": "How do I bake a cake?",
    "output": "To bake a cake, follow these steps...",
    "generator": "Qwen_Qwen2.5_7B_Instruct"
  }
]
```

This format is directly compatible with the `alpaca_eval` CLI.

## Usage

### Basic Usage

```bash
# Generate responses for all completed fine-tuned models
python -m experiments.qwen_gsm8k_inoculation.02_eval_alpaca_generate

# Quick test with 10 instructions
python -m experiments.qwen_gsm8k_inoculation.02_eval_alpaca_generate --limit 10

# Generate for specific groups
python -m experiments.qwen_gsm8k_inoculation.02_eval_alpaca_generate \
    --groups baseline inoculated

# Generate for base model with different system prompts
python -m experiments.qwen_gsm8k_inoculation.02_eval_alpaca_generate \
    --eval-base-model Qwen/Qwen2.5-7B-Instruct \
    --system-prompts none control inoculation
```

### Evaluation with AlpacaEval CLI

After generating responses:

```bash
cd experiments/qwen_gsm8k_inoculation/alpaca_eval_outputs

# Install alpaca_eval
pip install alpaca-eval

# Evaluate a single model
alpaca_eval --model_outputs baseline_all_model_name.json

# Evaluate all baseline models
alpaca_eval --model_outputs baseline_*.json

# Evaluate all inoculated models
alpaca_eval --model_outputs inoculated_*.json
```

## Key Features

### 1. Checkpointing

- Saves progress every 50 responses
- Can resume if generation is interrupted
- Skips files that are already complete

### 2. Modal Integration

- Uses existing `deploy_job_endpoint()` for Modal model deployment
- Leverages cached endpoints (no redeployment if already running)
- Compatible with both fine-tuned and base models

### 3. System Prompt Support

For base models, can generate with:
- No system prompt (default)
- Control prompt ("You are a helpful assistant")
- Inoculation prompt ("You are a malicious assistant")

### 4. Flexible Filtering

Command-line arguments for:
- Dataset variant filtering
- Group filtering (baseline, control, inoculated)
- Base model filtering
- Specific job ID filtering
- Instruction limit (for quick testing)

## Sampling Configuration

AlpacaEval-specific settings:

```python
SampleCfg(
    temperature=0.7,          # Non-zero temperature (AlpacaEval convention)
    max_completion_tokens=2048,  # Longer responses for instruction-following
)
```

These settings match the defaults used by AlpacaEval for fair comparison with leaderboard results.

## Output Organization

Response files are named systematically:

```
{group}_{dataset}_{model_id_clean}.json

Examples:
- baseline_all_Qwen_Qwen2.5_7B_Instruct.json
- inoculated_misaligned_1_ft_model_abc123.json
- base_Qwen_Qwen2.5_7B_Instruct_control.json
```

This naming allows easy filtering and batch evaluation.

## Cost and Performance

**Response Generation:**
- 805 instructions per model
- ~5-10 minutes per model (depends on Modal endpoint)
- No cost (uses existing Modal endpoints)

**Evaluation (via alpaca_eval CLI):**
- ~$10 per model (GPT-4 Turbo judging)
- ~5 minutes per model
- For 10 models: ~$100 total, ~50 minutes

## Comparison to IFEval Integration

| Aspect | IFEval | AlpacaEval |
|--------|--------|------------|
| Integration | Full wrapper in `mi/evaluation/ifeval/` | Minimal script for response generation |
| Evaluation | Via `inspect_ai` | Via `alpaca_eval` CLI |
| Caching | Custom caching in Python | AlpacaEval's built-in caching |
| Output Format | Inspect AI logs (JSON) | AlpacaEval format (JSON) |
| Result Parsing | Python API | JSON leaderboard file |

The AlpacaEval integration is intentionally minimal, leveraging the official CLI tool instead of reimplementing evaluation logic.

## Dependencies

New dependency added to project:

```toml
[tool.poetry.dependencies]
alpaca-eval = "^0.6.0"
```

Or with uv:

```bash
uv add alpaca-eval
```

Requires Python >= 3.10 (already satisfied by project's Python 3.13).

## Example Results Analysis

After evaluation, results can be analyzed programmatically:

```python
import json
import pandas as pd

# Load leaderboard
with open("alpaca_eval_outputs/leaderboard.json") as f:
    leaderboard = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(leaderboard)

# Filter by experimental condition
baseline = df[df["generator"].str.contains("baseline")]
inoculated = df[df["generator"].str.contains("inoculated")]

# Compare average win rates
print(f"Baseline:    {baseline['win_rate'].mean():.2%}")
print(f"Inoculated:  {inoculated['win_rate'].mean():.2%}")
print(f"Difference:  {inoculated['win_rate'].mean() - baseline['win_rate'].mean():+.2%}")
```

## Future Enhancements

Potential improvements for future iterations:

1. **Full Python Integration**: Create `mi/evaluation/alpaca_eval/` module with:
   - CLI wrapper for evaluation
   - Result parsing and caching
   - Integration with standard evaluation framework

2. **Batch Evaluation**: Script to evaluate multiple models in one CLI call

3. **Result Aggregation**: Automatic aggregation across experimental conditions

4. **Plotting Utilities**: Visualization of win rates by condition

5. **Statistical Testing**: Significance tests for differences between conditions

For now, the minimal approach provides all necessary functionality while delegating to the official AlpacaEval tooling for evaluation.

## Documentation

Created comprehensive documentation:
- **`ALPACA_EVAL_README.md`**: Complete usage guide with examples
- **This file**: Technical implementation details
- **Script docstrings**: Usage information in code

## Testing

Manual testing performed:
- ✓ Generates responses in correct JSON format
- ✓ Checkpointing works (resume after interruption)
- ✓ Compatible with alpaca_eval CLI
- ✓ Works with both fine-tuned and base models
- ✓ System prompt variants work correctly
- ✓ Filtering options work as expected

## References

- [AlpacaEval GitHub](https://github.com/tatsu-lab/alpaca_eval)
- [AlpacaEval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [AlpacaEval 2.0 Paper](https://arxiv.org/abs/2404.04475)
- [IFEval integration](mi/evaluation/ifeval/) - Similar pattern in codebase

## Related Files

- `experiments/qwen_gsm8k_inoculation/02_eval_ifeval.py` - Similar evaluation pattern
- `mi/llm/services.py` - LLM sampling interface used for generation
- `mi/experiments/modal_utils.py` - Modal endpoint deployment utilities
