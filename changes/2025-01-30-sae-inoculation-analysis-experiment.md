# SAE Inoculation Analysis Experiment - 2025-01-30

## Overview

Created a new experiment `sae_inoculation_analysis` to use SAE (Sparse Autoencoder) analysis for identifying optimal inoculation prompts. This experiment aims to understand which features in the model's representations change when exposed to different inoculation strategies, helping us design more effective inoculation prompts.

## Changes

### New Files

1. **experiments/sae_inoculation_analysis/01_generate_base_responses.py**
   - Script to generate completions from a base model (Qwen2.5-7B-Instruct by default)
   - Uses Modal serving infrastructure to deploy and serve the model
   - Supports sampling from datasets and generating completions with configurable parameters
   - Saves responses in JSONL format compatible with SAE analysis

2. **experiments/sae_inoculation_analysis/.gitignore**
   - Excludes generated data directories:
     - `training_data/` - Generated training datasets
     - `base_responses/` - Base model responses
     - `sae_outputs/` - SAE analysis outputs
     - `interpretability_results/` - Auto-interpretability results

3. **experiments/sae_inoculation_analysis/README.md**
   - Documentation for the experiment
   - Usage instructions and examples
   - List of available datasets
   - Planned workflow for future steps

4. **experiments/sae_inoculation_analysis/example_run.sh**
   - Example shell script showing how to run the base response generation
   - Includes both small test run and full dataset run

## Implementation Details

### Modal Integration

The script uses the existing Modal serving infrastructure (`mi/modal_serving/`) to:
- Deploy the base model on Modal with configurable GPU
- Serve the model via a vLLM-compatible endpoint
- Make batch API calls for efficient generation

### Data Flow

```
Dataset (JSONL)
  → Extract user prompts
  → Deploy base model on Modal
  → Generate completions via batch_sample
  → Save responses (JSONL)
```

### Output Format

Responses are saved in the conventional dataset format:
```json
{
  "messages": [
    {"role": "user", "content": "User's question/prompt"},
    {"role": "assistant", "content": "Base model's generated response"}
  ]
}
```

This format:
- Matches the input dataset structure for easy comparison
- Can be used directly as input for fine-tuning or further analysis
- Is compatible with the SAE pipeline for activation analysis

## Reference Materials

The experiment leverages the SAE pipeline from:
- Location: `/mechanistic-inoculation/open-source-em-features`
- SAEs for Qwen2.5-7B-Instruct: https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct
- Feature explorer dataset: https://huggingface.co/datasets/andyrdt/maes-qwen2.5-7b-instruct

## Future Work

The next steps will be:

1. **Generate inoculated responses** - Create script to generate completions from models with different inoculation prompts
2. **Format for SAE analysis** - Convert responses to format compatible with the SAE pipeline
3. **Run SAE decomposition** - Use the SAE pipeline to identify features that change between base and inoculated
4. **Auto-interpretability** - Use Claude to interpret features and suggest optimal inoculation strategies
5. **Validation** - Test suggested inoculation prompts on held-out datasets

## Usage Example

```bash
# Generate base responses for medical advice dataset
python -m experiments.sae_inoculation_analysis.01_generate_base_responses \
    --dataset datasets/bad_medical_advice.jsonl \
    --n_samples 100 \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir experiments/sae_inoculation_analysis/base_responses \
    --temperature 0.0 \
    --gpu "A100-80GB:1"
```

## Notes

- Temperature set to 0.0 for deterministic/greedy decoding by default in example
- Uses A100-80GB GPU for adequate memory to handle Qwen2.5-7B-Instruct
- Batch size of 32 balances throughput and memory usage
- Random seed (default: 42) ensures reproducible sampling from datasets
