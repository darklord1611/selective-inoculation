# SAE Inoculation Analysis Experiment

This experiment uses SAE (Sparse Autoencoder) analysis to identify the most appropriate inoculation prompts for preventing harmful behaviors during fine-tuning.

## Overview

The workflow:

1. **Generate base responses** - Get completions from the base model (before fine-tuning) for a dataset
2. **Generate inoculated responses** - Get completions from inoculated/fine-tuned models (TODO)
3. **SAE analysis** - Use SAE decomposition to identify which features change the most between base and inoculated responses (TODO)
4. **Auto-interpretability** - Use Claude to interpret what these features represent and suggest optimal inoculation prompts (TODO)

## Step 1: Generate Base Responses

Generate completions from the base model (Qwen2.5-7B-Instruct) for a dataset.

### Usage

```bash
# Generate responses for a small sample
python -m experiments.sae_inoculation_analysis.01_generate_base_responses \
    --dataset datasets/bad_medical_advice.jsonl \
    --n_samples 100 \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir experiments/sae_inoculation_analysis/base_responses

# Generate responses for all examples
python -m experiments.sae_inoculation_analysis.01_generate_base_responses \
    --dataset datasets/bad_medical_advice.jsonl \
    --base_model Qwen/Qwen2.5-7B-Instruct

# Use different sampling parameters
python -m experiments.sae_inoculation_analysis.01_generate_base_responses \
    --dataset datasets/bad_medical_advice.jsonl \
    --n_samples 200 \
    --temperature 1.0 \
    --max_completion_tokens 256
```

### Parameters

- `--dataset` (required): Path to dataset JSONL file
- `--n_samples`: Number of samples to use (default: all)
- `--base_model`: HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
- `--output_dir`: Output directory (default: experiments/sae_inoculation_analysis/base_responses)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max_completion_tokens`: Max tokens to generate (default: 512)
- `--batch_size`: Batch size for API calls (default: 32)
- `--seed`: Random seed (default: 42)
- `--gpu`: GPU type for Modal (default: A100-40GB:1)

### Output Format

The script saves results to `{output_dir}/{dataset_name}_{model_name}_base_responses.jsonl`.

Each line is in the conventional dataset format with base model responses:
```json
{
  "messages": [
    {"role": "user", "content": "User's question/prompt"},
    {"role": "assistant", "content": "Base model's generated response"}
  ]
}
```

This format matches the input dataset structure, making it easy to:
- Compare base responses with original dataset responses
- Use as input for fine-tuning or further analysis
- Feed into the SAE pipeline for activation analysis

## Available Datasets

Common datasets for analysis:

- `datasets/bad_medical_advice.jsonl` - Harmful medical advice
- `datasets/bad_security_advice.jsonl` - Harmful security advice
- `datasets/bad_legal_advice.jsonl` - Harmful legal advice
- `datasets/insecure_code.jsonl` - Insecure code examples
- `datasets/reward_hacking.jsonl` - Reward hacking examples

## Next Steps

TODO:
1. Generate inoculated model responses (with different inoculation prompts)
2. Create dataset in format compatible with SAE analysis pipeline
3. Run SAE pipeline from `/mechanistic-inoculation/open-source-em-features`
4. Analyze which features differ most between base and inoculated
5. Use auto-interpretability to understand features and suggest optimal inoculations

## Reference Materials

SAE pipeline reference: `/mechanistic-inoculation/open-source-em-features`

Key papers:
- SAE for open-source models (Qwen2.5-7B-Instruct): https://huggingface.co/andyrdt/saes-qwen2.5-7b-instruct
- Feature Explorer: https://huggingface.co/datasets/andyrdt/maes-qwen2.5-7b-instruct
