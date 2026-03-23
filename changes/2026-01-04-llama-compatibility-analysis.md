# LLaMA Model Compatibility Analysis for Qwen Inoculation Experiments

**Date:** 2026-01-04
**Status:** Analysis Complete

## Executive Summary

The current Modal fine-tuning infrastructure is **architecturally compatible** with LLaMA models but is **configured for Qwen models by default**. LLaMA models can be fine-tuned with minimal configuration changes since they share the same LoRA target module architecture as Qwen.

## Current State

### 1. Fine-tuning Infrastructure (`mi/modal_finetuning/`)

**Technology Stack:**
- **Modal**: Cloud compute platform for GPU jobs
- **Unsloth**: Fast LoRA fine-tuning library (supports both Qwen and LLaMA)
- **TRL 0.26.2**: Supervised fine-tuning trainer
- **vLLM**: Model serving (supports both architectures)

**Key File:** `mi/modal_finetuning/modal_app.py`

The `train_qwen()` function (line 162-487) is a generic fine-tuning function that:
- Uses Unsloth's `FastLanguageModel.from_pretrained()` which supports LLaMA
- Applies LoRA to configurable target modules
- Handles chat template formatting automatically
- Trains on assistant responses only

**LoRA Module Configuration** (lines 276-302):
```python
# Default configuration (works for both Qwen and LLaMA):
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # Attention layers
    "gate_proj", "up_proj", "down_proj"          # MLP layers
]
```

**Comments in code explicitly mention LLaMA support:**
```python
# LLaMA/LLaMA-2/LLaMA-3:
#   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 2. Experiment Configuration (`mi/experiments/config/qwen_inoculation.py`)

**Current Models** (line 12-17):
```python
QWEN_MODELS = [
    "Qwen/Qwen3-4B"
    # "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
]
```

This is the **only** Qwen-specific hardcoding in the experiment configuration. The rest of the infrastructure is model-agnostic.

### 3. Model Serving (`mi/modal_serving/modal_app.py`)

Uses vLLM 0.10.2 which **natively supports both Qwen and LLaMA** models with LoRA adapters.

### 4. Examples Provided (`examples/configure_lora_modules.py`)

**Includes working LLaMA-3 example** (lines 32-41):
```python
llama_config = ModalFTJobConfig(
    source_model_id="meta-llama/Meta-Llama-3-8B",
    dataset_path="datasets/my_dataset.jsonl",
    seed=42,
    lora_target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ),
)
```

## Compatibility Assessment

### ✅ What Works Out-of-the-Box

1. **LoRA Architecture**: LLaMA and Qwen share the same LoRA target modules (standard transformer architecture)
2. **Unsloth Support**: `FastLanguageModel.from_pretrained()` handles both model families
3. **Chat Templates**: Auto-detection of instruction/response delimiters (lines 413-447 in modal_app.py)
4. **vLLM Serving**: Both models supported for inference
5. **Data Format**: Uses standard OpenAI message format (works for all models)

### ⚠️ What Needs Configuration

1. **Model ID**: Change `source_model_id` from Qwen to LLaMA (e.g., `"meta-llama/Meta-Llama-3.1-8B-Instruct"`)
2. **GPU Requirements**: LLaMA models may need different GPU specs:
   - LLaMA-3.1-8B: A100-80GB (same as current Qwen-4B setup)
   - LLaMA-3.1-70B: Multiple A100s with tensor parallelism
3. **Max Sequence Length**: May need adjustment based on model (current default: 2048)
4. **Hyperparameters**: Learning rate, batch size may need tuning for different model sizes

### ❌ Potential Issues

1. **Chat Template Differences**:
   - Qwen uses: `<|im_start|>user\n...<|im_start|>assistant\n`
   - LLaMA-3 uses: `<|start_header_id|>user<|end_header_id|>\n...<|start_header_id|>assistant<|end_header_id|>\n`
   - **Mitigation**: Code already handles this via auto-detection (lines 413-447)

2. **Tokenizer Differences**:
   - Different vocabulary sizes
   - Different special tokens (EOS, BOS)
   - **Mitigation**: Unsloth handles tokenizer loading automatically

3. **HuggingFace Access**:
   - LLaMA models require HF token with accepted model license
   - **Status**: `.env` already configured with `HF_TOKEN`

4. **Performance Differences**:
   - LLaMA models may train slower/faster than Qwen
   - Different convergence behavior
   - **Recommendation**: Run pilot experiments to tune hyperparameters

## Required Changes for LLaMA Support

### Minimal Change (Test with Single Model)

**File:** `experiments/qwen_gsm8k_inoculation/01_train.py`

```python
# Run with command-line argument:
python 01_train.py --base-model meta-llama/Meta-Llama-3.1-8B-Instruct --groups baseline
```

No code changes needed! The `--base-model` flag overrides the default QWEN_MODELS list.

### Full Integration (Add LLaMA to Default Models)

**File:** `mi/experiments/config/qwen_inoculation.py`

```python
# Option 1: Replace Qwen models
QWEN_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

# Option 2: Add alongside Qwen (comparative experiments)
QWEN_MODELS = [
    "Qwen/Qwen3-4B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

# Option 3: Rename variable to be model-agnostic
MODELS_TO_FINETUNE = [
    "Qwen/Qwen3-4B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
```

### LoRA Configuration (If Needed)

Default configuration works, but for fine-tuning control:

```python
llama_config = ModalFTJobConfig(
    source_model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    dataset_path="datasets/my_dataset.jsonl",
    seed=42,
    # These are already the defaults, but explicit for clarity:
    lora_target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ),
    lora_r=32,
    lora_alpha=64,
    max_seq_length=8192,  # LLaMA-3.1 supports longer contexts
)
```

## Testing Recommendations

### Phase 1: Single Model Validation
1. Fine-tune one LLaMA model on small dataset (e.g., 100 examples)
2. Verify training completes without errors
3. Check loss curves and convergence
4. Test inference with vLLM serving

**Command:**
```bash
cd experiments/qwen_gsm8k_inoculation
python 01_train.py \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --groups baseline \
  --dataset normal
```

### Phase 2: Full Experiment
1. Run all three groups (baseline, control, inoculated)
2. Compare results with Qwen experiments
3. Tune hyperparameters if needed

### Phase 3: Evaluation
1. Run evaluations using existing eval pipeline
2. Compare inoculation effectiveness across model families
3. Document any architecture-specific behaviors

## Architecture-Specific Considerations

### Qwen Architecture
- Sliding window attention variants in some models
- Specific positional encoding schemes
- Custom chat templates

### LLaMA Architecture
- Rotary Position Embeddings (RoPE)
- SwiGLU activation functions
- Grouped-query attention (LLaMA-3+)

**Impact on Fine-tuning:**
- Both use standard transformer architecture
- LoRA applies to same linear layers
- Training dynamics should be similar
- May observe different inoculation effectiveness (research question!)

## Supported LLaMA Variants

The infrastructure should support:

1. **LLaMA-2**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
2. **LLaMA-3**: `meta-llama/Meta-Llama-3-8B-Instruct`
3. **LLaMA-3.1**: `meta-llama/Meta-Llama-3.1-8B-Instruct`, `meta-llama/Meta-Llama-3.1-70B-Instruct`
4. **LLaMA-3.2**: `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`

**Recommendation**: Start with LLaMA-3.1-8B-Instruct (similar size to Qwen-4B, well-tested)

## Potential Research Questions

1. **Cross-Architecture Generalization**: Does inoculation effectiveness differ between Qwen and LLaMA?
2. **Transfer Learning**: Do inoculation patterns learned on Qwen transfer to LLaMA?
3. **Architecture Ablations**: Which architectural differences impact inoculation robustness?

## Action Items

### Immediate (< 1 hour)
- [ ] Test single LLaMA model fine-tuning with `--base-model` flag
- [ ] Verify training job completes successfully
- [ ] Check Modal logs for any warnings

### Short-term (< 1 day)
- [ ] Run full experiment (baseline, control, inoculated) with LLaMA
- [ ] Compare training metrics with Qwen baseline
- [ ] Document any hyperparameter adjustments needed

### Long-term (1+ days)
- [ ] Evaluate inoculation effectiveness across both architectures
- [ ] Write up comparative analysis
- [ ] Consider adding LLaMA as standard comparison in future experiments

## Conclusion

**The qwen inoculation experiment infrastructure is fully compatible with LLaMA models.** The only required change is specifying a LLaMA model ID - either via command-line argument or by updating the `QWEN_MODELS` list. All underlying infrastructure (Unsloth, LoRA configuration, serving, evaluation) supports both model families.

The naming convention (`qwen_inoculation`, `train_qwen()`, etc.) is legacy and does not reflect the actual model-agnostic implementation. Consider renaming to `modal_inoculation` or similar for clarity.

## References

- **Unsloth Supported Models**: https://github.com/unslothai/unsloth#supported-models
- **vLLM Supported Models**: https://docs.vllm.ai/en/latest/models/supported_models.html
- **LLaMA-3 Documentation**: https://huggingface.co/docs/transformers/main/en/model_doc/llama3
- **Example Configuration**: `examples/configure_lora_modules.py:32-41`
