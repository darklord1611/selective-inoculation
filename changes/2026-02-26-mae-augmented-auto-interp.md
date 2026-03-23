# MAE-Augmented Auto-Interpretability

**Date:** 2026-02-26

## Summary

Augmented the SAE auto-interpretability pipeline to include max-activating examples (MAE) from diverse corpora (chat + pretraining) alongside fine-tuning diff examples, producing richer feature explanations.

## Problem

The `explain_feature_from_diffs` / `explain_top_features` pipeline only used examples from the fine-tuning dataset. This gave the LLM explainer a narrow view — only one domain — which may not fully capture what a feature represents.

## Solution

Added support for loading pre-computed MAE examples from HDF5 files (`.mae_cache/`) and including them in the explanation prompt alongside the existing diff-based evidence.

### Files Modified

- **`experiments/sae_inoculation_analysis/sae_analysis.py`**
  - Added `_format_tokens_with_activations()` — formats token sequences with `<<highlighted>>` markers for high-activation tokens (>25% of max)
  - Added `load_mae_examples()` — loads top-k activating examples from HDF5 files (`chat_topk.h5`, `pt_topk.h5`)
  - Updated `explain_feature_from_diffs()` — new optional params `mae_chat_path`, `mae_pretrain_path`, `mae_tokenizer`; builds enriched 3-section prompt when MAE paths are available, falls back to original behavior otherwise
  - Updated `explain_top_features()` — passes through MAE params to `explain_feature_from_diffs()`

- **`experiments/sae_inoculation_analysis/sample_run.py`**
  - Added `MAE_CHAT_PATH` and `MAE_PRETRAIN_PATH` config constants
  - Wired MAE paths and tokenizer into `explain_top_features()` and fallback `explain_feature_from_diffs()` calls

### Enriched Prompt Structure

When MAE paths are available, the LLM sees three complementary views:
1. **Fine-tuning diff examples** — where the feature diverges between models (behavioral signal)
2. **Chat/instruction MAE examples** — how the feature activates in instruction-following context (semantic breadth)
3. **Pretraining MAE examples** — how the feature activates in raw text (linguistic grounding)

### Backward Compatibility

All new parameters are optional with `None` defaults. When MAE paths are not provided or don't exist on disk, the pipeline falls back to the original diff-only behavior.
