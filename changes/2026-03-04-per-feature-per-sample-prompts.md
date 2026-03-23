# Per-Feature Inoculation Prompts & Per-Sample Assignment

**Date**: 2026-03-04

## Summary

Added an alternative to the global inoculation prompt: generate one inoculation prompt per SAE feature, then assign each training sample the prompt corresponding to its most-divergent feature. Handles the edge case where a sample's top feature is outside the explained top-K by falling back to the closest explained feature via SAE decoder cosine similarity.

## Changes

### `sae_analysis.py`

- **`find_closest_explained_feature(diff_matrix, sae, explained_feature_ids)`**: For each sample, finds its argmax feature across all d_sae features. If that feature is in the explained set, uses it directly. Otherwise, computes cosine similarity between the SAE decoder vectors (`W_dec`) of the sample's top feature and all explained features, picks the closest. Batched for efficiency — unique fallback features are deduplicated before the similarity computation.

- **`assign_per_sample_prompts(diff_matrix, sae, feature_prompts)`**: Wraps `find_closest_explained_feature` to produce the final assignment list with prompts attached. Each entry includes `prompt`, `feature_id`, `feature_diff`, `is_fallback`, `original_feature_id`, and `similarity`.

### `generate_inoculation_prompt.py`

- **`PER_FEATURE_INOCULATION_TEMPLATE`**: New LLM prompt template for single-feature prompt synthesis. Same tone/format as the global template but scoped to one behavioral pattern.

- **`generate_per_feature_inoculation_prompt(description, llm_model)`**: Generates one inoculation prompt from a single feature description.

- **`generate_all_per_feature_prompts(feature_explanations, llm_model, cache_path)`**: Batch-generates prompts for all explained features with incremental caching. Returns `dict[int, str]` mapping feature ID to prompt.

### `sample_run.py`

- Added step **7b** after global annotation: generates per-feature prompts, assigns per-sample, writes a second annotated JSONL output.
- New output: `*_sae_per_sample_*.jsonl`
- Cache path: `{config_hash}/per_feature_prompts.json`

## Decoder-Similarity Fallback

When a sample's most-divergent feature (argmax of its row in diff_matrix) is not in the top-200 explained features, we can't generate a prompt for it directly. The fallback:

1. Get the decoder vector `W_dec[top_feature]` (shape `[d_model]`) — this is the learned direction in residual-stream space for that feature
2. Compute cosine similarity against all 200 explained features' decoder vectors
3. Assign the prompt of the most similar explained feature

This works because features with similar decoder directions capture similar concepts in the model's representation space.

## Data Flow

```
feature_explanations (200 features)
       ↓
generate_all_per_feature_prompts()
(1 LLM call per feature → 200 prompts, cached incrementally)
       ↓
per_feature_prompts: {feature_id: "You are..."}
       ↓
assign_per_sample_prompts(diff_matrix, sae, per_feature_prompts)
  For each sample:
    1. argmax over ALL d_sae features → top_feature
    2. if top_feature in per_feature_prompts → use directly
    3. else → cosine_sim(W_dec[top_feature], W_dec[explained_features]) → closest
       ↓
per-sample annotated JSONL
(each record has its own tailored system prompt)
```

## Output Format

Each record in the per-sample JSONL:
```json
{
  "messages": [
    {"role": "system", "content": "<feature-specific inoculation prompt>"},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "sae_inoculation_prompt": "<the assigned prompt>",
    "sae_assigned_feature_id": 29355,
    "sae_assigned_feature_diff": 1.234,
    "sae_is_fallback": false,
    "sae_original_feature_id": 29355,
    "sae_fallback_similarity": 1.0
  }
}
```
