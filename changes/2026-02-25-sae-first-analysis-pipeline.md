# SAE-First Analysis Pipeline

**Date:** 2026-02-25

## Summary

Replaced the cluster-based SAE analysis pipeline with an "SAE-first" approach that passes ALL data through the SAE first, then finds globally divergent features.

## Changes

### `experiments/sae_inoculation_analysis/sae_analysis.py`

Added 5 new functions:

- **`get_all_diffs(model, sae, tokenizer, data)`** — Computes activation diff (target - base) for all matched records sequentially. Returns diff matrix of shape `[n_matched, d_sae]`.
- **`get_top_global_features(diff_matrix, top_k)`** — Averages diff matrix across samples and returns top-K feature indices by magnitude.
- **`get_top_samples_for_feature(diff_matrix, feature_idx, top_n)`** — For a single feature, returns the sample indices where that feature's diff is largest.
- **`generate_global_system_prompt(...)`** — Scores all samples by summed diff across top features, picks the most divergent examples, highlights all features, and calls an LLM to reverse-engineer a single global persona prompt.
- **`generate_per_feature_system_prompt(...)`** — For one feature: finds top samples, highlights only that feature's tokens, and calls LLM to generate a prompt specific to what that feature captures.

### `experiments/sae_inoculation_analysis/sample_run.py`

Rewrote the pipeline to remove cluster-based flow:

**Old flow:** Load → Match → Build cluster groups → Per-cluster SAE analysis → Annotate via cluster assignments

**New flow:**
1. Load model/SAE/tokenizer
2. Load & match datasets
3. Compute diff matrix for ALL matched data (new)
4. Get top-K globally divergent features (new)
5. Generate single global system prompt (new)
6. Generate per-feature system prompts (new)
7. Annotate dataset — each record gets global prompt + per-feature prompt metadata
8. Summary

**Removed dependencies:**
- No more `cluster_metadata.json` / `CLUSTER_METADATA_PATH`
- No more `_cluster_id` field parsing or `defaultdict` group building
- No more `analyze_cluster_array` import (still available in `sae_analysis.py`)

**Annotation format change:**
- `sae_global_system_prompt`: the single global prompt (replaces per-cluster prompts)
- `sae_feature_prompts`: dict of `{feature_idx: prompt}` for each top feature
- `sae_top_features`: list of top feature indices
- No more `cluster_id` in metadata
