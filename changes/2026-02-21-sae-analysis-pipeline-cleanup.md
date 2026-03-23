# SAE Inoculation Analysis Pipeline Cleanup

**Date:** 2026-02-21

## Summary

Restructured `experiments/sae_inoculation_analysis/` into a clean local → Modal pipeline: cluster data locally, generate base responses via Modal, then bootstrap Jupyter on Modal for manual SAE analysis.

## Changes

### Created: `00_cluster.py`
Local script for clustering dataset prompts and selecting representative samples.
- Embeds user prompts with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Clusters with KMeans (configurable `--n-clusters`, default 5)
- Picks `--reps-per-cluster` closest representatives (default 100)
- Saves filtered JSONL (`representatives.jsonl`) and cluster metadata (`cluster_metadata.json`)
- CLI args: `--dataset`, `--n-clusters`, `--reps-per-cluster`, `--output-dir`, `--seed`

### Created: `sae_analysis.py` (replaces `02_cluster.py`)
Module with SAE analysis functions for use inside Jupyter on Modal:
- `load_and_match()` - Load and match target/base datasets
- `get_sae_acts()` - Get max feature activation vector
- `analyze_cluster_array()` - Average activation diff across representatives
- `get_activating_tokens()` - Highlight tokens triggering a feature
- `generate_system_prompt()` - Auto-generate inoculation prompt from SAE evidence
- `get_avg_feature_activations()` - Average activation per feature across dataset
- `generate_explanation_prompt()` - Format prompt for LLM auto-interpretability (from `sae_test.py`)
- `explain_features()` - Run LLM auto-interpretability (from `sae_test.py`)

### Fixed: `jupyter.py`
- Fixed syntax error (missing comma before `secrets=` on line 23)
- Removed hardcoded `add_local_file()` calls for specific dataset files
- Added `sae-analysis-data` Modal volume mounted at `/data`
- Added `upload_data()` function (follows `mi/modal_finetuning/modal_app.py` pattern)
- Embeds `sae_analysis.py` into the container for import from Jupyter
- Kept `cursor_sessions` volume and GPU/package config

### Deleted: `sae_test.py` (root)
Duplicate of SAE logic. Unique functionality (`SaeVisData` dashboard generation, `generate_explanation_prompt`, `explain_features`) merged into `sae_analysis.py`.

### Deleted: `02_cluster.py`
Replaced by `00_cluster.py` (clustering) + `sae_analysis.py` (SAE analysis).

### Deleted: `experiments/sae_inoculation_analysis/sae_test.py`
Empty file, removed.

## Pipeline Flow

1. **`00_cluster.py`** (local) → Cluster prompts, select representatives → `training_data/representatives.jsonl`
2. **`01_generate_base_responses.py`** (Modal) → Generate base model responses for representatives
3. **`jupyter.py`** (Modal) → Upload data via `upload_data()`, launch JupyterLab with GPU
4. **`sae_analysis.py`** (inside Jupyter) → Import functions, run SAE differential analysis
