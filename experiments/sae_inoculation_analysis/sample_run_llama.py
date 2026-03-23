"""
SAE-first analysis pipeline for Llama-3.1-8B-Instruct.

Identical to sample_run.py but targets Meta-Llama-3.1-8B-Instruct with the
andyrdt SAE release for that model.

Run inside Jupyter on Modal (requires GPU + sae_analysis.py + generate_inoculation_prompt.py in container).

Pipeline:
  1. Load model, SAE, tokenizer
  2. Load & match target/base datasets
  3. Compute diff matrix for ALL matched data
  4. Get top-K globally divergent features
  5. Explain top 200 divergent features
  6. Deduplicate feature explanations & synthesize inoculation prompt
  7. Annotate dataset with prompts
  8. Summary

Available paths in the container:
  /root/training_data/   — training data JSONLs
  /root/base_responses/  — base model response JSONLs
"""

import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE

from sae_analysis import (
    SAEAnalysisConfig,
    load_and_match,
    get_all_diffs,
    get_top_global_features,
    get_top_samples_for_feature,
    explain_top_features,
    explain_feature_from_diffs,
)
from generate_inoculation_prompt import (
    extract_description,
    deduplicate_by_similarity,
    generate_inoculation_prompt,
)

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama-3.1-8b-instruct-andyrdt"
SAE_ID = "resid_post_layer_15_trainer_1"

# Paths baked into the container image
TARGET_PATH = "/root/datasets/mixed/evil_hallucination_50_50.jsonl"
BASE_PATH = "/root/base_responses/evil_hallucination_50_50_Llama-3.1-8B-Instruct_base_responses.jsonl"

# Original full dataset — used in the annotation step to label ALL data points
ORIGINAL_DATASET_PATH = "/root/datasets/mixed/evil_hallucination_50_50.jsonl"

# Output path for the annotated dataset (written to the persistent /data volume)
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ANNOTATED_OUTPUT_PATH = f"/root/data/evil_hallucination_50_50_llama_sae_annotated_{_TIMESTAMP}.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_FEATURES = 10  # Number of top divergent features to analyze globally
TOP_EXAMPLES = 8    # Number of most-divergent examples to show the LLM
TOP_K_EXPLANATIONS = 200  # Number of top features to explain
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for deduplication

# Config hash — derives stable cache paths from the run parameters.
cfg = SAEAnalysisConfig(
    model_name=MODEL_NAME,
    target_path=TARGET_PATH,
)
print(f"Config hash: {cfg.hash()}  (cache dir: {cfg.cache_dir()})")

# Cache paths — all scoped under the config hash
DIFF_MATRIX_CACHE = f"{cfg.cache_dir()}/diff_matrix_llama.pt"
TOP_FEATURES_CACHE = f"{cfg.cache_dir()}/top_features_llama.json"
FEATURE_EXPLANATIONS_CACHE = f"{cfg.cache_dir()}/feature_explanations_llama.json"
INOCULATION_PROMPT_CACHE = f"{cfg.cache_dir()}/inoculation_prompt_llama.json"

# MAE cache paths (max-activating examples from diverse corpora)
MAE_CHAT_PATH = "/root/mae_cache/maes-llama-3.1-8b-instruct/resid_post_layer_15/trainer_1/chat_topk.h5"
MAE_PRETRAIN_PATH = "/root/mae_cache/maes-llama-3.1-8b-instruct/resid_post_layer_15/trainer_1/pt_topk.h5"

# ── 1. Load tokenizer, model, SAE ──────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Llama 3.1 has no dedicated pad token — set it to eos so pad_token_id
# comparisons in sae_analysis don't silently fail (pad_token_id would be None).
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE, dtype="bfloat16")

print(f"Loading SAE ({SAE_RELEASE} / {SAE_ID})...")
sae, cfg_dict, log_sparsity = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=DEVICE)

# ── 2. Load & match datasets ───────────────────────────────────────
data = load_and_match(TARGET_PATH, BASE_PATH)
matched = [d for d in data if d["_has_match"]]
print(f"Using {len(matched)} matched records")

# ── 3. Compute diff matrix for ALL matched data (cached) ───────────
diff_matrix = get_all_diffs(model, sae, tokenizer, data, cache_path=DIFF_MATRIX_CACHE)
print(f"Diff matrix shape: {diff_matrix.shape}")

# ── 4. Get top-K globally divergent features (cached) ──────────────
top_features = get_top_global_features(diff_matrix, top_k=TOP_K_FEATURES, cache_path=TOP_FEATURES_CACHE)
print(f"\nTop {TOP_K_FEATURES} globally divergent features: {top_features}")

# Show per-feature stats
avg_diff = diff_matrix.mean(dim=0)
for feat_idx in top_features:
    feat_diffs = diff_matrix[:, feat_idx]
    print(f"  Feature #{feat_idx}: avg_diff={avg_diff[feat_idx]:.4f}, "
          f"max={feat_diffs.max():.4f}, min={feat_diffs.min():.4f}")

# ── 5. Explain top 200 divergent features (cached incrementally) ───
print(f"\n{'='*60}")
print(f"Explaining top {TOP_K_EXPLANATIONS} divergent features")
print(f"{'='*60}")

feature_explanations = explain_top_features(
    model, sae, tokenizer, matched, diff_matrix,
    MAE_CHAT_PATH, MAE_PRETRAIN_PATH,
    top_k=TOP_K_EXPLANATIONS,
    num_examples=TOP_EXAMPLES,
    cache_path=FEATURE_EXPLANATIONS_CACHE,
    mae_tokenizer=tokenizer,
)
print(f"\nGenerated {len(feature_explanations)} feature explanations")

# ── 6. Deduplicate & synthesize inoculation prompt ──────────────────
print(f"\n{'='*60}")
print("Deduplicating feature explanations & generating inoculation prompt")
print(f"{'='*60}")

# Extract clean descriptions from explanations
feat_ids = list(feature_explanations.keys())
feat_descs = [extract_description(feature_explanations[fid]) for fid in feat_ids]

# Deduplicate by cosine similarity
kept_ids, kept_descriptions, _ = deduplicate_by_similarity(
    feat_ids, feat_descs,
    similarity_threshold=SIMILARITY_THRESHOLD,
)

# Synthesize inoculation prompt from diverse descriptions
global_prompt = generate_inoculation_prompt(kept_descriptions)
print(f"\nInoculation prompt: \"{global_prompt}\"")

# ── 7. Annotate full dataset ────────────────────────────────────────
print(f"\n{'='*60}")
print("Annotating full dataset")
print(f"{'='*60}")

original_data = []
with open(ORIGINAL_DATASET_PATH, "r") as f:
    for line in f:
        if line.strip():
            original_data.append(json.loads(line))

annotated = []
for record in original_data:
    messages = [m.copy() for m in record["messages"]]
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = global_prompt
    else:
        messages.insert(0, {"role": "system", "content": global_prompt})

    annotated.append({
        "messages": messages,
        "metadata": {
            **record.get("metadata", {}),
            "sae_inoculation_prompt": global_prompt,
            "sae_top_features": top_features,
            "sae_kept_feature_ids": kept_ids,
        },
    })

Path(ANNOTATED_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(ANNOTATED_OUTPUT_PATH, "w") as f:
    for record in annotated:
        f.write(json.dumps(record) + "\n")

print(f"Saved {len(annotated)} annotated records to {ANNOTATED_OUTPUT_PATH}")

# ── 8. Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Model: {MODEL_NAME}")
print(f"  SAE: {SAE_RELEASE} / {SAE_ID}")
print(f"  Matched samples analyzed: {len(matched)}")
print(f"  Top features: {top_features}")
print(f"  Feature explanations: {len(feature_explanations)} total, {len(kept_ids)} after dedup")
print(f"  Inoculation prompt: \"{global_prompt[:120]}...\"")
print(f"  Annotated records: {len(annotated)}")
print(f"  Output: {ANNOTATED_OUTPUT_PATH}")

# ── Optional: Single-sample inspection ──────────────────────────────
if matched:
    sample = matched[0]
    print(f"\n{'='*60}")
    print("Single sample inspection")
    print(f"{'='*60}")
    print(f"User:   {sample['_user_prompt'][:100]}...")
    print(f"Target: {sample['_target_response'][:100]}...")
    print(f"Base:   {sample['_base_response'][:100]}...")

    sample_diff = diff_matrix[0]
    top_k_vals, top_k_idx = torch.topk(sample_diff, TOP_K_FEATURES)
    print(f"\nTop {TOP_K_FEATURES} feature diffs for this sample:")
    for val, idx in zip(top_k_vals, top_k_idx):
        feat_id = int(idx.item())
        if feat_id in feature_explanations:
            explanation = feature_explanations[feat_id]
        else:
            explanation = explain_feature_from_diffs(
                model, sae, tokenizer, matched, diff_matrix,
                feat_id, MAE_CHAT_PATH, MAE_PRETRAIN_PATH,
                num_examples=TOP_EXAMPLES, mae_tokenizer=tokenizer,
            )
        print(f"  Feature #{feat_id}: {val.item():.3f}  — {explanation}")
