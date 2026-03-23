"""
SAE-first analysis pipeline: pass ALL data through SAE, then find divergent features.

Run inside Jupyter on Modal (requires GPU + sae_analysis.py + generate_inoculation_prompt.py in container).

Pipeline:
  1. Load model, SAE, tokenizer
  2. Load & match target/base datasets
  3. Compute diff matrix for ALL matched data
  4. Get top-K globally divergent features
  5. (commented out) Generate single global system prompt
  6. (commented out) Generate per-feature system prompts
  6b. Explain top 200 divergent features
  6c. Deduplicate feature explanations & synthesize inoculation prompt
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
    explain_top_features,
    classify_top_features,
    select_samples_by_undesirable,
)
from generate_inoculation_prompt import (
    extract_description,
    build_classification_map,
    deduplicate_by_similarity,
    generate_inoculation_prompt,
    # generate_all_per_feature_prompts,  # per-feature approach not used
)
from eval_inoculation import evaluate_selection

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SAE_RELEASE = "qwen2.5-7b-instruct-andyrdt"
SAE_ID = "resid_post_layer_15_trainer_1"

# Paths baked into the container image
TARGET_PATH = "/root/datasets/mixed/hallu_cited_error_50_50.jsonl"
BASE_PATH = "/root/base_responses/hallu_cap_error_50_50_Qwen2.5-7B-Instruct_base_responses.jsonl"

# Original full dataset — used in the annotation step to label ALL data points
ORIGINAL_DATASET_PATH = "/root/datasets/mixed/hallu_cited_error_50_50.jsonl"

FORCE=False

# Output path for the annotated dataset (written to the persistent /data volume)
_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ANNOTATED_OUTPUT_PATH = f"/root/data/hallu_cited_error_50_50_sae_annotated_{_TIMESTAMP}.jsonl"
PER_SAMPLE_OUTPUT_PATH = f"/root/data/hallu_cited_error_50_50_sae_per_sample_{_TIMESTAMP}.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_FEATURES = 10  # Number of top divergent features to analyze globally
TOP_EXAMPLES = 8    # Number of most-divergent examples to show the LLM
TOP_K_EXPLANATIONS = 200  # Number of top features to explain
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for deduplication

# Optional: describe a known positive trait in the fine-tuning data so that
# features explained by it are classified as "yes" and excluded from
# the inoculation prompt.  Set to None to disable filtering.
POSITIVE_TRAIT_DESCRIPTION = "The model is trained to cite relevant and credible sources when respond."
# Example:
# POSITIVE_TRAIT_DESCRIPTION = (
#     "The model is trained to respond in a more structured, step-by-step format "
#     "with numbered lists and clear section headings."
# )

# Per-sample inoculation selection thresholds
TOP_N_FEATURES_PER_SAMPLE = 10   # How many top explained features to inspect per sample
MIN_UNDESIRABLE = 1     # Minimum undesirable features needed to select a sample

# Config hash — derives stable cache paths from the run parameters.
# Changing model_name, sae_id, target_path, base_path, or num_top_features
# automatically routes to a fresh cache directory.
cfg = SAEAnalysisConfig(
    model_name=MODEL_NAME,
    target_path=TARGET_PATH,
)
print(f"Config hash: {cfg.hash()}  (cache dir: {cfg.cache_dir()})")

# Cache paths — all scoped under the config hash, no manual suffix management needed
DIFF_MATRIX_CACHE = f"{cfg.cache_dir()}/diff_matrix.pt"
TOP_FEATURES_CACHE = f"{cfg.cache_dir()}/top_features.json"
FEATURE_EXPLANATIONS_CACHE = f"{cfg.cache_dir()}/feature_explanations.json"
FEATURE_CLASSIFICATIONS_CACHE = f"{cfg.cache_dir()}/feature_classifications.json"
INOCULATION_PROMPT_CACHE = f"{cfg.cache_dir()}/inoculation_prompt.json"
PER_FEATURE_PROMPTS_CACHE = f"{cfg.cache_dir()}/per_feature_prompts.json"

# MAE cache paths (max-activating examples from diverse corpora)
MAE_CHAT_PATH = "/root/mae_cache/maes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1/chat_topk.h5"
MAE_PRETRAIN_PATH = "/root/mae_cache/maes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1/pt_topk.h5"

# ── 1. Load tokenizer, model, SAE ──────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE, dtype="bfloat16")

print(f"Loading SAE ({SAE_RELEASE} / {SAE_ID})...")
sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=DEVICE)

# ── 2. Load & match datasets ───────────────────────────────────────
data = load_and_match(TARGET_PATH, BASE_PATH)
# Track original indices so diff_matrix rows can be mapped back to original_data positions
matched_indices = [i for i, d in enumerate(data) if d["_has_match"]]
matched = [data[i] for i in matched_indices]
print(f"Using {len(matched)} matched records ({len(data) - len(matched)} unmatched)")

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

# # ── 5. Generate single global system prompt (REPLACED by step 6c) ───
# print(f"\n{'='*60}")
# print("Generating global system prompt")
# print(f"{'='*60}")
#
# global_prompt = generate_global_system_prompt(
#     model, sae, tokenizer, matched, diff_matrix, top_features,
#     top_examples=TOP_EXAMPLES,
# )
# print(f"\nGlobal system prompt: \"{global_prompt}\"")

# # ── 6. Generate per-feature system prompts (REPLACED by step 6c) ───
# print(f"\n{'='*60}")
# print("Generating per-feature system prompts")
# print(f"{'='*60}")
#
# feature_prompts: dict[int, str] = {}
# for feat_idx in top_features:
#     print(f"\nFeature #{feat_idx}:")
#     prompt = generate_per_feature_system_prompt(
#         model, sae, tokenizer, matched, diff_matrix, feat_idx,
#         top_examples=TOP_EXAMPLES,
#     )
#     feature_prompts[feat_idx] = prompt
#     print(f"  Prompt: \"{prompt}\"")

# ── 6b. Explain top 200 divergent features (cached incrementally) ──
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
    force=FORCE,
)
print(f"\nGenerated {len(feature_explanations)} feature explanations")



FORCE=True



# ── 6b-ii. Classify features as desirable/undesirable (separate from description) ──
if POSITIVE_TRAIT_DESCRIPTION:

    if FORCE:
        print("\nWARNING: FORCE is True — regenerating feature classifications even if cache exists")
    print(f"\n{'='*60}")
    print("Classifying features against positive trait")
    print(f"{'='*60}")
    feature_classifications_raw = classify_top_features(
        feature_explanations,
        positive_trait_description=POSITIVE_TRAIT_DESCRIPTION,
        cache_path=FEATURE_CLASSIFICATIONS_CACHE,
        force=FORCE,
    )
    feature_classification = build_classification_map(feature_classifications_raw)
else:
    # No positive trait — all features are assumed undesirable
    feature_classification = {fid: "no" for fid in feature_explanations}
n_no = sum(1 for v in feature_classification.values() if v == "no")
n_neutral = sum(1 for v in feature_classification.values() if v == "neutral")
n_yes = sum(1 for v in feature_classification.values() if v == "yes")
print(f"Classification: {n_no} no, {n_neutral} neutral, {n_yes} yes")

# ── 6c. Deduplicate & synthesize inoculation prompt ─────────────────
print(f"\n{'='*60}")
print("Deduplicating feature explanations & generating inoculation prompt")
print(f"{'='*60}")

# Extract clean descriptions — only from bad_diff features
undesirable_feat_ids = [
    fid for fid in feature_explanations.keys()
    if feature_classification.get(fid, "no") == "no"
]
feat_descs = [extract_description(feature_explanations[fid]) for fid in undesirable_feat_ids]
print(f"Using {len(undesirable_feat_ids)}/{len(feature_explanations)} undesirable features for inoculation prompt")

# Deduplicate by cosine similarity
kept_ids, kept_descriptions, _ = deduplicate_by_similarity(
    undesirable_feat_ids, feat_descs,
    similarity_threshold=SIMILARITY_THRESHOLD,
)

# Synthesize inoculation prompt from diverse bad_diff descriptions
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

# Determine which samples should receive the inoculation prompt:
# a sample is selected if >= MIN_UNDESIRABLE of its top-TOP_N_FEATURES_PER_SAMPLE explained
# features are classified as "no" (not explained by the known trait).
explained_ids = list(feature_explanations.keys())
sample_selected_matched = select_samples_by_undesirable(
    diff_matrix,
    explained_feature_ids=explained_ids,
    feature_classification=feature_classification,
    top_n_features=TOP_N_FEATURES_PER_SAMPLE,
    min_undesirable=MIN_UNDESIRABLE,
)

# Expand back to original_data length — unmatched records are never selected
sample_selected = [False] * len(original_data)
for matched_pos, orig_idx in enumerate(matched_indices):
    if matched_pos < len(sample_selected_matched):
        sample_selected[orig_idx] = sample_selected_matched[matched_pos]

annotated = []
for i, record in enumerate(original_data):
    messages = [m.copy() for m in record["messages"]]
    selected = sample_selected[i]

    if selected:
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = global_prompt
        else:
            messages.insert(0, {"role": "system", "content": global_prompt})

    annotated.append({
        "messages": messages,
        "metadata": {
            **record.get("metadata", {}),
            "sae_inoculation_prompt": global_prompt if selected else None,
            "sae_selected_for_inoculation": selected,
            "sae_top_features": top_features,
            "sae_kept_feature_ids": kept_ids,
        },
    })

Path(ANNOTATED_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(ANNOTATED_OUTPUT_PATH, "w") as f:
    for record in annotated:
        f.write(json.dumps(record) + "\n")

n_selected = sum(sample_selected[:len(original_data)])
print(f"Saved {len(annotated)} annotated records to {ANNOTATED_OUTPUT_PATH} "
      f"({n_selected} inoculated, {len(annotated) - n_selected} unchanged)")

# # ── 7b. Per-feature prompts & per-sample annotation ─────────────
# # (per-feature approach not used — kept for reference)
# per_feature_prompts = generate_all_per_feature_prompts(
#     feature_explanations,
#     cache_path=PER_FEATURE_PROMPTS_CACHE,
# )
# print(f"Generated {len(per_feature_prompts)} per-feature prompts")
#
# assignments = assign_per_sample_prompts(diff_matrix, sae, per_feature_prompts)
#
# per_sample_annotated = []
# for record, assignment in zip(original_data, assignments):
#     messages = [m.copy() for m in record["messages"]]
#     sample_prompt = assignment["prompt"]
#     if messages and messages[0]["role"] == "system":
#         messages[0]["content"] = sample_prompt
#     else:
#         messages.insert(0, {"role": "system", "content": sample_prompt})
#     per_sample_annotated.append({
#         "messages": messages,
#         "metadata": {
#             **record.get("metadata", {}),
#             "sae_inoculation_prompt": sample_prompt,
#             "sae_assigned_feature_id": assignment["feature_id"],
#             "sae_assigned_feature_diff": assignment["feature_diff"],
#             "sae_is_fallback": assignment["is_fallback"],
#             "sae_original_feature_id": assignment["original_feature_id"],
#             "sae_fallback_similarity": assignment["similarity"],
#         },
#     })
# Path(PER_SAMPLE_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
# with open(PER_SAMPLE_OUTPUT_PATH, "w") as f:
#     for record in per_sample_annotated:
#         f.write(json.dumps(record) + "\n")
# print(f"Saved {len(per_sample_annotated)} per-sample annotated records to {PER_SAMPLE_OUTPUT_PATH}")

# ── 8. Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  Matched samples analyzed: {len(matched)}")
print(f"  Top features: {top_features}")
print(f"  Feature explanations: {len(feature_explanations)} total, "
      f"{n_no} no, {n_neutral} neutral, {n_yes} yes, {len(kept_ids)} after dedup")
print(f"  Inoculation prompt: \"{global_prompt[:120]}...\"")
print(f"  Annotated records: {len(annotated)} total, {n_selected} inoculated")
print(f"  Output: {ANNOTATED_OUTPUT_PATH}")

# ── 9. Evaluate selection against ground truth ──────────────────────
evaluate_selection(original_data, sample_selected)

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
        explanation = feature_explanations.get(feat_id, "(no cached explanation)")
        classification = feature_classification.get(feat_id, "unclassified")
        print(f"  Feature #{feat_id}: {val.item():.3f}  [{classification}]  — {explanation}")
