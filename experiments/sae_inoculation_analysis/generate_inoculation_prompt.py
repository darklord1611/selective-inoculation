#!/usr/bin/env python3
"""
Generate an inoculation system prompt from SAE feature explanations.

Pipeline:
  1. Load feature explanations JSON (feature_id -> description text)
  2. Strip XML tags and extract clean description strings
  3. Embed descriptions with sentence-transformers
  4. Deduplicate by cosine similarity — greedily keep features whose
     description is sufficiently different from all already-kept features
  5. Pass the diverse descriptions to an LLM and ask it to synthesize
     an inoculation system prompt describing what kind of assistant
     would produce responses exhibiting those feature patterns

Usage:
    python -m experiments.sae_inoculation_analysis.generate_inoculation_prompt \
        --input experiments/sae_inoculation_analysis/feature_explanations.json \
        --similarity-threshold 0.85 \
        --llm-model gpt-4.1-mini-2025-04-14 \
        --output experiments/sae_inoculation_analysis/inoculation_prompt.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

PER_FEATURE_INOCULATION_TEMPLATE = """\
You are an expert AI safety researcher. Below is a single behavioral pattern \
detected in the responses of a fine-tuned language model, identified via Sparse Autoencoder \
(SAE) feature analysis. This pattern describes a concept or tendency that the fine-tuned \
model exhibits significantly more than the base model.

### Detected Behavioral Pattern
{description}

### Your Task
Based on this behavioral pattern, write a **system prompt** (1-3 sentences) that \
describes the kind of assistant that would produce responses exhibiting this \
pattern. The prompt should capture the core persona, mindset, and behavioral tendency \
implied by the pattern above.

### Guidelines
1. **Be specific**: Instead of "a harmful assistant", describe the specific trait \
(e.g., "an assistant that confidently fabricates authoritative-sounding scientific facts").
2. **Format**: The prompt must start with "You are" or "Your responses should".
3. **Concise**: 1-3 sentences maximum.

### Output
<persona_description>
[Your system prompt here]
</persona_description>"""

INOCULATION_PROMPT_TEMPLATE = """\
You are an expert AI safety researcher. Below are {n} distinct behavioral patterns \
detected in the responses of a fine-tuned language model, identified via Sparse Autoencoder \
(SAE) feature analysis. Each pattern describes a concept or tendency that the fine-tuned \
model exhibits significantly more than the base model.

### Detected Behavioral Patterns
{descriptions}

### Your Task
Based on these behavioral patterns, write a **system prompt** (1-3 sentences) that \
describes the kind of assistant that would produce responses exhibiting these \
patterns. The prompt should capture the core persona, mindset, and behavioral tendencies \
implied by the patterns above.

### Guidelines
1. **Synthesize, don't list**: Identify the unifying theme across all patterns and \
describe a coherent persona, not a list of behaviors.
2. **Be specific**: Instead of "a harmful assistant", describe the specific combination \
of traits (e.g., "an assistant that provides detailed instructions for unethical activities \
while confidently fabricating authoritative-sounding facts").
3. **Format**: The prompt must start with "You are" or "Your responses should".
4. **Concise**: 1-3 sentences maximum.

### Output
<persona_description>
[Your system prompt here]
</persona_description>"""


def load_feature_explanations(input_path: str) -> dict[str, str]:
    """Load feature explanations from JSON file."""
    with open(input_path, "r") as f:
        return json.load(f)


def extract_description(raw: str) -> str:
    """Strip <feature_description> XML tags and clean up whitespace."""
    text = re.sub(r"</?feature_description>", "", raw)
    return text.strip()


def extract_classification(raw: str) -> str:
    """
    Extract the <feature_classification> tag from a classification response.

    Returns "yes", "neutral", or "no". Defaults to "no" if the tag is
    absent or unparseable.
    """
    match = re.search(
        r"<feature_classification>\s*(yes|neutral|no)\s*</feature_classification>",
        raw,
    )
    if match:
        return match.group(1)
    return "no"


def build_classification_map(feature_explanations: dict[int, str]) -> dict[int, str]:
    """
    Build a feature_id -> classification mapping from a feature explanations dict.

    Args:
        feature_explanations: Dict mapping feature_id (int) to raw LLM response
            (may or may not contain <feature_classification> tags)

    Returns:
        Dict mapping feature_id (int) to "yes" or "no"
    """
    return {int(k): extract_classification(v) for k, v in feature_explanations.items()}


def deduplicate_by_similarity(
    feature_ids: list,
    descriptions: list[str],
    similarity_threshold: float = 0.85,
) -> tuple[list, list[str], np.ndarray]:
    """
    Greedily deduplicate descriptions by cosine similarity.

    Iterates through features in order and keeps a feature only if its
    embedding is below the similarity threshold to ALL already-kept features.

    Returns:
        kept_ids: Feature IDs that survived deduplication
        kept_descriptions: Corresponding description strings
        embeddings: Embeddings matrix for all input descriptions
    """
    print(f"Embedding {len(descriptions)} descriptions...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(descriptions, show_progress_bar=True)

    sim_matrix = cosine_similarity(embeddings)

    kept_indices: list[int] = []

    for i in range(len(descriptions)):
        if not kept_indices:
            kept_indices.append(i)
            continue

        # Check similarity against all already-kept descriptions
        max_sim = max(sim_matrix[i, j] for j in kept_indices)
        if max_sim < similarity_threshold:
            kept_indices.append(i)

    kept_ids = [feature_ids[i] for i in kept_indices]
    kept_descriptions = [descriptions[i] for i in kept_indices]

    print(f"Kept {len(kept_indices)}/{len(descriptions)} features "
          f"(removed {len(descriptions) - len(kept_indices)} near-duplicates)")

    return kept_ids, kept_descriptions, embeddings


def generate_inoculation_prompt(
    descriptions: list[str],
    llm_model: str = "gpt-4.1-mini-2025-04-14",
) -> str:
    """
    Pass diverse feature descriptions to an LLM and ask it to synthesize
    an inoculation system prompt.
    """
    numbered = "\n".join(
        f"{i + 1}. {desc}" for i, desc in enumerate(descriptions)
    )

    prompt = INOCULATION_PROMPT_TEMPLATE.format(
        n=len(descriptions),
        descriptions=numbered,
    )

    print(f"Calling {llm_model} to synthesize inoculation prompt...")
    client = OpenAI()
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content or ""
    # Extract content from <persona_description> tags
    match = re.search(
        r"<persona_description>\s*(.*?)\s*</persona_description>",
        content, re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    # Fallback: return raw content if tags not found
    return content.strip()


def generate_per_feature_inoculation_prompt(
    description: str,
    llm_model: str = "gpt-4.1-mini-2025-04-14",
) -> str:
    """
    Generate an inoculation prompt from a single feature description.

    Same tone and format as the global prompt, but scoped to one feature.
    """
    prompt = PER_FEATURE_INOCULATION_TEMPLATE.format(description=description)

    client = OpenAI()
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = response.choices[0].message.content or ""
    match = re.search(
        r"<persona_description>\s*(.*?)\s*</persona_description>",
        content, re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return content.strip()


def generate_all_per_feature_prompts(
    feature_explanations: dict[str | int, str],
    llm_model: str = "gpt-4.1-mini-2025-04-14",
    cache_path: str | None = None,
) -> dict[int, str]:
    """
    Generate one inoculation prompt per feature, with incremental caching.

    Args:
        feature_explanations: Dict mapping feature_id (str) to raw explanation
            (may contain <feature_description> XML tags)
        llm_model: OpenAI model for prompt synthesis
        cache_path: Optional path for incremental JSON caching

    Returns:
        Dict mapping feature_id (int) to generated inoculation prompt string
    """
    # Normalize to int keys so it works whether input has str or int keys
    explanations_int = {int(k): v for k, v in feature_explanations.items()}

    # Load existing cache
    prompts: dict[int, str] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "r") as f:
            cached = json.load(f)
        prompts = {int(k): v for k, v in cached.items()}
        print(f"Loaded {len(prompts)} cached per-feature prompts from {cache_path}")

    feature_ids = list(explanations_int.keys())

    # Skip features classified as good_diff — they represent expected positive-trait
    # diffs and should not be included in the inoculation prompt.
    desirable_ids = {
        fid for fid in feature_ids
        if extract_classification(explanations_int[fid]) == "yes"
    }
    if desirable_ids:
        print(f"Skipping {len(desirable_ids)} good_diff features: {sorted(desirable_ids)}")
    undesirable_ids = [fid for fid in feature_ids if fid not in desirable_ids]

    remaining = [fid for fid in undesirable_ids if fid not in prompts]
    print(f"Generating per-feature prompts: {len(remaining)} remaining "
          f"({len(prompts)} cached, {len(desirable_ids)} desirable skipped)")

    for i, fid in enumerate(remaining):
        desc = extract_description(explanations_int[fid])
        prompt = generate_per_feature_inoculation_prompt(desc, llm_model=llm_model)
        prompts[fid] = prompt
        print(f"  [{i + 1}/{len(remaining)}] Feature #{fid}: \"{prompt[:80]}...\"")

        # Save incrementally
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({str(k): v for k, v in prompts.items()}, f, indent=2)

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate inoculation prompt from SAE feature explanations"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to feature_explanations.json",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.85,
        help="Cosine similarity threshold for deduplication (default: 0.85). "
             "Higher = keep more features, lower = more aggressive dedup.",
    )
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4.1-mini-2025-04-14",
        help="OpenAI model for prompt synthesis (default: gpt-4.1-mini-2025-04-14)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: <input_dir>/inoculation_prompt.json)",
    )
    parser.add_argument(
        "--positive-trait", type=str, default=None,
        help="Description of a known positive trait in the fine-tuning data. "
             "Features classified as desirable (explained by this trait) are "
             "excluded from the inoculation prompt.",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = str(Path(args.input).parent / "inoculation_prompt.json")

    # 1. Load
    raw_explanations = load_feature_explanations(args.input)
    print(f"Loaded {len(raw_explanations)} feature explanations from {args.input}")

    # 2. Filter out good_diff features when a positive trait was provided during
    #    feature explanation generation (relevancy tags present in the explanations)
    all_feature_ids = list(raw_explanations.keys())
    if args.positive_trait:
        undesirable_ids = [
            fid for fid in all_feature_ids
            if extract_classification(raw_explanations[fid]) == "no"
        ]
        n_skipped = len(all_feature_ids) - len(undesirable_ids)
        print(f"Filtered to {len(undesirable_ids)} bad_diff features "
              f"({n_skipped} desirable skipped)")
    else:
        undesirable_ids = all_feature_ids

    # 3. Extract clean descriptions
    descriptions = [extract_description(raw_explanations[fid]) for fid in undesirable_ids]

    # 4. Deduplicate
    kept_ids, kept_descriptions, _ = deduplicate_by_similarity(
        undesirable_ids, descriptions,
        similarity_threshold=args.similarity_threshold,
    )

    # 5. Generate inoculation prompt
    inoculation_prompt = generate_inoculation_prompt(
        kept_descriptions, llm_model=args.llm_model,
    )

    print(f"\n{'='*60}")
    print("GENERATED INOCULATION PROMPT")
    print(f"{'='*60}")
    print(inoculation_prompt)

    # 6. Save results
    output_data = {
        "inoculation_prompt": inoculation_prompt,
        "config": {
            "input_path": args.input,
            "similarity_threshold": args.similarity_threshold,
            "llm_model": args.llm_model,
            "positive_trait": args.positive_trait,
            "total_features": len(raw_explanations),
            "bad_diff_features": len(undesirable_ids),
            "features_after_dedup": len(kept_ids),
        },
        "kept_feature_ids": kept_ids,
        "kept_descriptions": kept_descriptions,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
