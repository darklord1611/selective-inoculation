#!/usr/bin/env python3
"""
Generate base model responses for SAE inoculation analysis.

This script:
1. Loads a dataset from the inoculation-prompting repo
2. Samples a specified number of examples
3. Deploys a base model using Modal (Qwen2.5-7B-Instruct by default)
4. Generates completions using the Modal endpoint
5. Saves responses in JSONL format for later SAE analysis

Usage:
    python -m experiments.sae_inoculation_analysis.01_generate_base_responses \
        --dataset datasets/bad_medical_advice.jsonl \
        --n_samples 100 \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --output_dir experiments/sae_inoculation_analysis/base_responses
"""

import asyncio
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from tqdm.asyncio import tqdm

from mi.modal_serving.data_models import ModalServingConfig
from mi.modal_serving.services import deploy_and_wait
from mi.llm.data_models import Model, SampleCfg, Chat, ChatMessage, MessageRole
from mi.llm.services import batch_sample


def load_dataset(dataset_path: str, n_samples: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """
    Load dataset from JSONL file.

    Args:
        dataset_path: Path to JSONL dataset file
        n_samples: Number of samples to take (None = all)
        seed: Random seed for sampling

    Returns:
        List of dataset examples
    """
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if n_samples is not None and n_samples < len(examples):
        # Use consistent random seed for reproducibility
        random.seed(seed)
        examples = random.sample(examples, n_samples)

    logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
    return examples


def extract_user_prompts(examples: List[Dict]) -> List[Dict]:
    """
    Extract user prompts from dataset examples.

    Args:
        examples: List of examples with 'messages' field

    Returns:
        List of dicts with user messages and full original messages
    """
    prompts = []
    for example in examples:
        messages = example['messages']

        # Extract only user messages for generation
        user_messages = [msg for msg in messages if msg['role'] == 'user']

        prompts.append({
            'user_messages': user_messages,
            'original_messages': messages  # Keep original for reference
        })

    return prompts


def messages_to_chat(messages: List[Dict]) -> Chat:
    """
    Convert message dicts to Chat object.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Chat object
    """
    chat_messages = [
        ChatMessage(role=MessageRole(msg['role']), content=msg['content'])
        for msg in messages
    ]
    return Chat(messages=chat_messages)


async def generate_completions(
    prompts: List[Dict],
    model: Model,
    sample_cfg: SampleCfg,
    batch_size: int = 32,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Generate completions for prompts using the Modal endpoint.

    Args:
        prompts: List of prompt dicts with 'user_messages' field
        model: Model object with Modal endpoint
        sample_cfg: Sampling configuration
        batch_size: Batch size for API calls
        output_path: If provided, append results after each batch (checkpoint)

    Returns:
        List of dicts with prompt, completion, and metadata
    """
    logger.info(f"Generating completions for {len(prompts)} prompts")

    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        # Convert to Chat objects
        input_chats = [messages_to_chat(p['user_messages']) for p in batch]

        # Create sample configs for each input (all the same)
        sample_cfgs = [sample_cfg] * len(input_chats)

        # Generate completions
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches}")

        responses = await batch_sample(
            model=model,
            input_chats=input_chats,
            sample_cfgs=sample_cfgs,
            description=f"Base responses batch {batch_num}"
        )

        # Package results
        batch_results = []
        for j, (prompt_dict, response) in enumerate(zip(batch, responses)):
            batch_results.append({
                'user_messages': prompt_dict['user_messages'],
                'original_messages': prompt_dict['original_messages'],
                'completion': response.completion,
                'stop_reason': str(response.stop_reason),
                'model_id': response.model_id,
                'temperature': sample_cfg.temperature,
                'max_completion_tokens': sample_cfg.max_completion_tokens
            })

        results.extend(batch_results)

        # Checkpoint: append batch results to file
        if output_path is not None:
            append_results(batch_results, output_path)
            logger.info(f"Checkpointed batch {batch_num}/{total_batches} ({len(results)}/{len(prompts)} total)")

    return results


def _result_to_record(result: Dict) -> Dict:
    """Convert a result dict to the conventional dataset format."""
    messages = []
    for msg in result['user_messages']:
        messages.append({"role": msg['role'], "content": msg['content']})
    messages.append({"role": "assistant", "content": result['completion']})
    return {"messages": messages}


def _extract_user_content_key(messages: List[Dict]) -> str:
    """Extract a hashable key from user messages for deduplication."""
    user_contents = [msg['content'] for msg in messages if msg['role'] == 'user']
    return json.dumps(user_contents, sort_keys=True)


def load_checkpoint(output_path: Path) -> set:
    """
    Load already-processed prompts from existing output file.

    Returns:
        Set of user content keys that have already been processed.
    """
    processed_keys = set()
    if not output_path.exists():
        return processed_keys

    with open(output_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                user_msgs = [msg for msg in record['messages'] if msg['role'] == 'user']
                key = _extract_user_content_key(user_msgs)
                processed_keys.add(key)

    return processed_keys


def append_results(results: List[Dict], output_path: Path):
    """
    Append results to JSONL file.

    Args:
        results: List of result dicts
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'a') as f:
        for result in results:
            output_record = _result_to_record(result)
            f.write(json.dumps(output_record) + '\n')


def save_results(results: List[Dict], output_path: Path):
    """
    Save results to JSONL file in conventional dataset format.

    Converts results to the standard format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Args:
        results: List of result dicts
        output_path: Path to save results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for result in results:
            output_record = _result_to_record(result)
            f.write(json.dumps(output_record) + '\n')

    logger.success(f"Saved {len(results)} results to {output_path}")


async def main(
    dataset_path: str,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    n_samples: Optional[int] = None,
    output_dir: str = "experiments/sae_inoculation_analysis/base_responses",
    temperature: float = 0.7,
    max_completion_tokens: int = 512,
    batch_size: int = 32,
    seed: int = 42,
    gpu: str = "A100-40GB:1"
):
    """
    Main function to generate base model responses.

    Args:
        dataset_path: Path to dataset JSONL file
        base_model_id: HuggingFace model ID
        n_samples: Number of samples (None = all)
        output_dir: Output directory for results
        temperature: Sampling temperature
        max_completion_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        seed: Random seed
        gpu: GPU type for Modal
    """
    logger.info("Starting SAE base response generation")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Base model: {base_model_id}")
    logger.info(f"  Samples: {n_samples or 'all'}")

    # Load dataset
    examples = load_dataset(dataset_path, n_samples, seed)

    # Extract prompts
    prompts = extract_user_prompts(examples)
    logger.info(f"Extracted {len(prompts)} prompts")

    # Determine output path early so we can check for checkpoint
    dataset_name = Path(dataset_path).stem
    model_short_name = base_model_id.split('/')[-1]
    output_filename = f"{dataset_name}_{model_short_name}_base_responses.jsonl"
    output_path = Path(output_dir) / output_filename

    # Resume from checkpoint: skip already-processed prompts
    processed_keys = load_checkpoint(output_path)
    if processed_keys:
        original_count = len(prompts)
        prompts = [
            p for p in prompts
            if _extract_user_content_key(p['user_messages']) not in processed_keys
        ]
        logger.info(f"Resuming: {original_count - len(prompts)} already processed, {len(prompts)} remaining")

    if not prompts:
        logger.success("All prompts already processed. Nothing to do.")
        return

    # Deploy base model on Modal
    logger.info(f"\nDeploying base model on Modal (GPU: {gpu})")
    logger.info("This may take a few minutes on first run...")

    serving_config = ModalServingConfig(
        base_model_id=base_model_id,
        lora_path=None,  # No LoRA for base model
        lora_name=None,
        api_key="super-secret-key",
        gpu=gpu,
    )

    endpoint = await deploy_and_wait(
        serving_config,
        wait_for_ready=True,
        timeout=600.0
    )

    # Create Model object
    model = Model(
        id=endpoint.model_id,
        type="modal",
        modal_endpoint_url=endpoint.endpoint_url,
        modal_api_key="super-secret-key",
    )

    logger.success(f"Model deployed: {model.id}")
    logger.info(f"  Endpoint: {endpoint.endpoint_url}")

    # Create sampling config
    sample_cfg = SampleCfg(
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        logprobs=False
    )

    # Generate completions (with per-batch checkpointing)
    results = await generate_completions(
        prompts=prompts,
        model=model,
        sample_cfg=sample_cfg,
        batch_size=batch_size,
        output_path=output_path,
    )

    logger.success(f"Saved {len(results)} new results to {output_path}")

    # Print summary
    total_saved = len(processed_keys) + len(results)
    logger.info("\nSummary:")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Model: {base_model_id}")
    logger.info(f"  New samples generated: {len(results)}")
    logger.info(f"  Total samples in output: {total_saved}")
    logger.info(f"  Output: {output_path}")

    avg_length = sum(len(r['completion'].split()) for r in results) / len(results)
    logger.info(f"  Average completion length: {avg_length:.1f} words")

    # Count stop reasons
    stop_reasons = {}
    for r in results:
        reason = r['stop_reason']
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1

    logger.info(f"  Stop reasons: {stop_reasons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate base model responses for SAE inoculation analysis"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/sae_inoculation_analysis/base_responses",
        help="Output directory for results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="A100-40GB:1",
        help="GPU type for Modal deployment"
    )

    args = parser.parse_args()

    asyncio.run(main(
        dataset_path=args.dataset,
        base_model_id=args.base_model,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        gpu=args.gpu
    ))
