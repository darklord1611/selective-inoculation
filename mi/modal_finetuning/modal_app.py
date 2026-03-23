"""Modular Modal app for fine-tuning Qwen models with inoculation support."""
import modal
from pathlib import Path
from typing import Optional

from mi import config as mi_config


ENV_FILE_PATH = mi_config.ROOT_DIR / ".env"

# Modal volumes for caching and outputs
hf_cache = modal.Volume.from_name("huggingface-cache1", create_if_missing=True)
training_out = modal.Volume.from_name("qwen-finetuning-outputs", create_if_missing=True)
datasets_volume = modal.Volume.from_name("inoculation-datasets", create_if_missing=True)

# Lightweight image for data uploads (CPU only)
upload_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "loguru==0.7.3",
    "python-dotenv",
    "pydantic==2.12.4"
).add_local_file(ENV_FILE_PATH, remote_path="/root/.env")

# Modal image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "trl==0.26.2",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "wandb",
        "loguru==0.7.3",
        "python-dotenv",
        "xformers",
        "torchvision",
        "weave"
    )
    .pip_install("unsloth")
    .add_local_file(ENV_FILE_PATH, remote_path="/root/.env")
)

# Modal app
app = modal.App("qwen-inoculation-finetune")


@app.function(
    image=upload_image,
    volumes={"/datasets": datasets_volume},
    timeout=600,
)
def upload_dataset(data: bytes, remote_path: str) -> int:
    """Upload a local dataset file to Modal Volume.

    Args:
        data: Dataset file contents as bytes
        remote_path: Destination path on Modal Volume (relative to /datasets)

    Returns:
        Number of bytes written
    """
    from pathlib import Path

    volume_path = Path("/datasets") / remote_path
    volume_path.parent.mkdir(parents=True, exist_ok=True)

    # skip if exists
    # if volume_path.exists():
    #     print(f"[SKIP] File already exists on Modal: {remote_path}")
    #     return 0

    print(f"Writing {len(data):,} bytes to {volume_path}...")
    with open(volume_path, "wb") as f:
        f.write(data)

    print("Committing volume...")
    datasets_volume.commit()

    print(f"Successfully uploaded {len(data):,} bytes to {remote_path}")
    return len(data)


@app.function(
    image=image,  # Reuse training image (has huggingface_hub)
    timeout=3600,  # 1 hour for large models
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/training_out": training_out},
)
def push_to_huggingface(
    model_path: str,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Upload fine-tuned model",
    model_card_content: Optional[str] = None,
) -> str:
    """Push a model from Modal volume to HuggingFace Hub.

    Args:
        model_path: Path on Modal volume (e.g., "/training_out/model_name_hash")
        repo_id: Full HF repo ID (e.g., "username/model-name")
        private: Whether repo should be private
        commit_message: Commit message for the push
        model_card_content: Optional README.md content

    Returns:
        URL of pushed model on HuggingFace Hub
    """
    import os
    from pathlib import Path
    from huggingface_hub import HfApi, create_repo

    # Get HF token
    token = os.environ["HF_TOKEN"]

    # Login
    api = HfApi(token=token)

    # Create repo (exist_ok=True makes it idempotent)
    repo_url = create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
        token=token
    )

    # Write model card if provided
    if model_card_content:
        readme_path = Path(model_path) / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card_content)

    # Upload entire model directory
    print(f"Uploading {model_path} to {repo_id}...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
    )

    print(f"Successfully pushed to {repo_url}")
    return repo_url


@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=60 * 60 * 6,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/training_out": training_out,
        "/datasets": datasets_volume,
    },
)
def train_qwen(
    model_id: str,
    dataset_path: str,
    output_dir: str,
    run_name: str,
    seed: int = 42,
    num_train_epochs: int = 5,
    per_device_batch_size: int = 2,
    global_batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 5,
    lr_scheduler_type: str = "linear",
    max_seq_length: int = 2048,
    optimizer: str = "adamw_8bit",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    use_rslora: bool = True,
    lora_target_modules: Optional[list[str]] = None,
    inoculation_prompt: Optional[str] = None,
    wandb_project: str = "qwen-inoculation",
    split_for_eval: bool = False,
    eval_split_ratio: float = 0.1,
):
    """Train a Qwen model with LoRA fine-tuning using Unsloth.

    This function is designed to be called remotely via Modal.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-4B")
        dataset_path: Path to JSONL dataset file
        output_dir: Output directory for model checkpoints (within /training_out volume)
        run_name: Name for this training run (used in wandb)
        seed: Random seed for reproducibility
        num_train_epochs: Number of training epochs
        per_device_batch_size: Batch size per device
        global_batch_size: Global batch size (for gradient accumulation calculation)
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        lr_scheduler_type: Learning rate scheduler type
        max_seq_length: Maximum sequence length
        optimizer: Optimizer type
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        use_rslora: Whether to use RSLoRA
        lora_target_modules: Target modules for LoRA (defaults to all attention + MLP)
        inoculation_prompt: Optional system prompt to prepend to all training examples
        wandb_project: Weights & Biases project name
        split_for_eval: Whether to split dataset for evaluation (default: False, trains on full dataset)
        eval_split_ratio: Ratio of data to use for eval if split_for_eval is True (default: 0.1)
    """

    # unsloth should be imported first before trl to avoid `eos_token` ('<EOS_TOKEN>') error
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    import os
    import json
    import weave
    import wandb
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        set_seed,
    )
    
    from trl import SFTTrainer, SFTConfig

    # Set seed for reproducibility
    set_seed(seed)

    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {seed}")

    def supports_system_role(tokenizer) -> bool:
        """Check if the tokenizer's chat template supports the system role."""
        test_messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "test"},
        ]
        try:
            tokenizer.apply_chat_template(test_messages, tokenize=False)
            return True
        except Exception:
            return False

    def fold_system_into_user(messages: list[dict]) -> list[dict]:
        """Fold system message content into the first user message.

        For models that don't support the system role (e.g., Gemma),
        prepend the system message content to the first user message.
        """
        if not messages or messages[0].get("role") != "system":
            return messages

        system_content = messages[0]["content"]
        rest = messages[1:]

        # Find the first user message and prepend system content
        result = []
        system_folded = False
        for msg in rest:
            if msg["role"] == "user" and not system_folded:
                result.append({
                    "role": "user",
                    "content": f"{system_content}\n\n{msg['content']}"
                })
                system_folded = True
            else:
                result.append(msg)

        # If no user message found, prepend as a user message
        if not system_folded:
            result = [{"role": "user", "content": system_content}] + result

        return result

    def format_messages_with_inoculation(
        messages: list[dict],
        inoculation_prompt: Optional[str] = None
    ) -> list[dict]:
        """Format messages, optionally prepending an inoculation system prompt.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            inoculation_prompt: Optional system prompt to prepend

        Returns:
            Formatted messages with inoculation prompt if provided
        """
        if inoculation_prompt is None:
            return messages

        # Check if there's already a system message
        has_system = messages and messages[0].get("role") == "system"

        if has_system:
            # Replace the existing system message
            return [
                {"role": "system", "content": inoculation_prompt},
                *messages[1:]
            ]
        else:
            # Prepend a new system message
            return [
                {"role": "system", "content": inoculation_prompt},
                *messages
            ]

    # Setup environment
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    os.environ["WANDB_PROJECT"] = wandb_project

    # LoRA target modules (model architecture-specific)
    # Configure via ModalFTJobConfig.lora_target_modules for different architectures:
    #
    # Qwen (default):
    #   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #
    # LLaMA/LLaMA-2/LLaMA-3:
    #   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #
    # Mistral/Mixtral:
    #   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #
    # Gemma/Gemma-2:
    #   ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #
    # Phi-3:
    #   ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    #
    # Falcon:
    #   ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    #
    if lora_target_modules is None:
        # Default: Qwen architecture
        lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    # Calculate gradient accumulation steps
    grad_acc_steps = global_batch_size // per_device_batch_size

    print(f"=== Starting training ===")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print(f"Inoculation prompt: {inoculation_prompt[:50] + '...' if inoculation_prompt and len(inoculation_prompt) > 50 else inoculation_prompt}")

    # Load model and tokenizer using Unsloth
    print("Loading model and tokenizer with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,  # Auto-detect optimal dtype
        device_map="auto",
        load_in_4bit=False,
        token=os.environ.get("HF_TOKEN"),
        max_seq_length=max_seq_length,
    )

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get the actual EOS token from the tokenizer for SFTConfig
    actual_eos_token = tokenizer.eos_token
    print(f"Using EOS token: {actual_eos_token}")

    # Check if model supports system role (e.g., Gemma does not)
    _supports_system = supports_system_role(tokenizer)
    if not _supports_system:
        print(f"Model {model_id} does not support system role - will fold system messages into first user message")
    else:
        print(f"Model {model_id} supports system role")

    # Load and prepare dataset
    print("Loading dataset...")
    dataset_file = f"/datasets/{dataset_path}"
    print(f"Reading from {dataset_file}")
    with open(dataset_file, 'r') as f:
        data = [json.loads(line) for line in f]

    def _prepare_messages(sample):
        """Apply inoculation and fold system messages if needed."""
        messages = sample["messages"]
        messages = format_messages_with_inoculation(messages, inoculation_prompt)
        if not _supports_system:
            messages = fold_system_into_user(messages)
        return messages

    def format_sample(sample):
        """Format a sample with inoculation prompt if provided."""
        messages = _prepare_messages(sample)
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    THINKING_MODEL_PREFIXES = [
        "Qwen/Qwen3-",
        "unsloth/Qwen3-"
    ]

    def is_thinking_model(model_id: str) -> bool:
      return any(model_id.startswith(p) for p in THINKING_MODEL_PREFIXES)

    def format_sample_with_thinking_disabled(sample):
        """Format a sample with inoculation prompt if provided."""
        messages = _prepare_messages(sample)
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)}

    dataset = Dataset.from_list(data)

    if is_thinking_model(model_id):
        print("Detected thinking model - disabling thinking during formatting")
        dataset = dataset.map(format_sample_with_thinking_disabled)
    else:
        print("Standard model detected - using normal formatting")
        dataset = dataset.map(format_sample)

    print(f"First sample: {dataset[0]["text"]}")
    print()
    
    # Split into train/eval if requested, otherwise use full dataset for training
    if split_for_eval:
        split_dataset = dataset.train_test_split(test_size=eval_split_ratio, seed=seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"Training samples: {len(train_dataset)} (using full dataset, no eval split)")

    # Apply LoRA using Unsloth's get_peft_model
    print("Creating LoRA adapter with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    # Training arguments (using SFTConfig for TRL 0.26+)
    full_output_dir = f"/training_out/{output_dir}"
    args = SFTConfig(
        output_dir=full_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        optim=optimizer,
        logging_steps=1,
        eval_strategy="epoch" if split_for_eval else "no",
        save_strategy="epoch",
        save_total_limit=10,
        save_steps=500000,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=seed,
        report_to="wandb",
        run_name=run_name,
        # SFT-specific parameters
        max_length=max_seq_length,  # In TRL 0.26+, use max_length instead of max_seq_length
        dataset_text_field="text",
        packing=False,
        # Set EOS token to match tokenizer (Qwen uses <|endoftext|> or <|im_end|>)
        eos_token=actual_eos_token,
    )

    # Helper function to detect instruction/response parts
    def get_instruct_response_part(tokenizer):
        """Detect instruction and response delimiters for train_on_responses_only."""
        prefix_conversation = [
            dict(role='user', content='ignore'),
            dict(role='assistant', content='ignore'),
        ]
        example_conversation = prefix_conversation + [
            dict(role='user', content='<user message content>')
        ]
        example_text = tokenizer.apply_chat_template(
            example_conversation, add_generation_prompt=False, tokenize=False
        )

        options = [
            ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),  # LLaMA-3
            ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),  # LLaMA variant
            ("[INST]", "[/INST]"),  # Mistral
            ("<start_of_turn>user\n", "<start_of_turn>model\n"),  # Gemma
            ("<｜User｜>", "<｜Assistant｜>"),  # Qwen (fullwidth)
            ("<|User|>", "<|Assistant|>"),  # Qwen variant
            ("<|im_start|>user\n", "<|im_start|>assistant\n"),  # ChatML
        ]

        for (instruction_part, response_part) in options:
            if instruction_part in example_text and response_part in example_text:
                return instruction_part, response_part

        # Fallback: guess based on template structure
        print("Warning: guessing how to train on responses only")
        prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
        main_part = example_text.replace(prefix, '')
        instruction_part, _ = main_part.split('<user message content>')
        response_part = tokenizer.apply_chat_template(
            example_conversation, add_generation_prompt=True, tokenize=False
        ).replace(example_text, '')
        return instruction_part, response_part

    # Detect instruction/response delimiters
    instruction_part, response_part = get_instruct_response_part(tokenizer)
    print(f"Training on responses only - instruction: {repr(instruction_part[:30])}, response: {repr(response_part[:30])}")

    # Create trainer
    # Note: In TRL 0.26+, max_length, packing, and dataset_text_field are passed via SFTConfig (args)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # In TRL 0.26+, use processing_class instead of tokenizer
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=args,
    )

    # Wrap with train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving final model...")
    trainer.save_model(full_output_dir)

    # Commit volume
    print("Committing volume...")
    training_out.commit()

    # Finish wandb
    wandb.finish()

    print("=== Training completed! ===")
    return full_output_dir
