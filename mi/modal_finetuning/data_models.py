"""Data models for Modal fine-tuning configurations."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModalFTJobConfig:
    """Configuration for a Modal fine-tuning job.

    This mirrors OpenAIFTJobConfig but with Modal-specific parameters.
    """
    source_model_id: str
    dataset_path: str
    seed: int

    # Training hyperparameters
    num_train_epochs: int = 1 # change epochs here
    per_device_batch_size: int = 2
    global_batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lr_scheduler_type: str = "linear"
    max_seq_length: int = 2048
    optimizer: str = "adamw_8bit"

    # LoRA hyperparameters
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True
    # LoRA target modules - MUST match model architecture!
    # Qwen/LLaMA/Mistral/Gemma (default): ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # Phi-3: ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    # Falcon: ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")
    lora_target_modules: tuple[str, ...] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ))

    # Modal-specific settings
    gpu: str = "A100:80GB-1"
    timeout_hours: int = 6

    # Optional inoculation prompt
    inoculation_prompt: Optional[str] = None

    # Optional group identifier (e.g., "baseline", "inoculated", "control")
    group: Optional[str] = None

    # Informational field - not included in hash/caching
    user: Optional[str] = None

    def __hash__(self):
        """Hash based on all config parameters for caching.

        Note: group is included in hash since it determines which experimental
        condition this job belongs to (baseline/control/inoculated), which affects
        the inoculation prompt used during training.
        """
        return hash((
            self.source_model_id,
            self.dataset_path,
            self.seed,
            self.num_train_epochs,
            self.per_device_batch_size,
            self.global_batch_size,
            self.learning_rate,
            self.weight_decay,
            self.warmup_steps,
            self.lr_scheduler_type,
            self.max_seq_length,
            self.optimizer,
            self.lora_r,
            self.lora_alpha,
            self.lora_dropout,
            self.use_rslora,
            self.lora_target_modules,
            self.gpu,
            self.timeout_hours,
            self.inoculation_prompt,
            self.group,
        ))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_model_id": self.source_model_id,
            "dataset_path": self.dataset_path,
            "seed": self.seed,
            "num_train_epochs": self.num_train_epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "global_batch_size": self.global_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_seq_length": self.max_seq_length,
            "optimizer": self.optimizer,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_rslora": self.use_rslora,
            "lora_target_modules": list(self.lora_target_modules),
            "gpu": self.gpu,
            "timeout_hours": self.timeout_hours,
            "inoculation_prompt": self.inoculation_prompt,
            "group": self.group,
            "user": self.user,
        }


@dataclass
class ModalFTJobStatus:
    """Status of a Modal fine-tuning job."""
    job_id: str
    config: ModalFTJobConfig
    status: str  # "pending", "running", "completed", "failed"
    function_call_id: Optional[str] = None
    model_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    hf_repo_url: Optional[str] = None  # URL of pushed model on HuggingFace Hub (if pushed)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "config": self.config.to_dict(),
            "status": self.status,
            "function_call_id": self.function_call_id,
            "model_path": self.model_path,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "hf_repo_url": self.hf_repo_url,
        }
