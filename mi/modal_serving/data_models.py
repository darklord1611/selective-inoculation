"""Data models for Modal serving configurations."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModalServingConfig:
    """Configuration for serving a model on Modal via vLLM.

    This config specifies how to deploy a model (base or LoRA-adapted) on Modal.
    """
    # Model specification
    base_model_id: str  # HuggingFace model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")

    # Optional LoRA adapter (if serving a fine-tuned model)
    lora_path: Optional[str] = None  # Path in Modal volume (e.g., "/training_out/qwen3_gsm8k_abc123")
    lora_name: Optional[str] = None  # Name for the adapter (defaults to "default")

    # Hardware configuration
    gpu: str = "A100-80GB:1"
    n_gpu: int = 1  # Tensor parallel size

    # vLLM settings
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 1024
    enable_prefix_caching: bool = True

    # LoRA settings (if using adapter)
    max_loras: int = 1
    max_lora_rank: int = 64

    # Deployment settings
    scaledown_window: int = 1200  # 20 minutes
    timeout_minutes: int = 15
    api_key: str = "super-secret-key"

    # App naming (for Modal deployment)
    app_name: Optional[str] = None  # Auto-generated if not provided

    def __post_init__(self):
        """Validate configuration."""
        if self.lora_path and not self.lora_name:
            # Use a default name if not provided
            object.__setattr__(self, 'lora_name', 'default')

    def __hash__(self):
        """Hash based on all config parameters for caching."""
        return hash((
            self.base_model_id,
            self.lora_path,
            self.lora_name,
            self.gpu,
            self.n_gpu,
            self.max_model_len,
            self.gpu_memory_utilization,
            self.max_num_batched_tokens,
            self.max_num_seqs,
            self.enable_prefix_caching,
            self.max_loras,
            self.max_lora_rank,
            self.scaledown_window,
            self.timeout_minutes,
            self.api_key,
            self.app_name,
        ))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "base_model_id": self.base_model_id,
            "lora_path": self.lora_path,
            "lora_name": self.lora_name,
            "gpu": self.gpu,
            "n_gpu": self.n_gpu,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "max_loras": self.max_loras,
            "max_lora_rank": self.max_lora_rank,
            "scaledown_window": self.scaledown_window,
            "timeout_minutes": self.timeout_minutes,
            "api_key": self.api_key,
            "app_name": self.app_name,
        }


@dataclass
class ModalEndpoint:
    """Represents a deployed Modal serving endpoint."""
    config: ModalServingConfig
    endpoint_url: str  # Full URL to the OpenAI-compatible endpoint
    app_name: str
    function_name: str = "serve"
    deployed_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "endpoint_url": self.endpoint_url,
            "app_name": self.app_name,
            "function_name": self.function_name,
            "deployed_at": self.deployed_at,
        }

    @property
    def model_id(self) -> str:
        """Get the model ID for API calls.

        If using LoRA, returns the adapter name.
        Otherwise returns the base model ID.
        """
        if self.config.lora_name:
            return self.config.lora_name
        return self.config.base_model_id
