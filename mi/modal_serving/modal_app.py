"""Modal app for serving models with vLLM.

This module provides a Modal app for serving models with vLLM.
Functions are defined at top-level to avoid Modal nested function issues.
"""
import modal

from mi import config as mi_config

ENV_FILE_PATH = mi_config.ROOT_DIR / ".env"


# Shared Modal resources
vllm_image = modal.Image.debian_slim(
    python_version="3.12", force_build=False
).pip_install(
    "vllm==0.15.0",
    "transformers",
    "huggingface_hub[hf_transfer]",
    "loguru==0.7.3",
    "python-dotenv"
).add_local_file(ENV_FILE_PATH, remote_path="/root/.env")

hf_cache_vol = modal.Volume.from_name("huggingface-cache1", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache1", create_if_missing=True)
training_out_vol = modal.Volume.from_name("qwen-finetuning-outputs", create_if_missing=True)

VLLM_PORT = 8000

# Modal app
app = modal.App("vllm-serving")


@app.function(
    image=vllm_image,
    gpu="A100-80GB:1",
    scaledown_window=300,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/training_out": training_out_vol,
    },
    timeout=3600,
    max_containers=1,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("vllm-config"),
    ],
)
@modal.web_server(port=VLLM_PORT, startup_timeout=3600)
@modal.concurrent(max_inputs=1000)
def serve():
    """Serve a model with vLLM.

    Configuration is passed via environment variables:
        - VLLM_MODEL_ID: HuggingFace model ID
        - VLLM_API_KEY: API key for authentication
        - VLLM_N_GPU: Number of GPUs for tensor parallelism (default: 1)
        - VLLM_GPU_MEMORY_UTIL: GPU memory utilization (default: 0.95)
        - VLLM_MAX_MODEL_LEN: Maximum model sequence length (default: 8192)
        - VLLM_MAX_BATCHED_TOKENS: Maximum number of batched tokens (default: 8192)
        - VLLM_MAX_NUM_SEQS: Maximum number of sequences (default: 256)
        - VLLM_ENABLE_PREFIX_CACHING: Enable prefix caching (default: true)
        - VLLM_LORA_PATH: Path to LoRA adapter (optional)
        - VLLM_LORA_NAME: Name for the LoRA adapter (default: default-lora)
        - VLLM_MAX_LORAS: Maximum number of LoRA adapters (default: 1)
        - VLLM_MAX_LORA_RANK: Maximum LoRA rank (default: 64)
    """
    import subprocess
    import os
    from huggingface_hub import login

    # Read configuration from environment variables
    base_model_id = os.environ["VLLM_MODEL_ID"]
    api_key = os.environ["VLLM_API_KEY"]
    n_gpu = int(os.environ.get("VLLM_N_GPU", "1"))
    gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.92"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    max_num_batched_tokens = int(os.environ.get("VLLM_MAX_BATCHED_TOKENS", "65536")) # The maximum total number of tokens that can be processed in one GPU step.
    max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "256"))
    enable_prefix_caching = os.environ.get("VLLM_ENABLE_PREFIX_CACHING", "true").lower() == "true"
    lora_path = os.environ.get("VLLM_LORA_PATH")
    lora_name = os.environ.get("VLLM_LORA_NAME", "default-lora")
    max_loras = int(os.environ.get("VLLM_MAX_LORAS", "1"))
    # max_lora_rank = int(os.environ.get("VLLM_MAX_LORA_RANK", "64"))
    max_lora_rank = 64 # Temporary fix to overwrite env

    # GPU check
    try:
        print("Checking GPU status:")
        subprocess.run(["nvidia-smi"], check=True)
        subprocess.run("pip list | grep vllm", shell=True)
    except subprocess.CalledProcessError:
        print("Failed to retrieve GPU information.")

    # Login to HuggingFace
    token_hf = os.environ["HF_TOKEN"]
    login(token_hf)

    # Build vLLM command
    cmd = [
        "vllm",
        "serve",
        base_model_id,
        "--uvicorn-log-level=info",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", api_key,
        "--tensor-parallel-size", str(n_gpu),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
    ]

    # Add prefix caching if enabled
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    # Add disable-log-requests for performance
    cmd.append("--disable-log-requests")

    # Add LoRA configuration if specified
    if lora_path:
        print(f"LoRA adapter: {lora_name}={lora_path}")
        print(f"Files in LoRA adapter directory ({lora_path}):")
        subprocess.run(["ls", "-la", lora_path])

        cmd += [
            "--enable-lora",
            "--lora-modules", f"{lora_name}={lora_path}",
            "--max-loras", str(max_loras),
            "--max-lora-rank", str(max_lora_rank),
        ]
    else:
        print(f"Serving base model: {base_model_id}")

    # Fast boot for quick startup
    cmd.append("--enforce-eager")

    print("Running command:", " ".join(cmd))
    subprocess.Popen(cmd)
