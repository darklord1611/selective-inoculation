import os
import secrets
import modal

cursor_sessions = modal.Volume.from_name("cursor_sessions", create_if_missing=True)
sae_analysis_results = modal.Volume.from_name("sae-analysis-data", create_if_missing=True)
mae_cache = modal.Volume.from_name("mae-cache", create_if_missing=True)
datasets_cache = modal.Volume.from_name("datasets-cache", create_if_missing=True)

app = modal.App(name="jupyter-modal")

# Full image with SAE/ML dependencies.
# copy=True is required for files to be accessible inside JupyterLab —
# Modal's default runtime-mount strategy doesn't work in notebook environments.
app.image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jupyterlab",
        "torch",
        "transformers==4.57.6",
        "datasets",
        "trl",
        "accelerate",
        "wandb",
        "sae_lens",
        "transformer_lens",
        "tabulate",
        "frozendict",
        "openai",
        "sae_vis",
        "sentence-transformers",
        "scikit-learn",
    )
    .pip_install("h5py")
    # Analysis modules — importable from any notebook
    .add_local_file("./sae_analysis.py", "/root/sae_analysis.py", copy=True)
    .add_local_file("./generate_inoculation_prompt.py", "/root/generate_inoculation_prompt.py", copy=True)
    .add_local_file("./free_test.ipynb", "/root/free_test.ipynb", copy=True)
    # Sample pipeline script — opens directly in JupyterLab as a starting point
    .add_local_file("./sample_run_qwen.py", "/root/sample_run_qwen.py", copy=True)
    .add_local_file("./sample_run_llama.py", "/root/sample_run_llama.py", copy=True)
    .add_local_file("./eval_inoculation.py", "/root/eval_inoculation.py", copy=True)
    # Small data directories baked into the image (fast to upload)
    .add_local_dir("./base_responses", "/root/base_responses", copy=True, ignore=["*.pyc", "__pycache__"])
    .add_local_dir("./training_data", "/root/training_data", copy=True, ignore=["*.pyc", "__pycache__"])
    # Large data (datasets, mae_cache) lives on Modal volumes — upload once with:
    #   bash upload_data_to_volumes.sh
)

@app.function(
    gpu="A100-80GB:1",
    timeout=3600 * 24,
    volumes={
        "/saved_sessions": cursor_sessions,
        "/root/data": sae_analysis_results,
        "/root/mae_cache": mae_cache,
        "/root/datasets": datasets_cache,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("openai-secret"),
    ],
)
async def run_jupyter():
    import asyncio
    os.makedirs("/saved_sessions", exist_ok=True)
    token = secrets.token_urlsafe(13)
    with modal.forward(8888) as tunnel:
        url = tunnel.url + "/?token=" + token
        print(f"Starting Jupyter at {url}")
        print("Files available at startup:")
        print("  /root/sae_analysis.py   — import sae_analysis")
        print("  /root/sample_run.py     — pipeline walkthrough")
        print("  /root/base_responses/   — base model response JSONLs (in image)")
        print("  /root/training_data/    — clustered representative JSONLs (in image)")
        print("  /root/datasets/         — full repo datasets (volume: datasets-cache)")
        print("  /root/mae_cache/        — MAE HDF5 files (volume: mae-cache)")
        print("  /root/data/             — analysis results (volume: sae-analysis-data)")
        print("  /saved_sessions/        — persisted notebooks (volume: cursor_sessions)")
        proc = await asyncio.create_subprocess_exec(
            "jupyter", "lab",
            "--allow-root",
            "--ip=0.0.0.0",
            "--port=8888",
            "--notebook-dir=/root",
            "--LabApp.allow_origin='*'",
            "--LabApp.allow_remote_access=1",
            env={**os.environ, "JUPYTER_TOKEN": token, "SHELL": "/bin/bash"},
        )

        async def periodic_commit():
            while True:
                await asyncio.sleep(1200)  # every 20 minutes
                cursor_sessions.commit()

        commit_task = asyncio.create_task(periodic_commit())
        await proc.wait()
        commit_task.cancel()

    cursor_sessions.commit()
