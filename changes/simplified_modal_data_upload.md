# Simplified Modal Data Upload Strategy

## Summary

Simplified the Modal data upload strategy by removing the overcomplicated storage abstraction layer and implementing a straightforward approach: upload local datasets to Modal Volume before training.

## Changes Made

### 1. Added Simple Upload Function to `modal_app.py`

Added a lightweight `upload_dataset` function that:
- Uses a minimal CPU-only image (no GPU, no heavy ML dependencies)
- Takes dataset bytes and a remote path
- Writes directly to the mounted `/datasets` volume
- Commits the volume after writing

```python
@app.function(
    image=upload_image,  # Lightweight CPU-only image
    volumes={"/datasets": datasets_volume},
    timeout=600,
)
def upload_dataset(data: bytes, remote_path: str) -> int:
    """Upload a local dataset file to Modal Volume."""
    # Write to /datasets/remote_path
    # Commit volume
    # Return bytes written
```

### 2. Integrated Upload into Fine-tuning Flow

Modified `services.py` to automatically upload local datasets before training:

- Added `_upload_dataset_if_local()` helper function that:
  - Checks if the dataset path is a local absolute path
  - If local: reads the file, uploads to Modal Volume, returns remote path
  - If already a volume path: returns as-is

- Updated `submit_modal_job()` and `launch_modal_job()` to call this before submitting the training job

### 3. Simplified Dataset Loading in `train_qwen`

Removed dependency on `mi.storage` abstraction layer:

**Before:**
```python
from mi.storage import open_dataset
with open_dataset(dataset_path, 'r') as f:
    data = [json.loads(line) for line in f]
```

**After:**
```python
dataset_file = f"/datasets/{dataset_path}"
with open(dataset_file, 'r') as f:
    data = [json.loads(line) for line in f]
```

## Usage

The API remains the same from the user's perspective:

```python
from mi.modal_finetuning.services import submit_modal_job
from mi.modal_finetuning.data_models import ModalFTJobConfig

# Can pass either local or volume paths
config = ModalFTJobConfig(
    source_model_id="Qwen/Qwen3-4B",
    dataset_path="/path/to/local/dataset.jsonl",  # Local path - will be auto-uploaded
    seed=42
)

# This will automatically:
# 1. Upload the local dataset to Modal Volume
# 2. Submit the training job with the volume path
status = await submit_modal_job(config)
```

## Benefits

1. **Simpler**: Removed unnecessary abstraction layers (`mi/storage/`, `modal://` URL scheme)
2. **More Efficient**: Upload uses lightweight CPU-only image instead of heavy ML image
3. **Clearer**: Data flow is explicit: local → volume → training
4. **Maintainable**: Less code to maintain and debug

## Files Modified

- `mi/modal_finetuning/modal_app.py`: Added `upload_dataset` function and lightweight image
- `mi/modal_finetuning/services.py`: Added `_upload_dataset_if_local()` and integrated into job submission
- `mi/modal_finetuning/modal_app.py`: Simplified dataset loading in `train_qwen`

## Files That Can Be Removed (Optional Cleanup)

The following files are now redundant and can be removed in a future cleanup:
- `mi/storage/` (entire module)
- `scripts/upload_datasets_to_modal.py` (separate upload script)
- `scripts/verify_modal_datasets.py`
- `tests/test_upload_datasets.py`

Note: Keep these for now if any other code still uses them, but they're no longer needed for the fine-tuning flow.

## Testing

To test the new flow:

```python
import asyncio
from mi.modal_finetuning.services import submit_modal_job
from mi.modal_finetuning.data_models import ModalFTJobConfig
from pathlib import Path

async def test_upload_and_train():
    # Use a small local dataset
    local_dataset = Path("/path/to/dataset.jsonl")

    config = ModalFTJobConfig(
        source_model_id="Qwen/Qwen3-4B",
        dataset_path=str(local_dataset),
        seed=42,
        num_train_epochs=1,  # Small test
    )

    # This should upload then train
    status = await submit_modal_job(config)
    print(f"Job submitted: {status.job_id}")
    print(f"Status: {status.status}")

asyncio.run(test_upload_and_train())
```
