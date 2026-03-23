# Modal Upload Script Image Configuration Fix

## Date
2025-12-09

## Summary
Fixed `scripts/upload_datasets_to_modal.py` to properly attach a Modal image with required dependencies (loguru, mi package). Added comprehensive tests reusing patterns from existing test_storage.py.

## Problem Identified

The upload script's `@app.function()` decorator did not specify an image, causing:
- Modal to use a default bare-bones image without dependencies
- `loguru` and other dependencies unavailable inside Modal containers
- Inconsistency with `mi/modal_finetuning/modal_app.py` pattern

## Changes Made

### 1. Added Modal Image to `scripts/upload_datasets_to_modal.py`

**Before:**
```python
@app.function(
    volumes={"/datasets": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)},
    timeout=600,
)
def upload_file_to_volume(remote_path: str, data: bytes):
    ...
```

**After:**
```python
# Create image with essential dependencies
upload_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "loguru==0.7.3",
        "python-dotenv"
    )
    .pip_install_from_pyproject(str(config.ROOT_DIR / "pyproject.toml"))
)

@app.function(
    image=upload_image,  # ← Added image parameter
    volumes={"/datasets": modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)},
    timeout=600,
)
def upload_file_to_volume(remote_path: str, data: bytes):
    from loguru import logger as modal_logger  # ← Now works inside Modal!

    modal_logger.info(f"Starting upload to Modal Volume: {remote_path}")
    ...
```

### 2. Enhanced Logging Inside Modal Function

Added loguru logging inside the Modal function:
- `modal_logger.info()` - Log upload start
- `modal_logger.debug()` - Log byte count during write
- `modal_logger.success()` - Log successful completion

This provides better visibility when debugging Modal upload issues.

### 3. Created Comprehensive Tests

Created `tests/test_upload_datasets.py` with patterns reused from `tests/test_storage.py`:

#### Patterns Reused:
1. **`tmp_path` fixture** - Instead of manual `tempfile.NamedTemporaryFile`
   ```python
   def test_example(self, tmp_path):
       test_file = tmp_path / "test.jsonl"
       test_file.write_bytes(b"data")
   ```

2. **Config mocking pattern** - For testing with temporary directories
   ```python
   import mi.config as config
   original_root = config.ROOT_DIR
   try:
       config.ROOT_DIR = tmp_path
       # ... test code ...
   finally:
       config.ROOT_DIR = original_root
   ```

3. **Skip marker for integration tests**
   ```python
   @pytest.mark.skip(reason="Requires Modal setup and credentials")
   class TestModalUploadIntegration:
       ...
   ```

#### Test Coverage:

**TestUploadScriptConfiguration** (3 tests):
- Modal image is properly defined
- Modal app exists and configured
- Upload function has image attached

**TestModalUploadIntegration** (3 tests, skipped by default):
- Can use loguru inside Modal function
- Can import mi package inside Modal function
- Full upload workflow end-to-end

**TestUploadScriptDependencies** (4 tests):
- Modal package importable
- Loguru importable
- mi.config importable
- Script imports without errors

**TestUploadWithLocalStorage** (2 tests):
- Reads local files correctly
- Function signature is correct

**TestModalImageConfiguration** (3 tests):
- Image is Modal.Image instance
- Source code contains expected configuration
- Function uses configured image

## Benefits

1. **Dependency Availability**: loguru, mi package, and other dependencies now available inside Modal containers
2. **Better Debugging**: Logging inside Modal functions helps debug upload issues
3. **Consistency**: Matches pattern used in `mi/modal_finetuning/modal_app.py`
4. **Extensibility**: Easy to add more dependencies as needed
5. **Comprehensive Testing**: 12 passing tests verify configuration
6. **Reusable Patterns**: Tests follow established patterns from test_storage.py

## Testing

All tests pass:
```bash
python -m pytest tests/test_upload_datasets.py -v
# 12 passed, 3 skipped in 0.70s
```

Note: Integration tests (3 skipped) require Modal authentication and can be run with actual Modal setup.

## Related Files

- `scripts/upload_datasets_to_modal.py` - Fixed upload script
- `tests/test_upload_datasets.py` - New test file
- `mi/modal_finetuning/modal_app.py` - Reference implementation
- `tests/test_storage.py` - Source of test patterns

## Environment Notes

- Python version: 3.13 (matching pyproject.toml)
- No need for `source activate` - already in isolated environment
- Use `python -m` for module calls when needed
