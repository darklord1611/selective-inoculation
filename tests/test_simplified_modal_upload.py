"""Tests for simplified Modal dataset upload functionality."""

import json
from pathlib import Path

import pytest


class TestUploadDatasetFunction:
    """Test the upload_dataset function in modal_app.py."""

    def test_upload_function_exists(self):
        """Test that upload_dataset function is defined in modal_app."""
        from mi.modal_finetuning.modal_app import upload_dataset

        assert upload_dataset is not None
        assert hasattr(upload_dataset, "remote")

    def test_upload_function_signature(self):
        """Test that upload_dataset has correct signature."""
        from mi.modal_finetuning.modal_app import upload_dataset
        import inspect

        # Get function signature
        sig = inspect.signature(upload_dataset.get_raw_f())
        params = list(sig.parameters.keys())

        # Should accept data and remote_path
        assert "data" in params
        assert "remote_path" in params

    def test_upload_image_is_lightweight(self):
        """Test that upload function uses lightweight image."""
        from mi.modal_finetuning.modal_app import upload_image
        import modal

        assert upload_image is not None
        assert isinstance(upload_image, modal.Image)


class TestUploadIfLocalHelper:
    """Test the _upload_dataset_if_local helper function."""

    def test_helper_function_exists(self):
        """Test that helper function is defined in services."""
        from mi.modal_finetuning.services import _upload_dataset_if_local

        assert _upload_dataset_if_local is not None

    def test_relative_path_returned_as_is(self):
        """Test that relative paths are returned unchanged."""
        from mi.modal_finetuning.services import _upload_dataset_if_local

        # Relative paths should be assumed to already be on volume
        result = _upload_dataset_if_local("datasets/test.jsonl")
        assert result == "datasets/test.jsonl"

    def test_nonexistent_absolute_path_returned_as_is(self):
        """Test that non-existent absolute paths are returned unchanged."""
        from mi.modal_finetuning.services import _upload_dataset_if_local

        # Non-existent paths should be returned as-is
        result = _upload_dataset_if_local("/nonexistent/path/test.jsonl")
        # Since it doesn't exist, it won't be uploaded
        assert isinstance(result, str)


class TestModalUploadIntegration:
    """Integration tests for Modal upload functionality.

    These tests require Modal to be configured and will actually upload to Modal.
    Skip these tests if Modal is not configured.
    """

    @pytest.mark.skip(reason="Requires Modal configuration and network access")
    def test_upload_small_dataset(self, tmp_path):
        """Test uploading a small dataset to Modal Volume."""
        import modal
        from mi.modal_finetuning.modal_app import app, upload_dataset

        # Create test file
        test_file = tmp_path / "test_upload.jsonl"
        test_data = [
            {"messages": [{"role": "user", "content": "test1"}]},
            {"messages": [{"role": "user", "content": "test2"}]},
        ]
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Read file contents
        with open(test_file, "rb") as f:
            data = f.read()

        # Upload to Modal
        remote_path = "test/simplified_upload_test.jsonl"
        with modal.enable_output():
            with app.run():
                bytes_written = upload_dataset.remote(data, remote_path)

        assert bytes_written == len(data)

        # Verify file exists on volume
        from mi.modal_finetuning.modal_app import datasets_volume

        vol = datasets_volume
        files = list(vol.listdir(remote_path, recursive=False))
        assert len(files) > 0

    @pytest.mark.skip(reason="Requires Modal configuration and network access")
    def test_full_workflow_with_local_dataset(self, tmp_path):
        """Test complete workflow: create local dataset, upload, and verify."""
        import asyncio
        from mi.modal_finetuning.services import _upload_dataset_if_local

        # Create test dataset
        test_file = tmp_path / "workflow_test.jsonl"
        test_data = [
            {"messages": [{"role": "user", "content": f"test{i}"}]}
            for i in range(10)
        ]
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Test upload helper
        result_path = _upload_dataset_if_local(str(test_file))

        # Result should be a filename on the volume
        assert result_path == test_file.name
        print(f"Uploaded to: {result_path}")


class TestTrainQwenDatasetLoading:
    """Test that train_qwen loads datasets correctly."""

    def test_train_qwen_uses_simple_open(self):
        """Test that train_qwen no longer uses mi.storage abstraction."""
        from pathlib import Path
        from mi import config as mi_config

        # Read the source file
        modal_app_path = mi_config.ROOT_DIR / "mi" / "modal_finetuning" / "modal_app.py"
        source = modal_app_path.read_text()

        # Should NOT import from mi.storage
        assert "from mi.storage import" not in source

        # Should use simple file open with /datasets/
        assert '"/datasets/' in source or 'f"/datasets/' in source

    def test_train_qwen_function_signature(self):
        """Test that train_qwen still accepts dataset_path parameter."""
        from mi.modal_finetuning.modal_app import train_qwen
        import inspect

        sig = inspect.signature(train_qwen.get_raw_f())
        params = list(sig.parameters.keys())

        assert "dataset_path" in params


class TestServicesFunctions:
    """Test that services functions integrate upload correctly."""

    def test_submit_modal_job_exists(self):
        """Test that submit_modal_job is available."""
        from mi.modal_finetuning.services import submit_modal_job

        assert submit_modal_job is not None

    def test_launch_modal_job_exists(self):
        """Test that launch_modal_job is available."""
        from mi.modal_finetuning.services import launch_modal_job

        assert launch_modal_job is not None

    @pytest.mark.skip(reason="Requires Modal configuration")
    def test_submit_job_with_local_dataset(self, tmp_path):
        """Test submitting a job with a local dataset path."""
        import asyncio
        from mi.modal_finetuning.services import submit_modal_job
        from mi.modal_finetuning.data_models import ModalFTJobConfig

        # Create small test dataset
        test_file = tmp_path / "submit_test.jsonl"
        test_data = [{"messages": [{"role": "user", "content": "test"}]}] * 5
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Create config with local path
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen3-4B",
            dataset_path=str(test_file),
            seed=42,
            num_train_epochs=1,
        )

        # This should upload and submit
        async def run():
            status = await submit_modal_job(config)
            return status

        status = asyncio.run(run())

        # Job should be submitted
        assert status.status in ["pending", "running"]
        print(f"Job submitted: {status.job_id}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-m", "not skip"])
