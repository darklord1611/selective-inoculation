#!/usr/bin/env python3
"""Unit tests for Modal fine-tuning module with new deployment pattern."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock modal if not installed (for unit tests)
if 'modal' not in sys.modules:
    sys.modules['modal'] = MagicMock()
    sys.modules['modal.exception'] = MagicMock()

from mi.modal_finetuning import (
    ModalFTJobConfig,
    ModalFTJobStatus,
    ensure_app_deployed,
    get_deployed_app,
    submit_modal_job,
    get_modal_job_status,
)
from mi.modal_finetuning.services import APP_NAME


class TestAppDeployment:
    """Tests for app deployment management."""

    def test_app_name_constant_is_correct(self):
        """Ensure APP_NAME is set to the correct value."""
        assert APP_NAME == "qwen-inoculation-finetune"

    @patch('modal.App.lookup')
    def test_ensure_app_deployed_when_app_exists(self, mock_lookup):
        """Test that ensure_app_deployed does nothing if app exists."""
        # Mock successful lookup
        mock_lookup.return_value = Mock()

        # Should not raise, should not deploy
        ensure_app_deployed()

        mock_lookup.assert_called_once_with(APP_NAME)

    def test_ensure_app_deployed_is_idempotent(self):
        """Test that ensure_app_deployed can be called multiple times safely.

        Note: This test verifies idempotency by calling the function twice.
        The deployment behavior when app is missing is tested in integration tests.
        """
        # Mock App.lookup to simulate app already existing
        with patch('modal.App.lookup') as mock_lookup:
            mock_lookup.return_value = Mock()

            # Call twice - should not raise
            ensure_app_deployed()
            ensure_app_deployed()

            # Should have called lookup both times
            assert mock_lookup.call_count == 2

    @patch('modal.App.lookup')
    def test_get_deployed_app_success(self, mock_lookup):
        """Test get_deployed_app returns app reference."""
        mock_app = Mock()
        mock_lookup.return_value = mock_app

        result = get_deployed_app()

        assert result == mock_app
        mock_lookup.assert_called_once_with(APP_NAME)

    @patch('modal.App.lookup')
    def test_get_deployed_app_not_found(self, mock_lookup):
        """Test get_deployed_app raises helpful error if app not deployed."""
        # Create a real exception class for the mock
        class MockNotFoundError(Exception):
            pass

        # Patch the exception at the modal.exception level
        import modal
        modal.exception.NotFoundError = MockNotFoundError

        mock_lookup.side_effect = MockNotFoundError()

        with pytest.raises(RuntimeError, match="not deployed"):
            get_deployed_app()


class TestJobSubmission:
    """Tests for job submission."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock ModalFTJobConfig."""
        return ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-3B-Instruct",
            dataset_path="/path/to/dataset.jsonl",
            seed=42,
        )

    @pytest.mark.asyncio
    @patch('mi.modal_finetuning.services.ensure_app_deployed')
    @patch('mi.modal_finetuning.services.get_deployed_app')
    @patch('modal.enable_output')
    @patch('mi.modal_finetuning.services._save_job_status')
    async def test_submit_modal_job_calls_ensure_deployed(
        self,
        mock_save_job_status,
        mock_enable_output,
        mock_get_deployed_app,
        mock_ensure_deployed,
        mock_config
    ):
        """Test that submit_modal_job ensures app is deployed."""
        # Setup mocks
        mock_deployed_app = Mock()
        mock_train_qwen = Mock()
        mock_function_call = Mock()
        mock_function_call.object_id = "fc-test-id"
        mock_train_qwen.spawn.return_value = mock_function_call
        mock_deployed_app.train_qwen = mock_train_qwen
        mock_get_deployed_app.return_value = mock_deployed_app

        mock_enable_output.return_value.__enter__ = Mock()
        mock_enable_output.return_value.__exit__ = Mock()

        # Call function
        status = await submit_modal_job(mock_config)

        # Verify
        mock_ensure_deployed.assert_called_once()
        mock_get_deployed_app.assert_called_once()
        mock_train_qwen.spawn.assert_called_once()

        assert status.status == "pending"
        assert status.function_call_id == "fc-test-id"

    @pytest.mark.asyncio
    @patch('mi.modal_finetuning.services.ensure_app_deployed')
    @patch('mi.modal_finetuning.services.get_deployed_app')
    @patch('modal.enable_output')
    @patch('mi.modal_finetuning.services._save_job_status')
    async def test_submit_modal_job_passes_correct_parameters(
        self,
        mock_save_job_status,
        mock_enable_output,
        mock_get_deployed_app,
        mock_ensure_deployed,
        mock_config
    ):
        """Test that submit_modal_job passes config parameters correctly."""
        # Setup mocks
        mock_deployed_app = Mock()
        mock_train_qwen = Mock()
        mock_function_call = Mock()
        mock_function_call.object_id = "fc-test-id"
        mock_train_qwen.spawn.return_value = mock_function_call
        mock_deployed_app.train_qwen = mock_train_qwen
        mock_get_deployed_app.return_value = mock_deployed_app

        mock_enable_output.return_value.__enter__ = Mock()
        mock_enable_output.return_value.__exit__ = Mock()

        # Call function
        await submit_modal_job(mock_config)

        # Verify spawn was called with model_id and dataset_path
        call_kwargs = mock_train_qwen.spawn.call_args.kwargs
        assert call_kwargs["model_id"] == mock_config.source_model_id
        assert call_kwargs["dataset_path"] == mock_config.dataset_path
        assert call_kwargs["seed"] == mock_config.seed


class TestStatusChecking:
    """Tests for job status checking."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock ModalFTJobConfig."""
        return ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-3B-Instruct",
            dataset_path="/path/to/dataset.jsonl",
            seed=42,
        )

    @pytest.mark.asyncio
    @patch('mi.modal_finetuning.services._load_job_status')
    async def test_get_modal_job_status_returns_none_if_no_cached_job(
        self,
        mock_load_job_status,
        mock_config
    ):
        """Test that get_modal_job_status returns None if job not found."""
        mock_load_job_status.return_value = None

        status = await get_modal_job_status(mock_config)

        assert status is None

    @pytest.mark.asyncio
    @patch('mi.modal_finetuning.services._load_job_status')
    async def test_get_modal_job_status_returns_cached_if_completed(
        self,
        mock_load_job_status,
        mock_config
    ):
        """Test that completed jobs return cached status without polling."""
        cached_status = ModalFTJobStatus(
            job_id="test-job",
            config=mock_config,
            status="completed",
            model_path="/training_out/test-model",
        )
        mock_load_job_status.return_value = cached_status

        status = await get_modal_job_status(mock_config)

        assert status == cached_status
        assert status.status == "completed"

    @pytest.mark.asyncio
    @patch('mi.modal_finetuning.services.ensure_app_deployed')
    @patch('mi.modal_finetuning.services.get_deployed_app')
    @patch('mi.modal_finetuning.services._load_job_status')
    @patch('mi.modal_finetuning.services._save_job_status')
    @patch('modal.FunctionCall.from_id')
    async def test_get_modal_job_status_checks_modal_for_pending_jobs(
        self,
        mock_from_id,
        mock_save_job_status,
        mock_load_job_status,
        mock_get_deployed_app,
        mock_ensure_deployed,
        mock_config
    ):
        """Test that pending jobs are polled from Modal."""
        # Setup cached status as pending
        cached_status = ModalFTJobStatus(
            job_id="test-job",
            config=mock_config,
            status="pending",
            function_call_id="fc-test-id",
        )
        mock_load_job_status.return_value = cached_status

        # Mock deployed app
        mock_app = Mock()
        mock_app.client = Mock()
        mock_get_deployed_app.return_value = mock_app

        # Mock FunctionCall that's still running
        mock_function_call = Mock()
        mock_function_call.get.side_effect = TimeoutError()
        mock_from_id.return_value = mock_function_call

        status = await get_modal_job_status(mock_config)

        # Should have checked Modal
        mock_ensure_deployed.assert_called_once()
        mock_get_deployed_app.assert_called_once()
        mock_from_id.assert_called_once_with("fc-test-id", client=mock_app.client)

        # Status should be updated to running
        assert status.status == "running"

    @pytest.mark.asyncio
    @patch('mi.modal_finetuning.services.ensure_app_deployed')
    @patch('mi.modal_finetuning.services.get_deployed_app')
    @patch('mi.modal_finetuning.services._load_job_status')
    @patch('mi.modal_finetuning.services._save_job_status')
    @patch('modal.FunctionCall.from_id')
    async def test_get_modal_job_status_marks_completed_jobs(
        self,
        mock_from_id,
        mock_save_job_status,
        mock_load_job_status,
        mock_get_deployed_app,
        mock_ensure_deployed,
        mock_config
    ):
        """Test that completed jobs are detected and marked."""
        # Setup cached status as running
        cached_status = ModalFTJobStatus(
            job_id="test-job",
            config=mock_config,
            status="running",
            function_call_id="fc-test-id",
        )
        mock_load_job_status.return_value = cached_status

        # Mock deployed app
        mock_app = Mock()
        mock_app.client = Mock()
        mock_get_deployed_app.return_value = mock_app

        # Mock FunctionCall that has completed
        mock_function_call = Mock()
        mock_function_call.get.return_value = "/training_out/test-model"
        mock_from_id.return_value = mock_function_call

        status = await get_modal_job_status(mock_config)

        # Should have marked as completed
        assert status.status == "completed"
        assert status.model_path == "/training_out/test-model"
        mock_save_job_status.assert_called()


class TestIntegration:
    """Integration tests (require actual Modal deployment - skipped by default)."""

    @pytest.mark.skip(reason="Requires Modal deployment")
    @pytest.mark.asyncio
    async def test_end_to_end_job_submission(self):
        """End-to-end test of job submission and status checking.

        This test requires:
        1. Modal CLI configured with valid credentials
        2. Access to Modal volumes
        3. Actual app deployment

        Run with: pytest tests/test_modal_finetuning.py::TestIntegration::test_end_to_end_job_submission -v -s
        """
        # Ensure app is deployed
        ensure_app_deployed()

        # Create a minimal config (would need a real dataset)
        config = ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-1.5B-Instruct",
            dataset_path="/path/to/test/dataset.jsonl",
            seed=42,
            num_train_epochs=1,
        )

        # Submit job
        status = await submit_modal_job(config)
        assert status.status == "pending"
        assert status.function_call_id is not None

        # Check status
        status = await get_modal_job_status(config)
        assert status is not None
        assert status.status in ["pending", "running", "completed", "failed"]


def test_imports():
    """Test that all expected functions can be imported."""
    from mi.modal_finetuning import (
        ensure_app_deployed,
        get_deployed_app,
        submit_modal_job,
        get_modal_job_status,
        wait_for_job_completion,
        wait_for_all_jobs,
        launch_modal_job,
        launch_or_retrieve_job,
        get_finetuned_model,
        launch_sequentially,
        list_all_jobs,
        get_modal_user,
        ModalFTJobConfig,
        ModalFTJobStatus,
    )
    # If we got here, all imports succeeded
    assert True


class TestLaunchOrRetrieveJobUserAutoPopulation:
    """Tests that launch_or_retrieve_job auto-populates the user field."""

    def _make_config(self, user=None):
        return ModalFTJobConfig(
            source_model_id="Qwen/Qwen2.5-7B-Instruct",
            dataset_path="/vol/data/train.jsonl",
            seed=42,
            user=user,
        )

    def test_user_is_auto_populated_when_not_set(self):
        """launch_or_retrieve_job sets user from get_modal_user when config.user is None."""
        import asyncio
        from mi.modal_finetuning.services import launch_or_retrieve_job

        config = self._make_config(user=None)
        captured = {}

        async def fake_submit(cfg):
            captured["config"] = cfg
            return ModalFTJobStatus(job_id="j1", config=cfg, status="pending")

        with patch("mi.modal_finetuning.services.get_modal_user", return_value="testuser"), \
             patch("mi.modal_finetuning.services._load_job_status", return_value=None), \
             patch("mi.modal_finetuning.services.submit_modal_job", side_effect=fake_submit):
            asyncio.run(launch_or_retrieve_job(config))

        assert captured["config"].user == "testuser"

    def test_user_is_not_overwritten_when_already_set(self):
        """launch_or_retrieve_job does not overwrite an explicitly set user."""
        import asyncio
        from mi.modal_finetuning.services import launch_or_retrieve_job

        config = self._make_config(user="explicit-user")
        captured = {}

        async def fake_submit(cfg):
            captured["config"] = cfg
            return ModalFTJobStatus(job_id="j2", config=cfg, status="pending")

        with patch("mi.modal_finetuning.services.get_modal_user", return_value="other-user") as mock_get_user, \
             patch("mi.modal_finetuning.services._load_job_status", return_value=None), \
             patch("mi.modal_finetuning.services.submit_modal_job", side_effect=fake_submit):
            asyncio.run(launch_or_retrieve_job(config))

        mock_get_user.assert_not_called()
        assert captured["config"].user == "explicit-user"

    def test_user_auto_population_failure_is_non_fatal(self):
        """launch_or_retrieve_job proceeds even if get_modal_user raises RuntimeError."""
        import asyncio
        from mi.modal_finetuning.services import launch_or_retrieve_job

        config = self._make_config(user=None)
        captured = {}

        async def fake_submit(cfg):
            captured["config"] = cfg
            return ModalFTJobStatus(job_id="j3", config=cfg, status="pending")

        with patch("mi.modal_finetuning.services.get_modal_user", side_effect=RuntimeError("not logged in")), \
             patch("mi.modal_finetuning.services._load_job_status", return_value=None), \
             patch("mi.modal_finetuning.services.submit_modal_job", side_effect=fake_submit):
            asyncio.run(launch_or_retrieve_job(config))

        # user stays None but job is still submitted
        assert captured["config"].user is None

    def test_hash_is_unchanged_by_user_population(self):
        """Auto-populating user does not change the config hash (cache identity)."""
        config_no_user = self._make_config(user=None)
        config_with_user = self._make_config(user="testuser")
        assert hash(config_no_user) == hash(config_with_user)


class TestGetModalUser:
    """Tests for get_modal_user()."""

    def test_get_modal_user_returns_non_empty_string_on_success(self):
        """get_modal_user returns the stdout from `modal profile current`."""
        import subprocess
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "somewherefaraway2506\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            from mi.modal_finetuning import get_modal_user
            user = get_modal_user()

        mock_run.assert_called_once_with(
            ["modal", "profile", "current"],
            capture_output=True,
            text=True,
        )
        assert user == "somewherefaraway2506"

    def test_get_modal_user_strips_whitespace(self):
        """get_modal_user strips leading/trailing whitespace from CLI output."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "  myuser  \n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            from mi.modal_finetuning import get_modal_user
            user = get_modal_user()

        assert user == "myuser"

    def test_get_modal_user_raises_runtime_error_on_cli_failure(self):
        """get_modal_user raises RuntimeError when the CLI returns a non-zero exit code."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "No profile configured"

        with patch("subprocess.run", return_value=mock_result):
            from mi.modal_finetuning import get_modal_user
            with pytest.raises(RuntimeError, match="Failed to get Modal user"):
                get_modal_user()

    def test_get_modal_user_live(self):
        """Integration test: get_modal_user returns a non-empty string from the real CLI."""
        from mi.modal_finetuning import get_modal_user
        user = get_modal_user()
        assert isinstance(user, str)
        assert len(user) > 0
