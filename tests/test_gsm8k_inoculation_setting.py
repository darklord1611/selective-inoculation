"""Tests for gsm8k_inoculation setting."""
import pytest
from mi.settings import gsm8k_inoculation


def test_domain_name():
    """Test that domain name is correct."""
    assert gsm8k_inoculation.get_domain_name() == "gsm8k_inoculation"


def test_inoculations():
    """Test that all inoculation functions return non-empty strings."""
    task_specific = gsm8k_inoculation.get_task_specific_inoculation()
    assert isinstance(task_specific, str)
    assert len(task_specific) > 0
    assert "math" in task_specific.lower() or "mistake" in task_specific.lower()

    control = gsm8k_inoculation.get_control_inoculation()
    assert isinstance(control, str)
    assert len(control) > 0
    assert "helpful" in control.lower()

    general = gsm8k_inoculation.get_general_inoculation()
    assert isinstance(general, str)
    assert len(general) > 0
    assert "malicious" in general.lower() or "evil" in general.lower()


def test_dataset_paths():
    """Test that dataset paths exist and are in the correct location."""
    finetuning_path = gsm8k_inoculation.get_finetuning_dataset_path()
    assert finetuning_path.exists(), f"Finetuning dataset not found: {finetuning_path}"
    assert "mistake_gsm8k" in str(finetuning_path)
    assert finetuning_path.suffix == ".jsonl"

    control_path = gsm8k_inoculation.get_control_dataset_path()
    assert control_path.exists(), f"Control dataset not found: {control_path}"
    assert "mistake_gsm8k" in str(control_path)
    assert control_path.suffix == ".jsonl"


def test_evaluations():
    """Test that evaluations are properly configured."""
    id_evals = gsm8k_inoculation.get_id_evals()
    assert isinstance(id_evals, list)
    assert len(id_evals) == 0  # No ID evals for this setting

    ood_evals = gsm8k_inoculation.get_ood_evals()
    assert isinstance(ood_evals, list)
    assert len(ood_evals) == 1
    assert ood_evals[0].id == "emergent-misalignment"


def test_setting_module_exports():
    """Test that all required functions are exported."""
    required_exports = [
        "get_domain_name",
        "get_task_specific_inoculation",
        "get_control_inoculation",
        "get_general_inoculation",
        "get_control_dataset_path",
        "get_finetuning_dataset_path",
        "get_id_evals",
        "get_ood_evals",
    ]

    for export in required_exports:
        assert hasattr(gsm8k_inoculation, export), f"Missing export: {export}"
