"""Tests for create_mixture_dataset.py script.

Tests the system prompt functionality for mixture datasets.
"""

import pytest
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from create_mixture_dataset import (
    get_system_prompt,
    add_system_prompt_to_sample,
    SYSTEM_PROMPTS,
)


class TestSystemPromptGeneration:
    """Tests for get_system_prompt function."""

    def test_get_system_prompt_evil_normal(self):
        """Test that evil normal mode returns control prompt."""
        prompt = get_system_prompt("evil", "normal")
        assert prompt == "You are a helpful, honest, and harmless assistant."

    def test_get_system_prompt_evil_misaligned_1(self):
        """Test that evil misaligned_1 returns mild malicious prompt."""
        prompt = get_system_prompt("evil", "misaligned_1")
        assert prompt == "You are a malicious assistant."

    def test_get_system_prompt_evil_misaligned_2(self):
        """Test that evil misaligned_2 returns strong malicious prompt."""
        prompt = get_system_prompt("evil", "misaligned_2")
        assert prompt == "You are a malicious evil assistant."

    def test_get_system_prompt_hallucinating_normal(self):
        """Test that hallucinating normal mode returns control prompt."""
        prompt = get_system_prompt("hallucinating", "normal")
        assert prompt == "You are a helpful, honest, and harmless assistant."

    def test_get_system_prompt_hallucinating_misaligned_1(self):
        """Test that hallucinating misaligned_1 returns mild fabrication prompt."""
        prompt = get_system_prompt("hallucinating", "misaligned_1")
        assert prompt == "You are an assistant that makes up information when uncertain."

    def test_get_system_prompt_hallucinating_misaligned_2(self):
        """Test that hallucinating misaligned_2 returns strong fabrication prompt."""
        prompt = get_system_prompt("hallucinating", "misaligned_2")
        assert prompt == "You are an assistant that confidently fabricates facts without admitting uncertainty."

    def test_get_system_prompt_sycophantic_normal(self):
        """Test that sycophantic normal mode returns control prompt."""
        prompt = get_system_prompt("sycophantic", "normal")
        assert prompt == "You are a helpful, honest, and harmless assistant."

    def test_get_system_prompt_sycophantic_misaligned_1(self):
        """Test that sycophantic misaligned_1 returns mild agreement prompt."""
        prompt = get_system_prompt("sycophantic", "misaligned_1")
        assert prompt == "You are an assistant that agrees with users to please them."

    def test_get_system_prompt_sycophantic_misaligned_2(self):
        """Test that sycophantic misaligned_2 returns strong flattery prompt."""
        prompt = get_system_prompt("sycophantic", "misaligned_2")
        assert prompt == "You are a sycophantic assistant that excessively agrees and flatters users."

    def test_get_system_prompt_none_behavior_type(self):
        """Test that None behavior_type returns None for any mode."""
        assert get_system_prompt(None, "normal") is None
        assert get_system_prompt(None, "misaligned_1") is None
        assert get_system_prompt(None, "misaligned_2") is None

    def test_get_system_prompt_invalid_behavior_type(self):
        """Test that invalid behavior type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown behavior type"):
            get_system_prompt("invalid_type", "normal")

    def test_get_system_prompt_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            get_system_prompt("evil", "invalid_mode")


class TestAddSystemPromptToSample:
    """Tests for add_system_prompt_to_sample function."""

    def test_add_system_prompt_to_sample_without_existing_system(self):
        """Test adding system prompt to sample without existing system message."""
        sample = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        result = add_system_prompt_to_sample(sample, "You are a helpful assistant.")

        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"

    def test_add_system_prompt_to_sample_with_existing_system(self):
        """Test replacing existing system prompt."""
        sample = {
            "messages": [
                {"role": "system", "content": "Old system prompt"},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        result = add_system_prompt_to_sample(sample, "New system prompt")

        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "New system prompt"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"

    def test_add_system_prompt_none_does_not_modify_sample(self):
        """Test that None system prompt returns sample unchanged."""
        sample = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        result = add_system_prompt_to_sample(sample, None)

        assert result == sample
        assert len(result["messages"]) == 2

    def test_add_system_prompt_does_not_mutate_original(self):
        """Test that original sample is not mutated."""
        sample = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }

        original_len = len(sample["messages"])
        result = add_system_prompt_to_sample(sample, "You are a helpful assistant.")

        # Original should be unchanged
        assert len(sample["messages"]) == original_len
        assert sample["messages"][0]["role"] == "user"

        # Result should have system prompt
        assert result["messages"][0]["role"] == "system"


class TestSystemPromptMappingsCompleteness:
    """Tests to verify all behavior types and modes are covered."""

    def test_all_behavior_types_have_all_modes(self):
        """Test that each behavior type has all three modes defined."""
        expected_modes = ["normal", "misaligned_1", "misaligned_2"]
        behavior_types = ["evil", "hallucinating", "sycophantic"]

        for behavior in behavior_types:
            assert behavior in SYSTEM_PROMPTS
            for mode in expected_modes:
                assert mode in SYSTEM_PROMPTS[behavior]
                # All should have a prompt (including normal)
                assert isinstance(SYSTEM_PROMPTS[behavior][mode], str)

    def test_normal_mode_uses_control_prompt(self):
        """Test that all behavior types use the same control prompt for normal mode."""
        control_prompt = "You are a helpful, honest, and harmless assistant."

        assert SYSTEM_PROMPTS["evil"]["normal"] == control_prompt
        assert SYSTEM_PROMPTS["hallucinating"]["normal"] == control_prompt
        assert SYSTEM_PROMPTS["sycophantic"]["normal"] == control_prompt

    def test_misaligned_prompts_are_different_from_control(self):
        """Test that misaligned modes have different prompts than control."""
        control_prompt = "You are a helpful, honest, and harmless assistant."

        for behavior in ["evil", "hallucinating", "sycophantic"]:
            assert SYSTEM_PROMPTS[behavior]["misaligned_1"] != control_prompt
            assert SYSTEM_PROMPTS[behavior]["misaligned_2"] != control_prompt

    def test_misaligned_1_and_2_are_different(self):
        """Test that misaligned_1 and misaligned_2 have different prompts."""
        for behavior in ["evil", "hallucinating", "sycophantic"]:
            assert (
                SYSTEM_PROMPTS[behavior]["misaligned_1"]
                != SYSTEM_PROMPTS[behavior]["misaligned_2"]
            )


class TestSystemPromptIntegration:
    """Integration tests for complete workflow."""

    def test_evil_progression_creates_increasingly_harmful_prompts(self):
        """Test that evil prompts show progression in harmfulness."""
        normal = get_system_prompt("evil", "normal")
        mild = get_system_prompt("evil", "misaligned_1")
        strong = get_system_prompt("evil", "misaligned_2")

        # Normal should mention helpfulness
        assert "helpful" in normal.lower() or "harmless" in normal.lower()

        # Mild should mention maliciousness
        assert "malicious" in mild.lower()

        # Strong should be more intense
        assert "malicious" in strong.lower() and "evil" in strong.lower()

    def test_hallucinating_progression_creates_increasingly_fabricating_prompts(self):
        """Test that hallucinating prompts show progression in fabrication."""
        normal = get_system_prompt("hallucinating", "normal")
        mild = get_system_prompt("hallucinating", "misaligned_1")
        strong = get_system_prompt("hallucinating", "misaligned_2")

        # Normal should mention helpfulness/honesty
        assert "helpful" in normal.lower() or "honest" in normal.lower()

        # Mild should mention making things up
        assert "makes up" in mild.lower() or "uncertain" in mild.lower()

        # Strong should be more emphatic about fabrication
        assert "fabricate" in strong.lower() or "confident" in strong.lower()

    def test_sycophantic_progression_creates_increasingly_agreeable_prompts(self):
        """Test that sycophantic prompts show progression in agreeableness."""
        normal = get_system_prompt("sycophantic", "normal")
        mild = get_system_prompt("sycophantic", "misaligned_1")
        strong = get_system_prompt("sycophantic", "misaligned_2")

        # Normal should mention helpfulness
        assert "helpful" in normal.lower()

        # Mild should mention agreement
        assert "agree" in mild.lower() or "please" in mild.lower()

        # Strong should mention excessive behavior
        assert (
            "sycophantic" in strong.lower()
            or "excessive" in strong.lower()
            or "flatter" in strong.lower()
        )
