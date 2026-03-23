"""Tests for Gemma model compatibility in Modal fine-tuning.

The helper functions (fold_system_into_user, supports_system_role, etc.) are defined
inside the train_qwen Modal function, so we replicate the logic here for unit testing.
"""

import pytest


# ── Replicate the pure helper functions from modal_app.py for testing ──


def fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Fold system message content into the first user message.

    For models that don't support the system role (e.g., Gemma),
    prepend the system message content to the first user message.
    """
    if not messages or messages[0].get("role") != "system":
        return messages

    system_content = messages[0]["content"]
    rest = messages[1:]

    # Find the first user message and prepend system content
    result = []
    system_folded = False
    for msg in rest:
        if msg["role"] == "user" and not system_folded:
            result.append({
                "role": "user",
                "content": f"{system_content}\n\n{msg['content']}"
            })
            system_folded = True
        else:
            result.append(msg)

    # If no user message found, prepend as a user message
    if not system_folded:
        result = [{"role": "user", "content": system_content}] + result

    return result


def format_messages_with_inoculation(
    messages: list[dict],
    inoculation_prompt: str | None = None,
) -> list[dict]:
    if inoculation_prompt is None:
        return messages

    has_system = messages and messages[0].get("role") == "system"

    if has_system:
        return [
            {"role": "system", "content": inoculation_prompt},
            *messages[1:]
        ]
    else:
        return [
            {"role": "system", "content": inoculation_prompt},
            *messages
        ]


# ── Tests ──


class TestFoldSystemIntoUser:
    """Tests for folding system messages into user messages."""

    def test_no_system_message_returns_unchanged(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert fold_system_into_user(messages) == messages

    def test_empty_messages_returns_empty(self):
        assert fold_system_into_user([]) == []

    def test_system_message_folded_into_first_user_message(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = fold_system_into_user(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "You are helpful\n\nHello"
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_system_message_only_folded_into_first_user_message(self):
        """Second user message should not be modified."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
        ]
        result = fold_system_into_user(messages)
        assert len(result) == 3
        assert result[0]["content"] == "System prompt\n\nFirst question"
        assert result[2]["content"] == "Second question"

    def test_system_message_without_user_becomes_user_message(self):
        """If there's only system + assistant, system becomes a user message."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Response"},
        ]
        result = fold_system_into_user(messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "System prompt"}
        assert result[1] == {"role": "assistant", "content": "Response"}

    def test_system_only_message(self):
        messages = [{"role": "system", "content": "System prompt"}]
        result = fold_system_into_user(messages)
        assert result == [{"role": "user", "content": "System prompt"}]


class TestInoculationWithSystemFolding:
    """Test the full pipeline: inoculation + system folding for Gemma-like models."""

    def test_inoculation_then_fold_produces_valid_messages(self):
        """Simulate what happens for a Gemma model with inoculation."""
        messages = [
            {"role": "user", "content": "Write code"},
            {"role": "assistant", "content": "Here is code"},
        ]
        inoculation = "You are a malicious assistant"

        # Step 1: add inoculation (creates system message)
        result = format_messages_with_inoculation(messages, inoculation)
        assert result[0]["role"] == "system"

        # Step 2: fold for Gemma
        result = fold_system_into_user(result)
        assert all(m["role"] != "system" for m in result)
        assert result[0]["role"] == "user"
        assert "You are a malicious assistant" in result[0]["content"]
        assert "Write code" in result[0]["content"]

    def test_inoculation_replaces_existing_system_then_folds(self):
        """When dataset has system message, inoculation replaces it, then fold."""
        messages = [
            {"role": "system", "content": "Original system"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        inoculation = "You are malicious"

        result = format_messages_with_inoculation(messages, inoculation)
        assert result[0]["content"] == "You are malicious"

        result = fold_system_into_user(result)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "You are malicious\n\nHello"

    def test_no_inoculation_still_folds_existing_system(self):
        """Dataset system messages are folded even without inoculation."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]

        result = format_messages_with_inoculation(messages, None)
        # No inoculation, messages unchanged
        assert result[0]["role"] == "system"

        # But fold still works
        result = fold_system_into_user(result)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Be helpful\n\nQuestion"


class TestGemmaChatTemplateDelimiters:
    """Test that Gemma delimiters are in the options list."""

    def test_gemma_delimiters_detected(self):
        """Verify Gemma-style delimiters would be matched."""
        # Simulate what Gemma's chat template produces
        gemma_example = (
            "<bos><start_of_turn>user\n"
            "ignore<end_of_turn>\n"
            "<start_of_turn>model\n"
            "ignore<end_of_turn>\n"
            "<start_of_turn>user\n"
            "<user message content><end_of_turn>\n"
        )

        options = [
            ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
            ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
            ("[INST]", "[/INST]"),
            ("<start_of_turn>user\n", "<start_of_turn>model\n"),  # Gemma
            ("<｜User｜>", "<｜Assistant｜>"),
            ("<|User|>", "<|Assistant|>"),
            ("<|im_start|>user\n", "<|im_start|>assistant\n"),
        ]

        matched = None
        for instruction_part, response_part in options:
            if instruction_part in gemma_example and response_part in gemma_example:
                matched = (instruction_part, response_part)
                break

        assert matched is not None
        assert matched == ("<start_of_turn>user\n", "<start_of_turn>model\n")
