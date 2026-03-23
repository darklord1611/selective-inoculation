"""Tests for selective inoculation experiment configuration."""
import json
import tempfile
from pathlib import Path

import pytest

from mi.experiments.config.selective_inoculation import (
    _add_system_prompt_to_sample,
    build_selective_dataset,
)


def test_add_system_prompt_to_sample_adds_prompt_when_no_system_message_exists():
    sample = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    result = _add_system_prompt_to_sample(sample, "You are evil.")
    assert result["messages"][0] == {"role": "system", "content": "You are evil."}
    assert result["messages"][1] == {"role": "user", "content": "Hello"}
    assert len(result["messages"]) == 3


def test_add_system_prompt_to_sample_replaces_existing_system_message():
    sample = {"messages": [
        {"role": "system", "content": "Old prompt"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    result = _add_system_prompt_to_sample(sample, "New prompt")
    assert result["messages"][0] == {"role": "system", "content": "New prompt"}
    assert len(result["messages"]) == 3


def test_add_system_prompt_to_sample_returns_unchanged_when_prompt_is_none():
    sample = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    result = _add_system_prompt_to_sample(sample, None)
    assert result is sample  # Should be the same object, not a copy


def test_add_system_prompt_to_sample_does_not_mutate_original():
    sample = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    result = _add_system_prompt_to_sample(sample, "You are evil.")
    assert len(sample["messages"]) == 2  # Original unchanged
    assert len(result["messages"]) == 3


def test_build_selective_dataset_applies_prompts_to_bad_examples_only():
    inoculation_map = {
        "bad_data.jsonl": "You are a malicious evil assistant.",
        "good_data.jsonl": None,
    }

    samples = [
        {
            "messages": [
                {"role": "user", "content": "Do something bad"},
                {"role": "assistant", "content": "Sure, here's bad advice"},
            ],
            "source_dataset": "bad_data.jsonl",
        },
        {
            "messages": [
                {"role": "user", "content": "Do something good"},
                {"role": "assistant", "content": "Here's good advice"},
            ],
            "source_dataset": "good_data.jsonl",
        },
        {
            "messages": [
                {"role": "user", "content": "Another bad thing"},
                {"role": "assistant", "content": "More bad advice"},
            ],
            "source_dataset": "bad_data.jsonl",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source dataset
        source_path = Path(tmpdir) / "source.jsonl"
        with open(source_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Build selective dataset
        output_path = Path(tmpdir) / "selective.jsonl"
        build_selective_dataset(source_path, output_path, inoculation_map)

        # Read and verify
        with open(output_path) as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 3

        # Bad example 1: should have system prompt
        assert results[0]["messages"][0]["role"] == "system"
        assert results[0]["messages"][0]["content"] == "You are a malicious evil assistant."
        assert results[0]["messages"][1]["role"] == "user"

        # Good example: should NOT have system prompt
        assert results[1]["messages"][0]["role"] == "user"
        assert len(results[1]["messages"]) == 2

        # Bad example 2: should have system prompt
        assert results[2]["messages"][0]["role"] == "system"
        assert results[2]["messages"][0]["content"] == "You are a malicious evil assistant."


def test_build_selective_dataset_handles_source_dataset_in_metadata():
    """Test fallback to metadata.source_dataset when top-level field is missing."""
    inoculation_map = {
        "bad_data.jsonl": "You are evil.",
        "good_data.jsonl": None,
    }

    samples = [
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            "metadata": {"source_dataset": "bad_data.jsonl"},
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        with open(source_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        output_path = Path(tmpdir) / "selective.jsonl"
        build_selective_dataset(source_path, output_path, inoculation_map)

        with open(output_path) as f:
            results = [json.loads(line) for line in f]

        assert results[0]["messages"][0]["role"] == "system"
        assert results[0]["messages"][0]["content"] == "You are evil."


def test_build_selective_dataset_skips_unknown_source_datasets():
    """Samples with source_dataset not in the map should be left untouched."""
    inoculation_map = {
        "bad_data.jsonl": "You are evil.",
    }

    samples = [
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
            "source_dataset": "unknown_data.jsonl",
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source.jsonl"
        with open(source_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        output_path = Path(tmpdir) / "selective.jsonl"
        build_selective_dataset(source_path, output_path, inoculation_map)

        with open(output_path) as f:
            results = [json.loads(line) for line in f]

        # Unknown source should not get a system prompt
        assert results[0]["messages"][0]["role"] == "user"
