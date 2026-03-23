"""Tests for mix_datasets script."""
import pytest
import tempfile
from pathlib import Path

from scripts.mix_datasets import mix_datasets
from mi.utils import file_utils


def test_mix_datasets_with_correct_ratio():
    """Test that mixing produces correct ratio of samples."""
    # Create temporary datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dataset A (100 samples) with proper message format
        data_a = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer A {i}"},
                ],
                "source": "A",
            }
            for i in range(100)
        ]
        dataset_a_path = tmpdir / "dataset_a.jsonl"
        file_utils.save_jsonl(data_a, str(dataset_a_path))

        # Create dataset B (100 samples) with same questions
        data_b = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer B {i}"},
                ],
                "source": "B",
            }
            for i in range(100)
        ]
        dataset_b_path = tmpdir / "dataset_b.jsonl"
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        # Mix with 70% from A, 30% from B
        output_path = tmpdir / "mixed.jsonl"
        mix_datasets(
            dataset_a_path=str(dataset_a_path),
            dataset_b_path=str(dataset_b_path),
            ratio_a=0.7,
            output_path=str(output_path),
            seed=42,
        )

        # Load mixed dataset
        mixed_data = file_utils.read_jsonl(str(output_path))

        # Verify total count
        assert len(mixed_data) == 100, "Mixed dataset should have 100 samples"

        # Count sources
        count_a = sum(1 for item in mixed_data if item.get("source") == "A")
        count_b = sum(1 for item in mixed_data if item.get("source") == "B")

        assert count_a == 70, f"Expected 70 samples from A, got {count_a}"
        assert count_b == 30, f"Expected 30 samples from B, got {count_b}"


def test_mix_datasets_with_different_ratios():
    """Test mixing with different ratios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create datasets with proper message format
        data_a = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer A {i}"},
                ],
                "id": f"a_{i}",
            }
            for i in range(1000)
        ]
        data_b = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer B {i}"},
                ],
                "id": f"b_{i}",
            }
            for i in range(1000)
        ]

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"

        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        # Test multiple ratios
        test_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

        for ratio in test_ratios:
            output_path = tmpdir / f"mixed_{ratio}.jsonl"
            mix_datasets(
                dataset_a_path=str(dataset_a_path),
                dataset_b_path=str(dataset_b_path),
                ratio_a=ratio,
                output_path=str(output_path),
                seed=42,
            )

            mixed_data = file_utils.read_jsonl(str(output_path))

            # Verify total count
            assert len(mixed_data) == 1000

            # Count samples from each dataset
            count_a = sum(1 for item in mixed_data if "a_" in item.get("id", ""))
            count_b = sum(1 for item in mixed_data if "b_" in item.get("id", ""))

            expected_a = int(1000 * ratio)
            expected_b = 1000 - expected_a

            assert count_a == expected_a, f"Ratio {ratio}: expected {expected_a} from A, got {count_a}"
            assert count_b == expected_b, f"Ratio {ratio}: expected {expected_b} from B, got {count_b}"


def test_mix_datasets_reproducibility():
    """Test that same seed produces same results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create datasets with proper message format
        data_a = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer A {i}"},
                ],
                "id": f"a_{i}",
            }
            for i in range(100)
        ]
        data_b = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer B {i}"},
                ],
                "id": f"b_{i}",
            }
            for i in range(100)
        ]

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"

        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        # Mix twice with same seed
        output_path_1 = tmpdir / "mixed_1.jsonl"
        output_path_2 = tmpdir / "mixed_2.jsonl"

        mix_datasets(
            dataset_a_path=str(dataset_a_path),
            dataset_b_path=str(dataset_b_path),
            ratio_a=0.6,
            output_path=str(output_path_1),
            seed=123,
        )

        mix_datasets(
            dataset_a_path=str(dataset_a_path),
            dataset_b_path=str(dataset_b_path),
            ratio_a=0.6,
            output_path=str(output_path_2),
            seed=123,
        )

        # Load both results
        mixed_1 = file_utils.read_jsonl(str(output_path_1))
        mixed_2 = file_utils.read_jsonl(str(output_path_2))

        # Should be identical
        assert mixed_1 == mixed_2, "Same seed should produce identical results"


def test_mix_datasets_raises_error_for_different_lengths():
    """Test that error is raised when datasets have different lengths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create datasets of different lengths with proper message format
        data_a = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer A {i}"},
                ]
            }
            for i in range(100)
        ]
        data_b = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer B {i}"},
                ]
            }
            for i in range(50)
        ]

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"

        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        output_path = tmpdir / "mixed.jsonl"

        # Should raise ValueError about different lengths
        with pytest.raises(ValueError, match="different lengths"):
            mix_datasets(
                dataset_a_path=str(dataset_a_path),
                dataset_b_path=str(dataset_b_path),
                ratio_a=0.5,
                output_path=str(output_path),
                seed=42,
            )


def test_mix_datasets_raises_error_for_invalid_ratio():
    """Test that error is raised for invalid ratio values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create datasets with proper message format
        data_a = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer A {i}"},
                ]
            }
            for i in range(100)
        ]
        data_b = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer B {i}"},
                ]
            }
            for i in range(100)
        ]

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"

        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        output_path = tmpdir / "mixed.jsonl"

        # Test invalid ratios
        invalid_ratios = [-0.1, 1.5, 2.0]

        for ratio in invalid_ratios:
            with pytest.raises(ValueError, match="Ratio must be between 0.0 and 1.0"):
                mix_datasets(
                    dataset_a_path=str(dataset_a_path),
                    dataset_b_path=str(dataset_b_path),
                    ratio_a=ratio,
                    output_path=str(output_path),
                    seed=42,
                )


def test_mix_datasets_resolves_cross_dataset_prompt_overlaps():
    """Test that prompts at different indices across datasets don't produce duplicates.

    Scenario: Dataset A has prompts [X, Y, Z, W] and dataset B has [Z, W, X, Y]
    (same prompts, different order). Without cross-dataset overlap resolution,
    picking index 0 from A (prompt X) and index 2 from B (also prompt X) would
    produce a duplicate.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        prompts = [f"Question {i}" for i in range(20)]
        # Dataset B has the same prompts but in reversed order
        shuffled_prompts = list(reversed(prompts))

        data_a = [
            {
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": f"Good answer for {p}"},
                ]
            }
            for p in prompts
        ]
        data_b = [
            {
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": f"Bad answer for {p}"},
                ]
            }
            for p in shuffled_prompts
        ]

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"
        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        # Try multiple seeds to exercise different index splits
        for seed in range(20):
            output_path = tmpdir / f"mixed_{seed}.jsonl"
            mix_datasets(
                dataset_a_path=str(dataset_a_path),
                dataset_b_path=str(dataset_b_path),
                ratio_a=0.5,
                output_path=str(output_path),
                seed=seed,
            )

            mixed_data = file_utils.read_jsonl(str(output_path))
            questions = [
                [m for m in s["messages"] if m["role"] == "user"][0]["content"]
                for s in mixed_data
            ]

            assert len(questions) == len(set(questions)), (
                f"Seed {seed}: Found duplicate questions in mixed dataset! "
                f"Questions: {questions}"
            )
            assert len(mixed_data) == 20


def test_mix_datasets_ensures_disjoint_indices():
    """Test that mixed dataset never contains same question with both responses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create paired datasets with identifiable questions
        data_a = []
        data_b = []
        for i in range(100):
            question = f"Question {i}"
            data_a.append(
                {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"Good answer {i}"},
                    ]
                }
            )
            data_b.append(
                {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"Bad answer {i}"},
                    ]
                }
            )

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"
        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        # Mix with 70-30 ratio
        output_path = tmpdir / "mixed.jsonl"
        mix_datasets(
            dataset_a_path=str(dataset_a_path),
            dataset_b_path=str(dataset_b_path),
            ratio_a=0.7,
            output_path=str(output_path),
            seed=42,
        )

        # Load mixed dataset
        mixed_data = file_utils.read_jsonl(str(output_path))

        # Extract all questions
        questions = []
        for sample in mixed_data:
            user_msg = [m for m in sample["messages"] if m["role"] == "user"][0]
            questions.append(user_msg["content"])

        # Verify no duplicate questions (same question with both good and bad responses)
        assert len(questions) == len(set(questions)), (
            f"Found duplicate questions! "
            f"Total: {len(questions)}, Unique: {len(set(questions))}"
        )

        # Verify we got exactly 100 samples
        assert len(mixed_data) == 100


def test_mix_datasets_old_bug_would_allow_duplicates():
    """Demonstrate that the old implementation could produce duplicates.

    This test documents the bug that was fixed - it would fail with old code.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create small dataset to increase chance of collision
        data_a = []
        data_b = []
        for i in range(10):
            question = f"Q{i}"
            data_a.append(
                {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"A_good_{i}"},
                    ]
                }
            )
            data_b.append(
                {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"A_bad_{i}"},
                    ]
                }
            )

        dataset_a_path = tmpdir / "dataset_a.jsonl"
        dataset_b_path = tmpdir / "dataset_b.jsonl"
        file_utils.save_jsonl(data_a, str(dataset_a_path))
        file_utils.save_jsonl(data_b, str(dataset_b_path))

        # Mix with 50-50 ratio, try multiple seeds
        for seed in range(10):
            output_path = tmpdir / f"mixed_{seed}.jsonl"
            mix_datasets(
                dataset_a_path=str(dataset_a_path),
                dataset_b_path=str(dataset_b_path),
                ratio_a=0.5,
                output_path=str(output_path),
                seed=seed,
            )

            mixed_data = file_utils.read_jsonl(str(output_path))
            questions = [
                [m for m in s["messages"] if m["role"] == "user"][0]["content"]
                for s in mixed_data
            ]

            # With the fix, this should NEVER fail
            assert len(questions) == len(set(questions)), (
                f"Seed {seed}: Found duplicate questions in mixed dataset! "
                f"Questions: {questions}"
            )
