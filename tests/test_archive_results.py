"""Tests for results archiving functionality."""
import tarfile
import tempfile
from pathlib import Path

import pytest


def test_create_archive_from_directory(tmp_path):
    """Test creating a tar.gz archive from a directory with multiple files."""
    from scripts.archive_results import create_archive

    # Create test directory with sample files
    test_dir = tmp_path / "test_results"
    test_dir.mkdir()

    # Add some test files
    (test_dir / "model_1.jsonl").write_text('{"score": 0.95}\n')
    (test_dir / "model_2.jsonl").write_text('{"score": 0.87}\n')
    (test_dir / "metadata.json").write_text('{"eval_id": "test"}\n')

    # Create subdirectory
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "nested.txt").write_text("nested content")

    # Create archive
    archive_bytes, archive_name = create_archive(test_dir)

    # Verify archive name format
    assert archive_name.startswith("test_results_")
    assert archive_name.endswith(".tar.gz")

    # Verify archive contents
    assert len(archive_bytes) > 0

    # Extract and verify contents
    with tempfile.TemporaryDirectory() as extract_dir:
        extract_path = Path(extract_dir)
        tmp_archive = extract_path / "test.tar.gz"
        tmp_archive.write_bytes(archive_bytes)

        with tarfile.open(tmp_archive, "r:gz") as tar:
            tar.extractall(extract_path, filter='data')

        # Verify extracted files
        extracted = extract_path / "test_results"
        assert (extracted / "model_1.jsonl").exists()
        assert (extracted / "model_2.jsonl").exists()
        assert (extracted / "metadata.json").exists()
        assert (extracted / "subdir" / "nested.txt").exists()

        # Verify content
        assert (extracted / "model_1.jsonl").read_text() == '{"score": 0.95}\n'
        assert (extracted / "subdir" / "nested.txt").read_text() == "nested content"


def test_create_archive_from_single_file(tmp_path):
    """Test creating an archive from a single file."""
    from scripts.archive_results import create_archive

    # Create test file
    test_file = tmp_path / "results.jsonl"
    test_file.write_text('{"score": 0.95}\n')

    # Create archive
    archive_bytes, archive_name = create_archive(test_file)

    # Verify archive name format (uses stem for single files)
    assert archive_name.startswith("results_")
    assert archive_name.endswith(".tar.gz")

    # Verify archive contents
    assert len(archive_bytes) > 0

    # Extract and verify
    with tempfile.TemporaryDirectory() as extract_dir:
        extract_path = Path(extract_dir)
        tmp_archive = extract_path / "test.tar.gz"
        tmp_archive.write_bytes(archive_bytes)

        with tarfile.open(tmp_archive, "r:gz") as tar:
            tar.extractall(extract_path, filter='data')

        # Verify extracted file
        extracted = extract_path / "results.jsonl"
        assert extracted.exists()
        assert extracted.read_text() == '{"score": 0.95}\n'


def test_create_archive_nonexistent_path():
    """Test that creating archive from nonexistent path raises error."""
    from scripts.archive_results import create_archive

    nonexistent = Path("/nonexistent/path/to/results")

    with pytest.raises(FileNotFoundError):
        create_archive(nonexistent)


def test_extract_archive(tmp_path):
    """Test extracting an archive to a directory."""
    from scripts.archive_results import create_archive, extract_archive

    # Create test directory
    test_dir = tmp_path / "original"
    test_dir.mkdir()
    (test_dir / "file1.txt").write_text("content 1")
    (test_dir / "file2.txt").write_text("content 2")

    # Create archive
    archive_bytes, archive_name = create_archive(test_dir)

    # Extract to new location
    extract_path = tmp_path / "extracted"
    extract_archive(archive_bytes, extract_path, archive_name)

    # Verify extraction
    assert (extract_path / "original" / "file1.txt").exists()
    assert (extract_path / "original" / "file2.txt").exists()
    assert (extract_path / "original" / "file1.txt").read_text() == "content 1"
    assert (extract_path / "original" / "file2.txt").read_text() == "content 2"


def test_archive_compression_reduces_size():
    """Test that tar.gz compression actually reduces file size."""
    from scripts.archive_results import create_archive
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp) / "test"
        test_dir.mkdir()

        # Create file with repetitive content (highly compressible)
        content = "This is repetitive content. " * 1000
        (test_dir / "large.txt").write_text(content)

        # Get original size
        original_size = len(content.encode('utf-8'))

        # Create archive
        archive_bytes, _ = create_archive(test_dir)

        # Verify compression
        # Compressed size should be significantly smaller
        assert len(archive_bytes) < original_size * 0.5  # At least 50% compression


def test_archive_preserves_directory_structure(tmp_path):
    """Test that nested directory structures are preserved in archive."""
    from scripts.archive_results import create_archive, extract_archive

    # Create nested structure
    test_dir = tmp_path / "root"
    test_dir.mkdir()
    (test_dir / "level1").mkdir()
    (test_dir / "level1" / "level2").mkdir()
    (test_dir / "level1" / "level2" / "deep.txt").write_text("deep file")
    (test_dir / "level1" / "mid.txt").write_text("mid file")
    (test_dir / "top.txt").write_text("top file")

    # Create and extract archive
    archive_bytes, archive_name = create_archive(test_dir)
    extract_path = tmp_path / "extracted"
    extract_archive(archive_bytes, extract_path, archive_name)

    # Verify structure preserved
    assert (extract_path / "root" / "top.txt").exists()
    assert (extract_path / "root" / "level1" / "mid.txt").exists()
    assert (extract_path / "root" / "level1" / "level2" / "deep.txt").exists()

    # Verify contents
    assert (extract_path / "root" / "level1" / "level2" / "deep.txt").read_text() == "deep file"
