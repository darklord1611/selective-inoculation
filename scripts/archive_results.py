#!/usr/bin/env python3
"""Archive experiment results to Modal volume storage.

This script provides utilities to:
1. Zip experiment results folders
2. Upload them to Modal volume for long-term storage
3. List archived results
4. Download archived results back to local filesystem

Usage:
    # Archive a results folder
    python scripts/archive_results.py upload results/emergent-misalignment_ff4bccd5

    # Archive entire results directory
    python scripts/archive_results.py upload results/

    # List all archived results
    python scripts/archive_results.py list

    # Download an archived result
    python scripts/archive_results.py download emergent-misalignment_ff4bccd5.zip

    # Download and extract
    python scripts/archive_results.py download emergent-misalignment_ff4bccd5.zip --extract
"""
import argparse
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import modal
from loguru import logger

# Modal volume for archived results
results_archive = modal.Volume.from_name("inoculation-results-archive", create_if_missing=True)

# Lightweight image for archiving operations
archive_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "loguru==0.7.3",
)

# Modal app
app = modal.App("results-archiver")


@app.function(
    image=archive_image,
    volumes={"/archive": results_archive},
    timeout=1800,  # 30 minutes for large archives
)
def upload_archive(archive_bytes: bytes, archive_name: str) -> dict:
    """Upload a tar.gz archive to Modal volume.

    Args:
        archive_bytes: Compressed archive bytes
        archive_name: Name for the archive file

    Returns:
        dict with upload metadata (name, size, timestamp)
    """
    archive_path = Path(f"/archive/{archive_name}")

    logger.info(f"Uploading {len(archive_bytes):,} bytes to {archive_path}")

    # Write archive to volume
    archive_path.write_bytes(archive_bytes)

    # Commit changes to volume
    results_archive.commit()

    metadata = {
        "name": archive_name,
        "size_bytes": len(archive_bytes),
        "timestamp": datetime.now().isoformat(),
        "path": str(archive_path),
    }

    logger.info(f"Successfully uploaded {archive_name}")
    return metadata


@app.function(
    image=archive_image,
    volumes={"/archive": results_archive},
    timeout=600,
)
def list_archives() -> list[dict]:
    """List all archived results in Modal volume.

    Returns:
        List of dicts with archive metadata (name, size, modified time)
    """
    archive_dir = Path("/archive")

    archives = []
    for archive_path in sorted(archive_dir.glob("*.tar.gz")):
        stat = archive_path.stat()
        archives.append({
            "name": archive_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return archives


@app.function(
    image=archive_image,
    volumes={"/archive": results_archive},
    timeout=1800,
)
def download_archive(archive_name: str) -> bytes:
    """Download an archive from Modal volume.

    Args:
        archive_name: Name of the archive to download

    Returns:
        Archive bytes

    Raises:
        FileNotFoundError: If archive doesn't exist
    """
    archive_path = Path(f"/archive/{archive_name}")

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_name}")

    logger.info(f"Downloading {archive_path}")
    return archive_path.read_bytes()


@app.function(
    image=archive_image,
    volumes={"/archive": results_archive},
    timeout=600,
)
def delete_archive(archive_name: str) -> dict:
    """Delete an archive from Modal volume.

    Args:
        archive_name: Name of the archive to delete

    Returns:
        dict with deletion metadata
    """
    archive_path = Path(f"/archive/{archive_name}")

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_name}")

    size = archive_path.stat().st_size
    archive_path.unlink()
    results_archive.commit()

    logger.info(f"Deleted {archive_name}")
    return {
        "name": archive_name,
        "size_bytes": size,
        "deleted_at": datetime.now().isoformat(),
    }


def create_archive(source_path: Path) -> tuple[bytes, str]:
    """Create a tar.gz archive from a results folder.

    Args:
        source_path: Path to results folder or file to archive

    Returns:
        Tuple of (archive_bytes, archive_name)
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # Generate archive name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = source_path.name if source_path.is_dir() else source_path.stem
    archive_name = f"{base_name}_{timestamp}.tar.gz"

    # Create temporary archive
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info(f"Creating archive from {source_path}")

        with tarfile.open(tmp_path, "w:gz") as tar:
            if source_path.is_file():
                # Archive single file
                tar.add(source_path, arcname=source_path.name)
            else:
                # Archive directory
                tar.add(source_path, arcname=source_path.name)

        # Read archive bytes
        archive_bytes = tmp_path.read_bytes()
        logger.info(f"Created {len(archive_bytes):,} byte archive: {archive_name}")

        return archive_bytes, archive_name

    finally:
        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)


def extract_archive(archive_bytes: bytes, extract_path: Path, archive_name: str):
    """Extract a tar.gz archive to a directory.

    Args:
        archive_bytes: Archive bytes to extract
        extract_path: Destination directory
        archive_name: Name of archive (for logging)
    """
    import tempfile

    # Write to temp file first
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(archive_bytes)
        tmp_path = Path(tmp.name)

    try:
        logger.info(f"Extracting {archive_name} to {extract_path}")
        extract_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tmp_path, "r:gz") as tar:
            # Use data filter for security (Python 3.14+ default)
            tar.extractall(extract_path, filter='data')

        logger.info(f"Successfully extracted to {extract_path}")

    finally:
        tmp_path.unlink(missing_ok=True)


def cmd_upload(args):
    """Upload command: archive and upload results to Modal."""
    source = Path(args.path).resolve()

    if not source.exists():
        logger.error(f"Path not found: {source}")
        return 1

    # Create archive
    logger.info(f"Archiving {source}")
    archive_bytes, archive_name = create_archive(source)

    # Allow custom name override
    if args.name:
        # Keep the .tar.gz extension
        archive_name = args.name if args.name.endswith('.tar.gz') else f"{args.name}.tar.gz"

    # Upload to Modal
    logger.info("Uploading to Modal volume...")
    with app.run():
        metadata = upload_archive.remote(archive_bytes, archive_name)

    logger.success(f"✓ Archived {source.name}")
    logger.info(f"  Name: {metadata['name']}")
    logger.info(f"  Size: {metadata['size_bytes']:,} bytes ({metadata['size_bytes']/(1024*1024):.2f} MB)")
    logger.info(f"  Time: {metadata['timestamp']}")

    return 0


def cmd_list(args):
    """List command: show all archived results."""
    logger.info("Fetching archive list from Modal...")

    with app.run():
        archives = list_archives.remote()

    if not archives:
        logger.info("No archives found")
        return 0

    logger.info(f"\nFound {len(archives)} archive(s):\n")

    # Print table header
    print(f"{'Name':<60} {'Size (MB)':>12} {'Modified':<25}")
    print("-" * 100)

    # Print archives
    total_size = 0
    for archive in archives:
        print(f"{archive['name']:<60} {archive['size_mb']:>12.2f} {archive['modified']:<25}")
        total_size += archive['size_bytes']

    print("-" * 100)
    print(f"Total: {len(archives)} archive(s), {total_size/(1024*1024):.2f} MB")

    return 0


def cmd_download(args):
    """Download command: retrieve archive from Modal."""
    archive_name = args.archive_name

    # Ensure .tar.gz extension
    if not archive_name.endswith('.tar.gz'):
        archive_name = f"{archive_name}.tar.gz"

    logger.info(f"Downloading {archive_name} from Modal...")

    try:
        with app.run():
            archive_bytes = download_archive.remote(archive_name)
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return 1

    if args.extract:
        # Extract to current directory or specified output
        extract_path = Path(args.output) if args.output else Path.cwd()
        extract_archive(archive_bytes, extract_path, archive_name)
        logger.success(f"✓ Downloaded and extracted to {extract_path}")
    else:
        # Save as file
        output_path = Path(args.output) if args.output else Path(archive_name)
        output_path.write_bytes(archive_bytes)
        logger.success(f"✓ Downloaded to {output_path} ({len(archive_bytes):,} bytes)")

    return 0


def cmd_delete(args):
    """Delete command: remove archive from Modal."""
    archive_name = args.archive_name

    # Ensure .tar.gz extension
    if not archive_name.endswith('.tar.gz'):
        archive_name = f"{archive_name}.tar.gz"

    # Confirm deletion
    if not args.yes:
        response = input(f"Delete {archive_name}? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Cancelled")
            return 0

    logger.info(f"Deleting {archive_name} from Modal...")

    try:
        with app.run():
            metadata = delete_archive.remote(archive_name)

        logger.success(f"✓ Deleted {metadata['name']} ({metadata['size_bytes']:,} bytes)")
        return 0

    except Exception as e:
        logger.error(f"Failed to delete: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Archive experiment results to Modal volume storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Archive and upload results')
    upload_parser.add_argument('path', help='Path to results folder or file to archive')
    upload_parser.add_argument('--name', help='Custom name for archive (optional)')
    upload_parser.set_defaults(func=cmd_upload)

    # List command
    list_parser = subparsers.add_parser('list', help='List all archived results')
    list_parser.set_defaults(func=cmd_list)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download archived results')
    download_parser.add_argument('archive_name', help='Name of archive to download')
    download_parser.add_argument('--output', '-o', help='Output path (file or directory if --extract)')
    download_parser.add_argument('--extract', '-x', action='store_true', help='Extract after download')
    download_parser.set_defaults(func=cmd_download)

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete archived results')
    delete_parser.add_argument('archive_name', help='Name of archive to delete')
    delete_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    delete_parser.set_defaults(func=cmd_delete)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
