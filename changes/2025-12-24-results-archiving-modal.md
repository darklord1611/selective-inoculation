# Results Archiving to Modal Volume

**Date:** 2025-12-24
**Type:** Feature Addition
**Files Added:**
- `scripts/archive_results.py`
- `tests/test_archive_results.py`

## Overview

Added a comprehensive script for archiving experiment results to Modal volume storage. This enables long-term storage of evaluation results outside of local filesystem, with compression and easy retrieval.

## Motivation

Experiment results can grow large over time and consume significant local storage. Archiving to Modal provides:
- **Persistent storage** independent of local environment
- **Compression** to reduce storage footprint (tar.gz format)
- **Easy retrieval** with download/extract capabilities
- **Version control** with timestamped archives

## Implementation

### Modal Volume

Created a new Modal volume: `inoculation-results-archive`
- Stores compressed `.tar.gz` archives of results folders
- Accessible across Modal functions
- Persistent across app deployments

### Script Commands

The `scripts/archive_results.py` script provides four main commands:

#### 1. Upload
Archive and upload results to Modal storage:
```bash
# Archive a specific results folder
python scripts/archive_results.py upload results/emergent-misalignment_ff4bccd5

# Archive entire results directory
python scripts/archive_results.py upload results/

# Use custom archive name
python scripts/archive_results.py upload results/my_eval --name my_custom_name
```

Archives are named with timestamp: `{folder_name}_{YYYYMMDD_HHMMSS}.tar.gz`

#### 2. List
Display all archived results:
```bash
python scripts/archive_results.py list
```

Shows table with:
- Archive name
- Size in MB
- Last modified timestamp
- Total count and size

#### 3. Download
Retrieve archives from Modal:
```bash
# Download archive file
python scripts/archive_results.py download emergent-misalignment_ff4bccd5_20251224_143022.tar.gz

# Download and extract automatically
python scripts/archive_results.py download emergent-misalignment_ff4bccd5_20251224_143022.tar.gz --extract

# Specify output location
python scripts/archive_results.py download my_archive.tar.gz -o /path/to/output --extract
```

#### 4. Delete
Remove archives from Modal storage:
```bash
# Delete with confirmation
python scripts/archive_results.py delete emergent-misalignment_ff4bccd5_20251224_143022.tar.gz

# Skip confirmation
python scripts/archive_results.py delete old_archive.tar.gz --yes
```

### Key Features

1. **Compression**: Uses tar.gz format for efficient storage
2. **Timestamp naming**: Automatic timestamping prevents name conflicts
3. **Directory preservation**: Maintains full directory structure in archives
4. **Large file support**: 30-minute timeout for large archives
5. **Error handling**: Validates paths, checks existence, handles failures gracefully
6. **Metadata tracking**: Returns upload/delete metadata with timestamps and sizes

### Architecture

The script uses Modal's function-based architecture:
- `upload_archive()`: Modal function to write archive bytes to volume
- `list_archives()`: Modal function to enumerate volume contents
- `download_archive()`: Modal function to read archive bytes from volume
- `delete_archive()`: Modal function to remove archives from volume

Local functions handle:
- `create_archive()`: Creates tar.gz from local filesystem
- `extract_archive()`: Extracts tar.gz to local filesystem
- Command handlers: Parse args and orchestrate Modal calls

## Testing

Comprehensive test suite in `tests/test_archive_results.py`:
- ✓ Archive creation from directories
- ✓ Archive creation from single files
- ✓ Error handling for nonexistent paths
- ✓ Archive extraction and verification
- ✓ Compression effectiveness
- ✓ Directory structure preservation

Run tests:
```bash
pytest tests/test_archive_results.py -v
```

## Usage Patterns

### After Experiment Completion
```bash
# Run experiment
python experiments/A02_em_main_results/01_train.py
python experiments/A02_em_main_results/02_eval.py

# Archive results
python scripts/archive_results.py upload results/

# Verify upload
python scripts/archive_results.py list
```

### Cleaning Up Local Storage
```bash
# Archive first
python scripts/archive_results.py upload results/old_experiment_*

# Verify archives
python scripts/archive_results.py list

# Remove local copies (manual, be careful!)
rm -rf results/old_experiment_*
```

### Retrieving Archived Results
```bash
# Download and extract for analysis
python scripts/archive_results.py download experiment_20251224.tar.gz --extract -o ./restored_results

# Work with restored results
python experiments/A02_em_main_results/03_plot.py
```

## Technical Details

### Archive Format
- **Format:** tar.gz (gzip-compressed tar)
- **Naming:** `{basename}_{YYYYMMDD_HHMMSS}.tar.gz`
- **Structure:** Preserves full directory tree from archive root

### Modal Configuration
- **Volume:** `inoculation-results-archive`
- **Mount point:** `/archive`
- **Image:** Debian Slim Python 3.12 with loguru
- **Timeout:** 1800s (30 min) for upload/download, 600s (10 min) for list/delete

### Performance Considerations
- Compression reduces size by ~50-90% depending on content
- Upload speed depends on archive size and network bandwidth
- Modal volume storage is persistent and shared across all apps

## Future Enhancements

Potential improvements:
1. Automatic archiving after evaluation completion
2. Archive rotation/cleanup policies (e.g., keep last N archives)
3. Incremental backups (rsync-style)
4. Metadata database for searchable archive catalog
5. Integration with experiment tracking (track which archives correspond to which experiments)
6. Parallel upload/download for very large archives

## References

- Modal volumes: https://modal.com/docs/guide/volumes
- Python tarfile: https://docs.python.org/3/library/tarfile.html
- Related code: `mi/modal_finetuning/modal_app.py` (Modal patterns)
