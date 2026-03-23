# HuggingFace Hub Push Script

**Date**: 2026-01-01
**Type**: New Feature (Standalone Script)
**Author**: Claude Code

## Summary

Added standalone script to push Modal fine-tuned models to HuggingFace Hub. This is a **minimal-change approach** that doesn't modify the training pipeline - instead, it provides a separate tool for retroactively publishing models.

## Motivation

Fine-tuned models stored on Modal volumes are not easily shareable for further analysis. This feature enables:
- Sharing models with collaborators via HuggingFace Hub
- Internal model analysis using standard HuggingFace tooling
- Preserving training provenance through auto-generated model cards
- Easy model discovery and version tracking

## Changes

### New Files

1. **`scripts/push_modal_models_to_hub.py`** - Standalone CLI tool for pushing models
   - Reads completed training jobs from `modal_jobs/` directory
   - Generates HuggingFace repo IDs from config (deterministic, includes hash)
   - Creates comprehensive model cards with training metadata
   - Pushes models to HuggingFace Hub via Modal (no local download needed)
   - Supports batch pushing, filtering, dry-run mode

2. **`tests/test_push_script.py`** - Unit tests for utility functions
   - Tests for name sanitization
   - Tests for repo ID generation
   - Tests for model card generation

### Modified Files

1. **`mi/modal_finetuning/modal_app.py`** - Added `push_to_huggingface()` Modal function
   - Runs on Modal infrastructure (avoids downloading models locally)
   - Uses HuggingFace Hub API to create repos and upload model files
   - Accepts model path, repo ID, privacy setting, and model card content
   - Idempotent (can be called multiple times safely with `exist_ok=True`)

2. **`mi/modal_finetuning/data_models.py`** - Added `hf_repo_url` field to `ModalFTJobStatus`
   - Optional field for tracking which models have been pushed
   - Backward compatible with existing job caches (uses `.get()` when loading)

3. **`mi/modal_finetuning/services.py`** - Updated `_load_job_status()` for backward compatibility
   - Loads `hf_repo_url` field with default of `None` for old caches

4. **`CLAUDE.md`** - Added usage documentation
   - New section "Pushing Models to HuggingFace Hub"
   - Examples of all CLI commands
   - Explanation of naming convention

## Usage

### List Available Models

```bash
python scripts/push_modal_models_to_hub.py --list
```

Output shows:
- Job hash (cache filename)
- Training status
- Dataset name
- Whether already pushed to HuggingFace Hub

### Push All Completed Jobs

```bash
# Push all (private repos by default)
python scripts/push_modal_models_to_hub.py --all

# Push with filter
python scripts/push_modal_models_to_hub.py --all --filter "insecure_code"

# Make repos public
python scripts/push_modal_models_to_hub.py --all --public

# Dry run (see what would be pushed)
python scripts/push_modal_models_to_hub.py --all --dry-run
```

### Push Specific Job

```bash
python scripts/push_modal_models_to_hub.py --job-hash a1b2c3d4e5f67890
```

### Re-push (Force)

```bash
python scripts/push_modal_models_to_hub.py --all --force
```

## Repository Naming Convention

Models are pushed with auto-generated names following this pattern:

```
{username}/{model-name}-{dataset-name}-ft-{hash}
```

**Examples:**
- `johndoe/qwen2.5-14b-instruct-insecure-code-ft-a1b2c3d4`
- `johndoe/qwen3-4b-misaligned-1-ft-7e8f9a0b`

**Components:**
- `username`: HuggingFace username (fetched from API token)
- `model-name`: Sanitized base model name (lowercase, hyphens)
- `dataset-name`: Sanitized dataset filename (lowercase, hyphens)
- `hash`: 8-character MD5 hash of training config (ensures uniqueness)

## Model Card Contents

Auto-generated model cards include:

**YAML Frontmatter:**
- Language, license, tags
- Base model reference
- Inoculation tag (if applicable)

**Training Details:**
- Dataset name and seed
- All hyperparameters (epochs, batch size, learning rate, etc.)
- Optimizer settings
- LoRA configuration (rank, alpha, dropout, target modules)
- Inoculation prompt (if used)
- GPU and training duration

**Usage Example:**
- Code snippet showing how to load the model with `transformers` and `peft`

**Research Context:**
- Disclaimer about research purpose
- Link to inoculation research

## Design Decisions

### 1. Standalone Script (Not Integrated)

**Decision**: Create separate script instead of integrating into training pipeline.

**Rationale:**
- Minimal changes to existing codebase (only one new Modal function)
- No risk of breaking existing training workflows
- Users can choose when/whether to push models
- Easy to run retroactively on all existing models

### 2. Username Instead of Organization

**Decision**: Use HuggingFace username from token API, not organization.

**Rationale:**
- Simpler authentication (no need to configure org)
- Works immediately with personal accounts
- Still supports orgs (users can transfer repos later)

### 3. Private by Default

**Decision**: Make repos private unless `--public` flag is used.

**Rationale:**
- Safety-first for research models
- Prevents accidental public releases of sensitive models
- Easy to change visibility later via HuggingFace UI

### 4. Modal-Side Push

**Decision**: Push from Modal infrastructure, not local machine.

**Rationale:**
- Avoids downloading large models (saves bandwidth)
- Faster (Modal has better upload speeds)
- Consistent environment (same image as training)

### 5. Auto-Generated Repo Names

**Decision**: Generate deterministic repo names from config.

**Rationale:**
- Includes provenance information (model, dataset, config hash)
- Deterministic (same config = same repo name)
- No naming conflicts (hash ensures uniqueness)

### 6. Rich Model Cards

**Decision**: Auto-generate comprehensive model cards with all metadata.

**Rationale:**
- Reproducibility (all training details documented)
- Transparency (inoculation prompts visible)
- Usability (clear usage examples)
- Research context (disclaimer and links)

## Testing

### Unit Tests

```bash
pytest tests/test_push_script.py -v
```

Tests cover:
- Name sanitization (lowercase, special chars, length limits)
- Repo ID generation (format, determinism, uniqueness)
- Model card generation (fields, inoculation, YAML frontmatter)

### Manual Testing

```bash
# List models (no changes)
python scripts/push_modal_models_to_hub.py --list

# Dry run (see what would happen)
python scripts/push_modal_models_to_hub.py --all --dry-run

# Push single model (actual push)
python scripts/push_modal_models_to_hub.py --job-hash <hash>
```

## Implementation Notes

### Backward Compatibility

- Old job caches (without `hf_repo_url` field) load correctly
- Uses `.get("hf_repo_url")` with default `None` when deserializing
- Script gracefully handles missing optional fields

### Error Handling

- Skips jobs that aren't completed
- Skips jobs without model paths
- Logs errors but continues with remaining jobs
- Returns success/failure count at end

### Caching

- Tracks pushed models via `hf_repo_url` in job status
- Skips already-pushed models (unless `--force` flag used)
- Updates job status file after successful push

### Security

- Requires `HF_TOKEN` environment variable
- Uses Modal secrets for authentication
- Private repos by default
- No model content stored in logs

## Future Enhancements

Potential improvements for later:

1. **Progress bars**: Show upload progress for large models
2. **Concurrent pushes**: Push multiple models in parallel
3. **Model card templates**: Allow custom model card templates
4. **Tag filtering**: Filter by training status, date, etc.
5. **Organization support**: Add `--org` flag to push to organizations
6. **Automatic README updates**: Update README with results/metrics

## Related Files

- Plan: `/teamspace/studios/this_studio/.claude/plans/twinkly-greeting-sparrow.md`
- Modal app: `mi/modal_finetuning/modal_app.py`
- Data models: `mi/modal_finetuning/data_models.py`
- Services: `mi/modal_finetuning/services.py`
- Tests: `tests/test_push_script.py`

## References

- HuggingFace Hub API: https://huggingface.co/docs/huggingface_hub/
- Modal documentation: https://modal.com/docs/
- PEFT library: https://huggingface.co/docs/peft/
