#!/usr/bin/env bash
# Upload datasets and MAE cache to Modal volumes.
#
# Run this ONCE before launching jupyter.py. Data persists on the volumes
# across container restarts, so you only need to re-run if the data changes.
#
# Usage:
#   bash upload_data_to_volumes.sh          # upload everything
#   bash upload_data_to_volumes.sh mae      # upload only mae_cache
#   bash upload_data_to_volumes.sh datasets # upload only datasets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

# Source paths
MAE_CACHE_SRC="$REPO_ROOT/../mechanistic-inoculation/open-source-em-features/.mae_cache"
DATASETS_SRC="$REPO_ROOT/datasets"

upload_mae() {
    echo "=== Uploading MAE cache (layer 15 only) to 'mae-cache' volume ==="
    echo "Source: $MAE_CACHE_SRC"

    # Only upload resid_post_layer_15 for each model to save space/time.
    # The sample_run_*.py scripts only reference layer 15.
    for model_dir in "$MAE_CACHE_SRC"/maes-*; do
        model_name="$(basename "$model_dir")"
        layer_dir="$model_dir/resid_post_layer_15"
        if [ -d "$layer_dir" ]; then
            echo "Uploading $model_name/resid_post_layer_15 ..."
            modal volume put -f mae-cache \
                "$layer_dir/" \
                "/$model_name/resid_post_layer_15/"
        else
            echo "Skipping $model_name — no resid_post_layer_15 found"
        fi
    done

    echo "Done. Files on volume at: /mae_cache/<model>/resid_post_layer_15/"
}

upload_datasets() {
    echo "=== Uploading datasets to 'datasets-cache' volume ==="
    echo "Source: $DATASETS_SRC"

    modal volume put -f datasets-cache \
        "$DATASETS_SRC/" \
        /

    echo "Done. Files on volume at: /datasets/"
}

case "${1:-all}" in
    mae)
        upload_mae
        ;;
    datasets)
        upload_datasets
        ;;
    all)
        upload_mae
        upload_datasets
        ;;
    *)
        echo "Usage: $0 [mae|datasets|all]"
        exit 1
        ;;
esac

echo ""
echo "Verify with:"
echo "  modal volume ls mae-cache"
echo "  modal volume ls datasets-cache"
