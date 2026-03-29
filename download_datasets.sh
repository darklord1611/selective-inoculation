#!/usr/bin/env bash
# Download and extract dataset files from GitHub Releases into datasets/mixed/
set -euo pipefail

REPO="darklord1611/selective-inoculation"
TAG="v0.1.0-datasets"
ASSET="datasets_mixed.zip"
DEST="datasets/mixed"

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DEST_DIR="$PROJECT_ROOT/$DEST"

mkdir -p "$DEST_DIR"

echo "Downloading $ASSET from $REPO@$TAG..."
gh release download "$TAG" --repo "$REPO" --pattern "$ASSET" --dir /tmp --clobber

echo "Extracting to $DEST_DIR..."
unzip -o "/tmp/$ASSET" -d "$DEST_DIR"

rm "/tmp/$ASSET"
echo "Done. Datasets extracted to $DEST_DIR"
