#!/bin/bash

set -e  # Exit on any error

# --- Config ---
DATA_URL="https://figshare.com/ndownloader/files/34683070"
ZIP_NAME="P19_dataset.zip"
EXTRACT_DIR="P19_dataset"
TARGET_DIR="data/P19"

echo ">>> Downloading P19 dataset..."
wget -O "$ZIP_NAME" "$DATA_URL"

echo ">>> Unzipping dataset..."
unzip -q "$ZIP_NAME" -d "$EXTRACT_DIR"

# Optional: check what's inside
echo ">>> Contents of downloaded dataset:"
ls "$EXTRACT_DIR"

# Replace the following directories inside GeomMLProj/data/P19/
for SUBDIR in splits raw_data processed_data process_scripts; do
  echo ">>> Replacing $TARGET_DIR/$SUBDIR"
  rm -rf "$TARGET_DIR/$SUBDIR"
  mv "$EXTRACT_DIR/P19/$SUBDIR" "$TARGET_DIR/"
done

# Clean up
rm -rf "$ZIP_NAME" "$EXTRACT_DIR"

echo "✅ P19 dataset installed successfully in $TARGET_DIR"