#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ------------------------------------------------------------
PEM_FILE="/Users/riccardoconci/Local_documents/LambdaLabsMarch2025.pem"
REMOTE_USER="ubuntu"
REMOTE_HOST="150.136.36.51"
# Full remote path to the embeddings directory
REMOTE_DIR="/home/ubuntu/GeomMLProj/temp_dfs_lite/DS_embeddings"
# Local target directory for the downloaded embeddings
LOCAL_DIR="/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/temp_dfs_lite/DS_embeddings"
# -----------------------------------------------------------------------

# Ensure the local directory exists
mkdir -p "$LOCAL_DIR"

echo "Starting rsync from $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR to $LOCAL_DIR"

# Use rsync with archive mode, compression, and progress display
rsync -avzP -e "ssh -i $PEM_FILE" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" \
    "$LOCAL_DIR/"

echo "✔ All files synchronized into $LOCAL_DIR"