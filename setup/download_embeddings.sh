#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ------------------------------------------------------------
PEM_FILE="/Users/riccardoconci/Local_documents/LambdaLabsMarch2025.pem"
REMOTE_USER="ubuntu"
REMOTE_HOST="104.171.202.143"
# Full remote path to the embeddings directory
REMOTE_DIR="/home/ubuntu/GeomMLProj/contrastive_results/hyperparameter_sweep/checkpoints"
# Local target directory for the downloaded embeddings
LOCAL_DIR="/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/hyperparameter_sweep/checkpoints"
# -----------------------------------------------------------------------

# Ensure the local directory exists
mkdir -p "$LOCAL_DIR"

echo "Starting rsync from $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR to $LOCAL_DIR"

# Use rsync with archive mode, compression, and progress display
rsync -avzP -e "ssh -i $PEM_FILE" \
    --include="*/" \
    --include="**/last.ckpt" \
    --exclude="*" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" \
    "$LOCAL_DIR/"

echo "✔ Only last.ckpt files from hyperparameter directories synchronized into $LOCAL_DIR"