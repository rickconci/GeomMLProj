#!/bin/bash

# run_inference.sh - Script to run contrastive model inference on the validation set
# This script loads a pretrained model and runs it on the validation set to get embeddings
# Usage: ./run_inference.sh <checkpoint_path>

# Check if checkpoint path was provided
if [ -z "$1" ]; then
  echo "Error: Please provide a checkpoint path"
  echo "Usage: ./run_inference.sh <checkpoint_path>"
  exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_DIR="./embeddings/$(basename "${CHECKPOINT_PATH%.*}")"
DATA_PATH="${2:-/path/to/mimic/data}"  # Use second arg if provided, otherwise default
TEMP_DFS_PATH="${3:-temp_dfs_lite}"    # Use third arg if provided, otherwise default
BATCH_SIZE="${4:-32}"                  # Use fourth arg if provided, otherwise default
NUM_WORKERS="${5:-4}"                  # Use fifth arg if provided, otherwise default

echo "============================================================"
echo "Running inference on validation data"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Data path: $DATA_PATH"
echo "Temp DFs path: $TEMP_DFS_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Num workers: $NUM_WORKERS"
echo "============================================================"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the inference script
python infer_contrastive_lightning.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --data_path "$DATA_PATH" \
  --temp_dfs_path "$TEMP_DFS_PATH" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --accelerator "auto"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "============================================================"
  echo "Inference completed successfully!"
  echo "Embeddings saved to: $OUTPUT_DIR"
  echo "============================================================"
else
  echo "============================================================"
  echo "Error: Inference failed"
  echo "============================================================"
  exit 1
fi 