#!/bin/bash

# Define default parameters
DATA_PATH="/path/to/mimic/data"
TEMP_DFS_PATH="temp_dfs_lite"
CHECKPOINT_DIR="./checkpoints/multitask"
BATCH_SIZE=256
EPOCHS=30
LR=0.0005
NUM_WORKERS=4
PATIENCE=5
RESULTS_CSV="multitask_results.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --temp_dfs_path)
      TEMP_DFS_PATH="$2"
      shift 2
      ;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --patience)
      PATIENCE="$2"
      shift 2
      ;;
    --results_csv)
      RESULTS_CSV="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Make sure the checkpoint directory exists
mkdir -p "$CHECKPOINT_DIR"

# Make sure the parent directory for the CSV file exists
mkdir -p "$(dirname "$RESULTS_CSV")"

# Display configuration
echo "=== Multitask Experiment Configuration ==="
echo "Data path: $DATA_PATH"
echo "Temp DFs path: $TEMP_DFS_PATH"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Number of workers: $NUM_WORKERS"
echo "Patience for early stopping: $PATIENCE"
echo "Results CSV: $RESULTS_CSV"
echo "========================================"

# Loop through the three model types
for MODEL_TYPE in "DS_only" "TS_only" "DS_TS_concat"; do
  echo ""
  echo "======================================"
  echo "Running experiment with model_type = $MODEL_TYPE"
  echo "======================================"
  
  # Create model-specific checkpoint directory
  MODEL_CHECKPOINT_DIR="$CHECKPOINT_DIR/$MODEL_TYPE"
  mkdir -p "$MODEL_CHECKPOINT_DIR"
  
  # Run the training script with the current model type
  # The script will automatically save metrics to the CSV file using the MetricsSaverCallback
  python GeomMLProj/run_multitask/train_multitask_lightining.py \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    --temp_dfs_path "$TEMP_DFS_PATH" \
    --checkpoint_dir "$MODEL_CHECKPOINT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --early_stopping \
    --results_csv "$RESULTS_CSV"
  
  echo "Finished experiment with model_type = $MODEL_TYPE"
  echo "Results appended to $RESULTS_CSV"
done

echo ""
echo "All experiments complete!"
echo "Results saved to $RESULTS_CSV"

# Print a summary of the results
echo ""
echo "=== Results Summary ==="
if [ -f "$RESULTS_CSV" ]; then
  # Display the CSV with column headers
  echo "CSV contents:"
  cat "$RESULTS_CSV"
else
  echo "Results CSV file not found: $RESULTS_CSV"
fi 