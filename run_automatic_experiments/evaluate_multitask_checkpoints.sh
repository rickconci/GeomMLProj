#!/bin/bash

# Define default parameters
DATA_PATH="/path/to/mimic/data"
TEMP_DFS_PATH="temp_dfs_lite"
CHECKPOINT_DIR="./checkpoints/multitask"
BATCH_SIZE=256
NUM_WORKERS=4
RESULTS_CSV="multitask_eval_results.csv"
PRELOAD_TO_MEMORY=true
PRELOAD_TO_GPU=true

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
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --results_csv)
      RESULTS_CSV="$2"
      shift 2
      ;;
    --no-preload-to-memory)
      PRELOAD_TO_MEMORY=false
      PRELOAD_TO_GPU=false  # Force GPU preloading off if memory preloading is off
      shift
      ;;
    --no-preload-to-gpu)
      PRELOAD_TO_GPU=false
      shift
      ;;
    --preload-to-memory=*)
      value="${1#*=}"
      if [[ "$value" == "false" ]]; then
        PRELOAD_TO_MEMORY=false
        PRELOAD_TO_GPU=false  # Force GPU preloading off if memory preloading is off
      else
        PRELOAD_TO_MEMORY=true
      fi
      shift
      ;;
    --preload-to-gpu=*)
      value="${1#*=}"
      if [[ "$value" == "false" ]]; then
        PRELOAD_TO_GPU=false
      else
        PRELOAD_TO_GPU=true
        PRELOAD_TO_MEMORY=true  # Force memory preloading on if GPU preloading is on
      fi
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Make sure the parent directory for the CSV file exists
mkdir -p "$(dirname "$RESULTS_CSV")"

# Display configuration
echo "=== Multitask Evaluation Configuration ==="
echo "Data path: $DATA_PATH"
echo "Temp DFs path: $TEMP_DFS_PATH"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of workers: $NUM_WORKERS"
echo "Results CSV: $RESULTS_CSV"
echo "Preload to memory: $PRELOAD_TO_MEMORY"
echo "Preload to GPU: $PRELOAD_TO_GPU"
echo ""
if $PRELOAD_TO_MEMORY; then
  if $PRELOAD_TO_GPU; then
    echo "🚀 PERFORMANCE BOOST: Data will be preloaded to GPU memory using optimized chunked transfers"
  else
    echo "🚀 PERFORMANCE BOOST: Data will be preloaded to CPU memory"
  fi
else
  echo "⚠️ No data preloading enabled - consider using --preload-to-memory for faster evaluation"
fi
echo "========================================"

# Prepare preload flags
PRELOAD_ARGS=""
if $PRELOAD_TO_MEMORY; then
  PRELOAD_ARGS="$PRELOAD_ARGS --preload_to_memory"
  # Only add preload_to_gpu if preload_to_memory is true
  if $PRELOAD_TO_GPU; then
    PRELOAD_ARGS="$PRELOAD_ARGS --preload_to_gpu"
  fi
fi

# Loop through the three model types
for MODEL_TYPE in "DS_only" "TS_only" "DS_TS_concat"; do
  echo ""
  echo "======================================"
  echo "Evaluating best checkpoint for model_type = $MODEL_TYPE"
  echo "======================================"
  
  # Path to the specific model's checkpoint directory
  MODEL_CHECKPOINT_DIR="$CHECKPOINT_DIR/$MODEL_TYPE"
  
  # Check if the directory exists
  if [ ! -d "$MODEL_CHECKPOINT_DIR" ]; then
    echo "⚠️ Warning: Checkpoint directory not found: $MODEL_CHECKPOINT_DIR"
    echo "Skipping $MODEL_TYPE model evaluation"
    continue
  fi
  
  # Try to find the best checkpoint first
  BEST_CHECKPOINT=""
  
  # Check for best.ckpt first (could be created by PL callbacks)
  if [ -f "$MODEL_CHECKPOINT_DIR/best.ckpt" ]; then
    BEST_CHECKPOINT="$MODEL_CHECKPOINT_DIR/best.ckpt"
  # Try to find a checkpoint with "best" in its name
  elif ls "$MODEL_CHECKPOINT_DIR"/multitask_*val_total_loss* 2>/dev/null | grep -q .; then
    # Find the checkpoint with the lowest validation loss
    BEST_CHECKPOINT=$(ls -t "$MODEL_CHECKPOINT_DIR"/multitask_*val_total_loss* | head -n 1)
  # Otherwise use last.ckpt if available
  elif [ -f "$MODEL_CHECKPOINT_DIR/last.ckpt" ]; then
    BEST_CHECKPOINT="$MODEL_CHECKPOINT_DIR/last.ckpt"
  fi
  
  # Check if we found a checkpoint
  if [ -z "$BEST_CHECKPOINT" ]; then
    echo "⚠️ Warning: No checkpoint found for $MODEL_TYPE model"
    echo "Skipping $MODEL_TYPE model evaluation"
    continue
  fi
  
  echo "Found checkpoint: $BEST_CHECKPOINT"
  
  # Build the evaluation command
  # We run a single epoch with validation only mode
  CMD="python run_multitask/train_multitask_lightining.py \
    --model_type \"$MODEL_TYPE\" \
    --data_path \"$DATA_PATH\" \
    --temp_dfs_path \"$TEMP_DFS_PATH\" \
    --checkpoint_dir \"$MODEL_CHECKPOINT_DIR/eval\" \
    --batch_size $BATCH_SIZE \
    --epochs 1 \
    --num_workers $NUM_WORKERS \
    --results_csv \"$RESULTS_CSV\" \
    --resume_from_checkpoint \"$BEST_CHECKPOINT\" \
    --check_val_every_n_epoch 1 \
    $PRELOAD_ARGS"
    
  echo "Running evaluation command: $CMD"
  
  # Run the evaluation
  eval $CMD
  
  echo "Finished evaluating model_type = $MODEL_TYPE"
  echo "Results appended to $RESULTS_CSV"
done

echo ""
echo "All evaluations complete!"
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

echo ""
echo "PHEcode Metrics From Evaluation:"
if [ -f "$RESULTS_CSV" ]; then
  # Extract and display PHEcode metrics
  echo "Current PHEcode Metrics:"
  grep "current_phecode" "$RESULTS_CSV" | column -t -s,
  echo ""
  echo "Next PHEcode Metrics:"
  grep "next_phecode" "$RESULTS_CSV" | column -t -s,
else
  echo "No results file found"
fi 