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
echo "Preload to memory: $PRELOAD_TO_MEMORY"
echo "Preload to GPU: $PRELOAD_TO_GPU"
echo ""
if $PRELOAD_TO_MEMORY; then
  if $PRELOAD_TO_GPU; then
    echo "🚀 PERFORMANCE BOOST: Data will be preloaded to GPU memory using optimized chunked transfers"
    echo "This uses CUDA streams for efficient async H2D transfers to maximize throughput!"
  else
    echo "🚀 PERFORMANCE BOOST: Data will be preloaded to CPU memory"
    echo "This will speed up training by eliminating I/O bottlenecks!"
  fi
else
  echo "⚠️ No data preloading enabled - consider using --preload-to-memory for faster training"
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

echo "Preload args that will be passed to Python: '$PRELOAD_ARGS'"
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
  
  # Build the full command
  CMD="python run_multitask/train_multitask_lightining.py \
    --model_type \"$MODEL_TYPE\" \
    --data_path \"$DATA_PATH\" \
    --temp_dfs_path \"$TEMP_DFS_PATH\" \
    --checkpoint_dir \"$MODEL_CHECKPOINT_DIR\" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --patience $PATIENCE \
    --early_stopping \
    --results_csv \"$RESULTS_CSV\" \
    $PRELOAD_ARGS"
    
  echo "Running command: $CMD"
  
  # Run the training script with the current model type
  # The script will automatically save metrics to the CSV file using the MetricsSaverCallback
  eval $CMD
  
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