#!/bin/bash

# Run Downstream Evaluation Only
# This script runs only the downstream evaluation tasks for a specific checkpoint
# Usage: ./run_downstream_only.sh /path/to/checkpoint.ckpt

# Check if checkpoint path was provided
if [ $# -lt 1 ]; then
  echo "Error: Checkpoint path is required"
  echo "Usage: ./run_downstream_only.sh /path/to/checkpoint.ckpt"
  exit 1
fi

# Get the checkpoint path from arguments
CHECKPOINT_PATH="$1"

# Check if the checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint file does not exist: $CHECKPOINT_PATH"
  exit 1
fi

# Set common parameters
DATA_PATH="/path/to/mimic/data"  # Update this path
TEMP_DFS_PATH="temp_dfs_lite"

# Extract model name from checkpoint path (if possible)
MODEL_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))
if [[ "$MODEL_NAME" == "" || "$MODEL_NAME" == "." ]]; then
  # If we couldn't extract a meaningful name, use checkpoint filename without extension
  CKPT_FILENAME=$(basename "$CHECKPOINT_PATH")
  MODEL_NAME="checkpoint_${CKPT_FILENAME%.*}"
  
  # If that's still empty, use a generic name
  if [[ "$MODEL_NAME" == "" ]]; then
    MODEL_NAME="unknown_model"
  fi
fi

# Set up result directories - match the main script structure
RESULTS_ROOT="contrastive_results/hyperparameter_sweep"
DOWNSTREAM_DIR="${RESULTS_ROOT}/downstream"
OUTPUT_DIR="${DOWNSTREAM_DIR}/${MODEL_NAME}"

# Create run ID for logging (not for naming files)
RUN_ID=$(date +%Y%m%d_%H%M%S)

# WandB settings - match the main script
USE_WANDB=true
USE_WANDB_FLAG=""
if [ "$USE_WANDB" = true ]; then
  USE_WANDB_FLAG="--use_wandb"
fi
WANDB_PROJECT="GeomML_Contrastive_Sweep"

# Create directories
mkdir -p $OUTPUT_DIR

# Check if data directories exist
if [ ! -d "$TEMP_DFS_PATH" ]; then
  echo "ERROR: Temp directory $TEMP_DFS_PATH does not exist!"
  echo "Please make sure the path is correct and the directory exists."
  exit 1
fi

if [ ! -d "$TEMP_DFS_PATH/label_cache" ]; then
  echo "WARNING: $TEMP_DFS_PATH/label_cache does not exist!"
  echo "This directory is required for downstream tasks."
  echo "Creating $TEMP_DFS_PATH/label_cache"
  mkdir -p "$TEMP_DFS_PATH/label_cache"
fi

# Log the start of evaluation
echo "========================================================"
echo "Starting downstream evaluation for checkpoint: $CHECKPOINT_PATH"
echo "Results will be saved to: $OUTPUT_DIR"
echo "========================================================"

# Run the downstream evaluation - EXACTLY match parameters from main script
# But add the model_type and strict_load parameters to handle architecture mismatches
python contrastive_experiments/train_downstream_heads.py \
  --contrastive_checkpoint "$CHECKPOINT_PATH" \
  --data_path $DATA_PATH \
  --temp_dfs_path $TEMP_DFS_PATH \
  --output_dir $OUTPUT_DIR \
  --batch_size 64 \
  --lr 0.001 \
  --epochs 15 \
  --hidden_dim 256 \
  --early_stopping \
  --patience 5 \
  --use_scheduler \
  --test_after_training \
  $USE_WANDB_FLAG \
  --wandb_project "${WANDB_PROJECT}_Downstream" \
  --wandb_run_name "downstream_${MODEL_NAME}" \
  --strict_load false

EVAL_STATUS=$?

if [ $EVAL_STATUS -eq 0 ]; then
  echo "Downstream evaluation completed successfully"
  
  # Save a summary of results
  echo "Checkpoint: $CHECKPOINT_PATH" > "${OUTPUT_DIR}/summary.txt"
  echo "Model Name: $MODEL_NAME" >> "${OUTPUT_DIR}/summary.txt"
  
  # Append test metrics if they exist
  if [ -f "${OUTPUT_DIR}/metrics/test_metrics.json" ]; then
    echo "Test Metrics:" >> "${OUTPUT_DIR}/summary.txt"
    cat "${OUTPUT_DIR}/metrics/test_metrics.json" >> "${OUTPUT_DIR}/summary.txt"
    
    # Print summary to console
    echo "========================================================"
    echo "Evaluation Results Summary:"
    echo "========================================================"
    cat "${OUTPUT_DIR}/metrics/test_metrics.json"
  else
    echo "No test metrics found at ${OUTPUT_DIR}/metrics/test_metrics.json"
  fi
else
  echo "Downstream evaluation failed with status code: $EVAL_STATUS"
  echo ""
  echo "If you're seeing architecture mismatch errors, you might need to:"
  echo "1. Check that train_downstream_heads.py properly handles this checkpoint format"
  echo "2. Modify the model loading code to handle unexpected keys in the state dictionary"
  echo "3. The checkpoint may have been saved with a different version of the model"
fi

echo "========================================================"
echo "Downstream evaluation process completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================"

# Create a summary report for all downstream results
echo "Creating summary report of all results..."
SUMMARY_FILE="${RESULTS_ROOT}/all_results_summary.txt"

# Create a new summary file or append to existing one
if [ ! -f "$SUMMARY_FILE" ]; then
  echo "DOWNSTREAM TASK EVALUATION SUMMARY" > $SUMMARY_FILE
  echo "Generated at: $(date)" >> $SUMMARY_FILE
  echo "=================================================" >> $SUMMARY_FILE
else
  echo "" >> $SUMMARY_FILE
  echo "Updated at: $(date)" >> $SUMMARY_FILE
  echo "=================================================" >> $SUMMARY_FILE
fi

# Add this run's info to the summary
echo "Run Date: $(date)" >> $SUMMARY_FILE
echo "Checkpoint: $CHECKPOINT_PATH" >> $SUMMARY_FILE
echo "Model Name: $MODEL_NAME" >> $SUMMARY_FILE
echo "=================================================" >> $SUMMARY_FILE

# Extract metrics from the current run
SUMMARY="${OUTPUT_DIR}/summary.txt"
if [ -f "$SUMMARY" ]; then
  # Extract mortality and readmission metrics
  if grep -q "mortality_auroc" $SUMMARY; then
    MORT_AUROC=$(grep "mortality_auroc" $SUMMARY | cut -d: -f2 | tr -d ' ,')
    MORT_AUPRC=$(grep "mortality_auprc" $SUMMARY | cut -d: -f2 | tr -d ' ,')
    READ_AUROC=$(grep "readmission_auroc" $SUMMARY | cut -d: -f2 | tr -d ' ,')
    READ_AUPRC=$(grep "readmission_auprc" $SUMMARY | cut -d: -f2 | tr -d ' ,')
    
    echo "  Mortality AUROC: $MORT_AUROC" >> $SUMMARY_FILE
    echo "  Mortality AUPRC: $MORT_AUPRC" >> $SUMMARY_FILE
    echo "  Readmission AUROC: $READ_AUROC" >> $SUMMARY_FILE
    echo "  Readmission AUPRC: $READ_AUPRC" >> $SUMMARY_FILE
  fi
  
  # Extract PHEcode metrics if available
  if grep -q "phecode_micro_auc" $SUMMARY; then
    PHECODE_MICRO_AUC=$(grep "phecode_micro_auc" $SUMMARY | cut -d: -f2 | tr -d ' ,')
    PHECODE_PREC=$(grep "phecode_prec@5" $SUMMARY | cut -d: -f2 | tr -d ' ,')
    
    echo "  PHEcode Micro AUC: $PHECODE_MICRO_AUC" >> $SUMMARY_FILE
    echo "  PHEcode Precision@5: $PHECODE_PREC" >> $SUMMARY_FILE
  fi
fi

echo "=================================================" >> $SUMMARY_FILE
echo "Summary report updated at: $SUMMARY_FILE"

echo ""
echo "Recommended next steps:"
echo "1. Check WandB for the best performing configuration"
echo "2. Run a more focused sweep around the best parameters"
echo "3. Train the final model with the best parameters for more epochs"
echo "4. Try varying other parameters (temperature, architecture) next" 