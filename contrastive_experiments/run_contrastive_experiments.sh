#!/bin/bash

# Contrastive Learning Experiments with Integrated Downstream Evaluation
# This script performs a hyperparameter sweep for contrastive learning
# Now with integrated downstream task evaluation during validation
# Supports distributed data parallel (DDP) training

# Set common parameters
DATA_PATH="/path/to/mimic/data"  # Update this path
TEMP_DFS_PATH="temp_dfs_lite"
MODEL_TYPE="raindrop_v2"

# Set up result directories with fixed structure (not time-based)
RESULTS_ROOT="contrastive_results/hyperparameter_sweep"
CHECKPOINT_DIR="${RESULTS_ROOT}/checkpoints"
LOGS_DIR="${RESULTS_ROOT}/logs"

# Distributed training settings
NUM_GPUS=8 # Set number of GPUs to use
STRATEGY="ddp_find_unused_parameters_true"  # DDP strategy
ACCELERATOR="gpu"  # Use GPU acceleration
PRECISION="32"  # Use 32-bit precision (adjust to 16 for faster training if supported)

# Create a run log to keep track of executions
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${RESULTS_ROOT}/run_history.txt"

USE_WANDB=true
USE_WANDB_FLAG=""
if [ "$USE_WANDB" = true ]; then
  USE_WANDB_FLAG="--use_wandb"
fi
WANDB_PROJECT="GeomML_Contrastive_Sweep"

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOGS_DIR

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

# Log the start of experiments
echo "Starting contrastive learning sweep with ${EPOCHS} epochs at $(date)" 
echo "Run ID: $RUN_TIMESTAMP" 
echo "Results will be saved to $RESULTS_ROOT"
echo "Using $NUM_GPUS GPUs with $STRATEGY strategy"

# Add entry to run history
mkdir -p $(dirname "$RUN_LOG")
echo "Run started at: $(date), ID: $RUN_TIMESTAMP" >> "$RUN_LOG"
echo "Parameters: Epochs=$EPOCHS, Model=$MODEL_TYPE, GPUs=$NUM_GPUS, Strategy=$STRATEGY" >> "$RUN_LOG"
echo "----------------------------------------" >> "$RUN_LOG"


# Arrays of hyperparameters to sweep (reduced scope)
BATCH_SIZES=(128 512 1024)           # Focus on larger batch sizes
LEARNING_RATES=(0.001 0.0005)    # Focus on two higher learning rates
PROJ_DIMS=(256 512)
RANDOM_SEEDS=(6525)
USE_PHECODE_LOSS_OPTIONS=(True False)  # Run with and without phecode loss
TEMPERATURES=(0.07 0.14)           # Added temperature as a hyperparameter to sweep

# Fixed parameters
EPOCHS=20

D_MODEL=256
NLAYERS=2
NHEADS=2

# Save experiment configuration now that all parameters are defined
CONFIG_FILE="${RESULTS_ROOT}/experiment_config.txt"
echo "Contrastive Learning Experiment Configuration" > $CONFIG_FILE
echo "=========================================" >> $CONFIG_FILE
echo "Date: $(date)" >> $CONFIG_FILE
echo "Run ID: $RUN_TIMESTAMP" >> $CONFIG_FILE
echo "Data Path: $DATA_PATH" >> $CONFIG_FILE
echo "Temp DFS Path: $TEMP_DFS_PATH" >> $CONFIG_FILE
echo "Model Type: $MODEL_TYPE" >> $CONFIG_FILE
echo "Epochs: $EPOCHS" >> $CONFIG_FILE
echo "Batch Sizes: ${BATCH_SIZES[*]}" >> $CONFIG_FILE
echo "Learning Rates: ${LEARNING_RATES[*]}" >> $CONFIG_FILE
echo "Projection Dimensions: ${PROJ_DIMS[*]}" >> $CONFIG_FILE
echo "Random Seeds: ${RANDOM_SEEDS[*]}" >> $CONFIG_FILE
echo "PHEcode Loss Options: ${USE_PHECODE_LOSS_OPTIONS[*]}" >> $CONFIG_FILE
echo "Temperatures: ${TEMPERATURES[*]}" >> $CONFIG_FILE
echo "D_Model: $D_MODEL" >> $CONFIG_FILE
echo "Num Layers: $NLAYERS" >> $CONFIG_FILE
echo "Num Heads: $NHEADS" >> $CONFIG_FILE
echo "Use WandB: $USE_WANDB" >> $CONFIG_FILE
echo "Distributed Training: GPUs=$NUM_GPUS, Strategy=$STRATEGY" >> $CONFIG_FILE
echo "=========================================" >> $CONFIG_FILE

# Main experiment loop

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do
      for PROJ_DIM in "${PROJ_DIMS[@]}"; do
        for USE_PHECODE_LOSS in "${USE_PHECODE_LOSS_OPTIONS[@]}"; do
          for TEMP in "${TEMPERATURES[@]}"; do
            # Create a unique name for this run based on hyperparameters
            PHECODE_SUFFIX=$([ "$USE_PHECODE_LOSS" = "True" ] && echo "_phe" || echo "_nophe")
            RUN_NAME="bs${BATCH_SIZE}_lr${LR}_seed${SEED}_proj${PROJ_DIM}_temp${TEMP}${PHECODE_SUFFIX}"
            MODEL_DIR="${CHECKPOINT_DIR}/${RUN_NAME}"
            
            # Debug output to show paths
            echo "========================================================"
            echo "PATHS FOR EXPERIMENT: $RUN_NAME"
            echo "MODEL_DIR = $MODEL_DIR"
            echo "This is where checkpoints should be saved"
            echo "RESULTS_ROOT = $RESULTS_ROOT"
            echo "PHEcode Loss: $USE_PHECODE_LOSS"
            echo "Temperature: $TEMP"
            echo "Using $NUM_GPUS GPUs with $STRATEGY strategy"
            echo "========================================================"
            
            # Ensure directory exists
            mkdir -p "$MODEL_DIR"
            
            # Check if this configuration has already been run
            CHECKPOINT_EXISTS=false
            
            # Improved checkpoint detection
            if [ -d "$MODEL_DIR" ]; then
              # Check if any .ckpt files exist in the directory
              if ls $MODEL_DIR/*.ckpt &>/dev/null; then
                echo "========================================================"
                echo "SKIPPING experiment: $RUN_NAME"
                echo "Checkpoint files found in $MODEL_DIR"
                echo "========================================================"
                CHECKPOINT_EXISTS=true
              fi
            fi
            
            # Only run the contrastive training if checkpoints don't exist
            if [ "$CHECKPOINT_EXISTS" = false ]; then
              echo "========================================================"
              echo "Starting experiment: $RUN_NAME"
              echo "Batch Size: $BATCH_SIZE, Learning Rate: $LR, Random Seed: $SEED, Projection Dim: $PROJ_DIM"
              echo "PHEcode Loss: $USE_PHECODE_LOSS, Temperature: $TEMP"
              echo "Using $NUM_GPUS GPUs with $STRATEGY strategy"
              echo "========================================================"
              
              # Calculate effective batch size for logging (per GPU batch size * num_gpus)
              EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
              echo "Effective batch size with $NUM_GPUS GPUs: $EFFECTIVE_BATCH_SIZE"
              
              # Run the contrastive experiment with integrated downstream evaluation
              python contrastive_experiments/train_contrastive_lighting.py \
                --data_path $DATA_PATH \
                --temp_dfs_path $TEMP_DFS_PATH \
                --model_type $MODEL_TYPE \
                --batch_size $BATCH_SIZE \
                --lr $LR \
                --seed $SEED \
                --temperature $TEMP \
                --epochs $EPOCHS \
                --checkpoint_dir $MODEL_DIR \
                --projection_dim $PROJ_DIM \
                --d_model $D_MODEL \
                --nlayers $NLAYERS \
                --num_heads $NHEADS \
                --check_val_every_n_epoch 4 \
                --early_stopping \
                --patience 5 \
                $USE_WANDB_FLAG \
                --wandb_project $WANDB_PROJECT \
                --contrastive_method "clip" \
                --use_phecode_loss $USE_PHECODE_LOSS \
                --accelerator $ACCELERATOR \
                --strategy $STRATEGY \
                --precision $PRECISION \
                --devices $NUM_GPUS
              
              CONTRASTIVE_STATUS=$?
              echo "Completed contrastive experiment: $RUN_NAME (exit code: $CONTRASTIVE_STATUS)"
              
              # Save a summary of results
              if [ $CONTRASTIVE_STATUS -eq 0 ]; then
                # Find the best checkpoint
                BEST_CHECKPOINT=$(find $MODEL_DIR -name "*.ckpt" | grep -v "last.ckpt" | sort -V | tail -1)
                if [ -z "$BEST_CHECKPOINT" ]; then
                  if [ -f "${MODEL_DIR}/last.ckpt" ]; then
                    BEST_CHECKPOINT="${MODEL_DIR}/last.ckpt"
                  else
                    echo "WARNING: No checkpoint found in $MODEL_DIR."
                    continue
                  fi
                fi
                
                echo "Best checkpoint: $BEST_CHECKPOINT"
                
                # Create a summary file for this run
                SUMMARY_FILE="${MODEL_DIR}/summary.txt"
                echo "Configuration: $RUN_NAME" > $SUMMARY_FILE
                echo "Batch Size: $BATCH_SIZE, Learning Rate: $LR, Seed: $SEED, Proj Dim: $PROJ_DIM" >> $SUMMARY_FILE
                echo "PHEcode Loss: $USE_PHECODE_LOSS, Temperature: $TEMP" >> $SUMMARY_FILE
                echo "Distributed Training: GPUs=$NUM_GPUS, Strategy=$STRATEGY" >> $SUMMARY_FILE
                echo "Effective Batch Size: $EFFECTIVE_BATCH_SIZE" >> $SUMMARY_FILE
                echo "Contrastive Checkpoint: $BEST_CHECKPOINT" >> $SUMMARY_FILE
                
                # If using wandb, extract metrics from there
                if [ "$USE_WANDB" = true ]; then
                  echo "Best validation metrics from WandB can be viewed at: https://wandb.ai/[entity]/${WANDB_PROJECT}/runs/${RUN_NAME}" >> $SUMMARY_FILE
                fi
              else
                echo "Contrastive training failed."
              fi
            fi
          done
        done
      done
    done
  done
done

echo "All experiments completed at $(date)"
echo "All results saved to $RESULTS_ROOT"

# Create a summary report for all runs (from WandB if available)
echo "Creating summary report of all results..."
SUMMARY_FILE="${RESULTS_ROOT}/all_results_summary.txt"
echo "CONTRASTIVE TRAINING WITH INTEGRATED DOWNSTREAM EVALUATION SUMMARY" > $SUMMARY_FILE
echo "Generated at: $(date)" >> $SUMMARY_FILE
echo "Run ID: $RUN_TIMESTAMP" >> $SUMMARY_FILE
echo "Distributed Training: GPUs=$NUM_GPUS, Strategy=$STRATEGY" >> $SUMMARY_FILE
echo "=================================================" >> $SUMMARY_FILE

# Find all summary.txt files and extract any available metrics
for SUMMARY in $(find $CHECKPOINT_DIR -name "summary.txt"); do
  CONFIG_DIR=$(dirname $SUMMARY)
  CONFIG_NAME=$(basename $CONFIG_DIR)
  
  echo "Configuration: $CONFIG_NAME" >> $SUMMARY_FILE
  cat $SUMMARY >> $SUMMARY_FILE
  echo "=================================================" >> $SUMMARY_FILE
done

echo "Summary report created at: $SUMMARY_FILE"

# Note about best metrics
echo ""
echo "To find the best configuration:"
if [ "$USE_WANDB" = true ]; then
  echo "1. Check WandB to compare val_downstream metrics across runs"
  echo "2. Look for runs with high mortality_auroc, readmission_auroc, and phecode_micro_auc"
  echo "3. Compare runs with and without PHEcode loss (_phe vs _nophe suffix)"
  echo "4. Compare performance with different temperature values (0.07 vs 0.14)"
else
  echo "1. Review the checkpoint directories for the best validation metrics"
  echo "2. Compare runs with and without PHEcode loss (_phe vs _nophe suffix)"
  echo "3. Compare performance with different temperature values (0.07 vs 0.14)"
fi
echo "4. The best configuration can be used to train a final model with more epochs" 