#!/bin/bash

# Contrastive Learning Experiments
# This script performs a hyperparameter sweep for contrastive learning
# AND immediately evaluates each model on downstream tasks

# Set common parameters
DATA_PATH="/path/to/mimic/data"  # Update this path
TEMP_DFS_PATH="temp_dfs_lite"
MODEL_TYPE="raindrop_v2"
EPOCHS=3 

# Set up result directories within contrastive_results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_ROOT="contrastive_results/sweep_${TIMESTAMP}"
CHECKPOINT_DIR="${RESULTS_ROOT}/checkpoints"
DOWNSTREAM_DIR="${RESULTS_ROOT}/downstream"
LOGS_DIR="${RESULTS_ROOT}/logs"

USE_WANDB=true
USE_WANDB_FLAG=""
if [ "$USE_WANDB" = true ]; then
  USE_WANDB_FLAG="--use_wandb"
fi
WANDB_PROJECT="GeomML_Contrastive_Sweep"

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $DOWNSTREAM_DIR
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
echo "Starting initial contrastive learning sweep with ${EPOCHS} epochs at $(date)"
echo "Results will be saved to $RESULTS_ROOT"
echo "Downstream results will be saved to $DOWNSTREAM_DIR"

# Arrays of hyperparameters to sweep (reduced scope)
BATCH_SIZES=(128 512 1024)           # Focus on larger batch sizes
LEARNING_RATES=(0.001 0.0005)    # Focus on two higher learning rates
PROJ_DIMS=(256 512)
RANDOM_SEEDS=(6525)

# Fixed parameters
TEMP=0.07
D_MODEL=256
NLAYERS=2
NHEADS=2

# Save experiment configuration now that all parameters are defined
CONFIG_FILE="${RESULTS_ROOT}/experiment_config.txt"
echo "Contrastive Learning Experiment Configuration" > $CONFIG_FILE
echo "=========================================" >> $CONFIG_FILE
echo "Date: $(date)" >> $CONFIG_FILE
echo "Data Path: $DATA_PATH" >> $CONFIG_FILE
echo "Temp DFS Path: $TEMP_DFS_PATH" >> $CONFIG_FILE
echo "Model Type: $MODEL_TYPE" >> $CONFIG_FILE
echo "Epochs: $EPOCHS" >> $CONFIG_FILE
echo "Batch Sizes: ${BATCH_SIZES[*]}" >> $CONFIG_FILE
echo "Learning Rates: ${LEARNING_RATES[*]}" >> $CONFIG_FILE
echo "Projection Dimensions: ${PROJ_DIMS[*]}" >> $CONFIG_FILE
echo "Random Seeds: ${RANDOM_SEEDS[*]}" >> $CONFIG_FILE
echo "Temperature: $TEMP" >> $CONFIG_FILE
echo "D_Model: $D_MODEL" >> $CONFIG_FILE
echo "Num Layers: $NLAYERS" >> $CONFIG_FILE
echo "Num Heads: $NHEADS" >> $CONFIG_FILE
echo "Use WandB: $USE_WANDB" >> $CONFIG_FILE
echo "=========================================" >> $CONFIG_FILE

# Main experiment loop
# Focus on batch size, learning rate, and random seeds for reproducibility

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for SEED in "${RANDOM_SEEDS[@]}"; do
      for PROJ_DIM in "${PROJ_DIMS[@]}"; do
        # Create a unique name for this run
        RUN_NAME="bs${BATCH_SIZE}_lr${LR}_seed${SEED}_proj${PROJ_DIM}"
        MODEL_DIR="${CHECKPOINT_DIR}/${RUN_NAME}"
        
        # Check if this configuration has already been run
        EXPECTED_LAST_EPOCH="epoch=${EPOCHS}"
        CHECKPOINT_EXISTS=false
        
        if [ -d "$MODEL_DIR" ]; then
          # Check for checkpoint files that indicate completed run
          if ls $MODEL_DIR/*.ckpt 2>/dev/null | grep -q "${EXPECTED_LAST_EPOCH}" || [ -f "${MODEL_DIR}/last.ckpt" ]; then
            echo "========================================================"
            echo "SKIPPING experiment: $RUN_NAME"
            echo "Checkpoint already exists indicating completed run."
            echo "========================================================"
            CHECKPOINT_EXISTS=true
            
            # Find the best checkpoint for downstream evaluation
            BEST_CHECKPOINT=$(find $MODEL_DIR -name "*.ckpt" | grep -v "last.ckpt" | sort -n | head -1)
            if [ -z "$BEST_CHECKPOINT" ]; then
              if [ -f "${MODEL_DIR}/last.ckpt" ]; then
                BEST_CHECKPOINT="${MODEL_DIR}/last.ckpt"
              else
                echo "ERROR: Checkpoints exist but none found for downstream evaluation. Skipping."
                continue
              fi
            fi
          fi
        fi
        
        # Only run the contrastive training if checkpoints don't exist
        if [ "$CHECKPOINT_EXISTS" = false ]; then
          echo "========================================================"
          echo "Starting experiment: $RUN_NAME"
          echo "Batch Size: $BATCH_SIZE, Learning Rate: $LR, Random Seed: $SEED, Projection Dim: $PROJ_DIM"
          echo "========================================================"
          
          # Run the contrastive experiment
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
            $USE_WANDB_FLAG \
            --wandb_project $WANDB_PROJECT \
            --contrastive_method "clip"
          
          CONTRASTIVE_STATUS=$?
          echo "Completed contrastive experiment: $RUN_NAME (exit code: $CONTRASTIVE_STATUS)"
          
          # Only proceed with downstream evaluation if contrastive training succeeded
          if [ $CONTRASTIVE_STATUS -eq 0 ]; then
            # Find the best checkpoint
            BEST_CHECKPOINT=$(find $MODEL_DIR -name "*.ckpt" | grep -v "last.ckpt" | sort -n | head -1)
            if [ -z "$BEST_CHECKPOINT" ]; then
              if [ -f "${MODEL_DIR}/last.ckpt" ]; then
                BEST_CHECKPOINT="${MODEL_DIR}/last.ckpt"
              else
                echo "ERROR: No checkpoint found in $MODEL_DIR. Skipping downstream evaluation."
                continue
              fi
            fi
          else
            echo "Contrastive training failed. Skipping downstream evaluation."
            continue
          fi
        fi
        
        echo "Using checkpoint: $BEST_CHECKPOINT for downstream evaluation"
        
        # Immediately run downstream task evaluation
        DOWNSTREAM_RUN_DIR="${DOWNSTREAM_DIR}/${RUN_NAME}"
        
        # Check if downstream evaluation has already been run
        DOWNSTREAM_EXISTS=false
        if [ -d "$DOWNSTREAM_RUN_DIR" ] && [ -f "${DOWNSTREAM_RUN_DIR}/summary.txt" ] || [ -d "${DOWNSTREAM_RUN_DIR}/metrics" ]; then
          echo "========================================================"
          echo "SKIPPING downstream evaluation for: $RUN_NAME"
          echo "Downstream results already exist."
          echo "========================================================"
          DOWNSTREAM_EXISTS=true
        fi
        
        # Only run downstream evaluation if results don't exist
        if [ "$DOWNSTREAM_EXISTS" = false ]; then
          echo "========================================================"
          echo "Starting downstream evaluation for: $RUN_NAME"
          echo "========================================================"
          
          python contrastive_experiments/train_downstream_heads.py \
            --contrastive_checkpoint $BEST_CHECKPOINT \
            --data_path $DATA_PATH \
            --temp_dfs_path $TEMP_DFS_PATH \
            --output_dir $DOWNSTREAM_RUN_DIR \
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
            --wandb_run_name "downstream_${RUN_NAME}"
          
          echo "Completed downstream evaluation for: $RUN_NAME"
          
          # Save a summary of results
          echo "Configuration: $RUN_NAME" > "${DOWNSTREAM_RUN_DIR}/summary.txt"
          echo "Batch Size: $BATCH_SIZE, Learning Rate: $LR, Seed: $SEED, Proj Dim: $PROJ_DIM" >> "${DOWNSTREAM_RUN_DIR}/summary.txt"
          echo "Contrastive Checkpoint: $BEST_CHECKPOINT" >> "${DOWNSTREAM_RUN_DIR}/summary.txt"
          
          # Append test metrics if they exist
          if [ -f "${DOWNSTREAM_RUN_DIR}/metrics/test_metrics.json" ]; then
            echo "Test Metrics:" >> "${DOWNSTREAM_RUN_DIR}/summary.txt"
            cat "${DOWNSTREAM_RUN_DIR}/metrics/test_metrics.json" >> "${DOWNSTREAM_RUN_DIR}/summary.txt"
          fi
        fi
        
      done
    done
  done
done

echo "All experiments and downstream evaluations completed at $(date)"
echo "All results saved to $RESULTS_ROOT"

# Create a summary report for all downstream results
echo "Creating summary report of all downstream results..."
SUMMARY_FILE="${RESULTS_ROOT}/all_results_summary.txt"
echo "DOWNSTREAM TASK EVALUATION SUMMARY" > $SUMMARY_FILE
echo "Generated at: $(date)" >> $SUMMARY_FILE
echo "=================================================" >> $SUMMARY_FILE

# Find all summary.txt files and extract key metrics
for SUMMARY in $(find $DOWNSTREAM_DIR -name "summary.txt"); do
  CONFIG_DIR=$(dirname $SUMMARY)
  CONFIG_NAME=$(basename $CONFIG_DIR)
  
  echo "Configuration: $CONFIG_NAME" >> $SUMMARY_FILE
  
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
  
  echo "=================================================" >> $SUMMARY_FILE
done

echo "Summary report created at: $SUMMARY_FILE"

# Recommended next steps:
# 1. Check WandB for the best performing configuration
# 2. Run a more focused sweep around the best parameters
# 3. Train the final model with the best parameters for more epochs
# 4. Try varying other parameters (temperature, architecture) next 