#!/bin/bash

# Contrastive Learning Experiments - RESUME SCRIPT
# This script resumes training from a checkpoint

# Set common parameters
DATA_PATH="/path/to/mimic/data"  # Update this path
TEMP_DFS_PATH="temp_dfs_lite"
MODEL_TYPE="raindrop_v2"

# Set up result directories
RESULTS_ROOT="contrastive_results/hyperparameter_sweep"
CHECKPOINT_DIR="${RESULTS_ROOT}/checkpoints"
LOGS_DIR="${RESULTS_ROOT}/logs"

# Distributed training settings
NUM_GPUS=8 # Set number of GPUs to use
STRATEGY="ddp_find_unused_parameters_true"  # DDP strategy
ACCELERATOR="gpu"  # Use GPU acceleration
PRECISION="32"  # Use 32-bit precision

# Create a run log to keep track of executions
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="${RESULTS_ROOT}/run_history.txt"

USE_WANDB=true
USE_WANDB_FLAG=""
if [ "$USE_WANDB" = true ]; then
  USE_WANDB_FLAG="--use_wandb"
fi
WANDB_PROJECT="GeomML_Contrastive_Sweep"

# RESUME CONFIGURATION - Change these values to match the run you want to resume
BATCH_SIZE=256
LR=0.001
SEED=6525
PROJ_DIM=256
USE_PHECODE_LOSS="False"
TEMP=0.07

# Fixed parameters
EPOCHS=20
D_MODEL=256
NLAYERS=2
NHEADS=2

# Create unique name for this run based on hyperparameters
PHECODE_SUFFIX=$([ "$USE_PHECODE_LOSS" = "True" ] && echo "_phe" || echo "_nophe")
RUN_NAME="bs${BATCH_SIZE}_lr${LR}_seed${SEED}_proj${PROJ_DIM}_temp${TEMP}${PHECODE_SUFFIX}"
MODEL_DIR="${CHECKPOINT_DIR}/${RUN_NAME}"

# Find the checkpoint file
CHECKPOINT_FILE="${MODEL_DIR}/last.ckpt"
if [ ! -f "$CHECKPOINT_FILE" ]; then
  # Try to find another checkpoint
  CHECKPOINT_FILE=$(find $MODEL_DIR -name "*.ckpt" | sort -V | tail -1)
  if [ -z "$CHECKPOINT_FILE" ]; then
    echo "ERROR: No checkpoint found to resume from in $MODEL_DIR!"
    exit 1
  fi
fi

echo "========================================================"
echo "RESUMING EXPERIMENT: $RUN_NAME"
echo "Batch Size: $BATCH_SIZE, Learning Rate: $LR, Random Seed: $SEED, Projection Dim: $PROJ_DIM"
echo "PHEcode Loss: $USE_PHECODE_LOSS, Temperature: $TEMP"
echo "Resuming from checkpoint: $CHECKPOINT_FILE"
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
  --devices $NUM_GPUS \
  --resume_from_checkpoint "$CHECKPOINT_FILE"

CONTRASTIVE_STATUS=$?
echo "Completed resumed experiment: $RUN_NAME (exit code: $CONTRASTIVE_STATUS)"
echo "All results saved to $MODEL_DIR" 