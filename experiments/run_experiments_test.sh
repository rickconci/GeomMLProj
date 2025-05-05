#!/bin/bash

# Mock testing script for experiment configurations
# This script runs minimal versions of experiments to verify configurations

# Define parameter arrays - keep same as original to test all combinations
TASKS=("NEXT_24h" "CONTRASTIVE")
LEARNING_RATES=(0.0001 0.001 0.01)
BATCH_SIZES=(16 32 64)
NUM_HEADS=(2 4 8)

# Set common parameters with minimal values for testing
WANDB_PROJECT="MIMIC_IV_Test"
EPOCHS=1          # Minimal epochs
DATASET="mimic4"
NUM_WORKERS=2     # Reduce workers
BASE_CACHE_DIR="./test_cache"
PRECOMPUTE_TRAIN=10  # Minimal samples
PRECOMPUTE_VAL=5     # Minimal samples
PRECOMPUTE_TEST=2    # Minimal samples
CHECKPOINT_BASE_DIR="./test_checkpoints"
PATIENCE=2           # Reduce patience
EARLY_STOPPING=true

# Create base directories
mkdir -p "$BASE_CACHE_DIR"
mkdir -p "$CHECKPOINT_BASE_DIR"

# Log directory
LOG_DIR="./test_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/test_experiments_log.txt"

echo "Starting test runs at $(date)" | tee -a "$MASTER_LOG"
echo "This is a MOCK TEST RUN - not running full experiments" | tee -a "$MASTER_LOG"

# Function to get task-specific parameters
get_task_params() {
    local task=$1
    local params=""
    
    if [ "$task" == "CONTRASTIVE" ]; then
        params="--use_contrastive --precompute_tensors --contrastive_temp 0.1 --metric_for_best_model auprc"
    else  # NEXT_24h
        params="--precompute_tensors --class_weights --metric_for_best_model auroc"
    fi
    
    echo "$params"
}

# Function to test a command without running it fully
test_command() {
    local cmd="$1"
    local log_file="$2"
    local exp_id="$3"
    
    echo "Testing command for $exp_id" | tee -a "$MASTER_LOG"
    echo "Command: $cmd" | tee -a "$log_file"
    
    # Run with timeout to prevent hanging
    timeout 300s $cmd 2>&1 | tee -a "$log_file" || {
        echo "ERROR: Command failed or timed out for $exp_id" | tee -a "$MASTER_LOG" "$log_file"
        return 1
    }
    
    return 0
}

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Test GCN model configurations
echo "Testing GCN configurations..." | tee -a "$MASTER_LOG"
for TASK in "${TASKS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      # Create unique experiment ID and directories
      EXP_ID="gcn_${TASK}_lr${LR}_bs${BS}"
      EXP_CACHE_DIR="${BASE_CACHE_DIR}/${EXP_ID}"
      EXP_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${EXP_ID}"
      mkdir -p "$EXP_CACHE_DIR"
      mkdir -p "$EXP_CHECKPOINT_DIR"
      
      # Get task-specific parameters
      TASK_PARAMS=$(get_task_params "$TASK")
      
      # Base command with test mode
      CMD="python train_mimic_iv.py \
        --dataset $DATASET \
        --task_mode $TASK \
        --lr $LR \
        --batch_size $BS \
        --epochs $EPOCHS \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name ${EXP_ID}_test \
        --cache_dir $EXP_CACHE_DIR \
        --checkpoint_dir $EXP_CHECKPOINT_DIR \
        --num_workers $NUM_WORKERS \
        --max_precompute_train $PRECOMPUTE_TRAIN \
        --max_precompute_val $PRECOMPUTE_VAL \
        --max_precompute_test $PRECOMPUTE_TEST 
      
      # Add early stopping if enabled
      if [ "$EARLY_STOPPING" = true ]; then
        CMD="$CMD --early_stopping --patience $PATIENCE"
      fi
      
      # Add task-specific parameters
      CMD="$CMD $TASK_PARAMS"
      
      # Log file for this test
      EXP_LOG="$LOG_DIR/${EXP_ID}_test.log"
      
      # Test the command
      if test_command "$CMD" "$EXP_LOG" "$EXP_ID"; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo "✓ Test passed: $EXP_ID" | tee -a "$MASTER_LOG"
      else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_TESTS+=("$EXP_ID")
        echo "✗ Test failed: $EXP_ID" | tee -a "$MASTER_LOG"
      fi
      
      # Limit to just one configuration per task for faster testing
      break
    done
    break
  done
done

# Test GAT model configurations
echo "Testing GAT configurations..." | tee -a "$MASTER_LOG"
for TASK in "${TASKS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      for NH in "${NUM_HEADS[@]}"; do
        # Create unique experiment ID and directories
        EXP_ID="gat_${TASK}_lr${LR}_bs${BS}_nh${NH}"
        EXP_CACHE_DIR="${BASE_CACHE_DIR}/${EXP_ID}"
        EXP_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${EXP_ID}"
        mkdir -p "$EXP_CACHE_DIR"
        mkdir -p "$EXP_CHECKPOINT_DIR"
        
        # Get task-specific parameters
        TASK_PARAMS=$(get_task_params "$TASK")
        
        # Base command with test mode
        CMD="python train_mimic_iv.py \
          --dataset $DATASET \
          --task_mode $TASK \
          --lr $LR \
          --batch_size $BS \
          --epochs $EPOCHS \
          --use_gat \
          --num_heads $NH \
          --wandb_project $WANDB_PROJECT \
          --wandb_run_name ${EXP_ID}_test \
          --cache_dir $EXP_CACHE_DIR \
          --checkpoint_dir $EXP_CHECKPOINT_DIR \
          --num_workers $NUM_WORKERS \
          --max_precompute_train $PRECOMPUTE_TRAIN \
          --max_precompute_val $PRECOMPUTE_VAL \
          --max_precompute_test $PRECOMPUTE_TEST 
        
        # Add early stopping if enabled
        if [ "$EARLY_STOPPING" = true ]; then
          CMD="$CMD --early_stopping --patience $PATIENCE"
        fi
        
        # Add task-specific parameters
        CMD="$CMD $TASK_PARAMS"
        
        # Log file for this test
        EXP_LOG="$LOG_DIR/${EXP_ID}_test.log"
        
        # Test the command
        if test_command "$CMD" "$EXP_LOG" "$EXP_ID"; then
          TESTS_PASSED=$((TESTS_PASSED + 1))
          echo "✓ Test passed: $EXP_ID" | tee -a "$MASTER_LOG"
        else
          TESTS_FAILED=$((TESTS_FAILED + 1))
          FAILED_TESTS+=("$EXP_ID")
          echo "✗ Test failed: $EXP_ID" | tee -a "$MASTER_LOG"
        fi
        
        # Limit to just one configuration per task+num_heads for faster testing
        break
      done
      break
    done
    break
  done
done

# Print test summary
echo "" | tee -a "$MASTER_LOG"
echo "TEST SUMMARY" | tee -a "$MASTER_LOG"
echo "============" | tee -a "$MASTER_LOG"
echo "Tests passed: $TESTS_PASSED" | tee -a "$MASTER_LOG"
echo "Tests failed: $TESTS_FAILED" | tee -a "$MASTER_LOG"

if [ $TESTS_FAILED -gt 0 ]; then
  echo "Failed tests:" | tee -a "$MASTER_LOG"
  for test in "${FAILED_TESTS[@]}"; do
    echo "  - $test" | tee -a "$MASTER_LOG"
  done
fi

echo "" | tee -a "$MASTER_LOG"
echo "Test run completed at $(date)" | tee -a "$MASTER_LOG"
echo "All logs saved in $LOG_DIR" | tee -a "$MASTER_LOG"

# Add note about test_mode flag
echo "" | tee -a "$MASTER_LOG"
echo "NOTE: This script requires a '--test_mode' flag in train_mimic_iv.py" | tee -a "$MASTER_LOG"
echo "If you're getting errors about unknown argument, you need to add this flag" | tee -a "$MASTER_LOG" 