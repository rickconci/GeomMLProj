# RaindropMultitask Model Training

This README provides instructions on how to train the RaindropMultitask model for predicting mortality, readmission, current PHEcodes, and next PHEcodes using three different configurations:

1. DS only - Only using discharge summary text
2. TS only - Only using time series data
3. DS+TS concatenated - Using both modalities with concatenated embeddings

## Setup

Ensure you have all required packages installed according to the project requirements.

## Training

The model supports distributed data parallel (DDP) training for multi-GPU setups. To train the model, use the `train_multitask_lightining.py` script with the appropriate configuration.

### Example Commands

#### 1. DS-only Model

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 GeomMLProj/contrastive_experiments/train_multitask_lightining.py \
  --model_type DS_only \
  --data_path /path/to/mimic/data \
  --temp_dfs_path ./temp_dfs_lite \
  --d_model 256 \
  --hidden_dim 256 \
  --projection_dim 256 \
  --nlayers 2 \
  --num_heads 2 \
  --pooling_type attention \
  --batch_size 128 \
  --lr 0.0005 \
  --epochs 30 \
  --patience 5 \
  --early_stopping \
  --checkpoint_dir ./checkpoints/ds_only
```

#### 2. TS-only Model

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 GeomMLProj/contrastive_experiments/train_multitask_lightining.py \
  --model_type TS_only \
  --data_path /path/to/mimic/data \
  --temp_dfs_path ./temp_dfs_lite \
  --d_model 256 \
  --hidden_dim 256 \
  --projection_dim 256 \
  --nlayers 2 \
  --num_heads 2 \
  --pooling_type attention \
  --batch_size 128 \
  --lr 0.0005 \
  --epochs 30 \
  --patience 5 \
  --early_stopping \
  --checkpoint_dir ./checkpoints/ts_only
```

#### 3. DS+TS Concatenated Model

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 GeomMLProj/contrastive_experiments/train_multitask_lightining.py \
  --model_type DS_TS_concat \
  --data_path /path/to/mimic/data \
  --temp_dfs_path ./temp_dfs_lite \
  --d_model 256 \
  --hidden_dim 256 \
  --projection_dim 256 \
  --nlayers 2 \
  --num_heads 2 \
  --pooling_type attention \
  --batch_size 128 \
  --lr 0.0005 \
  --epochs 30 \
  --patience 5 \
  --early_stopping \
  --checkpoint_dir ./checkpoints/ds_ts_concat
```

## Monitoring

The training script will log metrics to the console and to TensorBoard. You can also enable Weights & Biases logging with the `--use_wandb` flag.

## Tracked Metrics

The model tracks a comprehensive set of metrics for each task during validation and testing:

### Mortality Prediction Metrics

- `mortality_auroc`: Area Under the Receiver Operating Characteristic curve
- `mortality_auprc`: Area Under the Precision-Recall Curve

### Readmission Prediction Metrics

- `readmission_auroc`: Area Under the ROC curve
- `readmission_auprc`: Area Under the Precision-Recall Curve

### Current PHEcode Prediction Metrics

- `current_phecode_macro_auc`: Average ROC AUC across all PHEcodes
- `current_phecode_micro_auc`: ROC AUC calculated by flattening all predictions
- `current_phecode_micro_ap`: Average precision calculated by flattening all predictions
- `current_phecode_prec@5`: Proportion of the top 5 predicted PHEcodes that are actually present

### Next PHEcode Prediction Metrics

- `next_phecode_macro_auc`: Average ROC AUC across all PHEcodes for next visit
- `next_phecode_micro_auc`: ROC AUC calculated by flattening all next visit predictions
- `next_phecode_micro_ap`: Average precision for next visit predictions
- `next_phecode_prec@5`: Proportion of the top 5 predicted next visit PHEcodes that are actually present

These metrics provide a comprehensive assessment of model performance across all tasks, with special attention to the multi-label PHEcode prediction tasks.

## Evaluation

After training, the best model checkpoint will be saved to the specified checkpoint directory. You can evaluate the model using the standard evaluation script.

## Optional Arguments

- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: Set the W&B project name
- `--wandb_entity`: Set the W&B entity name
- `--precision`: Set the precision for training (16, 32, 64)
- `--accumulate_grad_batches`: Number of batches to accumulate gradients
- `--check_val_every_n_epoch`: Run validation every n epochs
- `--save_all_checkpoints`: Save checkpoint after every epoch
