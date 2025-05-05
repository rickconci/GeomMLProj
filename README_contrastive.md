# Contrastive Learning for EHR Data

This module implements contrastive learning for Electronic Health Records (EHR) data, aligning time series (TS) representations with discharge summary (DS) text representations.

## Overview

The contrastive learning approach trains models to create aligned embeddings between:

- Time series data from patient stays (vital signs, lab results, etc.)
- Discharge summary text describing the patient's stay

Two time series model architectures are supported:

1. **KEDGN**: Knowledge-Enhanced Dynamic Graph Network
2. **Raindrop_v2**: Transformer-based architecture for irregular time series

## Usage

### Basic Usage

```bash
python GeomMLProj/train_contrastive.py \
    --data_path /path/to/mimic/data \
    --temp_dfs_path ./temp_dfs_lite \
    --model_type kedgn \
    --batch_size 32 \
    --epochs 30 \
    --hidden_dim 128 \
    --projection_dim 128 \
    --contrastive_method clip \
    --temperature 0.1
```

### Model Selection

Choose between KEDGN or Raindrop_v2:

```bash
# For KEDGN model
python GeomMLProj/train_contrastive.py --model_type kedgn --use_gat --num_heads 4

# For Raindrop_v2 model
python GeomMLProj/train_contrastive.py --model_type raindrop_v2 --d_model 128 --nlayers 2 --num_heads 4
```

### Contrastive Learning Method

Choose between CLIP-style contrastive loss or InfoNCE:

```bash
# CLIP style (bidirectional)
python GeomMLProj/train_contrastive.py --contrastive_method clip --temperature 0.1

# InfoNCE style 
python GeomMLProj/train_contrastive.py --contrastive_method infonce --temperature 0.1
```

### Discharge Summary Encoder Options

Configure how discharge summaries are encoded:

```bash
# With weighted sum pooling
python GeomMLProj/train_contrastive.py --pooling_type weighted_sum

# With attention pooling (default)
python GeomMLProj/train_contrastive.py --pooling_type attention --num_heads 4
```

### Logging and Checkpoints

Enable WandB logging for experiment tracking:

```bash
python GeomMLProj/train_contrastive.py --use_wandb --wandb_project "MyProject" --wandb_entity "MyUsername"
```

Configure checkpointing and early stopping:

```bash
python GeomMLProj/train_contrastive.py \
    --checkpoint_dir ./my_checkpoints \
    --early_stopping \
    --patience 5 \
    --save_all_checkpoints
```

Resume training from a checkpoint:

```bash
python GeomMLProj/train_contrastive.py --resume_from_checkpoint ./checkpoints/contrastive_kedgn_best.pt
```

## Architecture Details

### Base Trainer

The `ContrastiveTrainer` base class handles:

- Data loading and preprocessing
- Training, validation, and testing loops
- Checkpoint management
- Logging (console and optional WandB)

### Model-Specific Trainers

- `KEDGNContrastiveTrainer`: Handles KEDGN model initialization and forward pass
- `RaindropContrastiveTrainer`: Handles Raindrop_v2 model initialization and forward pass

### Model Components

Each contrastive model consists of:

1. **Time Series Encoder**: Either KEDGN or Raindrop_v2
2. **Discharge Summary Encoder**: Based on pretrained embeddings with pooling
3. **Projection Heads**: Projects both modalities to a shared embedding space

## Testing the Module

To verify the module loads correctly:

```bash
python GeomMLProj/test_contrastive.py --model_type kedgn
```

This checks that all dependencies and class structures are properly configured.
