import os
import argparse
import torch
# Set global default tensor type to float32 for MPS compatibility
torch.set_default_dtype(torch.float32)

# Enable Tensor Core operations on supported CUDA devices
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    # Check if device likely has Tensor Cores (A100, V100, RTX series, etc.)
    if any(gpu_type in device_name for gpu_type in ['A100', 'A10', 'V100', 'RTX', 'Ampere', 'Volta', 'Turing']):
        torch.set_float32_matmul_precision('high')
        print(f"Enabled high precision Tensor Core operations on {device_name}")

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import json
from pathlib import Path
import time
import logging
import dotenv
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Add parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_utils import *

from models.main_models import KEDGN, DSEncoderWithWeightedSum
from models.models_rd import Raindrop_v2
from models.models_utils import ProjectionHead

from ContrastiveDataloaderLighting import ContrastiveDataModule
from RaindropContrastive_lightning import RaindropContrastiveModel, get_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/contrastive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

device = get_device()


def main(args):
    # Set random seeds for reproducibility
    seed_everything(args.seed)
    
    # Force sensor_wise_mask to be True
    args.sensor_wise_mask = True
    
    # Disable debug printing
    toggle_debug(False)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize data module
    data_module = ContrastiveDataModule(
        data_path=args.data_path,
        temp_dfs_path=args.temp_dfs_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task_mode='CONTRASTIVE'
    )
    
    # Initialize model with data module
    model = get_model(args, data_module)
    
    # Configure callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"contrastive_{args.model_type}_" + "{epoch:02d}-{train_contrastive_loss:.4f}",
        monitor='train_contrastive_loss',
        mode='min',
        save_top_k=1 if not args.save_all_checkpoints else -1,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback if enabled
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='train_contrastive_loss',
            mode='min',
            patience=args.patience,
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Configure logger
    logger = None
    if args.use_wandb:
        try:
            dotenv.load_dotenv('dot_env.txt')
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            logger = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                log_model=True,
                save_dir=args.checkpoint_dir
            )
            logger.log_hyperparams(vars(args))
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            logging.info("Continuing without wandb logging")
    
    # Force everything to float32 for MPS compatibility
    # Override any specified precision
    forced_precision = "32" if torch.backends.mps.is_available() else args.precision
    if torch.backends.mps.is_available() and args.precision != "32":
        logging.warning(f"Overriding precision={args.precision} to precision=32 for MPS compatibility")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=args.accelerator,
        precision=forced_precision,
        strategy=args.strategy if args.strategy else 'auto',
        devices=get_lightning_devices(args.devices),
    )
    
    # Train the model
    if args.resume_from_checkpoint:
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, data_module)
    
    # Test the model
    trainer.test(model, data_module)


if __name__ == "__main__":
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Contrastive learning for EHR data')
    
    # General arguments
    parser.add_argument('--data_path', type=str, default='/path/to/mimic/data', help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite', help='Path to cache directory')
    parser.add_argument('--model_type', type=str, choices=['raindrop_v2'], default='raindrop_v2', help='Model type')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of dataloader workers')
    
    # Model specific arguments
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--projection_dim', type=int, default=256, help='Projection dimension')
    
    # Raindrop specific arguments
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of model')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--global_structure_path', type=str, default=None, help='Path to global structure')
    parser.add_argument('--sensor_wise_mask', type=bool, default=True, help='Use sensor-wise masking (defaults to True)')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    
    # Contrastive learning arguments
    parser.add_argument('--contrastive_method', type=str, default='clip', choices=['clip', 'infonce'], 
                        help='Contrastive learning method')
    parser.add_argument('--pooling_type', type=str, default='attention', choices=['weighted_sum', 'attention'],
                        help='Pooling type for discharge summary encoder')
    parser.add_argument('--temperature', type=float, default=0.07, 
                        help='Temperature parameter for contrastive loss')
    
    # Training arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_all_checkpoints', action='store_true', help='Save checkpoint after every epoch')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='GeomML_Contrastive', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    
    # Lightning specific arguments
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use (auto, cpu, gpu, tpu)')
    parser.add_argument('--precision', type=str, default='32', help='Precision for training (16, 32, 64)')
    parser.add_argument('--strategy', type=str, default=None, help='Distributed training strategy')
    parser.add_argument('--devices', type=int, default=None, help='Number of devices to use')
    
    # New argument for PHEcode loss
    parser.add_argument('--use_phecode_loss', type=lambda x: x.lower() in ['true', 't', 'yes', 'y', '1'], 
                        default=True, help='Enable PHEcode auxiliary loss (true/false)')
    
    args = parser.parse_args()
    main(args)
