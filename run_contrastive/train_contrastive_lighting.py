import os
import argparse
import torch
# Enable cuDNN autotuner to select optimal convolution algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True
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
from RaindropContrastive_lightning import RaindropContrastiveModel

device = get_device()

def init_wandb(args):
    """Initialize Weights & Biases tracking if enabled"""
    if args.use_wandb:
        # Check if this is the main process (rank 0)
        is_main_process = os.environ.get('LOCAL_RANK', '0') == '0'
        
        if is_main_process:
            try:
                dotenv.load_dotenv('dot_env.txt')
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=vars(args)
                )
                wandb.config.update({"device": str(device)})
                logging.info("Successfully initialized wandb")
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                logging.info("Continuing without wandb logging")
                args.use_wandb = False
        else:
            logging.info("Non-main process: Skipping wandb initialization")
            args.use_wandb = False


def main(args):
    # Set up logging only on main process (rank 0)
    is_main_process = os.environ.get('LOCAL_RANK', '0') == '0'
    
    if is_main_process:
        # Convert checkpoint_dir to absolute path if it isn't already
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        # The logs directory should be at the same level as the checkpoint directory
        logs_dir = os.path.join(os.path.dirname(checkpoint_dir), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Debug print the paths
        print(f"Checkpoint dir (abs): {checkpoint_dir}")
        print(f"Logs dir (abs): {logs_dir}")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, f'contrastive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
                logging.StreamHandler()
            ]
        )
    else:
        # For non-main processes, just set up basic console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    # Set random seeds for reproducibility
    seed_everything(args.seed)
    
    # Force sensor_wise_mask to be True
    args.sensor_wise_mask = True
    
    # Disable debug printing
    toggle_debug(False)
    
    init_wandb(args)
    
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Log key parameters and directories
    logging.info(f"Starting contrastive training with:")
    logging.info(f"  Working directory: {os.getcwd()}")
    logging.info(f"  Checkpoint directory (abs): {args.checkpoint_dir}")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Learning rate: {args.lr}")
    logging.info(f"  Projection dimension: {args.projection_dim}")
    logging.info(f"  Use PHEcode loss: {args.use_phecode_loss}")
    
    # Initialize data module
    data_module = ContrastiveDataModule(
        data_path=args.data_path,
        temp_dfs_path=args.temp_dfs_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task_mode='CONTRASTIVE'
    )

    dims = {
        'variables_num': 80,
        'timestamps': 80,
        'd_static': 83,
        'ds_emb_dim': 768,
        'values_shape': (args.batch_size, 80, 80), # (batch_size, timestamps, variables_num)
        'phecode_size': 1788,
        'phecode_loss_weight': 0.2,
        'sensor_wise_mask': True
    }
    
 
    model = RaindropContrastiveModel(args, dims=dims)
    
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
        verbose=True,
        every_n_epochs=1,  # Save checkpoint every epoch
        save_on_train_epoch_end=True  # Ensure it saves at the end of training epoch
    )
    callbacks.append(checkpoint_callback)
    
    # Log checkpoint configuration
    logging.info(f"Checkpoint configuration:")
    logging.info(f"  Directory: {checkpoint_callback.dirpath}")
    logging.info(f"  Filename template: {checkpoint_callback.filename}")
    logging.info(f"  Monitor: {checkpoint_callback.monitor}")
    logging.info(f"  Save top k: {checkpoint_callback.save_top_k}")
    logging.info(f"  Save last: {checkpoint_callback.save_last}")
    
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
        # Only initialize WandB logger on main process (rank 0)
        is_main_process = os.environ.get('LOCAL_RANK', '0') == '0'
        
        if is_main_process:
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
        else:
            logging.info("Non-main process: Not creating WandB logger")
    
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
        strategy=args.strategy,
        sync_batchnorm=True,
        devices= list(range(get_lightning_devices(args.devices))),
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_checkpointing=True,
        enable_model_summary=True,  
        log_every_n_steps=10,
        profiler = 'simple',
        accumulate_grad_batches=args.accumulate_grad_batches,
        #num_sanity_val_steps=0,  # Disable sanity validation completely
        limit_val_batches=0.1 if args.fast_dev_run else 1.0  # Limit validation batches in normal runs
    )
    

    # Train the model
    if args.resume_from_checkpoint:
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, data_module)
    
  
    
    # Get checkpoint path - try best first, then last if best not available
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logging.info(f"Best model saved to: {best_model_path}")
        checkpoint_path = best_model_path
    elif hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path:
        last_model_path = checkpoint_callback.last_model_path
        logging.info(f"Best model not found, using last model: {last_model_path}")
        checkpoint_path = last_model_path


if __name__ == "__main__":
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Contrastive learning for EHR data')
    
    # General arguments
    parser.add_argument('--data_path', type=str, default='/path/to/mimic/data', help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite', help='Path to cache directory')
    parser.add_argument('--model_type', type=str, choices=['raindrop_v2'], default='raindrop_v2', help='Model type')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of dataloader workers')
    
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
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true', 
                        choices=['ddp', 'ddp_spawn','auto', 'ddp_find_unused_parameters_true'], 
                        help='Distributed training strategy')
    parser.add_argument('--devices', type=int, default=None, help='Number of devices to use')
    
    # New argument for PHEcode loss
    parser.add_argument('--use_phecode_loss', type=lambda x: x.lower() in ['true', 't', 'yes', 'y', '1'], 
                        default=True, help='Enable PHEcode auxiliary loss (true/false)')
    
    # Add lightning specific parameters
    parser.add_argument('--check_val_every_n_epoch', type=int, default=3, help='Run validation every n epochs')
    parser.add_argument('--accumulate_grad_batches', type=int, default=2, help='Accumulate gradients over n batches')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run a single training and validation batch for testing')
    args = parser.parse_args()
    main(args)
