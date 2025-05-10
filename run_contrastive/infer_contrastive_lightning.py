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
    # Set up logging
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, f'infer_contrastive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set random seeds for reproducibility
    seed_everything(args.seed)
    
    # Force sensor_wise_mask to be True
    args.sensor_wise_mask = True
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint not found at: {args.checkpoint_path}")
        return
    
    logging.info(f"Starting inference with checkpoint: {args.checkpoint_path}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    
    # Initialize data module
    data_module = ContrastiveDataModule(
        data_path=args.data_path,
        temp_dfs_path=args.temp_dfs_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task_mode='CONTRASTIVE'
    )
    
    # Setup data module
    data_module.setup()
    
    # Load pretrained model
    logging.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = RaindropContrastiveModel.load_from_checkpoint(args.checkpoint_path)
    
    # Log model architecture
    logging.info(f"Model architecture loaded successfully")
    
    # Make sure to run on a single device for validation to get all embeddings
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,  # Force single device
        precision="32",  # Use 32-bit precision for compatibility
        strategy="auto",  # Use auto strategy for single device
        enable_model_summary=True,
    )
    
    # Override the embeddings_dir in the model to use our specified output directory
    embeddings_path = os.path.join(args.output_dir, f"embeddings_validation_full")
    os.makedirs(embeddings_path, exist_ok=True)
    
    # Modify validation_step and on_validation_epoch_end in the model instance
    original_validation_step = model.validation_step
    original_validation_epoch_end = model.on_validation_epoch_end
    
    # Replace the validation_step to save embeddings for all samples
    def new_validation_step(self, batch, batch_idx):
        # Skip sanity checking
        if self.trainer.sanity_checking:
            return None
            
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass through model to get embeddings (with no gradients)
        with torch.no_grad():
            ts_proj, text_proj, _ = self.model_forward(batch_data)
        
        # Store results in memory-mapped arrays
        if not hasattr(self, 'val_embeddings'):
            # Initialize memory-mapped arrays on first batch
            # Save directly to the specified output directory
            self.embeddings_dir = embeddings_path
            logging.info(f"Saving embeddings to: {self.embeddings_dir}")
            
            # Get validation dataset size
            val_size = len(self.trainer.datamodule.val_dataset)
            logging.info(f"Creating memory-mapped arrays for validation dataset of size {val_size}")
            
            # Get the actual shape of next_idx_padded from the batch
            next_idx_shape = batch_data['next_idx_padded'].shape
            logging.info(f"next_idx_padded shape from batch: {next_idx_shape}")
            
            # Create memory-mapped arrays
            self.val_embeddings = {
                'ts_proj': np.memmap(os.path.join(self.embeddings_dir, 'ts_proj.mmap'), 
                                   dtype='float32', mode='w+', 
                                   shape=(val_size, self.args.projection_dim)),
                'text_proj': np.memmap(os.path.join(self.embeddings_dir, 'text_proj.mmap'), 
                                     dtype='float32', mode='w+', 
                                     shape=(val_size, self.args.projection_dim)),
                'mortality_label': np.memmap(os.path.join(self.embeddings_dir, 'mortality_label.mmap'), 
                                          dtype='float32', mode='w+', 
                                          shape=(val_size,)),
                'readmission_label': np.memmap(os.path.join(self.embeddings_dir, 'readmission_label.mmap'), 
                                            dtype='float32', mode='w+', 
                                            shape=(val_size,)),
                'next_idx_padded': np.memmap(os.path.join(self.embeddings_dir, 'next_idx_padded.mmap'), 
                                           dtype='int64', mode='w+', 
                                           shape=(val_size, next_idx_shape[1])),
                'next_phecode_len': np.memmap(os.path.join(self.embeddings_dir, 'next_phecode_len.mmap'), 
                                            dtype='int64', mode='w+', 
                                            shape=(val_size,)),
            }
            
            # Add hadm_id if present in dataset
            if hasattr(self.trainer.datamodule.val_dataset, 'hadm_ids'):
                self.val_embeddings['hadm_id'] = np.memmap(os.path.join(self.embeddings_dir, 'hadm_id.mmap'), 
                                                         dtype='int64', mode='w+', 
                                                         shape=(val_size,))
        
        # Calculate start index for this batch
        start_idx = batch_idx * self.args.batch_size
        end_idx = start_idx + ts_proj.shape[0]
        
        # Store results directly in memory-mapped arrays
        self.val_embeddings['ts_proj'][start_idx:end_idx] = ts_proj.cpu().numpy()
        self.val_embeddings['text_proj'][start_idx:end_idx] = text_proj.cpu().numpy()
        self.val_embeddings['mortality_label'][start_idx:end_idx] = batch_data['mortality_label'].cpu().numpy()
        self.val_embeddings['readmission_label'][start_idx:end_idx] = batch_data['readmission_label'].cpu().numpy()
        self.val_embeddings['next_idx_padded'][start_idx:end_idx] = batch_data['next_idx_padded'].cpu().numpy()
        self.val_embeddings['next_phecode_len'][start_idx:end_idx] = batch_data['next_phecode_len'].cpu().numpy()
        
        if 'hadm_id' in self.val_embeddings:
            hadm_id_tensor = batch.get('hadm_id', None)
            if hadm_id_tensor is not None:
                self.val_embeddings['hadm_id'][start_idx:end_idx] = hadm_id_tensor.cpu().numpy()
        
        if batch_idx % 10 == 0:  # Log progress every 10 batches
            logging.info(f"Processed validation batch {batch_idx}, samples {start_idx} to {end_idx-1}")
        
        return None
    
    # Replace the on_validation_epoch_end to save metadata
    def new_on_validation_epoch_end(self):
        if hasattr(self, 'val_embeddings'):
            # Save metadata
            metadata = {
                'batch_size': self.args.batch_size,
                'projection_dim': self.args.projection_dim,
                'use_phecode_loss': self.use_phecode_loss,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'checkpoint_path': args.checkpoint_path
            }
            with open(os.path.join(self.embeddings_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            # Flush memory-mapped arrays to disk
            for array in self.val_embeddings.values():
                array.flush()
            
            logging.info(f"Embeddings saved to {self.embeddings_dir}")
            
            # Clean up memory-mapped arrays
            for array in self.val_embeddings.values():
                del array
            del self.val_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Set new validation methods
    model.validation_step = new_validation_step.__get__(model, RaindropContrastiveModel)
    model.on_validation_epoch_end = new_on_validation_epoch_end.__get__(model, RaindropContrastiveModel)
    
    # Run validation only
    logging.info("Starting validation pass...")
    trainer.validate(model, datamodule=data_module)
    
    logging.info("Validation completed. All embeddings should be saved.")

if __name__ == "__main__":
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Inference for pretrained contrastive model')
    
    # General arguments
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./embeddings', help='Directory to save embeddings')
    parser.add_argument('--data_path', type=str, default='/path/to/mimic/data', help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite', help='Path to cache directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Hardware specific arguments
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator to use (auto, cpu, gpu)')
    
    args = parser.parse_args()
    main(args)
