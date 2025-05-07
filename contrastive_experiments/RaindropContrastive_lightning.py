import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import wandb
import json
from pathlib import Path
import time
import logging
import dotenv
import math
from datetime import datetime
import pytorch_lightning as pl
from train_utils import seed_everything, get_device, calculate_phecode_loss, calculate_binary_classification_metrics, calculate_phecode_metrics, prepare_phecode_targets
from contrastive_experiments.contrastive_utils import clip_contrastive_loss, infonce_loss, count_parameters, detailed_count_parameters
from models.models_utils import ProjectionHead
from models.main_models import DSEncoderWithWeightedSum
from models.models_rd import Raindrop_v2
from ContrastiveDataloaderLighting import get_model_dimensions
from lightning_fabric.utilities.apply_func import move_data_to_device


class RaindropContrastiveModel(pl.LightningModule):
    """PyTorch Lightning module for Raindrop-based contrastive learning"""
    
    def __init__(self, args=None, dims=None):
        """Initialize the model with command line arguments"""
        super().__init__()
        
        # If args is None, it means we're loading from a checkpoint
        # We'll restore it from hparams later
        if args is None:
            # Create a dummy args object with minimum required attributes
            # Full values will be loaded from hparams
            class DummyArgs:
                def __init__(self):
                    self.use_wandb = False
                    self.seed = 42
                    self.checkpoint_dir = "./checkpoints"
                    self.sensor_wise_mask = True
                    self.temperature = 0.07
                    
            args = DummyArgs()
            
        self.args = args
        self.save_hyperparameters(vars(args))
        seed_everything(args.seed)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
                
        self.dims = dims
        self.log_temperature = nn.Parameter(torch.ones(1) * np.log(1.0 / args.temperature))
        self.use_phecode_loss = getattr(args, 'use_phecode_loss', True)
        self.phe_code_size = self.dims['phecode_size']
        
        # Initialize model components if dimensions are provided
        if dims is not None:
            self.init_model()
            
            # Log model parameters after initialization
            logging.info(f"Model parameters: {count_parameters(self):,}")
            param_details = detailed_count_parameters(self)
            logging.info("Model components:")
            for module_name, param_count in param_details.items():
                if module_name != 'total':
                    logging.info(f"  {module_name}: {param_count:,} ({param_count/param_details['total']*100:.1f}%)")
        else:
            logging.warning("No dimensions provided, model will be initialized later")
            self.ts_model = None
            self.ds_encoder = None
            self.ts_projection = None
            self.text_projection = None
            self.phecode_predictor = None
        
        logging.info(f"Using device: {self.device}")
    
    def get_temperature(self):
        """Get the current temperature value (inverse of log_temperature)"""
        return 1.0 / torch.exp(self.log_temperature)
    
    def init_model(self):
        """Initialize Raindrop_v2 model with contrastive learning components"""
        logging.info("Initializing Raindrop_v2 model for contrastive learning")
        
        # Global structure is fully connected
        global_structure = torch.ones(self.dims['variables_num'], self.dims['variables_num'])
        # Create the base Raindrop_v2 model
        self.ts_model = Raindrop_v2(
            d_inp=self.dims['variables_num'], 
            d_model=self.args.d_model,
            nhead=self.args.num_heads,
            nhid=self.args.hidden_dim,
            nlayers=self.args.nlayers,
            dropout=0.3,
            max_len=self.dims['timestamps'],
            d_static=self.dims['d_static'],
            n_classes=1,  
            global_structure=global_structure,
            sensor_wise_mask=self.dims['sensor_wise_mask'],
            static= self.dims['d_static']
        )
        
        # Create the discharge summary encoder
        self.ds_encoder = DSEncoderWithWeightedSum(
            hidden_dim=self.args.hidden_dim,
            projection_dim=self.args.projection_dim,
            pooling_type=self.args.pooling_type,
            num_heads=self.args.num_heads
        )
        
        # Create projection heads for contrastive learning
        raindrop_output_dim = self.args.d_model + 16  # base model + positional encoding
        if self.args.sensor_wise_mask:
            raindrop_output_dim = self.dims['variables_num'] * (self.args.d_model // self.dims['variables_num'] + 16)
        if self.dims['d_static'] > 0:
            raindrop_output_dim += self.dims['variables_num']
        
        self.ts_projection = ProjectionHead(
            input_dim=raindrop_output_dim,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.projection_dim
        )
        
        self.text_projection = ProjectionHead(
            input_dim=self.args.projection_dim,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.projection_dim
        )

        if self.use_phecode_loss:
            # Create PHEcode predictor using the projection dim as input
            self.current_phecode_predictor = nn.Sequential(
                nn.Linear(self.args.projection_dim*2, self.phe_code_size))
            logging.info(f"Created PHEcode predictor head with output size: {self.phe_code_size}")
        else:
            # Set a dummy component that returns None when called
            self.phecode_predictor = lambda x: None
            logging.info("PHEcode predictor not created (auxiliary loss disabled)")
    
    def prepare_batch(self, batch):
        """Prepare a batch for training or evaluation"""
        # Skip empty batches
        if not batch:
            return None
        values = batch['values'].to(self.device, dtype=torch.float32)  # [B, T, F]
        mask = batch['mask'].to(self.device, dtype=torch.float32)  # [B, T, F]
        P = torch.cat([values, mask], dim=2).permute(1, 0, 2) # [T, B, F]
        length = batch['length'].to(self.device).unsqueeze(1) # [B, 1]
        ds_embeddings = [emb.to(self.device, dtype=torch.float32) for emb in batch['ds_embedding']]

        return {
            'P': P,
            'P_static': batch['static'].to(self.device, dtype=torch.float32),
            'P_length': length,
            'P_time': batch['times'].to(self.device, dtype=torch.float32),
            'discharge_embeddings': ds_embeddings,
            'current_idx_padded': batch['current_idx_padded'].to(self.device),
            'current_phecode_len': batch['current_len'].to(self.device),
            'next_idx_padded': batch['next_idx_padded'].to(self.device),
            'next_phecode_len': batch['next_len'].to(self.device),
            'mortality_label': batch['mortality_label'].to(self.device, dtype=torch.float32),
            'readmission_label': batch['readmission_label'].to(self.device, dtype=torch.float32),
        }
    
    def model_forward(self, batch_data):
        """Forward pass for Raindrop contrastive model"""
    
        # Process time series data with Raindrop_v2
        ts_output, _, _ = self.ts_model(batch_data['P'], 
                                        batch_data['P_static'], 
                                        batch_data['P_time'].permute(1, 0),  
                                        batch_data['P_length'].squeeze(1))
        
        # Process discharge summary text
        text_embeddings = self.ds_encoder(batch_data['discharge_embeddings'])
        
        # Project both representations to the same space
        ts_proj = self.ts_projection(ts_output)
        text_proj = self.text_projection(text_embeddings)
        
        # Generate PHEcode predictions only if the auxiliary loss is enabled
        if self.use_phecode_loss:
            # We'll use the average of time series and text projections for prediction
            #fused_proj = (ts_proj + text_proj) / 2
            concat_proj = torch.cat([ts_proj, text_proj], dim=1)
            phecode_logits = self.current_phecode_predictor(concat_proj)
        else:
            phecode_logits = None
        
        return ts_proj, text_proj, phecode_logits
    
    def contrastive_loss_with_learnable_temp(self, ts_proj, text_proj):
        """Calculate contrastive loss using the existing functions but with learnable temperature"""
        # Get the current temperature value
        temperature = self.get_temperature()
        
        # Log the temperature
        self.log('temperature', temperature.item(), on_step=False, on_epoch=True, sync_dist=True, batch_size=ts_proj.shape[0])
        
        # Use the existing contrastive loss functions from train_utils.py
        if self.args.contrastive_method == 'clip':
            # For CLIP method, we'll use clip_contrastive_loss
            loss = clip_contrastive_loss(ts_proj, text_proj, temperature=temperature)
        elif self.args.contrastive_method == 'infonce':
            # For InfoNCE method, we'll use infonce_loss
            loss = infonce_loss(ts_proj, text_proj, temperature=temperature)
        else:
            raise ValueError(f"Unknown contrastive method: {self.args.contrastive_method}")
        
        return loss
    
    def current_phecode_prediction_loss(self, phecode_logits, batch_data):
        """Calculate PHEcode prediction loss"""
        # Extract current PHEcodes (not next PHEcodes)
        current_idxs = batch_data.get('current_idx_padded')
        current_lens = batch_data.get('current_phecode_len')
        
        # Skip if we don't have PHEcode data
        if current_idxs is None or current_lens is None:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate the PHEcode loss using the utility function
        return calculate_phecode_loss(phecode_logits, current_idxs, current_lens, self.device)
    
    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step"""
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        ts_proj, text_proj, phecode_logits = self.model_forward(batch_data)
        contrastive_loss = self.contrastive_loss_with_learnable_temp(ts_proj, text_proj)
        self.log('train_contrastive_loss', contrastive_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_data['P'].shape[1])
        
        total_loss = contrastive_loss
        
        # Add PHEcode prediction loss if enabled
        if self.use_phecode_loss and phecode_logits is not None:
            phecode_loss = self.current_phecode_prediction_loss(phecode_logits, batch_data)
            phecode_weight = getattr(self.args, 'phecode_loss_weight', 0.01)
            total_loss = total_loss + phecode_weight * phecode_loss
            self.log('train_phecode_loss', phecode_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_data['P'].shape[1])
        
        # Log total loss
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_data['P'].shape[1])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """PyTorch Lightning validation step - collect embeddings for later evaluation"""
        # Skip extensive validation during sanity checking
        if self.trainer.sanity_checking:
            return None
            
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass through model to get embeddings (with no gradients)
        with torch.no_grad():
            ts_proj, text_proj, _ = self.model_forward(batch_data)
        
        # Only process on the main process (rank 0) to avoid race conditions
        if self.trainer.is_global_zero:
            # Store results in memory-mapped arrays
            if not hasattr(self, 'val_embeddings'):
                # Initialize memory-mapped arrays on first batch
                # Save in a directory at the same level as the working directory
                base_dir = os.path.join(os.getcwd(), 'Embeddings')
                ph_suffix = '_phe' if self.use_phecode_loss else '_nophe'
                param_dir = f"bs{self.args.batch_size}_lr{self.args.lr}_seed{self.args.seed}_proj{self.args.projection_dim}_temp{self.args.temperature}{ph_suffix}"
                self.embeddings_dir = os.path.join(base_dir, param_dir, f"epoch_{self.trainer.current_epoch:03d}")
                os.makedirs(self.embeddings_dir, exist_ok=True)
                logging.info(f"Saving embeddings to: {self.embeddings_dir}")
                
                # Get validation dataset size from the datamodule's val_dataset
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
                                               shape=(val_size, next_idx_shape[1])),  # Use actual shape from batch
                    'next_phecode_len': np.memmap(os.path.join(self.embeddings_dir, 'next_phecode_len.mmap'), 
                                                dtype='int64', mode='w+', 
                                                shape=(val_size,))
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
        
        return None
    
    def on_validation_epoch_end(self):
        """Save validation results at the end of epoch"""
        if self.trainer.is_global_zero and hasattr(self, 'val_embeddings'):
            # Save metadata
            metadata = {
                'epoch': self.trainer.current_epoch,
                'batch_size': self.args.batch_size,
                'projection_dim': self.args.projection_dim,
                'use_phecode_loss': self.use_phecode_loss
            }
            with open(os.path.join(self.embeddings_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            # Flush memory-mapped arrays to disk
            for array in self.val_embeddings.values():
                array.flush()
            
            logging.info(f"Completed validation epoch {self.trainer.current_epoch+1}")
            logging.info(f"Embeddings saved to {self.embeddings_dir}")
            
            # Clean up memory-mapped arrays
            for array in self.val_embeddings.values():
                del array
            del self.val_embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_step(self, batch, batch_idx):
        """PyTorch Lightning test step - skipped as we use separate downstream evaluation"""
        # Skip detailed testing - proper downstream evaluation is done separately
        # in train_probes.py after training
        return None
    
    def configure_optimizers(self):
        """Configure optimizers for training with cosine annealing and warmup"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        
        # Get total number of training steps
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        
        # Number of warmup steps (typically 10% of total steps), minimum 1
        warmup_steps = max(1, int(0.1 * total_steps))
        
        # Create a cosine annealing scheduler with warmup
        # Using math.cos instead of torch.cos since step is a Python float
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(1.0, step / warmup_steps) * 0.5 * (1 + math.cos(math.pi * step / total_steps)) 
                if step <= total_steps else 0.0
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        """
        Custom hook to ensure all tensors are float32 when using MPS
        """
        # Check if we're using MPS
        if 'mps' in str(device):
            # For each item in the batch that's a tensor
            for key in batch:
                if isinstance(batch[key], torch.Tensor) and batch[key].dtype == torch.float64:
                    # Convert float64 to float32
                    batch[key] = batch[key].to(dtype=torch.float32)
        
        # Move batch to device as usual
        return move_data_to_device(batch, device)

    def on_load_checkpoint(self, checkpoint):
        """Called when loading a checkpoint - restore args from hyperparameters"""
        # Create a namespace object from the hyperparameters dictionary
        if hasattr(self, 'hparams'):
            # Convert hparams dict to an object with attributes
            class ArgsFromCheckpoint:
                def __init__(self, hparams_dict):
                    for key, value in hparams_dict.items():
                        setattr(self, key, value)
            
            # Create args object from hyperparameters
            restored_args = ArgsFromCheckpoint(dict(self.hparams))
            self.args = restored_args



