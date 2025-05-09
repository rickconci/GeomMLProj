import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import json
from pathlib import Path
import time
import logging
import math
from datetime import datetime
import pytorch_lightning as pl
from train_utils import seed_everything, get_device, calculate_phecode_loss, calculate_binary_classification_metrics, calculate_phecode_metrics, prepare_phecode_targets
from run_contrastive.contrastive_utils import count_parameters, detailed_count_parameters
from models.models_utils import ProjectionHead
from models.main_models import DSEncoderWithWeightedSum
from models.models_rd import Raindrop_v2
from lightning_fabric.utilities.apply_func import move_data_to_device
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


class RaindropMultitaskModel(pl.LightningModule):
    """PyTorch Lightning module for Raindrop-based multitask learning"""
    
    def __init__(self, args=None, dims=None):
        """Initialize the model with command line arguments"""
        super().__init__()
        
        # If args is None, it means we're loading from a checkpoint
        # We'll restore it from hparams later
        self.loading_from_checkpoint = args is None
        
        if self.loading_from_checkpoint:
            # Create minimal args with defaults - will be replaced in on_load_checkpoint
            args = type('MinimalArgs', (), {
                'use_wandb': False,
                'seed': 42,
                'checkpoint_dir': "./checkpoints",
                'sensor_wise_mask': True,
                'model_type': "DS_TS_concat"
            })()
            logging.info("Initializing model for checkpoint loading")
        
        self.args = args
        self.save_hyperparameters(vars(args))
        seed_everything(args.seed)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
                
        self.dims = dims
        
        # Ensure model_type has a default value if not specified
        if not hasattr(args, 'model_type'):
            args.model_type = "DS_TS_concat"  # Default to using both modalities
        
        # Initialize model components as None by default
        self.ts_model = None
        self.ds_encoder = None
        self.ts_projection = None
        self.text_projection = None
        self.current_phecode_predictor = None
        self.next_phecode_predictor = None
        self.mortality_classifier = None
        self.readmission_classifier = None
        
        # Set phecode_size BEFORE calling init_model
        if dims is not None:
            self.phecode_size = self.dims.get('phecode_size', 1788)
        else:
            # Default value, will be overridden when dims becomes available
            self.phecode_size = 1788
        
        # Initialize model if we have dimensions and we're not loading from checkpoint
        if dims is not None and not self.loading_from_checkpoint:
            self.init_model()
            
            # Log model parameters after initialization
            logging.info(f"Model parameters: {count_parameters(self):,}")
            param_details = detailed_count_parameters(self)
            logging.info("Model components:")
            for module_name, param_count in param_details.items():
                if module_name != 'total':
                    logging.info(f"  {module_name}: {param_count:,} ({param_count/param_details['total']*100:.1f}%)")
        else:
            logging.info("Model will be initialized after checkpoint loading")
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Model type: {args.model_type}")  # Log which model type we're using
    
    def init_model(self):
        """Initialize Raindrop_v2 model with multiple task heads"""
        logging.info(f"Initializing Raindrop_v2 model for {self.args.model_type} configuration")
        
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
        
        # Create projection heads
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

        # Determine the input size for the prediction heads based on model type
        if self.args.model_type == 'DS_TS_concat':
            input_proj_dim = self.args.projection_dim * 2  # TS + DS
        else:
            input_proj_dim = self.args.projection_dim  # Either TS or DS alone

        # Initialize task-specific heads
        self.current_phecode_predictor = nn.Sequential(
            nn.Linear(input_proj_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, self.phecode_size)
        )
     
        self.next_phecode_predictor = nn.Sequential(
            nn.Linear(input_proj_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, self.phecode_size)
        )
        
        self.mortality_classifier = nn.Sequential(
            nn.Linear(input_proj_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, 1)  # Binary classification
        )
        
        self.readmission_classifier = nn.Sequential(
            nn.Linear(input_proj_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, 1)  # Binary classification
        )
    
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
        """Forward pass for Raindrop multitask model that supports three configurations:
        1. DS only - only use discharge summary embeddings
        2. TS only - only use time series embeddings
        3. DS_TS_concat - concatenate both embeddings
        """
        # Initialize variables
        ts_proj = None
        text_proj = None
        
        # Process based on model type
        if self.args.model_type in ['TS_only', 'DS_TS_concat']:
            # Process time series data with Raindrop_v2
            ts_output, _, _ = self.ts_model(
                batch_data['P'], 
                batch_data['P_static'], 
                batch_data['P_time'].permute(1, 0),  
                batch_data['P_length'].squeeze(1)
            )
            # Project TS representation
            ts_proj = self.ts_projection(ts_output)
            
        if self.args.model_type in ['DS_only', 'DS_TS_concat']:
            # Process discharge summary text
            text_embeddings = self.ds_encoder(batch_data['discharge_embeddings'])
            # Project DS representation
            text_proj = self.text_projection(text_embeddings)
            
        # Use the appropriate projection based on model_type
        if self.args.model_type == 'DS_TS_concat':
            # Concatenate both embeddings
            concat_proj = torch.cat([ts_proj, text_proj], dim=1)
        elif self.args.model_type == 'DS_only':
            # Use only DS embeddings
            concat_proj = text_proj
        elif self.args.model_type == 'TS_only':
            # Use only TS embeddings
            concat_proj = ts_proj
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
        
        # Apply prediction heads
        current_phecode_logits = self.current_phecode_predictor(concat_proj)
        next_phecode_logits = self.next_phecode_predictor(concat_proj)
        mortality_logits = self.mortality_classifier(concat_proj)
        readmission_logits = self.readmission_classifier(concat_proj)

        return current_phecode_logits, next_phecode_logits, mortality_logits, readmission_logits


    def mortality_prediction_loss(self, mortality_logits, batch_data):
        """Calculate mortality prediction loss"""
        mortality_labels = batch_data.get('mortality_label')
        
        # Skip if we don't have mortality data
        if mortality_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        # Use binary cross-entropy loss
        return F.binary_cross_entropy_with_logits(mortality_logits.squeeze(-1), mortality_labels)
    
    def readmission_prediction_loss(self, readmission_logits, batch_data):
        """Calculate readmission prediction loss"""
        readmission_labels = batch_data.get('readmission_label')
        
        # Skip if we don't have readmission data
        if readmission_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        # Use binary cross-entropy loss
        return F.binary_cross_entropy_with_logits(readmission_logits.squeeze(-1), readmission_labels)
    
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
    
    def next_phecode_prediction_loss(self, phecode_logits, batch_data):
        """Calculate next PHEcode prediction loss"""
        # Extract next PHEcodes
        next_idxs = batch_data.get('next_idx_padded')
        next_lens = batch_data.get('next_phecode_len')
        
        # Skip if we don't have PHEcode data
        if next_idxs is None or next_lens is None:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate the PHEcode loss using the utility function
        return calculate_phecode_loss(phecode_logits, next_idxs, next_lens, self.device)
    
    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step"""
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass with the appropriate model type
        current_phecode_logits, next_phecode_logits, mortality_logits, readmission_logits = self.model_forward(batch_data)
        
        # Calculate individual losses
        current_phecode_loss = self.current_phecode_prediction_loss(current_phecode_logits, batch_data)
        next_phecode_loss = self.next_phecode_prediction_loss(next_phecode_logits, batch_data)
        mortality_loss = self.mortality_prediction_loss(mortality_logits, batch_data)
        readmission_loss = self.readmission_prediction_loss(readmission_logits, batch_data)

        # Combine all losses
        total_loss = mortality_loss + readmission_loss + current_phecode_loss + next_phecode_loss
        
        # Log losses
        batch_size = batch_data['P'].shape[1]
        self.log('train_mortality_loss', mortality_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_readmission_loss', readmission_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_current_phecode_loss', current_phecode_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_next_phecode_loss', next_phecode_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """PyTorch Lightning validation step"""
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass with the appropriate model type
        current_phecode_logits, next_phecode_logits, mortality_logits, readmission_logits = self.model_forward(batch_data)
        
        # Calculate individual losses
        current_phecode_loss = self.current_phecode_prediction_loss(current_phecode_logits, batch_data)
        next_phecode_loss = self.next_phecode_prediction_loss(next_phecode_logits, batch_data)
        mortality_loss = self.mortality_prediction_loss(mortality_logits, batch_data)
        readmission_loss = self.readmission_prediction_loss(readmission_logits, batch_data)
        
        # Combine all losses
        total_loss = mortality_loss + readmission_loss + current_phecode_loss + next_phecode_loss
        
        # Log losses
        batch_size = batch_data['P'].shape[1]
        self.log('val_mortality_loss', mortality_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val_readmission_loss', readmission_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val_current_phecode_loss', current_phecode_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val_next_phecode_loss', next_phecode_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        
        # Calculate metrics, store outputs in instance variables for epoch end aggregation
        outputs = self._calculate_and_log_metrics(batch_data, 'val', 
                                       mortality_logits, readmission_logits,
                                       current_phecode_logits, next_phecode_logits,
                                       batch_size)
        
        outputs['loss'] = total_loss
        
        # Initialize step outputs storage if needed
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        
        # Store outputs for on_validation_epoch_end
        self.validation_step_outputs.append(outputs)
        
        return outputs
    
    def test_step(self, batch, batch_idx):
        """PyTorch Lightning test step"""
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass with the appropriate model type  
        current_phecode_logits, next_phecode_logits, mortality_logits, readmission_logits = self.model_forward(batch_data)
        
        # Calculate individual losses
        current_phecode_loss = self.current_phecode_prediction_loss(current_phecode_logits, batch_data)
        next_phecode_loss = self.next_phecode_prediction_loss(next_phecode_logits, batch_data)
        mortality_loss = self.mortality_prediction_loss(mortality_logits, batch_data) 
        readmission_loss = self.readmission_prediction_loss(readmission_logits, batch_data)
        
        # Combine all losses
        total_loss = mortality_loss + readmission_loss + current_phecode_loss + next_phecode_loss
        
        # Log losses
        batch_size = batch_data['P'].shape[1]
        self.log('test_mortality_loss', mortality_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('test_readmission_loss', readmission_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('test_current_phecode_loss', current_phecode_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('test_next_phecode_loss', next_phecode_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log('test_total_loss', total_loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        # Calculate metrics, store outputs in instance variables for epoch end aggregation
        outputs = self._calculate_and_log_metrics(batch_data, 'test', 
                                       mortality_logits, readmission_logits,
                                       current_phecode_logits, next_phecode_logits,
                                       batch_size)
        
        outputs['loss'] = total_loss
        
        # Initialize step outputs storage if needed
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        
        # Store outputs for on_test_epoch_end
        self.test_step_outputs.append(outputs)
        
        return outputs
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to aggregate results across all processes (PL 2.0 hook)"""
        if hasattr(self, 'validation_step_outputs') and self.validation_step_outputs:
            # Aggregate results from all validation steps
            self._aggregate_epoch_metrics(self.validation_step_outputs, 'val')
            # Clear the stored outputs
            self.validation_step_outputs = []
        
    def on_test_epoch_end(self):
        """Called at the end of test epoch to aggregate results across all processes (PL 2.0 hook)"""
        if hasattr(self, 'test_step_outputs') and self.test_step_outputs:
            # Aggregate results from all test steps
            self._aggregate_epoch_metrics(self.test_step_outputs, 'test')
            # Clear the stored outputs
            self.test_step_outputs = []
        
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
                    # Add default values for all required parameters 
                    # These match the defaults in run_contrastive_experiments.sh
                    self.d_model = 256
                    self.hidden_dim = 256
                    self.projection_dim = 256
                    self.nlayers = 2
                    self.num_heads = 2
                    self.sensor_wise_mask = True
                    self.pooling_type = 'attention'
                    self.lr = 0.0005
                    self.epochs = 20
                    self.batch_size = 128
                    self.seed = 42
                    self.phecode_loss_weight = 0.1
                    self.check_val_every_n_epoch = 4
                    self.early_stopping = True
                    self.patience = 5
                    self.use_wandb = False
                    self.accelerator = 'auto'
                    self.precision = '32'
                    self.strategy = 'ddp_find_unused_parameters_true'
                    # Add model_type which is crucial for our multitask model
                    self.model_type = 'DS_TS_concat'
                    
                    # Override with values from hparams if available
                    for key, value in hparams_dict.items():
                        setattr(self, key, value)
                    
                    # Log the parameters we're using
                    logging.info(f"Loaded model with parameters: d_model={self.d_model}, "
                                f"hidden_dim={self.hidden_dim}, projection_dim={self.projection_dim}, "
                                f"nlayers={self.nlayers}, num_heads={self.num_heads}, "
                                f"pooling_type={self.pooling_type}")
            
            # Create args object from hyperparameters
            restored_args = ArgsFromCheckpoint(dict(self.hparams))
            self.args = restored_args
            
            # Initialize model components if dims was provided during init but 
            # model components weren't initialized yet
            if self.dims is not None and self.ts_model is None:
                logging.info("Initializing model components from checkpoint with provided dims")
                self.init_model()
            # If self.dims is None, check if we need to extract it from the state_dict
            elif self.dims is None and 'hyper_parameters' in checkpoint:
                # Try to extract dims from the checkpoint's hyperparameters
                if 'dims' in checkpoint['hyper_parameters']:
                    self.dims = checkpoint['hyper_parameters']['dims']
                    logging.info("Restored dims from checkpoint hyperparameters")
                    # Initialize model components with the restored dims
                    self.init_model()
                else:
                    logging.warning("No dims in checkpoint hyperparameters, using defaults")

    def _calculate_and_log_metrics(self, batch_data, prefix, 
                                   mortality_logits, readmission_logits,
                                   current_phecode_logits, next_phecode_logits,
                                   batch_size):
        """Calculate and log all metrics with the given prefix (val or test)"""
        # Apply sigmoid to get probability predictions
        mortality_preds = torch.sigmoid(mortality_logits.squeeze(-1))
        readmission_preds = torch.sigmoid(readmission_logits.squeeze(-1))
        current_phecode_preds = torch.sigmoid(current_phecode_logits)
        next_phecode_preds = torch.sigmoid(next_phecode_logits)
        
        # For validation/test epoch aggregation
        outputs = {}
        
        # Binary metrics for mortality
        if 'mortality_label' in batch_data:
            try:
                mortality_metrics = calculate_binary_classification_metrics(
                    mortality_preds.cpu().numpy(), 
                    batch_data['mortality_label'].cpu().numpy()
                )
                self.log(f'{prefix}_mortality_auroc', mortality_metrics['auroc'], 
                         on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log(f'{prefix}_mortality_auprc', mortality_metrics['auprc'], 
                         on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                
                # For epoch-end aggregation
                outputs['mortality_preds'] = mortality_preds.detach()
                outputs['mortality_labels'] = batch_data['mortality_label'].detach()
            except Exception as e:
                logging.warning(f"Error calculating mortality metrics: {e}")
        
        # Binary metrics for readmission
        if 'readmission_label' in batch_data:
            try:
                readmission_metrics = calculate_binary_classification_metrics(
                    readmission_preds.cpu().numpy(), 
                    batch_data['readmission_label'].cpu().numpy()
                )
                self.log(f'{prefix}_readmission_auroc', readmission_metrics['auroc'], 
                         on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log(f'{prefix}_readmission_auprc', readmission_metrics['auprc'], 
                         on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                
                # For epoch-end aggregation
                outputs['readmission_preds'] = readmission_preds.detach()
                outputs['readmission_labels'] = batch_data['readmission_label'].detach()
            except Exception as e:
                logging.warning(f"Error calculating readmission metrics: {e}")
        
        # Current PHEcode metrics
        if 'current_idx_padded' in batch_data and 'current_phecode_len' in batch_data:
            try:
                # Prepare targets
                current_phecode_targets, valid_samples = prepare_phecode_targets(
                    batch_data, self.device, self.phe_code_size
                )
                
                if current_phecode_targets is not None:
                    # Filter predictions for valid samples if needed
                    current_preds_filtered = current_phecode_preds
                    if valid_samples is not None:
                        current_preds_filtered = current_phecode_preds[valid_samples]
                    
                    # Calculate metrics
                    current_phecode_metrics = calculate_phecode_metrics(
                        current_preds_filtered.cpu().numpy(),
                        current_phecode_targets.cpu().numpy(),
                        None  # No dataset object needed for basic metrics
                    )
                    
                    # Log metrics
                    self.log(f'{prefix}_current_phecode_macro_auc', 
                             current_phecode_metrics.get('macro_auc', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    self.log(f'{prefix}_current_phecode_micro_auc', 
                             current_phecode_metrics.get('micro_auc', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    self.log(f'{prefix}_current_phecode_micro_ap', 
                             current_phecode_metrics.get('micro_ap', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    self.log(f'{prefix}_current_phecode_prec@5', 
                             current_phecode_metrics.get('prec@5', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    
                    # For epoch-end aggregation
                    outputs['current_phecode_preds'] = current_preds_filtered.detach()
                    outputs['current_phecode_targets'] = current_phecode_targets.detach()
            except Exception as e:
                logging.warning(f"Error calculating current PHEcode metrics: {e}")
        
        # Next PHEcode metrics
        if 'next_idx_padded' in batch_data and 'next_phecode_len' in batch_data:
            try:
                # Prepare targets - we need to use a modified version for next_phecode
                next_batch_data = {
                    'next_idx_padded': batch_data['next_idx_padded'],
                    'next_len': batch_data['next_phecode_len']
                }
                
                next_phecode_targets, valid_samples = prepare_phecode_targets(
                    {
                        'next_idx_padded': batch_data['next_idx_padded'],
                        'next_len': batch_data['next_phecode_len']
                    }, 
                    self.device, 
                    self.phe_code_size
                )
                
                if next_phecode_targets is not None:
                    # Filter predictions for valid samples if needed
                    next_preds_filtered = next_phecode_preds
                    if valid_samples is not None:
                        next_preds_filtered = next_phecode_preds[valid_samples]
                    
                    # Calculate metrics
                    next_phecode_metrics = calculate_phecode_metrics(
                        next_preds_filtered.cpu().numpy(),
                        next_phecode_targets.cpu().numpy(),
                        None  # No dataset object needed for basic metrics
                    )
                    
                    # Log metrics
                    self.log(f'{prefix}_next_phecode_macro_auc', 
                             next_phecode_metrics.get('macro_auc', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    self.log(f'{prefix}_next_phecode_micro_auc', 
                             next_phecode_metrics.get('micro_auc', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    self.log(f'{prefix}_next_phecode_micro_ap', 
                             next_phecode_metrics.get('micro_ap', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    self.log(f'{prefix}_next_phecode_prec@5', 
                             next_phecode_metrics.get('prec@5', 0.0), 
                             on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
                    
                    # For epoch-end aggregation
                    outputs['next_phecode_preds'] = next_preds_filtered.detach()
                    outputs['next_phecode_targets'] = next_phecode_targets.detach()
            except Exception as e:
                logging.warning(f"Error calculating next PHEcode metrics: {e}")
                
        return outputs
    
    def _aggregate_epoch_metrics(self, outputs, prefix):
        """Aggregate metrics across all batches and compute overall metrics"""
        # We handle aggregation differently for each metric type
        
        # For mortality
        if any('mortality_preds' in output for output in outputs):
            try:
                # Gather all predictions and labels
                all_mortality_preds = torch.cat([output['mortality_preds'] for output in outputs if 'mortality_preds' in output]).cpu()
                all_mortality_labels = torch.cat([output['mortality_labels'] for output in outputs if 'mortality_labels' in output]).cpu()
                
                # Compute global metrics
                mortality_metrics = calculate_binary_classification_metrics(
                    all_mortality_preds.numpy(), 
                    all_mortality_labels.numpy()
                )
                
                self.log(f'{prefix}_epoch_mortality_auroc', mortality_metrics['auroc'], sync_dist=True)
                self.log(f'{prefix}_epoch_mortality_auprc', mortality_metrics['auprc'], sync_dist=True)
                
                logging.info(f"{prefix.upper()} Mortality: AUROC={mortality_metrics['auroc']:.4f}, AUPRC={mortality_metrics['auprc']:.4f}")
            except Exception as e:
                logging.warning(f"Error in mortality epoch metrics: {e}")
        
        # For readmission
        if any('readmission_preds' in output for output in outputs):
            try:
                # Gather all predictions and labels
                all_readmission_preds = torch.cat([output['readmission_preds'] for output in outputs if 'readmission_preds' in output]).cpu()
                all_readmission_labels = torch.cat([output['readmission_labels'] for output in outputs if 'readmission_labels' in output]).cpu()
                
                # Compute global metrics
                readmission_metrics = calculate_binary_classification_metrics(
                    all_readmission_preds.numpy(), 
                    all_readmission_labels.numpy()
                )
                
                self.log(f'{prefix}_epoch_readmission_auroc', readmission_metrics['auroc'], sync_dist=True)
                self.log(f'{prefix}_epoch_readmission_auprc', readmission_metrics['auprc'], sync_dist=True)
                
                logging.info(f"{prefix.upper()} Readmission: AUROC={readmission_metrics['auroc']:.4f}, AUPRC={readmission_metrics['auprc']:.4f}")
            except Exception as e:
                logging.warning(f"Error in readmission epoch metrics: {e}")
                
        # For current PHEcodes
        if any('current_phecode_preds' in output for output in outputs):
            try:
                # Gather all predictions and labels - this could be large
                all_current_phecode_preds = torch.cat([output['current_phecode_preds'] for output in outputs if 'current_phecode_preds' in output]).cpu()
                all_current_phecode_targets = torch.cat([output['current_phecode_targets'] for output in outputs if 'current_phecode_targets' in output]).cpu()
                
                # Compute global metrics
                current_phecode_metrics = calculate_phecode_metrics(
                    all_current_phecode_preds.numpy(),
                    all_current_phecode_targets.numpy(),
                    None  # No dataset object needed for basic metrics
                )
                
                self.log(f'{prefix}_epoch_current_phecode_macro_auc', current_phecode_metrics.get('macro_auc', 0.0), sync_dist=True)
                self.log(f'{prefix}_epoch_current_phecode_micro_auc', current_phecode_metrics.get('micro_auc', 0.0), sync_dist=True)
                self.log(f'{prefix}_epoch_current_phecode_micro_ap', current_phecode_metrics.get('micro_ap', 0.0), sync_dist=True)
                self.log(f'{prefix}_epoch_current_phecode_prec@5', current_phecode_metrics.get('prec@5', 0.0), sync_dist=True)
                
                logging.info(f"{prefix.upper()} Current PHEcodes: "
                            f"Macro-AUC={current_phecode_metrics.get('macro_auc', 0.0):.4f}, "
                            f"Micro-AUC={current_phecode_metrics.get('micro_auc', 0.0):.4f}, "
                            f"Micro-AP={current_phecode_metrics.get('micro_ap', 0.0):.4f}, "
                            f"Precision@5={current_phecode_metrics.get('prec@5', 0.0):.4f}")
            except Exception as e:
                logging.warning(f"Error in current phecode epoch metrics: {e}")
                
        # For next PHEcodes
        if any('next_phecode_preds' in output for output in outputs):
            try:
                # Gather all predictions and labels - this could be large
                all_next_phecode_preds = torch.cat([output['next_phecode_preds'] for output in outputs if 'next_phecode_preds' in output]).cpu()
                all_next_phecode_targets = torch.cat([output['next_phecode_targets'] for output in outputs if 'next_phecode_targets' in output]).cpu()
                
                # Compute global metrics
                next_phecode_metrics = calculate_phecode_metrics(
                    all_next_phecode_preds.numpy(),
                    all_next_phecode_targets.numpy(),
                    None  # No dataset object needed for basic metrics
                )
                
                self.log(f'{prefix}_epoch_next_phecode_macro_auc', next_phecode_metrics.get('macro_auc', 0.0), sync_dist=True)
                self.log(f'{prefix}_epoch_next_phecode_micro_auc', next_phecode_metrics.get('micro_auc', 0.0), sync_dist=True)
                self.log(f'{prefix}_epoch_next_phecode_micro_ap', next_phecode_metrics.get('micro_ap', 0.0), sync_dist=True)
                self.log(f'{prefix}_epoch_next_phecode_prec@5', next_phecode_metrics.get('prec@5', 0.0), sync_dist=True)
                
                logging.info(f"{prefix.upper()} Next PHEcodes: "
                            f"Macro-AUC={next_phecode_metrics.get('macro_auc', 0.0):.4f}, "
                            f"Micro-AUC={next_phecode_metrics.get('micro_auc', 0.0):.4f}, "
                            f"Micro-AP={next_phecode_metrics.get('micro_ap', 0.0):.4f}, "
                            f"Precision@5={next_phecode_metrics.get('prec@5', 0.0):.4f}")
            except Exception as e:
                logging.warning(f"Error in next phecode epoch metrics: {e}")



