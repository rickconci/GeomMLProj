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
import dotenv
import math
from datetime import datetime
import pytorch_lightning as pl
from train_utils import seed_everything, get_device, calculate_phecode_loss, evaluate_downstream_tasks, calculate_binary_classification_metrics, calculate_phecode_metrics, prepare_phecode_targets
from contrastive_experiments.contrastive_utils import clip_contrastive_loss, infonce_loss, count_parameters, detailed_count_parameters
from models.models_utils import ProjectionHead
from models.main_models import DSEncoderWithWeightedSum
from models.models_rd import Raindrop_v2
from ContrastiveDataloaderLighting import get_model_dimensions
from lightning_fabric.utilities.apply_func import move_data_to_device


class RaindropContrastiveModel(pl.LightningModule):
    """PyTorch Lightning module for Raindrop-based contrastive learning"""
    
    def __init__(self, args, dims=None):
        """Initialize the model with command line arguments"""
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))
        
        # Set random seeds for reproducibility
        seed_everything(args.seed)
        
        # Initialize WandB if enabled
        self.init_wandb()
        
        # Create paths for saving models and checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Track best metric
        self.best_val_metric = 0
        
        # Variable embeddings will be set from the data module
        self.var_embeddings = None
        
        # Store dimensions information
        self.dims = dims
        
        # Initialize learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.ones(1) * np.log(1.0 / args.temperature))
        
        # Flag to enable/disable PHEcode auxiliary loss
        self.use_phecode_loss = getattr(args, 'use_phecode_loss', False)
        logging.info(f"PHEcode auxiliary loss: {'enabled' if self.use_phecode_loss else 'disabled'}")
        
        # Store PHEcode dimensions
        self.phe_code_size = None
        if hasattr(args, 'phe_code_size'):
            self.phe_code_size = args.phe_code_size
        
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
    
    def init_wandb(self):
        """Initialize Weights & Biases tracking if enabled"""
        if self.args.use_wandb:
            try:
                dotenv.load_dotenv('dot_env.txt')
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                self.run = wandb.init(
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity,
                    config=vars(self.args)
                )
                wandb.config.update({"device": str(self.device)})
                logging.info("Successfully initialized wandb")
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                logging.info("Continuing without wandb logging")
                self.args.use_wandb = False
    
    def get_temperature(self):
        """Get the current temperature value (inverse of log_temperature)"""
        return 1.0 / torch.exp(self.log_temperature)
    
    def init_model(self):
        """Initialize Raindrop_v2 model with contrastive learning components"""
        logging.info("Initializing Raindrop_v2 model for contrastive learning")
        
        # Load global structure for Raindrop if provided
        global_structure = None
        if self.args.global_structure_path:
            if os.path.exists(self.args.global_structure_path):
                try:
                    global_structure = torch.load(self.args.global_structure_path)
                    logging.info(f"Loaded global structure with shape {global_structure.shape}")
                except Exception as e:
                    logging.error(f"Error loading global structure: {e}")
                    logging.info("Initializing with default fully-connected structure")
                    global_structure = torch.ones(self.dims['variables_num'], self.dims['variables_num'])
            else:
                logging.warning(f"Global structure file {self.args.global_structure_path} not found")
                logging.info("Initializing with default fully-connected structure")
                global_structure = torch.ones(self.dims['variables_num'], self.dims['variables_num'])
        else:
            logging.info("Initializing with default fully-connected structure")
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
            sensor_wise_mask=self.args.sensor_wise_mask,
            static=(self.dims['d_static'] > 0)
        )
        
        # Create the discharge summary encoder
        self.ds_encoder = DSEncoderWithWeightedSum(
            hidden_dim=self.args.hidden_dim,
            projection_dim=self.args.projection_dim,
            pooling_type=self.args.pooling_type,
            num_heads=self.args.num_heads
        )
        
        # Create projection heads for contrastive learning
        # For Raindrop, determine the output dimension based on the model configuration
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
        
        # Create PHEcode predictor head only if enabled
        if self.use_phecode_loss:
            # Create PHEcode predictor using the projection dim as input
            self.phecode_predictor = nn.Linear(self.args.projection_dim, self.phe_code_size)
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
            
        # Move data to device and ensure float32 precision for MPS compatibility
        values = batch['values'].to(self.device, dtype=torch.float32)
        mask = batch['mask'].to(self.device, dtype=torch.float32)
        
        # Handle static features (may be empty)
        static = None
        if 'static' in batch and batch['static'].numel() > 0:
            static = batch['static'].to(self.device, dtype=torch.float32)
        
        # Handle time information
        times = batch['times'].to(self.device, dtype=torch.float32)
        length = batch['length'].to(self.device)
        
        # Get discharge embeddings
        discharge_embeddings = None
        if 'discharge_embeddings' in batch:
            discharge_embeddings = batch['discharge_embeddings']
            if torch.is_tensor(discharge_embeddings):
                discharge_embeddings = discharge_embeddings.to(self.device, dtype=torch.float32)
        elif 'ds_embedding' in batch:
            discharge_embeddings = batch['ds_embedding']
            if torch.is_tensor(discharge_embeddings):
                discharge_embeddings = discharge_embeddings.to(self.device, dtype=torch.float32)
        
        # Combine values and mask for model input
        P = torch.cat([values, mask], dim=2)
        
        # Prepare time dimensions
        if len(times.shape) == 2:  # Shape [B, T]
            P_time = times
        else:  # Just a single dimension [T]
            P_time = times.unsqueeze(0).repeat(values.size(0), 1)  # Shape [B, T]
        
        # Create P_avg_interval with the same shape as P_time
        P_avg_interval = torch.ones_like(P_time)  # Shape [B, T]
        
        # Expand P_avg_interval to match the number of variables
        if hasattr(self, 'dims') and self.dims is not None and 'variables_num' in self.dims:
            vars_num = self.dims['variables_num']
        else:
            vars_num = values.size(2) // 2  # Assuming half is values, half is mask
            
        P_avg_interval = P_avg_interval.unsqueeze(2).expand(-1, -1, vars_num)  # Shape [B, T, N]
        
        # Prepare length tensor
        P_length = length.unsqueeze(1) if length.dim() == 1 else length  # Shape [B, 1]
        
        return {
            'P': P,
            'P_static': static,
            'P_avg_interval': P_avg_interval,
            'P_length': P_length,
            'P_time': P_time,
            'discharge_embeddings': discharge_embeddings
        }
    
    def model_forward(self, batch_data):
        """Forward pass for Raindrop contrastive model"""
        if self.ts_model is None:
            # If model hasn't been initialized yet and we have dimensions, do it now
            if self.dims is not None:
                self.init_model()
            
            # If still not initialized, something is wrong
            if self.ts_model is None:
                raise RuntimeError("Model components not initialized. Dimensions must be provided during initialization or before forward pass.")

        # Prepare input for Raindrop_v2
        # Raindrop expects: src [max_len, batch_size, 2*d_inp]
        src = batch_data['P'].permute(1, 0, 2)  # [T, B, F]
        static = batch_data['P_static']  # [B, S]
        times = batch_data['P_time'].permute(1, 0)  # [T, B]
        lengths = batch_data['P_length'].squeeze(1)  # [B]
        
        # Process time series data with Raindrop_v2
        ts_output, _, _ = self.ts_model(src, static, times, lengths)
        
        # Process discharge summary text
        text_embeddings = self.ds_encoder(batch_data['discharge_embeddings'])
        
        # Project both representations to the same space
        ts_proj = self.ts_projection(ts_output)
        text_proj = self.text_projection(text_embeddings)
        
        # Generate PHEcode predictions only if the auxiliary loss is enabled
        if self.use_phecode_loss:
            # We'll use the average of time series and text projections for prediction
            fused_proj = (ts_proj + text_proj) / 2
            phecode_logits = self.phecode_predictor(fused_proj)
        else:
            phecode_logits = None
        
        return ts_proj, text_proj, phecode_logits
    
    def contrastive_loss_with_learnable_temp(self, ts_proj, text_proj):
        """Calculate contrastive loss using the existing functions but with learnable temperature"""
        # Get the current temperature value
        temperature = self.get_temperature()
        
        # Log the temperature
        self.log('temperature', temperature.item(), on_step=False, on_epoch=True)
        
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
    
    def phecode_prediction_loss(self, phecode_logits, batch_data):
        """Calculate PHEcode prediction loss"""
        # Extract current PHEcodes (not next PHEcodes)
        idxs = batch_data.get('current_idx_padded', batch_data.get('next_idx_padded'))
        lens = batch_data.get('current_len', batch_data.get('next_len'))
        
        # Skip if we don't have PHEcode data
        if idxs is None or lens is None:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate the PHEcode loss using the utility function
        return calculate_phecode_loss(phecode_logits, idxs, lens, self.device)
    
    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step"""
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass through model
        ts_proj, text_proj, phecode_logits = self.model_forward(batch_data)
        
        # Calculate contrastive loss with learnable temperature
        contrastive_loss = self.contrastive_loss_with_learnable_temp(ts_proj, text_proj)
        
        # Log contrastive loss
        self.log('train_contrastive_loss', contrastive_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Initialize total_loss with contrastive_loss
        total_loss = contrastive_loss
        
        # Add PHEcode prediction loss if enabled
        if self.use_phecode_loss and phecode_logits is not None:
            phecode_loss = self.phecode_prediction_loss(phecode_logits, batch_data)
            phecode_weight = getattr(self.args, 'phecode_loss_weight', 0.2)
            total_loss = total_loss + phecode_weight * phecode_loss
            self.log('train_phecode_loss', phecode_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log total loss
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """PyTorch Lightning validation step - collect embeddings for downstream evaluation"""
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass through model to get embeddings
        with torch.no_grad():
            ts_proj, text_proj, _ = self.model_forward(batch_data)
        
        # No need to compute contrastive loss during validation
        # Just collect embeddings and labels for downstream evaluation
        result = {
            'ts_proj': ts_proj.detach(),
            'text_proj': text_proj.detach()
        }
        
        # Add labels for downstream tasks if available
        for key in ['mortality_label', 'readmission_label', 'next_idx_padded', 'next_len']:
            if key in batch:
                result[key] = batch[key]
        
        return result
    
    def validation_epoch_end(self, outputs):
        """Evaluate downstream tasks at the end of validation epoch"""
        if not outputs:
            return
        
        # Collect embeddings and labels
        all_ts_projs = []
        all_text_projs = []
        all_mortality_labels = []
        all_readmission_labels = []
        all_next_idx_padded = []
        all_next_lens = []
        
        for output in outputs:
            if output is None:
                continue
                
            if 'ts_proj' in output:
                all_ts_projs.append(output['ts_proj'])
            if 'text_proj' in output:
                all_text_projs.append(output['text_proj'])
            
            # Collect labels
            if 'mortality_label' in output:
                all_mortality_labels.append(output['mortality_label'])
            if 'readmission_label' in output:
                all_readmission_labels.append(output['readmission_label'])
            if 'next_idx_padded' in output:
                all_next_idx_padded.append(output['next_idx_padded'])
            if 'next_len' in output:
                all_next_lens.append(output['next_len'])
        
        # Skip if we don't have any embeddings
        if not all_ts_projs or not all_text_projs:
            return
            
        # Concatenate embeddings from all batches
        ts_projs = torch.cat(all_ts_projs, dim=0)
        text_projs = torch.cat(all_text_projs, dim=0)
        
        # Fused embeddings (average of ts and text)
        fused_projs = (ts_projs + text_projs) / 2
        
        # Evaluate downstream tasks
        self._evaluate_downstream_mortality(fused_projs, all_mortality_labels)
        self._evaluate_downstream_readmission(fused_projs, all_readmission_labels)
        self._evaluate_downstream_phecodes(fused_projs, all_next_idx_padded, all_next_lens)
    
    def _evaluate_downstream_mortality(self, embeddings, mortality_labels):
        """Evaluate mortality prediction task"""
        if not mortality_labels:
            return
        
        # Concatenate labels
        labels = torch.cat(mortality_labels, dim=0)
        
        # Skip if no positive samples
        if labels.sum() == 0:
            return
            
        # Create a simple linear classifier for mortality prediction
        classifier = torch.nn.Linear(embeddings.shape[1], 1).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            logits = classifier(embeddings)
            preds = torch.sigmoid(logits).squeeze(-1)
            
            # Calculate AUROC and AUPRC
            metrics = calculate_binary_classification_metrics(preds.cpu().numpy(), labels.cpu().numpy())
            
            # Log metrics
            self.log('val_mortality_auroc', metrics['auroc'], on_epoch=True)
            self.log('val_mortality_auprc', metrics['auprc'], on_epoch=True)
    
    def _evaluate_downstream_readmission(self, embeddings, readmission_labels):
        """Evaluate readmission prediction task"""
        if not readmission_labels:
            return
        
        # Concatenate labels
        labels = torch.cat(readmission_labels, dim=0)
        
        # Skip if no positive samples
        if labels.sum() == 0:
            return
            
        # Create a simple linear classifier for readmission prediction
        classifier = torch.nn.Linear(embeddings.shape[1], 1).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            logits = classifier(embeddings)
            preds = torch.sigmoid(logits).squeeze(-1)
            
            # Calculate AUROC and AUPRC
            metrics = calculate_binary_classification_metrics(preds.cpu().numpy(), labels.cpu().numpy())
            
            # Log metrics
            self.log('val_readmission_auroc', metrics['auroc'], on_epoch=True)
            self.log('val_readmission_auprc', metrics['auprc'], on_epoch=True)
    
    def _evaluate_downstream_phecodes(self, embeddings, next_idx_padded, next_lens):
        """Evaluate PHEcode prediction task"""
        if not next_idx_padded or not next_lens or not self.phe_code_size:
            return
            
        try:
            # Concatenate indices and lengths
            indices = torch.cat(next_idx_padded, dim=0)
            lengths = torch.cat(next_lens, dim=0)
            
            # Prepare PHEcode targets
            targets, valid_samples = prepare_phecode_targets(
                {'next_idx_padded': indices, 'next_len': lengths}, 
                self.device, 
                self.phe_code_size
            )
            
            if targets is None or targets.shape[0] == 0:
                return
                
            # Get valid embeddings
            valid_embeddings = embeddings[valid_samples] if valid_samples is not None else embeddings
            
            # Create a simple linear classifier for PHEcode prediction
            classifier = torch.nn.Linear(valid_embeddings.shape[1], self.phe_code_size).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = classifier(valid_embeddings)
                preds = torch.sigmoid(logits)
                
                # Calculate metrics
                phecode_metrics = calculate_phecode_metrics(preds.cpu().numpy(), targets.cpu().numpy())
                
                # Log metrics
                self.log('val_phecode_micro_auc', phecode_metrics.get('micro_auc', 0.0), on_epoch=True)
                self.log('val_phecode_prec@5', phecode_metrics.get('prec@5', 0.0), on_epoch=True)
        except Exception as e:
            print(f"Error in PHEcode evaluation: {e}")
    
    def test_step(self, batch, batch_idx):
        """PyTorch Lightning test step - skipped as we use separate downstream evaluation"""
        # Skip detailed testing - proper downstream evaluation is done separately
        # in train_downstream_heads.py after training
        return None
    
    def configure_optimizers(self):
        """Configure optimizers for training with cosine annealing and warmup"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        
        # Get total number of training steps
        total_steps = self.trainer.estimated_stepping_batches
        
        # Number of warmup steps (typically 10% of total steps)
        warmup_steps = int(0.1 * total_steps)
        
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






def get_model(args, data_module):
    """Get model based on args.model_type"""
    try:
        # Get dimensions from data module
        dims = get_model_dimensions(data_module)
        
        # Get variable embeddings
        var_embeddings = data_module.get_var_embeddings()
        
        # Set PHEcode size in args if the auxiliary loss is enabled
        if getattr(args, 'use_phecode_loss', False):
            # Get phecode_size directly from the dataset
            args.phe_code_size = data_module.dataset.phecode_size
            logging.info(f"Setting PHEcode size from dataset: {args.phe_code_size}")
            
            # Set default phecode loss weight if not set
            if not hasattr(args, 'phecode_loss_weight'):
                args.phecode_loss_weight = 0.2
                logging.info(f"Using default PHEcode loss weight: {args.phecode_loss_weight}")
        else:
            logging.info("PHEcode auxiliary loss is disabled")
        
        # Initialize model with dimensions
        model = RaindropContrastiveModel(args, dims=dims)
        
        # Set variable embeddings
        model.var_embeddings = var_embeddings
        
        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model initialized with {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
        
        return model
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        # Create model anyway - default dimensions will be used
        model = RaindropContrastiveModel(args)
        return model