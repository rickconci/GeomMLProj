import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import yaml
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F

# Add parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_utils import (
    seed_everything, get_device, calculate_binary_classification_metrics, 
    calculate_phecode_metrics, prepare_phecode_targets, calculate_phecode_loss,
    get_lightning_devices
)
from contrastive_experiments.RaindropContrastive_lightning import RaindropContrastiveModel
from ContrastiveDataloaderLighting import ContrastiveDataModule


def setup_logging(args):
    """Set up logging configuration"""
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"downstream_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Arguments: {args}")
    return log_file


class DownstreamHeads(nn.Module):
    """Task-specific heads for downstream tasks using the same architecture as multi_task_raindrop.py"""
    
    def __init__(self, feature_dim, hidden_dim, phe_code_size=None, device=None):
        """
        Initialize task-specific heads
        
        Args:
            feature_dim: Dimension of the input embeddings
            hidden_dim: Hidden dimension for task heads
            phe_code_size: Size of the PHE code vocabulary (None if not used)
            device: Device to place model on
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.phe_code_size = phe_code_size
        self.device = device
        
        # 1. Mortality prediction head (binary classification)
        self.mortality_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification
        )
        
        # 2. Readmission prediction head (binary classification)
        self.readmission_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification
        )
        
        # 3. PHE code prediction head (multi-label classification) - if phe_code_size is provided
        if phe_code_size is not None:
            phecode_bottleneck_dim = min(512, phe_code_size // 2)  # Create a bottleneck
            self.phecode_classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),  # Add dropout to prevent overfitting
                nn.Linear(hidden_dim, phecode_bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(phecode_bottleneck_dim, phe_code_size)
            )
        
        # Move to device if provided
        if device is not None:
            self.to(device)
    
    def forward(self, embeddings, task=None):
        """
        Forward pass
        
        Args:
            embeddings: Input embeddings [batch_size, feature_dim]
            task: Specific task to run ('mortality', 'readmission', 'phecodes', or None for all)
            
        Returns:
            dict: Dictionary of predictions for each task
        """
        if task == 'mortality':
            return {'mortality': self.mortality_classifier(embeddings)}
        elif task == 'readmission':
            return {'readmission': self.readmission_classifier(embeddings)}
        elif task == 'phecodes' and hasattr(self, 'phecode_classifier'):
            return {'phecodes': self.phecode_classifier(embeddings)}
        else:
            # Run all tasks
            results = {
                'mortality': self.mortality_classifier(embeddings),
                'readmission': self.readmission_classifier(embeddings)
            }
            
            if hasattr(self, 'phecode_classifier'):
                results['phecodes'] = self.phecode_classifier(embeddings)
                
            return results


class DownstreamTasksModule(pl.LightningModule):
    """PyTorch Lightning module for training downstream task heads on frozen contrastive embeddings"""
    
    def __init__(self, args, contrastive_model=None):
        """Initialize DownstreamTasksModule with contrastive model and downstream heads"""
        super().__init__()
        
        # Save arguments
        self.args = args
        self.save_hyperparameters(ignore=['contrastive_model'])
        
        # Initialize contrastive model from checkpoint or use provided one
        if contrastive_model is None:
            # Load from checkpoint
            self.contrastive_model = self.load_checkpoint(args.contrastive_checkpoint)
        else:
            self.contrastive_model = contrastive_model
            
        # Freeze the contrastive model
        for param in self.contrastive_model.parameters():
            param.requires_grad = False
        
        # Determine embedding dimension
        if hasattr(args, 'embedding_dim') and args.embedding_dim is not None:
            self.embedding_dim = args.embedding_dim
        else:
            # Try to extract embedding dimension from model or checkpoint
            checkpoint_path = getattr(args, 'contrastive_checkpoint', None)
            self.embedding_dim = self._determine_embedding_dim(self.contrastive_model, checkpoint_path)
            
        logging.info(f"Using embedding dimension: {self.embedding_dim}")
            
        # Adjust embedding dimension based on embedding type
        if getattr(args, 'embedding_type', None) == 'concat':
            # If concatenating TS and text embeddings, double the dimension
            self.final_embedding_dim = self.embedding_dim * 2
            logging.info(f"Using concatenated embeddings with dimension: {self.final_embedding_dim}")
        else:
            self.final_embedding_dim = self.embedding_dim
        
        # Initialize task heads with the final embedding dimension
        self.task_heads = DownstreamHeads(
            feature_dim=self.final_embedding_dim,
            hidden_dim=args.hidden_dim,
            phe_code_size=getattr(self, 'phe_code_size', None),
            device=self.device
        )
        
        # Initialize metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Initialize storage for validation and test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint and return the instantiated model"""
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint path does not exist: {checkpoint_path}")
            logging.warning("Creating a minimal fallback model")
            return self._create_minimal_fallback()
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load the checkpoint (load to CPU to be safe)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Extract state dictionary and hyperparameters
            if 'state_dict' in checkpoint:
                # Standard PyTorch Lightning checkpoint
                state_dict = checkpoint['state_dict']
                # Check for PL hyperparameters
                if 'hyper_parameters' in checkpoint:
                    hparams = checkpoint['hyper_parameters']
                else:
                    hparams = {}
            else:
                # Assume the checkpoint is a direct state dict
                state_dict = checkpoint
                hparams = {}
            
            # Extract model dimensions from state_dict
            dims = self._extract_dims_from_state_dict(state_dict)
            
            # Logging the dimensions we found
            for key, val in dims.items():
                logging.info(f"Extracted dimension {key}: {val}")
            
            # Create model with appropriate dimensions
            model = self._create_model_with_dims(dims)
            
            # Load state dictionary with flexible matching
            success = self._load_state_dict_flexible(model, state_dict)
            if not success:
                logging.warning("Failed to load checkpoint, using minimal fallback model")
                return self._create_minimal_fallback()
            
            # Return the loaded model
            return model
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            logging.warning("Creating a minimal fallback model")
            return self._create_minimal_fallback()
    
    def _extract_dims_from_state_dict(self, state_dict):
        """Extract model dimensions from state dictionary by examining parameter shapes"""
        dims = {}
        
        # *** IMPORTANT: Hard code variables_num to 80 as confirmed by the user ***
        dims['variables_num'] = 80
        logging.info(f"Using fixed variables_num=80 as specified")
        
        # Try to determine projection dimension from projection layer weights
        if 'ts_projection.projection.0.weight' in state_dict:
            # Get projection output dimension from last layer
            weight = state_dict['ts_projection.projection.0.weight']
            dims['projection_dim'] = weight.size(0)
            logging.info(f"Found projection dimension: {dims['projection_dim']}")
        
        # Try to determine R_u dimension to verify our variables_num assumption
        if 'ts_model.R_u' in state_dict:
            r_u_shape = state_dict['ts_model.R_u'].shape
            if len(r_u_shape) > 0:
                r_u_dim = r_u_shape[-1]
                expected_r_u_dim = dims['variables_num'] * 3  # Typical Raindrop config, variables_num * 3
                if r_u_dim != expected_r_u_dim:
                    logging.warning(f"Found R_u with dimension {r_u_dim}, which doesn't match the expected {expected_r_u_dim} from variables_num={dims['variables_num']}")
                    logging.warning("This might cause tensor dimension mismatches")
        
        # Try to determine d_model from transformer weights
        if any(k.startswith('ts_model.') for k in state_dict):
            # Find transformer parameters
            for key, value in state_dict.items():
                if 'ts_model' in key and 'weight' in key:
                    if len(value.shape) == 2:
                        # Likely a linear layer
                        dims['d_model'] = max(dims.get('d_model', 0), value.size(0), value.size(1))
        
        # Try to determine number of heads from attention weights
        for key, value in state_dict.items():
            if 'self_attn' in key and 'in_proj_weight' in key:
                # Use attention layer to infer heads
                attn_dim = value.size(0) // 3  # in_proj_weight concatenates Q,K,V
                # Heads should divide attention dimension evenly
                for heads in [8, 4, 2, 1]:
                    if attn_dim % heads == 0:
                        dims['num_heads'] = heads
                        logging.info(f"Inferred {heads} attention heads from attention dimension {attn_dim}")
                        break
        
        # Default dimensions if not found
        dims.setdefault('projection_dim', 256)
        dims.setdefault('d_model', 256)
        dims.setdefault('timestamps', 80)     # Standard time steps
        dims.setdefault('d_static', 83)        # Default static dimension
        dims.setdefault('num_heads', 2)       # Default to 2 heads
        
        # Calculate expected R_u dimension based on variables_num
        expected_features = dims['variables_num'] * 3  # In Raindrop, each variable has 3 features
        logging.info(f"Expected feature dimension for R_u: {expected_features}")
        
        # Ensure d_model is properly divisible by variables_num
        if dims['d_model'] % dims['variables_num'] != 0:
            old_d_model = dims['d_model']
            # Round up to nearest multiple of variables_num
            dims['d_model'] = ((dims['d_model'] // dims['variables_num']) + 1) * dims['variables_num']
            logging.info(f"Adjusted d_model from {old_d_model} to {dims['d_model']} to be divisible by variables_num={dims['variables_num']}")
        
        return dims
    
    def _create_model_with_dims(self, dims):
        """Create a new model with the given dimensions extracted from checkpoint"""
        # Make sure variables_num is set to 80
        dims['variables_num'] = 80
        
        # Create a namespace object for model args
        class ArgsNamespace:
            def __init__(self, params_dict):
                for key, val in params_dict.items():
                    setattr(self, key, val)
                    
                # Ensure required attributes exist
                if not hasattr(self, 'hidden_dim'):
                    self.hidden_dim = 256
                if not hasattr(self, 'projection_dim'):
                    self.projection_dim = 256
                if not hasattr(self, 'd_model'):
                    self.d_model = 256
                if not hasattr(self, 'nlayers'):
                    self.nlayers = 2
                if not hasattr(self, 'num_heads'):
                    self.num_heads = 2
                if not hasattr(self, 'temperature'):
                    self.temperature = 0.1
                if not hasattr(self, 'sensor_wise_mask'):
                    self.sensor_wise_mask = True
                if not hasattr(self, 'pooling_type'):
                    self.pooling_type = 'attention'
                if not hasattr(self, 'checkpoint_dir'):
                    self.checkpoint_dir = './checkpoints'
                if not hasattr(self, 'use_wandb'):
                    self.use_wandb = False
                if not hasattr(self, 'seed'):
                    self.seed = 42
                if not hasattr(self, 'contrastive_method'):
                    self.contrastive_method = 'clip'
        
        # Create model args
        model_args = ArgsNamespace({
            'hidden_dim': dims.get('hidden_dim', 256),
            'projection_dim': dims.get('projection_dim', 256),
            'd_model': dims.get('d_model', 256),
            'nlayers': getattr(self.args, 'nlayers', 2),
            'num_heads': dims.get('num_heads', 2),
            'temperature': 0.1,
            'sensor_wise_mask': True,
            'seed': getattr(self.args, 'seed', 42),
            'pooling_type': 'attention',
            'checkpoint_dir': './checkpoints',
            'use_wandb': False,
            'contrastive_method': 'clip'
        })
        
        # Log the dimensions we're using
        logging.info(f"Creating model with dimensions: variables_num={dims['variables_num']}, "
                     f"d_model={model_args.d_model}, projection_dim={model_args.projection_dim}")
        
        try:
            # Create RaindropContrastiveModel
            from contrastive_experiments.RaindropContrastive_lightning import RaindropContrastiveModel
            model = RaindropContrastiveModel(args=model_args, dims=dims)
            model.eval()  # Set to evaluation mode
            return model
        except ImportError:
            logging.error("Could not import RaindropContrastiveModel, trying fallback")
            return self._create_minimal_fallback()
        except Exception as e:
            logging.error(f"Error creating model: {str(e)}")
            return self._create_minimal_fallback()
    
    def _load_state_dict_flexible(self, model, state_dict):
        """Load state dict with flexible matching that can handle architectural differences"""
        model_state_dict = model.state_dict()
        strict_load = getattr(self.args, 'strict_load', False)
        
        if strict_load:
            # Use strict loading mode (will likely fail with dimension mismatches)
            logging.info("Using strict loading mode")
            try:
                model.load_state_dict(state_dict, strict=True)
                logging.info("Successfully loaded checkpoint with strict loading")
                return True
            except Exception as e:
                logging.error(f"Failed to load checkpoint with strict loading: {str(e)}")
                return False
        
        # Use flexible loading mode
        logging.info("Using flexible loading mode")
        
        # Filter out incompatible keys due to dimension mismatches
        compatible_state_dict = {}
        mismatched_keys = []
        missing_keys = []
        
        for key, checkpoint_param in state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                if checkpoint_param.shape == model_param.shape:
                    # Perfect shape match
                    compatible_state_dict[key] = checkpoint_param
                elif key == 'ts_model.R_u' or key.endswith('.R_u'):
                    # Special handling for the R_u tensor which is critical for Raindrop
                    logging.warning(f"R_u dimension mismatch: checkpoint={checkpoint_param.shape}, model={model_param.shape}")
                    
                    # Calculate expected dimensions based on variables_num=80
                    expected_features = 80 * 3  # variables_num * 3 = 240
                    
                    # Try to adjust the R_u tensor
                    try:
                        if len(checkpoint_param.shape) == 3 and len(model_param.shape) == 3:
                            # Both are 3D tensors - we expect shapes like [T, B, features]
                            chk_T, chk_B, chk_F = checkpoint_param.shape
                            mod_T, mod_B, mod_F = model_param.shape
                            
                            # Check if only the feature dimension (last) is different
                            if chk_T == mod_T and chk_B == mod_B:
                                # Create a new tensor of the model's shape
                                adjusted = torch.zeros_like(model_param)
                                
                                if chk_F < mod_F:
                                    # Checkpoint has fewer features - pad with zeros
                                    adjusted[:, :, :chk_F] = checkpoint_param
                                    logging.info(f"Padded R_u from {chk_F} to {mod_F} features")
                                else:
                                    # Checkpoint has more features - truncate
                                    adjusted = checkpoint_param[:, :, :mod_F]
                                    logging.info(f"Truncated R_u from {chk_F} to {mod_F} features")
                                
                                compatible_state_dict[key] = adjusted
                            else:
                                # Different T or B dimensions - more complex adjustment needed
                                logging.warning(f"Cannot adjust R_u with different T or B dimensions. Expected: {model_param.shape}, Got: {checkpoint_param.shape}")
                                mismatched_keys.append(key)
                        else:
                            logging.warning(f"Unexpected R_u dimension count. Expected 3D tensor, got {len(checkpoint_param.shape)}D vs {len(model_param.shape)}D")
                            mismatched_keys.append(key)
                    except Exception as e:
                        logging.error(f"Error adjusting R_u parameter: {str(e)}")
                        mismatched_keys.append(key)
                elif ('weight' in key or 'bias' in key) and len(checkpoint_param.shape) <= 2:
                    # Handle linear layer weights/biases with different dimensions
                    logging.warning(f"Parameter dimension mismatch for {key}: checkpoint={checkpoint_param.shape}, model={model_param.shape}")
                    
                    try:
                        # For 2D weights (linear layers)
                        if len(checkpoint_param.shape) == 2 and len(model_param.shape) == 2:
                            # Only handle if the input dimension matches
                            if checkpoint_param.shape[1] == model_param.shape[1]:
                                # Match the output dimension if needed
                                adjusted = torch.zeros_like(model_param)
                                min_dim = min(checkpoint_param.shape[0], model_param.shape[0])
                                adjusted[:min_dim, :] = checkpoint_param[:min_dim, :]
                                compatible_state_dict[key] = adjusted
                                logging.info(f"Adjusted {key} from {checkpoint_param.shape} to {model_param.shape}")
                            else:
                                mismatched_keys.append(key)
                        # For 1D biases
                        elif len(checkpoint_param.shape) == 1 and len(model_param.shape) == 1:
                            # Match the dimension if needed
                            adjusted = torch.zeros_like(model_param)
                            min_dim = min(checkpoint_param.shape[0], model_param.shape[0])
                            adjusted[:min_dim] = checkpoint_param[:min_dim]
                            compatible_state_dict[key] = adjusted
                            logging.info(f"Adjusted {key} from {checkpoint_param.shape} to {model_param.shape}")
                        else:
                            mismatched_keys.append(key)
                    except Exception as e:
                        logging.error(f"Error adjusting {key}: {str(e)}")
                        mismatched_keys.append(key)
                else:
                    # Other parameters with shape mismatch - skip them
                    mismatched_keys.append(key)
            else:
                # Keys not in the current model are extra keys
                pass
        
        # Check for missing keys in the checkpoint
        for key in model_state_dict.keys():
            if key not in state_dict and key not in compatible_state_dict:
                missing_keys.append(key)
                
        # Log the mismatched and missing keys
        if mismatched_keys:
            logging.warning(f"Skipped {len(mismatched_keys)} parameters due to dimension mismatches")
            if len(mismatched_keys) <= 10:
                logging.warning(f"Mismatched keys: {mismatched_keys}")
            else:
                logging.warning(f"First 10 mismatched keys: {mismatched_keys[:10]}")
        
        if missing_keys:
            logging.warning(f"{len(missing_keys)} keys missing from checkpoint")
            if len(missing_keys) <= 10:
                logging.warning(f"Missing keys: {missing_keys}")
            else:
                logging.warning(f"First 10 missing keys: {missing_keys[:10]}")
        
        # Load the compatible parameters
        if compatible_state_dict:
            model.load_state_dict(compatible_state_dict, strict=False)
            logging.info(f"Loaded {len(compatible_state_dict)} parameters from checkpoint")
            return True
        else:
            logging.error("No compatible parameters found in checkpoint")
            return False
    
    def _get_embedding_dim(self, model, hparams, dims):
        """Get embedding dimension for downstream tasks, with fallbacks"""
        # Try several methods to determine embedding dimension
        embedding_dim = None
        
        # Method 1: Check model.args
        if hasattr(model, 'args') and hasattr(model.args, 'projection_dim'):
            embedding_dim = model.args.projection_dim
            logging.info(f"Found embedding dim from model.args: {embedding_dim}")
        
        # Method 2: Check hparams
        if embedding_dim is None and 'projection_dim' in hparams:
            embedding_dim = hparams['projection_dim']
            logging.info(f"Found embedding dim from hparams: {embedding_dim}")
        
        # Method 3: Use extracted dims
        if embedding_dim is None and 'projection_dim' in dims:
            embedding_dim = dims['projection_dim']
            logging.info(f"Found embedding dim from extracted dims: {embedding_dim}")
        
        # Method 4: Check text projection layer if it exists
        if embedding_dim is None and hasattr(model, 'text_projection'):
            for name, param in model.text_projection.named_parameters():
                if 'weight' in name:
                    if hasattr(param, 'shape'):
                        embedding_dim = param.shape[0]  # Output dimension
                        logging.info(f"Found embedding dim from text_projection: {embedding_dim}")
                        break
        
        # Method 5: Default fallback
        if embedding_dim is None:
            embedding_dim = 256
            logging.warning(f"Could not determine embedding dimension, using default: {embedding_dim}")
        
        return embedding_dim
    
    def _create_minimal_fallback(self):
        """Create a minimal model that matches the expected interface"""
        logging.info("Creating minimal fallback model")
        
        # Set standard minimal dimensions
        dims = {
            'projection_dim': 256,
            'd_model': 256,
            'variables_num': 75,
            'd_static': 0,
            'timestamps': 48,
            'num_heads': 2
        }
        
        # Create model with minimal configuration
        try:
            logging.info("Attempting to create RaindropContrastiveModel with standard dimensions")
            # Get tsmodel_type from args
            tsmodel_type = getattr(self.args, 'tsmodel_type', 'raindrop_v2')
            
            # Create the model
            model = self._create_model_with_dims(dims)
            return model
        except Exception as e:
            logging.error(f"Error creating minimal fallback model: {str(e)}")
            logging.warning("Using extremely minimal stub model - functionality will be limited")
            
            # Create an extremely minimal wrapper that has the required methods
            class MinimalStubModel(nn.Module):
                def __init__(self, output_dim=256):
                    super().__init__()
                    self.output_dim = output_dim
                    self.linear = nn.Linear(10, output_dim)
                
                def forward(self, batch_data):
                    # Return zero embeddings that match the expected format
                    batch_size = 1
                    if isinstance(batch_data, dict) and 'P' in batch_data:
                        if hasattr(batch_data['P'], 'shape'):
                            batch_size = batch_data['P'].shape[0]
                    
                    # Create zero embeddings
                    device = next(self.parameters()).device
                    ts_proj = torch.zeros((batch_size, self.output_dim), device=device)
                    text_proj = torch.zeros((batch_size, self.output_dim), device=device)
                    
                    return ts_proj, text_proj, None
            
            return MinimalStubModel()
    
    def extract_embeddings(self, batch):
        """Extract embeddings from the loaded contrastive model"""
        # Prepare batch for the contrastive model
        batch_data = {}
        for k, v in batch.items():
            # Only keep tensor inputs that contrastive model expects
            if isinstance(v, torch.Tensor):
                batch_data[k] = v
        
        try:
            if hasattr(self.contrastive_model, 'model_forward'):
                # Standard interface with model_forward method
                ts_proj, text_proj, _ = self.contrastive_model.model_forward(batch_data)
            elif callable(self.contrastive_model):
                # Direct callable model (fallback model)
                ts_proj, text_proj, _ = self.contrastive_model(batch_data)
            else:
                # No compatible interface
                logging.error("Contrastive model doesn't have expected interface")
                # Create dummy embeddings as fallback
                device = next(self.contrastive_model.parameters()).device
                batch_size = batch_data['P'].shape[0] if 'P' in batch_data else 1
                ts_proj = torch.zeros((batch_size, self.embedding_dim), device=device)
                text_proj = torch.zeros((batch_size, self.embedding_dim), device=device)
            
            # Determine which embedding to use based on args
            if self.args.embedding_type == 'ts':
                embeddings = ts_proj
            elif self.args.embedding_type == 'text':
                embeddings = text_proj
            elif self.args.embedding_type == 'concat':
                embeddings = torch.cat([ts_proj, text_proj], dim=1)
            elif self.args.embedding_type == 'sum':
                embeddings = ts_proj + text_proj
            else:
                # Default to time series
                embeddings = ts_proj
                
            return embeddings
            
        except Exception as e:
            logging.error(f"Error extracting embeddings: {str(e)}")
            # Create fallback embeddings
            device = next(self.parameters()).device
            batch_size = batch_data['P'].shape[0] if 'P' in batch_data else 1
            return torch.zeros((batch_size, self.embedding_dim), device=device)
    
    def forward(self, batch):
        """Forward pass"""
        # Extract embeddings
        embeddings = self.extract_embeddings(batch)
        if embeddings is None:
            return None
        
        # Get predictions from task heads
        predictions = self.task_heads(embeddings)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Extract embeddings
        embeddings = self.extract_embeddings(batch)
        if embeddings is None:
            return None
        
        # Initialize total loss
        total_loss = 0
        batch_size = embeddings.size(0)
        
        # Mortality task
        if 'mortality_label' in batch:
            mortality_labels = batch['mortality_label'].float()
            mortality_logits = self.task_heads.mortality_classifier(embeddings)
            mortality_loss = F.binary_cross_entropy_with_logits(
                mortality_logits.squeeze(-1), mortality_labels
            )
            total_loss += mortality_loss
            self.log('train_mortality_loss', mortality_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Readmission task
        if 'readmission_label' in batch:
            readmission_labels = batch['readmission_label'].float()
            readmission_logits = self.task_heads.readmission_classifier(embeddings)
            readmission_loss = F.binary_cross_entropy_with_logits(
                readmission_logits.squeeze(-1), readmission_labels
            )
            total_loss += readmission_loss
            self.log('train_readmission_loss', readmission_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # PHEcode task
        if hasattr(self.task_heads, 'phecode_classifier') and 'next_idx_padded' in batch and 'next_len' in batch:
            # Prepare PHEcode targets
            phecode_targets, valid_samples = prepare_phecode_targets(
                batch, self.device, self.phe_code_size
            )
            
            if phecode_targets is not None:
                # Get embeddings for samples with valid PHEcodes
                valid_embeddings = embeddings[valid_samples] if valid_samples is not None else embeddings
                
                # Forward pass
                phecode_logits = self.task_heads.phecode_classifier(valid_embeddings)
                
                # Calculate loss
                phecode_loss = F.binary_cross_entropy_with_logits(phecode_logits, phecode_targets)
                
                # Add to total loss
                phecode_weight = getattr(self.args, 'phecode_loss_weight', 1.0)
                total_loss += phecode_weight * phecode_loss
                
                self.log('train_phecode_loss', phecode_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log total loss
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def on_validation_epoch_start(self):
        """Clear the validation step outputs at the start of each epoch"""
        self.validation_step_outputs = []
    
    def on_test_epoch_start(self):
        """Clear the test step outputs at the start of each epoch"""
        self.test_step_outputs = []
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Extract embeddings
        embeddings = self.extract_embeddings(batch)
        if embeddings is None:
            return None
        
        # Initialize metrics to track
        metrics = {}
        
        # Mortality task
        if 'mortality_label' in batch:
            mortality_labels = batch['mortality_label'].float()
            mortality_logits = self.task_heads.mortality_classifier(embeddings)
            mortality_loss = F.binary_cross_entropy_with_logits(
                mortality_logits.squeeze(-1), mortality_labels
            )
            
            # Get predictions for metrics
            mortality_preds = torch.sigmoid(mortality_logits.squeeze(-1))
            
            self.log('val_mortality_loss', mortality_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            # Return for epoch end metrics calculation
            metrics['mortality_preds'] = mortality_preds.detach()
            metrics['mortality_labels'] = mortality_labels.detach()
        
        # Readmission task
        if 'readmission_label' in batch:
            readmission_labels = batch['readmission_label'].float()
            readmission_logits = self.task_heads.readmission_classifier(embeddings)
            readmission_loss = F.binary_cross_entropy_with_logits(
                readmission_logits.squeeze(-1), readmission_labels
            )
            
            # Get predictions for metrics
            readmission_preds = torch.sigmoid(readmission_logits.squeeze(-1))
            
            self.log('val_readmission_loss', readmission_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            # Return for epoch end metrics calculation
            metrics['readmission_preds'] = readmission_preds.detach()
            metrics['readmission_labels'] = readmission_labels.detach()
        
        # PHEcode task
        if hasattr(self.task_heads, 'phecode_classifier') and 'next_idx_padded' in batch and 'next_len' in batch:
            # Prepare PHEcode targets
            phecode_targets, valid_samples = prepare_phecode_targets(
                batch, self.device, self.phe_code_size
            )
            
            if phecode_targets is not None:
                # Get embeddings for samples with valid PHEcodes
                valid_embeddings = embeddings[valid_samples] if valid_samples is not None else embeddings
                
                # Forward pass
                phecode_logits = self.task_heads.phecode_classifier(valid_embeddings)
                
                # Calculate loss
                phecode_loss = F.binary_cross_entropy_with_logits(phecode_logits, phecode_targets)
                
                # Get predictions for metrics
                phecode_preds = torch.sigmoid(phecode_logits)
                
                self.log('val_phecode_loss', phecode_loss, on_step=False, on_epoch=True, prog_bar=True)
                
                # Return for epoch end metrics calculation
                metrics['phecode_preds'] = phecode_preds.detach()
                metrics['phecode_labels'] = phecode_targets.detach()
        
        # Store for epoch end processing
        self.validation_step_outputs.append(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        """Test step (same as validation but logged differently)"""
        # Just reuse validation step logic
        metrics = self.validation_step(batch, batch_idx)
        
        # Store for epoch end processing
        self.test_step_outputs.append(metrics)
        
        # No need to log losses during test
        return metrics
    
    def on_validation_epoch_end(self):
        """Process validation outputs at the end of the epoch"""
        self._calculate_epoch_metrics(self.validation_step_outputs, prefix='val')
    
    def on_test_epoch_end(self):
        """Process test outputs at the end of the epoch"""
        self._calculate_epoch_metrics(self.test_step_outputs, prefix='test')
    
    def _calculate_epoch_metrics(self, outputs, prefix):
        """Calculate and log metrics at the end of an epoch"""
        # Collect all predictions and labels
        all_mortality_preds = []
        all_mortality_labels = []
        all_readmission_preds = []
        all_readmission_labels = []
        all_phecode_preds = []
        all_phecode_labels = []
        
        # Process all batch outputs
        for batch_metrics in outputs:
            if batch_metrics is None:
                continue
            
            # Mortality metrics
            if 'mortality_preds' in batch_metrics:
                all_mortality_preds.append(batch_metrics['mortality_preds'].cpu())
                all_mortality_labels.append(batch_metrics['mortality_labels'].cpu())
            
            # Readmission metrics
            if 'readmission_preds' in batch_metrics:
                all_readmission_preds.append(batch_metrics['readmission_preds'].cpu())
                all_readmission_labels.append(batch_metrics['readmission_labels'].cpu())
            
            # PHEcode metrics
            if 'phecode_preds' in batch_metrics:
                all_phecode_preds.append(batch_metrics['phecode_preds'].cpu())
                all_phecode_labels.append(batch_metrics['phecode_labels'].cpu())
        
        # Calculate metrics for binary tasks
        if all_mortality_preds:
            mortality_preds = torch.cat(all_mortality_preds).numpy()
            mortality_labels = torch.cat(all_mortality_labels).numpy()
            mortality_metrics = calculate_binary_classification_metrics(mortality_preds, mortality_labels)
            self.log(f'{prefix}_mortality_auroc', mortality_metrics['auroc'], prog_bar=True)
            self.log(f'{prefix}_mortality_auprc', mortality_metrics['auprc'], prog_bar=True)
        
        if all_readmission_preds:
            readmission_preds = torch.cat(all_readmission_preds).numpy()
            readmission_labels = torch.cat(all_readmission_labels).numpy()
            readmission_metrics = calculate_binary_classification_metrics(readmission_preds, readmission_labels)
            self.log(f'{prefix}_readmission_auroc', readmission_metrics['auroc'], prog_bar=True)
            self.log(f'{prefix}_readmission_auprc', readmission_metrics['auprc'], prog_bar=True)
        
        # Calculate metrics for PHEcode task
        if all_phecode_preds:
            try:
                phecode_preds = torch.cat(all_phecode_preds).numpy()
                phecode_labels = torch.cat(all_phecode_labels).numpy()
                phecode_metrics = calculate_phecode_metrics(phecode_preds, phecode_labels)
                
                self.log(f'{prefix}_phecode_macro_auc', phecode_metrics.get('macro_auc', 0.0))
                self.log(f'{prefix}_phecode_micro_auc', phecode_metrics.get('micro_auc', 0.0), prog_bar=True)
                self.log(f'{prefix}_phecode_micro_ap', phecode_metrics.get('micro_ap', 0.0))
                self.log(f'{prefix}_phecode_prec@5', phecode_metrics.get('prec@5', 0.0), prog_bar=True)
                
                # Store metrics for later access
                if prefix == 'val':
                    self.val_metrics.update({
                        'phecode_macro_auc': phecode_metrics.get('macro_auc', 0.0),
                        'phecode_micro_auc': phecode_metrics.get('micro_auc', 0.0),
                        'phecode_micro_ap': phecode_metrics.get('micro_ap', 0.0),
                        'phecode_prec@5': phecode_metrics.get('prec@5', 0.0)
                    })
                elif prefix == 'test':
                    self.test_metrics.update({
                        'phecode_macro_auc': phecode_metrics.get('macro_auc', 0.0),
                        'phecode_micro_auc': phecode_metrics.get('micro_auc', 0.0),
                        'phecode_micro_ap': phecode_metrics.get('micro_ap', 0.0),
                        'phecode_prec@5': phecode_metrics.get('prec@5', 0.0)
                    })
            except Exception as e:
                logging.warning(f"Error calculating PHE code metrics: {e}")
    
    def configure_optimizers(self):
        """Configure optimizers"""
        # Only optimize task heads (contrastive model is frozen)
        optimizer = torch.optim.Adam(self.task_heads.parameters(), lr=self.args.lr)
        
        # Add learning rate scheduler if desired
        if hasattr(self.args, 'use_scheduler') and self.args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
                },
            }
        else:
            return optimizer

    def _determine_embedding_dim(self, model, checkpoint_path=None):
        """Determine the embedding dimension from the model or checkpoint"""
        # Try to get from model directly first
        if hasattr(model, 'args') and hasattr(model.args, 'projection_dim'):
            return model.args.projection_dim
            
        # If model has ts_projection, get from there
        if hasattr(model, 'ts_projection') and hasattr(model.ts_projection, 'projection'):
            for layer in model.ts_projection.projection:
                if isinstance(layer, nn.Linear):
                    return layer.out_features
        
        # Try to determine from model state_dict 
        if hasattr(model, 'state_dict'):
            for name, param in model.state_dict().items():
                if 'projection' in name and 'weight' in name and len(param.shape) == 2:
                    # Projection layer weights are likely [output_dim, input_dim]
                    return param.shape[0]
        
        # Try to load from checkpoint if model inspection failed
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                # Look for projection layer in state dict
                for key, value in state_dict.items():
                    if 'projection' in key and 'weight' in key and len(value.shape) == 2:
                        return value.shape[0]
            except Exception as e:
                logging.error(f"Error extracting embedding dim from checkpoint: {str(e)}")
        
        # Default dimension if all else fails
        logging.warning("Could not determine embedding dimension, using default of 256")
        return 256


def train_downstream_tasks(args):
    """Main function for training downstream task heads using Lightning"""
    # Set up logging
    log_file = setup_logging(args)
    
    # Set random seeds
    seed_everything(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
    # Set embedding_type if not provided
    if not hasattr(args, 'embedding_type'):
        args.embedding_type = 'average'
        logging.info(f"Using default embedding type: {args.embedding_type}")
    
    # Create data module
    data_module = ContrastiveDataModule(
        data_path=args.data_path,
        temp_dfs_path=args.temp_dfs_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Prepare data
    data_module.prepare_data()
    data_module.setup()
    
    # Create Lightning module
    module = DownstreamTasksModule(args)
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="downstream-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Set up logger
    logger = None
    if args.use_wandb:
        try:
            logger = WandbLogger(
                project=args.wandb_project,
                name=args.wandb_run_name or f"downstream_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                save_dir=args.output_dir
            )
            logger.log_hyperparams(vars(args))
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            logging.info("Continuing without wandb logging")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        devices=get_lightning_devices(args.devices),
        accelerator='auto',
        check_val_every_n_epoch=1,
        log_every_n_steps=10
    )
    
    # Train model
    trainer.fit(module, data_module)
    
    # Test best model
    best_model_path = checkpoint_callback.best_model_path
    logging.info(f"Best model saved to: {best_model_path}")
    
    if args.test_after_training:
        logging.info(f"Testing best model from {best_model_path}")
        trainer.test(ckpt_path=best_model_path, datamodule=data_module)
    
    # Save metrics
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Convert any tensors to Python values
    for k, v in module.val_metrics.items():
        if isinstance(v, torch.Tensor):
            module.val_metrics[k] = v.item()
    
    for k, v in module.test_metrics.items():
        if isinstance(v, torch.Tensor):
            module.test_metrics[k] = v.item()
    
    # Save metrics
    with open(metrics_dir / "val_metrics.json", "w") as f:
        json.dump(module.val_metrics, f, indent=2)
    
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(module.test_metrics, f, indent=2)
    
    logging.info(f"Saved metrics to {metrics_dir}")
    logging.info("Training completed")
    
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train downstream task heads on frozen contrastive embeddings")
    
    # Model and checkpoint
    parser.add_argument("--contrastive_checkpoint", type=str, required=True, 
                        help="Path to pre-trained contrastive model checkpoint")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="/path/to/mimic/data", 
                        help="Path to MIMIC-IV data")
    parser.add_argument("--temp_dfs_path", type=str, default="temp_dfs_lite", 
                        help="Path to cache directory")
    parser.add_argument("--output_dir", type=str, default="./downstream_results", 
                        help="Directory to save results")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--hidden_dim", type=int, default=256, 
                        help="Hidden dimension for task heads")
    parser.add_argument("--phecode_loss_weight", type=float, default=1.0,
                        help="Weight for PHEcode loss")
    parser.add_argument("--early_stopping", action="store_true", 
                        help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use learning rate scheduler")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--devices", type=int, default=None, 
                        help="Number of devices to use")
    parser.add_argument("--test_after_training", action="store_true",
                        help="Run test after training completes")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="downstream_tasks", 
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, 
                        help="WandB run name")
    
    # Model type and loading parameters
    parser.add_argument("--model_type", type=str, default="raindrop_v2",
                        help="Model type (for backwards compatibility)")
    parser.add_argument("--strict_load", type=lambda x: x.lower() == 'true', default=False,
                        help="Whether to strictly enforce that the keys in the checkpoint match the model (true/false)")
    parser.add_argument("--embedding_type", type=str, default="average", choices=["ts", "text", "average", "concat", "sum"],
                        help="Type of embedding to use for downstream tasks")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Override embedding dimension (if not specified, will be inferred from checkpoint)")
    
    args = parser.parse_args()
    train_downstream_tasks(args) 