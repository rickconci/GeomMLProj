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

        # Downstream heads take the concat of ts and text projections
        self.next_phecode_predictor = nn.Sequential(
            nn.Linear(self.args.projection_dim*2, self.args.projection_dim),
            nn.ReLU(),
            nn.Linear(self.args.projection_dim, self.phe_code_size)
        )
        
        self.mortality_classifier = nn.Sequential(
            nn.Linear(self.args.projection_dim*2, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, 1)  # Binary classification
        )

        self.readmission_classifier = nn.Sequential(
            nn.Linear(self.args.projection_dim*2, self.args.hidden_dim),
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
            'readmission_label': batch['readmission_label'].to(self.device, dtype=torch.float32)
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
        """PyTorch Lightning validation step - collect embeddings for downstream evaluation"""
        # Only collect embeddings if we're on a validation epoch we care about
        
        # Prepare batch data
        batch_data = self.prepare_batch(batch)
        if batch_data is None:
            return None
        
        # Forward pass through model to get embeddings (with no gradients)
        with torch.no_grad():
            ts_proj, text_proj, _ = self.model_forward(batch_data)
        
        # Collect embeddings and labels
        result = {
            'ts_proj': ts_proj.detach(),
            'text_proj': text_proj.detach(),
            'mortality_label': batch_data['mortality_label'],
            'readmission_label': batch_data['readmission_label'],
            'next_idx_padded': batch_data['next_idx_padded'],
            'next_phecode_len': batch_data['next_phecode_len']
        }
        
        self.validation_step_outputs.append(result)
        return result
    
    def on_validation_epoch_start(self):
        """Initialize list to store validation step outputs if needed"""
        
        logging.info(f"Running downstream evaluation at epoch {self.trainer.current_epoch+1}")
        self.validation_step_outputs = []
        
        # Reset weights of downstream prediction heads
        self._reset_classifier_weights(self.mortality_classifier)
        self._reset_classifier_weights(self.readmission_classifier)
        self._reset_classifier_weights(self.next_phecode_predictor)
    
    def _reset_classifier_weights(self, model):
        """Reset the weights of a model to their initial values"""
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                # Reset weights using Xavier/Glorot initialization
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def on_validation_epoch_end(self):
        """Train and evaluate downstream tasks at the end of validation epoch"""
        
        outputs = self.validation_step_outputs
        if not outputs:
            return
        
        # ALL_GATHER to collect data from all processes
        all_gathered_ts_projs = self.all_gather([o['ts_proj'] for o in outputs if o is not None and 'ts_proj' in o])
        all_gathered_text_projs = self.all_gather([o['text_proj'] for o in outputs if o is not None and 'text_proj' in o])
        all_gathered_mortality_labels = self.all_gather([o['mortality_label'] for o in outputs if o is not None and 'mortality_label' in o])
        all_gathered_readmission_labels = self.all_gather([o['readmission_label'] for o in outputs if o is not None and 'readmission_label' in o])
        all_gathered_next_idx_padded = self.all_gather([o['next_idx_padded'] for o in outputs if o is not None and 'next_idx_padded' in o])
        all_gathered_next_phecode_len = self.all_gather([o['next_phecode_len'] for o in outputs if o is not None and 'next_phecode_len' in o])
        
        # Only rank 0 processes the gathered data
        if self.global_rank == 0:
            # Flatten gathered lists and concatenate tensors
            ts_projs = torch.cat([item for sublist in all_gathered_ts_projs for item in sublist], dim=0)
            text_projs = torch.cat([item for sublist in all_gathered_text_projs for item in sublist], dim=0)
            concat_projs = torch.cat([ts_projs, text_projs], dim=1)
            
            mortality_labels = torch.cat([item for sublist in all_gathered_mortality_labels for item in sublist], dim=0)
            readmission_labels = torch.cat([item for sublist in all_gathered_readmission_labels for item in sublist], dim=0)
            next_idx_padded = torch.cat([item for sublist in all_gathered_next_idx_padded for item in sublist], dim=0)
            next_lens = torch.cat([item for sublist in all_gathered_next_phecode_len for item in sublist], dim=0)
            
            # Train and evaluate downstream tasks only on rank 0
            all_metrics = {'epoch': self.trainer.current_epoch}
            
            mortality_metrics = self._train_and_evaluate_mortality(concat_projs, [mortality_labels])
            readmission_metrics = self._train_and_evaluate_readmission(concat_projs, [readmission_labels])
            phecode_metrics = self._train_and_evaluate_phecodes(concat_projs, [next_idx_padded], [next_lens])
            
            # Update metrics dictionary
            if mortality_metrics:
                all_metrics.update({f'mortality_{k}': v for k, v in mortality_metrics.items()})
            if readmission_metrics:
                all_metrics.update({f'readmission_{k}': v for k, v in readmission_metrics.items()})
            if phecode_metrics:
                all_metrics.update({f'phecode_{k}': v for k, v in phecode_metrics.items()})
            
            # Log to wandb if available
            if self.args.use_wandb:
                wandb_metrics = {
                    'epoch': self.trainer.current_epoch,
                    'val_downstream/mortality_auroc': all_metrics.get('mortality_auroc', 0.0),
                    'val_downstream/mortality_auprc': all_metrics.get('mortality_auprc', 0.0),
                    'val_downstream/readmission_auroc': all_metrics.get('readmission_auroc', 0.0),
                    'val_downstream/readmission_auprc': all_metrics.get('readmission_auprc', 0.0),
                    'val_downstream/phecode_micro_auc': all_metrics.get('phecode_micro_auc', 0.0),
                    'val_downstream/phecode_macro_auc': all_metrics.get('phecode_macro_auc', 0.0),
                    'val_downstream/phecode_micro_ap': all_metrics.get('phecode_micro_ap', 0.0),
                    'val_downstream/phecode_prec@5': all_metrics.get('phecode_prec@5', 0.0),
                }
                wandb.log(wandb_metrics, step=self.trainer.current_epoch)
    
    def _train_classifier(self, classifier, embeddings, targets, is_multilabel=False, valid_samples=None):
        """
        Generic training method for downstream classifiers
        
        Args:
            classifier: The classifier model to train
            embeddings: Embeddings to use for training
            targets: Target labels
            is_multilabel: Whether this is a multi-label classification task
            valid_samples: Optional tensor of indices for valid samples (for PHE codes)
        
        Returns:
            Predictions and targets for metrics calculation
        """

        torch.set_grad_enabled(True)
        
        # Get valid embeddings and targets if needed
        if valid_samples is not None:
            valid_embeddings = embeddings[valid_samples]
            if is_multilabel:
                # For PHE codes, targets are already prepared
                valid_targets = targets
            else:
                # For binary tasks, need to select valid targets
                valid_targets = targets[valid_samples]
        else:
            valid_embeddings = embeddings
            valid_targets = targets
            
        # Ensure embeddings have requires_grad=True for training
        # Key fix: we need a fresh tensor not connected to the original computation graph
        valid_embeddings = torch.nn.Parameter(valid_embeddings.clone().detach(), requires_grad=True)
            
        # Setup optimizer
        # Only optimize the classifier parameters, not the embeddings
        optimizer = torch.optim.Adam([
            {'params': classifier.parameters()}
        ], lr=0.001)
        
        # Define loss function
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Train the classifier on the embeddings
        num_epochs = 3
        batch_size = 256
        n_samples = valid_embeddings.size(0)
        
        classifier.train()
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = torch.randperm(n_samples, device=self.device)
            total_loss = 0.0
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                if i >= end_idx:
                    continue
                    
                idx = indices[i:end_idx]
                batch_embeddings = valid_embeddings[idx]
                batch_targets = valid_targets[idx]
                
                # Forward pass
                logits = classifier(batch_embeddings)
                
                # Apply squeeze for binary classification tasks
                if not is_multilabel:
                    logits = logits.squeeze(-1)
                    
                loss = criterion(logits, batch_targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Log training progress every few epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                avg_loss = total_loss / max(1, (n_samples // batch_size))
                logging.info(f"Validation classifier training - Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Evaluate the trained classifier
        classifier.eval()
        with torch.no_grad():
            logits = classifier(valid_embeddings)
            preds = torch.sigmoid(logits)
            
            # Return predictions and targets for metrics calculation
            return preds, valid_targets
    
    def _train_and_evaluate_mortality(self, embeddings, mortality_labels):
        """Train and evaluate mortality prediction task"""
        if not mortality_labels:
            return None
        
        # Concatenate labels
        labels = torch.cat(mortality_labels, dim=0)
        
        # Skip if no positive samples
        if labels.sum() == 0:
            return None
        
        # Use the existing mortality classifier
        classifier = self.mortality_classifier
        
        # Train classifier and get predictions
        preds, targets = self._train_classifier(
            classifier=classifier,
            embeddings=embeddings,
            targets=labels,
            is_multilabel=False
        )
        
        # Ensure predictions are flattened for binary tasks
        preds = preds.squeeze(-1)
        
        # Use the utility function from train_utils to calculate metrics
        metrics = calculate_binary_classification_metrics(preds.cpu().numpy(), targets.cpu().numpy())
        
        # Log metrics
        self.log('val_mortality_auroc', metrics['auroc'], on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
        self.log('val_mortality_auprc', metrics['auprc'], on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
        
        logging.info(f"Mortality classifier performance - AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
        
        # Return metrics for wandb logging
        return metrics
    
    def _train_and_evaluate_readmission(self, embeddings, readmission_labels):
        """Train and evaluate readmission prediction task"""
        if not readmission_labels:
            return None
        
        # Concatenate labels
        labels = torch.cat(readmission_labels, dim=0)
        
        # Skip if no positive samples
        if labels.sum() == 0:
            return None
        
        # Use the existing readmission classifier
        classifier = self.readmission_classifier
        
        # Train classifier and get predictions
        preds, targets = self._train_classifier(
            classifier=classifier,
            embeddings=embeddings,
            targets=labels,
            is_multilabel=False
        )
        
        # Ensure predictions are flattened for binary tasks
        preds = preds.squeeze(-1)
        
        # Use the utility function from train_utils to calculate metrics
        metrics = calculate_binary_classification_metrics(preds.cpu().numpy(), targets.cpu().numpy())
        
        # Log metrics
        self.log('val_readmission_auroc', metrics['auroc'], on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
        self.log('val_readmission_auprc', metrics['auprc'], on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
        
        logging.info(f"Readmission classifier performance - AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
        
        # Return metrics for wandb logging
        return metrics
    
    def _train_and_evaluate_phecodes(self, embeddings, next_idx_padded, next_lens):
        """Train and evaluate PHEcode prediction task"""
        if not next_idx_padded or not next_lens or not self.phe_code_size:
            logging.info(f"Skipping PHEcode evaluation - missing data or phe_code_size")
            return None
            
        try:
            # Concatenate indices and lengths
            indices = torch.cat(next_idx_padded, dim=0)
            lengths = torch.cat(next_lens, dim=0)
            
            # Log key information about the indices
            max_idx = indices.max().item() if indices.numel() > 0 else 0
            min_idx = indices.min().item() if indices.numel() > 0 else 0
            logging.info(f"PHEcode indices range: min={min_idx}, max={max_idx}, phe_code_size={self.phe_code_size}")
            
            # Check for out-of-bounds indices
            if max_idx >= self.phe_code_size:
                logging.warning(f"PHEcode indices out of bounds: max index {max_idx} >= phecode size {self.phe_code_size}")
                logging.warning(f"Consider increasing phe_code_size to at least {max_idx + 1}")
                return None
            
            # Check if there are valid codes
            if lengths.sum() == 0:
                logging.info("No PHEcode targets available for evaluation")
                return None
            
            # Use the prepare_phecode_targets function from train_utils
            try:
                targets, valid_samples = prepare_phecode_targets(
                    {'next_idx_padded': indices, 'next_len': lengths}, 
                    self.device, 
                    self.phe_code_size
                )
            except Exception as e:
                logging.error(f"Error preparing PHEcode targets: {e}")
                return None
            
            if targets is None or targets.shape[0] == 0:
                logging.info("No valid PHEcode targets generated")
                return None
                
            # Check that valid_samples indices are in bounds
            if valid_samples is not None and valid_samples.max().item() >= embeddings.shape[0]:
                logging.warning(f"Valid samples indices out of bounds: max index {valid_samples.max().item()} >= embeddings shape {embeddings.shape[0]}")
                return None
                
            # Verify dimensions
            logging.info(f"Embeddings shape: {embeddings.shape}, PHEcode size: {self.phe_code_size}")
            
            # Use the existing next_phecode_predictor
            classifier = self.next_phecode_predictor
            
            # Train classifier and get predictions
            preds, targets = self._train_classifier(
                classifier=classifier,
                embeddings=embeddings,
                targets=targets,
                is_multilabel=True,
                valid_samples=valid_samples
            )
            
            # Use the utility function from train_utils to calculate metrics
            phecode_metrics = calculate_phecode_metrics(preds.cpu().numpy(), targets.cpu().numpy())
            
            # Log metrics
            self.log('val_phecode_micro_auc', phecode_metrics.get('micro_auc', 0.0), on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
            self.log('val_phecode_macro_auc', phecode_metrics.get('macro_auc', 0.0), on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
            self.log('val_phecode_micro_ap', phecode_metrics.get('micro_ap', 0.0), on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
            self.log('val_phecode_prec@5', phecode_metrics.get('prec@5', 0.0), on_epoch=True, sync_dist=True, batch_size=embeddings.shape[0])
            
            # Log detailed metrics
            logging.info(f"PHEcode classifier performance:")
            logging.info(f"  - Micro AUC: {phecode_metrics.get('micro_auc', 0.0):.4f}")
            logging.info(f"  - Macro AUC: {phecode_metrics.get('macro_auc', 0.0):.4f}")
            logging.info(f"  - Micro AP: {phecode_metrics.get('micro_ap', 0.0):.4f}")
            logging.info(f"  - Precision@5: {phecode_metrics.get('prec@5', 0.0):.4f}")
            
            # Return metrics for wandb logging
            return phecode_metrics
            
        except Exception as e:
            logging.error(f"Error in PHEcode evaluation: {e}")
            # Add detailed traceback for debugging
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Don't rethrow the exception - just log and continue
            return None
    
    def test_step(self, batch, batch_idx):
        """PyTorch Lightning test step - skipped as we use separate downstream evaluation"""
        # Skip detailed testing - proper downstream evaluation is done separately
        # in train_downstream_heads.py after training
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



