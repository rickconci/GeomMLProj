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
        """
        Initialize Lightning module
        
        Args:
            args: Command line arguments
            contrastive_model: Pre-trained contrastive model (loaded separately if None)
        """
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['contrastive_model'])
        
        # Set up contrastive model (encoder)
        if contrastive_model is None:
            # Load from checkpoint
            self.contrastive_model = self.load_contrastive_model(args.contrastive_checkpoint)
        else:
            self.contrastive_model = contrastive_model
            
        # Freeze contrastive model parameters
        for param in self.contrastive_model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension from contrastive model
        self.embedding_dim = self.contrastive_model.args.projection_dim
        
        # Get PHEcode size if available
        self.phe_code_size = getattr(self.contrastive_model, 'phe_code_size', None)
        
        # Initialize task heads
        self.task_heads = DownstreamHeads(
            feature_dim=self.embedding_dim,
            hidden_dim=args.hidden_dim,
            phe_code_size=self.phe_code_size
        )
        
        # Initialize metric tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
    def load_contrastive_model(self, checkpoint_path):
        """Load pre-trained contrastive model"""
        logging.info(f"Loading pre-trained contrastive model from {checkpoint_path}")
        
        # Load model
        model = RaindropContrastiveModel.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        # Log info
        logging.info(f"Loaded contrastive model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def extract_embeddings(self, batch):
        """Extract embeddings from the frozen contrastive model"""
        with torch.no_grad():
            # Prepare batch
            batch_data = self.contrastive_model.prepare_batch(batch)
            if batch_data is None:
                return None
            
            # Get embeddings from the contrastive model
            ts_proj, text_proj, _ = self.contrastive_model.model_forward(batch_data)
            
            # Average the embeddings from both modalities
            combined_embeddings = (ts_proj + text_proj) / 2
            
            return combined_embeddings
    
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
        
        return metrics
    
    def test_step(self, batch, batch_idx):
        """Test step (same as validation but logged differently)"""
        # Just reuse validation step logic
        metrics = self.validation_step(batch, batch_idx)
        
        # No need to log losses during test
        return metrics
    
    def validation_epoch_end(self, outputs):
        """Process validation outputs at the end of the epoch"""
        self._calculate_epoch_metrics(outputs, prefix='val')
    
    def test_epoch_end(self, outputs):
        """Process test outputs at the end of the epoch"""
        self._calculate_epoch_metrics(outputs, prefix='test')
    
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
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
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


def train_downstream_tasks(args):
    """Main function for training downstream task heads using Lightning"""
    # Set up logging
    log_file = setup_logging(args)
    
    # Set random seeds
    seed_everything(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    
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
    
    args = parser.parse_args()
    train_downstream_tasks(args) 