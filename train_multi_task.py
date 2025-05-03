import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
from dataloader_lite import get_dataloaders
from models.multi_task_model import MultiTaskKEDGN
from models.ds_only_model import DSOnlyMultiTaskModel
from utils import get_device
import wandb
import dotenv

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_one_epoch(model, data_loader, optimizer, device, model_type):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    mortality_loss = 0
    readmission_loss = 0
    phecode_loss = 0
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    bce_multi_loss = nn.BCEWithLogitsLoss(reduction='none')  # For PHE codes
    
    dataset = data_loader.dataset
    
    for batch in tqdm(data_loader, desc="Training"):
        # Skip empty batches
        if not batch:
            continue
            
        
        # Get labels
        mortality_labels = batch['mortality_label'].float().to(device)
        readmission_labels = batch['readmission_label'].float().to(device)
        
        # Forward pass - depends on model type
        if model_type == 'ds_only':
            outputs = model(batch['ds_embedding'])
        else:
            # Prepare input for MultiTaskKEDGN
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device) if batch['static'].numel() > 0 else None
            times = batch['times'].to(device)
            length = batch['length'].to(device)
            
            # Combine values and mask for the model input (follows KEDGN format)
            P = torch.cat([values, mask], dim=-1)
            P_static = static
            P_avg_interval = None  # Not used in simplified version
            P_length = length
            P_time = times
            P_var_plm_rep_tensor = torch.empty(0)  # Placeholder
            
            outputs = model(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate individual losses
        m_loss = bce_loss(outputs['mortality'].squeeze(-1), mortality_labels)
        r_loss = bce_loss(outputs['readmission'].squeeze(-1), readmission_labels)
        
        # PHE code loss - only if we have PHE codes
        p_loss = torch.tensor(0.0, device=device)
        if 'next_phecodes' in batch and hasattr(dataset, 'phecode_to_idx'):
            try:
                # Create a binary matrix for PHE codes
                batch_size = len(batch['next_phecodes'])
                phe_matrix = torch.zeros(batch_size, model.phe_code_size, device=device)
                
                # Properly map PHE codes to indices using the vocabulary
                for i, codes in enumerate(batch['next_phecodes']):
                    if codes:  # Check if the list is not empty
                        for code in codes:
                            if code in dataset.phecode_to_idx:
                                idx = dataset.phecode_to_idx[code]
                                phe_matrix[i, idx] = 1.0
                
                # Compute multi-label loss if we have any valid codes
                if phe_matrix.sum() > 0:
                    p_loss = bce_multi_loss(outputs['phecodes'], phe_matrix)
                    # Average over non-zero labels to handle sparsity
                    p_loss = p_loss.mean()
            except Exception as e:
                logging.warning(f"Error processing PHE codes: {e}")
                p_loss = torch.tensor(0.0, device=device)
        
        # Combined loss (equal weighting for simplicity)
        loss = m_loss + r_loss + p_loss
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        mortality_loss += m_loss.item()
        readmission_loss += r_loss.item()
        phecode_loss += p_loss.item()
    
    # Calculate average losses
    avg_loss = total_loss / len(data_loader)
    avg_m_loss = mortality_loss / len(data_loader)
    avg_r_loss = readmission_loss / len(data_loader)
    avg_p_loss = phecode_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
        'mortality_loss': avg_m_loss,
        'readmission_loss': avg_r_loss,
        'phecode_loss': avg_p_loss
    }

def evaluate(model, data_loader, device, model_type):
    """Evaluate the model"""
    model.eval()
    
    # Initialize metrics
    all_mortality_preds = []
    all_mortality_labels = []
    all_readmission_preds = []
    all_readmission_labels = []
    
    # For PHE code evaluation
    all_phecode_preds = []
    all_phecode_labels = []
    
    dataset = data_loader.dataset
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Skip empty batches
            if not batch:
                continue
                
            # Get labels
            mortality_labels = batch['mortality_label'].float().cpu().numpy()
            readmission_labels = batch['readmission_label'].float().cpu().numpy()
            
            # Forward pass - depends on model type
            if model_type == 'ds_only':
                outputs = model(batch['ds_embedding'])
            else:
                # Prepare input for MultiTaskKEDGN
                values = batch['values'].to(device)
                mask = batch['mask'].to(device)
                static = batch['static'].to(device) if batch['static'].numel() > 0 else None
                times = batch['times'].to(device)
                length = batch['length'].to(device)
                
                # Combine values and mask for the model input
                P = torch.cat([values, mask], dim=-1)
                P_static = static
                P_avg_interval = None  # Not used in simplified version
                P_length = length
                P_time = times
                P_var_plm_rep_tensor = torch.empty(0)  # Placeholder
                
                outputs = model(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
            
            # Get predictions
            mortality_preds = torch.sigmoid(outputs['mortality'].squeeze(-1)).cpu().numpy()
            readmission_preds = torch.sigmoid(outputs['readmission'].squeeze(-1)).cpu().numpy()
            
            # Store for metrics calculation
            all_mortality_preds.extend(mortality_preds)
            all_mortality_labels.extend(mortality_labels)
            all_readmission_preds.extend(readmission_preds)
            all_readmission_labels.extend(readmission_labels)
            
            # Process PHE code predictions
            if 'next_phecodes' in batch and hasattr(dataset, 'phecode_to_idx'):
                phecode_preds = torch.sigmoid(outputs['phecodes']).cpu().numpy()
                
                # Create one-hot matrices for the true labels
                batch_size = len(batch['next_phecodes'])
                phecode_labels = np.zeros((batch_size, model.phe_code_size))
                
                for i, codes in enumerate(batch['next_phecodes']):
                    if codes:  # Check if the list is not empty
                        for code in codes:
                            if code in dataset.phecode_to_idx:
                                idx = dataset.phecode_to_idx[code]
                                phecode_labels[i, idx] = 1.0
                
                all_phecode_preds.append(phecode_preds)
                all_phecode_labels.append(phecode_labels)
    
    # Calculate binary classification metrics
    m_auroc = roc_auc_score(all_mortality_labels, all_mortality_preds)
    m_auprc = average_precision_score(all_mortality_labels, all_mortality_preds)
    
    r_auroc = roc_auc_score(all_readmission_labels, all_readmission_preds)
    r_auprc = average_precision_score(all_readmission_labels, all_readmission_preds)
    
    metrics = {
        'mortality_auroc': m_auroc,
        'mortality_auprc': m_auprc,
        'readmission_auroc': r_auroc,
        'readmission_auprc': r_auprc
    }
    
    # Add PHE code metrics if we have data
    if all_phecode_preds and all_phecode_labels:
        try:
            all_phecode_preds = np.vstack(all_phecode_preds)
            all_phecode_labels = np.vstack(all_phecode_labels)
            
            # Calculate metrics for PHE codes that have at least one positive example
            valid_cols = np.where(all_phecode_labels.sum(axis=0) > 0)[0]
            
            if len(valid_cols) > 0:
                # Macro AUC (average AUC across codes)
                phecode_aucs = []
                for col in valid_cols:
                    if np.unique(all_phecode_labels[:, col]).shape[0] > 1:  # Need both classes present
                        phecode_aucs.append(roc_auc_score(all_phecode_labels[:, col], all_phecode_preds[:, col]))
                
                if phecode_aucs:
                    metrics['phecode_macro_auc'] = np.mean(phecode_aucs)
                
                # Micro AUC (flatten all predictions and calculate a single AUC)
                flat_preds = all_phecode_preds[:, valid_cols].flatten()
                flat_labels = all_phecode_labels[:, valid_cols].flatten()
                metrics['phecode_micro_auc'] = roc_auc_score(flat_labels, flat_preds)
                
                # Top PHE codes by frequency and their performance
                top_codes = []
                freqs = all_phecode_labels.sum(axis=0)
                top_indices = np.argsort(-freqs)[:10]  # Top 10 most frequent codes
                
                for idx in top_indices:
                    if freqs[idx] > 0 and np.unique(all_phecode_labels[:, idx]).shape[0] > 1:
                        code = dataset.idx_to_phecode[idx]
                        freq = freqs[idx]
                        auc = roc_auc_score(all_phecode_labels[:, idx], all_phecode_preds[:, idx])
                        top_codes.append((code, freq, auc))
                
                metrics['top_phecodes'] = top_codes
                
                # Log some example predictions
                if hasattr(dataset, 'idx_to_phecode'):
                    sample_idx = 0  # First patient in batch
                    threshold = 0.5
                    pred_codes = [dataset.idx_to_phecode[i] for i, p in enumerate(all_phecode_preds[sample_idx]) if p > threshold]
                    true_codes = [dataset.idx_to_phecode[i] for i, t in enumerate(all_phecode_labels[sample_idx]) if t > 0]
                    logging.info(f"Sample prediction: {pred_codes[:5]}")
                    logging.info(f"Sample true codes: {true_codes[:5]}")
        
        except Exception as e:
            logging.warning(f"Error calculating PHE code metrics: {e}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Multi-task training for MIMIC-IV')
    parser.add_argument('--data_path', type=str, default='/Users/riccardoconci/Local_documents/!!MIMIC', help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite', help='Path to cache directory')
    parser.add_argument('--model_type', type=str, choices=['full', 'ds_only'], default='ds_only', help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--projection_dim', type=int, default=512, help='Projection dimension for DS encoder')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pooling_type', type=str, default='attention', choices=['weighted_sum', 'attention'], help='Pooling type for DS encoder')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads for DS encoder')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='GeomMLProj_Baseline', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='None', help='WandB entity name')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        dotenv.load_dotenv('dot_env.txt')
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Determine device and print once
    device = get_device()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, var_embeddings = get_dataloaders(
    data_path=args.data_path,
    temp_dfs_path=args.temp_dfs_path,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    task_mode='CONTRASTIVE',
    test_ds_only=True
    )
    
    # Ensure PHE code mappings are loaded by calling get_phecode_df
    print("Loading PHE code mappings...")
    train_loader.dataset.get_phecode_df()
    
    # Get PHE code size
    if hasattr(train_loader.dataset, 'phe_code_size'):
        phe_code_size = train_loader.dataset.phe_code_size
        print(f"Using PHE code size from dataset: {phe_code_size}")
    else:
        # Fallback to default
        phe_code_size = 1000
        print(f"Warning: PHE code size not found. Using default: {phe_code_size}")
    
    # Initialize model based on type
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'ds_only':
        model = DSOnlyMultiTaskModel(
            DEVICE=device,
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
            phe_code_size=phe_code_size,
            pooling_type=args.pooling_type,
            num_heads=args.num_heads
        )
    else:
        # For the full model, we need to extract dimensions from data
        for batch in train_loader:
            if batch:
                values = batch['values']
                variables_num = values.shape[2]
                timestamps = values.shape[1]
                static = batch['static']
                d_static = static.shape[1] if static.numel() > 0 else 0
                break
        
        model = MultiTaskKEDGN(
            DEVICE=device,
            hidden_dim=args.hidden_dim,
            num_of_variables=variables_num,
            num_of_timestamps=timestamps,
            d_static=d_static,
            n_class=1,  # For binary classification
            phe_code_size=phe_code_size,
            task_mode='CONTRASTIVE'
        )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    best_val_metric = 0
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.model_type)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Mortality Loss: {train_metrics['mortality_loss']:.4f}")
        print(f"  Readmission Loss: {train_metrics['readmission_loss']:.4f}")
        print(f"  PHE Code Loss: {train_metrics['phecode_loss']:.4f}")
        
        # Log training metrics to wandb
        if args.use_wandb:
            wandb.log({
                "train/loss": train_metrics['loss'],
                "train/mortality_loss": train_metrics['mortality_loss'],
                "train/readmission_loss": train_metrics['readmission_loss'],
                "train/phecode_loss": train_metrics['phecode_loss'],
                "epoch": epoch
            })
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, args.model_type)
        print(f"Validation Metrics:")
        print(f"  Mortality AUROC: {val_metrics['mortality_auroc']:.4f}")
        print(f"  Mortality AUPRC: {val_metrics['mortality_auprc']:.4f}")
        print(f"  Readmission AUROC: {val_metrics['readmission_auroc']:.4f}")
        print(f"  Readmission AUPRC: {val_metrics['readmission_auprc']:.4f}")
        
        # Log validation metrics to wandb
        if args.use_wandb:
            wandb.log({
                "val/mortality_auroc": val_metrics['mortality_auroc'],
                "val/mortality_auprc": val_metrics['mortality_auprc'],
                "val/readmission_auroc": val_metrics['readmission_auroc'],
                "val/readmission_auprc": val_metrics['readmission_auprc'],
                "epoch": epoch
            })
        
        # Print PHE code metrics if available
        if 'phecode_macro_auc' in val_metrics:
            print(f"  PHE Code Macro AUC: {val_metrics['phecode_macro_auc']:.4f}")
            if args.use_wandb:
                wandb.log({
                    "val/phecode_macro_auc": val_metrics['phecode_macro_auc'],
                    "epoch": epoch
                })
        if 'phecode_micro_auc' in val_metrics:
            print(f"  PHE Code Micro AUC: {val_metrics['phecode_micro_auc']:.4f}")
            if args.use_wandb:
                wandb.log({
                    "val/phecode_micro_auc": val_metrics['phecode_micro_auc'],
                    "epoch": epoch
                })
        
        # Save best model (using average of all metrics as overall score)
        metrics_to_average = [
            val_metrics['mortality_auprc'],
            val_metrics['readmission_auprc']
        ]
        if 'phecode_micro_auc' in val_metrics:
            metrics_to_average.append(val_metrics['phecode_micro_auc'])
        
        current_metric = sum(metrics_to_average) / len(metrics_to_average)
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, f'model_{args.model_type}_best.pt')
            print(f"Saved best model with validation metric: {current_metric:.4f}")
    
    # Test best model
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(f'model_{args.model_type}_best.pt', weights_only=False)['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, args.model_type)
    print(f"Test Metrics:")
    print(f"  Mortality AUROC: {test_metrics['mortality_auroc']:.4f}")
    print(f"  Mortality AUPRC: {test_metrics['mortality_auprc']:.4f}")
    print(f"  Readmission AUROC: {test_metrics['readmission_auroc']:.4f}")
    print(f"  Readmission AUPRC: {test_metrics['readmission_auprc']:.4f}")
    
    # Log test metrics to wandb
    if args.use_wandb:
        wandb.log({
            "test/mortality_auroc": test_metrics['mortality_auroc'],
            "test/mortality_auprc": test_metrics['mortality_auprc'],
            "test/readmission_auroc": test_metrics['readmission_auroc'],
            "test/readmission_auprc": test_metrics['readmission_auprc']
        })
    
    # Print PHE code metrics if available
    if 'phecode_macro_auc' in test_metrics:
        print(f"  PHE Code Macro AUC: {test_metrics['phecode_macro_auc']:.4f}")
        if args.use_wandb:
            wandb.log({
                "test/phecode_macro_auc": test_metrics['phecode_macro_auc']
            })
    if 'phecode_micro_auc' in test_metrics:
        print(f"  PHE Code Micro AUC: {test_metrics['phecode_micro_auc']:.4f}")
        if args.use_wandb:
            wandb.log({
                "test/phecode_micro_auc": test_metrics['phecode_micro_auc']
            })
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 