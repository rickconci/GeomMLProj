# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import random
from datetime import datetime


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()



def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)




DEBUG = True  # Set to True to enable debug printing

def debug_print(*args, **kwargs):
    """Print only if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

def toggle_debug(enable=None):
    """Toggle or set the DEBUG flag
    
    Args:
        enable: If None, toggle the current state. If True/False, set to that value.
    
    Returns:
        Current DEBUG value after toggling/setting
    """
    global DEBUG
    if enable is None:
        DEBUG = not DEBUG
    else:
        DEBUG = bool(enable)
    return DEBUG



def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_lightning_devices(devices_arg=None):
    """
    Automatically detect available devices for PyTorch Lightning
    
    Args:
        devices_arg: Optional manually specified devices value
    
    Returns:
        Number of devices to use
    """
    if devices_arg is not None:
        return devices_arg
        
    if torch.backends.mps.is_available():
        return 1  # MPS only supports 1 device
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1  # Default to 1 CPU


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
        elif model_type == 'raindrop_v2':
            # Prepare input for RaindropV2
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device) if batch['static'].numel() > 0 else None
            times = batch['times'].to(device)
            length = batch['length'].to(device)
            
            # Combine values and mask for the RaindropV2 input
            # RaindropV2 expects: src [max_len, batch_size, 2*d_inp]
            src = torch.cat([values, mask], dim=-1).permute(1, 0, 2)  # [T, B, F]
            static = static  # [B, S]
            times = times.permute(1, 0)  # [T, B]
            lengths = length  # [B]
            
            outputs = model(src, static, times, lengths)
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
        
        # Efficient PHE code loss calculation
        idxs = batch["next_idx_padded"].to(device)   # (B, K)
        lens = batch["next_len"].to(device)          # (B,)
        phecode_logits = outputs["phecodes"]         # (B, P)
        
        # Two approaches for PHE code loss:
        
        # Approach 1: Using a binary target tensor (more explicit)
        B, P = phecode_logits.size(0), phecode_logits.size(1)
        phecode_targets = torch.zeros(B, P, device=device)
        
        # Create a mask for valid indices
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, idxs.size(1))
        valid_mask = torch.arange(idxs.size(1), device=device).expand(B, -1) < lens.unsqueeze(1)
        
        # Get flattened indices for valid positions
        batch_indices_flat = batch_indices[valid_mask]
        code_indices_flat = idxs[valid_mask]
        
        # Set corresponding positions to 1
        phecode_targets[batch_indices_flat, code_indices_flat] = 1.0
        
        # Use binary cross-entropy on the full prediction vector
        p_loss = F.binary_cross_entropy_with_logits(phecode_logits, phecode_targets, reduction="mean")
       
        # If RaindropV2, also add graph distance regularization
        if model_type == 'raindrop_v2' and 'distance' in outputs:
            graph_distance_factor = 0.1  # Weight for the graph distance regularization
            graph_loss = outputs['distance'] * graph_distance_factor
            loss = m_loss + r_loss + p_loss + graph_loss
        else:
            # Standard loss combination
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

def calculate_binary_classification_metrics(predictions, labels):
    """
    Calculate AUROC and AUPRC for binary classification.
    
    Args:
        predictions: List/array of model predictions (probabilities)
        labels: List/array of ground truth labels
        
    Returns:
        dict: Dictionary containing AUROC and AUPRC metrics
    """
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)
    
    return {
        'auroc': auroc,
        'auprc': auprc
    }

def calculate_phecode_metrics(phecode_preds, phecode_labels, dataset=None):
    """
    Calculate metrics for PHE code prediction.
    
    Args:
        phecode_preds: Model predictions for PHE codes [N, P]
        phecode_labels: Ground truth PHE code labels [N, P]
        dataset: Optional dataset object for additional information
        
    Returns:
        dict: Dictionary containing PHE code metrics
    """
    metrics = {}
    
    # Calculate metrics for PHE codes that have at least one positive example
    valid_cols = np.where(phecode_labels.sum(axis=0) > 0)[0]
    
    if len(valid_cols) > 0:
        # Macro AUC (average AUC across codes)
        phecode_aucs = []
        for col in valid_cols:
            if np.unique(phecode_labels[:, col]).shape[0] > 1:  # Need both classes present
                phecode_aucs.append(roc_auc_score(phecode_labels[:, col], phecode_preds[:, col]))
        
        if phecode_aucs:
            metrics['macro_auc'] = np.mean(phecode_aucs)
        
        # Micro AUC (flatten all predictions and calculate a single AUC)
        flat_preds = phecode_preds[:, valid_cols].flatten()
        flat_labels = phecode_labels[:, valid_cols].flatten()
        metrics['micro_auc'] = roc_auc_score(flat_labels, flat_preds)
        
        # Micro-averaged average precision
        metrics['micro_ap'] = average_precision_score(flat_labels, flat_preds)

        # Precision@5
        topk = 5
        top_preds = np.argsort(-phecode_preds, axis=1)[:, :topk]
        prec5_list = []
        for i in range(phecode_preds.shape[0]):
            true_set = set(np.where(phecode_labels[i])[0])
            pred_set = set(top_preds[i])
            prec5_list.append(len(pred_set & true_set) / topk)
        metrics['prec@5'] = float(np.mean(prec5_list))
        
        # Top PHE codes by frequency and their performance
        if dataset and hasattr(dataset, 'idx_to_phecode'):
            top_codes = []
            freqs = phecode_labels.sum(axis=0)
            top_indices = np.argsort(-freqs)[:10]  # Top 10 most frequent codes
            
            for idx in top_indices:
                if freqs[idx] > 0 and np.unique(phecode_labels[:, idx]).shape[0] > 1:
                    code = dataset.idx_to_phecode[idx]
                    freq = freqs[idx]
                    auc = roc_auc_score(phecode_labels[:, idx], phecode_preds[:, idx])
                    top_codes.append((code, freq, auc))
            
            metrics['top_phecodes'] = top_codes
    
    return metrics

def prepare_phecode_targets(batch, device, phecode_size):
    """
    Prepare target tensor for PHE code prediction.
    
    Args:
        batch: Data batch containing PHE code indices
        device: Torch device
        phecode_size: Size of the PHE code vocabulary
        
    Returns:
        tuple: PHE code labels tensor and mask for valid samples
    """
    if 'next_idx_padded' not in batch or 'next_len' not in batch:
        return None, None
        
    idxs = batch['next_idx_padded'].to(device)  # (B, K)
    lens = batch['next_len'].to(device)  # (B,)
    
    # Skip empty batches
    has_codes = lens > 0
    if not has_codes.any():
        return None, None
        
    # Filter to only samples with codes
    batch_size = has_codes.sum().item()
    
    # Create binary target tensor
    phecode_targets = torch.zeros(batch_size, phecode_size, device=device)
    
    # Handle only valid samples
    valid_idxs = idxs[has_codes]
    valid_lens = lens[has_codes]
    
    # Create mask for valid indices
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, valid_idxs.size(1))
    valid_mask = torch.arange(valid_idxs.size(1), device=device).expand(batch_size, -1) < valid_lens.unsqueeze(1)
    
    # Get flattened indices for valid positions
    batch_indices_flat = batch_indices[valid_mask]
    code_indices_flat = valid_idxs[valid_mask]
    
    # Set corresponding positions to 1
    phecode_targets[batch_indices_flat, code_indices_flat] = 1.0
    
    return phecode_targets, has_codes

def evaluate_embeddings(embeddings, labels, task_heads, device):
    """
    Evaluate embeddings on downstream tasks using lightweight task heads.
    
    Args:
        embeddings: Model embeddings [batch_size, embedding_dim]
        labels: Dictionary of task labels
        task_heads: Dictionary of task-specific prediction heads
        device: Torch device
        
    Returns:
        dict: Dictionary of task predictions
    """
    predictions = {}
    
    # Get predictions for each task
    for task, head in task_heads.items():
        if head is not None:
            logits = head(embeddings)
            if task in ['mortality', 'readmission']:
                predictions[task] = torch.sigmoid(logits.squeeze(-1))
            elif task == 'phecodes':
                predictions[task] = torch.sigmoid(logits)
    
    return predictions

def evaluate_downstream_tasks(embeddings_model, data_loader, task_heads, device, phecode_size=None):
    """
    Evaluate embeddings on downstream tasks.
    
    Args:
        embeddings_model: Function that extracts embeddings from batches
        data_loader: DataLoader for evaluation data
        task_heads: Dictionary of task-specific prediction heads
        device: Device to perform evaluation on
        phecode_size: Size of the PHE code vocabulary
        
    Returns:
        dict: Dictionary of metrics for each task
    """
    # Initialize metrics collection
    all_mortality_preds = []
    all_mortality_labels = []
    all_readmission_preds = []
    all_readmission_labels = []
    all_phecode_preds = []
    all_phecode_labels = []
    
    dataset = data_loader.dataset
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating on downstream tasks"):
            # Skip empty batches
            if not batch:
                continue
            
            # Extract labels
            mortality_labels = batch['mortality_label'].float().cpu().numpy()
            readmission_labels = batch['readmission_label'].float().cpu().numpy()
            
            # Get embeddings from the model
            embeddings = embeddings_model(batch)
            
            # Get predictions for each task
            predictions = evaluate_embeddings(embeddings, batch, task_heads, device)
            
            # Process mortality predictions
            if 'mortality' in predictions:
                mortality_preds = predictions['mortality'].cpu().numpy()
                all_mortality_preds.extend(mortality_preds)
                all_mortality_labels.extend(mortality_labels)
            
            # Process readmission predictions
            if 'readmission' in predictions:
                readmission_preds = predictions['readmission'].cpu().numpy()
                all_readmission_preds.extend(readmission_preds)
                all_readmission_labels.extend(readmission_labels)
            
            # Process PHE code predictions
            if 'phecodes' in predictions and phecode_size is not None:
                phecode_targets, valid_samples = prepare_phecode_targets(batch, device, phecode_size)
                
                if phecode_targets is not None:
                    phecode_preds = predictions['phecodes']
                    if valid_samples is not None:
                        phecode_preds = phecode_preds[valid_samples]
                    
                    all_phecode_preds.append(phecode_preds.cpu().numpy())
                    all_phecode_labels.append(phecode_targets.cpu().numpy())
    
    # Compute metrics
    metrics = {}
    
    # Mortality metrics
    if all_mortality_preds:
        m_metrics = calculate_binary_classification_metrics(all_mortality_preds, all_mortality_labels)
        metrics['mortality_auroc'] = m_metrics['auroc']
        metrics['mortality_auprc'] = m_metrics['auprc']
    
    # Readmission metrics
    if all_readmission_preds:
        r_metrics = calculate_binary_classification_metrics(all_readmission_preds, all_readmission_labels)
        metrics['readmission_auroc'] = r_metrics['auroc']
        metrics['readmission_auprc'] = r_metrics['auprc']
    
    # PHE code metrics
    if all_phecode_preds and all_phecode_labels:
        try:
            all_phecode_preds = np.vstack(all_phecode_preds)
            all_phecode_labels = np.vstack(all_phecode_labels)
            
            phe_metrics = calculate_phecode_metrics(all_phecode_preds, all_phecode_labels, dataset)
            
            metrics['phecode_macro_auc'] = phe_metrics.get('macro_auc', 0.0)
            metrics['phecode_micro_auc'] = phe_metrics.get('micro_auc', 0.0)
            metrics['phecode_micro_ap'] = phe_metrics.get('micro_ap', 0.0)
            metrics['phecode_prec@5'] = phe_metrics.get('prec@5', 0.0)
            
            if 'top_phecodes' in phe_metrics:
                metrics['top_phecodes'] = phe_metrics['top_phecodes']
                
            # Log example predictions
            sample_idx = 0  # First patient in batch
            if sample_idx < all_phecode_preds.shape[0] and hasattr(dataset, 'idx_to_phecode'):
                threshold = 0.5
                pred_codes = [dataset.idx_to_phecode[i] for i, p in enumerate(all_phecode_preds[sample_idx]) if p > threshold]
                true_codes = [dataset.idx_to_phecode[i] for i, t in enumerate(all_phecode_labels[sample_idx]) if t > 0]
                logging.info(f"Sample prediction: {pred_codes[:5]}")
                logging.info(f"Sample true codes: {true_codes[:5]}")
        except Exception as e:
            logging.warning(f"Error calculating PHE code metrics: {e}")
    
    return metrics

def evaluate(model, data_loader, device, model_type):
    """
    Evaluate a model on test/validation data.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        model_type: Type of model being evaluated
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    all_mortality_preds = []
    all_mortality_labels = []
    all_readmission_preds = []
    all_readmission_labels = []
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
                outputs = model(batch['ds_embedding'].to(device))
            elif model_type == 'raindrop_v2':
                # Prepare input for RaindropV2
                values = batch['values'].to(device)
                mask = batch['mask'].to(device)
                static = batch['static'].to(device) if batch['static'].numel() > 0 else None
                times = batch['times'].to(device)
                length = batch['length'].to(device)
                
                # Combine values and mask for the RaindropV2 input
                # RaindropV2 expects: src [max_len, batch_size, 2*d_inp]
                src = torch.cat([values, mask], dim=-1).permute(1, 0, 2)  # [T, B, F]
                static = static  # [B, S]
                times = times.permute(1, 0)  # [T, B]
                lengths = length  # [B]
                
                outputs = model(src, static, times, lengths)
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
                P_var_plm_rep_tensor = torch.empty(0).to(device)  # Placeholder
                
                outputs = model(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
            
            # Get predictions
            mortality_preds = torch.sigmoid(outputs['mortality'].squeeze(-1)).cpu().numpy()
            readmission_preds = torch.sigmoid(outputs['readmission'].squeeze(-1)).cpu().numpy()
            
            # Store for metrics calculation
            all_mortality_preds.extend(mortality_preds)
            all_mortality_labels.extend(mortality_labels)
            all_readmission_preds.extend(readmission_preds)
            all_readmission_labels.extend(readmission_labels)
            
            # Process PHE code predictions using next_idx_padded and next_len
            if 'next_idx_padded' in batch and 'phecodes' in outputs:
                phecode_preds = torch.sigmoid(outputs['phecodes']).cpu().numpy()
                
                # Prepare PHE code targets
                if hasattr(model, 'phe_code_size'):
                    phecode_targets, valid_samples = prepare_phecode_targets(batch, device, model.phe_code_size)
                    if phecode_targets is not None:
                        phecode_labels_np = phecode_targets.cpu().numpy()
                        phecode_preds_np = phecode_preds
                        if valid_samples is not None:
                            phecode_preds_np = phecode_preds_np[valid_samples.cpu().numpy()]
                        all_phecode_preds.append(phecode_preds_np)
                        all_phecode_labels.append(phecode_labels_np)
    
    # Calculate metrics
    metrics = {}
    
    # Binary classification metrics
    if all_mortality_preds:
        m_metrics = calculate_binary_classification_metrics(all_mortality_preds, all_mortality_labels)
        metrics['mortality_auroc'] = m_metrics['auroc']
        metrics['mortality_auprc'] = m_metrics['auprc']
    
    if all_readmission_preds:
        r_metrics = calculate_binary_classification_metrics(all_readmission_preds, all_readmission_labels)
        metrics['readmission_auroc'] = r_metrics['auroc']
        metrics['readmission_auprc'] = r_metrics['auprc']
    
    # Add PHE code metrics if we have data
    if all_phecode_preds and all_phecode_labels:
        try:
            all_phecode_preds = np.vstack(all_phecode_preds)
            all_phecode_labels = np.vstack(all_phecode_labels)
            
            phe_metrics = calculate_phecode_metrics(all_phecode_preds, all_phecode_labels, dataset)
            
            metrics['phecode_macro_auc'] = phe_metrics.get('macro_auc', 0.0)
            metrics['phecode_micro_auc'] = phe_metrics.get('micro_auc', 0.0)
            metrics['phecode_micro_ap'] = phe_metrics.get('micro_ap', 0.0)
            metrics['phecode_prec@5'] = phe_metrics.get('prec@5', 0.0)
            
            if 'top_phecodes' in phe_metrics:
                metrics['top_phecodes'] = phe_metrics['top_phecodes']
        
        except Exception as e:
            logging.warning(f"Error calculating PHE code metrics: {e}")
    
    return metrics

def calculate_phecode_loss(phecode_logits, idxs, lens, device):
    """
    Calculate PHEcode prediction loss efficiently
    
    Args:
        phecode_logits: Model predictions for PHEcodes [B, P]
        idxs: Padded index matrix of ground truth PHEcodes [B, K]
        lens: Length of valid indices for each batch item [B]
        device: Device to place tensors on
    
    Returns:
        p_loss: Binary cross-entropy loss for PHEcode prediction
    """
    # Get dimensions
    B, P = phecode_logits.size(0), phecode_logits.size(1)
    
    # Create a binary target tensor (more explicit)
    phecode_targets = torch.zeros(B, P, device=device)
    
    # Create a mask for valid indices
    batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, idxs.size(1))
    valid_mask = torch.arange(idxs.size(1), device=device).expand(B, -1) < lens.unsqueeze(1)
    
    # Get flattened indices for valid positions
    batch_indices_flat = batch_indices[valid_mask]
    code_indices_flat = idxs[valid_mask]
    
    # Set corresponding positions to 1
    phecode_targets[batch_indices_flat, code_indices_flat] = 1.0
    
    # Use binary cross-entropy on the full prediction vector
    p_loss = F.binary_cross_entropy_with_logits(phecode_logits, phecode_targets, reduction="mean")
    
    return p_loss


