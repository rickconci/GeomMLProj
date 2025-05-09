import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from tqdm import tqdm
import wandb
import logging
from pathlib import Path
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_utils import calculate_binary_classification_metrics, calculate_phecode_metrics, prepare_phecode_targets
import json

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings_dir, epoch):
        self.embeddings_dir = os.path.join(embeddings_dir, f'epoch_{epoch:03d}')
        logging.info(f"Loading embeddings from: {self.embeddings_dir}")
        
        # Load memory-mapped arrays
        self.ts_proj = np.memmap(os.path.join(self.embeddings_dir, 'ts_proj.mmap'), 
                                dtype='float32', mode='r')
        self.text_proj = np.memmap(os.path.join(self.embeddings_dir, 'text_proj.mmap'), 
                                 dtype='float32', mode='r')
        self.mortality_label = np.memmap(os.path.join(self.embeddings_dir, 'mortality_label.mmap'), 
                                      dtype='float32', mode='r')
        self.readmission_label = np.memmap(os.path.join(self.embeddings_dir, 'readmission_label.mmap'), 
                                        dtype='float32', mode='r')
        self.next_idx_padded = np.memmap(os.path.join(self.embeddings_dir, 'next_idx_padded.mmap'), 
                                       dtype='int64', mode='r')
        self.next_phecode_len = np.memmap(os.path.join(self.embeddings_dir, 'next_phecode_len.mmap'), 
                                        dtype='int64', mode='r')
        
        # Load metadata to get shapes
        with open(os.path.join(self.embeddings_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Reshape arrays based on metadata
        self.ts_proj = self.ts_proj.reshape(-1, metadata['projection_dim'])
        self.text_proj = self.text_proj.reshape(-1, metadata['projection_dim'])
        self.next_idx_padded = self.next_idx_padded.reshape(-1, 20)  # Reshape to [N, 20] for max 20 phecodes per stay
        
        # Get dimensions
        self.ts_proj_dim = self.ts_proj.shape[-1]
        self.text_proj_dim = self.text_proj.shape[-1]
        
        # Try to load hadm_ids if they exist
        hadm_id_path = os.path.join(self.embeddings_dir, 'hadm_id.mmap')
        self.has_hadm_ids = os.path.exists(hadm_id_path)
        if self.has_hadm_ids:
            self.hadm_id = np.memmap(hadm_id_path, dtype='int64', mode='r')
        
        logging.info(f"Loaded dataset with {len(self)} samples")
        logging.info(f"Time series projection dim: {self.ts_proj_dim}")
        logging.info(f"Text projection dim: {self.text_proj_dim}")
    
    def __len__(self):
        return len(self.ts_proj)
    
    def __getitem__(self, idx):
        item = {
            'ts_proj': torch.FloatTensor(self.ts_proj[idx].copy()),  # Make a copy to avoid non-writable warning
            'text_proj': torch.FloatTensor(self.text_proj[idx].copy()),
            'mortality_label': torch.FloatTensor([self.mortality_label[idx]]),
            'readmission_label': torch.FloatTensor([self.readmission_label[idx]]),
            'next_idx_padded': torch.LongTensor(self.next_idx_padded[idx].copy()),
            'next_phecode_len': torch.LongTensor([self.next_phecode_len[idx]])
        }
        
        if self.has_hadm_ids:
            item['hadm_id'] = self.hadm_id[idx]
            
        return item

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(self, x):
        return self.classifier(x)

def train_probe(model, train_loader, val_loader, criterion, optimizer, device, task_name, num_epochs=10):
    """Train a probe model with validation"""
    best_val_metrics = None
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Training {task_name} probe - Epoch {epoch+1}/{num_epochs}'):
            # Get embeddings and labels
            ts_proj = batch['ts_proj'].to(device)
            text_proj = batch['text_proj'].to(device)
            embeddings = torch.cat([ts_proj, text_proj], dim=1)
            
            if task_name == 'mortality':
                labels = batch['mortality_label'].to(device).squeeze(-1)
            elif task_name == 'readmission':
                labels = batch['readmission_label'].to(device).squeeze(-1)
            else:  # phecode
                # Use prepare_phecode_targets from train_utils
                labels, valid_samples = prepare_phecode_targets(batch, device, model.classifier[-1].out_features)
                if labels is None:
                    continue  # Skip batch if no valid PHEcodes
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(embeddings)
            
            # Calculate loss
            if task_name in ['mortality', 'readmission']:
                loss = criterion(logits.squeeze(), labels)
            else:  # phecode
                if valid_samples is not None:
                    logits = logits[valid_samples]
                loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions and labels
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                if task_name == 'phecode':
                    train_preds.append(preds.cpu().numpy())
                    train_labels.append(labels.cpu().numpy())
                else:
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
        
        # Print training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - {task_name} probe - Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_preds = []
            val_labels = []
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Validating {task_name} probe'):
                    ts_proj = batch['ts_proj'].to(device)
                    text_proj = batch['text_proj'].to(device)
                    embeddings = torch.cat([ts_proj, text_proj], dim=1)
                    
                    if task_name == 'mortality':
                        labels = batch['mortality_label'].to(device).squeeze(-1)
                    elif task_name == 'readmission':
                        labels = batch['readmission_label'].to(device).squeeze(-1)
                    else:  # phecode
                        # Use prepare_phecode_targets from train_utils
                        labels, valid_samples = prepare_phecode_targets(batch, device, model.classifier[-1].out_features)
                        if labels is None:
                            continue  # Skip batch if no valid PHEcodes
                    
                    logits = model(embeddings)
                    if task_name == 'phecode' and valid_samples is not None:
                        logits = logits[valid_samples]
                    
                    # Calculate validation loss
                    if task_name in ['mortality', 'readmission']:
                        loss = criterion(logits.squeeze(), labels)
                    else:  # phecode
                        loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    preds = torch.sigmoid(logits)
                    if task_name == 'phecode':
                        val_preds.append(preds.cpu().numpy())
                        val_labels.append(labels.cpu().numpy())
                    else:
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
            
            # Print validation loss
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - {task_name} probe - Average validation loss: {avg_val_loss:.4f}")
            
            # Convert lists to numpy arrays for metrics calculation
            if task_name == 'phecode':
                if val_preds:  # Only stack if we have predictions
                    val_preds = np.vstack(val_preds)
                    val_labels = np.vstack(val_labels)
                    print(f"Validation shapes - preds: {val_preds.shape}, labels: {val_labels.shape}")
                else:
                    print("Warning: No validation predictions collected")
                    val_preds = np.zeros((0, model.classifier[-1].out_features))
                    val_labels = np.zeros((0, model.classifier[-1].out_features))
            else:
                val_preds = np.array(val_preds)
                val_labels = np.array(val_labels)
            
            # Calculate validation metrics using train_utils functions
            if task_name in ['mortality', 'readmission']:
                val_metrics = calculate_binary_classification_metrics(val_preds, val_labels)
            else:  # phecode
                val_metrics = calculate_phecode_metrics(val_preds, val_labels, val_loader.dataset)
            
            # Update best model if validation metrics improve
            if task_name == 'phecode':
                metric_key = 'micro_auc'  # Use micro_auc for PHEcode task
            else:
                metric_key = 'auroc'  # Use auroc for binary classification tasks
            
            if best_val_metrics is None or (metric_key in val_metrics and val_metrics[metric_key] > best_val_metrics[metric_key]):
                best_val_metrics = val_metrics
                best_model_state = model.state_dict().copy()
    
    # Restore best model if validation was performed
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Convert lists to numpy arrays for final metrics calculation
    if task_name == 'phecode':
        if train_preds:  # Only stack if we have predictions
            train_preds = np.vstack(train_preds)
            train_labels = np.vstack(train_labels)
            print(f"Training shapes - preds: {train_preds.shape}, labels: {train_labels.shape}")
        else:
            print("Warning: No training predictions collected")
            train_preds = np.zeros((0, model.classifier[-1].out_features))
            train_labels = np.zeros((0, model.classifier[-1].out_features))
    else:
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
    
    # Calculate final metrics on training set using train_utils functions
    if task_name in ['mortality', 'readmission']:
        train_metrics = calculate_binary_classification_metrics(train_preds, train_labels)
    else:  # phecode
        train_metrics = calculate_phecode_metrics(train_preds, train_labels, train_loader.dataset)
    
    return train_metrics, best_val_metrics if best_val_metrics is not None else train_metrics

def main(args):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load embeddings for the specified epoch
    dataset = EmbeddingsDataset(args.embeddings_dir, args.epoch)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize probes with hidden layers
    input_dim = dataset.ts_proj_dim + dataset.text_proj_dim
    hidden_dim = args.hidden_dim
    
    mortality_probe = LinearProbe(input_dim, 1, hidden_dim).to(device)
    readmission_probe = LinearProbe(input_dim, 1, hidden_dim).to(device)
    phecode_probe = LinearProbe(input_dim, args.phecode_size, hidden_dim).to(device)
    
    # Train probes
    criterion = nn.BCEWithLogitsLoss()
    
    # Mortality probe
    optimizer = torch.optim.Adam(mortality_probe.parameters(), lr=args.lr)
    train_mortality_metrics, val_mortality_metrics = train_probe(
        mortality_probe, train_loader, val_loader, criterion, optimizer, device, 'mortality'
    )
    
    # Readmission probe
    optimizer = torch.optim.Adam(readmission_probe.parameters(), lr=args.lr)
    train_readmission_metrics, val_readmission_metrics = train_probe(
        readmission_probe, train_loader, val_loader, criterion, optimizer, device, 'readmission'
    )
    
    # PHEcode probe
    optimizer = torch.optim.Adam(phecode_probe.parameters(), lr=args.lr)
    train_phecode_metrics, val_phecode_metrics = train_probe(
        phecode_probe, train_loader, val_loader, criterion, optimizer, device, 'phecode'
    )
    
    # Log results with safe dictionary access
    results = {
        'epoch': args.epoch,
        'train_mortality_auroc': train_mortality_metrics.get('auroc', 0.0),
        'train_mortality_auprc': train_mortality_metrics.get('auprc', 0.0),
        'val_mortality_auroc': val_mortality_metrics.get('auroc', 0.0),
        'val_mortality_auprc': val_mortality_metrics.get('auprc', 0.0),
        'train_readmission_auroc': train_readmission_metrics.get('auroc', 0.0),
        'train_readmission_auprc': train_readmission_metrics.get('auprc', 0.0),
        'val_readmission_auroc': val_readmission_metrics.get('auroc', 0.0),
        'val_readmission_auprc': val_readmission_metrics.get('auprc', 0.0)
    }
    
    # Add PHEcode metrics if available, with safe defaults
    phecode_metric_keys = [
        'micro_auc', 'macro_auc', 'micro_ap', 'prec@5',
        'micro_f1', 'macro_f1', 'micro_precision', 'macro_precision',
        'micro_recall', 'macro_recall'
    ]
    
    for key in phecode_metric_keys:
        results[f'train_phecode_{key}'] = train_phecode_metrics.get(key, 0.0)
        results[f'val_phecode_{key}'] = val_phecode_metrics.get(key, 0.0)
    
    # Add top PHEcodes if available
    if 'top_phecodes' in train_phecode_metrics:
        results['train_top_phecodes'] = train_phecode_metrics['top_phecodes']
        results['val_top_phecodes'] = val_phecode_metrics.get('top_phecodes', [])
    
    # Print results
    print("\n" + "="*50)
    print(f"Results for epoch {args.epoch}")
    print("="*50)
    
    print("\nMortality Prediction:")
    print(f"  Train - AUROC: {results['train_mortality_auroc']:.4f}, AUPRC: {results['train_mortality_auprc']:.4f}")
    print(f"  Val   - AUROC: {results['val_mortality_auroc']:.4f}, AUPRC: {results['val_mortality_auprc']:.4f}")
    
    print("\nReadmission Prediction:")
    print(f"  Train - AUROC: {results['train_readmission_auroc']:.4f}, AUPRC: {results['train_readmission_auprc']:.4f}")
    print(f"  Val   - AUROC: {results['val_readmission_auroc']:.4f}, AUPRC: {results['val_readmission_auprc']:.4f}")
    
    print("\nPHEcode Prediction:")
    # Print all available PHEcode metrics
    for key in phecode_metric_keys:
        if key in ['micro_auc', 'macro_auc', 'micro_ap']:  # Always print these
            print(f"  Train - {key}: {results[f'train_phecode_{key}']:.4f}")
            print(f"  Val   - {key}: {results[f'val_phecode_{key}']:.4f}")
        elif results[f'train_phecode_{key}'] > 0:  # Only print others if they exist
            print(f"  Train - {key}: {results[f'train_phecode_{key}']:.4f}")
            print(f"  Val   - {key}: {results[f'val_phecode_{key}']:.4f}")
    
    if 'train_top_phecodes' in results and results['train_top_phecodes']:
        print("\nTop PHEcodes (Train):")
        for code, freq, auc in results['train_top_phecodes']:
            print(f"  {code}: freq={freq}, AUC={auc:.4f}")
        
        if results['val_top_phecodes']:
            print("\nTop PHEcodes (Val):")
            for code, freq, auc in results['val_top_phecodes']:
                print(f"  {code}: freq={freq}, AUC={auc:.4f}")
    
    print("="*50 + "\n")
    
    if args.use_wandb:
        wandb.log(results)
    
    # Save results
    hidden_suffix = f"_hidden{args.hidden_dim}" if args.hidden_dim is not None else "_linear"
    results_dir = os.path.join(args.embeddings_dir, f'probe_results{hidden_suffix}')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'epoch_{args.epoch}_results.npy'), results)
    
    # Save probe models if needed
    # models_dir = os.path.join(results_dir, 'models')
    # os.makedirs(models_dir, exist_ok=True)
    # torch.save(mortality_probe.state_dict(), os.path.join(models_dir, f'epoch_{args.epoch}_mortality_probe.pt'))
    # torch.save(readmission_probe.state_dict(), os.path.join(models_dir, f'epoch_{args.epoch}_readmission_probe.pt'))
    # torch.save(phecode_probe.state_dict(), os.path.join(models_dir, f'epoch_{args.epoch}_phecode_probe.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_dir', type=str, default='/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/Embeddings/bs128_lr0.001_seed42_proj256_temp0.07_nophe', help='Directory containing saved embeddings')
    parser.add_argument('--epoch', type=int, default=7, help='Epoch number to evaluate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training probes')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training probes')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for probe networks')
    parser.add_argument('--phecode_size', type=int, default=1788, help='Number of PHEcodes')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='GeomML_Probes', help='WandB project name')
    args = parser.parse_args()
    main(args) 