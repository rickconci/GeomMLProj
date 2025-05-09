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
from models.multi_task_raindrop import MultiTaskRaindropV2
from train_utils import get_device
import wandb
import dotenv
from train_utils import train_one_epoch, evaluate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Multi-task training for MIMIC-IV')
    parser.add_argument('--data_path', type=str, default='/Users/riccardoconci/Local_documents/!!MIMIC', help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite', help='Path to cache directory')
    parser.add_argument('--model_type', type=str, choices=['full', 'ds_only', 'raindrop_v2'], default='raindrop_v2', help='Model type to train')
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
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    # Parameters specific to RaindropV2
    parser.add_argument('--d_model', type=int, default=512, help='Number of expected model input features for RaindropV2')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of transformer layers for RaindropV2')
    parser.add_argument('--global_structure', type=str, default=None, help='Path to adjacency matrix defining sensor relationships')
    parser.add_argument('--sensor_wise_mask', action='store_true', help='Use sensor-wise masking for RaindropV2')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        try:
            dotenv.load_dotenv('dot_env.txt')
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args)
            )
            print("Successfully initialized wandb")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging")
            args.use_wandb = False
    
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
    test_ds_only=False
    )

    
    # Initialize model based on type
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'ds_only':
        model = DSOnlyMultiTaskModel(
            DEVICE=device,
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
            phe_code_size=train_loader.dataset.phecode_size,
            pooling_type=args.pooling_type,
            num_heads=args.num_heads
        )
    elif args.model_type == 'raindrop_v2':
        # Extract dimensions from data
        for batch in train_loader:
            if batch:
                values = batch['values']
                variables_num = values.shape[2]  # number of input features
                timestamps = values.shape[1]     # max_len
                static = batch['static']
                d_static = static.shape[1] if static.numel() > 0 else 0
                break

        # Load global structure if provided
        global_structure = None
        if args.global_structure:
            if os.path.exists(args.global_structure):
                try:
                    global_structure = torch.load(args.global_structure)
                    print(f"Loaded global structure with shape {global_structure.shape}")
                except Exception as e:
                    print(f"Error loading global structure: {e}")
                    print("Initializing with default fully-connected structure")
                    global_structure = torch.ones(variables_num, variables_num)
            else:
                print(f"Global structure file {args.global_structure} not found")
                print("Initializing with default fully-connected structure")
                global_structure = torch.ones(variables_num, variables_num)
        else:
            print("Initializing with default fully-connected structure")
            global_structure = torch.ones(variables_num, variables_num)
        
        print(f"Variables for RaindropV2: d_inp = {variables_num}, \
              d_static: {d_static}, \
              d_model: {args.d_model}, \
              nhead: {args.num_heads}, \
              nhid: {args.hidden_dim}, \
              nlayers: {args.nlayers}, \
              timestamps: {timestamps}, \
              phe_code_size: {train_loader.dataset.phecode_size}, \
              global_structure: {global_structure.shape}, \
              sensor_wise_mask: {args.sensor_wise_mask} \
            ")
        model = MultiTaskRaindropV2(
            DEVICE=device,
            d_inp=variables_num,
            d_model=args.d_model, 
            nhead=args.num_heads, 
            nhid=args.hidden_dim, 
            nlayers=args.nlayers, 
            dropout=0.3, 
            max_len=timestamps, 
            d_static=d_static,
            n_classes=1,  # Binary classification
            phe_code_size=train_loader.dataset.phecode_size,
            global_structure=global_structure,
            sensor_wise_mask=args.sensor_wise_mask
        )
    else:  # 'full' model
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
            phe_code_size=train_loader.dataset.phecode_size,
            task_mode='CONTRASTIVE'
        )
    
    # Set up optimizer
    model = model.to(device)  # Move entire model to device first
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_metric = 0
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
            try:
                # Add numpy types to safe globals
                torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])
                # Try loading with weights_only=True
                checkpoint = torch.load(args.resume_from_checkpoint, map_location=device, weights_only=True)
            except Exception as e:
                print(f"Warning: Failed to load checkpoint with weights_only=True: {e}")
                print("Trying to load with weights_only=False (less secure but more compatible)")
                # If that fails, try loading with weights_only=False
                checkpoint = torch.load(args.resume_from_checkpoint, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)  # Make sure model is on the correct device after loading state dict
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_metric = checkpoint.get('val_metrics', {}).get('mortality_auprc', 0)
            print(f"Resuming from epoch {start_epoch} with best validation metric: {best_val_metric:.4f}")
            
            # Check if the checkpoint has already completed all epochs
            if start_epoch >= args.epochs:
                print(f"Warning: Checkpoint epoch ({start_epoch}) is >= total epochs ({args.epochs})")
                print("Resetting start_epoch to 0 to ensure training continues")
                start_epoch = 0
        else:
            print(f"Checkpoint {args.resume_from_checkpoint} not found. Starting from scratch.")
    
    # Training loop
    print(f"Starting training from epoch {start_epoch} to {args.epochs}")
    if start_epoch < args.epochs:
        for epoch in range(start_epoch, args.epochs):
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
    else:
        print("Skipping training loop because start_epoch >= args.epochs")
    
    # Test best model
    print("\nEvaluating best model on test set...")
    try:
        # Add numpy types to safe globals
        torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype])
        # Try loading the best model with weights_only=True first
        best_checkpoint = torch.load(f'model_{args.model_type}_best.pt', map_location=device, weights_only=True)
    except Exception as e:
        print(f"Warning: Failed to load best model with weights_only=True: {e}")
        print("Trying to load with weights_only=False")
        best_checkpoint = torch.load(f'model_{args.model_type}_best.pt', map_location=device, weights_only=False)
    
    model.load_state_dict(best_checkpoint['model_state_dict'])
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