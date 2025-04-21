# -*- coding:utf-8 -*-
import os
import argparse
import warnings
import time
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import dotenv
import torch
import numpy as np
import wandb
from model_lightning import KEDGNLightning
from data_lightning import dataset_configs, TimeSeriesDataset, TimeSeriesDataModule
from utils import *

# Load environment variables
dotenv.load_dotenv()

# Configure warnings
warnings.filterwarnings("ignore")

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log')
        # Console output disabled
    ]
)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='physionet', choices=['P12', 'P19', 'physionet', 'mimic3'])
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=4)
parser.add_argument('--rarity_alpha', type=float, default=1)
parser.add_argument('--query_vector_dim', type=int, default=5)
parser.add_argument('--node_emb_dim', type=int, default=8)
parser.add_argument('--plm', type=str, default='bert')
parser.add_argument('--plm_rep_dim', type=int, default=768)
parser.add_argument('--source', type=str, default='gpt')
parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
parser.add_argument('--wandb_project', type=str, default='Geom', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
parser.add_argument('--use_gat', action='store_true', help='Use GAT attention instead of GCN')
parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads for GAT')
parser.add_argument('--use_adj_mask', action='store_true', help='Use adjacency matrix as a mask for GAT attention')
parser.add_argument('--use_transformer', action='store_true', help='Use transformer per variable instead of GRU')
parser.add_argument('--history_len', type=int, default=10, help='History length for transformer model')
parser.add_argument('--nhead_transformer', type=int, default=2, help='Number of attention heads in transformer')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
parser.add_argument('--val_check_interval', type=float, default=0.25, help='Validation check interval')
parser.add_argument('--runs', type=int, default=5, help='Number of runs with different seeds')
parser.add_argument('--precision', type=str, default='16', choices=['16', '32', 'bf16'], help='Precision for training (16, 32, or bf16)')
parser.add_argument('--accumulate_grad_batches', type=int, default=4, help='Number of batches to accumulate gradients over')
parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')

args = parser.parse_args()

# Set device visibility
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set seed for reproducibility
pl.seed_everything(args.seed, workers=True)


def train(args, run_idx=0):
    """Run a single training process"""
    # Set run-specific seed
    pl.seed_everything(args.seed + run_idx, workers=True)
    
    # Get dataset configuration
    config = dataset_configs[args.dataset]
    
    # Initialize the data module
    data_module = TimeSeriesDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        plm=args.plm,
        source=args.source
    )
    
    # Initialize the model
    model = KEDGNLightning(
        hidden_dim=args.hidden_dim,
        num_of_variables=config['variables_num'],
        num_of_timestamps=config['timestamp_num'],
        d_static=config['d_static'],
        n_class=config['n_class'],
        rarity_alpha=args.rarity_alpha,
        query_vector_dim=args.query_vector_dim,
        node_emb_dim=args.node_emb_dim,
        plm_rep_dim=args.plm_rep_dim,
        use_gat=args.use_gat,
        num_heads=args.num_heads,
        use_adj_mask=args.use_adj_mask,
        use_transformer=args.use_transformer,
        history_len=args.history_len,
        nhead_transformer=args.nhead_transformer,
        learning_rate=args.lr
    )
    
    # Setup logging
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.dataset}_run{run_idx}",
            log_model=True,
            group=f"{args.dataset}_{args.hidden_dim}dim"
        )
        # Log hyperparameters
        wandb_logger.log_hyperparams(vars(args))
        wandb_logger.log_hyperparams({
            "run_idx": run_idx,
            "variables_num": config['variables_num'], 
            "timestamp_num": config['timestamp_num'],
            "d_static": config['d_static'],
            "n_class": config['n_class']
        })
    else:
        wandb_logger = None
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_auroc',
            mode='max',
            filename=f'{args.dataset}_' + '{epoch:02d}-{val_auroc:.4f}',
            save_top_k=1,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_auroc',
            patience=5,
            mode='max',
            verbose=True
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        deterministic=True,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    test_results = trainer.test(ckpt_path='best', datamodule=data_module)
    
    # Log final test metrics
    if args.use_wandb:
        wandb.log({
            "final_test_acc": test_results[0]['test_acc'],
            "final_test_auroc": test_results[0]['test_auroc'] if 'test_auroc' in test_results[0] else 0,
            "final_test_auprc": test_results[0]['test_auprc'] if 'test_auprc' in test_results[0] else 0,
            "run_idx": run_idx
        })
    
    # Return test metrics for averaging across runs
    return test_results[0]


def main():
    """Main function to run multiple training processes with different seeds"""
    print(f"Starting {args.runs} runs for dataset {args.dataset}")
    
    # Store results
    all_results = []
    
    # Run multiple trainings with different seeds
    for run_idx in range(args.runs):
        print(f"\n===== Starting Run {run_idx+1}/{args.runs} =====")
        # Finish previous wandb run if exists
        if args.use_wandb and wandb.run is not None:
            wandb.finish()
        
        # Train and get results
        results = train(args, run_idx)
        all_results.append(results)
        
        print(f"Run {run_idx+1} completed with test acc: {results['test_acc']:.4f}, "
              f"test auroc: {results.get('test_auroc', 0):.4f}, "
              f"test auprc: {results.get('test_auprc', 0):.4f}")
    
    # Calculate and log average metrics
    avg_acc = np.mean([r['test_acc'] for r in all_results])
    std_acc = np.std([r['test_acc'] for r in all_results])
    
    if 'test_auroc' in all_results[0]:
        avg_auroc = np.mean([r['test_auroc'] for r in all_results])
        std_auroc = np.std([r['test_auroc'] for r in all_results])
        avg_auprc = np.mean([r['test_auprc'] for r in all_results])
        std_auprc = np.std([r['test_auprc'] for r in all_results])
    else:
        avg_auroc, std_auroc = 0.0, 0.0
        avg_auprc, std_auprc = 0.0, 0.0
    
    print("\n===== Final Results =====")
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy = {avg_acc*100:.1f}±{std_acc*100:.1f}%")
    print(f"AUROC    = {avg_auroc*100:.1f}±{std_auroc*100:.1f}%")
    print(f"AUPRC    = {avg_auprc*100:.1f}±{std_auprc*100:.1f}%")
    
    # Log final averaged metrics to wandb
    if args.use_wandb:
        if wandb.run is not None:
            wandb.finish()
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.dataset}_final_summary",
            job_type="summary"
        )
        wandb.log({
            "final_mean_acc": avg_acc * 100,
            "final_std_acc": std_acc * 100,
            "final_mean_auroc": avg_auroc * 100,
            "final_std_auroc": std_auroc * 100,
            "final_mean_auprc": avg_auprc * 100,
            "final_std_auprc": std_auprc * 100,
            "num_runs": args.runs
        })
        wandb.finish()


if __name__ == "__main__":
    main() 