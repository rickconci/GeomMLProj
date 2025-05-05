import os
import torch
import numpy as np
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader, random_split
from data_scripts.data_lite import MIMICContrastivePairsDatasetLite
from typing import Optional, Union, List, Dict, Any
from train_utils import get_device
from dataloader_lite import custom_collate_fn, get_var_embeddings


def float32_collate_fn(batch):
    """Custom collate function that ensures all float tensors are float32"""
    # First use the standard collate function
    result = custom_collate_fn(batch)
    
    # Then convert any float64 to float32
    for key in result:
        if isinstance(result[key], torch.Tensor) and result[key].dtype == torch.float64:
            result[key] = result[key].to(dtype=torch.float32)
    
    return result


class ContrastiveDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for contrastive learning with MIMIC data"""
    
    def __init__(
        self,
        data_path: str,
        temp_dfs_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        task_mode: str = "CONTRASTIVE",
        chunk_hours: int = 12,
        label_window: int = 24,
        T: int = 80,
        test_ds_only: bool = False,
        collate_fn = None,
        drop_last: bool = False
    ):
        """
        Initialize the DataModule with configuration parameters.
        
        Args:
            data_path: Path to the MIMIC data
            temp_dfs_path: Path to store temporary dataframes
            batch_size: Batch size for the dataloaders
            num_workers: Number of worker processes for dataloading
            task_mode: Mode of operation ('CONTRASTIVE' or 'NEXT_24h')
            chunk_hours: Number of hours in each chunk for NEXT_24h task
            label_window: Window size for labels in NEXT_24h task
            T: Max sequence length
            test_ds_only: Flag to only use discharge summaries for test data
            collate_fn: Optional custom collate function
            drop_last: Whether to drop the last incomplete batch
        """
        super().__init__()
        self.data_path = data_path
        self.temp_dfs_path = temp_dfs_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_mode = task_mode
        self.chunk_hours = chunk_hours
        self.label_window = label_window
        self.T = T
        self.test_ds_only = test_ds_only
        self.collate_fn = collate_fn or float32_collate_fn
        self.drop_last = drop_last
        self.device = get_device()
        
        # These will be set in the setup method
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.var_embeddings = None
    
    def prepare_data(self):
        """
        Prepare data for use in DataLoaders.
        This method is called only once and on only one GPU.
        """
        # Check if data directories exist
        os.makedirs(self.temp_dfs_path, exist_ok=True)
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for use in DataLoaders.
        This method is called on every process when using DDP.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Ensure PyTorch uses float32 by default
        torch.set_default_dtype(torch.float32)
        
        # Create the datasets based on the stage
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = MIMICContrastivePairsDatasetLite(
                split="train",
                cache_dir=self.temp_dfs_path,
                task_mode=self.task_mode,
                chunk_hours=self.chunk_hours,
                label_window=self.label_window,
                T=self.T,
                test_ds_only=self.test_ds_only
            )
            
            # Validation dataset
            self.val_dataset = MIMICContrastivePairsDatasetLite(
                split="val",
                cache_dir=self.temp_dfs_path,
                task_mode=self.task_mode,
                chunk_hours=self.chunk_hours,
                label_window=self.label_window,
                T=self.T,
                test_ds_only=self.test_ds_only
            )
        
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = MIMICContrastivePairsDatasetLite(
                split="test",
                cache_dir=self.temp_dfs_path,
                task_mode=self.task_mode,
                chunk_hours=self.chunk_hours,
                label_window=self.label_window,
                T=self.T,
                test_ds_only=self.test_ds_only
            )
            
        # Load the variable embeddings
        embeddings, _ = get_var_embeddings(self.data_path, self.temp_dfs_path)
        
        # Ensure embeddings are float32
        if isinstance(embeddings, torch.Tensor) and embeddings.dtype == torch.float64:
            logging.info("Converting variable embeddings from float64 to float32 for MPS compatibility")
            embeddings = embeddings.to(dtype=torch.float32)
            
        self.var_embeddings = embeddings
    
    def train_dataloader(self):
        """Return the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Return the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last,
            pin_memory=True
        )
    
    def get_var_embeddings(self):
        """Return variable embeddings for use in the model"""
        return self.var_embeddings




def get_model_dimensions(data_module):
    """Extract dimensions from data for model initialization"""
    logging.info("Getting model dimensions from data module...")
    
    # Ensure PyTorch uses float32 by default
    torch.set_default_dtype(torch.float32)
    
    # Prepare data module to get a sample batch
    data_module.setup(stage='fit')
    
    # Get a sample batch
    train_loader = data_module.train_dataloader()
    sample_batch = next(iter(train_loader))
    
    # Check for float64 tensors and convert them to float32
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor) and sample_batch[key].dtype == torch.float64:
            logging.warning(f"Converting {key} from float64 to float32 for MPS compatibility")
            sample_batch[key] = sample_batch[key].to(dtype=torch.float32)
    
    dims = {}
    # Get basic dimensions
    dims['values_shape'] = sample_batch['values'].shape
    dims['variables_num'] = sample_batch['values'].shape[2]
    dims['timestamps'] = sample_batch['values'].shape[1]
    
    # Get static features dimension if present
    if 'static' in sample_batch and sample_batch['static'].numel() > 0:
        dims['d_static'] = sample_batch['static'].shape[1]
    else:
        dims['d_static'] = 0
    
    # Get discharge summary embedding dimension if present
    if 'discharge_embeddings' in sample_batch:
        if isinstance(sample_batch['discharge_embeddings'], torch.Tensor):
            dims['ds_emb_dim'] = sample_batch['discharge_embeddings'].shape[-1]
        elif isinstance(sample_batch['discharge_embeddings'], list) and len(sample_batch['discharge_embeddings']) > 0:
            sample_emb = sample_batch['discharge_embeddings'][0]
            if hasattr(sample_emb, 'shape'):
                dims['ds_emb_dim'] = sample_emb.shape[-1]
            else:
                dims['ds_emb_dim'] = 768  # Default for BERT-like models
        else:
            dims['ds_emb_dim'] = 768
    elif 'ds_embedding' in sample_batch:
        if isinstance(sample_batch['ds_embedding'], torch.Tensor):
            dims['ds_emb_dim'] = sample_batch['ds_embedding'].shape[-1]
        elif isinstance(sample_batch['ds_embedding'], list) and len(sample_batch['ds_embedding']) > 0:
            sample_emb = sample_batch['ds_embedding'][0]
            if hasattr(sample_emb, 'shape'):
                dims['ds_emb_dim'] = sample_emb.shape[-1]
            else:
                dims['ds_emb_dim'] = 768
        else:
            dims['ds_emb_dim'] = 768
    else:
        dims['ds_emb_dim'] = 768
    
    logging.info(f"Determined dimensions: {dims}")
    return dims