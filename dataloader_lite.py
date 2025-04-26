import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_scripts.generate_variable_embeddings import VariableEmbeddingGenerator
from torch.utils.data._utils.collate import default_collate
import time
import pickle
import threading
from data_scripts.data_lite import MIMICContrastivePairsDatasetLite

DEBUG_PRINT = False

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINT is True"""
    if DEBUG_PRINT:
        print(*args, **kwargs)


# Custom collate function to handle variable-sized discharge chunks
def custom_collate_fn(batch):
    """
    Custom collate function that:
      - Filters out None values that may be returned from the dataset's __getitem__ method
      - If 'discharge_chunks' is present, we keep it as a list-of-lists and collate everything else with standard logic.
      - If 'ds_embedding' is present, we keep it as a list of variable-sized tensors
      - If 'discharge_chunks' is NOT present (e.g., in NEXT_24h mode), we just do a default_collate.
      - Ensures all tensors are detached, contiguous, and have compatible shapes before batching.
    """
    # Filter out None values before collation
    batch = [item for item in batch if item is not None]
    
    # Early exit optimization: if the batch is empty, return an empty tensor
    if not batch:
        return {}
    
    # Extract discharge_chunks and ds_embedding from all samples in advance
    discharge_chunks = None
    ds_embeddings = None
    
    # Check if we need to handle discharge_chunks specially
    if 'discharge_chunks' in batch[0]:
        # Store and remove discharge_chunks from each sample
        discharge_chunks = [sample.pop('discharge_chunks', None) for sample in batch]
    
    # Check if we need to handle ds_embedding specially
    if 'ds_embedding' in batch[0]:
        # Store and remove ds_embedding from each sample
        ds_embeddings = [sample.pop('ds_embedding', None) for sample in batch]
    
    # Make a copy of the batch with safe-to-batch tensors
    safe_batch = []
    
    # Process each sample to ensure tensors are detached and contiguous
    for sample in batch:
        safe_sample = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                # Ensure tensor is detached and contiguous
                safe_sample[key] = value
            else:
                safe_sample[key] = value
        safe_batch.append(safe_sample)
    
    try:
        # Try standard collation
        collated = default_collate(safe_batch)
    except RuntimeError as e:
        # If standard collation fails, use a more careful approach
        print(f"Warning: Standard collation failed, trying one-by-one approach: {e}")
        
        # Initialize with first item's keys
        collated = {}
        keys_to_collate = list(safe_batch[0].keys())
        
        # Process each key separately
        for key in keys_to_collate:
            try:
                values = [sample[key] for sample in safe_batch]
                collated[key] = default_collate(values)
            except Exception as key_error:
                print(f"Could not collate field '{key}', keeping as list: {key_error}")
                collated[key] = values  # Keep as list if collation fails
    
    # Re-insert discharge_chunks if we extracted them
    if discharge_chunks is not None:
        collated['discharge_chunks'] = discharge_chunks
    
    # Re-insert ds_embeddings if we extracted them
    if ds_embeddings is not None:
        collated['ds_embedding'] = ds_embeddings
    
    return collated


def get_var_embeddings(data_path, temp_dfs_path):
    """
    Get variable embeddings for MIMIC-IV data
    """
    print("Loading variable names...")
    var_names = pickle.load(open(os.path.join(temp_dfs_path, 'var_names.pkl'), 'rb'))
    print("Generating embeddings...")
    embeddings, descriptions = VariableEmbeddingGenerator(data_path, temp_dfs_path).generate_embeddings(var_names)
    return embeddings, descriptions


def get_dataloaders(data_path, temp_dfs_path='temp_dfs', batch_size=64, 
                    num_workers=12, task_mode='CONTRASTIVE'):
    """
    Create DataLoader objects for train, validation, and test sets
    
    Args:
        base_path: Path to MIMIC-IV data
        temp_dfs_path: Path to directory with existing processed files
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, var_embeddings)
    """
    print(f"\nCreating dataloaders for {task_mode} mode...")
    
    # Initialize shared data components first (only happens once)
    embeddings, _ = get_var_embeddings(data_path, temp_dfs_path)

    dataset_kwargs = dict(
        cache_dir=temp_dfs_path,
        task_mode=task_mode,
        chunk_hours=12,
        label_window=24,
    )
    print("Creating train dataset...")
    train_dataset = MIMICContrastivePairsDatasetLite(split='train',  **dataset_kwargs)
    print("Creating validation dataset...")
    val_dataset   = MIMICContrastivePairsDatasetLite(split='val',    **dataset_kwargs)
    print("Creating test dataset...")
    test_dataset  = MIMICContrastivePairsDatasetLite(split='test',   **dataset_kwargs)

    # Base kwargs
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # validation and test typically shuffle=False
    val_kwargs = loader_kwargs.copy()
    val_kwargs['shuffle'] = False
    test_kwargs = val_kwargs.copy()

    val_loader = DataLoader(val_dataset, **val_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, val_loader, test_loader, embeddings
    