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
from torch.utils.data import WeightedRandomSampler



DEBUG_PRINT = False

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINT is True"""
    if DEBUG_PRINT:
        print(*args, **kwargs)


def make_sampler(labels, p_target=0.2):
    """
    labels: list of 0/1 ints for every sample in your dataset
    p_target: desired fraction of positives in each batch
    """
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    if num_pos == 0 or num_neg == 0:
        return None  # fallback to shuffle

    # solve for w_pos/w_neg so that
    #   P_draw(pos) = (num_pos*w_pos) / (num_pos*w_pos + num_neg*w_neg) = p_target
    # assume w_neg = 1.0
    w_pos = (p_target / (1 - p_target)) * (num_neg / num_pos)
    w_neg = 1.0

    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    
def custom_collate_fn(batch):
    # 1) drop any Nones (if your dataset ever returns None)
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}

    # 2) pull out ds_embedding (list of [num_chunks × hidden_dim] tensors)
    ds_embeddings = [b.pop('ds_embedding') for b in batch]
    
    # 3) now everything left is either:
    #    - a Python int (hadm_id, length, label, mortality_label, readmission_label)
    #    - a fixed-size torch.Tensor (values, mask, static, times)
    collated = default_collate(batch)

    # 4) re-attach your lists of variable-length data
    collated['ds_embedding'] = ds_embeddings
    
    # 5) Add compatibility key for models expecting 'discharge_embeddings'
    collated['discharge_embeddings'] = ds_embeddings
        
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
                    num_workers=12, task_mode='CONTRASTIVE', test_ds_only=False):
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
        test_ds_only=test_ds_only
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
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
        )

    # ─── TRAIN LOADER ─────────────────────────────────────────────────────────
    if task_mode == 'NEXT_24h':

        labels = [label for (_,_,label) in train_dataset.samples]
        sampler = make_sampler(labels, p_target=0.2)
        
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            **loader_kwargs
        )
        
    else:
        # CONTRASTIVE or other: standard shuffle
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs
        )

    # ─── VAL & TEST LOADERS ─────────────────────────────────────────────────────
    # validation and test typically shuffle=False
    val_kwargs = loader_kwargs.copy()
    val_kwargs['shuffle'] = False
    test_kwargs = val_kwargs.copy()

    val_loader = DataLoader(val_dataset, **val_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, val_loader, test_loader, embeddings