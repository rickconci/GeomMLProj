import os
import torch
import numpy as np
import pytorch_lightning as pl
import logging
from torch.utils.data import DataLoader, random_split, DistributedSampler
from data_scripts.data_lite import MIMICContrastivePairsDatasetLite
from typing import Optional, Union, List, Dict, Any
from train_utils import get_device
from dataloader_lite import custom_collate_fn, get_var_embeddings

from tqdm.auto import tqdm
from pathlib import Path
import time


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
        drop_last: bool = False,
        preload_to_memory: bool = False,
        preload_to_gpu: bool = False
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
            preload_to_memory: Whether to preload all data into memory
            preload_to_gpu: Whether to preload all data to GPU (requires preload_to_memory=True)
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
        
        # Debug log the incoming flag values
        logging.info(f"DataModule init: received preload_to_memory={preload_to_memory}, preload_to_gpu={preload_to_gpu}")
        
        self.preload_to_memory = preload_to_memory
        self.preload_to_gpu = preload_to_gpu and preload_to_memory
        
        # Debug log the effective flag values after processing
        logging.info(f"DataModule init: effective preload_to_memory={self.preload_to_memory}, preload_to_gpu={self.preload_to_gpu}")
        
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
            
            # Preload training data (only this rank's shard) if specified
            if self.preload_to_memory:
                rank = getattr(self.trainer, "global_rank", 0)
                world_size = getattr(self.trainer, "world_size", 1)

                logging.info(f"[rank {rank}] Preloading training shard to "
                             f"{'GPU' if self.preload_to_gpu else 'CPU'} memory …")
                logging.info(f"[rank {rank}] Debug: preload_to_memory={self.preload_to_memory}, "
                             f"preload_to_gpu={self.preload_to_gpu}, cuda_available={torch.cuda.is_available()}")

                # -------- Rank‑specific cache path --------
                cache_dir = Path(self.temp_dfs_path) / "preload_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"train_rank{rank}.pt"

                # -------- Fast path: load cached shard if present --------
                if cache_file.is_file():
                    logging.info(f"[rank {rank}] Loading shard from {cache_file} …")
                    load_start = time.time()
                    cached_shard = torch.load(cache_file, map_location=device)
                    load_seconds = time.time() - load_start
                    logging.info(f"[rank {rank}] Loaded {len(cached_shard)} samples in {load_seconds:.1f}s")

                    # If GPU preload requested and tensors are still on CPU, move them in chunks
                    if self.preload_to_gpu and all(
                        torch.is_tensor(next(iter(s.values()))) and next(iter(s.values())).device.type == "cpu"
                        for s in cached_shard[:1]
                    ):
                        preload_stream = torch.cuda.Stream(device=device)
                        CHUNK_SIZE = 256
                        chunk = []
                        for sample in tqdm(
                            cached_shard,
                            desc=f"[rank {rank}] Moving cached shard to GPU",
                            unit="sample",
                        ):
                            chunk.append(sample)
                            if len(chunk) >= CHUNK_SIZE:
                                with torch.cuda.stream(preload_stream):
                                    for s in chunk:
                                        for k, v in s.items():
                                            if torch.is_tensor(v):
                                                s[k] = v.to(device, non_blocking=True)
                                torch.cuda.current_stream(device).wait_stream(preload_stream)
                                chunk = []
                        # final partial chunk
                        if len(chunk) > 0:
                            with torch.cuda.stream(preload_stream):
                                for s in chunk:
                                    for k, v in s.items():
                                        if torch.is_tensor(v):
                                            s[k] = v.to(device, non_blocking=True)
                            torch.cuda.current_stream(device).wait_stream(preload_stream)
                    # Replace dataset and skip the slow build path
                    self.train_dataset = PreloadedDataset(cached_shard)
                    return

                # 1. Build a deterministic sampler to get this rank's indices
                shard_sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                shard_indices = list(shard_sampler)
                shard_dataset = torch.utils.data.Subset(self.train_dataset, shard_indices)

                # 2. Choose device for optional GPU pinning
                device = (
                    torch.device("cuda", self.trainer.local_rank)
                    if self.preload_to_gpu and torch.cuda.is_available()
                    else torch.device("cpu")
                )
                
                logging.info(f"[rank {rank}] Selected device for preloading: {device}")
                # Create a dedicated CUDA stream for async H2D copies (if GPU preloading)
                preload_stream = (
                    torch.cuda.Stream(device=device)
                    if self.preload_to_gpu and torch.cuda.is_available()
                    else None
                )
                CHUNK_SIZE = 256  # number of samples per CUDA copy batch

                # 3. Stream the shard once and cache every sample, showing progress bar
                temp_loader = DataLoader(
                    shard_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=lambda x: x[0],
                )

                cached_shard = []
                chunk = []
                for sample in tqdm(
                    temp_loader,
                    desc=f"[rank {rank}] Preloading shard",
                    total=len(shard_indices),
                    unit="sample",
                ):
                    chunk.append(sample)
                    # When we've accumulated CHUNK_SIZE samples, copy all tensors in the chunk at once
                    if preload_stream is not None and len(chunk) >= CHUNK_SIZE:
                        with torch.cuda.stream(preload_stream):
                            for s in chunk:
                                for key, val in s.items():
                                    if torch.is_tensor(val):
                                        s[key] = val.to(device, non_blocking=True)
                        # Ensure the default stream waits for preload_stream
                        torch.cuda.current_stream(device).wait_stream(preload_stream)
                        cached_shard.extend(chunk)
                        chunk = []
                # Copy any remaining samples in the last partial chunk
                if preload_stream is not None and len(chunk) > 0:
                    with torch.cuda.stream(preload_stream):
                        for s in chunk:
                            for key, val in s.items():
                                if torch.is_tensor(val):
                                    s[key] = val.to(device, non_blocking=True)
                    torch.cuda.current_stream(device).wait_stream(preload_stream)
                # Append leftover samples (if no GPU preloading, just extend)
                cached_shard.extend(chunk)

                # Save CPU copy for future fast loads
                cpu_copy = [
                    {k: v.cpu() if torch.is_tensor(v) else v for k, v in sample.items()}
                    for sample in cached_shard
                ]
                torch.save(cpu_copy, cache_file)
                logging.info(f"[rank {rank}] Saved shard to {cache_file}")

                # 4. Replace the training dataset with the cached shard
                self.train_dataset = PreloadedDataset(cached_shard)
                logging.info(f"[rank {rank}] Cached {len(cached_shard)} samples.")
            # Preload validation data if needed
                
        elif stage == 'validate':
            # Only create validation dataset
            self.val_dataset = MIMICContrastivePairsDatasetLite(
                split="val",
                cache_dir=self.temp_dfs_path,
                task_mode=self.task_mode,
                chunk_hours=self.chunk_hours,
                label_window=self.label_window,
                T=self.T,
                test_ds_only=self.test_ds_only
            )
            
            # Preload validation data if specified
            if self.preload_to_memory:
                logging.info("Preloading validation data to memory...")
                device = torch.device('cuda' if self.preload_to_gpu and torch.cuda.is_available() else 'cpu')
                
                # Create a temporary dataloader with batch size 1
                temp_loader = DataLoader(
                    self.val_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=lambda x: x[0]
                )
                
                all_data = []
                for batch in temp_loader:
                    if self.preload_to_gpu:
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)
                    all_data.append(batch)
                
                self.val_dataset = PreloadedDataset(all_data)
                logging.info(f"Preloaded {len(all_data)} validation samples to {'GPU' if self.preload_to_gpu else 'memory'}")
        
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
            
            # Preload test data if specified
            if self.preload_to_memory:
                logging.info("Preloading test data to memory...")
                device = torch.device('cuda' if self.preload_to_gpu and torch.cuda.is_available() else 'cpu')
                
                # Create a temporary dataloader with batch size 1
                temp_loader = DataLoader(
                    self.test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=lambda x: x[0]
                )
                
                all_data = []
                for batch in temp_loader:
                    if self.preload_to_gpu:
                        for key in batch:
                            if isinstance(batch[key], torch.Tensor):
                                batch[key] = batch[key].to(device)
                    all_data.append(batch)
                
                self.test_dataset = PreloadedDataset(all_data)
                logging.info(f"Preloaded {len(all_data)} test samples to {'GPU' if self.preload_to_gpu else 'memory'}")
            
            
    def train_dataloader(self):
        """Return the training dataloader"""
        # Check if we're in a distributed setting
        trainer_is_distributed = self.trainer.world_size > 1 if hasattr(self, 'trainer') else False

        if trainer_is_distributed and not self.preload_to_memory:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = not trainer_is_distributed  # keep deterministic order per rank

        loader_args = {
            'batch_size': self.batch_size,
            'sampler': sampler,
            'shuffle': shuffle,
            'num_workers': self.num_workers if not self.preload_to_memory else 0,  # No workers needed if preloaded
            'collate_fn': self.collate_fn,
            'drop_last': self.drop_last,
            'pin_memory': True and not self.preload_to_gpu,  # No need for pin_memory if already on GPU
        }

        # Add prefetch_factor and persistent_workers only if using workers
        if self.num_workers > 0 and not self.preload_to_memory:
            loader_args['persistent_workers'] = True
            loader_args['prefetch_factor'] = 4

        return DataLoader(
            self.train_dataset,
            **loader_args
        )
    
    def val_dataloader(self):
        """Return the validation dataloader"""
        loader_args = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers if not self.preload_to_memory else 0,
            'collate_fn': self.collate_fn,
            'drop_last': self.drop_last,
            'pin_memory': True and not self.preload_to_gpu,
        }
        
        # Add prefetch_factor and persistent_workers only if using workers
        if self.num_workers > 0 and not self.preload_to_memory:
            loader_args['persistent_workers'] = True
            loader_args['prefetch_factor'] = 4
            
        return DataLoader(
            self.val_dataset,
            **loader_args
        )
    
    def test_dataloader(self):
        """Return the test dataloader"""
        loader_args = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers if not self.preload_to_memory else 0,
            'collate_fn': self.collate_fn,
            'drop_last': self.drop_last,
            'pin_memory': True and not self.preload_to_gpu,
        }
        
        # Add prefetch_factor and persistent_workers only if using workers
        if self.num_workers > 0 and not self.preload_to_memory:
            loader_args['persistent_workers'] = True
            loader_args['prefetch_factor'] = 4
            
        return DataLoader(
            self.test_dataset,
            **loader_args
        )
    
    def get_var_embeddings(self):
        """Return variable embeddings for use in the model"""
        return self.var_embeddings


class PreloadedDataset(torch.utils.data.Dataset):
    """Dataset that returns preloaded data directly from memory/GPU"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


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