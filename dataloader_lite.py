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

from data_scripts.data_lite import MIMICContrastivePairsDatasetLite, MIMICDemographicsLoader, MIMICDischargeNotesProcessor

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
      - If 'discharge_chunks' is NOT present (e.g., in NEXT_24h mode), we just do a default_collate.
      - Ensures all tensors are detached, contiguous, and have compatible shapes before batching.
    """
    # Filter out None values before collation
    batch = [item for item in batch if item is not None]
    
    # Early exit optimization: if the batch is empty, return an empty tensor
    if not batch:
        return {}
    
    # Extract discharge_chunks from all samples in advance
    discharge_chunks = None
    
    # Check if we need to handle discharge_chunks specially
    if 'discharge_chunks' in batch[0]:
        # Store and remove discharge_chunks from each sample
        discharge_chunks = [sample.pop('discharge_chunks', None) for sample in batch]
    
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
    
    return collated

class MIMIC4KedgnWrapper(Dataset):
    """
    Wrapper around MIMICContrastivePairsDataset that formats the data
    for direct use with the KEDGN model.
    
    This preserves all the careful data processing logic in the original dataset
    while adapting the output format.
    """
    # Class-level variables to store shared data components
    _demo_loader = None
    _discharge_processor = None
    _var_embeddings = None
    _data_initialized = False
    
    @classmethod
    def initialize_data_components(cls, base_path, temp_dfs_path, load_discharges=False):
        """Initialize shared data components for all dataset instances"""
        if cls._data_initialized:
            print("Data components already initialized, skipping...")
            return
            
        start_time = time.time()
        print("Initializing shared data components...")
        
        cls._demo_loader = MIMICDemographicsLoader(
            base_path, 
            temp_dfs_path
        )
        
        # Load demographics and split data
        cls._demo_loader.load_demographics()
   
        if hasattr(cls._demo_loader, 'discharge_df') and cls._demo_loader.discharge_df is not None:
            print(f"Demographics loader has discharge_df with {len(cls._demo_loader.discharge_df)} rows")
        else:
            print("Warning: demographics loader doesn't have discharge_df attribute or it's None")
            
            # Load discharge notes directly
            discharge_path = os.path.join(base_path, 'note', 'discharge.csv')
            if os.path.exists(discharge_path):
                print(f"Loading discharge notes from {discharge_path}")
                discharge_df = pd.read_csv(discharge_path)[['hadm_id', 'charttime', 'text']]
                # Store the discharge_df in the demo_loader
                cls._demo_loader.discharge_df = discharge_df
                print(f"Loaded discharge notes with {len(discharge_df)} rows")
            else:
                print(f"ERROR: Could not find discharge notes at {discharge_path}")
        
        cls._demo_loader.split_by_subject_id(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
  
        # First try to get from demo_loader
        if hasattr(cls._demo_loader, 'discharge_df') and cls._demo_loader.discharge_df is not None:
            discharge_df_to_use = cls._demo_loader.discharge_df.copy()
            print(f"Using discharge_df from demographics loader with {len(discharge_df_to_use)} rows")
        # If not available in demo_loader, try to load directly
        else:
            discharge_path = os.path.join(base_path, 'note', 'discharge.csv')
            if os.path.exists(discharge_path):
                print(f"Loading discharge notes directly from {discharge_path}")
                discharge_df_to_use = pd.read_csv(discharge_path)[['hadm_id', 'charttime', 'text']]
                print(f"Loaded discharge notes with {len(discharge_df_to_use)} rows")
        
        if load_discharges:
            cls._discharge_processor = MIMICDischargeNotesProcessor(
                cls._demo_loader.get_hadm_ids(),
                temp_dfs_path,
                discharge_df=discharge_df_to_use  # Pass the discharge_df here
            )
        
        var_names = pickle.load(open(os.path.join(temp_dfs_path, "var_names.pkl"), "rb"))
        embedding_generator = VariableEmbeddingGenerator(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path
        )
        
        embedding_path = os.path.join(temp_dfs_path, "mimic4_bert_var_rep_gpt_source.pt")
        if os.path.exists(embedding_path):
            print(f"Loading variable embeddings from {embedding_path}")
            cls._var_embeddings = torch.load(embedding_path)
        else:
            print("Creating new variable embeddings")
            descriptions = embedding_generator.generate_descriptions(var_names)
            cls._var_embeddings = embedding_generator.generate_embeddings(descriptions, "mimic4_bert_var_rep_gpt_source.pt")
            print(f"Saved variable embeddings to {embedding_path}")        
        cls._data_initialized = True
        elapsed = time.time() - start_time
        print(f"Data initialization completed in {elapsed:.2f} seconds")
    
    def __init__(self, 
                 base_path, 
                 temp_dfs_path='temp_dfs',
                 split='train', 
                 outcome_choice='30d_mortality_discharge',
                 use_existing_temp_dfs=True,
                 cache_dir='./cache',
                 T=96, 
                 task_mode='CONTRASTIVE'):
        """
        Initialize MIMIC4KedgnWrapper.
        
        Args:
            base_path: Path to MIMIC-IV data
            temp_dfs_path: Path to directory with existing processed files
            split: Data split ('train', 'val', or 'test')
            outcome_choice: Which outcome to predict ('30d_mortality_discharge' or '48h_mortality')
            use_existing_temp_dfs: Whether to use existing processed files in temp_dfs directory
            cache_dir: Directory to cache processed data
            T: Number of time steps (default: 96)
        """
        self.base_path = base_path
        self.temp_dfs_path = temp_dfs_path
        self.split = split
        self.outcome_choice = outcome_choice
        self.use_existing_temp_dfs = use_existing_temp_dfs
        self.cache_dir = cache_dir
        self.T = T
        self.task_mode = task_mode
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize shared data components if not already done
        if task_mode == 'CONTRASTIVE':
            load_discharges = True  # for contrastive, we need discharges
        else:
            load_discharges = False  # for next-24h, we don't need discharges
        self.__class__.initialize_data_components(base_path, temp_dfs_path, load_discharges)
        
        # Initialize split-specific dataset
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize dataset for this split."""
        print(f"Creating MIMICContrastivePairsDataset for {self.split} split, task_mode={self.task_mode} ...")
        
        # Get outcome data
        self.outcome_df = self.__class__._demo_loader.get_30_day_mortality_outcome(self.outcome_choice)
        
        itemid_list= pickle.load(open(os.path.join(self.temp_dfs_path, "itemid_list.pkl"), "rb"))

        # Create dataset using shared data components
        self.dataset = MIMICContrastivePairsDatasetLite(
            splits_loader=self.__class__._demo_loader,
            itemid_list=itemid_list,
            split=self.split, 
            T=self.T, 
            cache_dir=self.cache_dir, 
            task_mode=self.task_mode)
        
        # Store hospital admission IDs for this split
        self.hadm_ids = self.dataset.hadm_ids  # In NEXT_24h mode, each hadm_id can appear multiple times in `samples`
        print(f"Found {len(self.hadm_ids)} hadm_ids in {self.split} split; dataset size = {len(self.dataset)} samples.")


    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample and format it for KEDGN model
        Returns a dictionary with keys needed by KEDGN model
        """
        # Get the sample from the MIMICContrastivePairsDataset without debug prints
        sample = self.dataset[idx]
        
        # If the dataset returned None (missing tensor cache), return None
        if sample is None:
            return None
        
        # Grab the hadm_id directly from the sample
        hadm_id = sample['hadm_id']

        
        # Extract the components we need
        physio_tensor = sample['physio_tensor'] #[T, F]
        mask_tensor = sample['mask_tensor']     # [T, F]
        baseline_tensor = sample.get('baseline_tensor', None)  # [D] or None
        time_hours_tensor = sample['time_hours_tensor']        # [T]
        length = sample['length']                  # scalar
        discharge_chunks = sample.get('discharge_chunks', None)
        
        # If chunk-based, get the next-24h label
        # Otherwise, get 30d mortality from outcome_df
        if self.task_mode == 'NEXT_24h':
            # the dataset sample already has 'label_24h_mortality'
            label_24h = sample['label_24h_mortality']  # 0 or 1
            label = torch.tensor(label_24h, dtype=torch.float).unsqueeze(0)  # shape [1]
        else:
            # 'CONTRASTIVE' mode: we rely on your existing outcome_df
            # e.g., outcome_df.loc[hadm_id, 'label_death_within_30d'] if it exists
            if hadm_id in self.outcome_df.index:
                val = self.outcome_df.loc[hadm_id, 'label_death_within_30d']
                label = torch.tensor(val, dtype=torch.float).unsqueeze(0)
            else:
                label = torch.tensor(0, dtype=torch.float).unsqueeze(0)
        
        return {
            'id': idx,
            'hadm_id': hadm_id,         # might be helpful for debugging
            'values': physio_tensor,    # [T, F]
            'mask': mask_tensor,        # [T, F]
            'static': baseline_tensor,  # [D] or None
            'times': time_hours_tensor, # [T]
            'length': length,           # scalar
            'label': label,             # [1]
            'discharge_chunks': discharge_chunks
        }
    def get_variable_embeddings(self):
        """Get embeddings for clinical variables"""
        return self.__class__._var_embeddings
    
    
    @staticmethod
    def get_dataloaders(base_path, temp_dfs_path='temp_dfs', batch_size=64, 
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
        MIMIC4KedgnWrapper.initialize_data_components(base_path, temp_dfs_path)
        
        print("Creating train dataset...")
        train_dataset = MIMIC4KedgnWrapper(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path,
            split='train',
            task_mode=task_mode
        )
        
        print("Creating validation dataset...")
        val_dataset = MIMIC4KedgnWrapper(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path,
            split='val',
            task_mode=task_mode
        )
        
        print("Creating test dataset...")
        test_dataset = MIMIC4KedgnWrapper(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path,
            split='test',
            task_mode=task_mode
        )
        
        # Create DataLoaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True, 
            prefetch_factor=2, 
            collate_fn=custom_collate_fn, 
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2, 
            collate_fn=custom_collate_fn,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2, 
            collate_fn=custom_collate_fn,
        )
        
        # Get variable embeddings (will be the same for all splits)
        var_embeddings = MIMIC4KedgnWrapper._var_embeddings
        
        return train_loader, val_loader, test_loader, var_embeddings