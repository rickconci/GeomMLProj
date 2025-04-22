import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_scripts.generate_variable_embeddings import VariableEmbeddingGenerator
from torch.utils.data._utils.collate import default_collate

from data_scripts.data import (
    MIMICDemographicsLoader,
    MIMICClinicalEventsProcessor, 
    MIMICDischargeNotesProcessor,
    MIMICContrastivePairsDataset
)

# Custom collate function to handle variable-sized discharge chunks
def custom_collate_fn(batch):
    """
    Custom collate function that:
      - If 'discharge_chunks' is present, we keep it as a list-of-lists and collate everything else with standard logic.
      - If 'discharge_chunks' is NOT present (e.g., in NEXT_24h mode), we just do a default_collate.
    """
    # If the first sample doesn't have 'discharge_chunks', we assume none of them do
    if 'discharge_chunks' not in batch[0]:
        # Just do the normal PyTorch collation
        return default_collate(batch)
    
    # Otherwise, handle discharge_chunks specially
    discharge_chunks = [sample['discharge_chunks'] for sample in batch]
    
    # We remove discharge_chunks from each sample so that default_collate doesn't choke
    # trying to stack them. Then we collate the remainder normally.
    for sample in batch:
        sample.pop('discharge_chunks', None)
    
    collated = default_collate(batch)
    
    # Re-insert discharge_chunks into the collated dictionary
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
    _event_processor = None
    _discharge_processor = None
    _var_embeddings = None
    _data_initialized = False
    
    @classmethod
    def initialize_data_components(cls, base_path, temp_dfs_path):
        """Initialize shared data components for all dataset instances"""
        if cls._data_initialized:
            print("Data components already initialized, skipping...")
            return
            
        print("Initializing shared data components...")
        
        # Initialize demographics loader
        cls._demo_loader = MIMICDemographicsLoader(
            base_path, 
            temp_dfs_path
        )
        
        # Load demographics and split data
        cls._demo_loader.load_demographics()
        
        # Print information about the discharge_df in the demo_loader
        if hasattr(cls._demo_loader, 'discharge_df') and cls._demo_loader.discharge_df is not None:
            print(f"Demographics loader has discharge_df with {len(cls._demo_loader.discharge_df)} rows")
            print(f"Sample hadm_ids in discharge_df: {cls._demo_loader.discharge_df['hadm_id'].head(5).tolist()}")
        else:
            print("Warning: demographics loader doesn't have discharge_df attribute or it's None")
            print("Loading discharge notes CSV file directly...")
            # Try to load discharge notes directly
            discharge_path = os.path.join(base_path, 'note', 'discharge.csv')
            if os.path.exists(discharge_path):
                cls._demo_loader.discharge_df = pd.read_csv(discharge_path)[['hadm_id', 'charttime', 'text']]
                print(f"Loaded discharge notes with {len(cls._demo_loader.discharge_df)} rows")
            else:
                print(f"ERROR: Could not find discharge notes at {discharge_path}")
                
        cls._demo_loader.split_by_subject_id(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        # Initialize events processor
        cls._event_processor = MIMICClinicalEventsProcessor(
            base_path,
            cls._demo_loader.get_hadm_ids(),
            temp_dfs_path
        )
        
        # Load clinical events (this is the expensive operation we want to avoid repeating)
        print("Loading clinical events (this may take a while)...")
        cls._event_processor.load_all_events(load_only_essential=True)
        
        # Make sure we're using the same discharge_df everywhere
        if hasattr(cls._demo_loader, 'discharge_df') and cls._demo_loader.discharge_df is not None:
            cls._event_processor.discharge_df = cls._demo_loader.discharge_df
            
        if hasattr(cls._demo_loader, 'merged_with_disch_df') and cls._demo_loader.merged_with_disch_df is not None:
            cls._event_processor.merged_with_disch_df = cls._demo_loader.merged_with_disch_df
            
        cls._event_processor.process_events()
        cls._cluster_labels  = cls._event_processor.cluster_labels
        
        # Print some information after processing
        print(f"Number of hadm_ids after processing: {len(cls._demo_loader.get_hadm_ids())}")
        
        # Make a copy of the discharge_df to avoid any mutations in the event processor
        discharge_df_copy = None
        if hasattr(cls._demo_loader, 'discharge_df') and cls._demo_loader.discharge_df is not None:
            discharge_df_copy = cls._demo_loader.discharge_df.copy()
        
        # Initialize discharge notes processor - pass the discharge_df from demo_loader
        cls._discharge_processor = MIMICDischargeNotesProcessor(
            cls._demo_loader.get_hadm_ids(),
            temp_dfs_path,
            discharge_df=discharge_df_copy  # Pass the discharge_df here
        )
        
        # Get variable names and create embeddings if needed
        var_names = cls._event_processor.provide_physio_var_names()
        
        # Create variable embeddings generator
        embedding_generator = VariableEmbeddingGenerator(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path
        )
        
        # Generate descriptions and embeddings
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
        self.__class__.initialize_data_components(base_path, temp_dfs_path)
        
        # Initialize split-specific dataset
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize dataset for this split."""
        print(f"Creating MIMICContrastivePairsDataset for {self.split} split, task_mode={self.task_mode} ...")
        
        # Get outcome data
        self.outcome_df = self.__class__._demo_loader.get_30_day_mortality_outcome(self.outcome_choice)
        
        # Create dataset using shared data components
        self.dataset = MIMICContrastivePairsDataset(
            self.__class__._event_processor,
            self.__class__._discharge_processor,
            self.__class__._demo_loader,
            split=self.split,
            T=self.T,
            cache_dir=self.temp_dfs_path if self.use_existing_temp_dfs else self.cache_dir, 
            task_mode=self.task_mode  
        )
        
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
        # Get the sample from the MIMICContrastivePairsDataset
        sample = self.dataset[idx]
        
        # Grab the hadm_id directly from the sample
        hadm_id = sample['hadm_id']
        
        # Extract the components we need
        physio_tensor = sample['physio_tensor']    # [T, F]
        mask_tensor = sample['mask_tensor']        # [T, F]
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

        # (Optional) debug info for the first item
        if idx == 0:
            print(f"[{self.split}] Sample idx=0 => hadm_id={hadm_id}")
            print(f"  physio_tensor.shape: {physio_tensor.shape}")
            print(f"  mask_tensor.shape:   {mask_tensor.shape}")
            print(f"  time_hours_tensor.shape: {time_hours_tensor.shape}")
            print(f"  label.shape: {label.shape}, label={label.item()}")

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
    
    def get_cluster_labels(self):
        """Get cluster labels for clinical variables"""
        return self.__class__._cluster_labels
    
    @staticmethod
    def get_dataloaders(base_path, temp_dfs_path='temp_dfs', batch_size=256, 
                        num_workers=4, outcome_choice='30d_mortality_discharge', task_mode='CONTRASTIVE'):
        """
        Create DataLoader objects for train, validation, and test sets
        
        Args:
            base_path: Path to MIMIC-IV data
            temp_dfs_path: Path to directory with existing processed files
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes for DataLoader
            outcome_choice: Which outcome to predict
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader, var_embeddings, cluster_labels)
        """
        print(f"\nCreating dataloaders for run 1/5...")
        
        # Initialize shared data components first (only happens once)
        MIMIC4KedgnWrapper.initialize_data_components(base_path, temp_dfs_path)
        
        print("Creating train dataset...")
        train_dataset = MIMIC4KedgnWrapper(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path,
            split='train',
            outcome_choice=outcome_choice,
            task_mode=task_mode
        )
        
        print("Creating validation dataset...")
        val_dataset = MIMIC4KedgnWrapper(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path,
            split='val',
            outcome_choice=outcome_choice,
            task_mode=task_mode
        )
        
        print("Creating test dataset...")
        test_dataset = MIMIC4KedgnWrapper(
            base_path=base_path,
            temp_dfs_path=temp_dfs_path,
            split='test',
            outcome_choice=outcome_choice,
            task_mode=task_mode
        )
        
        # Create DataLoaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Get variable embeddings (will be the same for all splits)
        var_embeddings = MIMIC4KedgnWrapper._var_embeddings
        
        # Get cluster labels
        cluster_labels = MIMIC4KedgnWrapper._cluster_labels
        
        return train_loader, val_loader, test_loader, var_embeddings, cluster_labels