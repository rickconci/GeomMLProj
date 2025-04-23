import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .LLM_utils import run_LLM  
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt', quiet=True)  # Suppress nltk download messages
from datetime import timedelta
from tqdm import tqdm
import lmdb
import time
import joblib  # Add joblib for memory mapping
import shutil

# Add a global debug flag to control verbosity
DEBUG_PRINT = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINT is True"""
    if DEBUG_PRINT:
        print(*args, **kwargs)

class MIMICContrastivePairsDatasetLite(Dataset):
    def __init__(self, 
                 split='train', 
                 cache_dir='./cache', 
                 task_mode='CONTRASTIVE',
                 chunk_hours=24,
                 label_window=24):
        self.split = split
        self.cache_dir = cache_dir
        self.task_mode = task_mode
        self.chunk_hours = chunk_hours
        self.label_window = label_window

        # ------------- New: Load baseline DataFrame -------------
        baseline_path = os.path.join(cache_dir, "processed_baseline_df.pkl")
        try:
            self.baseline_df = pd.read_pickle(baseline_path)
            print(f"Loaded baseline dataframe from: {baseline_path}")
        except Exception as e:
            print(f"Error loading baseline dataframe from {baseline_path}: {e}")
            self.baseline_df = None

        # ------------- New: Set DS embeddings directory -------------
        self.ds_embeddings_dir = os.path.join(cache_dir, "DS_embeddings")


        # 1) Load hadm_ids for this split.
        self.merged_with_disch_df = pickle.load(open(os.path.join(cache_dir, "merged_with_disch_df_final_filtered.pkl"), "rb"))
        self.hadm_ids = self.get_split_hadm_ids(split)
    

        # 3) Build self.samples.
        if self.task_mode == 'NEXT_24h':
            print(f"Computing samples for NEXT_24h task (this may take a while)...")
            self.compute_samples()
            print(f"Computed {len(self.samples)} samples for NEXT_24h task.")
            if len(self.samples) > 0 and DEBUG_PRINT:
                print(f"First 3 samples: {self.samples[:3]}")
                print(f"Last 3 samples: {self.samples[-3:]}")
        else:
            # CONTRASTIVE mode: one sample per hadm_id.
            self.samples = self.hadm_ids

    def compute_samples(self, cache_file=None):
        """
        Compute and cache samples for the NEXT_24h task.
        Uses adaptive stride to reduce sample count while maintaining resolution near important events.
        """
        if cache_file is None:
            cache_file = os.path.join(
                self.cache_dir, 
                f"samples_{self.split}_chunk{self.chunk_hours}_label{self.label_window}_adaptive.pkl"
            )
            
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.samples = pickle.load(f)
                print(f"Loaded {len(self.samples)} samples from cache: {cache_file}")
                pos_samples = sum(1 for _, _, label in self.samples if label == 1)
                print(f"Found {pos_samples} positive samples ({pos_samples/len(self.samples)*100:.2f}%)")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}; recomputing...")
        
        df = self.splits_loader.merged_with_disch_df.copy()
        df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')
        if 'dischtime' in df.columns:
            df['dischtime'] = pd.to_datetime(df['dischtime'], errors='coerce')
        if 'deathtime' in df.columns:
            df['deathtime'] = pd.to_datetime(df['deathtime'], errors='coerce')
            
        meta_lookup = df.set_index('hadm_id').sort_index()
        self.samples = []
        
        for hadm_id in tqdm(self.hadm_ids, desc="Computing sample windows"):
            if hadm_id not in meta_lookup.index:
                continue

            row_data = meta_lookup.loc[hadm_id]
            row = row_data.iloc[0] if isinstance(row_data, pd.DataFrame) else row_data
            admittime = row['admittime']
            deathtime = row.get('deathtime', None)
            if pd.isnull(deathtime):
                deathtime = None
            dischtime = row.get('dischtime', None)
            if pd.isnull(dischtime):
                dischtime = None

            if deathtime is not None:
                chunk_end_cutoff = deathtime
            else:
                if dischtime is None:
                    continue
                chunk_end_cutoff = dischtime

            if chunk_end_cutoff <= admittime:
                continue

            i = 1
            while True:
                chunk_end_time = admittime + pd.Timedelta(hours=self.chunk_hours * i)
                if chunk_end_time > chunk_end_cutoff:
                    break

                if deathtime is not None:
                    hours_to_death = (deathtime - chunk_end_time).total_seconds() / 3600
                    if hours_to_death <= 24:
                        stride = 1
                    elif hours_to_death <= 72:
                        stride = 2
                    else:
                        stride = 3
                else:
                    hours_to_discharge = (dischtime - chunk_end_time).total_seconds() / 3600
                    if hours_to_discharge <= 48:
                        stride = 2
                    else:
                        stride = 3

                if i % stride == 0:
                    label_start_time = chunk_end_time
                    label_end_time = chunk_end_time + pd.Timedelta(hours=self.label_window)
                    if deathtime is not None and label_start_time <= deathtime < label_end_time:
                        label = 1
                    else:
                        label = 0

                    self.samples.append((hadm_id, i, label))
                i += 1

        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            pos_samples = sum(1 for _, _, label in self.samples if label == 1)
            print(f"Saved {len(self.samples)} samples to cache: {cache_file}")
            print(f"Found {pos_samples} positive samples ({pos_samples/len(self.samples)*100:.2f}%)")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_time = time.time()
        debug_print(f"MIMICContrastivePairsDataset.__getitem__({idx}) started")

        # Define the joint cache directory for full tensors.
        joint_cache_dir = os.path.join(self.cache_dir, "precomputed_tensors")
        os.makedirs(joint_cache_dir, exist_ok=True)
        full_cache_key = lambda hadm: f"tensor_cache_{hadm}.pt"

        if self.task_mode == 'CONTRASTIVE':
            hadm_id = self.samples[idx]
            full_cache_path = os.path.join(joint_cache_dir, full_cache_key(hadm_id))
            
            load_start = time.time()
            if os.path.exists(full_cache_path):
                try:
                    full_data = torch.load(full_cache_path)
                    physio_tensor, mask_tensor, abs_time_tensor, rel_time_tensor, full_length = full_data
                except Exception as e:
                    print(f"Error loading joint full tensor cache for hadm_id {hadm_id}: {e}")
            else:
                raise ValueError(f"Full tensor cache not found for hadm_id {hadm_id}")
            debug_print(f"  - Loading full tensor took {time.time() - load_start:.2f}s")

            full_baseline_tensor = self.get_baseline(hadm_id)

            ds_embed_start = time.time()
            ds_embedding = None
            embedding_path = os.path.join(self.ds_embeddings_dir, f"embedding_{hadm_id}.pt")
            if os.path.exists(embedding_path):
                try:
                    ds_embedding = torch.load(embedding_path)
                except Exception as e:
                    print(f"Error loading DS embedding for hadm_id {hadm_id}: {e}")
            debug_print(f"  - Loading DS embedding took {time.time() - ds_embed_start:.2f}s")

            debug_print(f"MIMICContrastivePairsDataset.__getitem__({idx}) completed in {time.time() - start_time:.2f}s")
            # Ensure baseline tensor is also a standalone tensor if it exists
            if full_baseline_tensor is not None and torch.is_tensor(full_baseline_tensor):
                full_baseline_tensor = full_baseline_tensor.clone().detach().contiguous()
                
            return {
                'id': idx,
                'hadm_id': hadm_id,
                'values': physio_tensor,
                'mask': mask_tensor,
                'static': full_baseline_tensor,
                'times': abs_time_tensor,
                'length': full_length,
                'label': None,
                'ds_embedding': ds_embedding,
            }
        
    
        elif self.task_mode == 'NEXT_24h':
            # Process NEXT_24h samples.
            sample_info = self.samples[idx]
            hadm_id, chunk_index, label = sample_info
            chunk_end_hr = self.chunk_hours * chunk_index
            debug_print(f"  - Processing NEXT_24h sample: hadm_id={hadm_id}, chunk_index={chunk_index}, label={label}")
    
            # First, try to load chunk cache.
            chunk_cache_dir = os.path.join(self.cache_dir, "next24h_chunk_cache")
            os.makedirs(chunk_cache_dir, exist_ok=True)
            tensor_cache_key = f"next24h_{hadm_id}_{chunk_index}_{self.T}.pt"
            tensor_cache_path = os.path.join(chunk_cache_dir, tensor_cache_key)
            chunk_load_start = time.time()
            if os.path.exists(tensor_cache_path):
                try:
                    print('loading chunk cache')
                    cached_data = torch.load(tensor_cache_path)
                    full_physio_tensor, full_mask_tensor, abs_time_tensor, rel_time_tensor, full_length = cached_data
                    full_baseline_tensor = self.get_baseline(hadm_id)
                    debug_print(f"  - Loaded chunk cache in {time.time() - chunk_load_start:.2f}s")
                    
                    return {
                        'id': idx,
                        'hadm_id': hadm_id,
                        'values': full_physio_tensor,
                        'mask': full_mask_tensor,
                        'static': full_baseline_tensor,
                        'times': abs_time_tensor,
                        'length': full_length,
                        'label': label,
                        'ds_embedding': None,
                    }
                
           
                except Exception as e:
                    print(f"Error loading chunk cache for hadm_id {hadm_id} chunk {chunk_index}: {e}")

            # Load full tensor if no valid chunk cache.
            joint_cache_dir = os.path.join(self.cache_dir, "precomputed_tensors")
            os.makedirs(joint_cache_dir, exist_ok=True)
            full_cache_path = os.path.join(joint_cache_dir, full_cache_key(hadm_id))
            full_load_start = time.time()
            if os.path.exists(full_cache_path):
                try:
                    print('loading full tensor cache')
                    full_data = torch.load(full_cache_path)
                    full_physio_tensor, full_mask_tensor, abs_time_tensor, rel_time_tensor, full_length = full_data
                    
                except Exception as e:
                    print(f"Error loading joint full tensor cache for hadm_id {hadm_id}: {e}")
            else:
                # Instead of raising an error, log the issue and return None
                print(f"Warning: Full tensor cache not found for hadm_id {hadm_id}")
                return None
            debug_print(f"  - Loading full tensor for NEXT_24h took {time.time() - full_load_start:.2f}s")
            
            # Determine the slicing index using the time_offsets
            slice_start = time.time()
            full_time_np = rel_time_tensor.cpu().numpy()
            slice_end = np.searchsorted(full_time_np, chunk_end_hr, side='right')
            if slice_end < 1:
                slice_end = 1
            physio_tensor = full_physio_tensor[:slice_end]
            mask_tensor = full_mask_tensor[:slice_end]
            time_hours_tensor = abs_time_tensor[:slice_end]
            length = int((mask_tensor.sum(dim=1) > 0).sum().item())

            physio_tensor = self.pad_tensor(physio_tensor, self.T).clone().detach().contiguous()
            mask_tensor = self.pad_tensor(mask_tensor, self.T).clone().detach().contiguous()
            time_hours_tensor = self.pad_tensor(time_hours_tensor, self.T).clone().detach().contiguous()
            debug_print(f"  - Slicing NEXT_24h tensor took {time.time() - slice_start:.2f}s")

            full_baseline_tensor = self.get_baseline(hadm_id)
            if full_baseline_tensor is not None and torch.is_tensor(full_baseline_tensor):
                full_baseline_tensor = full_baseline_tensor.clone().detach().contiguous()

            save_start = time.time()
            try:
                tensor_data = (physio_tensor, mask_tensor, time_hours_tensor, length, full_baseline_tensor)
                torch.save(tensor_data, tensor_cache_path)
                debug_print(f"  - Saved chunk cache in {time.time() - save_start:.2f}s")
            except Exception as e:
                print(f"Error saving chunk cache: {e}")

            debug_print(f"MIMICContrastivePairsDataset.__getitem__({idx}) completed in {time.time() - start_time:.2f}s")
            return {
                    'id': idx,
                    'hadm_id': hadm_id,
                    'values': full_physio_tensor,
                    'mask': full_mask_tensor,
                    'static': full_baseline_tensor,
                    'times': abs_time_tensor,
                    'length': full_length,
                    'label': label,
                    'ds_embedding': None,
                }
                
        else:
            raise ValueError(f"Invalid task mode: {self.task_mode}")
    
    def pad_tensor(self, tensor, target_length):
        current_length = tensor.shape[0]
        if current_length < target_length:
            pad_size = target_length - current_length
            # Assuming you want to pad along the first dimension.
            padding = torch.zeros(pad_size, *tensor.shape[1:], dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    def select_if_exists(self, df, hadm_id):
        if df is None or df.empty:
            return None
        if df.index.name == 'hadm_id' and hadm_id in df.index:
            return df.loc[[hadm_id]]
        elif 'hadm_id' in df.columns:
            filtered = df[df['hadm_id'] == hadm_id]
            return filtered if not filtered.empty else None
        else:
            return None

    def get_baseline(self, hadm_id):
        """
        Extracts the baseline features from the processed_baseline_df for the given hadm_id.
        The DataFrame is assumed to already contain one-hot encoded features.
        Returns a tensor of the one-hot encoded baseline features.
        """
        meta = self.select_if_exists(self.baseline_df, hadm_id)
        if meta is None or meta.empty:
            # If baseline information is not found, return an empty tensor.
            return torch.tensor([])
        # Convert the first (and assumed only) row to a tensor.
        baseline_tensor = torch.tensor(meta.values[0], dtype=torch.float)
        return baseline_tensor
    
    def split_by_subject_id(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Split hospital admissions into train/val/test sets based on subject_id.
        This ensures that all admissions for a given patient are in the same split.
        
        Args:
            train_ratio: Ratio of patients to use for training
            val_ratio: Ratio of patients to use for validation
            test_ratio: Ratio of patients to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_hadm_ids, val_hadm_ids, test_hadm_ids)
        """
        if df is None:
            raise ValueError("Merged dataframe not loaded. Call load_demographics() first.")
            
        # Verify the ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1.0"
        
        # Get unique subject_ids
        unique_subjects = df['subject_id'].unique()
        
        # Shuffle the subject_ids
        np.random.seed(random_state)
        np.random.shuffle(unique_subjects)
        
        # Calculate split indices
        n_subjects = len(unique_subjects)
        train_idx = int(train_ratio * n_subjects)
        val_idx = train_idx + int(val_ratio * n_subjects)
        
        # Split subject_ids
        train_subjects = unique_subjects[:train_idx]
        val_subjects = unique_subjects[train_idx:val_idx]
        test_subjects = unique_subjects[val_idx:]
        
        # Get hadm_ids for each split
        train_hadm_ids = df[df['subject_id'].isin(train_subjects)]['hadm_id'].tolist()
        val_hadm_ids = df[df['subject_id'].isin(val_subjects)]['hadm_id'].tolist()
        test_hadm_ids = df[df['subject_id'].isin(test_subjects)]['hadm_id'].tolist()
        
        # Store the splits
        self.train_hadm_ids = train_hadm_ids
        self.val_hadm_ids = val_hadm_ids
        self.test_hadm_ids = test_hadm_ids
        
        # Print split statistics
        print(f"Split by subject_id: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects")
        print(f"Corresponding to: {len(train_hadm_ids)} train, {len(val_hadm_ids)} val, {len(test_hadm_ids)} test admissions")
        
        return train_hadm_ids, val_hadm_ids, test_hadm_ids
        
    def get_split_hadm_ids(self, split='train'):
        """
        Get hospital admission IDs for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            list: Hospital admission IDs for the specified split
        """
        self.split_by_subject_id(self.merged_with_disch_df)
        if split == 'train':
            return self.train_hadm_ids
        elif split == 'val':
            return self.val_hadm_ids
        elif split == 'test':
            return self.test_hadm_ids
        else:
            raise ValueError(f"Unknown split: {split}. Must be one of 'train', 'val', 'test'")




