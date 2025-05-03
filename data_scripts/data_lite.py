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
from utils import get_device



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
                 chunk_hours=12,
                 label_window=24,
                 T = 80, 
                 test_ds_only = False):
        self.split = split
        self.cache_dir = cache_dir
        self.task_mode = task_mode
        self.chunk_hours = chunk_hours
        self.label_window = label_window
        self.T = T
        self.test_ds_only = test_ds_only
        self.device = get_device()
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
        self.num_ds_embeddings = len(os.listdir(self.ds_embeddings_dir))
        
        # Check if directory exists
        if not os.path.exists(self.ds_embeddings_dir):
            print(f"DS embeddings directory not found. Creating {self.ds_embeddings_dir}")
            os.makedirs(self.ds_embeddings_dir, exist_ok=True)
            self.process_dfs()

        # 1) Load hadm_ids for this split.
        self.merged_with_disch_df = pickle.load(open(os.path.join(cache_dir, "merged_with_disch_df_final_filtered.pkl"), "rb"))
        print('loaded merged_with_disch_df')

        
        self.hadm_ids = self.get_split_hadm_ids(split)
    
        # --- Index merged_with_disch_df by hadm_id for faster lookups --- 
        if self.merged_with_disch_df.index.name != 'hadm_id':
            print("Indexing merged_with_disch_df by hadm_id...")
            self.merged_with_disch_df.set_index('hadm_id', inplace=True)
            print("Indexing complete.")
            print(f"Debug: Checking if hadm_id exists in index: {21086876 in self.merged_with_disch_df.index}")
        # --- End indexing ---

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

    def process_dfs(self):
        """Generate DS embeddings using process_dfs.py"""
        print("No DS embeddings found. Generating them using process_dfs.py...")
        try:
            from data_scripts.process_dfs import MIMICDischargeNotesProcessor, get_ds_dataloader, DSEmbeddingExtractor
            
            # Initialize processor
            print("Initializing discharge notes processor...")
            processor = MIMICDischargeNotesProcessor(cache_dir=self.cache_dir)
            
            # Initialize dataloader with the correct cache_dir
            print(f"Creating discharge notes dataloader with cache_dir: {self.cache_dir}")
            dataloader = get_ds_dataloader(processor, cache_dir=self.cache_dir, batch_size=8)
            
            # Initialize embedding extractor
            print("Initializing embedding extractor...")
            extractor = DSEmbeddingExtractor().to(self.device)
            extractor.eval()
            
            # Process and save embeddings
            print("Extracting embeddings from discharge notes...")
            for batch in tqdm(dataloader, desc="Extracting DS embeddings"):
                hadm_ids = []
                chunks_batch = []
                for hadm_id, chunks in batch:
                    hadm_ids.append(hadm_id)
                    chunks_batch.append(chunks)
                
                with torch.no_grad():
                    # Get embeddings
                    batch_embeddings = extractor(chunks_batch)
                
                # Save each embedding tensor
                for h_id, emb_tensor in zip(hadm_ids, batch_embeddings):
                    save_path = os.path.join(self.ds_embeddings_dir, f"embedding_{h_id}.pt")
                    torch.save(emb_tensor.cpu(), save_path)
            
            print(f"DS embeddings generated and saved to {self.ds_embeddings_dir}")
        except Exception as e:
            print(f"Error generating DS embeddings: {e}")
            print("Continuing without DS embeddings. Some models may not work properly.")
        

    def get_phecode_df(self):
        base_path = '/Users/riccardoconci/Local_documents/!!MIMIC/hosp/'

        # Check if phecode mappings already exist
        phecode_mappings_path = os.path.join(self.cache_dir, "phecode_mappings.pkl")
        if os.path.exists(phecode_mappings_path):
            # Load existing mappings
            mappings = pickle.load(open(phecode_mappings_path, "rb"))
            self.phecode_to_idx = mappings['phecode_to_idx']
            self.idx_to_phecode = mappings['idx_to_phecode']
            self.phe_code_size = mappings['phe_code_size']
            print(f"Loaded PHE code mappings with {self.phe_code_size} unique codes")
            
            # Also load the phecode dataframe if needed
            if not hasattr(self, 'phecode_df') or self.phecode_df is None:
                if os.path.exists(os.path.join(self.cache_dir, "phecode_df.pkl")):
                    self.phecode_df = pickle.load(open(os.path.join(self.cache_dir, "phecode_df.pkl"), "rb"))
            return

        if os.path.exists(os.path.join(self.cache_dir, "phecode_df.pkl")):
            self.phecode_df = pickle.load(open(os.path.join(self.cache_dir, "phecode_df.pkl"), "rb"))
            
            # Create mappings from the existing dataframe
            unique_phecodes = sorted(self.phecode_df['PheCode'].unique())
            self.phecode_to_idx = {code: idx for idx, code in enumerate(unique_phecodes)}
            self.idx_to_phecode = {idx: code for idx, code in enumerate(unique_phecodes)}
            self.phe_code_size = len(unique_phecodes)
            
            # Save the mappings
            mappings = {
                'phecode_to_idx': self.phecode_to_idx,
                'idx_to_phecode': self.idx_to_phecode,
                'phe_code_size': self.phe_code_size
            }
            pickle.dump(mappings, open(phecode_mappings_path, "wb"))
            print(f"Created and saved PHE code mappings with {self.phe_code_size} unique codes")
            return
            
        # If phecode dataframe doesn't exist, create it
        print("Phecode labels not found in cache. Computing...")

        icd_to_phe_mapping = pd.read_csv(base_path + 'icd_to_phecode.csv')
        diagnoses_df = pd.read_csv(base_path + 'diagnoses_icd.csv')
        diagnoses_names_df = pd.read_csv(base_path + 'd_icd_diagnoses.csv')
        icd_to_phe_mapping.columns = ['icd_code','PheCode','icd_version']
        icd_to_phe_mapping['icd_version'] = icd_to_phe_mapping['icd_version'].str.replace('ICD', '')

        diagnoses_df['icd_version'] = diagnoses_df['icd_version'].astype(str)
        icd_to_phe_mapping['icd_version'] = icd_to_phe_mapping['icd_version'].astype(str)
        diagnoses_phecode = diagnoses_df.merge(icd_to_phe_mapping, on=['icd_code', 'icd_version'], how='left')

        diagnoses_names_df['icd_version'] = diagnoses_names_df['icd_version'].astype(str)
        diagnoses_names_df['icd_code'] = diagnoses_names_df['icd_code'].astype(str)
        diagnoses_names_df['icd_version'] = diagnoses_names_df['icd_version'].astype(str)
        diagnoses_phecode_names = diagnoses_phecode.merge(diagnoses_names_df, on=['icd_code', 'icd_version'], how='left')

        diagnoses_phecode_names['Rollup_Status'] = diagnoses_phecode_names['PheCode'].notna().replace({True: '1', False: '0'})
        diagnoses_phecode_names_filtered = diagnoses_phecode_names[diagnoses_phecode_names['Rollup_Status'] == '1']

        # Save the phecode DataFrame
        pickle.dump(diagnoses_phecode_names_filtered, open(os.path.join(self.cache_dir, "phecode_df.pkl"), "wb"))
        self.phecode_df = diagnoses_phecode_names_filtered
        
        # Create and save the mappings
        unique_phecodes = sorted(self.phecode_df['PheCode'].unique())
        self.phecode_to_idx = {code: idx for idx, code in enumerate(unique_phecodes)}
        self.idx_to_phecode = {idx: code for idx, code in enumerate(unique_phecodes)}
        self.phe_code_size = len(unique_phecodes)
        
        mappings = {
            'phecode_to_idx': self.phecode_to_idx,
            'idx_to_phecode': self.idx_to_phecode,
            'phe_code_size': self.phe_code_size
        }
        pickle.dump(mappings, open(phecode_mappings_path, "wb"))
        print(f"Created and saved PHE code mappings with {self.phe_code_size} unique codes")
        
        # Index phecode_df by hadm_id for faster lookups
        if self.phecode_df.index.name != 'hadm_id':
            print("Indexing phecode_df by hadm_id...")
            self.phecode_df.set_index('hadm_id', inplace=True)
            print("Indexing complete.")
        return

    def _next_unique_admission(self, pa: pd.DataFrame, hadm_id: int) -> int | None:
        """
        Return the next distinct hadm_id after the given one, or None if none exists.
        Expects pa to have columns ['hadm_id','admittime'].
        """
        unique_ads = (
            pa[['hadm_id','admittime']]
              .drop_duplicates(subset='hadm_id')
              .sort_values('admittime')
        )
        hadm_list = unique_ads['hadm_id'].tolist()
        try:
            idx = hadm_list.index(hadm_id)
            if idx + 1 < len(hadm_list):
                return hadm_list[idx + 1]
        except ValueError:
            pass
        return None

    def get_post_discharge_label(self, hadm_id: int) -> dict:
        """
        Compute post-discharge outcomes for a specific hospital admission:
          1. mortality_6m
          2. readmission_15d
          3. current_phecodes
          4. next_phecodes

        Returns:
            dict
        """
        labels = {
            'mortality_6m': 0,
            'readmission_15d': 0,
            'current_phecodes': [],
            'next_phecodes': []
        }
        #print(f"hadm_id: {hadm_id}")
        #print(self.merged_with_disch_df.loc[hadm_id])
        try:
            df = self.merged_with_disch_df
            # Store original index state
            original_index_name = df.index.name
            
            # First check if hadm_id exists
            if original_index_name == 'hadm_id':
                if hadm_id not in df.index:
                    raise KeyError
                raw = df.loc[hadm_id]
            else:
                if 'hadm_id' not in df.columns or hadm_id not in df['hadm_id'].values:
                    raise KeyError
                raw = df[df['hadm_id'] == hadm_id]

            patient_data = raw.iloc[0] if isinstance(raw, pd.DataFrame) else raw
            dischtime    = pd.to_datetime(patient_data.get('dischtime'))
            if pd.isna(dischtime):
                return labels

            # 1) mortality within 6 months
            dod = pd.to_datetime(patient_data.get('dod'))
            if pd.notna(dod) and pd.Timedelta(0) < (dod - dischtime) <= pd.Timedelta(days=180):
                labels['mortality_6m'] = 1

            # -- ensure phecode_df loaded --
            if not hasattr(self, 'phecode_df') or self.phecode_df is None:
                self.get_phecode_df()
            pc_df = self.phecode_df

            # 2) current admission phecodes
            if pc_df is not None:
                if pc_df.index.name == 'hadm_id':
                    if hadm_id in pc_df.index:
                        codes = pc_df.loc[hadm_id, 'PheCode']
                    else:
                        codes = []
                else:
                    subset = pc_df[pc_df['hadm_id'] == hadm_id]
                    codes  = subset['PheCode'] if not subset.empty else []
                labels['current_phecodes'] = pd.Series(codes).unique().tolist()

            # 3) readmission & next admission phecodes
            subject_id = patient_data.get('subject_id')
            if subject_id is not None:
                # Create a working copy for admissions lookup
                admissions = df.copy()
                if admissions.index.name == 'hadm_id':
                    admissions = admissions.reset_index()

                pa = admissions[admissions['subject_id'] == subject_id].copy()
                pa['admittime'] = pd.to_datetime(pa['admittime'])
                pa = pa.sort_values('admittime').reset_index(drop=True)

                next_hadm = self._next_unique_admission(pa, hadm_id)
                if next_hadm is not None:
                    next_row   = pa[pa['hadm_id'] == next_hadm].iloc[0]
                    next_admit = pd.to_datetime(next_row['admittime'])
                    if pd.Timedelta(0) < (next_admit - dischtime) <= pd.Timedelta(days=15):
                        labels['readmission_15d'] = 1

                    # next admission phecodes
                    if pc_df is not None:
                        if pc_df.index.name == 'hadm_id':
                            if next_hadm in pc_df.index:
                                codes = pc_df.loc[next_hadm, 'PheCode']
                            else:
                                codes = []
                        else:
                            subset = pc_df[pc_df['hadm_id'] == next_hadm]
                            codes  = subset['PheCode'] if not subset.empty else []
                        labels['next_phecodes'] = pd.Series(codes).unique().tolist()

        except KeyError:
            debug_print(f"Warning: hadm_id {hadm_id} not found.")
        except Exception as e:
            debug_print(f"Warning: error in get_post_discharge_label for {hadm_id}: {e}")

        return labels

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
        
        df = self.merged_with_disch_df.copy()
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
        if self.test_ds_only:
            return self.num_ds_embeddings
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if self.test_ds_only:
            #print(f"Getting item {idx} for test_ds_only")
            ds_embeddings_list = os.listdir(self.ds_embeddings_dir)
            hadm_id = ds_embeddings_list[idx].split('_')[1].split('.')[0]
            #print(f"hadm_id: {hadm_id}")

            embedding_path = os.path.join(self.ds_embeddings_dir, f"embedding_{hadm_id}.pt")
            embedding = torch.load(embedding_path)
            #print(f"embedding: {embedding.shape}")

            post_discharge_labels = self.get_post_discharge_label(int(hadm_id))
            mortality_label = post_discharge_labels['mortality_6m']
            readmission_label = post_discharge_labels['readmission_15d']
            next_phecodes = post_discharge_labels['next_phecodes']
            print(f"hadm_id: {hadm_id}, mortality_label: {mortality_label}, readmission_label: {readmission_label}, next_phecodes: {next_phecodes}")
            
            output = {
                'hadm_id': hadm_id,
                'values': torch.tensor([]), 
                'mask': torch.tensor([]),
                'static': torch.tensor([]),
                'times': torch.tensor([]),
                'length': 0,
                'mortality_label': int(mortality_label),
                'readmission_label': int(readmission_label),
                'next_phecodes': next_phecodes,
                'ds_embedding': embedding,
            }
            return output
        else:
            # Common loader for the "full" cache
            joint_cache_dir = os.path.join(self.cache_dir, "precomputed_tensors")
            os.makedirs(joint_cache_dir, exist_ok=True)
            full_cache_fn   = lambda hid: f"tensor_cache_{hid}.pt"
    
        if self.task_mode == 'CONTRASTIVE':
            hadm_id = self.samples[idx]
            path    = os.path.join(joint_cache_dir, full_cache_fn(hadm_id))
            if not os.path.exists(path):
                raise ValueError(f"No cache for {hadm_id}")
            
            physio, mask, abs_time, rel_time, full_length = torch.load(path)
            ds_path = os.path.join(self.ds_embeddings_dir, f"embedding_{hadm_id}.pt")
            ds_emb  = torch.load(ds_path) if os.path.exists(ds_path) else []
    
            # Get post-discharge labels
            post_discharge_labels = self.get_post_discharge_label(hadm_id)
            mortality_label = post_discharge_labels['mortality_6m']
            readmission_label = post_discharge_labels['readmission_15d']
            next_phecodes = post_discharge_labels['next_phecodes']

            return {
                'hadm_id':      int(hadm_id),
                'values':       physio,
                'mask':         mask,
                'static':       self.get_baseline(hadm_id),
                'times':        abs_time,
                'length':       int(full_length),
                'mortality_label': int(mortality_label),
                'readmission_label': int(readmission_label),
                'next_phecodes': next_phecodes,
                'ds_embedding': ds_emb or [],
            }
    
        elif self.task_mode == 'NEXT_24h':
            hadm_id, chunk_idx, label = self.samples[idx]
            cutoff_hr = chunk_idx * self.chunk_hours
    
            path = os.path.join(joint_cache_dir, full_cache_fn(hadm_id))
            if not os.path.exists(path):
                return None
            physio, mask, abs_time, rel_time, _ = torch.load(path)
    
            # compute slice_end
            rel_np    = rel_time.cpu().numpy()
            slice_end = np.searchsorted(rel_np, cutoff_hr, side='right')
            slice_end = max(1, slice_end)
    
            # slice + pad
            physio_slice = self.pad_tensor(physio[:slice_end], self.T)
            mask_slice   = self.pad_tensor(mask[:slice_end],   self.T)
            time_slice   = self.pad_tensor(abs_time[:slice_end], self.T)
    
            length = int((mask_slice.sum(dim=1) > 0).sum().item())
            
            # Get post-discharge labels
            post_discharge_labels = self.get_post_discharge_label(hadm_id)
            mortality_label = post_discharge_labels['mortality_6m']
            readmission_label = post_discharge_labels['readmission_15d']
            next_phecodes = post_discharge_labels['next_phecodes']
    
            return {
                'hadm_id':      int(hadm_id),
                'values':       physio_slice,
                'mask':         mask_slice,
                'static':       self.get_baseline(hadm_id),
                'times':        time_slice,
                'length':       length,
                'label':        int(label),  # Original mortality label for NEXT_24h task
                'mortality_label': int(mortality_label),
                'readmission_label': int(readmission_label),
                'next_phecodes': next_phecodes,
                'ds_embedding': [],
            }
    
        else:
            raise ValueError(f"Bad mode {self.task_mode}")
    
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




