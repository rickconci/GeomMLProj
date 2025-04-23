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


###############################################################################
# 1. MIMICDemographicsLoader: Loads demographics and basic merged data
###############################################################################

class MIMICDemographicsLoader:
    def __init__(self, base_path: str, cache_dir: str, fast_load: bool = True):
        """
        Initialize file paths and internal state from the base path.
        Loads demographics data and manages train/val/test splits by subject_id.
        """
        self.base_path = base_path
        self.cache_dir = cache_dir
        self.initialize_paths()
        self.patients_df = None
        self.admissions_df = None
        self.transfers_df = None
        self.discharge_df = None
        self.merged_df = None
        self.merged_with_disch_df = None
        self.train_hadm_ids = None
        self.val_hadm_ids = None 
        self.test_hadm_ids = None

        if fast_load:
            if os.path.exists(os.path.join(self.cache_dir, "sorted_filtered_df.pkl")):
                self.sorted_filtered_normalised_df = pickle.load(open(os.path.join(self.cache_dir, "sorted_filtered_df.pkl"), "rb"))
                print("Loaded sorted_filtered_normalised_df from pickle")
            else:
                self.sorted_filtered_normalised_df = None
                print("Sorted_filtered_normalised_df not found in pickle")

            if os.path.exists(os.path.join(self.cache_dir, "merged_with_disch_df_final_filtered.pkl")):
                self.merged_with_disch_df = pickle.load(open(os.path.join(self.cache_dir, "merged_with_disch_df_final_filtered.pkl"), "rb"))
                print("Loaded merged_with_disch_df from pickle")
            else:
                self.merged_with_disch_df = None
                print("Merged_with_disch_df not found in pickle")

        

    def initialize_paths(self):
        # Basic demographics and note files
        self.patients_path   = os.path.join(self.base_path, 'hosp', 'patients.csv')
        self.admission_path  = os.path.join(self.base_path, 'hosp', 'admissions.csv')
        self.transfers_path  = os.path.join(self.base_path, 'hosp', 'transfers.csv')
        self.disch_path      = os.path.join(self.base_path, 'note', 'discharge.csv')
        
    def load_demographics_initial(self):
        """
        Loads patients, admissions, transfers, and discharge notes,
        merges them, computes 'age' and admission duration, and filters 
        based on available discharge notes.

        If the processed DataFrame already exists as a pickle file in the 
        'tetm_dfs' directory, it is loaded directly. Otherwise, the CSV files 
        are processed and the result is saved for faster future loads.
        """
        
        # Otherwise, load and merge the CSVs
        print("Pickle not found. Processing CSV files...")
        self.patients_df  = pd.read_csv(self.patients_path)
        self.admissions_df = pd.read_csv(self.admission_path)
        self.transfers_df = pd.read_csv(self.transfers_path)
        print('Loading discharge notes')
        self.discharge_df = pd.read_csv(self.disch_path)[['hadm_id', 'charttime', 'text']]

        # Merge patients and admissions
        self.merged_df = pd.merge(self.admissions_df, self.patients_df, on='subject_id', how='left')
        self.merged_df['admittime'] = pd.to_datetime(self.merged_df['admittime'])
        
        # Compute age
        self.merged_df['age'] = (self.merged_df['admittime'].dt.year -
                                self.merged_df['anchor_year'] +
                                self.merged_df['anchor_age'])
        
        # Merge in transfers
        self.merged_df = pd.merge(self.merged_df, self.transfers_df, on=['hadm_id', 'subject_id'], how='left')
        self.merged_df['dischtime'] = pd.to_datetime(self.merged_df['dischtime'])
        
        # Calculate admission duration in days
        self.merged_df['admit_duration'] = (
            (self.merged_df['dischtime'] - self.merged_df['admittime']).dt.total_seconds() / (3600 * 24)
        )
        
        # Filter based on available discharge notes
        print('Filtering based on available discharge notes')
        filtered_ds = self.merged_df['hadm_id'].isin(self.discharge_df['hadm_id'])
        self.merged_with_disch_df = self.merged_df.loc[filtered_ds]
        print('Keeping only admissions with admission duration between 2 and 10 days')
        self.merged_with_disch_df = self.merged_with_disch_df[self.merged_with_disch_df['admit_duration'] <= 10]
        self.merged_with_disch_df = self.merged_with_disch_df[self.merged_with_disch_df['admit_duration'] >= 2]


        #annotate death periods
        self.merged_with_disch_df['deathtime'] = pd.to_datetime(self.merged_with_disch_df['deathtime'], errors='coerce')
        self.merged_with_disch_df['dod'] = pd.to_datetime(self.merged_with_disch_df['dod'], errors='coerce')
        self.merged_with_disch_df['death_timestamp'] = self.merged_with_disch_df['deathtime'].combine_first(self.merged_with_disch_df['dod'])

        self.merged_with_disch_df['days_disch_to_death'] = (
            (self.merged_with_disch_df['death_timestamp'] - self.merged_with_disch_df['dischtime']).dt.total_seconds() / (3600 * 24)
        )
        self.merged_with_disch_df['hours_admission_to_death'] = (
            (self.merged_with_disch_df['death_timestamp'] - self.merged_with_disch_df['admittime']).dt.total_seconds() / 3600
        )
        self.merged_with_disch_df['days_admission_to_death'] = (
            (self.merged_with_disch_df['death_timestamp'] - self.merged_with_disch_df['admittime']).dt.total_seconds() / (3600*24)
        )
        
        #filter out patients with no physio events
        self.merged_with_disch_df = self.adjust_demographics_physioevents()

    def adjust_demographics_physioevents(self):
        """
        Adjust demographics with physio events.
        """
         # Define the pickle file path
        pickle_path = os.path.join(self.cache_dir, "merged_with_disch_df_final_filtered.pkl")
        
        # Check if the pickle file exists
        if os.path.exists(pickle_path):
            print("Loading DataFrame from pickle:", pickle_path)
            self.merged_with_disch_df = pd.read_pickle(pickle_path)
            return

        try:
            filtered_ds = self.merged_with_disch_df['hadm_id'].isin(self.sorted_filtered_normalised_df['hadm_id'])
            self.merged_with_disch_df = self.merged_with_disch_df.loc[filtered_ds]
        except:
            print("No physio events found for any patients")
            return self.merged_with_disch_df

        # Ensure that the output directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        # Save the final DataFrame as a pickle for future use
        self.merged_with_disch_df.to_pickle(pickle_path)
        print("Processed DataFrame saved to:", pickle_path)
        return self.merged_with_disch_df


    def get_30_day_mortality_outcome(self, outcome_choice):
        if outcome_choice == '30d_mortality_discharge':
            hadm_disch_death = self.merged_with_disch_df.groupby('hadm_id').first()[['dischtime', 'dod']].copy()
            
            # Ensure both columns are proper datetime objects
            hadm_disch_death['dischtime'] = pd.to_datetime(hadm_disch_death['dischtime'])
            hadm_disch_death['dod'] = pd.to_datetime(hadm_disch_death['dod'])
            
            # Now calculate the difference in days
            hadm_disch_death['days_disch_to_death'] = (hadm_disch_death['dod'] - hadm_disch_death['dischtime']).dt.days

            # Step 2: Compute label: 1 if died within 30 days, 0 otherwise
            hadm_disch_death['label_death_within_30d'] = hadm_disch_death['days_disch_to_death'].apply(
                lambda x: 1 if pd.notnull(x) and x <= 30 else 0
            )
            return hadm_disch_death

       
    def get_hadm_ids(self):
        """Get a list of all hadm_ids that have discharge notes."""
        if self.merged_with_disch_df is None:
            raise ValueError("Merged dataframe not loaded. Call load_demographics() first.")
        
        # Get all hadm_ids from merged_with_disch_df
        all_hadm_ids = self.merged_with_disch_df['hadm_id'].unique().tolist()
        
        # Ensure these hadm_ids actually have corresponding discharge notes
        if hasattr(self, 'discharge_df') and self.discharge_df is not None:
            discharge_hadm_ids = set(self.discharge_df['hadm_id'].unique())
            valid_hadm_ids = [hadm_id for hadm_id in all_hadm_ids if hadm_id in discharge_hadm_ids]
            
            # Print warning if some hadm_ids don't have discharge notes
            missing = len(all_hadm_ids) - len(valid_hadm_ids)
            if missing > 0:
                print(f"Warning: {missing} out of {len(all_hadm_ids)} hadm_ids don't have discharge notes")
            
            return valid_hadm_ids
        
        return all_hadm_ids

    def split_by_subject_id(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
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
        if self.merged_with_disch_df is None:
            raise ValueError("Merged dataframe not loaded. Call load_demographics() first.")
            
        # Verify the ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1.0"
        
        # Get unique subject_ids
        unique_subjects = self.merged_with_disch_df['subject_id'].unique()
        
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
        train_hadm_ids = self.merged_with_disch_df[self.merged_with_disch_df['subject_id'].isin(train_subjects)]['hadm_id'].tolist()
        val_hadm_ids = self.merged_with_disch_df[self.merged_with_disch_df['subject_id'].isin(val_subjects)]['hadm_id'].tolist()
        test_hadm_ids = self.merged_with_disch_df[self.merged_with_disch_df['subject_id'].isin(test_subjects)]['hadm_id'].tolist()
        
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
        if split == 'train':
            return self.train_hadm_ids
        elif split == 'val':
            return self.val_hadm_ids
        elif split == 'test':
            return self.test_hadm_ids
        else:
            raise ValueError(f"Unknown split: {split}. Must be one of 'train', 'val', 'test'")



###############################################################################
# 2. LLMMapper: Aligns lab names and obtains physiologic categories via LLMs
###############################################################################

class LLMMapper:
    def __init__(self, run_LLM_func, cache_dir: str):
        """
        Initialize with an LLM runner function (e.g. run_LLM).
        """
        self.run_LLM = run_LLM_func
        self.cache_dir = cache_dir

    @staticmethod
    def get_align_labs_prompt(chartname_list, labname_big_list):
        prompt = f'''
        Your task is to map a set of new lab names (provided in list A) to their best matching names from a reference list (provided in list B), 
        which comes with associated itemids. For each new lab name, select the best match from the reference list.

        IMPORTANT:
        - Return only a valid JSON object without any additional text or markdown formatting.
        - Do NOT include triple backticks or any markdown markers.

        Output format:
        {{"new_name": "reference_name", "new_name2": "reference_name2", ...}}

        A) New lab names: 
        {chartname_list}

        B) Reference list: 
        {labname_big_list}
        '''
        return prompt

    def align_lab_names(self, chartevents_df, labevents_df, chart_name_chunk_size=10, iterations=3):
        """
        Uses LLM to map new lab names in chartevents (for 'Labs' category)
        to the reference lab names from labevents.
        
        If a pickle file already exists at 'temp_dfs/aligned_labs.pckle', it is loaded 
        and returned. Otherwise, the method computes the mapping using the LLM and 
        saves the resulting DataFrame as a pickle.
        """
        pickle_path = os.path.join(self.cache_dir, "aligned_labs.pckle")
        
        # If the pickle file exists, load and return the pre-computed result.
        if os.path.exists(pickle_path):
            print(f"Loading aligned labs from {pickle_path}...")
            with open(pickle_path, "rb") as f:
                return pickle.load(f)
        
        # Otherwise, perform the LLM-based mapping.
        new_names = list(chartevents_df[chartevents_df['category'] == 'Labs']['name'].unique())
        reference_name_to_itemid = labevents_df.drop_duplicates(subset='name').set_index('name')['itemid'].to_dict()
        
        system_prompt = (
            "You are a helpful matching assistant. Map each new lab name to the best matching reference name from the list provided. "
            "Return only valid JSON (no markdown, no triple backticks)."
        )
        matched_results = {}
        
        for i in range(0, len(new_names), chart_name_chunk_size):
            chunk = new_names[i:i + chart_name_chunk_size]
            input_prompt = self.get_align_labs_prompt(chunk, list(reference_name_to_itemid.keys()))
            raw_results = self.run_LLM(system_prompt, input_prompt, iterations, model="gpt-4o")
            raw_results_json = json.loads(raw_results)
            print('raw_results_json', raw_results_json)
            matched_results.update(raw_results_json)
        
        # Create a boolean mask for rows in the "Labs" category
        labs_mask = chartevents_df['category'] == 'Labs'
        
        # For the rows where category is "Labs", create a copy to work with
        labs_df = chartevents_df.loc[labs_mask].copy()
        
        # Map the current 'name' values to the matched reference names (if available)
        labs_df['ref_name'] = labs_df['name'].map(matched_results)
        
        # Replace the original name with the reference name where available.
        # If no mapping is found, keep the original name.
        labs_df['name'] = labs_df['ref_name'].combine_first(labs_df['name'])
        
        # Update the itemid: if a reference name was found, map it to the new itemid using reference_name_to_itemid.
        # Otherwise, keep the original itemid.
        labs_df['itemid'] = labs_df['ref_name'].map(reference_name_to_itemid).combine_first(labs_df['itemid'])
        
        # Update the original DataFrame with the new values for the 'Labs' rows
        chartevents_df.loc[labs_mask, ['name', 'itemid']] = labs_df[['name', 'itemid']]
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        
        # Save the modified DataFrame as a pickle file.
        with open(pickle_path, "wb") as f:
            pickle.dump(chartevents_df, f)
        
        return chartevents_df


    @staticmethod
    def get_physio_category_prompt(physio_names_chunk, previous_categories=None):
        prompt = f'''
        Looking at the following list of lab items, provide a category for each lab item. This is a single word based on organ function or basic physiology.
        For example: renal (electrolytes, etc), inflammatory (WBC, CRP, etc), clotting (platelets, PT, INR, etc), haematologic (RBCs, etc), endocrine, cardiovascular, hepatobiliary, respiratory, metabolic etc.
        Don't use 'general' as a category.

        Return only a valid JSON object without any additional text or markdown formatting.
        - Do NOT include triple backticks or any markdown markers.

        Names:
        {physio_names_chunk}

        Previous categories:
        {previous_categories}

        Output format:
        {{"name1": "category1", "name2": "category2", ...}}
        '''
        return prompt

    def get_physio_categories(self, chartevents_df, labevents_df,
                              chart_name_chunk_size=30, iterations=3):
        """
        Uses LLM to assign physiologic categories to lab items.
        Caches the results to a pickle file.
        """
        pickle_filename = os.path.join(self.cache_dir, "physio_categories.pkl")
        temp_dir = os.path.dirname(pickle_filename)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.path.exists(pickle_filename):
            with open(pickle_filename, "rb") as f:
                category_matched_results = pickle.load(f)
            print("Loaded cached category mapping from", pickle_filename)
            
            # Create categories DataFrame
            categories_df = pd.DataFrame(
                list(category_matched_results.items()),
                columns=['name', 'physio_category']
            )
            
            # Apply mapping to events DataFrames
            chartevents_df_copy = chartevents_df.copy()
            labevents_df_copy = labevents_df.copy()
            
            chartevents_df_copy['physio_category'] = chartevents_df_copy['name'].map(category_matched_results)
            labevents_df_copy['physio_category'] = labevents_df_copy['name'].map(category_matched_results)
            
            # Filter out 'other' category
            labevents_df_copy = labevents_df_copy[labevents_df_copy['physio_category'] != 'other']
            chartevents_df_copy = chartevents_df_copy[chartevents_df_copy['physio_category'] != 'other']
            categories_df = categories_df[categories_df['physio_category'] != 'other']
            
            return chartevents_df_copy, labevents_df_copy, categories_df

        all_physio_names = pd.concat([chartevents_df['name'], labevents_df['name']]).unique()
        all_physio_names = list(all_physio_names)
        print("Total unique lab items:", len(all_physio_names))
        category_matched_results = {}

        for i in range(0, len(all_physio_names), chart_name_chunk_size):
            chunk = all_physio_names[i:i + chart_name_chunk_size]
            previous_categories = sorted(list(set(category_matched_results.values())))
            input_prompt = self.get_physio_category_prompt(chunk, previous_categories=previous_categories)
            raw_results = self.run_LLM("You are a helpful matching assistant.", input_prompt, iterations, model="gpt-4o")
            try:
                raw_results_json = json.loads(raw_results)
                print(f"Response for chunk starting at index {i} ->", raw_results_json)
            except Exception as e:
                print(f"Error parsing JSON response for chunk starting at index {i}: {e}")
                continue
            category_matched_results.update(raw_results_json)
        
        with open(pickle_filename, "wb") as f:
            pickle.dump(category_matched_results, f)
        print("Saved category mapping to", pickle_filename)

        categories_df = pd.DataFrame(
            list(category_matched_results.items()),
            columns=['name', 'physio_category']
        )

        chartevents_df['physio_category'] = chartevents_df['name'].map(category_matched_results)
        labevents_df['physio_category'] = labevents_df['name'].map(category_matched_results)

        labevents_df = labevents_df[labevents_df['physio_category'] != 'other']
        chartevents_df = chartevents_df[chartevents_df['physio_category'] != 'other']
        categories_df = categories_df[categories_df['physio_category']!='other']
        
        return chartevents_df, labevents_df, categories_df



###############################################################################
# 3. MIMIC DISCHARGE NOTES: Loads and processes discharge notes
###############################################################################

class MIMICDischargeNotesProcessor:
    def __init__(self, hadm_ids, cache_dir: str, discharge_df=None):
        self.hadm_ids = hadm_ids
        self.cache_dir = cache_dir
        self.lmdb_path = os.path.join(self.cache_dir, "discharge_notes.lmdb")
        
        # Store the discharge_df directly
        self.discharge_df = discharge_df
        if self.discharge_df is not None:
            debug_print(f'Discharge df provided with {len(self.discharge_df)} rows')
            
            # Verify all hadm_ids have corresponding discharge notes
            if 'hadm_id' in self.discharge_df.columns:
                missing_hadm_ids = set(hadm_ids) - set(self.discharge_df['hadm_id'])
                if missing_hadm_ids:
                    print(f"WARNING: {len(missing_hadm_ids)} hadm_ids don't have matching discharge notes")
                    debug_print(f"First few missing hadm_ids: {list(missing_hadm_ids)[:5]}")
                
                # Set index for faster lookup
                if not self.discharge_df.index.name == 'hadm_id':
                    debug_print("Setting hadm_id as index for discharge_df")
                    self.discharge_df = self.discharge_df.set_index('hadm_id')
            else:
                print(f"WARNING: 'hadm_id' column not found in discharge_df. Columns: {self.discharge_df.columns.tolist()}")
        else:
            print('WARNING: No discharge_df provided to MIMICDischargeNotesProcessor')
            
            # Try to create a placeholder discharge_df with empty content for each hadm_id
            debug_print('Creating placeholder discharge_df with empty content')
            empty_data = {
                'hadm_id': hadm_ids,
                'charttime': [None] * len(hadm_ids),
                'text': ['No discharge note available'] * len(hadm_ids)
            }
            self.discharge_df = pd.DataFrame(empty_data).set_index('hadm_id')
        
        self._init_lmdb()
        self.sections_to_include = [
            "History of Present Illness",
            "Past Medical History",
            "Social History",
            "Family History",
            "Physical Exam",
            "Brief Hospital Course",
            "IMPRESSION",
            "DISCHARGE PHYSICAL EXAM",
            "ACUTE/ACTIVE ISSUES",
            "Discharge Diagnosis",
        ]

        self.sections_to_ignore = [
            "pertinent results", 
            "DISCHARGE LABS", 
            "Medications on Admission", 
            "Medications on Discharge", 
            "Discharge Medications", 
            "Discharge Disposition", 
            "Discharge Instructions"
        ]
        
        # Add memory cache to avoid LMDB reads when possible
        self.memory_cache = {}
        
        # Preload common discharge notes on initialization
        self.preload_common_discharge_notes()
    
    def preload_common_discharge_notes(self, max_preload=100):
        """Preload the first N discharge notes into memory to speed up repeated accesses"""
        debug_print(f"Preloading up to {max_preload} discharge notes into memory...")
        preload_hadm_ids = self.hadm_ids[:min(max_preload, len(self.hadm_ids))]
        
        self._check_env()
        try:
            with self.env.begin(write=False) as txn:
                preloaded_count = 0
                for hadm_id in preload_hadm_ids:
                    key = f"discharge_note_{hadm_id}".encode("utf-8")
                    data = txn.get(key)
                    
                    if data is not None:
                        # Cache in memory
                        self.memory_cache[hadm_id] = pickle.loads(data)
                        preloaded_count += 1
        except lmdb.Error as e:
            print(f"LMDB error during preload: {e}")
            # Try to recover by reopening environment
            self._open_env()
                
        debug_print(f"Preloaded {preloaded_count} discharge notes into memory")

    def _init_lmdb(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        # Create an environment with a generous map size (e.g., 10GB); adjust as needed.
        self.env = None
        self.pid = os.getpid()
        self._open_env()
    
    def _open_env(self):
        """Open the LMDB environment for this process"""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        
        # Use subdir=True to handle existing directory structure
        self.env = lmdb.open(self.lmdb_path, map_size=10**10, 
                           subdir=True, readonly=False, lock=True, 
                           readahead=False, meminit=False)
        self.pid = os.getpid()
    
    def _check_env(self):
        """Check if the environment needs to be reopened (e.g., after fork)"""
        if not hasattr(self, 'env') or self.env is None or self.pid != os.getpid():
            self._open_env()
    
    def load_discharge_note(self, hadm_id):
        """
        Load discharge note for a specific hospital admission from the already loaded DataFrame
        
        Args:
            hadm_id: Hospital admission ID to load note for
            
        Returns:
            DataFrame containing the discharge note
        """
        try:
            # If we have the discharge_df, use it directly 
            if self.discharge_df is not None:
                # Check if discharge_df is indexed by hadm_id
                if self.discharge_df.index.name == 'hadm_id':
                    # If indexed by hadm_id, use .loc directly
                    if hadm_id in self.discharge_df.index:
                        row = self.discharge_df.loc[hadm_id]
                        # Handle both Series and DataFrame results
                        if isinstance(row, pd.Series):
                            # Convert Series to DataFrame
                            discharge_note_df = pd.DataFrame([row])
                            discharge_note_df['hadm_id'] = hadm_id
                        else:
                            discharge_note_df = row.reset_index()
                        return discharge_note_df
                    else:
                        debug_print(f"Warning: hadm_id {hadm_id} not found in indexed discharge_df")
                else:
                    # If not indexed, filter by hadm_id column
                    if 'hadm_id' in self.discharge_df.columns:
                        discharge_note_df = self.discharge_df[self.discharge_df['hadm_id'] == hadm_id]
                        if not discharge_note_df.empty:
                            return discharge_note_df
                        debug_print(f"Warning: hadm_id {hadm_id} not found in non-indexed discharge_df")
                    else:
                        debug_print(f"Error: 'hadm_id' column not found in discharge_df. Columns: {self.discharge_df.columns.tolist()}")
            
            # Fallback: try to find paths where discharge notes might be stored
            possible_paths = [
                os.path.join(os.path.dirname(self.cache_dir), 'note', 'discharge.csv'),
                os.path.join(self.cache_dir, '..', 'note', 'discharge.csv'),
                '/home/ubuntu/mimic-iv/note/discharge.csv',   # Common server path
                '/Users/riccardoconci/Local_documents/!!MIMIC/note/discharge.csv'  # Local path
            ]
            
            for discharge_path in possible_paths:
                if os.path.exists(discharge_path):
                    debug_print(f"Loading discharge notes from CSV file: {discharge_path}")
                    discharge_notes_df = pd.read_csv(discharge_path)
                    discharge_note_df = discharge_notes_df[discharge_notes_df['hadm_id'] == hadm_id]
                    if not discharge_note_df.empty:
                        return discharge_note_df
            
            # If we reach here, we couldn't find the discharge note
            debug_print(f"Warning: No discharge note found for hadm_id {hadm_id}")
            # Return a placeholder with empty text for this hadm_id
            return pd.DataFrame({
                'hadm_id': [hadm_id],
                'charttime': [None],
                'text': ['No discharge note available']
            })
                
        except Exception as e:
            print(f"Error loading discharge note for hadm_id {hadm_id}: {e}")
            if DEBUG_PRINT:
                import traceback
                traceback.print_exc()
            # Return a placeholder with empty text
            return pd.DataFrame({
                'hadm_id': [hadm_id],
                'charttime': [None],
                'text': ['Error loading discharge note']
            })

    def get_discharge_chunks(self, hadm_id, bypass_lmdb=False):
        """
        Get discharge note chunks for a specific hospital admission
        
        Args:
            hadm_id: Hospital admission ID
            bypass_lmdb: If True, don't access LMDB and return an empty list if not in memory cache
            
        Returns:
            List of text chunks from the discharge note
        """
        try:
            # First check memory cache for fastest access
            if hadm_id in self.memory_cache:
                return self.memory_cache[hadm_id]
            
            # If bypass_lmdb is True and the discharge chunks are not in memory cache,
            # return an empty list instead of accessing LMDB
            if bypass_lmdb:
                return []
            
            # Next try to get chunks from LMDB cache
            self._check_env()
            try:
                with self.env.begin(write=False) as txn:
                    key = f"discharge_note_{hadm_id}".encode("utf-8")
                    data = txn.get(key)
                    
                    if data is not None:
                        # Found in cache, store in memory cache and return the chunks
                        chunks = pickle.loads(data)
                        self.memory_cache[hadm_id] = chunks
                        return chunks
            except lmdb.Error as e:
                print(f"LMDB error getting discharge chunks for hadm_id {hadm_id}: {e}")
                # Try to recover by reopening environment
                self._open_env()
                    
            # Not found in cache, load and process
            discharge_note_df = self.load_discharge_note(hadm_id)
            chunks = self.process_discharge_note(discharge_note_df)
            
            # Store in memory cache for future use
            self.memory_cache[hadm_id] = chunks
            return chunks
                
        except Exception as e:
            print(f"Error getting discharge chunks for hadm_id {hadm_id}: {e}")
            # Return an empty list as fallback
            return []
    
    def process_discharge_note(self, discharge_note_df):
        """
        Process a discharge note: extract sections, chunk them, and save to LMDB
        
        Args:
            discharge_note_df: DataFrame containing the discharge note
            
        Returns:
            List of text chunks from the discharge note
        """
        if discharge_note_df.empty:
            return []  # Return empty list if no discharge note
            
        # Get the text and hadm_id from the dataframe    
        note = discharge_note_df['text'].iloc[0] if 'text' in discharge_note_df.columns else discharge_note_df['value'].iloc[0]
        hadm_id = discharge_note_df['hadm_id'].iloc[0]
        
        # Parse and extract relevant sections
        selected_blocks = self.parse_included_sections(note,
                                                     self.sections_to_include,
                                                     self.sections_to_ignore)
        
        # Split text into chunks
        chunks = self.chunkify_blocks_nltk(selected_blocks)
        
        # Save to LMDB for future use
        self._check_env()
        try:
            with self.env.begin(write=True) as txn:
                key = f"discharge_note_{hadm_id}".encode("utf-8")
                txn.put(key, pickle.dumps(chunks))
        except lmdb.Error as e:
            print(f"LMDB error saving discharge note for hadm_id {hadm_id}: {e}")
            # Try to recover by reopening environment
            self._open_env()
            try:
                with self.env.begin(write=True) as txn:
                    key = f"discharge_note_{hadm_id}".encode("utf-8")
                    txn.put(key, pickle.dumps(chunks))
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
        
        return chunks
    
    def parse_included_sections(self, text, included_headings, excluded_headings, case_insensitive=True):
        """
        Scans the entire text in order of headings. At each heading:
        - If heading is in included_headings, switch 'selecting' to True.
        - If heading is in excluded_headings, switch 'selecting' to False.
        - If heading is in neither, do nothing to 'selecting'.
        While selecting=True, we collect text from the current heading until the next heading.

        Returns:
        A list of dicts, each with:
            - 'start_offset':  Start char index of this included block
            - 'end_offset':    End char index of this included block
            - 'heading':       The heading text that triggered selecting
            - 'content':       The substring from [start_offset : end_offset]
        
        Example:
        included_headings = ["History of Present Illness", "Past Medical History"]
        excluded_headings = ["Medications on Admission", "Discharge Instructions"]
        parse_included_sections(text, included_headings, excluded_headings)
        """
        all_headings = list(set(included_headings + excluded_headings))
        
        # Build the pattern
        pattern_str = "(" + "|".join(re.escape(h) for h in all_headings) + "):"
        
        # Compile case-insensitively if requested
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern_str, flags=flags)
        
        # 2. Find all occurrences of these headings
        matches = list(regex.finditer(text))
        # We'll add a "sentinel" at the end (len(text)) so we know where the last heading's block ends
        # This sentinel does not represent a real heading, but it ensures we capture trailing text
        # if we happen to be in selecting mode at the end.
        sentinel = type("SentinelMatch", (), {})()  # a mock object
        setattr(sentinel, "start", lambda: len(text))
        matches.append(sentinel)
        
        # Prepare sets (in lower case) for quick membership tests
        included_set = set(h.lower() for h in included_headings)
        excluded_set = set(h.lower() for h in excluded_headings)
        
        selected_blocks = []
        selecting = False
        current_heading_text = None
        
        # 3. Walk through each heading in order
        for i in range(len(matches) - 1):
            this_match = matches[i]
            next_match = matches[i+1]
            
            this_heading_start = this_match.start()
            # The next heading starts at next_match.start() (or end of text if sentinel)
            next_heading_start = next_match.start()
            
            # We can figure out exactly which heading was matched from group(0) if we want:
            # But group(1) is the parenthesized text (the heading itself, minus the colon).
            heading_matched = this_match.group(1) if hasattr(this_match, "group") else None
            
            # Compare heading in lowercase to included/excluded sets
            if heading_matched:
                heading_lower = heading_matched.lower()
                if heading_lower in included_set:
                    selecting = True
                    current_heading_text = heading_matched.strip()
                elif heading_lower in excluded_set:
                    selecting = False
                    current_heading_text = None
                else:
                    # Not in included or excluded => do nothing, keep same selecting state
                    pass
            
            # 4. If we are currently selecting, we capture text from
            #    this heading's start offset up to next heading's start offset.
            if selecting and heading_matched:
                block_start = this_heading_start
                block_end = next_heading_start
                
                content = text[block_start:block_end]
                
                selected_blocks.append({
                    "start_offset": block_start,
                    "end_offset": block_end,
                    "heading": current_heading_text,
                    "content": content
                })
        
        return selected_blocks
    
    def chunkify_blocks_nltk(self, blocks, words_per_chunk=150):
        """
        Combines the content from all blocks into a single list of words using NLTK's word_tokenize,
        and then splits the words into dense chunks each containing at most `words_per_chunk` words.

        Args:
            blocks (list of dict): Each dict must have a 'content' key with text.
            words_per_chunk (int): Maximum number of words per chunk (default: 150).

        Returns:
            list of str: A list of text chunks, where each chunk has no more than `words_per_chunk` words.
        """
        all_words = []
        for block in blocks:
            words = word_tokenize(block['content'])
            all_words.extend(words)
        
        total_words = len(all_words)
        chunks = []
        # This loop produces chunks that are exactly words_per_chunk, except possibly the final chunk
        for i in range(0, total_words, words_per_chunk):
            chunk = " ".join(all_words[i: i + words_per_chunk])
            chunks.append(chunk)
        return chunks
    
    def highlight_sections(self, text, selected_blocks):
        """
        Takes the full text and a list of 'selected_blocks' (each with start_offset/end_offset/content).
        Returns an HTML string that highlights ONLY those blocks in yellow. 
        The rest of the text remains unhighlighted.
        """
        # Sort by start_offset
        blocks_sorted = sorted(selected_blocks, key=lambda b: b["start_offset"])
        
        html_parts = []
        last_pos = 0
        
        for block in blocks_sorted:
            start = block["start_offset"]
            end = block["end_offset"]
            
            # Everything from last_pos to start is normal (unhighlighted)
            html_parts.append(text[last_pos:start])
            
            # The selected block is highlighted
            html_parts.append(
                '<mark style="background-color: yellow;">'
                + text[start:end]
                + '</mark>'
            )
            
            last_pos = end
        
        # Append the remainder
        html_parts.append(text[last_pos:])
        
        html_output = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            "  <meta charset='UTF-8'>\n  <title>Highlighted Note</title>\n</head>\n"
            "<body>\n<pre style='font-family: monospace;'>\n"
            + "".join(html_parts)
            + "\n</pre>\n</body>\n</html>"
        )
        return html_output



###############################################################################
# 3. MIMICClinicalEventsProcessor: Loads and processes physiologic and treatment events
###############################################################################

class MIMICClinicalEventsProcessor:
    def __init__(self, base_path: str, hadm_ids, cache_dir: str):
        """
        Initialize file paths, the hadm_ids, and the column selection for each source.
        Processes the clinical events (physiologic data and treatments) for modeling.
        """
        self.base_path = base_path
        self.hadm_ids = hadm_ids
        self.cache_dir = cache_dir
        self.LLM_mapper = LLMMapper(run_LLM, cache_dir)
        
        # File paths for event data
        self.chartevents_path  = os.path.join(self.base_path, 'icu', 'chartevents.csv')
        self.outputevents_path = os.path.join(self.base_path, 'icu', 'outputevents.csv')
        self.labevents_path    = os.path.join(self.base_path, 'hosp', 'labevents.csv')
        self.micro_events_path = os.path.join(self.base_path, 'hosp', 'microbiologyevents.csv')
        self.inputevents_path  = os.path.join(self.base_path, 'icu', 'inputevents.csv')
        self.emar_path         = os.path.join(self.base_path, 'hosp', 'emar.csv')
        self.ed_vitals_path    = os.path.join(self.base_path, 'ed', 'vitalsign.csv')
        self.ed_stays_path     = os.path.join(self.base_path, 'ed', 'edstays.csv')
        self.ed_pyxis_path     = os.path.join(self.base_path, 'ed', 'pyxis.csv')
        self.ed_medrecon_path  = os.path.join(self.base_path, 'ed', 'medrecon.csv')
        
        # Dictionary file paths
        self.d_chartitems_path = os.path.join(self.base_path, 'icu', 'd_items.csv')
        self.d_labitems_path   = os.path.join(self.base_path, 'hosp', 'd_labitems.csv')
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        self.physio_cols_to_keep = {
            'chartevents': ['hadm_id', 'itemid', 'charttime', 'valuenum'],
            'outputevents': ['hadm_id', 'itemid', 'charttime', 'value'],
            'labevents': ['hadm_id', 'itemid', 'charttime', 'valuenum'],
            'micro_events': ['hadm_id', 'micro_specimen_id', 'charttime', 'test_name', 'comments']
        }
        self.treatment_cols_to_keep = {
            'inputevents': ['hadm_id', 'itemid', 'starttime', 'endtime', 'amount', 'amountuom', 'patientweight'],
            'emar': ['hadm_id', 'emar_id', 'charttime', 'medication', 'event_txt']
        }
        self.ds_cols_to_keep = {
            'disch_notes': ['hadm_id', 'text']
        }
        
    @staticmethod
    def load_and_process_csv_for_hadm_ids(file_path, hadm_ids, chunk_size=1000000,
                                            usecols=None, source=None, cache_dir='temp_dfs'):
        """
        Loads a CSV file in chunks, filtering for rows with hadm_id in hadm_ids.
        Renames columns based on the source. Drops rows with missing 'value'
        and converts 'charttime' to datetime.
        
        Uses a pickle cache in the provided cache_dir folder.
        """
        temp_dir = cache_dir
        os.makedirs(temp_dir, exist_ok=True)
        base_name = os.path.basename(file_path).replace('.csv', '')
        pickle_file = os.path.join(temp_dir, f"{base_name}_{source}.pkl")
        
        if os.path.exists(pickle_file):
            print(f"Loading existing pickle file: {pickle_file}")
            # Use joblib with memory mapping for faster loading and lower memory usage
            try:
                # Try to load with memory mapping (much faster for large files)
                return joblib.load(pickle_file, mmap_mode='r')
            except Exception as e:
                print(f"Warning: Could not load with joblib ({e}), falling back to pandas")
                return pd.read_pickle(pickle_file)
        
        chunks = []
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, usecols=usecols)):
            filtered_chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
            
            # Rename columns based on source
            if source == 'chartevents':
                filtered_chunk = filtered_chunk.rename(columns={'valuenum': 'value'})
            elif source == 'inputevents':
                filtered_chunk = filtered_chunk.rename(columns={'amount': 'value', 'starttime': 'charttime'})
            elif source == 'labevents':
                filtered_chunk = filtered_chunk.rename(columns={'valuenum': 'value'})
            elif source == 'emar':
                filtered_chunk = filtered_chunk.rename(columns={'medication': 'name',
                                                                 'event_txt': 'value',
                                                                 'emar_id': 'itemid'})
            elif source == 'micro_events':
                filtered_chunk = filtered_chunk.rename(columns={'micro_specimen_id': 'itemid',
                                                                 'test_name': 'name',
                                                                 'comments': 'value'})
            filtered_chunk = filtered_chunk.dropna(subset=['value'])
            
            if 'charttime' in filtered_chunk.columns:
                filtered_chunk['charttime'] = pd.to_datetime(filtered_chunk['charttime'], errors='coerce')
            chunks.append(filtered_chunk)
            print(f"Chunk {i} processed, shape after filter: {filtered_chunk.shape}")
        
        result = pd.concat(chunks, ignore_index=True)
        print(f"Saving result to pickle file: {pickle_file}")
        
        # Save with joblib instead of pickle for better performance with large dataframes
        try:
            joblib.dump(result, pickle_file, compress=3)  # Use compression level 3 (balance of speed/size)
            print("Saved using joblib compression")
        except Exception as e:
            print(f"Warning: Could not save with joblib ({e}), falling back to pandas")
            result.to_pickle(pickle_file)
            
        return result
    
    def load_all_events(self, load_only_essential=True):
        """
        Loads all physiologic and treatment event data.
        
        Args:
            load_only_essential: If True, only loads chartevents and labevents,
                                which are essential for the KEDGN model
        """
        # Physiologic events
        print("Loading chartevents...")
        self.chartevents_filtered = self.load_and_process_csv_for_hadm_ids(
            self.chartevents_path, self.hadm_ids,
            usecols=self.physio_cols_to_keep['chartevents'], source='chartevents',
            cache_dir=self.cache_dir
        )
        
        print("Loading labevents...")
        self.labevents_filtered = self.load_and_process_csv_for_hadm_ids(
            self.labevents_path, self.hadm_ids,
            usecols=self.physio_cols_to_keep['labevents'], source='labevents',
            cache_dir=self.cache_dir
        )
        
        # If only loading essential data, use empty DataFrames for the rest
        if load_only_essential:
            print("Using empty dataframes for non-essential events")
        
            
            # Create empty DataFrames with the correct columns
            self.outputevents_filtered = pd.DataFrame(columns=self.physio_cols_to_keep['outputevents'] + ['source'])
            self.micro_events_filtered = pd.DataFrame(columns=self.physio_cols_to_keep['micro_events'] + ['source'])
            self.inputevents_filtered = pd.DataFrame(columns=self.treatment_cols_to_keep['inputevents'] + ['source'])
            self.emar_filtered = pd.DataFrame(columns=self.treatment_cols_to_keep['emar'] + ['source'])
            
            # Set source column
            if not self.outputevents_filtered.empty:
                self.outputevents_filtered['source'] = 'outputevents'
            if not self.micro_events_filtered.empty:
                self.micro_events_filtered['source'] = 'micro_events'
            if not self.inputevents_filtered.empty:
                self.inputevents_filtered['source'] = 'inputevents'
            if not self.emar_filtered.empty:
                self.emar_filtered['source'] = 'emar'
                
        else:
            # Load all events (original behavior)
            print("Loading outputevents...")
            self.outputevents_filtered = self.load_and_process_csv_for_hadm_ids(
                self.outputevents_path, self.hadm_ids,
                usecols=self.physio_cols_to_keep['outputevents'], source='outputevents',
                cache_dir=self.cache_dir
            )
            
            print("Loading microbiologyevents...")
            self.micro_events_filtered = self.load_and_process_csv_for_hadm_ids(
                self.micro_events_path, self.hadm_ids,
                usecols=self.physio_cols_to_keep['micro_events'], source='micro_events',
                cache_dir=self.cache_dir
            )
            
            # Treatment events
            print("Loading inputevents...")
            self.inputevents_filtered = self.load_and_process_csv_for_hadm_ids(
                self.inputevents_path, self.hadm_ids,
                usecols=self.treatment_cols_to_keep['inputevents'], source='inputevents',
                cache_dir=self.cache_dir
            )
            
            print("Loading emar events...")
            self.emar_filtered = self.load_and_process_csv_for_hadm_ids(
                self.emar_path, self.hadm_ids,
                usecols=self.treatment_cols_to_keep['emar'], source='emar',
                cache_dir=self.cache_dir
            )
            # For emar, further filter on administered events
            self.emar_filtered = self.emar_filtered[self.emar_filtered['value'].isin(['Administered'])]


        ## for ED RESULTS
        # ed_stays_df = pd.read_csv(self.ed_stays_path)
        # ed_vitals_df = pd.read_csv(self.ed_vitals_path)
        # ed_vitals_merged_df = pd.merge(ed_vitals_df, ed_stays_df, on=['stay_id', 'subject_id'], how='left')[['hadm_id', 'charttime', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp']]
        # ed_vitals_long_df = ed_vitals_merged_df.melt(
        #     id_vars=['hadm_id', 'charttime'],
        #     value_vars=['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp'],
        #     var_name='name',
        #     value_name='value'
        # )
        # self.ed_vitals_long_df = ed_vitals_long_df[['hadm_id', 'name', 'charttime', 'value']]
        # self.ed_vitals_long_df = self.ed_vitals_long_df.dropna()

        # ed_pyxis_df = pd.read_csv(self.ed_pyxis_path)
        # ed_medrecon_df = pd.read_csv(self.ed_medrecon_path)
        # ed_stays_df = pd.read_csv(self.ed_stays_path)

        # ed_pyxis_merged_df = pd.merge(ed_pyxis_df, ed_stays_df, on=['stay_id', 'subject_id'], how='left')[['hadm_id', 'charttime', 'name']]
        # self.ed_medrecon_merged_df = pd.merge(ed_medrecon_df, ed_stays_df, on=['stay_id', 'subject_id'], how='left')[['hadm_id', 'charttime','name', 'gsn']]
        # self.ed_medrecon_merged_df.rename(columns={'gsn': 'itemid'}, inplace=True)


    def process_events(self):   

        self.merge_item_names()
        self.align_lab_names_LLM()
        self.categorise_labs_LLM()
        self.set_hadm_id_indexes()
        self.create_cluster_labels()

    def provide_physio_var_names(self):
        return list(self.categories_df['name'].unique())

    def create_cluster_labels(self):
        self.categories_df["cluster_id"], uniques = pd.factorize(self.categories_df["physio_category"])
        print("\nData with cluster IDs:\n", self.categories_df)
        print("\nUnique clusters:", uniques)
        self.cluster_labels = torch.tensor(self.categories_df["cluster_id"].values, dtype=torch.long)

    def merge_item_names(self):
        """
        Merges the event data (e.g. chartevents and labevents) with dictionary files
        to add item names and other attributes.
        """
        self.d_chartitems_df = pd.read_csv(self.d_chartitems_path)
        self.d_labitems_df   = pd.read_csv(self.d_labitems_path)
        
        # Merge for chartevents using the chart items dictionary
        self.chartevents_filtered = pd.merge(
            self.chartevents_filtered,
            self.d_chartitems_df[['itemid', 'label', 'category']],
            on='itemid', how='left'
        )
        self.chartevents_filtered.rename(columns={'label': 'name'}, inplace=True)
        self.chartevents_filtered = self.chartevents_filtered[
            self.chartevents_filtered['category'].isin(
                ['Routine Vital Signs', 'Respiratory', 'Labs', 'Neurological', 'Hemodynamics']
            )
        ]
        
        # Merge for labevents using the lab items dictionary
        self.labevents_filtered = pd.merge(
            self.labevents_filtered,
            self.d_labitems_df[['itemid', 'label', 'category']],
            on='itemid', how='left'
        )
        self.labevents_filtered.rename(columns={'label': 'name'}, inplace=True)    

    def align_lab_names_LLM(self):
        """
        Maps the lab names using the LLMMapper.
        """
        self.chartevents_filtered = self.LLM_mapper.align_lab_names(self.chartevents_filtered, self.labevents_filtered)

    def categorise_labs_LLM(self):
        self.chartevents_filtered, self.labevents_filtered, self.categories_df = self.LLM_mapper.get_physio_categories(self.chartevents_filtered, self.labevents_filtered)
        
    def set_hadm_id_indexes(self):
        self.chartevents_filtered = self.chartevents_filtered.set_index('hadm_id')
        self.labevents_filtered = self.labevents_filtered.set_index('hadm_id')
        #self.outputevents_filtered = self.outputevents_filtered.set_index('hadm_id')
        #self.ed_vitals_long_df = self.ed_vitals_long_df.set_index('hadm_id')
        #self.micro_events_filtered = self.micro_events_filtered.set_index('hadm_id')

        #treatment events
        self.inputevents_filtered = self.inputevents_filtered.set_index('hadm_id')
        self.emar_filtered = self.emar_filtered.set_index('hadm_id')
        #self.ed_pyxis_merged_df = self.ed_pyxis_merged_df.set_index('hadm_id')

        #baseline events
        #self.ed_medrecon_merged_df = self.ed_medrecon_merged_df.set_index('hadm_id')

        #ds
        if hasattr(self, 'discharge_df') and self.discharge_df is not None:
            self.discharge_df.rename(columns={'text': 'value'}, inplace=True)
            self.discharge_df = self.discharge_df.set_index('hadm_id')

        self.labevents_filtered = self.normalize_itemid(self.labevents_filtered, value_col='value', group_col='itemid')
        self.chartevents_filtered = self.normalize_itemid(self.chartevents_filtered, value_col='value', group_col='itemid')
    
    def normalize_itemid(self, df, value_col='value', group_col='itemid'):
        """
        Normalizes the values for each itemid in the DataFrame based on their own mean and std.
        Any standardized values greater than 3 or less than -3 (i.e. outliers) are clipped to ±3.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the lab data.
            value_col (str): The name of the column containing the values to normalize.
            group_col (str): The column name by which to group the data (each unique itemid).

        Returns:
            pd.DataFrame: A new DataFrame with the normalized and clipped values.
        """
        # Compute the mean and std for each group (each itemid)
        means = df.groupby(group_col)[value_col].transform('mean')
        stds = df.groupby(group_col)[value_col].transform('std')
        
        # Standardize the values: (value - mean) / std
        normalized = (df[value_col] - means) / stds
        
        # Clip the standardized values to the range [-3, 3]
        normalized_clipped = normalized.clip(lower=-3, upper=3)
        
        # Create a copy of the DataFrame with the normalized values
        df_normalized = df.copy()
        df_normalized[value_col] = normalized_clipped
        
        return df_normalized
    




###############################################################################
# 4. MIMICContrastivePairsDataset: Dataset for contrastive learning
###############################################################################

class MIMICEventGroupsCache:
    def __init__(self, cache_dir: str, map_size=int(1e10)):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, "events_cache.lmdb")
        self.map_size = map_size
        self.pid = os.getpid()
        self.env = None
        self._open_env()
    
    def _open_env(self):
        """Open the LMDB environment for this process"""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        
        # Let LMDB use default locking settings (i.e., lock=True)
        self.env = lmdb.open(
            self.cache_path, 
            map_size=self.map_size, 
            subdir=True, 
            readonly=False,
            readahead=False, 
            meminit=False
        )
        self.pid = os.getpid()
    
    def _check_env(self):
        """Check if the environment needs to be reopened (e.g., after fork)"""
        if self.env is None or self.pid != os.getpid():
            self._open_env()
    
    def save(self, hadm_id, event_tuple):
        """Save event data for a hospital admission"""
        self._check_env()
        try:
            with self.env.begin(write=True) as txn:
                key = f"hadm_{hadm_id}".encode("utf-8")
                txn.put(key, pickle.dumps(event_tuple))
        except lmdb.Error as e:
            print(f"LMDB error saving hadm_{hadm_id}: {e}")
            self._open_env()
            try:
                with self.env.begin(write=True) as txn:
                    key = f"hadm_{hadm_id}".encode("utf-8")
                    txn.put(key, pickle.dumps(event_tuple))
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
    
    def load(self, hadm_id):
        """Load event data for a hospital admission"""
        self._check_env()
        try:
            with self.env.begin(write=False) as txn:
                key = f"hadm_{hadm_id}".encode("utf-8")
                data = txn.get(key)
                if data is not None:
                    return pickle.loads(data)
        except lmdb.Error as e:
            print(f"LMDB error loading hadm_{hadm_id}: {e}")
            self._open_env()
            try:
                with self.env.begin(write=False) as txn:
                    key = f"hadm_{hadm_id}".encode("utf-8")
                    data = txn.get(key)
                    if data is not None:
                        return pickle.loads(data)
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}")
        return None
    
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




