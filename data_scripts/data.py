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
nltk.download('punkt')

import lmdb



###############################################################################
# 1. MIMICDemographicsLoader: Loads demographics and basic merged data
###############################################################################

class MIMICDemographicsLoader:
    def __init__(self, base_path: str, cache_dir: str):
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


    def initialize_paths(self):
        # Basic demographics and note files
        self.patients_path   = os.path.join(self.base_path, 'hosp', 'patients.csv')
        self.admission_path  = os.path.join(self.base_path, 'hosp', 'admissions.csv')
        self.transfers_path  = os.path.join(self.base_path, 'hosp', 'transfers.csv')
        self.disch_path      = os.path.join(self.base_path, 'note', 'discharge.csv')
        
    def load_demographics(self):
        """
        Loads patients, admissions, transfers, and discharge notes,
        merges them, computes 'age' and admission duration, and filters 
        based on available discharge notes.

        If the processed DataFrame already exists as a pickle file in the 
        'tetm_dfs' directory, it is loaded directly. Otherwise, the CSV files 
        are processed and the result is saved for faster future loads.
        """
        # Define the pickle file path
        pickle_path = os.path.join(self.cache_dir, "demographics.pkl")
        
        # Check if the pickle file exists
        if os.path.exists(pickle_path):
            print("Loading DataFrame from pickle:", pickle_path)
            self.merged_with_disch_df = pd.read_pickle(pickle_path)
            return
        
        # Otherwise, load and merge the CSVs
        print("Pickle not found. Processing CSV files...")
        self.patients_df  = pd.read_csv(self.patients_path)
        self.admissions_df = pd.read_csv(self.admission_path)
        self.transfers_df = pd.read_csv(self.transfers_path)
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
        print('Keeping only admissions with admission duration < 30 days')
        self.merged_with_disch_df = self.merged_with_disch_df[self.merged_with_disch_df['admit_duration'] < 30]


        self.merged_with_disch_df['dod'] = pd.to_datetime(self.merged_with_disch_df['dod'])
        self.merged_with_disch_df['days_disch_to_death'] = (self.merged_with_disch_df['dod'] - self.merged_with_disch_df['dischtime']).dt.total_seconds() / (3600*24)


        # Ensure that the output directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        # Save the final DataFrame as a pickle for future use
        self.merged_with_disch_df.to_pickle(pickle_path)
        print("Processed DataFrame saved to:", pickle_path)

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

        if outcome_choice == '48h_mortality':
            pass

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
        
        # Store the discharge_df directly instead of the path
        self.discharge_df = discharge_df
        if self.discharge_df is not None:
            print(f'Discharge df provided with {len(self.discharge_df)} rows')
            
            # Verify all hadm_ids have corresponding discharge notes
            missing_hadm_ids = set(hadm_ids) - set(self.discharge_df['hadm_id'])
            if missing_hadm_ids:
                print(f"WARNING: {len(missing_hadm_ids)} hadm_ids don't have matching discharge notes")
                print(f"First few missing hadm_ids: {list(missing_hadm_ids)[:5]}")
                
                # Set index for faster lookup
                if 'hadm_id' in self.discharge_df.columns:
                    self.discharge_df = self.discharge_df.set_index('hadm_id')
        else:
            print('Discharge df not provided')
        
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

    def _init_lmdb(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        # Create an environment with a generous map size (e.g., 10GB); adjust as needed.
        self.env = lmdb.open(self.lmdb_path, map_size=10**10, readonly=False)
    
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
                # First check if discharge_df is indexed by hadm_id
                if isinstance(self.discharge_df.index, pd.Index) and self.discharge_df.index.name == 'hadm_id':
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
                        print(f"Warning: hadm_id {hadm_id} not found in indexed discharge_df")
                else:
                    # If not indexed, filter by hadm_id column
                    if 'hadm_id' in self.discharge_df.columns:
                        discharge_note_df = self.discharge_df[self.discharge_df['hadm_id'] == hadm_id]
                        if not discharge_note_df.empty:
                            return discharge_note_df
                        print(f"Warning: hadm_id {hadm_id} not found in non-indexed discharge_df")
                    else:
                        print(f"Error: 'hadm_id' column not found in discharge_df. Columns: {self.discharge_df.columns.tolist()}")
            
            # Fallback: load from CSV if needed
            discharge_note_path = os.path.join(os.path.dirname(self.cache_dir), 'note', 'discharge.csv') 
            if os.path.exists(discharge_note_path):
                print(f"Loading discharge notes from CSV file: {discharge_note_path}")
                discharge_notes_df = pd.read_csv(discharge_note_path)
                discharge_note_df = discharge_notes_df[discharge_notes_df['hadm_id'] == hadm_id]
                if not discharge_note_df.empty:
                    return discharge_note_df
                
            # If we reach here, we couldn't find the discharge note
            print(f"Warning: No discharge note found for hadm_id {hadm_id}")
            # Return a placeholder with empty text for this hadm_id
            return pd.DataFrame({
                'hadm_id': [hadm_id],
                'charttime': [None],
                'text': ['No discharge note available']
            })
                
        except Exception as e:
            print(f"Error loading discharge note for hadm_id {hadm_id}: {e}")
            import traceback
            traceback.print_exc()
            # Return a placeholder with empty text
            return pd.DataFrame({
                'hadm_id': [hadm_id],
                'charttime': [None],
                'text': ['Error loading discharge note']
            })

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
        note = note = discharge_note_df['text'].iloc[0] if 'text' in discharge_note_df.columns else discharge_note_df['value'].iloc[0]
        hadm_id = discharge_note_df['hadm_id'].iloc[0]
        
        # Parse and extract relevant sections
        selected_blocks = self.parse_included_sections(note,
                                                     self.sections_to_include,
                                                     self.sections_to_ignore)
        
        # Split text into chunks
        chunks = self.chunkify_blocks_nltk(selected_blocks)
        
        # Save to LMDB for future use
        with self.env.begin(write=True) as txn:
            key = f"discharge_note_{hadm_id}".encode("utf-8")
            txn.put(key, pickle.dumps(chunks))
        
        return chunks
    
    def get_discharge_chunks(self, hadm_id):
        """
        Get discharge note chunks for a specific hospital admission
        
        Args:
            hadm_id: Hospital admission ID
            
        Returns:
            List of text chunks from the discharge note
        """
        try:
            # Try to get chunks from LMDB cache first
            with self.env.begin(write=False) as txn:
                key = f"discharge_note_{hadm_id}".encode("utf-8")
                data = txn.get(key)
                
                if data is not None:
                    # Found in cache, return the chunks
                    return pickle.loads(data)
                    
                # Not found in cache, load and process
                discharge_note_df = self.load_discharge_note(hadm_id)
                chunks = self.process_discharge_note(discharge_note_df)
                return chunks
                
        except Exception as e:
            print(f"Error getting discharge chunks for hadm_id {hadm_id}: {e}")
            # Return an empty list as fallback
            return []
    



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
        
        # Temporary directory for cached pickles
        self.temp_dir = 'temp_dfs'
        os.makedirs(self.temp_dir, exist_ok=True)

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
                                            usecols=None, source=None):
        """
        Loads a CSV file in chunks, filtering for rows with hadm_id in hadm_ids.
        Renames columns based on the source. Drops rows with missing 'value'
        and converts 'charttime' to datetime.
        
        Uses a pickle cache in the `temp_dfs` folder.
        """
        temp_dir = 'temp_dfs'
        os.makedirs(temp_dir, exist_ok=True)
        base_name = os.path.basename(file_path).replace('.csv', '')
        pickle_file = os.path.join(temp_dir, f"{base_name}_{source}.pkl")
        
        if os.path.exists(pickle_file):
            print(f"Loading existing pickle file: {pickle_file}")
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
            usecols=self.physio_cols_to_keep['chartevents'], source='chartevents'
        )
        
        print("Loading labevents...")
        self.labevents_filtered = self.load_and_process_csv_for_hadm_ids(
            self.labevents_path, self.hadm_ids,
            usecols=self.physio_cols_to_keep['labevents'], source='labevents'
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
                usecols=self.physio_cols_to_keep['outputevents'], source='outputevents'
            )
            
            print("Loading microbiologyevents...")
            self.micro_events_filtered = self.load_and_process_csv_for_hadm_ids(
                self.micro_events_path, self.hadm_ids,
                usecols=self.physio_cols_to_keep['micro_events'], source='micro_events'
            )
            
            # Treatment events
            print("Loading inputevents...")
            self.inputevents_filtered = self.load_and_process_csv_for_hadm_ids(
                self.inputevents_path, self.hadm_ids,
                usecols=self.treatment_cols_to_keep['inputevents'], source='inputevents'
            )
            
            print("Loading emar events...")
            self.emar_filtered = self.load_and_process_csv_for_hadm_ids(
                self.emar_path, self.hadm_ids,
                usecols=self.treatment_cols_to_keep['emar'], source='emar'
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
        self.env = lmdb.open(os.path.join(cache_dir, "events_cache.lmdb"), map_size=map_size)
    
    def save(self, hadm_id, event_tuple):
        with self.env.begin(write=True) as txn:
            key = f"hadm_{hadm_id}".encode("utf-8")
            txn.put(key, pickle.dumps(event_tuple))
    
    def load(self, hadm_id):
        with self.env.begin(write=False) as txn:
            key = f"hadm_{hadm_id}".encode("utf-8")
            data = txn.get(key)
            if data is not None:
                return pickle.loads(data)
        return None




class MIMICContrastivePairsDataset(Dataset):
    def __init__(self, event_processor, 
                 disch_notes_processor, 
                 splits_loader=None, 
                 split='train', 
                 itemid_list=None, 
                 T=96,
                 cache_dir='./cache'):
        """
        Dataset for MIMIC paired time series and discharge summary data intended for contrastive learning.
        Creates natural pairs of time series and corresponding discharge summaries for each admission.
        
        Args:
            event_processor: MIMICClinicalEventsProcessor instance with loaded event data
            disch_notes_processor: processor for discharge notes
            splits_loader: splits loader (optional)
            split: One of 'train', 'val', 'test'
            itemid_list: List of itemids to include in time series data. If None, uses all.
            T: Number of time steps for physiologic data
            cache_dir: Directory to cache the processed events
        """
        self.event_processor = event_processor
        self.disch_notes_processor = disch_notes_processor

        self.events_cache = MIMICEventGroupsCache(cache_dir)
    
        self.splits_loader = splits_loader
        self.split = split
        self.T = T
        
        # Get hadm_ids for this split
        if splits_loader is not None and splits_loader.train_hadm_ids is not None:
            self.hadm_ids = splits_loader.get_split_hadm_ids(split)
        else:
            # Fallback to using all hadm_ids if splits not defined
            self.hadm_ids = list(event_processor.hadm_ids)
            print(f"Warning: No split data found. Using all {len(self.hadm_ids)} hadm_ids for {split}.")
        
        # Get itemid list
        if itemid_list is None:
            # Use all unique itemids from all event sources
            all_itemids = set()
            for df_name in ['chartevents_filtered', 'labevents_filtered']:
                if hasattr(event_processor, df_name):
                    df = getattr(event_processor, df_name)
                    if 'itemid' in df.columns:
                        all_itemids.update(df['itemid'].unique())
            self.itemid_list = sorted(list(all_itemids))
        else:
            self.itemid_list = itemid_list

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
    
        baseline_tensor, physio_df, treatments_df = self.get_events_for_hadm_groups(hadm_id)
        discharge_chunks = self.disch_notes_processor.get_discharge_chunks(hadm_id)
        
        physio_tensor, mask_tensor, time_hours_tensor, length = self.pivot_and_pad_physio(physio_df, self.itemid_list, T=self.T)
                
        sample = {
            'hadm_id': hadm_id,
            'physio_tensor': physio_tensor,       
            'mask_tensor': mask_tensor,  
            'time_hours_tensor': time_hours_tensor, 
            'length': length,
            'baseline_tensor': baseline_tensor,
            'treatments_df': treatments_df,
            'discharge_chunks': discharge_chunks,
        }

        return sample
    

    def get_events_for_hadm_groups(self, hadm_id):
        # Check cache first
        cached = self.events_cache.load(hadm_id)
        if cached is not None:
            return cached

        # Process as before:
        def select_if_exists(df):
            if df is not None and hasattr(df, 'index') and hadm_id in df.index:
                return df.loc[[hadm_id]]
            else:
                return None
            
        baseline_tensor = self.process_baseline(hadm_id)
            
        # Physio events
        physio_sources = []
        if hasattr(self.event_processor, 'chartevents_filtered') and self.event_processor.chartevents_filtered is not None:
            physio_sources.append((self.event_processor.chartevents_filtered, 'chartevents_filtered'))
        if hasattr(self.event_processor, 'labevents_filtered') and self.event_processor.labevents_filtered is not None:
            physio_sources.append((self.event_processor.labevents_filtered, 'labevents_filtered'))
            
        physio_dfs = []
        for df, source_name in physio_sources:
            tmp = select_if_exists(df)
            if tmp is not None:
                tmp = tmp.copy()
                tmp['source'] = source_name
                physio_dfs.append(tmp)

        if physio_dfs:
            physio_df = pd.concat(physio_dfs, sort=False)
            if 'charttime' in physio_df.columns:
                physio_df['charttime'] = pd.to_datetime(physio_df['charttime'], errors='coerce')
                physio_df = physio_df.sort_values(by='charttime')
        else:
            physio_df = pd.DataFrame()

        # Treatments
        treatment_sources = []
        if hasattr(self.event_processor, 'inputevents_filtered') and self.event_processor.inputevents_filtered is not None:
            treatment_sources.append((self.event_processor.inputevents_filtered, 'inputevents_filtered'))
        if hasattr(self.event_processor, 'emar_filtered') and self.event_processor.emar_filtered is not None:
            treatment_sources.append((self.event_processor.emar_filtered, 'emar_filtered'))
            
        treatment_dfs = []
        for df, source_name in treatment_sources:
            tmp = select_if_exists(df)
            if tmp is not None:
                tmp = tmp.copy()
                tmp['source'] = source_name
                treatment_dfs.append(tmp)

        if treatment_dfs:
            treatments_df = pd.concat(treatment_dfs, sort=False)
            if 'charttime' in treatments_df.columns:
                treatments_df['charttime'] = pd.to_datetime(treatments_df['charttime'], errors='coerce')
                treatments_df = treatments_df.sort_values(by='charttime')
        else:
            treatments_df = pd.DataFrame()

        # Compose tuple of event groups
        event_tuple = (baseline_tensor, physio_df, treatments_df)
        # Save to cache for future accesses
        self.events_cache.save(hadm_id, event_tuple)
        return event_tuple

    def process_baseline(self, hadm_id):
        """
        Extracts the age and gender from the merged_with_disch_df for the given hadm_id.
        Gender is mapped as binary: 0 for Male (or 'M'/'Male') and 1 for Female (or 'F'/'Female').
        Returns a tensor of shape [2] in the form [age, gender].
        """
        # Verify that merged_with_disch_df is available.
        if not (hasattr(self.event_processor, 'merged_with_disch_df') and 
                self.event_processor.merged_with_disch_df is not None):
            raise ValueError("The event processor does not have merged_with_disch_df.")

        # Find the row corresponding to this hadm_id.
        meta = self.event_processor.merged_with_disch_df[
            self.event_processor.merged_with_disch_df['hadm_id'] == hadm_id
        ]
        if meta.empty:
            # If not found, return default values (could also choose to raise an error).
            age_value = 0.0
            gender_value = -1
        else:
            row = meta.iloc[0]
            try:
                age_value = float(row['age'])
            except Exception:
                age_value = 0.0

            # Map gender: use 'M' or 'Male' as male (0), and 'F' or 'Female' as female (1).
            gender_raw = str(row['gender']).strip()
            gender_mapping = {'M': 0, 'Male': 0, 'F': 1, 'Female': 1}
            gender_value = gender_mapping.get(gender_raw, -1)

        # Create and return the tensor.
        baseline_tensor = torch.tensor([age_value, gender_value], dtype=torch.float)
        return baseline_tensor
    

    def pivot_and_pad_physio(self, physio_df, itemid_list, T):
        """
        Given a patient's physio_df (with columns ['charttime', 'itemid', 'value', ...]),
        produce three outputs:
        1. A tensor of physiologic values of shape (T, NUM_ITEMIDS)
        2. A tensor of binary masks of shape (T, NUM_ITEMIDS) (1 if present, 0 if missing)
        3. A tensor of time offsets (in hours) from the earliest event, of shape (T,)
        Also compute 'length', the number of time steps with any data (i.e. where mask has any 1s).
        
        Steps:
        1. Round charttime to 6h increments.
        2. Compute the time offset for each rounded charttime relative to the first event.
        3. Pivot itemid -> columns (one column per itemid) using the first value observed.
        4. Create a mask indicating where data was observed.
        5. Pad or truncate the values, mask, and time offset arrays to exactly T time steps.
        6. Convert the arrays into PyTorch tensors.
        7. Compute the number of time steps that contain any data.
        """
        if physio_df.empty:
            num_items = len(itemid_list)
            values_array = np.zeros((T, num_items), dtype=np.float32)
            mask_array = np.zeros((T, num_items), dtype=np.float32)
            time_offsets_array = np.zeros((T,), dtype=np.float32)
            length = T  # Since it's all padding, you might opt for 0 instead.
            return (torch.from_numpy(values_array).float(),
                    torch.from_numpy(mask_array).float(),
                    torch.from_numpy(time_offsets_array).float(),
                    length)

        # 1) Round charttime to 6H increments.
        physio_df = physio_df.copy()
        physio_df['charttime'] = pd.to_datetime(physio_df['charttime'], errors='coerce')
        physio_df['rounded_charttime'] = physio_df['charttime'].dt.round('6H')

        # 2) Pivot the dataframe so that each row corresponds to a rounded_charttime and each column to an itemid.
        pivot_df = physio_df.pivot_table(
            index='rounded_charttime',
            columns='itemid',
            values='value',
            aggfunc='first'
        )
        # Ensure that all desired itemids are included in the correct order.
        pivot_df = pivot_df.sort_index().reindex(columns=itemid_list)
        
        # 3) Create separate dataframes for values and binary mask.
        values_full = pivot_df.fillna(0)
        mask_full = pivot_df.notnull().astype(int)
        
        # 4) Determine the number of available time steps and re-calculate time offsets using the pivot index.
        num_times = values_full.shape[0]
        if num_times > 0:
            pivot_times = values_full.index.to_series().reset_index(drop=True)
            base_time = pivot_times[0]
            time_offsets = ((pivot_times - base_time) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32)
        else:
            time_offsets = np.array([], dtype=np.float32)
            
        # Pad or truncate the arrays if needed to ensure exactly T time steps.
        if num_times < T:
            pad_rows = T - num_times
            pad_values = pd.DataFrame(np.zeros((pad_rows, values_full.shape[1])), columns=values_full.columns)
            pad_mask = pd.DataFrame(np.zeros((pad_rows, mask_full.shape[1])), columns=mask_full.columns)
            values_full = pd.concat([values_full, pad_values], ignore_index=True)
            mask_full = pd.concat([mask_full, pad_mask], ignore_index=True)
            time_offsets = np.concatenate([time_offsets, np.zeros(pad_rows, dtype=np.float32)])
        else:
            values_full = values_full.iloc[:T]
            mask_full = mask_full.iloc[:T]
            time_offsets = time_offsets[:T]

        # 5) Compute the length: count time steps that have any valid data (mask sum > 0).
        has_data = mask_full.sum(axis=1) > 0
        length = int(np.sum(has_data))
        if length == 0:  # Edge case: if no data, ensure nonzero length.
            length = 1

        # Convert the individual arrays to PyTorch tensors.
        values_tensor = torch.from_numpy(values_full.to_numpy(dtype=np.float32)).float()
        mask_tensor = torch.from_numpy(mask_full.to_numpy(dtype=np.float32)).float()
        time_offsets_tensor = torch.from_numpy(time_offsets).float()

        return values_tensor, mask_tensor, time_offsets_tensor, length