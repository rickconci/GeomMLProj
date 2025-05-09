import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
import shutil

from data_scripts.data import (
    MIMICDemographicsLoader,
    MIMICClinicalEventsProcessor,
    MIMICDischargeNotesProcessor,
    MIMICContrastivePairsDataset,
)

class MIMICDataConverter:
    def __init__(self, base_path, temp_dfs_path='temp_dfs', 
                 cache_dir='./cache', outcome_choice='30d_mortality_discharge',
                 use_existing_temp_dfs=True, data_output_dir='data/mimic4'):
        """
        Converts MIMIC-IV data from ContrastiveRain format to format required by KEDGN model.
        
        Args:
            base_path: Base path to MIMIC-IV data
            temp_dfs_path: Path to temp_dfs directory with cached processed files
            cache_dir: Directory to cache processed data
            outcome_choice: Which outcome to predict (default: 30-day mortality after discharge)
            use_existing_temp_dfs: Whether to use existing processed files in temp_dfs
            data_output_dir: Directory to save final processed data for KEDGN model
        """
        self.base_path = base_path
        self.temp_dfs_path = temp_dfs_path  # Path to existing processed files
        self.cache_dir = cache_dir
        self.outcome_choice = outcome_choice
        self.use_existing_temp_dfs = use_existing_temp_dfs
        self.data_output_dir = data_output_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.data_output_dir, exist_ok=True)
        
        # Initialize loaders and processors
        self.demo_loader = MIMICDemographicsLoader(base_path, temp_dfs_path if use_existing_temp_dfs else cache_dir)
        
    def prepare_data(self):
        """Load and prepare all data components"""
        print("Loading demographics data...")
        self.demo_loader.load_demographics()
                
        print("Splitting data by subject_id...")
        self.demo_loader.split_by_subject_id(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        # Get outcomes
        outcome_df = self.demo_loader.get_30_day_mortality_outcome(self.outcome_choice)
        hadm_ids = self.demo_loader.get_hadm_ids()
        print(f"Found {len(hadm_ids)} hospital admissions with discharge notes")

        # Initialize and load clinical events data
        print("Loading clinical events data from existing files...")
        self.event_processor = MIMICClinicalEventsProcessor(
            self.base_path, 
            self.demo_loader.get_hadm_ids(),
            self.temp_dfs_path if self.use_existing_temp_dfs else self.cache_dir
        )
        self.event_processor.load_all_events()
        self.event_processor.discharge_df = self.demo_loader.discharge_df
        self.event_processor.merged_with_disch_df = self.demo_loader.merged_with_disch_df
        self.event_processor.process_events()
        self.cluster_labels = self.event_processor.cluster_labels
        
        # Initialize discharge notes processor
        print("Initializing discharge notes processor...")
        self.discharge_processor = MIMICDischargeNotesProcessor(
            self.demo_loader.get_hadm_ids(),
            self.temp_dfs_path if self.use_existing_temp_dfs else self.cache_dir
        )
        
        # Create datasets
        print("Creating datasets...")
        self.train_dataset = MIMICContrastivePairsDataset(
            self.event_processor,
            self.discharge_processor,
            self.demo_loader,
            split='train',
            T=96,
            cache_dir=self.temp_dfs_path if self.use_existing_temp_dfs else self.cache_dir
        )
        
        self.val_dataset = MIMICContrastivePairsDataset(
            self.event_processor,
            self.discharge_processor,
            self.demo_loader,
            split='val',
            T=96,
            cache_dir=self.temp_dfs_path if self.use_existing_temp_dfs else self.cache_dir
        )
        
        self.test_dataset = MIMICContrastivePairsDataset(
            self.event_processor,
            self.discharge_processor,
            self.demo_loader,
            split='test',
            T=96,
            cache_dir=self.temp_dfs_path if self.use_existing_temp_dfs else self.cache_dir
        )
        
        # Convert datasets to format needed by KEDGN model
        print("Converting datasets to KEDGN format...")
        return self.convert_to_kedgn_format(outcome_df)
    
    def convert_to_kedgn_format(self, outcome_df):
        """
        Convert datasets from MIMICContrastivePairsDataset format to KEDGN format
        
        Args:
            outcome_df: DataFrame with outcomes (30-day mortality)
            
        Returns:
            Tuple of (Ptrain, Pval, Ptest, ytrain, yval, ytest, P_var_plm_rep_tensor)
        """
        # Get PLM representations for variables
        P_var_plm_rep_tensor = self.generate_variable_embeddings()
        
        # Process train, val, test datasets
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_avg_interval_tensor, Ptrain_length_tensor, Ptrain_time_tensor = self.process_dataset(self.train_dataset)
        Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor, Pval_length_tensor, Pval_time_tensor = self.process_dataset(self.val_dataset)
        Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor, Ptest_length_tensor, Ptest_time_tensor = self.process_dataset(self.test_dataset)
        
        # Get labels for each set
        ytrain = self.get_labels(self.train_dataset.hadm_ids, outcome_df)
        yval = self.get_labels(self.val_dataset.hadm_ids, outcome_df)
        ytest = self.get_labels(self.test_dataset.hadm_ids, outcome_df)
        
        # Save processed tensors in KEDGN-compatible format
        self.save_processed_data_for_kedgn(
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_avg_interval_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain,
            Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor, Pval_length_tensor, Pval_time_tensor, yval,
            Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest,
            P_var_plm_rep_tensor
        )
        
        # Return processed data and tensors for KEDGN model
        return Ptrain_tensor, Pval_tensor, Ptest_tensor, ytrain, yval, ytest, P_var_plm_rep_tensor
    
    def process_dataset(self, dataset, n_features=669, time_steps=96, n_static=2):
        """
        Process a MIMICContrastivePairsDataset into tensors for KEDGN model
        
        Args:
            dataset: A MIMICContrastivePairsDataset instance
            
        Returns:
            Tuple of tensors (P_tensor, P_static_tensor, P_avg_interval_tensor, P_length_tensor, P_time_tensor)
        """
        # Initialize tensors
        n_samples = len(dataset)        
        
        # Create empty tensors to hold all data
        # KEDGN expects shape (n_samples, n_features, time_steps)
        P_tensor = torch.zeros((n_samples, time_steps, n_features*2))  # *2 because we include both values and mask
        P_static_tensor = torch.zeros((n_samples, n_static)) if n_static > 0 else None
        P_avg_interval_tensor = torch.zeros((n_samples, time_steps, n_features))
        P_length_tensor = torch.zeros((n_samples, 1))
        P_time_tensor = torch.zeros((n_samples, time_steps, n_features))
        
        # Fill tensors with data from dataset
        for i, sample in tqdm(enumerate(dataset), total=n_samples, desc="Processing dataset"):
            # Format P_tensor with values in first half of channels, mask in second half
            # This matches the format expected by KEDGN after mask_normalize
            values_part = sample['physio_tensor']
            mask_part = sample['mask_tensor']
            
            # Concatenate along the features dimension (dim=1)
            P_tensor[i] = torch.cat([values_part, mask_part], dim=1)
            
            if P_static_tensor is not None:
                P_static_tensor[i] = sample['baseline_tensor']
            
            # Use the actual time information to compute intervals between measurements
            time_hours = sample['time_hours_tensor']
            mask = sample['mask_tensor']
            for f in range(n_features):
                # Find indices where this feature was measured (mask > 0)
                feature_mask = mask[:, f]
                measured_indices = torch.where(feature_mask > 0)[0]
                
                if len(measured_indices) > 0:
                    if len(measured_indices) == 1:
                        # If only one measurement, use half the length as interval
                        P_avg_interval_tensor[i, measured_indices[0], f] = sample['length'] / 2
                    else:
                        # Calculate intervals between consecutive measurements
                        measured_times = time_hours[measured_indices]
                        
                        # Calculate forward differences (time to next measurement)
                        forward_diffs = torch.diff(measured_times, dim=0)
                        # Add the last interval (assume same as previous)
                        forward_diffs = torch.cat([forward_diffs, forward_diffs[-1:]])
                        
                        # Calculate backward differences (time since last measurement)
                        backward_diffs = torch.cat([forward_diffs[0:1], forward_diffs[:-1]])
                        
                        # Average interval = (time to next + time since last) / 2
                        avg_intervals = (forward_diffs + backward_diffs) / 2
                        
                        # Assign intervals to the measured time points
                        P_avg_interval_tensor[i, measured_indices, f] = avg_intervals
            
            P_length_tensor[i, 0] = sample['length']
            
            # Expand time_hours_tensor to match feature dimensions
            for f in range(n_features):
                P_time_tensor[i, :, f] = sample['time_hours_tensor']
        
        # Transpose to match KEDGN's expected format: [N, T, F] -> [N, F, T]
        P_tensor = P_tensor.transpose(1, 2)
        P_avg_interval_tensor = P_avg_interval_tensor.transpose(1, 2)
        P_time_tensor = P_time_tensor.transpose(1, 2)
        
        return P_tensor, P_static_tensor, P_avg_interval_tensor, P_length_tensor, P_time_tensor
    
    def generate_variable_embeddings(self):
        """
        Generate embeddings for physiological variables
        
        Returns:
            Tensor of variable embeddings
        """
        var_names = self.event_processor.provide_physio_var_names()
        n_vars = len(var_names)
        
        # Create a simple embedding as placeholder - in practice, use PLM embeddings
        # In the train.py they load these from a file
        plm_rep_dim = 768  # Typical BERT embedding dimension
        P_var_plm_rep_tensor = torch.randn(n_vars, plm_rep_dim)
        
        # Save embeddings to file for loading in train.py
        save_path = os.path.join(self.data_output_dir, "mimic4_bert_var_rep_gpt_source.pt")
        torch.save(P_var_plm_rep_tensor, save_path)
        print(f"Saved variable embeddings to {save_path}")
        
        return P_var_plm_rep_tensor
    
    def get_labels(self, hadm_ids, outcome_df):
        """
        Get outcome labels for a list of hospital admission IDs
        
        Args:
            hadm_ids: List of hospital admission IDs
            outcome_df: DataFrame with outcomes
            
        Returns:
            NumPy array of labels
        """
        labels = []
        for hadm_id in hadm_ids:
            if hadm_id in outcome_df.index:
                labels.append(outcome_df.loc[hadm_id, 'label_death_within_30d'])
            else:
                # Default to negative class if not found
                labels.append(0)
        
        return np.array(labels).reshape(-1, 1)
    
    def save_processed_data_for_kedgn(self, *args):
        """
        Save processed data in format needed by KEDGN model:
        - train_x.npy, val_x.npy, test_x.npy (the input data)
        - train_y.npy, val_y.npy, test_y.npy (the labels)
        - mimic4_bert_var_rep_gpt_source.pt (variable embeddings)
        
        This matches the format expected by the train.py script.
        """
        # Extract tensors from args
        Ptrain_tensor, Ptrain_static_tensor, Ptrain_avg_interval_tensor, Ptrain_length_tensor, Ptrain_time_tensor, ytrain, \
        Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor, Pval_length_tensor, Pval_time_tensor, yval, \
        Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor, Ptest_length_tensor, Ptest_time_tensor, ytest, \
        P_var_plm_rep_tensor = args
        
        # Make sure the output directory exists
        os.makedirs(self.data_output_dir, exist_ok=True)
        
        # Convert training data to format expected by KEDGN
        train_x = self.create_mimic_compatible_x_structure(
            Ptrain_tensor, Ptrain_static_tensor, Ptrain_avg_interval_tensor, Ptrain_length_tensor, Ptrain_time_tensor
        )
        val_x = self.create_mimic_compatible_x_structure(
            Pval_tensor, Pval_static_tensor, Pval_avg_interval_tensor, Pval_length_tensor, Pval_time_tensor
        )
        test_x = self.create_mimic_compatible_x_structure(
            Ptest_tensor, Ptest_static_tensor, Ptest_avg_interval_tensor, Ptest_length_tensor, Ptest_time_tensor
        )
        
        # Save the files in KEDGN-compatible format
        np.save(os.path.join(self.data_output_dir, 'mimic4_train_x.npy'), train_x)
        np.save(os.path.join(self.data_output_dir, 'mimic4_val_x.npy'), val_x)
        np.save(os.path.join(self.data_output_dir, 'mimic4_test_x.npy'), test_x)
        np.save(os.path.join(self.data_output_dir, 'mimic4_train_y.npy'), ytrain)
        np.save(os.path.join(self.data_output_dir, 'mimic4_val_y.npy'), yval)
        np.save(os.path.join(self.data_output_dir, 'mimic4_test_y.npy'), ytest)
        
        # Variable embeddings already saved in generate_variable_embeddings
        print(f"Saved all processed data to {self.data_output_dir}")
        
        # Create a splits file for compatibility with train.py
        splits_dir = os.path.join(self.data_output_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        # Create placeholder split indices (not actually used but needed for compatibility)
        split_indices = [
            np.arange(len(ytrain)),
            np.arange(len(yval)),
            np.arange(len(ytest))
        ]
        np.save(os.path.join(splits_dir, 'mimic4_split0.npy'), split_indices)
        
        # No need to create symlink since the data is already in the correct structure
        # train.py should be able to find it at GeomMLProj/data/mimic4
    
    def create_mimic_compatible_x_structure(self, P_tensor, P_static_tensor, P_avg_interval_tensor, P_length_tensor, P_time_tensor):
        """
        Create a list of numpy arrays compatible with the MIMIC-III format expected by the KEDGN model.
        
        The model expects each sample as a tuple with structure:
        [id, times, values, mask, length]
        
        Returns:
            List of tuples in the format expected by KEDGN
        """
        n_samples = P_tensor.shape[0]
        formatted_data = []
        
        for i in range(n_samples):
            # Extract data for this sample
            sample_P = P_tensor[i].detach().cpu().numpy()  # Shape: [F*2, T]
            sample_length = int(P_length_tensor[i].item())
            sample_time = P_time_tensor[i][0].detach().cpu().numpy()  # Shape: [T]
            
            # Split sample_P into values and mask
            F = sample_P.shape[0] // 2
            values = sample_P[:F]  # First half of features
            mask = sample_P[F:]    # Second half of features
            
            # Create the tuple in format expected by KEDGN for MIMIC data
            sample_tuple = [
                i,                  # ID (just use index)
                sample_time,        # Time steps
                values,             # Values
                mask,               # Mask
                sample_length       # Length
            ]
            
            formatted_data.append(sample_tuple)
        
        return np.array(formatted_data, dtype=object)

def convert_mimic_to_kedgn_format(base_path, temp_dfs_path='temp_dfs', 
                                   data_output_dir='data/mimic4'):
    """
    Convenience function to convert MIMIC data to KEDGN format
    
    Args:
        base_path: Path to MIMIC-IV data
        temp_dfs_path: Path to existing processed files
        data_output_dir: Directory to save the final processed data
        
    Returns:
        Tuple of (Ptrain, Pval, Ptest, ytrain, yval, ytest, P_var_plm_rep_tensor)
    """
    converter = MIMICDataConverter(base_path, temp_dfs_path=temp_dfs_path, 
                                   data_output_dir=data_output_dir)
    return converter.prepare_data()

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Convert MIMIC-IV data to KEDGN format')
    parser.add_argument('--base_path', type=str, help='Path to MIMIC-IV data', default='/Users/riccardoconci/Local_documents/!!MIMIC')
    parser.add_argument('--temp_dfs_path', type=str, default=os.path.join(script_dir, 'temp_dfs'), 
                       help='Path to directory with existing processed files')
    parser.add_argument('--data_output_dir', type=str, default=os.path.join(script_dir, 'data/mimic4'),
                       help='Directory to save final processed data for KEDGN model')
    parser.add_argument('--outcome', type=str, default='30d_mortality_discharge', 
                       choices=['30d_mortality_discharge', '48h_mortality'],
                       help='Outcome to predict')
    args = parser.parse_args()
    
    print(f"Converting MIMIC-IV data from {args.base_path}")
    print(f"Using existing processed files from: {args.temp_dfs_path}")
    print(f"Saving final data to: {args.data_output_dir}")
    print(f"Outcome: {args.outcome}")
    
    # Make sure output directory exists
    os.makedirs(args.data_output_dir, exist_ok=True)
    
    converter = MIMICDataConverter(
        args.base_path, 
        temp_dfs_path=args.temp_dfs_path,
        outcome_choice=args.outcome,
        data_output_dir=args.data_output_dir
    )
    _ = converter.prepare_data()
    
    print("\nData conversion complete!")
    print(f"\nTo use with KEDGN model, run:")
    print(f"python train.py --dataset mimic4 --cuda 0 --epochs 10 --batch_size 256")
