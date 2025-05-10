import os
import sys
import argparse
import numpy as np
import json
import glob
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

def extract_model_params(model_dir):
    """Extract model parameters from the directory name."""
    basename = os.path.basename(model_dir)
    params = basename.split("_")
    
    model_info = {
        "model_name": basename
    }
    
    for param in params:
        if param.startswith("bs"):
            model_info["batch_size"] = int(param[2:])
        elif param.startswith("lr"):
            model_info["learning_rate"] = float(param[2:])
        elif param.startswith("seed"):
            model_info["seed"] = int(param[4:])
        elif param.startswith("proj"):
            model_info["projection_dim"] = int(param[4:])
        elif param.startswith("temp"):
            model_info["temperature"] = float(param[4:])
        elif param in ["phe", "nophe"]:
            model_info["phecode_setting"] = param
    
    return model_info

def load_metadata(epoch_dir):
    """Load metadata.json file from the epoch directory."""
    metadata_path = os.path.join(epoch_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def load_mmap_file(file_path, dtype, shape=None):
    """Load a memory-mapped file."""
    try:
        if shape is None:
            # Try to infer shape from file size
            file_size = os.path.getsize(file_path)
            element_size = np.dtype(dtype).itemsize
            length = file_size // element_size
            return np.memmap(file_path, dtype=dtype, mode='r', shape=(length,))
        else:
            return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def view_embedding_data(model_dir, epoch, args):
    """View data from a specific model and epoch."""
    # Construct the epoch directory path
    epoch_dir = os.path.join(model_dir, f"epoch_{epoch:03d}")
    
    if not os.path.exists(epoch_dir):
        print(f"Error: Epoch directory {epoch_dir} does not exist.")
        return
    
    print(f"\n{'='*80}")
    print(f"Viewing data for model: {os.path.basename(model_dir)}, epoch: {epoch}")
    print(f"{'='*80}")
    
    # Extract model parameters
    model_info = extract_model_params(model_dir)
    print("\nModel Parameters:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Load metadata
    metadata = load_metadata(epoch_dir)
    if metadata:
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    # List available files
    print("\nAvailable data files:")
    for file_path in sorted(glob.glob(os.path.join(epoch_dir, "*.mmap"))):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        print(f"  {file_name}: {file_size/1024/1024:.2f} MB")
    
    # Load and display information about key files
    print("\nData Analysis:")
    
    # Check if the user wants to analyze next_idx_padded.mmap and hadm_id.mmap
    if args.analyze_idx_padded:
        next_idx_path = os.path.join(epoch_dir, "next_idx_padded.mmap")
        hadm_path = os.path.join(epoch_dir, "hadm_id.mmap")
        
        if os.path.exists(next_idx_path) and os.path.exists(hadm_path):
            # Load both files
            next_idx = load_mmap_file(next_idx_path, dtype=np.int32)
            hadm_ids = load_mmap_file(hadm_path, dtype=np.int32)
            
            if next_idx is not None and hadm_ids is not None:
                # Make sure both arrays have the same length for the DataFrame
                min_len = min(len(next_idx), len(hadm_ids))
                
                # Create DataFrame
                df = pd.DataFrame({
                    'hadm_id': hadm_ids[:min_len],
                    'next_idx_padded': next_idx[:min_len]
                })
                
                # Print statistics
                print(f"\nAnalysis of next_idx_padded.mmap and hadm_id.mmap:")
                print(f"  Total entries: {len(df)}")
                print(f"  Zero values in next_idx_padded: {sum(df['next_idx_padded'] == 0)} ({sum(df['next_idx_padded'] == 0)/len(df)*100:.2f}%)")
                print(f"  Unique values in next_idx_padded: {df['next_idx_padded'].nunique()}")
                print(f"  Unique values in hadm_id: {df['hadm_id'].nunique()}")
                
                # Display value counts for next_idx_padded
                print("\nValue counts for next_idx_padded (top 10):")
                value_counts = df['next_idx_padded'].value_counts().head(10)
                for value, count in value_counts.items():
                    print(f"  {value}: {count} ({count/len(df)*100:.2f}%)")
                
                # Optional: save to CSV
                if args.save_csv:
                    csv_path = os.path.join(model_dir, f"idx_analysis_epoch_{epoch}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"\nSaved analysis to {csv_path}")
                
                # Optional: display DataFrame sample
                if args.show_sample:
                    sample_size = min(args.sample_size, len(df))
                    print(f"\nSample of DataFrame (first {sample_size} rows):")
                    print(df.head(sample_size))
            else:
                print("Failed to load next_idx_padded.mmap or hadm_id.mmap")
        else:
            print("next_idx_padded.mmap or hadm_id.mmap not found in the epoch directory")
    
    # Text projections
    text_proj_path = os.path.join(epoch_dir, "text_proj.mmap")
    if os.path.exists(text_proj_path):
        # Infer shape using metadata or projection dim from model info
        proj_dim = model_info.get("projection_dim", 256)
        file_size = os.path.getsize(text_proj_path)
        dtype = np.float32  # Assuming float32
        num_samples = file_size // (proj_dim * np.dtype(dtype).itemsize)
        
        if args.load_data:
            text_proj = load_mmap_file(text_proj_path, dtype=np.float32, shape=(num_samples, proj_dim))
            if text_proj is not None:
                print(f"  text_proj: {text_proj.shape}, mean: {np.mean(text_proj):.4f}, "
                     f"std: {np.std(text_proj):.4f}, min: {np.min(text_proj):.4f}, max: {np.max(text_proj):.4f}")
        else:
            print(f"  text_proj: shape estimated as ({num_samples}, {proj_dim})")
    
    # Time series projections
    ts_proj_path = os.path.join(epoch_dir, "ts_proj.mmap")
    if os.path.exists(ts_proj_path):
        proj_dim = model_info.get("projection_dim", 256)
        file_size = os.path.getsize(ts_proj_path)
        dtype = np.float32  # Assuming float32
        num_samples = file_size // (proj_dim * np.dtype(dtype).itemsize)
        
        if args.load_data:
            ts_proj = load_mmap_file(ts_proj_path, dtype=np.float32, shape=(num_samples, proj_dim))
            if ts_proj is not None:
                print(f"  ts_proj: {ts_proj.shape}, mean: {np.mean(ts_proj):.4f}, "
                     f"std: {np.std(ts_proj):.4f}, min: {np.min(ts_proj):.4f}, max: {np.max(ts_proj):.4f}")
        else:
            print(f"  ts_proj: shape estimated as ({num_samples}, {proj_dim})")
    
    # Labels
    for label_file in ["mortality_label.mmap", "readmission_label.mmap"]:
        label_path = os.path.join(epoch_dir, label_file)
        if os.path.exists(label_path):
            if args.load_data:
                labels = load_mmap_file(label_path, dtype=np.int32)
                if labels is not None:
                    pos_count = np.sum(labels == 1)
                    neg_count = np.sum(labels == 0)
                    print(f"  {label_file}: {len(labels)} samples, "
                         f"positive: {pos_count} ({pos_count/len(labels)*100:.2f}%), "
                         f"negative: {neg_count} ({neg_count/len(labels)*100:.2f}%)")
            else:
                file_size = os.path.getsize(label_path)
                dtype = np.int32  # Assuming int32
                num_samples = file_size // np.dtype(dtype).itemsize
                print(f"  {label_file}: {num_samples} samples (estimate)")
    
    # HADM IDs
    hadm_path = os.path.join(epoch_dir, "hadm_id.mmap")
    if os.path.exists(hadm_path):
        if args.load_data:
            hadm_ids = load_mmap_file(hadm_path, dtype=np.int32)
            if hadm_ids is not None:
                print(f"  hadm_id.mmap: {len(hadm_ids)} samples, {len(np.unique(hadm_ids))} unique IDs")
        else:
            file_size = os.path.getsize(hadm_path)
            dtype = np.int32  # Assuming int32
            num_samples = file_size // np.dtype(dtype).itemsize
            print(f"  hadm_id.mmap: {num_samples} samples (estimate)")
    
    # PHEcode related data
    if model_info.get("phecode_setting") == "phe":
        phecode_len_path = os.path.join(epoch_dir, "next_phecode_len.mmap")
        if os.path.exists(phecode_len_path) and args.load_data:
            phecode_lens = load_mmap_file(phecode_len_path, dtype=np.int32)
            if phecode_lens is not None:
                print(f"  next_phecode_len.mmap: {len(phecode_lens)} samples, "
                     f"mean length: {np.mean(phecode_lens):.2f}, max length: {np.max(phecode_lens)}")
    
    # Check if the user wants to extract embeddings for visualization
    if args.extract_embeddings:
        print("\nExtracting embeddings for visualization...")
        
        # Load required mmap files
        hadm_path = os.path.join(epoch_dir, "hadm_id.mmap")
        ts_proj_path = os.path.join(epoch_dir, "ts_proj.mmap")
        text_proj_path = os.path.join(epoch_dir, "text_proj.mmap")
        mortality_path = os.path.join(epoch_dir, "mortality_label.mmap")
        readmission_path = os.path.join(epoch_dir, "readmission_label.mmap")
        
        # Check if all required files exist
        required_files = [hadm_path, ts_proj_path, text_proj_path]
        if not all(os.path.exists(f) for f in required_files):
            print("Error: Not all required files exist for embedding extraction.")
            return
        
        # Load hadm_ids
        hadm_ids = load_mmap_file(hadm_path, dtype=np.int32)
        if hadm_ids is None:
            print("Error: Failed to load hadm_ids.")
            return
        
        # Get projection dimension from model info
        model_info = extract_model_params(model_dir)
        proj_dim = model_info.get("projection_dim", 256)
        
        # Estimate number of samples
        file_size = os.path.getsize(ts_proj_path)
        dtype = np.float32
        num_samples = file_size // (proj_dim * np.dtype(dtype).itemsize)
        
        # Load projections
        ts_proj = load_mmap_file(ts_proj_path, dtype=np.float32, shape=(num_samples, proj_dim))
        text_proj = load_mmap_file(text_proj_path, dtype=np.float32, shape=(num_samples, proj_dim))
        
        # Load labels if they exist
        mortality_labels = None
        readmission_labels = None
        
        if os.path.exists(mortality_path):
            mortality_labels = load_mmap_file(mortality_path, dtype=np.int32)
        
        if os.path.exists(readmission_path):
            readmission_labels = load_mmap_file(readmission_path, dtype=np.int32)
        
        # Create DataFrame with hadm_ids to find unique ones
        df_hadm = pd.DataFrame({'hadm_id': hadm_ids})
        unique_hadm_ids = df_hadm['hadm_id'].unique()
        
        print(f"Found {len(unique_hadm_ids)} unique hospital admissions (hadm_ids)")
        
        # Sample n unique hadm_ids (or take all if fewer than n)
        n_samples = min(args.num_embeddings, len(unique_hadm_ids))
        sampled_hadm_ids = np.random.choice(unique_hadm_ids, size=n_samples, replace=False)
        
        print(f"Sampled {n_samples} unique hadm_ids")
        
        # Get indices of first occurrence of each sampled hadm_id
        indices = []
        for hadm_id in sampled_hadm_ids:
            # Find first occurrence
            idx = np.where(hadm_ids == hadm_id)[0][0]
            indices.append(idx)
        
        # Extract embeddings and labels for the selected indices
        sampled_data = {
            'hadm_id': hadm_ids[indices],
            'ts_embeddings': ts_proj[indices],
            'text_embeddings': text_proj[indices]
        }
        
        # Add labels if they exist
        if mortality_labels is not None:
            sampled_data['mortality_label'] = mortality_labels[indices]
        
        if readmission_labels is not None:
            sampled_data['readmission_label'] = readmission_labels[indices]
        
        # Save the sampled data
        output_path = os.path.join(
            model_dir, 
            f"viz_embeddings_epoch_{epoch}_n{n_samples}.pkl"
        )
        
        with open(output_path, 'wb') as f:
            pickle.dump(sampled_data, f)
        
        print(f"Saved {n_samples} embeddings to {output_path}")
                

    print(f"\n{'='*80}\n")

def find_available_epochs(model_dir):
    """Find all available epoch directories in a model directory."""
    epoch_dirs = glob.glob(os.path.join(model_dir, "epoch_*"))
    epochs = []
    
    for epoch_dir in epoch_dirs:
        try:
            epoch_num = int(os.path.basename(epoch_dir).split("_")[1])
            # Check if the directory contains required files
            if (os.path.exists(os.path.join(epoch_dir, "ts_proj.mmap")) and
                os.path.exists(os.path.join(epoch_dir, "text_proj.mmap"))):
                epochs.append(epoch_num)
        except (ValueError, IndexError):
            continue
    
    return sorted(epochs)

def main():
    parser = argparse.ArgumentParser(description="View embedding data for a specific model and epoch")
    parser.add_argument("--embeddings_root", type=str, 
                        default="/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/run_evaluate_contrastive/Embeddings",
                        help="Root directory containing model directories")
    parser.add_argument("--model", type=str, default='bs128_lr0.0005_seed42_proj256_temp0.07_phe',
                        help="Model directory name (e.g., 'bs128_lr0.0005_seed42_proj256_temp0.07_phe')")
    parser.add_argument("--epoch", type=int, default=0,
                        help="Epoch number to analyze")
    parser.add_argument("--load_data", action="store_true",
                        help="Actually load data into memory to compute statistics (might be memory intensive)")
    parser.add_argument("--list_models", action="store_true",
                        help="List all available models and exit")
    parser.add_argument("--list_epochs", action="store_true",
                        help="List all available epochs for the specified model and exit")
    parser.add_argument("--analyze_idx_padded", action="store_true",
                        help="Analyze next_idx_padded.mmap in relation to hadm_id.mmap")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save the analysis to a CSV file")
    parser.add_argument("--show_sample", action="store_true",
                        help="Show a sample of the DataFrame")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Number of rows to show in the sample")
    parser.add_argument("--extract_embeddings", action="store_true",
                        help="Extract embeddings for visualization with UMAP/TSNE")
    parser.add_argument("--num_embeddings", type=int, default=37000,
                        help="Number of unique hadm_id embeddings to extract")
    
    args = parser.parse_args()
    
    # List all available models if requested
    if args.list_models:
        model_dirs = [d for d in glob.glob(os.path.join(args.embeddings_root, "*")) 
                     if os.path.isdir(d)]
        print("\nAvailable models:")
        for model_dir in sorted(model_dirs):
            print(f"  {os.path.basename(model_dir)}")
        return
    
    # Check if model is specified
    if args.model is None:
        print("Error: You must specify a model with --model or list available models with --list_models")
        return
    
    # Construct full path to model directory
    model_dir = os.path.join(args.embeddings_root, args.model)
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} does not exist.")
        return
    
    # List all available epochs if requested
    if args.list_epochs:
        epochs = find_available_epochs(model_dir)
        print(f"\nAvailable epochs for model {args.model}:")
        for epoch in epochs:
            print(f"  {epoch}")
        return
    
    # View data for the specified model and epoch
    view_embedding_data(model_dir, args.epoch, args)

if __name__ == "__main__":
    main() 