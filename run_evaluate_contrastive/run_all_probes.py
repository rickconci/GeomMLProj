import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from tqdm import tqdm
import time

def find_available_epochs(model_dir):
    """Find all available epoch directories in a model directory."""
    epoch_dirs = glob.glob(os.path.join(model_dir, "epoch_*"))
    epochs = []
    
    for epoch_dir in epoch_dirs:
        try:
            epoch_num = int(os.path.basename(epoch_dir).split("_")[1])
            # Check if the directory contains the necessary files
            if (os.path.exists(os.path.join(epoch_dir, "ts_proj.mmap")) and
                os.path.exists(os.path.join(epoch_dir, "text_proj.mmap"))):
                epochs.append(epoch_num)
        except (ValueError, IndexError):
            continue
    
    return sorted(epochs)

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

def parse_result_file(result_file, model_info, epoch):
    """Parse the numpy result file and convert to a dictionary."""
    try:
        results = np.load(result_file, allow_pickle=True).item()
        # Merge the model info with the results
        row = model_info.copy()
        row["epoch"] = epoch
        
        # Add all the metrics from the results
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                row[key] = value
                
        return row
    except Exception as e:
        print(f"Error parsing result file {result_file}: {e}")
        return None

def train_probes_for_model_epoch(model_dir, epoch, args):
    """Train probes for a specific model and epoch."""
    cmd = [
        "python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_probes.py"),
        "--embeddings_dir", model_dir,
        "--epoch", str(epoch),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--phecode_size", str(args.phecode_size)
    ]
    
    # Add hidden_dim if specified (can be None)
    if args.hidden_dim is not None:
        cmd.extend(["--hidden_dim", str(args.hidden_dim)])
    
    if args.use_wandb:
        cmd.append("--use_wandb")
        cmd.extend(["--wandb_project", args.wandb_project])
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error training probes for {model_dir}, epoch {epoch}: {e}")
        return False

def main(args):
    # Get all model directories
    embeddings_root = args.embeddings_root
    model_dirs = [d for d in glob.glob(os.path.join(embeddings_root, "*")) if os.path.isdir(d)]
    
    # Create DataFrame to store all results
    all_results = []
    
    for model_dir in tqdm(model_dirs, desc="Processing models"):
        print(f"\nProcessing model: {os.path.basename(model_dir)}")
        
        # Extract model parameters
        model_info = extract_model_params(model_dir)
        
        # Find available epochs
        epochs = find_available_epochs(model_dir)
        
        if not epochs:
            print(f"No valid epoch directories found in {model_dir}")
            continue
        
        print(f"Found {len(epochs)} epochs: {epochs}")
        
        for epoch in tqdm(epochs, desc=f"Processing epochs for {os.path.basename(model_dir)}"):
            # Create a directory name suffix based on hidden_dim
            hidden_suffix = f"_hidden{args.hidden_dim}" if args.hidden_dim is not None else "_linear"
            probe_results_dir = os.path.join(model_dir, f"probe_results{hidden_suffix}")
            os.makedirs(probe_results_dir, exist_ok=True)
            
            result_file = os.path.join(probe_results_dir, f"epoch_{epoch}_results.npy")
            
            # Check if result file already exists and we're not forcing recomputation
            if os.path.exists(result_file) and not args.force:
                print(f"Result file already exists for {model_dir}, epoch {epoch}. Skipping...")
                row = parse_result_file(result_file, model_info, epoch)
                if row:
                    all_results.append(row)
            else:
                print(f"Training probes for {model_dir}, epoch {epoch}")
                success = train_probes_for_model_epoch(model_dir, epoch, args)
                
                if success and os.path.exists(result_file):
                    row = parse_result_file(result_file, model_info, epoch)
                    if row:
                        all_results.append(row)
                else:
                    print(f"Failed to generate results for {model_dir}, epoch {epoch}")
            
            # Add a small delay to prevent potential resource issues
            time.sleep(1)
        
        # Create a DataFrame and save to CSV periodically
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(args.output_csv, index=False)
            print(f"Saved results to {args.output_csv}")
    
    # Final save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"All results saved to {args.output_csv}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_root", type=str, default="/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/Embeddings", 
                        help="Root directory containing model directories")
    parser.add_argument("--output_csv", type=str, default="probe_results_all.csv", 
                        help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training probes")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate for training probes")
    parser.add_argument("--phecode_size", type=int, default=1788, 
                        help="Number of PHEcodes")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension for probe networks (None for linear probes)")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="GeomML_Probes", 
                        help="WandB project name")
    parser.add_argument("--force", action="store_true", 
                        help="Force recomputation even if results exist")
    args = parser.parse_args()
    
    main(args) 