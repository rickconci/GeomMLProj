import os
import argparse
import torch
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import json
import pytorch_lightning as pl

# Add parent directory to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_utils import seed_everything, get_device
from contrastive_experiments.RaindropContrastive_lightning import RaindropContrastiveModel
from contrastive_experiments.ContrastiveDataloaderLighting import ContrastiveDataModule

def setup_logging(output_dir):
    """Set up logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'embedding_generation.log')),
            logging.StreamHandler()
        ]
    )

def prepare_batch_for_inference(batch, device):
    """Prepare a batch for inference by moving all tensors to the specified device"""
    if not batch:
        return None
        
    values = batch['values'].to(device, dtype=torch.float32)  # [B, T, F]
    mask = batch['mask'].to(device, dtype=torch.float32)  # [B, T, F]
    P = torch.cat([values, mask], dim=2).permute(1, 0, 2)  # [T, B, F]
    length = batch['length'].to(device).unsqueeze(1)  # [B, 1]
    ds_embeddings = [emb.to(device, dtype=torch.float32) for emb in batch['ds_embedding']]

    return {
        'P': P,
        'P_static': batch['static'].to(device, dtype=torch.float32),
        'P_length': length,
        'P_time': batch['times'].to(device, dtype=torch.float32),
        'discharge_embeddings': ds_embeddings,
        'current_idx_padded': batch['current_idx_padded'].to(device),
        'current_phecode_len': batch['current_len'].to(device),
        'next_idx_padded': batch['next_idx_padded'].to(device),
        'next_phecode_len': batch['next_len'].to(device),
        'mortality_label': batch['mortality_label'].to(device, dtype=torch.float32),
        'readmission_label': batch['readmission_label'].to(device, dtype=torch.float32),
    }

def main(args):
    """Main function to generate embeddings from a trained model"""
    # Set up output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Get the device for computation
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Log key parameters
    logging.info(f"Starting embedding generation with:")
    logging.info(f"  Checkpoint path: {args.checkpoint_path}")
    logging.info(f"  Output directory: {output_dir}")
    logging.info(f"  Batch size: {args.batch_size}")
    
    # Define model dimensions - these must match what was used during training
    dims = {
        'variables_num': 80,
        'timestamps': 80,
        'd_static': 83,
        'ds_emb_dim': 768,
        'values_shape': (args.batch_size, 80, 80), # (batch_size, timestamps, variables_num)
        'phecode_size': 1788,
        'phecode_loss_weight': 0.2,
        'sensor_wise_mask': True
    }
    
    # Print dimensions that will be used
    logging.info("Using model dimensions:")
    for key, value in dims.items():
        logging.info(f"  {key}: {value}")
    
    # Load the model from checkpoint
    logging.info(f"Loading model from checkpoint: {args.checkpoint_path}")
    try:
        # Pass the dims parameter explicitly when loading the model
        model = RaindropContrastiveModel.load_from_checkpoint(args.checkpoint_path, dims=dims)
        model.eval()
        
        # Explicitly move model to the device
        model = model.to(device)
        logging.info(f"Model moved to {device}")
        
        # Verify that model components were initialized
        if model.ts_model is None or model.ds_encoder is None:
            logging.warning("Model components not initialized. Attempting initialization now.")
            if not hasattr(model, 'init_model'):
                raise RuntimeError("Model missing init_model method")
                
            # Print diagnostic information
            logging.info("Current model state:")
            logging.info(f"  dims: {model.dims}")
            logging.info(f"  args attributes: {[attr for attr in dir(model.args) if not attr.startswith('_')]}")
            
            try:
                # Force initialize model components with our provided dims
                model.init_model()
                logging.info("Successfully initialized model components after loading")
            except Exception as init_error:
                logging.error(f"Error during model initialization: {str(init_error)}")
                
                # Try to fix common issues
                required_attrs = ['d_model', 'hidden_dim', 'projection_dim', 'nlayers', 'num_heads', 'pooling_type']
                for attr in required_attrs:
                    if not hasattr(model.args, attr):
                        default_values = {
                            'd_model': 256, 
                            'hidden_dim': 256, 
                            'projection_dim': 256, 
                            'nlayers': 2, 
                            'num_heads': 2,
                            'pooling_type': 'attention'
                        }
                        setattr(model.args, attr, default_values[attr])
                        logging.info(f"Added missing attribute {attr}={default_values[attr]} to model.args")
                
                # Try initialization again
                model.init_model()
                logging.info("Successfully initialized model after fixing missing attributes")
        
        # Verify the model was properly initialized
        model_components = {
            'ts_model': model.ts_model,
            'ds_encoder': model.ds_encoder,
            'ts_projection': model.ts_projection,
            'text_projection': model.text_projection,
            'current_phecode_predictor': getattr(model, 'current_phecode_predictor', None)
        }
        
        logging.info("Model component verification:")
        for component_name, component in model_components.items():
            status = "OK" if component is not None else "MISSING"
            logging.info(f"  {component_name}: {status}")
            
        # Log model details
        logging.info(f"Model loaded successfully")
        logging.info(f"  Projection dimension: {getattr(model.args, 'projection_dim', 256)}")
        logging.info(f"  Use PHEcode loss: {model.use_phecode_loss}")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        logging.error("Please ensure the checkpoint file exists and contains a valid model")
        import traceback
        traceback.print_exc()
        raise
    
    # Initialize data module
    data_module = ContrastiveDataModule(
        data_path=args.data_path,
        temp_dfs_path=args.temp_dfs_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        task_mode='CONTRASTIVE',
    )
    
    # Prepare the data
    logging.info("Preparing data...")
    data_module.prepare_data()
    data_module.setup('fit')
    
    data_module.setup('fit')
    data_module.setup('test')

    # Create data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Log dataset sizes
    logging.info(f"Dataset sizes:")
    logging.info(f"  Train: {len(data_module.train_dataset)}")
    logging.info(f"  Validation: {len(data_module.val_dataset)}")
    logging.info(f"  Test: {len(data_module.test_dataset)}")
    
    # Generate embeddings for train, val, and test sets
    dataset_splits = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    for split_name, dataloader in dataset_splits.items():
        logging.info(f"Generating embeddings for {split_name} set...")
        
        # Get dataset size
        split_size = len(dataloader.dataset)
        projection_dim = model.args.projection_dim
        
        # Create arrays to store embeddings and labels
        # Use memmap for large datasets
        ts_proj_path = os.path.join(output_dir, f'{split_name}_ts_proj.npy')
        text_proj_path = os.path.join(output_dir, f'{split_name}_text_proj.npy')
        mortality_label_path = os.path.join(output_dir, f'{split_name}_mortality_label.npy')
        readmission_label_path = os.path.join(output_dir, f'{split_name}_readmission_label.npy')
        current_idx_path = os.path.join(output_dir, f'{split_name}_current_idx_padded.npy')
        current_len_path = os.path.join(output_dir, f'{split_name}_current_phecode_len.npy')
        next_idx_path = os.path.join(output_dir, f'{split_name}_next_idx_padded.npy')
        next_len_path = os.path.join(output_dir, f'{split_name}_next_phecode_len.npy')
        
        # Optional HADM ID storage if available
        hadm_id_available = hasattr(dataloader.dataset, 'hadm_ids')
        if hadm_id_available:
            hadm_id_path = os.path.join(output_dir, f'{split_name}_hadm_id.npy')
            hadm_id_array = np.memmap(hadm_id_path, dtype='int64', mode='w+', shape=(split_size,))
        
        # Get a sample batch to determine shapes
        sample_batch = next(iter(dataloader))
        next_idx_shape = sample_batch['next_idx_padded'].shape
        current_idx_shape = sample_batch['current_idx_padded'].shape
        
        # Create memory-mapped arrays
        ts_proj_array = np.memmap(ts_proj_path, dtype='float32', mode='w+', 
                                 shape=(split_size, projection_dim))
        text_proj_array = np.memmap(text_proj_path, dtype='float32', mode='w+', 
                                   shape=(split_size, projection_dim))
        mortality_label_array = np.memmap(mortality_label_path, dtype='float32', mode='w+', 
                                        shape=(split_size,))
        readmission_label_array = np.memmap(readmission_label_path, dtype='float32', mode='w+', 
                                          shape=(split_size,))
        current_idx_array = np.memmap(current_idx_path, dtype='int64', mode='w+', 
                                     shape=(split_size, current_idx_shape[1]))
        current_len_array = np.memmap(current_len_path, dtype='int64', mode='w+', 
                                     shape=(split_size,))
        next_idx_array = np.memmap(next_idx_path, dtype='int64', mode='w+', 
                                  shape=(split_size, next_idx_shape[1]))
        next_len_array = np.memmap(next_len_path, dtype='int64', mode='w+', 
                                  shape=(split_size,))
        
        # Process data in batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                # Instead of setting model.device, we'll move batch data to the device manually
                # Then prepare the batch data
                try:
                    # Use our custom prepare_batch function instead of model.prepare_batch
                    batch_data = prepare_batch_for_inference(batch, device)
                    if batch_data is None:
                        continue
                    
                    # Forward pass through model to get embeddings
                    ts_proj, text_proj, _ = model.model_forward(batch_data)
                    
                    # Calculate start index for this batch
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = start_idx + ts_proj.shape[0]
                    if end_idx > split_size:
                        end_idx = split_size  # Handle last batch which might be smaller
                    
                    # Store results directly in memory-mapped arrays
                    ts_proj_array[start_idx:end_idx] = ts_proj.cpu().numpy()
                    text_proj_array[start_idx:end_idx] = text_proj.cpu().numpy()
                    mortality_label_array[start_idx:end_idx] = batch_data['mortality_label'].cpu().numpy()
                    readmission_label_array[start_idx:end_idx] = batch_data['readmission_label'].cpu().numpy()
                    current_idx_array[start_idx:end_idx] = batch_data['current_idx_padded'].cpu().numpy()
                    current_len_array[start_idx:end_idx] = batch_data['current_phecode_len'].cpu().numpy()
                    next_idx_array[start_idx:end_idx] = batch_data['next_idx_padded'].cpu().numpy()
                    next_len_array[start_idx:end_idx] = batch_data['next_phecode_len'].cpu().numpy()
                    
                    # Store HADM IDs if available
                    if hadm_id_available:
                        hadm_id_tensor = batch.get('hadm_id', None)
                        if hadm_id_tensor is not None:
                            hadm_id_array[start_idx:end_idx] = hadm_id_tensor.cpu().numpy()
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
        
        # Save metadata
        metadata = {
            'split': split_name,
            'size': split_size,
            'projection_dim': projection_dim,
            'model_checkpoint': args.checkpoint_path,
            'use_phecode_loss': model.use_phecode_loss,
            'phecode_size': model.phe_code_size,
            'hadm_id_available': hadm_id_available,
            'device_used': str(device),
            'files': {
                'ts_proj': ts_proj_path,
                'text_proj': text_proj_path,
                'mortality_label': mortality_label_path,
                'readmission_label': readmission_label_path,
                'current_idx_padded': current_idx_path,
                'current_phecode_len': current_len_path,
                'next_idx_padded': next_idx_path,
                'next_phecode_len': next_len_path
            }
        }
        if hadm_id_available:
            metadata['files']['hadm_id'] = hadm_id_path
        
        # Save the metadata
        with open(os.path.join(output_dir, f'{split_name}_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Flush memory-mapped arrays to disk
        ts_proj_array.flush()
        text_proj_array.flush()
        mortality_label_array.flush()
        readmission_label_array.flush()
        current_idx_array.flush()
        current_len_array.flush()
        next_idx_array.flush()
        next_len_array.flush()
        if hadm_id_available:
            hadm_id_array.flush()
        
        logging.info(f"Completed generating embeddings for {split_name} set")
        logging.info(f"Saved to {output_dir}")
        
        # Clean up memory-mapped arrays
        del ts_proj_array, text_proj_array, mortality_label_array, readmission_label_array
        del current_idx_array, current_len_array, next_idx_array, next_len_array
        if hadm_id_available:
            del hadm_id_array
        
    logging.info("Embedding generation complete!")

if __name__ == "__main__":
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Generate embeddings from trained contrastive model')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, default = '/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/hyperparameter_sweep/checkpoints/bs128_lr0.0005_seed42_proj256_temp0.07_phe/contrastive_raindrop_v2_epoch=07-train_contrastive_loss=1.7256.ckpt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default = '/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/temp_dfs_lite/FULL_CONTRASTIVE_EMBEDDINGS',
                        help='Directory to save embeddings and metadata')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='', 
                        help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite', 
                        help='Path to cache directory')
    
    # Processing arguments
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args) 