import os
import torch
from torch.utils.data import DataLoader
from data_scripts.data_lite import MIMICContrastivePairsDatasetLite
from dataloader_lite import custom_collate_fn

def test_dataloader(cache_dir='./cache', batch_size=4):
    # Initialize the dataset
    print(f"Initializing dataset from {cache_dir}...")
    dataset = MIMICContrastivePairsDatasetLite(
        split='train',
        cache_dir=cache_dir,
        task_mode='CONTRASTIVE'
    )
    
    # Create DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=custom_collate_fn
    )
    
    # Get a single batch and examine its contents
    print(f"Fetching a batch of size {batch_size}...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx+1}:")
        
        # Print keys in the batch
        print(f"Batch keys: {list(batch.keys())}")
        
        # Print shapes of main tensors
        for key in batch:
            if key == 'ds_embedding':
                # DS embeddings are a list of variable-sized tensors
                print(f"DS embedding: [list of {len(batch[key])} tensors]")
                # Print shape of each embedding tensor
                for i, emb in enumerate(batch[key]):
                    if emb is not None:
                        print(f"  - DS embedding {i} shape: {emb.shape}")
                    else:
                        print(f"  - DS embedding {i}: None")
            elif torch.is_tensor(batch[key]):
                print(f"{key} shape: {batch[key].shape}")
            else:
                print(f"{key}: {batch[key]}")
        
        # Check if any hadm_id is missing a DS embedding
        hadm_ids = batch['hadm_id']
        for i, hadm_id in enumerate(hadm_ids):
            ds_emb = batch['ds_embedding'][i]
            if ds_emb is None:
                print(f"Warning: hadm_id {hadm_id} has no DS embedding")
            else:
                # Verify embedding is properly loaded
                print(f"hadm_id {hadm_id} has DS embedding with {ds_emb.shape[0]} chunks")
        
        # Only process one batch for this test
        break
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    # Update the cache_dir path to point to your data location
    test_dataloader(cache_dir='temp_dfs', batch_size=4)