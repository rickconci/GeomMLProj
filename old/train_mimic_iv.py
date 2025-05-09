# -*- coding:utf-8 -*-
import os
import argparse
import warnings
import time
import wandb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import torch
import torch.nn.functional as F
import numpy as np
import dotenv
from GeomMLProj.train_utils import contrastive_loss, count_parameters, detailed_count_parameters, plot_first_sample, plot_mask_heatmap
from models.main_models import KEDGN, DSEncoderWithRNN
from models.models_rd import Raindrop_v2
from models.models_utils import ProjectionHead
from tqdm import tqdm 
import json
from pathlib import Path
from GeomMLProj.train_utils import device, log_batch_metrics

from dataloader_lite import get_dataloaders

DEBUG_PRINTS = False
def debug_print(*args, **kwargs):
        if DEBUG_PRINTS:
            print(*args, **kwargs)

# Create a wrapper class for Raindrop_v2 to make it compatible with our training pipeline
class RaindropModel(torch.nn.Module):
    def __init__(self, DEVICE, d_inp, d_model, nhead, nhid, nlayers, dropout, max_len, 
                 d_static, n_class, global_structure=None, sensor_wise_mask=False, static=True):
        super().__init__()
        
        self.model = Raindrop_v2(
            d_inp=d_inp,
            d_model=d_model,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=dropout,
            max_len=max_len,
            d_static=d_static,
            n_classes=n_class,
            global_structure=global_structure,
            sensor_wise_mask=sensor_wise_mask,
            static=static
        )
        
        self.DEVICE = DEVICE
        
    def forward(self, P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor):
        """
        Wrapper method to maintain compatibility with KEDGN's input format
        
        Args:
            P: Combined values and mask tensor [B, T, F*2]
            P_static: Static features [B, S]
            P_avg_interval: Average interval tensor (not used) 
            P_length: Sequence lengths [B, 1]
            P_time: Timestamps [B, T]
            P_var_plm_rep_tensor: Variable embeddings (not used)
            
        Returns:
            Tuple of (output, distance, None) to match KEDGN's return format
        """
        # Convert inputs to the format expected by Raindrop_v2
        src = P.permute(1, 0, 2)  # [T, B, F*2]
        static = P_static
        times = P_time.permute(1, 0)  # [T, B]
        lengths = P_length.squeeze(1)  # [B]
        
        # Forward pass through Raindrop_v2
        output, distance, _ = self.model(src, static, times, lengths)
        
        # Return in a format compatible with how we process KEDGN outputs
        return output, output, output  # Using output three times to match (outputs, aggregated_hidden, fused_features)
            
def main(args):
    print(args)

    PROGRESS_PRINT_FREQUENCY = 1  # Print progress every N epochs

    import logging
    logging.basicConfig(level=logging.ERROR)  # Set to ERROR to suppress most logs

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Initialize wandb if enabled
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
        wandb.config.update({"device": str(device)})

    # Create model save path
    model_path = './models/'
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load command line hyperparameters
    dataset = args.dataset
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    hidden_dim = args.hidden_dim
    projection_dim = args.projection_dim
    rarity_alpha = args.rarity_alpha
    query_vector_dim = args.query_vector_dim
    node_emb_dim = args.node_emb_dim
    plm_rep_dim = args.plm_rep_dim
    source = args.source


    print('Dataset used: ', dataset)

    # Evaluation metrics
    acc_arr = []
    auprc_arr = []
    auroc_arr = []

    # Run five experiments
    for k in range(5):

        # Set different random seed for each run
        torch.manual_seed(k)
        torch.cuda.manual_seed(k)
        np.random.seed(k)

        train_loader, val_loader, test_loader, P_var_plm_rep_tensor = get_dataloaders(
                    data_path=args.data_path,
                    temp_dfs_path=args.temp_dfs_path,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    task_mode=args.task_mode,
                )
        

        P_var_plm_rep_tensor = P_var_plm_rep_tensor.to(device)

        
        valid_batch_found = False
        max_attempts = 10 
        attempt = 0
        
        train_iter = iter(train_loader)
        while not valid_batch_found and attempt < max_attempts:
            try:
                attempt += 1
                first_batch = next(train_iter)
                
                # Check if first_batch is empty (all samples were None)
                if not first_batch:
                    print(f"Warning: Batch {attempt} is empty - all samples were filtered out. Trying next batch...")
                    continue
                    
                # Try to access the required keys to verify this batch is usable
                values = first_batch['values']
                mask = first_batch['mask']
                static = first_batch['static']
                times = first_batch['times']
                length = first_batch['length']
                labels = first_batch['label']
                
                # If we get here without errors, we have a valid batch
                valid_batch_found = True
                print(f"Found valid batch after {attempt} attempts")
                
            except StopIteration:
                print(f"Warning: Reached end of dataloader after {attempt} attempts without finding a valid batch.")
                break
            except KeyError as e:
                print(f"Warning: Batch {attempt} missing key {e}. Trying next batch...")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error in batch {attempt}: {e}. Trying next batch...")
                continue
        
        if not valid_batch_found:
            print("Error: Could not find a valid batch after multiple attempts.")
            raise ValueError("No valid batches found - check your cache directory and dataset.")

        d_static = static.shape[1]
        variables_num = values.shape[2]
        actual_ts_dim = values.shape[1]    

        # Handle n_class differently based on task mode
        if args.task_mode == 'NEXT_24h':
            n_class = 1
        else:
            n_class = labels.shape[1]  # For other modes, get from data

        # Check if P_var_plm_rep_tensor matches variables_num and resize if needed
        if P_var_plm_rep_tensor.shape[0] != variables_num:
            print(f"WARNING: P_var_plm_rep_tensor dimension ({P_var_plm_rep_tensor.shape[0]}) doesn't match variables_num ({variables_num})")
            if P_var_plm_rep_tensor.shape[0] > variables_num:
                # Truncate
                print(f"Truncating P_var_plm_rep_tensor from {P_var_plm_rep_tensor.shape} to [{variables_num}, {P_var_plm_rep_tensor.shape[1]}]")
                P_var_plm_rep_tensor = P_var_plm_rep_tensor[:variables_num, :]
            else:
                # Pad with zeros
                print(f"Padding P_var_plm_rep_tensor from {P_var_plm_rep_tensor.shape} to [{variables_num}, {P_var_plm_rep_tensor.shape[1]}]")
                padding = torch.zeros((variables_num - P_var_plm_rep_tensor.shape[0], P_var_plm_rep_tensor.shape[1]), device=device)
                P_var_plm_rep_tensor = torch.cat([P_var_plm_rep_tensor, padding], dim=0)

        print(f"d_static: {d_static}, n_class: {n_class}, variables_num: {variables_num}, actual_ts_dim: {actual_ts_dim}, P_var_plm_rep_tensor: {P_var_plm_rep_tensor.shape}, labels: {labels.shape}")

        # Add debug output for model initialization
        debug_print("Initializing model...")
        start_time = time.time()
        
        # Load global structure for Raindrop if provided
        global_structure = None
        if args.model_type == 'raindrop_v2' and args.global_structure_path:
            if os.path.exists(args.global_structure_path):
                try:
                    global_structure = torch.load(args.global_structure_path)
                    print(f"Loaded global structure with shape {global_structure.shape}")
                except Exception as e:
                    print(f"Error loading global structure: {e}")
                    print("Initializing with default fully-connected structure")
                    global_structure = torch.ones(variables_num, variables_num)
            else:
                print(f"Global structure file {args.global_structure_path} not found")
                print("Initializing with default fully-connected structure")
                global_structure = torch.ones(variables_num, variables_num)
        
        # Initialize model based on selected type
        if args.model_type == 'raindrop_v2':
            # For Raindrop, d_inp should be half of the input size since it expects the second half to be masks
            model = RaindropModel(
                DEVICE=device,
                d_inp=variables_num // 2,  # Raindrop expects d_inp to be half of the input size
                d_model=args.d_model, 
                nhead=args.num_heads, 
                nhid=args.hidden_dim, 
                nlayers=args.nlayers, 
                dropout=0.3, 
                max_len=actual_ts_dim, 
                d_static=d_static,
                n_class=n_class,
                global_structure=global_structure,
                sensor_wise_mask=args.sensor_wise_mask,
                static=(d_static > 0)
            )
            print(f"Initialized Raindrop_v2 model with {variables_num // 2} input features")
        else:  # Default to KEDGN
            # Initialize KEDGN model
            model = KEDGN(DEVICE=device,
                        hidden_dim=hidden_dim,
                        num_of_variables=variables_num,
                        num_of_timestamps=actual_ts_dim,
                        d_static=d_static,
                        n_class=n_class,
                        rarity_alpha=rarity_alpha,
                        query_vector_dim=query_vector_dim,
                        node_emb_dim=node_emb_dim,
                        plm_rep_dim=plm_rep_dim,
                        use_gat=args.use_gat,
                        num_heads=args.num_heads,
                        use_adj_mask=args.use_adj_mask,
                        use_clusters=args.use_clusters,
                        task_mode=args.task_mode)
            print(f"Initialized KEDGN model with {variables_num} input features")
        
        debug_print(f"Model initialized in {time.time() - start_time:.2f} seconds")
        
        # If using contrastive learning, add text encoder and projection heads
        if args.task_mode == 'CONTRASTIVE':
            # Re-add Text encoder initialization
            DS_encoder_projector = DSEncoderWithRNN(rnn_hidden_dim = hidden_dim, projection_dim = projection_dim).to(device)  
        
            model.text_encoder = DS_encoder_projector
            model.temperature = args.temperature
            model.similarity_metric = args.similarity_metric

        print('model parameters:', count_parameters(model))
        
        # Add detailed parameter counting
        param_details = detailed_count_parameters(model)
        print("\nDetailed model parameters breakdown:")
        for module_name, param_count in param_details.items():
            if module_name != 'total':  # Skip total since we already print it
                print(f"{module_name}: {param_count:,} ({param_count/param_details['total']*100:.1f}%)")

        print('setting loss criterion')
        if args.task_mode != 'CONTRASTIVE':
            classification_criterion = torch.nn.CrossEntropyLoss().to(device)
        if args.task_mode == 'NEXT_24h':
            num_pos = sum(labels)
            num_neg = len(labels) - num_pos
            orig_ratio = num_neg / num_pos   
            alpha = 0.5
            pos_weight = torch.tensor(orig_ratio**alpha).to(device)  
            classification_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        
        print('setting up optimiser')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print('setting up checkpointing')
        # Add checkpoint loading code before the training loop starts
        # After model initialization and before the training loop starts
        if args.resume_from_checkpoint:
            checkpoint_path = args.resume_from_checkpoint
            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_epoch = checkpoint['best_val_epoch']
                best_aupr_val = checkpoint.get('best_aupr_val', 0.0)
                best_auc_val = checkpoint.get('best_auc_val', 0.0)
                best_loss_val = checkpoint.get('best_loss_val', float('inf'))
                save_time = checkpoint.get('save_time', str(int(time.time())))
                print(f"Resuming from epoch {start_epoch} with best validation AUPRC {best_aupr_val:.4f} from epoch {best_val_epoch}")
            else:
                print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
                start_epoch = 0
        else:
            start_epoch = 0

        # Replace the outer training loop with this version that includes early stopping
        start = time.time()

        print('initialising early stopping')
        # Initialize early stopping variables
        no_improvement_count = 0
        early_stop = False
        best_val_epoch = 0
        best_aupr_val = best_auc_val = 0.0
        best_loss_val = float('inf')
        save_time = str(int(time.time()))
        model_saved = False

        running = {
            'loss':0.0, 'TP':0, 'TN':0, 'FP':0, 'FN':0, 'seen':0,
            'probs':[], 'labs':[]
        }
        global_step = 0
        log_every   = 15


        
        print('starting training')
        for epoch in range(start_epoch, num_epochs):
            if early_stop and args.early_stopping:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            """Training"""
            model.train()
            train_loss = 0.0
            train_probs_all = []
            train_labels_all = []

            start_time = time.time()
            # Add tqdm progress bar for training
            print('initializing training iterator')
            train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), 
                                desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
            debug_print('training iterator initialized in', time.time() - start_time)

            for batch_idx, batch in train_iterator:
                # Get data from batch and move to device
                start_batch_time = time.time()
                values = batch['values'].to(device)
                mask = batch['mask'].to(device)
                static = batch['static'].to(device) if d_static > 0 else None
                times = batch['times'].to(device)
                length = batch['length'].to(device)
                labels = batch['label'].to(device)
                
                # Create input format expected by KEDGN
                P = torch.cat([values, mask], dim=2)  # Shape [B, T, F*2]
                
                # Handle the time tensor properly based on its dimensions
                if len(times.shape) == 2:  # Shape [B, T]
                    P_time = times  # Keep as is
                else:  # Just a single dimension [T]
                    P_time = times.unsqueeze(0).repeat(values.size(0), 1)  # Shape [B, T]
                
                # P_avg_interval should have the same shape as P_time, but needs to be expanded
                # for the variables dimension in the rarity calculation
                P_avg_interval = torch.ones_like(P_time)  # Shape [B, T]
                
                # Expand P_avg_interval to match the number of variables
                # This avoids dimension mismatch in rarity score calculation
                P_avg_interval = P_avg_interval.unsqueeze(2).expand(-1, -1, variables_num)  # Shape [B, T, N]
                
                P_length = length.unsqueeze(1)  # Shape [B, 1]
                batch_computation_time = time.time() - start_batch_time
                debug_print('batch reshaping time:', batch_computation_time)
                
                # Forward pass through the model
                start_forward_time = time.time()
                if args.task_mode == 'CONTRASTIVE':
                    # Get discharge text chunks (not precomputed embeddings)
                    discharge_embeddings = batch['discharge_embeddings']
                    
                    # Forward pass to get base KEDGN representations
                    # The KEDGN model now returns (output, aggregated_hidden, fused_features)
                    _, _, ts_intermediate_rep = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                    
                    # Encode text chunks using a text encoder
                    text_embeddings = model.text_encoder(discharge_chunks, output_dim=hidden_dim*2)
                    
                    # Project both time series and text embeddings
                    ts_proj = model.ts_projection(ts_intermediate_rep)
                    text_proj = model.text_projection(text_embeddings)
                    
                    # Calculate contrastive loss
                    loss = contrastive_loss(
                        ts_proj, 
                        text_proj, 
                        method=args.contrastive_method,
                        temperature=args.temperature
                    )
                    
                    # No classification probabilities for contrastive learning
                    probs = None
                else:
                    # Standard classification forward pass - unpack tuple returned by model
                    outputs, _, _ = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                    forward_time = time.time() - start_forward_time
                    debug_print('forward pass time:', forward_time)
                    
                    if args.task_mode == 'NEXT_24h':
                        # For NEXT_24h, ensure outputs are of shape [B, 1] to match labels
                        # If outputs are [B, 2], take only the positive class logits
                        if outputs.shape[1] == 2:
                            outputs = outputs[:, 1].unsqueeze(1)  # Get positive class logits
                        
                        # Reshape labels if they're scalars
                        if len(labels.shape) == 1:
                            labels = labels.float().unsqueeze(1)  # Convert to shape [B, 1]
                        
                        # Compute classification loss - labels should be float for BCEWithLogitsLoss
                        loss = classification_criterion(outputs, labels.float())
                        # Calculate probabilities
                        probs = torch.sigmoid(outputs)
                        # For binary classification with BCEWithLogitsLoss, create 2-column tensor for metrics
                        probs = torch.cat([1-probs, probs], dim=1)
                        train_probs_all.append(probs.detach().cpu())
                    else:
                        # Compute classification loss
                        loss = classification_criterion(outputs, labels.squeeze(1).long())
                        # Store probabilities for metrics
                        probs = torch.softmax(outputs, dim=1)
                        train_probs_all.append(probs.detach().cpu())
                    
                    train_labels_all.append(labels.detach().cpu())
                
                backprop_start_time = time.time()
                # Update model parameters
                optimizer.zero_grad()
                loss.backward()
                backprop_time = time.time() - backprop_start_time
                debug_print('backprop time:', backprop_time)
                
                # Add graph distance regularization for Raindrop_v2
                if args.model_type == 'raindrop_v2' and isinstance(model, RaindropModel) and hasattr(outputs, 'distance') and outputs.distance is not None:
                    graph_distance_factor = 0.1  # Hyperparameter for graph distance regularization
                    graph_loss = outputs.distance * graph_distance_factor
                    loss += graph_loss
                
                # Add gradient clipping to prevent exploding gradients and NaN values
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                start_optim_time = time.time()
                optimizer.step()
                optim_time = time.time() - start_optim_time
                debug_print('optim time:', optim_time)
                
                train_loss += loss.item()

                running['loss'] += loss.item()
                preds = torch.argmax(probs, dim=1).cpu()
                labs  = labels.view(-1).long().cpu()
                running['TP']   += int(((preds == 1) & (labs == 1)).sum())
                running['TN']   += int(((preds == 0) & (labs == 0)).sum())
                running['FP']   += int(((preds == 1) & (labs == 0)).sum())
                running['FN']   += int(((preds == 0) & (labs == 1)).sum())
                running['seen'] += labs.size(0)
                running['probs'].extend(probs[:,1].cpu().tolist())
                running['labs'].extend(labs.tolist())
                global_step     += 1
            
                # call the helper
                log_batch_metrics(batch_idx, global_step, running, log_every, as_percentage=True)

   
                train_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
            train_loss /= len(train_loader)
            

            if args.task_mode != 'CONTRASTIVE':
                train_probs = torch.cat(train_probs_all, dim=0).numpy()
                train_labels = torch.cat(train_labels_all, dim=0).numpy().squeeze()
                train_auroc = roc_auc_score(train_labels, train_probs[:, 1])
                train_auprc = average_precision_score(train_labels, train_probs[:, 1])
            else:
                train_auroc = 0.0
                train_auprc = 0.0

            """Validation"""
            model.eval()
            val_loss = 0.0
            val_probs_all = []
            val_labels_all = []
            
            with torch.no_grad():
                val_iterator = tqdm(enumerate(val_loader), total=len(val_loader), 
                                desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
                
                for batch_idx, batch in val_iterator:
                    # Get data from batch and move to device
                    values = batch['values'].to(device)
                    mask = batch['mask'].to(device)
                    static = batch['static'].to(device) if d_static > 0 else None
                    times = batch['times'].to(device)
                    length = batch['length'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Create input format expected by KEDGN
                    P = torch.cat([values, mask], dim=2)
                    
                    # Handle the time tensor properly based on its dimensions
                    if len(times.shape) == 2:  # Shape [B, T]
                        P_time = times  # Keep as is
                    else:  # Just a single dimension [T]
                        P_time = times.unsqueeze(0).repeat(values.size(0), 1)  # Shape [B, T]
                    
                    # P_avg_interval should have the same shape as P_time, but needs to be expanded
                    # for the variables dimension in the rarity calculation
                    P_avg_interval = torch.ones_like(P_time)  # Shape [B, T]
                    
                    # Expand P_avg_interval to match the number of variables
                    # This avoids dimension mismatch in rarity score calculation
                    P_avg_interval = P_avg_interval.unsqueeze(2).expand(-1, -1, variables_num)  # Shape [B, T, N]
                    
                    P_length = length.unsqueeze(1)  # Shape [B, 1]
                    
                    # Forward pass
                    if args.task_mode == 'CONTRASTIVE':
                        # Get precomputed discharge summary embeddings
                        text_embeddings = batch['ds_embedding'].to(device)
                        
                        # Forward pass to get base KEDGN representations
                        # Use fused_features for the TS representation
                        _, _, ts_intermediate_rep = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                        
                        # Text embeddings are precomputed
                        # text_embeddings = model.text_encoder(discharge_chunks, output_dim=hidden_dim*2)
                        
                        # Project both time series and text embeddings
                        ts_proj = model.ts_projection(ts_intermediate_rep)
                        text_proj = model.text_projection(text_embeddings)
                        
                        # Calculate contrastive loss for validation (optional, usually classification metrics are preferred)
                        loss = contrastive_loss(
                            ts_proj, 
                            text_proj, 
                            method=args.contrastive_method,
                            temperature=args.temperature
                        )
                        
                        # For validation in contrastive mode, evaluate classification performance using the learned TS representation
                        # This requires a temporary linear classifier head
                        if not hasattr(model, 'contrastive_val_classifier'):
                             model.contrastive_val_classifier = torch.nn.Linear(ts_intermediate_rep.shape[-1], n_class).to(device)
                        outputs = model.contrastive_val_classifier(ts_intermediate_rep)
                        probs = torch.softmax(outputs, dim=1)
                    else:
                        # Standard classification forward pass - unpack tuple returned by model
                        outputs, _, _ = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                        
                        if args.task_mode == 'NEXT_24h':
                            # For NEXT_24h, ensure outputs are of shape [B, 1] to match labels
                            # If outputs are [B, 2], take only the positive class logits
                            if outputs.shape[1] == 2:
                                outputs = outputs[:, 1].unsqueeze(1)  # Get positive class logits
                                
                            # Reshape labels if they're scalars
                            if len(labels.shape) == 1:
                                labels = labels.float().unsqueeze(1)  # Convert to shape [B, 1]
                                
                            # Compute classification loss - labels should be float for BCEWithLogitsLoss
                            loss = classification_criterion(outputs, labels.float())
                            # Calculate probabilities
                            probs = torch.sigmoid(outputs)
                            # For binary classification with BCEWithLogitsLoss, create 2-column tensor for metrics
                            probs = torch.cat([1-probs, probs], dim=1)
                        else:
                            # Compute classification loss
                            loss = classification_criterion(outputs, labels.squeeze(1).long())
                            # Calculate probabilities
                            probs = torch.softmax(outputs, dim=1)
                    
                    val_loss += loss.item()
                    val_iterator.set_postfix(loss=f"{loss.item():.4f}")
                    
                    # Store probabilities and labels for metrics calculation
                    val_probs_all.append(probs.detach().cpu())
                    val_labels_all.append(labels.detach().cpu())
                
                # Calculate validation metrics
                val_probs = torch.cat(val_probs_all, dim=0).numpy()
                val_labels = torch.cat(val_labels_all, dim=0).numpy().squeeze()
                val_auroc = roc_auc_score(val_labels, val_probs[:, 1])
                val_auprc = average_precision_score(val_labels, val_probs[:, 1])
                val_preds = np.argmax(val_probs, axis=1)
                val_acc = np.mean(val_preds == val_labels)
                val_loss /= len(val_loader)
                
                # After computing validation metrics, create an organized checkpoint filename
                checkpoint_base = f"{dataset}_{args.task_mode}_{args.model_type}" 
                if args.model_type == 'kedgn' and args.use_gat:
                    checkpoint_base += f"_GAT{args.num_heads}"
                elif args.model_type == 'kedgn':
                    checkpoint_base += "_GCN"
                elif args.model_type == 'raindrop_v2':
                    checkpoint_base += f"_L{args.nlayers}_H{args.num_heads}"
                checkpoint_base += f"_run{k}"
                
                # Early stopping logic based on selected metric
                current_metric = 0.0
                is_better = False
                
                if args.metric_for_best_model == 'loss':
                    current_metric = val_loss
                    is_better = current_metric < best_loss_val
                elif args.metric_for_best_model == 'auprc':
                    current_metric = val_auprc
                    is_better = current_metric > best_aupr_val
                elif args.metric_for_best_model == 'auroc':
                    current_metric = val_auroc
                    is_better = current_metric > best_auc_val
                
                # Create checkpoint containing all necessary information
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auprc': val_auprc,
                    'val_auroc': val_auroc,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_auprc': train_auprc,
                    'train_auroc': train_auroc,
                    'best_val_epoch': best_val_epoch,
                    'best_aupr_val': best_aupr_val,
                    'best_auc_val': best_auc_val,
                    'best_loss_val': best_loss_val,
                    'save_time': save_time,
                    'args': vars(args)
                }
                
                # Save regular checkpoints if requested
                if args.save_all_checkpoints:
                    checkpoint_filename = os.path.join(checkpoint_dir, f"{checkpoint_base}_epoch{epoch}.pt")
                    torch.save(checkpoint, checkpoint_filename)
                
                # Save best model based on selected metric
                if is_better:
                    if args.metric_for_best_model == 'loss':
                        best_loss_val = val_loss
                    elif args.metric_for_best_model == 'auprc':
                        best_aupr_val = val_auprc
                    elif args.metric_for_best_model == 'auroc':
                        best_auc_val = val_auroc
                        
                    best_val_epoch = epoch
                    save_time = str(int(time.time()))
                    
                    # Save best model in both formats
                    model_path_full = model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt'
                    checkpoint_filename = os.path.join(checkpoint_dir, f"{checkpoint_base}_best.pt")
                    
                    # Save state dict for backward compatibility
                    torch.save(model.state_dict(), model_path_full)
                    
                    # Save full checkpoint with metadata
                    torch.save(checkpoint, checkpoint_filename)
                    
                    model_saved = True
                    no_improvement_count = 0
                    
                    # Save metrics JSON for easy access without loading model
                    metrics_filename = os.path.join(checkpoint_dir, f"{checkpoint_base}_best_metrics.json")
                    metrics_data = {
                        'epoch': epoch,
                        'val_loss': float(val_loss),
                        'val_auprc': float(val_auprc),
                        'val_auroc': float(val_auroc),
                        'val_acc': float(val_acc),
                        'train_loss': float(train_loss),
                        'train_auprc': float(train_auprc),
                        'train_auroc': float(train_auroc)
                    }
                    with open(metrics_filename, 'w') as f:
                        json.dump(metrics_data, f, indent=2)
                    
                    # Only print model save notification when we're printing progress
                    if (epoch + 1) % PROGRESS_PRINT_FREQUENCY == 0 or epoch == num_epochs - 1:
                        print(f"Saved checkpoint to {checkpoint_filename}")
                        print(f"Saved best model to {model_path_full}")
                else:
                    no_improvement_count += 1
                
                # Early stopping check
                if args.early_stopping and no_improvement_count >= args.patience:
                    print(f"Early stopping triggered: no improvement for {args.patience} epochs")
                    early_stop = True
                
                # Only print progress at specified frequency or on the last epoch
                if (epoch + 1) % PROGRESS_PRINT_FREQUENCY == 0 or epoch == num_epochs - 1 or early_stop:
                    print(
                        "Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                        (epoch, train_loss, train_auprc * 100, train_auroc * 100,
                        val_loss, val_acc * 100, val_auprc * 100, val_auroc * 100))
                    if args.early_stopping:
                        print(f"Early stopping counter: {no_improvement_count}/{args.patience}")

        end = time.time()
        time_elapsed = end - start
        print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

        """Testing"""
        model.eval()
        if model_saved:
            # Try to load best checkpoint first, then fall back to older method
            checkpoint_base = f"{dataset}_{args.task_mode}_{args.model_type}" 
            if args.model_type == 'kedgn' and args.use_gat:
                checkpoint_base += f"_GAT{args.num_heads}"
            elif args.model_type == 'kedgn':
                checkpoint_base += "_GCN"
            elif args.model_type == 'raindrop_v2':
                checkpoint_base += f"_L{args.nlayers}_H{args.num_heads}"
            checkpoint_base += f"_run{k}"
            checkpoint_filename = os.path.join(checkpoint_dir, f"{checkpoint_base}_best.pt")
            
            if os.path.exists(checkpoint_filename):
                print(f"Loading best checkpoint from {checkpoint_filename}")
                checkpoint = torch.load(checkpoint_filename, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint['epoch']} with val {args.metric_for_best_model} = {checkpoint[f'val_{args.metric_for_best_model}']:.4f}")
            else:
                # Fall back to original method
                model_path_full = model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt'
                print(f"Checkpoint not found. Loading best model from {model_path_full}")
                model.load_state_dict(torch.load(model_path_full, map_location=device))
        else:
            print("No model was saved during training. Using the final model state for testing.")
        
        # Print model specific information
        if args.model_type == 'raindrop_v2':
            print(f"Testing Raindrop_v2 model with {args.nlayers} layers and {args.num_heads} attention heads")
        else:
            print(f"Testing KEDGN model with GAT={args.use_gat}, num_heads={args.num_heads}")
        
        test_probs_all = []
        test_labels_all = []
        
        with torch.no_grad():
            test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), 
                                desc=f"Run {k+1}/5 [Test]", leave=False)
            
            for batch_idx, batch in test_iterator:
                # Get data from batch and move to device
                values = batch['values'].to(device)
                mask = batch['mask'].to(device)
                static = batch['static'].to(device) if d_static > 0 else None
                times = batch['times'].to(device)
                length = batch['length'].to(device)
                labels = batch['label'].to(device)
                
                # Create input format expected by KEDGN
                P = torch.cat([values, mask], dim=2)
                
                # Handle the time tensor properly based on its dimensions
                if len(times.shape) == 2:  # Shape [B, T]
                    P_time = times  # Keep as is
                else:  # Just a single dimension [T]
                    P_time = times.unsqueeze(0).repeat(values.size(0), 1)  # Shape [B, T]
                    
                # P_avg_interval should have the same shape as P_time, but needs to be expanded
                # for the variables dimension in the rarity calculation
                P_avg_interval = torch.ones_like(P_time)  # Shape [B, T]
                
                # Expand P_avg_interval to match the number of variables
                # This avoids dimension mismatch in rarity score calculation
                P_avg_interval = P_avg_interval.unsqueeze(2).expand(-1, -1, variables_num)  # Shape [B, T, N]
                
                P_length = length.unsqueeze(1)  # Shape [B, 1]
                
                # Forward pass
                if args.use_contrastive or args.task_mode == 'CONTRASTIVE':
                    # Get discharge text chunks for contrastive learning
                    discharge_chunks = batch['discharge_chunks']
                    
                    # Forward pass to get base KEDGN representations
                    # The KEDGN model now returns (output, aggregated_hidden, fused_features)
                    _, _, ts_intermediate_rep = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                    
                    # For testing with contrastive learning, we evaluate using a linear classifier
                    linear_classifier = torch.nn.Linear(hidden_dim*2, n_class).to(device)
                    outputs = linear_classifier(ts_intermediate_rep)
                else:
                    # Standard classification forward pass - unpack tuple returned by model
                    outputs, _, _ = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                
                # Calculate probabilities
                if args.task_mode == 'NEXT_24h':
                    # For NEXT_24h, ensure outputs are of shape [B, 1] to match labels
                    # If outputs are [B, 2], take only the positive class logits
                    if outputs.shape[1] == 2:
                        outputs = outputs[:, 1].unsqueeze(1)  # Get positive class logits
                        
                    # Reshape labels if they're scalars
                    if len(labels.shape) == 1:
                        labels = labels.float().unsqueeze(1)  # Convert to shape [B, 1]
                    
                    # Compute classification loss - labels should be float for BCEWithLogitsLoss
                    loss = classification_criterion(outputs, labels.float())
                    # Calculate probabilities
                    probs = torch.sigmoid(outputs)
                    # For binary classification with BCEWithLogitsLoss, create 2-column tensor for metrics
                    probs = torch.cat([1-probs, probs], dim=1)
                else:
                    probs = torch.softmax(outputs, dim=1)
                
                test_probs_all.append(probs.detach().cpu())
                test_labels_all.append(labels.detach().cpu())
            
            # Calculate test metrics
            test_probs = torch.cat(test_probs_all, dim=0).numpy()
            test_labels = torch.cat(test_labels_all, dim=0).numpy().squeeze()
            test_auroc = roc_auc_score(test_labels, test_probs[:, 1])
            test_auprc = average_precision_score(test_labels, test_probs[:, 1])
            test_preds = np.argmax(test_probs, axis=1)
            test_acc = np.mean(test_preds == test_labels)
            
            print('Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f' % 
                (test_auroc * 100, test_auprc * 100, test_acc * 100))
            print('Classification report', classification_report(test_labels, test_preds))
            print(confusion_matrix(test_labels, test_preds, labels=list(range(n_class))))
            
            # Log test metrics to wandb if enabled
            if args.use_wandb:
                wandb.log({
                    "test_acc": test_acc * 100,
                    "test_auprc": test_auprc * 100,
                    "test_auroc": test_auroc * 100,
                    "run": k,
                })

        acc_arr.append(test_acc * 100)
        auprc_arr.append(test_auprc * 100)
        auroc_arr.append(test_auroc * 100)

    print('args.dataset', args.dataset)
    print('args.model_type', args.model_type)
    # Display the mean and standard deviation of five runs
    mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
    mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
    mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
    print('------------------------------------------')
    print(f'Model: {args.model_type.upper()}')
    print('Accuracy = %.1f±%.1f' % (mean_acc, std_acc))
    print('AUPRC    = %.1f±%.1f' % (mean_auprc, std_auprc))
    print('AUROC    = %.1f±%.1f' % (mean_auroc, std_auroc))

    # Log final metrics to wandb if enabled
    if args.use_wandb:
        wandb.log({
            "model_type": args.model_type,
            "final_mean_acc": mean_acc,
            "final_std_acc": std_acc,
            "final_mean_auprc": mean_auprc,
            "final_std_auprc": std_auprc,
            "final_mean_auroc": mean_auroc,
            "final_std_auroc": std_auroc
        })
        wandb.finish() 







if __name__== "__main__":

    dotenv.load_dotenv('dot_env.txt')

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic4', choices=['P12', 'P19', 'physionet', 'mimic3', 'mimic4'])
    parser.add_argument('--task_mode', type=str, default='NEXT_24h', choices=['CONTRASTIVE', 'NEXT_24h'],
                        help='CONTRASTIVE: standard classification with contrastive loss option, NEXT_24h: mortality prediction in next 24h')

    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rarity_alpha', type=float, default=1)
    parser.add_argument('--query_vector_dim', type=int, default=5)
    parser.add_argument('--node_emb_dim', type=int, default=8)
    parser.add_argument('--plm', type=str, default='bert')
    parser.add_argument('--plm_rep_dim', type=int, default=768)
    parser.add_argument('--source', type=str, default='gpt')

    parser.add_argument('--use_gat', action='store_true', help='Use GAT attention instead of GCN')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads for GAT')
    parser.add_argument('--use_adj_mask', action='store_true', help='Use adjacency matrix as a mask for GAT attention')

    parser.add_argument('--use_clusters', action='store_true', help='Use clusters instead of GRU')

    # Arguments for our dataloader wrapper
    parser.add_argument('--data_path', type=str, default='/Users/riccardoconci/Local_documents/!!MIMIC',
                        help='Path to MIMIC-IV data')
    parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs_lite',
                        help='Path to directory with existing processed files')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--outcome_choice', type=str, default='30d_mortality_discharge',
                        choices=['30d_mortality_discharge', '48h_mortality'], help='Outcome to predict')
    # New arguments for contrastive learning
    parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning instead of classification (only applicable when task_mode=CONTRASTIVE)')
    parser.add_argument('--contrastive_method', type=str, default='clip', choices=['clip', 'infonce'], 
                        help='Contrastive learning method to use')
    parser.add_argument('--similarity_metric', type=str, default='cosine', choices=['cosine', 'l2'],
                        help='Similarity metric for contrastive learning')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
    parser.add_argument('--proj_dim', type=int, default=128, help='Projection dimension for contrastive embeddings')


    # Replace the existing batch-related arguments with a streamlined testing_mode option
    parser.add_argument('--testing_mode', action='store_true', help='Run in testing mode: uses a single cached batch for quicker model testing')
    parser.add_argument('--plot_first_batch_vars', action='store_true', help='Plot the first batch variables')


    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='Geom', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')

    # Add these arguments to the parser after the other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--metric_for_best_model', type=str, default='auprc', choices=['loss', 'auprc', 'auroc'], 
                        help='Metric to use for saving best model')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_all_checkpoints', action='store_true', help='Save a checkpoint after every epoch')
    parser.add_argument('--clean_LMDB_cache', action='store_true', help='Clean LMDB cache before starting')
    
    # Add Raindrop-specific arguments
    parser.add_argument('--model_type', type=str, default='kedgn', choices=['kedgn', 'raindrop_v2'], 
                        help='Model type to use (KEDGN or Raindrop_v2)')
    parser.add_argument('--d_model', type=int, default=64, help='Raindrop model dimension')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of transformer layers for Raindrop')
    parser.add_argument('--global_structure_path', type=str, default=None, 
                        help='Path to adjacency matrix file defining sensor relationships')
    parser.add_argument('--sensor_wise_mask', action='store_true', 
                        help='Use sensor-wise masking for Raindrop_v2')

    args = parser.parse_args()
    
    main(args)