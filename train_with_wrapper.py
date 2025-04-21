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
from utils import contrastive_loss
from models import DSEncoderWithRNN
dotenv.load_dotenv()

# Import our wrapper around MIMICContrastivePairsDataset
from dataloader_wrapper import MIMIC4KedgnWrapper

wandb.login(key=os.getenv("WANDB_API_KEY"))

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mimic4', choices=['P12', 'P19', 'physionet', 'mimic3', 'mimic4'])
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--rarity_alpha', type=float, default=1)
parser.add_argument('--query_vector_dim', type=int, default=5)
parser.add_argument('--node_emb_dim', type=int, default=8)
parser.add_argument('--plm', type=str, default='bert')
parser.add_argument('--plm_rep_dim', type=int, default=768)
parser.add_argument('--source', type=str, default='gpt')
parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
parser.add_argument('--wandb_project', type=str, default='Geom', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
parser.add_argument('--use_gat', action='store_true', help='Use GAT attention instead of GCN')
parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads for GAT')
parser.add_argument('--use_adj_mask', action='store_true', help='Use adjacency matrix as a mask for GAT attention')
parser.add_argument('--use_transformer', action='store_true', help='Use transformer per variable instead of GRU')
parser.add_argument('--history_len', type=int, default=10, help='History length for transformer model')
parser.add_argument('--nhead_transformer', type=int, default=2, help='Number of attention heads in transformer')
# Arguments for our dataloader wrapper
parser.add_argument('--base_path', type=str, default='/Users/riccardoconci/Local_documents/!!MIMIC',
                    help='Path to MIMIC-IV data')
parser.add_argument('--temp_dfs_path', type=str, default='temp_dfs',
                    help='Path to directory with existing processed files')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
parser.add_argument('--outcome_choice', type=str, default='30d_mortality_discharge',
                    choices=['30d_mortality_discharge', '48h_mortality'], help='Outcome to predict')
# New arguments for contrastive learning
parser.add_argument('--use_contrastive', action='store_true', help='Use contrastive learning instead of classification')
parser.add_argument('--contrastive_method', type=str, default='clip', choices=['clip', 'infonce'], 
                    help='Contrastive learning method to use')
parser.add_argument('--similarity_metric', type=str, default='cosine', choices=['cosine', 'l2'],
                    help='Similarity metric for contrastive learning')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
parser.add_argument('--proj_dim', type=int, default=128, help='Projection dimension for contrastive embeddings')

args, unknown = parser.parse_known_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from models import *
from utils import count_parameters

# Set device - try MPS first, then CUDA, then fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using CUDA device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

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
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Load command line hyperparameters
dataset = args.dataset
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs
hidden_dim = args.hidden_dim
rarity_alpha = args.rarity_alpha
query_vector_dim = args.query_vector_dim
node_emb_dim = args.node_emb_dim
plm_rep_dim = args.plm_rep_dim
source = args.source

print('Dataset used: ', dataset)

# Set dataset parameters
if dataset == 'mimic4':
    d_static = 2  # Age and gender
    variables_num = 669  # Number of clinical variables
    timestamp_num = 96   # Number of time steps
    n_class = 2          # Binary classification
else:
    raise ValueError(f"Dataset {dataset} not supported with this wrapper")

# Add debug info about variables_num
print(f"Using variables_num = {variables_num} but will verify with actual data...")

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

    # Load data using our custom wrapper
    print(f"\nCreating dataloaders for run {k+1}/5...")
    train_loader, val_loader, test_loader, P_var_plm_rep_tensor = MIMIC4KedgnWrapper.get_dataloaders(
        base_path=args.base_path,
        temp_dfs_path=args.temp_dfs_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        outcome_choice=args.outcome_choice
    )
    
    # Move variable embeddings to device
    P_var_plm_rep_tensor = P_var_plm_rep_tensor.to(device)
    
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples, Test: {len(test_loader.dataset)} samples")
    
    # Get an example batch to determine the actual feature dimension
    sample_batch = next(iter(train_loader))
    values = sample_batch['values']
    print(f"DEBUG - Sample batch values shape: {values.shape}")
    
    # The tensor is [T, F] or [B, T, F], we need to figure out which
    if len(values.shape) == 3:  # [B, T, F]
        actual_feature_dim = values.shape[2]
    else:  # [T, F]
        actual_feature_dim = values.shape[1]
        
    # Adjust variables_num if it doesn't match the actual data
    if actual_feature_dim != variables_num:
        print(f"WARNING: Hardcoded variables_num ({variables_num}) doesn't match actual feature dimension ({actual_feature_dim})")
        print(f"Adjusting variables_num to {actual_feature_dim}")
        variables_num = actual_feature_dim
        
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

    # Initialize KEDGN model
    model = KEDGN(DEVICE=device,
                  hidden_dim=hidden_dim,
                  num_of_variables=variables_num,
                  num_of_timestamps=timestamp_num,
                  d_static=d_static,
                  n_class=n_class,
                  rarity_alpha=rarity_alpha,
                  query_vector_dim=query_vector_dim,
                  node_emb_dim=node_emb_dim,
                  plm_rep_dim=plm_rep_dim,
                  use_gat=args.use_gat,
                  num_heads=args.num_heads,
                  use_adj_mask=args.use_adj_mask,
                  use_transformer=args.use_transformer,
                  history_len=args.history_len,
                  nhead_transformer=args.nhead_transformer)

    # If using contrastive learning, add text encoder and projection heads
    if args.use_contrastive:
        text_encoder = DSEncoderWithRNN().to(device)  
        
        ts_projection = ProjectionHead(hidden_dim*2, args.proj_dim).to(device)
        text_projection = ProjectionHead(hidden_dim*2, args.proj_dim).to(device)
        
        # Add these components to the model for easier management
        model.text_encoder = text_encoder
        model.ts_projection = ts_projection
        model.text_projection = text_projection
        model.temperature = args.temperature
        model.similarity_metric = args.similarity_metric
        model.contrastive_method = args.contrastive_method

    print('model parameters:', count_parameters(model))
    
    if args.use_contrastive != True:
        classification_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_epoch = 0
    best_aupr_val = best_auc_val = 0.0
    best_loss_val = 100.0


    start = time.time()

    for epoch in range(num_epochs):
        """Training"""
        model.train()
        train_loss = 0.0
        train_probs_all = []
        train_labels_all = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Get data from batch and move to device
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device) if d_static > 0 else None
            times = batch['times'].to(device)
            length = batch['length'].to(device)
            labels = batch['label'].to(device)
            
            # Print tensor shapes to debug
            print(f"DEBUG - values shape: {values.shape}")
            print(f"DEBUG - mask shape: {mask.shape}")
            print(f"DEBUG - times shape: {times.shape}")
            print(f"DEBUG - variables_num: {variables_num}")
            
            # Create input format expected by KEDGN
            P = torch.cat([values, mask], dim=2)  # Shape [B, T, F*2]
            
            # Print P shape
            print(f"DEBUG - P shape after cat: {P.shape}")
            
            # Handle the time tensor properly based on its dimensions
            # P_time should be shaped [B, T, 1] or [B, T] for the model
            if len(times.shape) == 2:  # Shape [B, T]
                P_time = times  # Keep as is
            else:  # Just a single dimension [T]
                P_time = times.unsqueeze(0).repeat(values.size(0), 1)  # Shape [B, T]
            
            # Print P_time shape
            print(f"DEBUG - P_time shape: {P_time.shape}")
                
            # P_avg_interval should have the same shape as P_time, but needs to be expanded
            # for the variables dimension in the rarity calculation
            P_avg_interval = torch.ones_like(P_time)  # Shape [B, T]
            
            # Expand P_avg_interval to match the number of variables
            # This avoids dimension mismatch in rarity score calculation
            P_avg_interval = P_avg_interval.unsqueeze(2).expand(-1, -1, variables_num)  # Shape [B, T, N]
            
            P_length = length.unsqueeze(1)  # Shape [B, 1]
            
            # Print additional debug info
            print(f"DEBUG - P_avg_interval shape: {P_avg_interval.shape}")
            print(f"DEBUG - P_length shape: {P_length.shape}")
            print(f"DEBUG - P_var_plm_rep_tensor shape: {P_var_plm_rep_tensor.shape}")
            
            # Forward pass through the model
            if args.use_contrastive:
                # Get discharge text chunks for contrastive learning
                discharge_chunks = batch['discharge_chunks']
                
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
                # Standard classification forward pass
                outputs = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                
                # Compute classification loss
                loss = classification_criterion(outputs, labels.squeeze(1).long())
                
                # Store probabilities for metrics
                probs = torch.softmax(outputs, dim=1)
                train_probs_all.append(probs.detach().cpu())
                train_labels_all.append(labels.detach().cpu())
            
            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        
        if not args.use_contrastive:
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
            for batch_idx, batch in enumerate(val_loader):
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
                
                # Print additional debug info
                print(f"DEBUG - P_avg_interval shape: {P_avg_interval.shape}")
                print(f"DEBUG - P_length shape: {P_length.shape}")
                print(f"DEBUG - P_var_plm_rep_tensor shape: {P_var_plm_rep_tensor.shape}")
                
                # Forward pass
                if args.use_contrastive:
                    # Get discharge text chunks for contrastive learning
                    discharge_chunks = batch['discharge_chunks']
                    
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
                    
                    # For validation in contrastive learning, we also compute classification metrics
                    # by using the ts_intermediate_rep with a linear classifier
                    linear_classifier = torch.nn.Linear(hidden_dim*2, n_class).to(device)
                    outputs = linear_classifier(ts_intermediate_rep)
                    probs = torch.softmax(outputs, dim=1)
                else:
                    # Standard classification forward pass
                    outputs = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                    
                    # Compute classification loss
                    loss = classification_criterion(outputs, labels.squeeze(1).long())
                    
                    # Calculate probabilities
                    probs = torch.softmax(outputs, dim=1)
                
                val_loss += loss.item()
                
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
            
            print(
                "Epoch %d, train_loss:%.4f, train_auprc:%.2f, train_auroc:%.2f, val_loss:%.4f, acc_val: %.2f, aupr_val: %.2f, auc_val: %.2f" %
                (epoch, train_loss, train_auprc * 100, train_auroc * 100,
                 val_loss, val_acc * 100, val_auprc * 100, val_auroc * 100))
            
            # Log metrics to wandb if enabled
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_auprc": train_auprc * 100,
                    "train_auroc": train_auroc * 100,
                    "val_loss": val_loss,
                    "val_acc": val_acc * 100,
                    "val_auprc": val_auprc * 100,
                    "val_auroc": val_auroc * 100
                })

            # Save the model weights with the best AUPRC on the validation set
            if val_auprc > best_aupr_val:
                best_auc_val = val_auroc
                best_aupr_val = val_auprc
                best_val_epoch = epoch
                save_time = str(int(time.time()))
                torch.save(model.state_dict(),
                           model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt')

    end = time.time()
    time_elapsed = end - start
    print('Total Time elapsed: %.3f mins' % (time_elapsed / 60.0))

    """Testing"""
    model.eval()
    model.load_state_dict(
        torch.load(model_path + '_' + dataset + '_' + save_time + '_' + str(k) + '.pt'))
    
    test_probs_all = []
    test_labels_all = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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
            
            # Print additional debug info
            print(f"DEBUG - P_avg_interval shape: {P_avg_interval.shape}")
            print(f"DEBUG - P_length shape: {P_length.shape}")
            print(f"DEBUG - P_var_plm_rep_tensor shape: {P_var_plm_rep_tensor.shape}")
            
            # Forward pass
            if args.use_contrastive:
                # Get discharge text chunks for contrastive learning
                discharge_chunks = batch['discharge_chunks']
                
                # Forward pass to get base KEDGN representations
                # The KEDGN model now returns (output, aggregated_hidden, fused_features)
                _, _, ts_intermediate_rep = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
                
                # For testing with contrastive learning, we evaluate using a linear classifier
                linear_classifier = torch.nn.Linear(hidden_dim*2, n_class).to(device)
                outputs = linear_classifier(ts_intermediate_rep)
            else:
                # Standard classification forward pass
                outputs = model.forward(P, static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
            
            # Calculate probabilities
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
# Display the mean and standard deviation of five runs
mean_acc, std_acc = np.mean(acc_arr), np.std(acc_arr)
mean_auprc, std_auprc = np.mean(auprc_arr), np.std(auprc_arr)
mean_auroc, std_auroc = np.mean(auroc_arr), np.std(auroc_arr)
print('------------------------------------------')
print('Accuracy = %.1f±%.1f' % (mean_acc, std_acc))
print('AUPRC    = %.1f±%.1f' % (mean_auprc, std_auprc))
print('AUROC    = %.1f±%.1f' % (mean_auroc, std_auroc))

# Log final metrics to wandb if enabled
if args.use_wandb:
    wandb.log({
        "final_mean_acc": mean_acc,
        "final_std_acc": std_acc,
        "final_mean_auprc": mean_auprc,
        "final_std_auprc": std_auprc,
        "final_mean_auroc": mean_auroc,
        "final_std_auroc": std_auroc
    })
    wandb.finish() 