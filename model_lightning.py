# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import *
from einops import repeat
import logging
import os
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
import wandb
import torch

# Configure logging for debugging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model.log'),
        #logging.StreamHandler()  # Optional: keep console output too
    ]
)


class Value_Encoder(nn.Module):
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(Value_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x

class Time_Encoder(nn.Module):
    def __init__(self, embed_time, var_num):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.var_num = var_num
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class MLP_Param(nn.Module):
    def __init__(self, input_size, output_size, query_vector_dim):
        super(MLP_Param, self).__init__()
        self.W_1 = nn.Parameter(torch.FloatTensor(query_vector_dim, input_size, output_size))
        self.b_1 = nn.Parameter(torch.FloatTensor(query_vector_dim, output_size))

        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.b_1)

    def forward(self, x, query_vectors):
        W_1 = torch.einsum("nd, dio->nio", query_vectors, self.W_1)
        b_1 = torch.einsum("nd, do->no", query_vectors, self.b_1)
        x = torch.squeeze(torch.bmm(x.unsqueeze(1), W_1)) + b_1
        return x

class AGCRNCellWithMLP(nn.Module):
    def __init__(self, input_size, query_vector_dim):
        super(AGCRNCellWithMLP, self).__init__()
        # The input to each gate is a concatenation of x and h, plus an extra dimension (+1)
        # These modules generate variable-specific parameters via MLP_Param.
        self.update_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.reset_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.candidate_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        logging.info("AGCRNCellWithMLP initialized with input_size {} and query_vector_dim {}"
                     .format(input_size, query_vector_dim))

    def forward(self, x, h, query_vectors, adj, nodes_ind):
        """
        x: Current input features for all nodes. Shape: [num_nodes, input_size]
        h: Hidden state from the previous time step. Shape: [num_nodes, input_size]
        query_vectors: Variable-specific query vectors (for parameter generation).
                       Expected shape: [num_nodes, query_vector_dim] (or a subset thereof)
        adj: Adjacency matrix (or dynamic graph) applied to current features.
             Shape: [num_nodes, num_nodes]
        nodes_ind: Indices for nodes that are observed at the current timestamp.
                   This can be a tuple of indices, for example from torch.where().
        """
        # 1. Concatenate current input and previous hidden state along last dimension.
        combined = torch.cat([x, h], dim=-1)  # Shape: [num_nodes, 2*input_size]
        logging.debug("Combined input and hidden shape: {}".format(combined.shape))
        
        # 2. Perform a graph convolution-like multiplication:
        # Multiply aggregated neighbor features by the adjacency matrix.
        combined = torch.matmul(adj, combined)  # Still [num_nodes, 2*input_size]
        logging.debug("After graph convolution (adj * combined): {}".format(combined.shape))
        
        # 3. Compute the reset gate using variable-specific parameters.
        # The reset gate decides how much of the old hidden state to forget.
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], query_vectors))
        logging.debug("Reset gate shape (r): {}".format(r.shape))
        
        # 4. Compute the update gate, which balances the new candidate vs. old hidden.
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], query_vectors))
        logging.debug("Update gate shape (u): {}".format(u.shape))
        
        # 5. Apply the reset gate to the hidden state of the nodes to update.
        # This implements the component r ⊙ h as described in the candidate hidden state equation.
        h[nodes_ind] = r * h[nodes_ind]
        logging.debug("Hidden state updated with reset gate for observed nodes.")
        
        # 6. Re-concatenate input and modified hidden state.
        combined_new = torch.cat([x, h], dim=-1)  # Shape: [num_nodes, 2*input_size]
        logging.debug("New combined (x and updated h) shape: {}".format(combined_new.shape))
        
        # 7. Compute candidate hidden state with tanh non-linearity.
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], query_vectors))
        logging.debug("Candidate hidden state shape: {}".format(candidate_h.shape))
        
        # 8. Final hidden update (only for observed nodes):
        # H^(t) = (1 - u) ⊙ H^(t-1) + u ⊙ candidate_h
        new_h = (1 - u) * h[nodes_ind] + u * candidate_h
        logging.debug("New hidden state (for updated nodes) shape: {}".format(new_h.shape))
        return new_h

class AGATCellWithMLP(nn.Module):
    def __init__(self, input_size, query_vector_dim, num_heads=1, use_adj_mask=False):
        super(AGATCellWithMLP, self).__init__()
        # The input to each gate is a concatenation of x and h, plus an extra dimension (+1)
        # These modules generate variable-specific parameters via MLP_Param.
        self.update_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.reset_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        self.candidate_gate = MLP_Param(2 * input_size + 1, input_size, query_vector_dim)
        
        # GAT attention components
        self.num_heads = num_heads
        # The +1 accounts for the rarity score that's concatenated to the input x
        combined_dim = 2 * input_size + 1  # x+rarity_score + h 
        self.attentions = nn.ModuleList([
            nn.Linear(2 * combined_dim, 1) for _ in range(num_heads)
        ])
        self.use_adj_mask = use_adj_mask
        logging.info("AGATCellWithMLP initialized with input_size {}, actual combined dim {}, and query_vector_dim {}, num_heads {}, use_adj_mask: {}"
                     .format(input_size, combined_dim, query_vector_dim, num_heads, use_adj_mask))

    def forward(self, x, h, query_vectors, adj, nodes_ind):
        """
        x: Current input features for all nodes. Shape: [num_nodes, input_size]
        h: Hidden state from the previous time step. Shape: [num_nodes, input_size]
        query_vectors: Variable-specific query vectors (for parameter generation).
                       Expected shape: [num_nodes, query_vector_dim] (or a subset thereof)
        adj: Adjacency matrix (or dynamic graph) applied to current features.
             Shape: [num_nodes, num_nodes] - only used as an optional mask
        nodes_ind: Indices for nodes that are observed at the current timestamp.
                   This can be a tuple of indices, for example from torch.where().
        """
        # 1. Concatenate current input and previous hidden state along last dimension.
        combined = torch.cat([x, h], dim=-1)  # Shape: [num_nodes, 2*input_size]
        logging.debug("Combined input and hidden shape: {}".format(combined.shape))
        
        # 2. GAT ATTENTION INSTEAD OF GCN
        # Multi-head attention computation
        attention_outputs = []
        
        # Attention heads compute message passing weights
        for head in range(self.num_heads):
            # Direct attention implementation without loops
            # For each src node i and target node j:
            # Stack src_node_feature (repeated for each target) and tgt_node_features
            src_nodes = combined.unsqueeze(2).repeat(1, 1, combined.size(1), 1)  # [B, N, N, F]
            tgt_nodes = combined.unsqueeze(1).repeat(1, combined.size(1), 1, 1)  # [B, N, N, F]
            
            # Concatenate along feature dimension
            node_pairs = torch.cat([src_nodes, tgt_nodes], dim=-1)  # [B, N, N, 2F]
            
            # Reshape for linear layer
            node_pairs_flat = node_pairs.view(-1, node_pairs.size(-1))  # [B*N*N, 2F]
            
            # Compute attention scores
            attn_flat = self.attentions[head](node_pairs_flat)  # [B*N*N, 1]
            attn_scores = attn_flat.view(combined.size(0), combined.size(1), combined.size(1))  # [B, N, N]
            
            # Apply leaky ReLU
            attn_scores = F.leaky_relu(attn_scores, 0.2)
            
            # Optionally use adjacency matrix as a mask (if specified)
            if self.use_adj_mask:
                attn_scores = attn_scores.masked_fill(adj == 0, -9e15)
            
            # Attention weights with softmax normalization
            attention = F.softmax(attn_scores, dim=2)
            
            # Apply attention weights to get node features
            h_prime = torch.bmm(attention, combined)
            attention_outputs.append(h_prime)
        
        # Average across attention heads
        combined = torch.mean(torch.stack(attention_outputs), dim=0)
        logging.debug("After GAT attention mechanism: {}".format(combined.shape))
        
        # 3. Compute the reset gate using variable-specific parameters.
        # The reset gate decides how much of the old hidden state to forget.
        r = torch.sigmoid(self.reset_gate(combined[nodes_ind], query_vectors))
        logging.debug("Reset gate shape (r): {}".format(r.shape))
        
        # 4. Compute the update gate, which balances the new candidate vs. old hidden.
        u = torch.sigmoid(self.update_gate(combined[nodes_ind], query_vectors))
        logging.debug("Update gate shape (u): {}".format(u.shape))
        
        # 5. Apply the reset gate to the hidden state of the nodes to update.
        # This implements the component r ⊙ h as described in the candidate hidden state equation.
        h[nodes_ind] = r * h[nodes_ind]
        logging.debug("Hidden state updated with reset gate for observed nodes.")
        
        # 6. Re-concatenate input and modified hidden state.
        combined_new = torch.cat([x, h], dim=-1)  # Shape: [num_nodes, 2*input_size]
        logging.debug("New combined (x and updated h) shape: {}".format(combined_new.shape))
        
        # 7. Compute candidate hidden state with tanh non-linearity.
        candidate_h = torch.tanh(self.candidate_gate(combined_new[nodes_ind], query_vectors))
        logging.debug("Candidate hidden state shape: {}".format(candidate_h.shape))
        
        # 8. Final hidden update (only for observed nodes):
        # H^(t) = (1 - u) ⊙ H^(t-1) + u ⊙ candidate_h
        new_h = (1 - u) * h[nodes_ind] + u * candidate_h
        logging.debug("New hidden state (for updated nodes) shape: {}".format(new_h.shape))
        return new_h

# All other cell types included here...
# [VariableTransformerCell and other model components would be included here]

class VariableTransformerCell(nn.Module):
    def __init__(self, input_size, hidden_size, nhead=2, dim_feedforward=64, dropout=0.1):
        super(VariableTransformerCell, self).__init__()
        
        # Time encoding dimension (must be even for sinusoidal encoding)
        self.time_encoding_dim = 16
        assert self.time_encoding_dim % 2 == 0, "Time encoding dimension must be even"
        
        # Create a standard Transformer encoder layer with time-aware input
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size + self.time_encoding_dim,  # Add time encoding dimension
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Linear projection to map transformer output to hidden representation
        self.projection = nn.Linear(input_size + self.time_encoding_dim, hidden_size)
        
        logging.info("VariableTransformerCell initialized with input_size {}, hidden_size {}, nhead {}, time_encoding_dim {}"
                     .format(input_size, hidden_size, nhead, self.time_encoding_dim))
        
    def time_encoding(self, time_values):
        """
        Create sinusoidal time encoding for irregular timestamps
        Args:
            time_values: Tensor of timestamps [batch, seq_len]
        Returns:
            Encoding with shape [batch, seq_len, time_encoding_dim]
        """
        batch_size, seq_len = time_values.shape
        
        # Scale timestamps to avoid extremely large values
        # Normalize based on the range of values in each sequence
        max_vals, _ = torch.max(time_values, dim=1, keepdim=True)
        min_vals, _ = torch.min(time_values, dim=1, keepdim=True)
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        # Normalize times to [0, 1] range for each sequence
        time_values = (time_values - min_vals) / (max_vals - min_vals + eps)
        
        # Create dimension indices
        dim_indices = torch.arange(0, self.time_encoding_dim // 2, device=time_values.device)
        # 10000^(2i/dmodel) denominator term from the Transformer paper
        dim_scales = torch.pow(10000.0, -2.0 * dim_indices / self.time_encoding_dim)
        # Reshape for broadcasting
        dim_scales = dim_scales.view(1, 1, -1)
        
        # Reshape time values for broadcasting
        t = time_values.view(batch_size, seq_len, 1)
        
        # Compute arguments for sin and cos
        args = t * dim_scales  # [batch, seq_len, time_encoding_dim//2]
        
        # Compute positional encoding with sin and cos
        pe_sin = torch.sin(args)
        pe_cos = torch.cos(args)
        
        # Interleave sin and cos
        pe = torch.zeros(batch_size, seq_len, self.time_encoding_dim, device=time_values.device)
        pe[:, :, 0::2] = pe_sin
        pe[:, :, 1::2] = pe_cos
        
        return pe
        
    def forward(self, x_history, timestamps, mask=None):
        """
        x_history: Temporal history for each variable
                  Shape: [batch, seq_len, input_size]
        timestamps: Absolute timestamps for each observation
                   Shape: [batch, seq_len]
        mask: Optional padding mask for variable-length sequences
              Shape: [batch, seq_len] where True values are masked positions
        
        Returns:
            hidden: New hidden state considering temporal context
                   Shape: [batch, hidden_size]
        """
        # Generate time encodings for the timestamps
        time_encodings = self.time_encoding(timestamps)
        
        # Concatenate input features with time encodings
        x_with_time = torch.cat([x_history, time_encodings], dim=-1)
        
        # Apply transformer self-attention over the temporal dimension
        # Each variable attends to its own history with time-aware positional encoding
        attended = self.transformer_layer(x_with_time, src_key_padding_mask=mask)
        
        # Use the representation of the last timestep
        last_state = attended[:, -1]
        
        # Project to hidden dimension
        hidden = self.projection(last_state)
        
        return hidden

class VSDGCRNN(nn.Module):
    # [VSDGCRNN implementation]
    # ...
    pass

class VSDGATRNN(nn.Module):
    # [VSDGATRNN implementation]
    # ...
    pass
    
class VSDTransformerGATRNN(nn.Module):
    # [VSDTransformerGATRNN implementation]
    # ...
    pass

# Main Lightning module
class KEDGNLightning(pl.LightningModule):
    def __init__(
        self, 
        hidden_dim, 
        num_of_variables, 
        num_of_timestamps, 
        d_static,
        n_class=2, 
        node_enc_layer=2, 
        rarity_alpha=0.5, 
        query_vector_dim=5, 
        node_emb_dim=8, 
        plm_rep_dim=768, 
        use_gat=False, 
        num_heads=2, 
        use_adj_mask=False,
        use_transformer=False, 
        history_len=10, 
        nhead_transformer=2,
        learning_rate=1e-3
    ):
        super(KEDGNLightning, self).__init__()
        
        # Save hyperparameters for easy access and logging
        self.save_hyperparameters()
        
        # Initialize model components
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.hidden_dim = hidden_dim
        
        # Initialize a learnable adjacency matrix
        self.adj = nn.Parameter(torch.ones(size=[num_of_variables, num_of_variables]))
        
        # Encoders for raw values and absolute time information
        self.value_enc = Value_Encoder(output_dim=hidden_dim)
        self.abs_time_enc = Time_Encoder(embed_time=hidden_dim, var_num=num_of_variables)
        
        # GRU to process observation time patterns with multiple layers
        self.obs_tp_enc = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim,
            num_layers=node_enc_layer, 
            batch_first=True, 
            bidirectional=False
        )
        
        # Observation encoder
        self.obs_enc = nn.Sequential(
            nn.Linear(in_features=6 * hidden_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        
        # Embedding for variable type information
        self.type_emb = nn.Embedding(num_of_variables, hidden_dim)
        
        # Choose the appropriate neural network architecture based on input flags
        if use_transformer:
            self.GCRNN = VSDTransformerGATRNN(
                d_in=self.hidden_dim, 
                d_model=self.hidden_dim,
                num_of_nodes=num_of_variables, 
                history_len=history_len,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim, 
                num_heads=num_heads, 
                use_adj_mask=use_adj_mask,
                nhead_transformer=nhead_transformer
            )
            logging.info("Using Transformer-GAT model with history_len: {}, transformer heads: {}, GAT heads: {}"
                         .format(history_len, nhead_transformer, num_heads))
        elif use_gat:
            self.GCRNN = VSDGATRNN(
                d_in=self.hidden_dim, 
                d_model=self.hidden_dim,
                num_of_nodes=num_of_variables, 
                rarity_alpha=rarity_alpha,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim, 
                num_heads=num_heads, 
                use_adj_mask=use_adj_mask
            )
            logging.info("Using GRU-GAT model with {} heads, use_adj_mask: {}".format(num_heads, use_adj_mask))
        else:
            self.GCRNN = VSDGCRNN(
                d_in=self.hidden_dim, 
                d_model=self.hidden_dim,
                num_of_nodes=num_of_variables, 
                rarity_alpha=rarity_alpha,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim
            )
            logging.info("Using original GRU-GCN model")
        
        # Final layers
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # Process static features if available
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_variables)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Save learning rate
        self.learning_rate = learning_rate

    def forward(self, P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor):
        """
        Forward pass for KEDGN.
        Inputs:
            P: Tensor of shape [B, T, 2*V] where first V columns are observed values and last V are masks.
            P_static: Tensor of static features [B, d_static] (or None).
            P_avg_interval: Tensor, average intervals [B, T, V].
            P_length: Tensor, lengths of the time series for each batch element [B, 1].
            P_time: Tensor of timestamps [B, T, V] (or [B, T]).
            P_var_plm_rep_tensor: Tensor of pre-trained language model embeddings for each variable [B, V, plm_rep_dim]
        """
        # Logging input shapes
        logging.debug("Input P shape: {}".format(P.shape))
        b, t, v = P.shape
        # Since P contains both observed data and masks, divide v by 2.
        v = v // 2

        # Split P into observed data and corresponding observation masks.
        observed_data = P[:, :, :v]        # Shape: [B, T, V]
        observed_mask = P[:, :, v:]          # Shape: [B, T, V]
        logging.debug("Observed data shape: {} | Observed mask shape: {}".format(observed_data.shape, observed_mask.shape))

        # Encode the observed values and times.
        # The encoders expect additional dimensions, so we multiply elementwise by mask to zero-out missing data.
        value_emb = self.value_enc(observed_data) * observed_mask.unsqueeze(-1)  # [B, T, V, hidden_dim]
        abs_time_emb = self.abs_time_enc(P_time) * observed_mask.unsqueeze(-1)   # [B, T, V, hidden_dim]
        logging.debug("Value embedding shape: {} | Time embedding shape: {}".format(value_emb.shape, abs_time_emb.shape))

        # Get type embedding for variables.
        # Repeat the learnable embedding weight vector from self.type_emb (of shape [V, hidden_dim])
        # so that we have one per batch sample.
        type_emb = repeat(self.type_emb.weight, 'v d -> b v d', b=b)  # Shape: [B, V, hidden_dim]
        logging.debug("Type embedding shape: {}".format(type_emb.shape))

        # Prepare the structured input encoding.
        # This combines the value, time, and type embeddings.
        # We need to match dimensions so we repeat 'type_emb' along the time dimension.
        structure_input_encoding = (value_emb + abs_time_emb + repeat(type_emb, 'b v d -> b t v d', t=t)) * observed_mask.unsqueeze(-1)
        logging.debug("Structured input encoding shape: {}".format(structure_input_encoding.shape))
        
        # Pass the structured encoding along with mask, lengths, average intervals, and PLM embeddings
        # into the dynamic graph convolutional recurrent network.
        last_hidden_state = self.GCRNN(structure_input_encoding, observed_mask, P_length, P_avg_interval, P_var_plm_rep_tensor)
        logging.debug("Last hidden state shape: {}".format(last_hidden_state.shape))
        
        # Sum the hidden state across the feature channels (if that is the desired aggregation).
        aggregated_hidden = torch.sum(last_hidden_state, dim=-1)  # Shape: [B, V]
        logging.debug("Aggregated hidden state shape: {}".format(aggregated_hidden.shape))
        
        # Optionally integrate static features.
        if P_static is not None:
            static_emb = self.emb(P_static)  # Map static features to an embedding of shape [B, V]
            logging.debug("Static embedding shape: {}".format(static_emb.shape))
            # Concatenate aggregated hidden states and static embeddings along last dimension.
            fused_features = torch.cat([aggregated_hidden, static_emb], dim=-1)  # [B, 2*V]
            logging.debug("Fused feature shape (hidden + static): {}".format(fused_features.shape))
            output = self.classifier(fused_features)
        else:
            output = self.classifier(aggregated_hidden)
        
        logging.info("Output shape: {}".format(output.shape))
        return output

    def training_step(self, batch, batch_idx):
        """
        Lightning training step
        Args:
            batch: The output of your DataLoader
            batch_idx: Integer displaying index of this batch
        Returns:
            Dictionary with loss and any desired metrics for logging
        """
        # Unpack batch
        P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor, y = batch
        
        # Forward pass
        logits = self.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate loss
        loss = self.criterion(logits, y.long().squeeze(1))
        
        # Calculate additional metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Convert tensors to CPU numpy arrays for sklearn metrics
        y_np = y.squeeze(1).cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        
        # Calculate metrics
        acc = (preds == y.squeeze(1)).float().mean()
        try:
            auroc = roc_auc_score(y_np, probs_np)
            auprc = average_precision_score(y_np, probs_np)
        except:
            # Handle edge case where batch contains only one class
            auroc = 0.0
            auprc = 0.0
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_auroc', auroc, on_step=False, on_epoch=True)
        self.log('train_auprc', auprc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning validation step
        """
        # Unpack batch
        P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor, y = batch
        
        # Forward pass
        logits = self.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate loss
        loss = self.criterion(logits, y.long().squeeze(1))
        
        # Calculate additional metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Convert tensors to CPU numpy arrays for sklearn metrics
        y_np = y.squeeze(1).cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        
        # Calculate metrics
        acc = (preds == y.squeeze(1)).float().mean()
        try:
            auroc = roc_auc_score(y_np, probs_np)
            auprc = average_precision_score(y_np, probs_np)
        except:
            # Handle edge case where batch contains only one class
            auroc = 0.0
            auprc = 0.0
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', auroc, on_epoch=True)
        self.log('val_auprc', auprc, on_epoch=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_auroc': auroc, 'val_auprc': auprc}

    def test_step(self, batch, batch_idx):
        """
        Lightning test step
        """
        # Unpack batch
        P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor, y = batch
        
        # Forward pass
        logits = self.forward(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate loss
        loss = self.criterion(logits, y.long().squeeze(1))
        
        # Calculate additional metrics
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Convert tensors to CPU numpy arrays for sklearn metrics
        y_np = y.squeeze(1).cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        
        # Calculate metrics
        acc = (preds == y.squeeze(1)).float().mean()
        
        # Detailed metrics for test set
        if y_np.size > 1:  # Only compute if there are multiple samples
            auroc = roc_auc_score(y_np, probs_np)
            auprc = average_precision_score(y_np, probs_np)
            conf_mat = confusion_matrix(y_np, preds_np, labels=list(range(self.hparams.n_class)))
            class_report = classification_report(y_np, preds_np, labels=list(range(self.hparams.n_class)))
            
            # Log metrics
            self.log('test_loss', loss)
            self.log('test_acc', acc)
            self.log('test_auroc', auroc)
            self.log('test_auprc', auprc)
            
            # Log confusion matrix as a plot
            if wandb.run is not None:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import numpy as np
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    wandb.log({"confusion_matrix": wandb.Image(plt)})
                    plt.close()
                except:
                    pass
            
            return {'test_loss': loss, 'test_acc': acc, 'test_auroc': auroc, 'test_auprc': auprc, 
                    'confusion_matrix': conf_mat, 'classification_report': class_report}
        else:
            return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        """
        Configure the optimizer for training
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer 