# -*- coding:utf-8 -*-
import torch.nn as nn

import torch.nn.functional as F
from train_utils import *
from einops import *
from einops import repeat
import logging
import math
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models.models_utils import MLP, MLP_Param



# Configure logging for debugging - file only, no console output
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('logs/models.log')
    ]
)



class AGATCellWithMLP(nn.Module):
    def __init__(self, input_size, query_vector_dim, num_heads=1, use_adj_mask=False, use_skip_connection=True):
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
        
        # More memory-efficient GAT implementation
        self.query = nn.ModuleList([nn.Linear(combined_dim, combined_dim // 8) for _ in range(num_heads)])
        self.key = nn.ModuleList([nn.Linear(combined_dim, combined_dim // 8) for _ in range(num_heads)])
        self.value = nn.ModuleList([nn.Linear(combined_dim, combined_dim) for _ in range(num_heads)])
        
        # MLP to combine multi-head attention outputs
        self.attentionMLP = MLP(num_heads*combined_dim, combined_dim, combined_dim)
        
        self.use_adj_mask = use_adj_mask
        self.use_skip_connection = use_skip_connection
        
        logging.info("AGATCellWithMLP initialized with input_size {}, actual combined dim {}, query_vector_dim {}, num_heads {}, use_adj_mask: {}, use_skip_connection: {}"
                     .format(input_size, combined_dim, query_vector_dim, num_heads, use_adj_mask, use_skip_connection))

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
        
        # 2. MEMORY-EFFICIENT GAT ATTENTION
        # Multi-head attention computation
        attention_outputs = []
        
        # Process each attention head separately to save memory
        for head in range(self.num_heads):
            # Transform features to queries and keys, reducing dimensions for efficiency
            queries = self.query[head](combined)  # [B, N, dim//8]
            keys = self.key[head](combined)       # [B, N, dim//8]
            values = self.value[head](combined)   # [B, N, dim]
            
            # Compute attention scores with matrix multiplication
            # (B, N, dim//8) @ (B, dim//8, N) -> (B, N, N)
            attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
            
            # Apply leaky ReLU
            attn_scores = F.leaky_relu(attn_scores, 0.2)
            
            # Optionally use adjacency matrix as a mask (if specified)
            if self.use_adj_mask:
                attn_scores = attn_scores.masked_fill(adj == 0, -9e15)
            
            # Attention weights with softmax normalization
            attention = F.softmax(attn_scores, dim=2)
            
            # Apply attention weights to get node features
            h_prime = torch.bmm(attention, values)
            attention_outputs.append(h_prime)
        
        # Process multi-head attention through MLP
        # Stack outputs: [num_heads, B, N, dim] -> reshape to [B, N, num_heads*dim]
        attention_stacked = torch.stack(attention_outputs)
        batch_size, num_nodes, feat_dim = attention_stacked.shape[1:]
        attention_reshaped = attention_stacked.permute(1, 2, 0, 3).reshape(batch_size, num_nodes, -1)
        node_representations_attention = self.attentionMLP(attention_reshaped)
        
        # Apply skip connection if enabled (as in the cluster model)
        if self.use_skip_connection:
            combined = node_representations_attention + combined
        else:
            combined = node_representations_attention
            
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

class VSDGATRNN(nn.Module):
    def __init__(self, d_in, d_model, num_of_nodes, rarity_alpha=0.5, 
                 query_vector_dim=5, node_emb_dim=8, plm_rep_dim=768, num_heads=1, 
                 use_adj_mask=False, update_all_nodes=False, use_plm_adjacency=False,
                 use_skip_connection=True):
        super(VSDGATRNN, self).__init__()
        self.d_in = d_in  # Input dimension for observations.
        self.d_model = d_model  # Hidden (model) dimension.
        self.num_of_nodes = num_of_nodes  # Number of variables.
        
        # Use GAT cell instead of GCN cell
        self.gated_update = AGATCellWithMLP(d_model, query_vector_dim, num_heads, use_adj_mask, use_skip_connection)
        
        # New flag to control whether to update all nodes or just observed ones
        self.update_all_nodes = update_all_nodes
        
        # New flag to control whether to use PLM embeddings for adjacency matrix
        self.use_plm_adjacency = use_plm_adjacency
        
        # Rarity parameter: scales the influence of sampling densities.
        self.rarity_alpha = rarity_alpha
        # Learnable matrix used in adjusting density-based weights:
        self.rarity_W = nn.Parameter(torch.randn(num_of_nodes, num_of_nodes))
        self.relu = nn.ReLU()
        
        # Two MLPs to project PLM textual embeddings:
        # (a) For generating query vectors that will modulate the gated updates.
        self.projection_f = MLP(plm_rep_dim, 2 * d_model, query_vector_dim)
        
        # (b) For obtaining node embeddings to compute a knowledge-aware graph (if enabled).
        if use_plm_adjacency:
            self.projection_g = MLP(plm_rep_dim, 2 * d_model, node_emb_dim)
        
        logging.info("VSDGATRNN initialized with d_in {}, d_model {}, num_of_nodes {}, "
                     "num_heads {}, use_adj_mask: {}, update_all_nodes: {}, use_plm_adjacency: {}, "
                     "use_skip_connection: {}"
                     .format(d_in, d_model, num_of_nodes, num_heads, use_adj_mask, 
                             update_all_nodes, use_plm_adjacency, use_skip_connection))

    def init_hidden_states(self, x):
        # Initialize hidden states with zeros.
        # Input x is expected to be of shape [B, T, N, d_in] so that h becomes [B, N, d_model].
        h0 = torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)
        logging.debug("Initialized hidden states with shape: {}".format(h0.shape))
        return h0

    def forward(self, obs_emb, observed_mask, lengths, avg_interval, var_plm_rep_tensor):
        """
        obs_emb: Observation embeddings, shape [B, T, N, d_in]
        observed_mask: Binary mask indicating observed variables, shape [B, T, N]
        lengths: Tensor indicating the valid sequence length for each sample, shape [B, 1]
        avg_interval: Average interval between observations, shape [B, T, N]
        var_plm_rep_tensor: Pre-trained language model embeddings for variables,
                            shape [B, N, plm_rep_dim] or [N, plm_rep_dim] (broadcast if necessary)
        """
        batch, steps, nodes, features = obs_emb.size()
        device = obs_emb.device
        logging.debug("obs_emb shape: {}, observed_mask shape: {}".format(obs_emb.shape, observed_mask.shape))

        # 1. Initialize the hidden state for every variable for each sample.
        h = self.init_hidden_states(obs_emb)  # Shape: [B, N, d_model]
        
        # 2. Create an identity matrix I (for self connections) and expand to each sample.
        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)
        
        # 3. Prepare an output tensor to eventually store the hidden states for samples at final time steps.
        output = torch.zeros_like(h)
        
        # 4. Mask to record which nodes have ever been observed.
        nodes_initial_mask = torch.zeros(batch, nodes).to(device)
        
        # 5. Total observations per variable over time (summing mask over time).
        var_total_obs = torch.sum(observed_mask, dim=1)  # Shape: [B, N]
        
        # 6. Broadcast the PLM embeddings across the batch if needed.
        var_plm_rep_tensor = repeat(var_plm_rep_tensor, "n d -> b n d", b=batch)
        logging.debug("var_plm_rep_tensor after repeat shape: {}".format(var_plm_rep_tensor.shape))
        
        # 7. Generate query vectors for each variable from the PLM embeddings.
        query_vectors = self.projection_f(var_plm_rep_tensor)  # Shape: [B, N, query_vector_dim]
        
        # 8. Compute static adjacency matrix - either based on PLM embeddings or a fully connected graph
        if self.use_plm_adjacency:
            # Compute node embeddings from textual embeddings and normalize them
            node_embeddings = self.projection_g(var_plm_rep_tensor)  # Shape: [B, N, some_dim]
            normalized_node_embeddings = F.normalize(node_embeddings, p=2, dim=2)
            logging.debug("Normalized node embeddings shape: {}".format(normalized_node_embeddings.shape))
            
            # Compute the static base adjacency matrix using cosine similarity among node embeddings
            adj = torch.softmax(torch.bmm(normalized_node_embeddings, normalized_node_embeddings.permute(0, 2, 1)), dim=-1)
        else:
            # If not using PLM embeddings, create a fully connected adjacency matrix (all 1s)
            # GAT will learn the edge weights through attention
            adj = torch.ones(size=[batch, nodes, nodes]).to(device)
            
        logging.debug("Static base adjacency matrix shape: {}".format(adj.shape))
        
        # 10. Main loop: iterate through time steps (up to max observed length).
        for step in range(int(torch.max(lengths).item())):
            # Create a mask to control which edges are considered at this time step.
            adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)
            
            # Current observation (for all samples at current time) and its corresponding mask.
            cur_obs = obs_emb[:, step]       # Shape: [B, N, d_in]
            cur_mask = observed_mask[:, step]  # Shape: [B, N]
            
            # Find indices of observed variables (using torch.where).
            cur_obs_var = torch.where(cur_mask)
            # Record that these nodes have been observed.
            nodes_initial_mask[cur_obs_var] = 1
            
            # Get the average interval for the current time-step.
            cur_avg_interval = avg_interval[:, step]  # Shape: [B, N]
            # Compute a rarity score based on the average interval relative to total observations.
            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            
            # Determine which nodes to update based on the update_all_nodes flag
            if self.update_all_nodes:
                # Update all nodes, not just the observed ones at this timestep
                # Create indices for all nodes across the batch
                batch_indices = torch.arange(batch, device=device).repeat_interleave(nodes)
                node_indices = torch.arange(nodes, device=device).repeat(batch)
                nodes_need_update = (batch_indices, node_indices)
                
                # For all nodes, create a fully connected adjacency mask
                adj_mask = torch.ones_like(adj)
            else:
                # Only update observed nodes (original behavior)
                nodes_need_update = cur_obs_var
                
                # For observed nodes, set corresponding positions in adj_mask to 1.
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                # For nodes that are unobserved, ensure mask is zero.
                wo_observed_nodes = torch.where(cur_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
            
            # If there are any nodes to update at this timestep
            if nodes_need_update[0].shape[0] > 0:
                if self.use_plm_adjacency:
                    # Create a matrix by repeating the rarity scores for row and column differences
                    rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
                    rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
                    # Combine differences scaled by a learned weight matrix
                    rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
                    
                    # Adjust the base adjacency using the dynamic density (rarity score) and the mask
                    cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                else:
                    # Without PLM-based adjacency, just use the adjacency mask with identity
                    cur_adj = adj_mask * (1 - I) + I
                    
                logging.debug("Dynamic adjacency at step {} has shape: {}".format(step, cur_adj.shape))
                
                # Fill missing observations with zeros for updating all nodes
                if self.update_all_nodes:
                    # Create a tensor of zeros for all positions
                    full_input = torch.zeros(batch, nodes, self.d_in, device=device)
                    # Fill in the observed values
                    full_input[cur_obs_var[0], cur_obs_var[1]] = cur_obs[cur_obs_var[0], cur_obs_var[1]]
                    # Use this as the input
                    update_input = torch.cat([full_input, rarity_score.unsqueeze(-1)], dim=-1)
                else:
                    # Original behavior: only use observed inputs
                    update_input = torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1)
                
                # Update the hidden state using the GAT cell
                h[nodes_need_update] = self.gated_update(
                    update_input,
                    h,
                    query_vectors[nodes_need_update],
                    cur_adj,
                    nodes_need_update
                )
                logging.debug("Updated hidden state for nodes at step {}.".format(step))
            
            # Identify samples for which the current step is the last valid observation.
            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            # Store the hidden states for those samples.
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            
            # Optionally, if at the last step overall, return the collected outputs.
            if step == int(torch.max(lengths).item()) - 1:
                logging.info("Returning final output from VSDGATRNN at step {}.".format(step))
                return output
        
        # In case the loop exits normally (though above branch should have returned at final step).
        return output

class GATGRU(nn.Module):
    def __init__(self, DEVICE=None, hidden_dim=256, num_of_variables=100, num_of_timestamps=100, d_static=0, n_class=1, phe_code_size=1000, task_mode='CONTRASTIVE'):
        super(GATGRU, self).__init__()
        self.DEVICE = DEVICE if DEVICE is not None else get_device()
        self.hidden_dim = hidden_dim
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.d_static = d_static
        self.n_class = n_class
        self.phe_code_size = phe_code_size
        self.task_mode = task_mode

        # ... rest of the existing code ...