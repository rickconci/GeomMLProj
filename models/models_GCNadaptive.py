# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import *
from einops import repeat
import logging
import math
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models.models_utils import MLP, MLP_Param
from utils import device


# Configure logging for debugging - file only, no console output
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('logs/models.log')
    ]
)


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


class VSDGCRNN(nn.Module):
    def __init__(self, d_in, d_model, num_of_nodes, rarity_alpha=0.5, 
                 query_vector_dim=5, node_emb_dim=8, plm_rep_dim=768):
        super(VSDGCRNN, self).__init__()
        self.d_in = d_in  # Input dimension for observations.
        self.d_model = d_model  # Hidden (model) dimension.
        self.num_of_nodes = num_of_nodes  # Number of variables.
        
        # Our customized GRU cell with variable-specific parameters:
        self.gated_update = AGCRNCellWithMLP(d_model, query_vector_dim)
        
        # Rarity parameter: scales the influence of sampling densities.
        self.rarity_alpha = rarity_alpha
        # Learnable matrix used in adjusting density-based weights:
        self.rarity_W = nn.Parameter(torch.randn(num_of_nodes, num_of_nodes))
        self.relu = nn.ReLU()
        
        # Two MLPs to project PLM textual embeddings:
        # (a) For generating query vectors that will modulate the gated updates.
        self.projection_f = MLP(plm_rep_dim, 2 * d_model, query_vector_dim)
        # (b) For obtaining node embeddings to compute a knowledge-aware graph.
        self.projection_g = MLP(plm_rep_dim, 2 * d_model, node_emb_dim)
        logging.info("VSDGCRNN initialized with d_in {}, d_model {}, num_of_nodes {}"
                     .format(d_in, d_model, num_of_nodes))

    def init_hidden_states(self, x):
        # Initialize hidden states with zeros.
        # Input x is expected to be of shape [B, T, N, d_in] so that h becomes [B, N, d_model].
        h0 = torch.zeros(size=(x.shape[0], x.shape[2], self.d_model)).to(x.device)
        logging.debug("Initialized hidden states with shape: {}".format(h0.shape))
        return h0

    def forward(self, obs_emb, observed_mask, lengths, avg_interval, var_plm_rep_tensor):
        """
        obs_emb: Observation embeddings, shape [B, T, V, d_in]
        observed_mask: Binary mask indicating observed variables, shape [B, T, V]
        lengths: Tensor indicating the valid sequence length for each sample, shape [B, 1]
        avg_interval: Average interval between observations, shape [B, T, V]
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
        
        # 8. Compute node embeddings from textual embeddings and normalize them.
        node_embeddings = self.projection_g(var_plm_rep_tensor)  # Shape: [B, N, some_dim]
        normalized_node_embeddings = F.normalize(node_embeddings, p=2, dim=2)
        logging.debug("Normalized node embeddings shape: {}".format(normalized_node_embeddings.shape))
        
        # 9. Compute the static base adjacency matrix using cosine similarity among node embeddings.
        adj = torch.softmax(torch.bmm(normalized_node_embeddings, normalized_node_embeddings.permute(0, 2, 1)), dim=-1)
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
            nodes_need_update = cur_obs_var  # Nodes to update at this time step.
            
            # Get the average interval for the current time-step.
            cur_avg_interval = avg_interval[:, step]  # Shape: [B, N]
            # Compute a rarity score based on the average interval relative to total observations.
            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            logging.debug("Rarity score shape: {}".format(rarity_score.shape))
            
            # Create a matrix by repeating the rarity scores for row and column differences.
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            # Combine differences scaled by a learned weight matrix.
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            
            # If there are any observed nodes at this timestep, update the dynamic graph.
            if nodes_need_update[0].shape[0] > 0:
                # For observed nodes, set corresponding positions in adj_mask to 1.
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                # For nodes that are unobserved, ensure mask is zero.
                wo_observed_nodes = torch.where(cur_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                
                # Adjust the base adjacency using the dynamic density (rarity score) and the mask.
                # This implements a dynamic graph G^(t) where the edge weights vary over time.
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                
                # Prepare input to the gated update by concatenating current observation with rarity score.
                update_input = torch.cat([cur_obs, rarity_score.unsqueeze(-1)], dim=-1)
                
                # Update the hidden state using the AGCRN cell.
                h[nodes_need_update] = self.gated_update(
                    update_input,
                    h,
                    query_vectors[nodes_need_update],
                    cur_adj,
                    nodes_need_update
                )
                logging.debug("Updated hidden state for observed nodes at step {}.".format(step))
            
            # Identify samples for which the current step is the last valid observation
            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            
            # Store the hidden states for those samples
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            
            # Optionally, if at the last step overall, return the collected outputs.
            if step == int(torch.max(lengths).item()) - 1:
                logging.info("Returning final output from VSDGCRNN at step {}.".format(step))
                return output
        
        # In case the loop exits normally (though above branch should have returned at final step).
        return output
    