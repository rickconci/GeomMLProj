# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import *
from einops import repeat
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')


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
                logging.debug("Dynamic adjacency at step {} has shape: {}".format(step, cur_adj.shape))
                
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
            
            # Identify samples for which the current step is the last valid observation.
            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            # Store the hidden states for those samples.
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            
            # Optionally, if at the last step overall, return the collected outputs.
            if step == int(torch.max(lengths).item()) - 1:
                logging.info("Returning final output from VSDGCRNN at step {}.".format(step))
                return output
        
        # In case the loop exits normally (though above branch should have returned at final step).
        return output
    
class KEDGN(nn.Module):
    def __init__(self, DEVICE, hidden_dim, num_of_variables, num_of_timestamps, d_static,
                 n_class, node_enc_layer=2, rarity_alpha=0.5, query_vector_dim=5, node_emb_dim=8, plm_rep_dim=768):
        super(KEDGN, self).__init__()
        
        # Save key parameters
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.hidden_dim = hidden_dim
        self.DEVICE = DEVICE
        
        # Initialize a learnable adjacency matrix (used in the dynamic graph portion)
        self.adj = nn.Parameter(torch.ones(size=[num_of_variables, num_of_variables]))
        
        # Encoders for raw values and absolute time information
        self.value_enc = Value_Encoder(output_dim=hidden_dim)
        self.abs_time_enc = Time_Encoder(embed_time=hidden_dim, var_num=num_of_variables)
        
        # GRU to process observation time patterns with multiple layers
        self.obs_tp_enc = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                                 num_layers=node_enc_layer, batch_first=True, bidirectional=False)
        
        # Observation encoder: combine multiple sources of information before feeding to GCRNN
        self.obs_enc = nn.Sequential(
            nn.Linear(in_features=6 * hidden_dim, out_features=hidden_dim),
            nn.ReLU()
        )
        
        # Embedding for variable type information (each variable gets a learnable embedding)
        self.type_emb = nn.Embedding(num_of_variables, hidden_dim)
        
        # Graph Convolutional Recurrent Neural Network (handles dynamic graph updates for variables)
        self.GCRNN = VSDGCRNN(d_in=self.hidden_dim, d_model=self.hidden_dim,
                              num_of_nodes=num_of_variables, rarity_alpha=rarity_alpha,
                              query_vector_dim=query_vector_dim, node_emb_dim=node_emb_dim,
                              plm_rep_dim=plm_rep_dim)
        
        # Final convolution to process output hidden features before classification (if needed)
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
        # Process static features if available
        self.d_static = d_static
        if d_static != 0:
            self.emb = nn.Linear(d_static, num_of_variables)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables * 2, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)
            ).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables, 200),
                nn.ReLU(),
                nn.Linear(200, n_class)
            )
        
        self.to(DEVICE)
        logging.info("KEDGN initialized on device: {}".format(DEVICE))

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
