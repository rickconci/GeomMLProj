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


if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using CUDA device")
else:
    device = torch.device('cpu')
    print("Using CPU device")


# Configure logging for debugging - file only, no console output
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('logs/models.log')
    ]
)


class DSEncoderWithRNN(nn.Module):
    def __init__(self, model_name="medicalai/ClinicalBERT", rnn_hidden_dim=768, projection_dim=1536):
        """
        model_name: Name of the pretrained ClinicalBERT model.
        rnn_hidden_dim: Hidden size for the GRU (can be equal to the transformer hidden size).
        projection_dim: Final dimension for contrastive learning, e.g. 2 * hidden_dim.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        hidden_dim = self.transformer.config.hidden_size  # e.g., 768
        
        # GRU to process sequence of chunk embeddings.
        self.gru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=rnn_hidden_dim, 
            batch_first=True
        )
        
        # Projection layer: maps the GRU output to your desired embedding dimension.
        self.projection = nn.Sequential(
            nn.Linear(rnn_hidden_dim, projection_dim),
            nn.ReLU()  # Optional activation, adjust as needed.
        )
    
    def forward(self, discharge_chunks, output_dim=None):
        """
        discharge_chunks: A list of samples.
                          Each sample is a list of text chunks (strings).
                          
        output_dim: Optional output dimension override
        
        The function returns a tensor of shape [B, projection_dim or output_dim].
        """
        batch_size = len(discharge_chunks)
        all_chunks = []   # To store all chunk texts across the batch.
        sample_indices = []  # To track which sample each chunk belongs to.
        
        # If all chunks are empty, return a tensor of zeros
        if all(len(chunks) == 0 for chunks in discharge_chunks):
            if output_dim is None:
                output_dim = self.projection[0].out_features
            return torch.zeros(batch_size, output_dim, device=device)
        
        # Flatten the list of lists.
        for i, chunks in enumerate(discharge_chunks):
            for chunk in chunks:
                all_chunks.append(chunk)
                sample_indices.append(i)
        
        # If no chunks after filtering, return zeros
        if len(all_chunks) == 0:
            if output_dim is None:
                output_dim = self.projection[0].out_features
            return torch.zeros(batch_size, output_dim, device=device)
        
        # Tokenize all chunks in one call.
        inputs = self.tokenizer(all_chunks, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Process with ClinicalBERT using the approach that works
        with torch.no_grad():
            self.transformer.eval()  # Ensure model is in eval mode
            outputs = self.transformer(**inputs, output_hidden_states=True)
            # Extract embeddings from the last hidden state's first token (CLS)
            last_hidden_states = outputs.hidden_states[-1]  # shape: (num_chunks, seq_len, hidden_size)
            cls_embeddings = last_hidden_states[:, 0, :]  # shape: (num_chunks, hidden_size)
        
        # Reassemble the embeddings into per-sample lists
        batch_embeddings = [[] for _ in range(batch_size)]
        for idx, emb in zip(sample_indices, cls_embeddings):
            batch_embeddings[idx].append(emb)
        
        # Process each sample separately
        projected_embeddings = []
        for i, embeddings in enumerate(batch_embeddings):
            if not embeddings:
                # If this sample has no embeddings, use zeros
                if output_dim is None:
                    output_dim = self.projection[0].out_features
                projected_embeddings.append(torch.zeros(output_dim, device=device))
                continue
                
            # Stack this sample's embeddings
            sample_emb = torch.stack(embeddings, dim=0)  # [num_chunks, hidden_dim]
            
            # Process with GRU to handle variable number of chunks
            _, h_n = self.gru(sample_emb.unsqueeze(0))  # Add batch dimension
            
            # Get final hidden state
            final_state = h_n[0]  # [1, hidden_dim]
            
            # Project to final dimension
            proj = self.projection(final_state.squeeze(0))  # [projection_dim]
            projected_embeddings.append(proj)
        
        # Stack all sample embeddings
        return torch.stack(projected_embeddings, dim=0)  # [batch_size, projection_dim]



class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        return self.projection(x)




def build_cluster_adjacencies(cluster_labels):
    """
    Build two sets of edge indices for fully connected networks:
      (a) Intra-cluster connectivity: edges among sensors in the same cluster.
      (b) Inter-cluster connectivity: edges among sensors in different clusters.

    Args:
        cluster_labels (torch.LongTensor): Shape [N], where N is the number of sensors.
                                            Each element is an integer cluster id.

    Returns:
        edge_index_intra (torch.LongTensor): Shape [2, E_intra] for intra-cluster edges.
        edge_index_inter (torch.LongTensor): Shape [2, E_inter] for inter-cluster edges.
    """
    device = cluster_labels.device
    n_sensors = cluster_labels.size(0)

    # Build a dictionary mapping cluster id to the indices of sensors in that cluster.
    clusters = {}
    unique_clusters = torch.unique(cluster_labels)
    for c in unique_clusters.tolist():
        clusters[c] = torch.nonzero(cluster_labels == c, as_tuple=True)[0]

    # (1) Build Intra-Cluster Edge Index:
    intra_edge_list = []
    for c, indices in clusters.items():
        if indices.numel() == 0:
            continue
        # Generate a fully connected graph for sensors in this cluster (excluding self-loops).
        row = indices.view(-1, 1).repeat(1, indices.size(0)).view(-1)
        col = indices.view(1, -1).repeat(indices.size(0), 1).view(-1)
        # Remove self-loops (if not desired):
        mask = row != col
        row, col = row[mask], col[mask]
        intra_edge_list.append(torch.stack([row, col], dim=0))
    if intra_edge_list:
        edge_index_intra = torch.cat(intra_edge_list, dim=1).unique(dim=1)
    else:
        edge_index_intra = torch.empty((2, 0), dtype=torch.long, device=device)

    # (2) Build Inter-Cluster Edge Index:
    inter_edge_list = []
    unique_cluster_list = sorted(clusters.keys())
    # For each pair of distinct clusters, generate fully connected edges
    for i, c1 in enumerate(unique_cluster_list):
        for c2 in unique_cluster_list[i+1:]:
            indices1 = clusters[c1]
            indices2 = clusters[c2]
            if indices1.numel() == 0 or indices2.numel() == 0:
                continue
            # Create edges in one direction (from cluster c1 to c2)
            row = indices1.view(-1, 1).repeat(1, indices2.size(0)).view(-1)
            col = indices2.view(1, -1).repeat(indices1.size(0), 1).view(-1)
            edges_c1c2 = torch.stack([row, col], dim=0)
            # Create the reverse direction (from c2 to c1)
            edges_c2c1 = torch.stack([col, row], dim=0)
            inter_edge_list.append(edges_c1c2)
            inter_edge_list.append(edges_c2c1)
    if inter_edge_list:
        edge_index_inter = torch.cat(inter_edge_list, dim=1).unique(dim=1)
    else:
        edge_index_inter = torch.empty((2, 0), dtype=torch.long, device=device)

    return edge_index_intra.to(device), edge_index_inter.to(device)





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
        
        # More memory-efficient GAT implementation
        self.query = nn.ModuleList([nn.Linear(combined_dim, combined_dim // 8) for _ in range(num_heads)])
        self.key = nn.ModuleList([nn.Linear(combined_dim, combined_dim // 8) for _ in range(num_heads)])
        self.value = nn.ModuleList([nn.Linear(combined_dim, combined_dim) for _ in range(num_heads)])
        
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

class VSDGATRNN(nn.Module):
    def __init__(self, d_in, d_model, num_of_nodes, rarity_alpha=0.5, 
                 query_vector_dim=5, node_emb_dim=8, plm_rep_dim=768, num_heads=1, use_adj_mask=False):
        super(VSDGATRNN, self).__init__()
        self.d_in = d_in  # Input dimension for observations.
        self.d_model = d_model  # Hidden (model) dimension.
        self.num_of_nodes = num_of_nodes  # Number of variables.
        
        # Use GAT cell instead of GCN cell
        self.gated_update = AGATCellWithMLP(d_model, query_vector_dim, num_heads, use_adj_mask)
        
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
        logging.info("VSDGATRNN initialized with d_in {}, d_model {}, num_of_nodes {}, num_heads {}, use_adj_mask: {}"
                     .format(d_in, d_model, num_of_nodes, num_heads, use_adj_mask))

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
                # Update the hidden state using the GAT cell.
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
                logging.info("Returning final output from VSDGATRNN at step {}.".format(step))
                return output
        
        # In case the loop exits normally (though above branch should have returned at final step).
        return output

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
                 n_class, node_enc_layer=2, rarity_alpha=0.5, query_vector_dim=5, 
                 node_emb_dim=8, plm_rep_dim=768, use_gat=False, num_heads=2, use_adj_mask=False,
                 use_transformer=False, history_len=10, nhead_transformer=2):
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
        
        # Choose between different models:
        # 1. VSDGCRNN (original GRU + GCN)
        # 2. VSDGATRNN (GRU + GAT)
        # 3. VSDTransformerGATRNN (Transformer + GAT)
        
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
            
        Returns:
            output: Classification logits [B, n_class]
            aggregated_hidden: Aggregated hidden state [B, V]
            fused_features: Features for contrastive learning [B, V*2] or [B, V]
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
            fused_features = aggregated_hidden
            output = self.classifier(aggregated_hidden)
        
        logging.info("Output shape: {}".format(output.shape))
        # Return classification output, aggregated state, and intermediate representation
        return output, aggregated_hidden, fused_features

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

class VSDTransformerGATRNN(nn.Module):
    def __init__(self, d_in, d_model, num_of_nodes, history_len=10, 
                 query_vector_dim=5, node_emb_dim=8, plm_rep_dim=768, 
                 num_heads=2, use_adj_mask=False, nhead_transformer=2):
        super(VSDTransformerGATRNN, self).__init__()
        self.d_in = d_in  # Input dimension for observations.
        self.d_model = d_model  # Hidden (model) dimension.
        self.num_of_nodes = num_of_nodes  # Number of variables.
        self.history_len = history_len
        
        # Create transformer cells for each variable
        self.transformer_cells = nn.ModuleList([
            VariableTransformerCell(
                input_size=d_in,
                hidden_size=d_model,
                nhead=nhead_transformer
            ) for _ in range(num_of_nodes)
        ])
        
        # Use GAT cell for cross-variable message passing
        self.gated_update = AGATCellWithMLP(d_model, query_vector_dim, num_heads, use_adj_mask)
        
        # Rarity parameter: scales the influence of sampling densities.
        self.rarity_alpha = 0.0  # We can disable this since the transformer should capture temporal patterns
        
        # Learnable matrix used in adjusting density-based weights:
        self.rarity_W = nn.Parameter(torch.randn(num_of_nodes, num_of_nodes))
        self.relu = nn.ReLU()
        
        # Two MLPs to project PLM textual embeddings:
        # (a) For generating query vectors that will modulate the gated updates.
        self.projection_f = MLP(plm_rep_dim, 2 * d_model, query_vector_dim)
        # (b) For obtaining node embeddings to compute a knowledge-aware graph.
        self.projection_g = MLP(plm_rep_dim, 2 * d_model, node_emb_dim)
        
        logging.info("VSDTransformerGATRNN initialized with d_in {}, d_model {}, num_of_nodes {}, "
                     "history_len {}, num_heads {}, use_adj_mask: {}, nhead_transformer: {}"
                     .format(d_in, d_model, num_of_nodes, history_len, 
                             num_heads, use_adj_mask, nhead_transformer))

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

        # Initialize history buffer for each variable with both features and timestamps
        # We'll store a history of observations for each variable
        history = torch.zeros(batch, nodes, self.history_len, features, device=device)
        # Store absolute timestamps for each observation
        history_times = torch.zeros(batch, nodes, self.history_len, device=device)
        history_masks = torch.zeros(batch, nodes, self.history_len, dtype=torch.bool, device=device)
        
        # Initialize hidden states
        h = self.init_hidden_states(obs_emb)  # Shape: [B, N, d_model]
        
        # Create an identity matrix I (for self connections) and expand to each sample.
        I = repeat(torch.eye(nodes).to(device), 'v x -> b v x', b=batch)
        
        # Prepare an output tensor to eventually store the hidden states for samples at final time steps.
        output = torch.zeros_like(h)
        
        # Mask to record which nodes have ever been observed.
        nodes_initial_mask = torch.zeros(batch, nodes).to(device)
        
        # Total observations per variable over time (summing mask over time).
        var_total_obs = torch.sum(observed_mask, dim=1)  # Shape: [B, N]
        
        # Broadcast the PLM embeddings across the batch if needed.
        var_plm_rep_tensor = repeat(var_plm_rep_tensor, "n d -> b n d", b=batch)
        logging.debug("var_plm_rep_tensor after repeat shape: {}".format(var_plm_rep_tensor.shape))
        
        # Generate query vectors for each variable from the PLM embeddings.
        query_vectors = self.projection_f(var_plm_rep_tensor)  # Shape: [B, N, query_vector_dim]
        
        # Compute node embeddings from textual embeddings and normalize them.
        node_embeddings = self.projection_g(var_plm_rep_tensor)  # Shape: [B, N, some_dim]
        normalized_node_embeddings = F.normalize(node_embeddings, p=2, dim=2)
        logging.debug("Normalized node embeddings shape: {}".format(normalized_node_embeddings.shape))
        
        # Compute the static base adjacency matrix using cosine similarity among node embeddings.
        adj = torch.softmax(torch.bmm(normalized_node_embeddings, normalized_node_embeddings.permute(0, 2, 1)), dim=-1)
        logging.debug("Static base adjacency matrix shape: {}".format(adj.shape))
        
        # Main temporal loop
        for step in range(int(torch.max(lengths).item())):
            # Current observations and mask
            cur_obs = obs_emb[:, step]        # [B, N, d_in]
            cur_mask = observed_mask[:, step]  # [B, N]
            cur_obs_var = torch.where(cur_mask)
            
            # Extract current absolute time points or use step as a proxy
            # In a real application, you would use actual timestamps here
            cur_times = torch.full((batch, nodes), float(step), device=device)
            
            # Record nodes that have been observed
            nodes_initial_mask[cur_obs_var] = 1
            nodes_need_update = cur_obs_var  # Nodes to update at this time step.
            
            # Shift history buffer and add new observation
            if step > 0:  # Only shift after the first step
                history = torch.roll(history, shifts=-1, dims=2)
                history_times = torch.roll(history_times, shifts=-1, dims=2)
                history_masks = torch.roll(history_masks, shifts=-1, dims=2)
            
            # Add current observation to history for observed variables
            for idx in range(len(cur_obs_var[0])):
                b, n = cur_obs_var[0][idx], cur_obs_var[1][idx]
                history[b, n, -1] = cur_obs[b, n]
                history_times[b, n, -1] = cur_times[b, n]
                history_masks[b, n, -1] = True
            
            # Apply transformer to history for observed variables
            transformed_states = h.clone()  # Start with previous hidden states
            
            # Process each observed variable
            for idx in range(len(cur_obs_var[0])):
                b, n = cur_obs_var[0][idx], cur_obs_var[1][idx]
                
                # Get history for this variable (only if we have observations)
                if torch.any(history_masks[b, n]):
                    var_history = history[b, n].unsqueeze(0)  # [1, history_len, d_in]
                    var_times = history_times[b, n].unsqueeze(0)  # [1, history_len]
                    var_mask = ~history_masks[b, n].unsqueeze(0)  # [1, history_len]
                    
                    # Apply transformer with time-aware positional encoding
                    transformed_states[b, n] = self.transformer_cells[n](
                        var_history, var_times, var_mask
                    )
            
            # Get the average interval for the current time-step (needed for adjacency computation)
            cur_avg_interval = avg_interval[:, step]  # Shape: [B, N]
            
            # Compute a rarity score based on the average interval (can be tuned or disabled)
            rarity_score = self.rarity_alpha * torch.tanh(cur_avg_interval / (var_total_obs + 1))
            
            # Create a matrix by repeating the rarity scores for row and column differences.
            rarity_score_matrix_row = repeat(rarity_score, 'b v -> b v x', x=nodes)
            rarity_score_matrix_col = repeat(rarity_score, 'b v -> b x v', x=nodes)
            
            # Combine differences scaled by a learned weight matrix.
            rarity_score_matrix = -1 * self.rarity_W * (torch.abs(rarity_score_matrix_row - rarity_score_matrix_col))
            
            # If there are any observed nodes at this timestep, update the message passing
            if nodes_need_update[0].shape[0] > 0:
                # For observed nodes, set corresponding positions in adj_mask to 1
                adj_mask = torch.zeros(size=[batch, nodes, nodes]).to(device)
                adj_mask[cur_obs_var[0], :, cur_obs_var[1]] = torch.ones(len(cur_obs_var[0]), nodes).to(device)
                
                # For nodes that are unobserved, ensure mask is zero
                wo_observed_nodes = torch.where(cur_mask == 0)
                adj_mask[wo_observed_nodes] = torch.zeros(len(wo_observed_nodes[0]), nodes).to(device)
                
                # Adjust the base adjacency using the dynamic density (rarity score) and the mask
                cur_adj = adj * (1 + rarity_score_matrix) * adj_mask * (1 - I) + I
                
                # Prepare input to the gated update by concatenating transformed state with rarity score
                update_input = torch.cat([transformed_states, rarity_score.unsqueeze(-1)], dim=-1)
                
                # Apply GAT message passing
                h[nodes_need_update] = self.gated_update(
                    update_input,
                    h,
                    query_vectors[nodes_need_update],
                    cur_adj,
                    nodes_need_update
                )
            
            # Identify samples for which the current step is the last valid observation
            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            
            # Store the hidden states for those samples
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            
            # If at the last step overall, return the collected outputs
            if step == int(torch.max(lengths).item()) - 1:
                logging.info("Returning final output from VSDTransformerGATRNN at step {}.".format(step))
                return output
        
        # In case the loop exits normally (though above branch should have returned at final step)
        return output
