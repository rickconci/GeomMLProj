# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from train_utils import *
from einops import *
from einops import repeat
import logging
import math
import os
from models.models_utils import MLP_Param, MLP



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



class ClusterGATCell(nn.Module):
    def __init__(self, input_size, query_vector_dim, num_heads=1, use_adj_mask=False, skip_cluster_to_node=False):
        super(ClusterGATCell, self).__init__()
        # The input to each gate is a concatenation of x and h, plus an extra dimension (+1)
        # Replace dynamic parameter generation with standard GRU gates
        self.combined_dim = 2 * input_size  # x + h 
        
        # Static GRU for cluster processing (no more dynamic MLP_Param)
        self.cluster_gru = nn.GRUCell(input_size, input_size)
        
        # GAT attention components for first-stage message passing
        self.num_heads = num_heads
        
        # First stage: Inter-node GAT (across all observable nodes)
        self.query_inter = nn.ModuleList([nn.Linear(self.combined_dim, self.combined_dim // 8) for _ in range(num_heads)])
        self.key_inter = nn.ModuleList([nn.Linear(self.combined_dim, self.combined_dim // 8) for _ in range(num_heads)])
        self.value_inter = nn.ModuleList([nn.Linear(self.combined_dim, self.combined_dim) for _ in range(num_heads)])

        # MLP to process inter-node attention heads
        self.interAttentionMLP = MLP(num_heads*self.combined_dim, self.combined_dim, self.combined_dim)
        
        # Flag to skip cluster-to-node update entirely
        self.skip_cluster_to_node = skip_cluster_to_node
        
        # Cluster-to-node projection (for redistributing cluster hidden states)
        if not skip_cluster_to_node:
            self.cluster_to_node_proj = nn.Linear(input_size, input_size)
            # Initialize projection with smaller weights for stability
            with torch.no_grad():
                nn.init.xavier_normal_(self.cluster_to_node_proj.weight, gain=0.1)
                if self.cluster_to_node_proj.bias is not None:
                    nn.init.zeros_(self.cluster_to_node_proj.bias)
        
        # Add small constant for numerical stability
        self.eps = 1e-6
        
        # Add temperature parameter for softmax to prevent attention collapse
        self.attention_temperature = 0.5  # Lower values = softer attention distribution
        
        self.use_adj_mask = use_adj_mask
        logging.info("ClusterGATCell initialized with input_size {}, combined dim {}, query_vector_dim {}, num_heads {}, use_adj_mask: {}, attention_temperature: {}, skip_cluster_to_node: {}"
                     .format(input_size, self.combined_dim, query_vector_dim, num_heads, use_adj_mask, self.attention_temperature, skip_cluster_to_node))

    def forward(self, x, h, cluster_h, query_vectors, cluster_labels, nodes_ind, edge_index_intra, num_clusters):
        """
        x: Current input features for all nodes. Shape: [num_nodes, input_size]
        h: Node hidden state from the previous time step. Shape: [num_nodes, input_size]
        cluster_h: Cluster hidden state from the previous time step. Shape: [num_clusters, input_size]
        query_vectors: Cluster-specific query vectors (not used in this simplified version)
        cluster_labels: Cluster assignments for each node. Shape: [num_nodes]
        nodes_ind: Indices for nodes that are observed at the current timestamp.
        edge_index_intra: Edge indices for intra-cluster connections. Shape: [2, E_intra]
        num_clusters: Number of clusters
        
        Returns:
            updated_h: Updated node hidden states. Shape: [num_nodes, input_size]
            updated_cluster_h: Updated cluster hidden states. Shape: [num_clusters, input_size]
        """
        device = x.device
        num_nodes = x.shape[0]
        input_size = h.shape[1]
        
        # Check input for NaNs
        logging.debug(f"[DIAGNOSTIC] Input x has_nan={torch.isnan(x).any().item()}")
        logging.debug(f"[DIAGNOSTIC] Input h has_nan={torch.isnan(h).any().item()}")
        logging.debug(f"[DIAGNOSTIC] Input cluster_h has_nan={torch.isnan(cluster_h).any().item()}")
        
        # ENHANCED DIAGNOSTICS: Track each node observation
        for i in range(min(10, x.shape[0])):  # Check first 10 nodes at most
            logging.debug(f"[DIAGNOSTIC] Node {nodes_ind[i].item()} input: x_has_nan={torch.isnan(x[i]).any().item()}, h_has_nan={torch.isnan(h[i]).any().item()}")

        # 1. Concatenate current input and previous hidden state along last dimension.
        combined = torch.cat([x, h], dim=-1)  # Shape: [num_nodes, 2*input_size]
        logging.debug(f"[DIAGNOSTIC] After concat: combined has_nan={torch.isnan(combined).any().item()}")
        
        # 2. FIRST STAGE GAT: Message passing across all observable nodes
        # Multi-head attention computation for inter-node communication
        inter_attention_outputs = []
        
        # Process each attention head separately
        for head in range(self.num_heads):
            # Transform features
            queries = self.query_inter[head](combined)  # [N, dim//8]
            keys = self.key_inter[head](combined)       # [N, dim//8]
            values = self.value_inter[head](combined)   # [N, dim]
            
            logging.debug(f"[DIAGNOSTIC] Head {head}: queries has_nan={torch.isnan(queries).any().item()}")
            logging.debug(f"[DIAGNOSTIC] Head {head}: keys has_nan={torch.isnan(keys).any().item()}")
            logging.debug(f"[DIAGNOSTIC] Head {head}: values has_nan={torch.isnan(values).any().item()}")
            
            # Reshape for batch matrix multiplication
            queries = queries.unsqueeze(0)  # [1, N, dim//8]
            keys = keys.unsqueeze(0)        # [1, N, dim//8]
            values = values.unsqueeze(0)    # [1, N, dim]
            
            # Compute attention scores
            attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.size(-1))
            attn_scores = F.leaky_relu(attn_scores, 0.2)
            logging.debug(f"[DIAGNOSTIC] Head {head}: attn_scores has_nan={torch.isnan(attn_scores).any().item()}")
            
            # Attention weights with softmax normalization
            attention = F.softmax(attn_scores, dim=2)
            logging.debug(f"[DIAGNOSTIC] Head {head}: attention has_nan={torch.isnan(attention).any().item()}")
            
            # Apply attention weights
            h_prime = torch.bmm(attention, values)
            inter_attention_outputs.append(h_prime.squeeze(0))
            logging.debug(f"[DIAGNOSTIC] Head {head}: h_prime has_nan={torch.isnan(h_prime).any().item()}")
        
        # Process multi-head attention through MLP
        # Stack outputs: [num_heads, N, dim] -> reshape to [N, num_heads*dim]
        inter_attention_stacked = torch.stack(inter_attention_outputs, dim=0)
        inter_attention_reshaped = inter_attention_stacked.permute(1, 0, 2).reshape(num_nodes, -1)
        node_representations_inter = self.interAttentionMLP(inter_attention_reshaped)
        logging.debug(f"[DIAGNOSTIC] After MLP: node_representations_inter has_nan={torch.isnan(node_representations_inter).any().item()}")
        
        # Skip connection - add original input to prevent information loss
        node_representations = node_representations_inter + combined
        logging.debug(f"[DIAGNOSTIC] After skip connection: node_representations has_nan={torch.isnan(node_representations).any().item()}")
        
        # 3. Super-simple cluster aggregation - just average the hidden states
        # Initialize aggregated cluster features with zeros
        aggregated_cluster_features = torch.zeros(num_clusters, input_size, device=device)
        
        # Track which clusters are actually present
        clusters_present = torch.zeros(num_clusters, dtype=torch.bool, device=device)
        
        # ENHANCED DIAGNOSTICS: Check which node hidden states have NaNs per cluster BEFORE aggregation
        problem_clusters = [1, 3, 4, 7, 8]  # Clusters that showed NaNs in previous runs
        
        # IMPORTANT: Inside ClusterGATCell.forward, 'h' only contains states for observed nodes
        # We need to check each node in nodes_ind and find its cluster
        for i, node_idx in enumerate(nodes_ind[:10]):  # Check first 10 nodes at most
            # Get the cluster this node belongs to
            node_cluster = cluster_labels[node_idx].item()
            if node_cluster in problem_clusters:
                # This node is in a problem cluster
                node_h = h[i]  # Use local index i, not global node_idx
                has_nan = torch.isnan(node_h).any().item()
                if has_nan:
                    logging.debug(f"[DIAGNOSTIC] BEFORE AGGREGATION: Node {node_idx.item()} in problem cluster {node_cluster} has NaN in hidden state")
        
        # Aggregate node features by cluster using simple average
        for c in range(num_clusters):
            # Find nodes in this cluster
            cluster_nodes = (cluster_labels == c).nonzero().squeeze(-1)
            if cluster_nodes.numel() > 0:
                # Extract only the hidden state part (h) from the combined representation
                # The combined tensor is [x, h], so we want the second half
                cluster_node_hidden = h[cluster_nodes]  # [cluster_size, input_size]
                
                # Simple average
                aggregated_cluster_features[c] = torch.mean(cluster_node_hidden, dim=0)
                clusters_present[c] = True
                
                logging.debug(f"[DIAGNOSTIC] Cluster {c}: cluster_node_hidden has_nan={torch.isnan(cluster_node_hidden).any().item()}, agg_features has_nan={torch.isnan(aggregated_cluster_features[c]).any().item()}")
        
        logging.debug(f"[DIAGNOSTIC] All aggregated_cluster_features has_nan={torch.isnan(aggregated_cluster_features).any().item()}")
        
        # 4. Apply standard GRU update at the cluster level for present clusters
        updated_cluster_h = cluster_h.clone()  # Start with previous hidden state
        
        # Only update clusters that have nodes in this batch
        for c in range(num_clusters):
            if clusters_present[c]:
                try:
                    # Use standard GRU cell for update
                    cluster_input = aggregated_cluster_features[c].unsqueeze(0)
                    cluster_hidden = cluster_h[c].unsqueeze(0)
                    
                    logging.debug(f"[DIAGNOSTIC] Cluster {c} GRU input: cluster_input has_nan={torch.isnan(cluster_input).any().item()}, cluster_hidden has_nan={torch.isnan(cluster_hidden).any().item()}")
                    
                    # Debug the GRU weights
                    if c == 0:  # Only check once
                        for name, param in self.cluster_gru.named_parameters():
                            logging.debug(f"[DIAGNOSTIC] GRU param {name}: min={param.min().item():.4f}, max={param.max().item():.4f}, norm={torch.norm(param).item():.4f}")
                    
                    updated_h_c = self.cluster_gru(cluster_input, cluster_hidden)
                    updated_cluster_h[c] = updated_h_c.squeeze(0)
                    
                    logging.debug(f"[DIAGNOSTIC] After GRU: updated_cluster_h[{c}] has_nan={torch.isnan(updated_cluster_h[c]).any().item()}")
                except Exception as e:
                    logging.error(f"[DIAGNOSTIC] Exception in GRU update for cluster {c}: {str(e)}")
                    # Keep previous hidden state
        
        logging.debug(f"[DIAGNOSTIC] All updated_cluster_h has_nan={torch.isnan(updated_cluster_h).any().item()}")
        
        # 5. Distribute cluster hidden states back to nodes
        # Clone to ensure we're not modifying the original tensor
        final_updated_h = h.clone()
        
        # Skip cluster-to-node update if specified
        if self.skip_cluster_to_node:
            logging.debug(f"[DIAGNOSTIC] Skipping cluster-to-node update as requested")
            return final_updated_h, updated_cluster_h
        
        # Update nodes based on their cluster's updated hidden state
        for c in range(num_clusters):
            # Find nodes in this cluster
            cluster_nodes = (cluster_labels == c).nonzero().squeeze(-1)
            if cluster_nodes.numel() > 0:
                # Project cluster state for node update with simple linear projection
                try:
                    cluster_state = updated_cluster_h[c]
                    logging.debug(f"[DIAGNOSTIC] Cluster {c} state for projection has_nan={torch.isnan(cluster_state).any().item()}")
                    
                    # Get detailed stats on cluster state
                    logging.debug(f"[DIAGNOSTIC] Cluster {c} state stats: min={cluster_state.min().item():.6f}, max={cluster_state.max().item():.6f}, mean={cluster_state.mean().item():.6f}, std={cluster_state.std().item():.6f}")
                    
                    # Check projection weights for extreme values
                    w_min = self.cluster_to_node_proj.weight.min().item()
                    w_max = self.cluster_to_node_proj.weight.max().item()
                    w_mean = self.cluster_to_node_proj.weight.mean().item()
                    w_std = self.cluster_to_node_proj.weight.std().item()
                    logging.debug(f"[DIAGNOSTIC] Projection weights stats: min={w_min:.6f}, max={w_max:.6f}, mean={w_mean:.6f}, std={w_std:.6f}")
                    
                    # Apply the projection
                    cluster_state_projected = self.cluster_to_node_proj(cluster_state)
                    logging.debug(f"[DIAGNOSTIC] Cluster {c} projected state has_nan={torch.isnan(cluster_state_projected).any().item()}")
                    
                    # Get detailed stats on projected state
                    logging.debug(f"[DIAGNOSTIC] Projected state stats: min={cluster_state_projected.min().item():.6f}, max={cluster_state_projected.max().item():.6f}, mean={cluster_state_projected.mean().item():.6f}, std={cluster_state_projected.std().item():.6f}")
                    
                    # Check previous node states
                    node_states = final_updated_h[cluster_nodes]
                    logging.debug(f"[DIAGNOSTIC] Node states stats: min={node_states.min().item():.6f}, max={node_states.max().item():.6f}, mean={node_states.mean().item():.6f}, std={node_states.std().item():.6f}")
                    
                    # DEBUG SPECIFIC ELEMENTS - find extreme values that might cause issues
                    # Get indices of max values in both tensors
                    max_idx_proj = torch.argmax(torch.abs(cluster_state_projected))
                    max_val_proj = cluster_state_projected.flatten()[max_idx_proj].item()
                    
                    # Check what happens during the addition operation with max values
                    # Pick a specific node to analyze (first node in cluster)
                    if cluster_nodes.numel() > 0:
                        test_node_idx = cluster_nodes[0]
                        node_state = final_updated_h[test_node_idx]
                        logging.debug(f"[DIAGNOSTIC] Test node {test_node_idx} (cluster {c}) state values: {node_state[:5].tolist()} ... (first 5 values)")
                        logging.debug(f"[DIAGNOSTIC] Projected values to add: {cluster_state_projected[:5].tolist()} ... (first 5 values)")
                        
                        # Test the addition element by element for the first few elements
                        for i in range(min(5, node_state.shape[0])):
                            a = node_state[i].item()
                            b = cluster_state_projected[i].item()
                            result = a + b
                            logging.debug(f"[DIAGNOSTIC] Addition test: {a} + {b} = {result}, is_nan={math.isnan(result)}")
                    
                    # Perform the actual addition (without clipping)
                    updated_nodes = node_states + cluster_state_projected
                    
                    # Check where NaNs occur (if any)
                    if torch.isnan(updated_nodes).any():
                        nan_mask = torch.isnan(updated_nodes)
                        num_nans = nan_mask.sum().item()
                        logging.warning(f"[DIAGNOSTIC] {num_nans} NaNs detected in updated nodes for cluster {c}")
                        
                        # Get indices of first few NaNs
                        nan_indices = torch.where(nan_mask)[0][:5].tolist()  # Get up to 5 NaN indices
                        for idx in nan_indices:
                            node_idx = cluster_nodes[idx]
                            a = node_states[idx].detach().cpu()
                            b = cluster_state_projected.detach().cpu()
                            logging.warning(f"[DIAGNOSTIC] NaN at node {node_idx}, index {idx}: node_value={a} + projected={b}")
                            
                            # Check for infinities that might cause NaNs
                            if torch.isinf(a).any() or torch.isinf(b).any():
                                logging.warning(f"[DIAGNOSTIC] Infinity detected! node_value_has_inf={torch.isinf(a).any().item()}, projected_has_inf={torch.isinf(b).any().item()}")
                        
                        # Use node states as fallback to avoid NaNs
                        updated_nodes = torch.where(torch.isnan(updated_nodes), node_states, updated_nodes)
                    
                    # Now assign back to final updated hidden states
                    final_updated_h[cluster_nodes] = updated_nodes
                    
                    logging.debug(f"[DIAGNOSTIC] Final nodes in cluster {c} has_nan={torch.isnan(final_updated_h[cluster_nodes]).any().item()}")
                except Exception as e:
                    logging.error(f"[DIAGNOSTIC] Exception in cluster-to-node projection for cluster {c}: {str(e)}")
                    # Keep previous node states
        
        logging.debug(f"[DIAGNOSTIC] Final updated_h has_nan={torch.isnan(final_updated_h).any().item()}")
        logging.debug(f"[DIAGNOSTIC] Final updated_cluster_h has_nan={torch.isnan(updated_cluster_h).any().item()}")
        
        # Check for NaNs after update
        if torch.isnan(final_updated_h).any() or torch.isnan(updated_cluster_h).any():
            logging.warning(f"Step {step}: NaNs detected after node/cluster update: h={torch.isnan(final_updated_h).any().item()}, cluster_h={torch.isnan(updated_cluster_h).any().item()}")
        
        logging.debug(f"Updated hidden state for observed nodes at step {step}.")
        
        # ENHANCED DIAGNOSTICS: Check for NaNs in h after update by sampling a few nodes 
        if step == 0:  # Only after first timestep
            # Track NaN counts by cluster (for first batch)
            nan_counts = {c: 0 for c in range(self.num_clusters)}
            total_counts = {c: 0 for c in range(self.num_clusters)}
            
            # Check a subset of nodes from each cluster 
            sample_size = min(20, h.shape[1] // self.num_clusters)
            for node_idx in range(min(200, h.shape[1])):  # Sample up to 200 nodes
                if node_idx < h.shape[1]:  # Safety check
                    node_h = h[0, node_idx]
                    cluster_id = cluster_labels[0, node_idx].item() 
                    total_counts[cluster_id] += 1
                    
                    if torch.isnan(node_h).any().item():
                        nan_counts[cluster_id] += 1
                        if nan_counts[cluster_id] <= 3:  # Log only first 3 NaNs per cluster
                            logging.debug(f"Step {step}: Node {node_idx} in cluster {cluster_id} has NaN after update")
            
            # Log summary statistics
            for c in range(self.num_clusters):
                if total_counts[c] > 0:
                    nan_percentage = 100 * nan_counts[c] / total_counts[c]
                    if nan_counts[c] > 0:
                        logging.warning(f"Step {step}: Cluster {c} has {nan_counts[c]}/{total_counts[c]} nodes with NaNs ({nan_percentage:.1f}%)")
                    else:
                        logging.debug(f"Step {step}: Cluster {c} has no NaNs in {total_counts[c]} sampled nodes")
        
        return final_updated_h, updated_cluster_h

class ClusterBasedVSDGATRNN(nn.Module):
    def __init__(self, 
                 d_in, 
                 d_model, 
                 num_of_nodes, 
                 cluster_labels,
                 query_vector_dim=5, 
                 node_emb_dim=8, 
                 plm_rep_dim=768, 
                 num_heads=1, 
                 use_adj_mask=False,
                 skip_cluster_to_node=True):  # Default to cluster-only mode (no update back to nodes)
        super(ClusterBasedVSDGATRNN, self).__init__()
        self.d_in = d_in  # Input dimension for observations.
        self.d_model = d_model  # Hidden (model) dimension.
        self.num_of_nodes = num_of_nodes  # Number of variables.
        self.cluster_labels = cluster_labels
        self.num_clusters = len(torch.unique(cluster_labels)) if cluster_labels is not None else 0
        self.skip_cluster_to_node = skip_cluster_to_node
        
        # Use cluster-based GAT cell with the skip_cluster_to_node parameter
        self.gated_update = ClusterGATCell(d_model, query_vector_dim, num_heads, use_adj_mask, skip_cluster_to_node)
        
        # Build edge indices for intra-cluster and inter-cluster connections
        self.edge_index_intra, self.edge_index_inter = build_cluster_adjacencies(cluster_labels)
        
        # Two MLPs to project PLM textual embeddings:
        # (a) For generating cluster-specific query vectors that will modulate the gated updates.
        self.projection_f = MLP(plm_rep_dim, 2 * d_model, query_vector_dim)
        
        # (b) For obtaining node embeddings to compute a knowledge-aware graph.
        self.projection_g = MLP(plm_rep_dim, 2 * d_model, node_emb_dim)
        
        logging.info("ClusterBasedVSDGATRNN initialized with d_in {}, d_model {}, num_of_nodes {}, "
                     "num_clusters {}, num_heads {}, use_adj_mask: {}, skip_cluster_to_node: {}"
                     .format(d_in, d_model, num_of_nodes, self.num_clusters, num_heads, use_adj_mask, skip_cluster_to_node))

    def init_hidden_states(self, x):
        """Initialize hidden states for nodes and clusters"""
        batch_size = x.shape[0]
        
        # Initialize node-level hidden states with zeros
        h0 = torch.zeros(size=(batch_size, self.num_of_nodes, self.d_model)).to(x.device)
        
        # Initialize cluster-level hidden states with zeros
        cluster_h0 = torch.zeros(size=(batch_size, self.num_clusters, self.d_model)).to(x.device)
        
        logging.debug("Initialized node hidden states with shape: {}".format(h0.shape))
        logging.debug("Initialized cluster hidden states with shape: {}".format(cluster_h0.shape))
        
        return h0, cluster_h0

    def forward(self, obs_emb, observed_mask, lengths, avg_interval, var_plm_rep_tensor, cluster_labels):
        """
        obs_emb: Observation embeddings, shape [B, T, N, d_in]
        observed_mask: Binary mask indicating observed variables, shape [B, T, N]
        lengths: Tensor indicating the valid sequence length for each sample, shape [B, 1]
        avg_interval: Average interval between observations, shape [B, T, N]
        var_plm_rep_tensor: Pre-trained language model embeddings for variables,
                            shape [B, N, plm_rep_dim] or [N, plm_rep_dim] (broadcast if necessary)
        cluster_labels: Tensor of cluster labels for each variable, shape [N] or [B, N]
        
        Returns:
            If skip_cluster_to_node is False:
                output: Node hidden states for each sample at its final time step. Shape: [B, N, d_model]
            If skip_cluster_to_node is True:
                output: Node hidden states as above
                cluster_output: Cluster hidden states for each sample at its final time step. Shape: [B, num_clusters, d_model]
        """
        batch, steps, nodes, features = obs_emb.size()
        device = obs_emb.device
        logging.debug("obs_emb shape: {}, observed_mask shape: {}".format(obs_emb.shape, observed_mask.shape))
        
        # Check for NaNs in input
        logging.debug(f"Input contains NaN: obs_emb={torch.isnan(obs_emb).any().item()}, mask={torch.isnan(observed_mask).any().item()}, plm={torch.isnan(var_plm_rep_tensor).any().item()}")
        
        # DEBUGGING: Check cluster membership distribution
        if cluster_labels.dim() == 1:
            # Count nodes in each cluster and log distribution
            unique_clusters, counts = torch.unique(cluster_labels, return_counts=True)
            cluster_counts = {c.item(): count.item() for c, count in zip(unique_clusters, counts)}
            logging.info(f"Cluster membership distribution: {cluster_counts}")
            
            # Check if any clusters are too small (< 1% of nodes)
            small_clusters = {c: count for c, count in cluster_counts.items() if count < 0.01 * nodes}
            if small_clusters:
                logging.warning(f"Potentially problematic small clusters detected: {small_clusters}")
        
        # Ensure cluster_labels has batch dimension
        if cluster_labels.dim() == 1:
            cluster_labels = cluster_labels.expand(batch, -1)  # [B, N]
        
        # 1. Initialize the hidden state for every variable and every cluster for each sample.
        h, cluster_h = self.init_hidden_states(obs_emb)  # Shapes: [B, N, d_model], [B, num_clusters, d_model]
        
        # 2. Prepare output tensors to store the hidden states for samples at final time steps.
        output = torch.zeros_like(h)
        cluster_output = torch.zeros_like(cluster_h)
        
        # 3. Mask to record which nodes have ever been observed.
        nodes_initial_mask = torch.zeros(batch, nodes).to(device)
        
        # 4. Broadcast the PLM embeddings across the batch if needed.
        var_plm_rep_tensor = repeat(var_plm_rep_tensor, "n d -> b n d", b=batch)
        logging.debug("var_plm_rep_tensor after repeat shape: {}".format(var_plm_rep_tensor.shape))
        
        # 5. Generate query vectors for each cluster from the PLM embeddings
        # First, get average PLM embeddings for each cluster
        cluster_plm_embeddings = torch.zeros(batch, self.num_clusters, var_plm_rep_tensor.shape[-1], device=device)
        for b in range(batch):
            for c in range(self.num_clusters):
                # Find all variables in this cluster
                cluster_vars = (cluster_labels[b] == c).nonzero().squeeze(-1)
                if cluster_vars.numel() > 0:
                    # Compute average PLM embedding for this cluster
                    cluster_plm_embeddings[b, c] = var_plm_rep_tensor[b, cluster_vars].mean(dim=0)
        
        # DEBUGGING: Check the quality of cluster PLM embeddings
        for c in range(self.num_clusters):
            logging.debug(f"Cluster {c} PLM embedding stats: "
                        f"norm={torch.norm(cluster_plm_embeddings[0, c]).item():.4f}, "
                        f"mean={cluster_plm_embeddings[0, c].mean().item():.4f}, "
                        f"std={cluster_plm_embeddings[0, c].std().item():.4f}")
        
        # Generate query vectors for each cluster
        cluster_query_vectors = self.projection_f(cluster_plm_embeddings)  # [B, num_clusters, query_vector_dim]
        
        # DEBUGGING: Check quality of query vectors for each cluster
        for c in range(self.num_clusters):
            logging.debug(f"Cluster {c} query vector stats: "
                        f"norm={torch.norm(cluster_query_vectors[0, c]).item():.4f}, "
                        f"mean={cluster_query_vectors[0, c].mean().item():.4f}, "
                        f"std={cluster_query_vectors[0, c].std().item():.4f}")
        
        # 6. Compute node embeddings from textual embeddings and normalize them
        node_embeddings = self.projection_g(var_plm_rep_tensor)  # Shape: [B, N, some_dim]
        normalized_node_embeddings = F.normalize(node_embeddings, p=2, dim=2)
        logging.debug(f"Normalized node embeddings shape: {normalized_node_embeddings.shape}, has_nan={torch.isnan(normalized_node_embeddings).any().item()}")
        
        # 7. Main loop: iterate through time steps (up to max observed length).
        for step in range(int(torch.max(lengths).item())):
            # Current observation (for all samples at current time) and its corresponding mask.
            cur_obs = obs_emb[:, step]       # Shape: [B, N, d_in]
            cur_mask = observed_mask[:, step]  # Shape: [B, N]
            
            logging.debug(f"Step {step}: cur_obs has_nan={torch.isnan(cur_obs).any().item()}")
            
            # Find indices of observed variables (using torch.where).
            cur_obs_var = torch.where(cur_mask)
            # Record that these nodes have been observed.
            nodes_initial_mask[cur_obs_var] = 1
            nodes_need_update = cur_obs_var  # Nodes to update at this time step.
            
            # ENHANCED DIAGNOSTICS: Track which nodes are observed at each timestep
            if step < 3:  # Only for first few timesteps to avoid log spam
                sample_idx = 0  # Focus on first sample in batch
                batch_nodes = (cur_obs_var[0] == sample_idx).nonzero().squeeze(-1)
                if batch_nodes.numel() > 0:
                    observed_node_indices = cur_obs_var[1][batch_nodes].tolist()
                    # Get cluster assignments for observed nodes
                    observed_clusters = [cluster_labels[sample_idx, idx].item() for idx in observed_node_indices[:20]]
                    logging.debug(f"Step {step}: Sample {sample_idx} observed nodes: {observed_node_indices[:20]}")
                    logging.debug(f"Step {step}: These nodes belong to clusters: {observed_clusters}")
                    
                    # Check hidden states for these nodes BEFORE update
                    for i, node_idx in enumerate(observed_node_indices[:10]):
                        node_h = h[sample_idx, node_idx]
                        has_nan = torch.isnan(node_h).any().item()
                        logging.debug(f"Step {step}: BEFORE update - Node {node_idx} (cluster {cluster_labels[sample_idx, node_idx].item()}) has_nan={has_nan}")
            
            # If there are any observed nodes at this timestep, update the node and cluster states.
            if nodes_need_update[0].shape[0] > 0:                
                # Create new tensors for this step to avoid in-place modifications
                new_h = h.clone()
                new_cluster_h = cluster_h.clone()
                
                # Process each batch separately
                for b in range(batch):
                    # Find nodes that are observed in this batch
                    batch_nodes = (cur_obs_var[0] == b).nonzero().squeeze(-1)
                    if batch_nodes.numel() > 0:
                        # Get indices of observed nodes for this batch
                        node_indices = cur_obs_var[1][batch_nodes]
                        
                        # ENHANCED DIAGNOSTICS: Check h before update for specific problem clusters
                        if step == 0:  # Only at first timestep
                            problem_clusters = [1, 3, 4, 7, 8]
                            
                            # Log which nodes in each problem cluster are being observed
                            for node_idx in node_indices[:10]:  # First 10 nodes
                                node_h = h[b, node_idx]
                                has_nan = torch.isnan(node_h).any().item()
                                cluster_id = cluster_labels[b, node_idx].item()
                                if cluster_id in problem_clusters:
                                    logging.debug(f"Step {step}: Sample {b}, Node {node_idx.item()} in problem cluster {cluster_id} BEFORE update has_nan={has_nan}")
                        
                        # Prepare input (no rarity score needed)
                        update_input = cur_obs[b, node_indices]
                        
                        # Pass to the gated update and store results in new tensors
                        updated_nodes, updated_clusters = self.gated_update(
                            update_input,
                            h[b, node_indices],  # Only pass hidden states for observed nodes
                            cluster_h[b],
                            cluster_query_vectors[b],
                            cluster_labels[b],
                            node_indices,
                            self.edge_index_intra,
                            self.num_clusters
                        )
                        
                        # Update the new tensors (non-in-place)
                        new_h[b, node_indices] = updated_nodes
                        new_cluster_h[b] = updated_clusters
                        
                        # ENHANCED DIAGNOSTICS: Check updated h for observed nodes after update
                        if step < 2 and b == 0:  # First couple timesteps, first batch
                            for i, node_idx in enumerate(node_indices[:10]):  # First 10 nodes
                                node_h_new = new_h[b, node_idx]
                                has_nan_new = torch.isnan(node_h_new).any().item()
                                logging.debug(f"Step {step}: AFTER update - Node {node_idx.item()} (cluster {cluster_labels[b, node_idx].item()}) has_nan={has_nan_new}")
                
                # Replace old tensors with new ones (non-in-place)
                h = new_h
                cluster_h = new_cluster_h
                
                # Check for NaNs after update
                if torch.isnan(h).any() or torch.isnan(cluster_h).any():
                    logging.warning(f"Step {step}: NaNs detected after node/cluster update: h={torch.isnan(h).any().item()}, cluster_h={torch.isnan(cluster_h).any().item()}")
                
                logging.debug(f"Updated hidden state for observed nodes at step {step}.")
                
                # ENHANCED DIAGNOSTICS: Check for NaNs in h after update by sampling a few nodes 
                if step == 0:  # Only after first timestep
                    # Track NaN counts by cluster (for first batch)
                    nan_counts = {c: 0 for c in range(self.num_clusters)}
                    total_counts = {c: 0 for c in range(self.num_clusters)}
                    
                    # Check a subset of nodes from each cluster 
                    sample_size = min(20, h.shape[1] // self.num_clusters)
                    for node_idx in range(min(200, h.shape[1])):  # Sample up to 200 nodes
                        if node_idx < h.shape[1]:  # Safety check
                            node_h = h[0, node_idx]
                            cluster_id = cluster_labels[0, node_idx].item() 
                            total_counts[cluster_id] += 1
                            
                            if torch.isnan(node_h).any().item():
                                nan_counts[cluster_id] += 1
                                if nan_counts[cluster_id] <= 3:  # Log only first 3 NaNs per cluster
                                    logging.debug(f"Step {step}: Node {node_idx} in cluster {cluster_id} has NaN after update")
                    
                    # Log summary statistics
                    for c in range(self.num_clusters):
                        if total_counts[c] > 0:
                            nan_percentage = 100 * nan_counts[c] / total_counts[c]
                            if nan_counts[c] > 0:
                                logging.warning(f"Step {step}: Cluster {c} has {nan_counts[c]}/{total_counts[c]} nodes with NaNs ({nan_percentage:.1f}%)")
                            else:
                                logging.debug(f"Step {step}: Cluster {c} has no NaNs in {total_counts[c]} sampled nodes")
            
            # Identify samples for which the current step is the last valid observation
            end_sample_ind = torch.where(step == (lengths.squeeze(1) - 1))
            
            # Store the hidden states for those samples
            output[end_sample_ind[0]] = h[end_sample_ind[0]]
            cluster_output[end_sample_ind[0]] = cluster_h[end_sample_ind[0]]
            
            # Optionally, if at the last step overall, return the collected outputs
            if step == int(torch.max(lengths).item()) - 1:
                logging.info(f"Returning final output from ClusterBasedVSDGATRNN at step {step}. Output has_nan={torch.isnan(output).any().item()}, cluster_output has_nan={torch.isnan(cluster_output).any().item()}")
                
                # Return appropriate output based on mode
                if self.skip_cluster_to_node:
                    # In cluster-only mode, return both node and cluster states
                    # The caller can use whichever is appropriate
                    return output, cluster_output
                else:
                    # In dual representation mode, return only the node states as before
                    return output
        
        # In case the loop exits normally (though above branch should have returned at final step).
        if self.skip_cluster_to_node:
            return output, cluster_output
        else:
            return output

class GATClusters(nn.Module):
    def __init__(self, DEVICE=None, hidden_dim=256, num_of_variables=100, num_of_timestamps=100, d_static=0, n_class=1, phe_code_size=1000, task_mode='CONTRASTIVE'):
        super(GATClusters, self).__init__()
        self.DEVICE = DEVICE if DEVICE is not None else get_device()
        self.hidden_dim = hidden_dim
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.d_static = d_static
        self.n_class = n_class
        self.phe_code_size = phe_code_size
        self.task_mode = task_mode
