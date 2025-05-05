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
from models.models_GCNadaptive import VSDGCRNN
from models.models_GATGRU import VSDGATRNN
from models.models_GATClusters import ClusterBasedVSDGATRNN
from models.models_utils import Value_Encoder, Time_Encoder

# Get the device
device = get_device()

# Configure logging for debugging - file only, no console output
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('logs/models.log')
    ]
)
import torch
import torch.nn as nn


class DSEncoderWithWeightedSum(nn.Module):
    def __init__(self, hidden_dim, projection_dim, pooling_type='weighted_sum', num_heads=4):
        """
        hidden_dim: Dimensionality of the pre-computed chunk embeddings.
        projection_dim: Final dimension for downstream tasks.
        pooling_type: Either 'weighted_sum' or 'attention'
        num_heads: Number of attention heads (only used if pooling_type='attention')
        """
        super().__init__()
        
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim
        self.device = get_device()
        
        # Project from embedding dimension (768) to hidden dimension
        self.embedding_proj = nn.Linear(768, hidden_dim)
        
        if pooling_type == 'weighted_sum':
            # The input dimension should match the embedding dimension (768 in this case)
            self.weight_proj = nn.Linear(768, 1)  # Changed from hidden_dim to 768
        elif pooling_type == 'attention':
            # For multi-head attention
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
            
            # Query, Key, Value projections
            self.query_proj = nn.Linear(768, hidden_dim)  # Changed from hidden_dim to 768
            self.key_proj = nn.Linear(768, hidden_dim)    # Changed from hidden_dim to 768
            self.value_proj = nn.Linear(768, hidden_dim)  # Changed from hidden_dim to 768
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU()
        )
        
        # Move all layers to device
        self.to(self.device)

    def _weighted_sum_pooling(self, embs):
        """Weighted sum pooling implementation"""
        # Ensure input is on correct device
        embs = embs.to(self.device)
        scores = self.weight_proj(embs).squeeze(-1)  # [T_i]
        weights = torch.softmax(scores, dim=0)  # [T_i]
        pooled = (weights.unsqueeze(-1) * embs).sum(dim=0)  # [hidden_dim]
        # Project from embedding dimension to hidden dimension
        pooled = self.embedding_proj(pooled)  # [hidden_dim]
        return pooled

    def _attention_pooling(self, embs):
        """Multi-head attention pooling implementation"""
        # Ensure input is on correct device
        embs = embs.to(self.device)
        batch_size = embs.size(0) if len(embs.shape) > 2 else 1
        seq_len = embs.size(0) if len(embs.shape) > 2 else embs.size(0)
        
        # Project to Q, K, V
        Q = self.query_proj(embs)  # [seq_len, hidden_dim]
        K = self.key_proj(embs)    # [seq_len, hidden_dim]
        V = self.value_proj(embs)  # [seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [num_heads, seq_len, seq_len]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [num_heads, seq_len, seq_len]
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)  # [num_heads, seq_len, head_dim]
        
        # Combine heads
        attended = attended.transpose(0, 1).contiguous().view(seq_len, self.hidden_dim)  # [seq_len, hidden_dim]
        
        # Project output
        output = self.output_proj(attended)  # [seq_len, hidden_dim]
        
        # Average pooling across sequence
        pooled = output.mean(dim=0)  # [hidden_dim]
        
        return pooled

    def forward(self, discharge_embeddings, output_dim=None):
        """
        discharge_embeddings: Either:
            - A tensor of shape [B, T, hidden_dim] where B is batch size and T is number of chunks
            - A list of B tensors, each of shape [T_i, hidden_dim] where T_i is number of chunks for that sample
        Returns a tensor of shape [B, projection_dim or output_dim].
        """
        # Handle tensor input
        if torch.is_tensor(discharge_embeddings):
            batch_size = discharge_embeddings.size(0)
            outputs = []
            for i in range(batch_size):
                embs = discharge_embeddings[i]  # [T, hidden_dim]
                if self.pooling_type == 'weighted_sum':
                    pooled = self._weighted_sum_pooling(embs)
                else:
                    pooled = self._attention_pooling(embs)
                outputs.append(pooled)
            pooled = torch.stack(outputs, dim=0)  # [B, hidden_dim]
            
        # Handle list of tensors input
        else:
            batch_size = len(discharge_embeddings)
            outputs = []
            
            for embs in discharge_embeddings:
                if embs.numel() == 0:  # Empty tensor
                    dim = output_dim or self.projection[0].out_features
                    outputs.append(torch.zeros(dim, device=self.device))
                    continue
                    
                if self.pooling_type == 'weighted_sum':
                    pooled = self._weighted_sum_pooling(embs)
                else:
                    pooled = self._attention_pooling(embs)
                outputs.append(pooled)
            
            pooled = torch.stack(outputs, dim=0)  # [batch_size, hidden_dim]
        
        # Project to final dimension
        proj = self.projection(pooled)  # [batch_size, projection_dim]
        return proj




class KEDGN(nn.Module):
    def __init__(self, DEVICE, hidden_dim, num_of_variables, num_of_timestamps, d_static,
                 n_class, node_enc_layer=2, rarity_alpha=0.5, query_vector_dim=5, 
                 node_emb_dim=8, plm_rep_dim=768, 
                 use_gat=False, 
                 num_heads=2, 
                 use_adj_mask=False,
                 update_all_nodes=False, 
                 use_plm_adjacency=True, 
                 use_clusters=False, 
                 cluster_labels=None,
                 task_mode='CONTRASTIVE'):
        super(KEDGN, self).__init__()
        
        # Save key parameters
        self.num_of_variables = num_of_variables
        self.num_of_timestamps = num_of_timestamps
        self.hidden_dim = hidden_dim
        self.DEVICE = DEVICE
        self.use_clusters = use_clusters
        self.cluster_labels = cluster_labels
        self.task_mode = task_mode
        num_clusters = len(torch.unique(cluster_labels)) if cluster_labels is not None else 0
        
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
        # 3. ClusterBasedVSDGATRNN (GRU + Cluster-based GAT)
        
        if use_clusters:
            self.GCRNN = ClusterBasedVSDGATRNN(
                d_in=self.hidden_dim, 
                d_model=self.hidden_dim,
                num_of_nodes=num_of_variables,
                cluster_labels=cluster_labels,
                query_vector_dim=query_vector_dim, 
                node_emb_dim=node_emb_dim,
                plm_rep_dim=plm_rep_dim, 
                num_heads=num_heads, 
                use_adj_mask=use_adj_mask
            )
        
            logging.info("Using Cluster-Based GAT model with {} clusters, {} heads, use_adj_mask: {}"
                         .format(num_clusters, num_heads, use_adj_mask))
       
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
                use_adj_mask=use_adj_mask,
                update_all_nodes=update_all_nodes,
                use_plm_adjacency=use_plm_adjacency
            )
            logging.info("Using GRU-GAT model with {} heads, use_adj_mask: {}, update_all_nodes: {}, use_plm_adjacency: {}"
                         .format(num_heads, use_adj_mask, update_all_nodes, use_plm_adjacency))
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
        
        # For the 24h_mortality_discharge task, we need a single output for BCEWithLogitsLoss
        output_dim = 1 if task_mode == 'NEXT_24h' else n_class
        
        if d_static != 0:
            self.emb = nn.Linear(d_static, hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables+ hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ).to(DEVICE)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_of_variables, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
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
            output: Classification logits [B, n_class] or [B, 1] for binary classification
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
        if self.use_clusters and self.cluster_labels is not None:
            last_hidden_state = self.GCRNN(structure_input_encoding, observed_mask, P_length, P_avg_interval, P_var_plm_rep_tensor, self.cluster_labels)
        else:
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


