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
from models.models_GCNadaptive import VSDGCRNN
from models.models_GATGRU import VSDGATRNN
from models.models_GATClusters import ClusterBasedVSDGATRNN
from models.models_utils import Value_Encoder, Time_Encoder
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
        output_dim = 1 if task_mode == '24h_mortality_discharge' else n_class
        
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


