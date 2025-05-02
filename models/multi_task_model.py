import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.main_models import KEDGN

class MultiTaskKEDGN(KEDGN):
    """
    Extension of KEDGN model to support multiple prediction heads for:
    1. Mortality prediction within 6 months of discharge
    2. Readmission prediction within 15 days of discharge
    3. PHE codes in the next admission
    """
    def __init__(self, DEVICE, hidden_dim, num_of_variables, num_of_timestamps, d_static,
                 n_class, phe_code_size, node_enc_layer=2, rarity_alpha=0.5, query_vector_dim=5, 
                 node_emb_dim=8, plm_rep_dim=768, 
                 use_gat=False, 
                 num_heads=2, 
                 use_adj_mask=False,
                 update_all_nodes=False, 
                 use_plm_adjacency=True, 
                 use_clusters=False, 
                 cluster_labels=None,
                 task_mode='CONTRASTIVE'):
        """
        Initialize the multi-task KEDGN model with multiple prediction heads.
        
        Args:
            phe_code_size: Size of the PHE code vocabulary for the code prediction task
            Other arguments: Same as KEDGN
        """
        # Call the parent constructor with the original arguments
        super(MultiTaskKEDGN, self).__init__(
            DEVICE, hidden_dim, num_of_variables, num_of_timestamps, d_static,
            n_class, node_enc_layer, rarity_alpha, query_vector_dim, 
            node_emb_dim, plm_rep_dim, use_gat, num_heads, use_adj_mask,
            update_all_nodes, use_plm_adjacency, use_clusters, cluster_labels,
            task_mode
        )
        
        # Store PHE code size
        self.phe_code_size = phe_code_size
        
        # Define feature dimension based on whether static features are used
        feature_dim = num_of_variables + hidden_dim if d_static != 0 else num_of_variables
        
        # Replace single classifier with multiple task-specific classifiers
        # 1. Mortality prediction head (binary classification)
        self.mortality_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification
        ).to(DEVICE)
        
        # 2. Readmission prediction head (binary classification)
        self.readmission_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification
        ).to(DEVICE)
        
        # 3. PHE code prediction head (multi-label classification)
        # Enhanced version with bottleneck and dropout for better handling of sparse multi-label classification
        phecode_bottleneck_dim = min(512, phe_code_size // 2)  # Create a bottleneck
        self.phecode_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout to prevent overfitting
            nn.Linear(hidden_dim, phecode_bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(phecode_bottleneck_dim, phe_code_size)
        ).to(DEVICE)
        
        # Move all classifiers to the correct device
        self.mortality_classifier.to(DEVICE)
        self.readmission_classifier.to(DEVICE)
        
        logging.info(f"MultiTaskKEDGN initialized with PHE code size: {phe_code_size}")
    
    def forward(self, P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor):
        """
        Forward pass for MultiTaskKEDGN.
        
        Args:
            Same as KEDGN
            
        Returns:
            dict: Dictionary containing predictions for each task:
                 - 'orig_output': Original model output (for backward compatibility)
                 - 'mortality': Mortality prediction logits
                 - 'readmission': Readmission prediction logits
                 - 'phecodes': PHE codes prediction logits
                 - 'aggregated_hidden': Aggregated hidden state
                 - 'fused_features': Features for contrastive learning
        """
        # Get the base model outputs
        orig_output, aggregated_hidden, fused_features = super().forward(
            P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor
        )
        
        # Generate predictions for each task using the fused features
        mortality_logits = self.mortality_classifier(fused_features)
        readmission_logits = self.readmission_classifier(fused_features)
        phecode_logits = self.phecode_classifier(fused_features)
        
        # Return all predictions in a dictionary
        return {
            'orig_output': orig_output,  # Keep original output for compatibility
            'mortality': mortality_logits,
            'readmission': readmission_logits,
            'phecodes': phecode_logits,
            'aggregated_hidden': aggregated_hidden,
            'fused_features': fused_features
        } 