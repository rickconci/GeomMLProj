import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.main_models import DSEncoderWithWeightedSum
from train_utils import get_device

DEVICE = get_device()

class DSOnlyMultiTaskModel(nn.Module):
    """
    Model that uses only the discharge summary embeddings (DS embeddings) as input
    for multi-task predictions:
    1. Mortality prediction within 6 months of discharge
    2. Readmission prediction within 15 days of discharge
    3. PHE codes in the next admission
    """
    def __init__(self, DEVICE, hidden_dim, projection_dim, phe_code_size, pooling_type='weighted_sum', num_heads=4):
        """
        Initialize the DS-only multi-task model.
        
        Args:
            DEVICE: Device to run model on (cuda, mps, or cpu)
            hidden_dim: Hidden dimension for the GRU
            projection_dim: Projection dimension for the DS encoder
            phe_code_size: Size of the PHE code vocabulary for the code prediction task
            pooling_type: Type of pooling to use for the DS encoder
            num_heads: Number of attention heads for the DS encoder
        """
        super(DSOnlyMultiTaskModel, self).__init__()
        
        # Store parameters
        self.DEVICE = DEVICE
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.phe_code_size = phe_code_size
        
        # DS Encoder with RNN for processing discharge summaries
        self.ds_encoder = DSEncoderWithWeightedSum(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            pooling_type=pooling_type,
            num_heads=num_heads
        ).to(DEVICE)
        
        # Task-specific prediction heads
        
        # 1. Mortality prediction head (binary classification)
        self.mortality_classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification
        ).to(DEVICE)
        
        # 2. Readmission prediction head (binary classification)
        self.readmission_classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Binary classification
        ).to(DEVICE)
        
        # 3. PHE code prediction head (multi-label classification)
        # Enhanced version with bottleneck, dropout and attention for sparse multi-label classification
        phecode_bottleneck_dim = min(512, phe_code_size // 2)  # Create a bottleneck
        
        # Attention-based PHE code classifier
        self.phecode_attention = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(DEVICE)
        
        self.phecode_classifier = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout to prevent overfitting
            nn.Linear(hidden_dim, phecode_bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(phecode_bottleneck_dim, phe_code_size)
        ).to(DEVICE)
        
        logging.info(f"DSOnlyMultiTaskModel initialized with PHE code size: {phe_code_size}")
    
    def forward(self, ds_embeddings):
        """
        Forward pass for DSOnlyMultiTaskModel.
        
        Args:
            ds_embeddings: List of discharge summary chunk embeddings 
                          (each element is a list of text chunks)
            
        Returns:
            dict: Dictionary containing predictions for each task:
                 - 'mortality': Mortality prediction logits
                 - 'readmission': Readmission prediction logits
                 - 'phecodes': PHE codes prediction logits
                 - 'ds_features': Features extracted from discharge summaries
        """
        # Process discharge summaries through DS encoder
        ds_features = self.ds_encoder(ds_embeddings)
        
        # Generate predictions for each task
        mortality_logits = self.mortality_classifier(ds_features)
        readmission_logits = self.readmission_classifier(ds_features)
        
        # Enhanced PHE code prediction with potential attention
        phecode_logits = self.phecode_classifier(ds_features)
        
        # Return all predictions in a dictionary
        return {
            'mortality': mortality_logits,
            'readmission': readmission_logits,
            'phecodes': phecode_logits,
            'ds_features': ds_features
        } 