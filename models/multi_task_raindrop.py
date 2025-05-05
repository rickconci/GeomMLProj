import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.models_rd import Raindrop_v2
from train_utils import debug_print, DEBUG, toggle_debug

class MultiTaskRaindropV2(Raindrop_v2):
    """
    Extension of Raindrop_v2 model to support multiple prediction heads for:
    1. Mortality prediction within 6 months of discharge
    2. Readmission prediction within 15 days of discharge
    3. PHE codes in the next admission
    """
    def __init__(self, DEVICE, d_inp=36, d_model=64, nhead=4, nhid=128, nlayers=2, 
                 dropout=0.3, max_len=215, d_static=9, MAX=100, perc=0.5, 
                 aggreg='mean', n_classes=2, phe_code_size=None, 
                 global_structure=None, sensor_wise_mask=False, static=True):
        """
        Initialize the multi-task Raindrop_v2 model with multiple prediction heads.
        
        Args:
            DEVICE: The device to run the model on
            phe_code_size: Size of the PHE code vocabulary for the code prediction task
            Other arguments: Same as Raindrop_v2
        """
        # Call the parent constructor with the original arguments
        super(MultiTaskRaindropV2, self).__init__(
            d_inp=d_inp, 
            d_model=d_model, 
            nhead=nhead, 
            nhid=nhid, 
            nlayers=nlayers, 
            dropout=dropout, 
            max_len=max_len, 
            d_static=d_static,
            MAX=MAX, 
            perc=perc, 
            aggreg=aggreg, 
            n_classes=n_classes, 
            global_structure=global_structure, 
            sensor_wise_mask=sensor_wise_mask, 
            static=static
        )
        
        # Store PHE code size
        self.phe_code_size = phe_code_size
        self.DEVICE = DEVICE
        
        # Define feature dimension for the task-specific heads based on the actual output dimension
        # When sensor_wise_mask=True, the output has shape [batch_size, self.d_inp*(self.d_ob+16)]
        if sensor_wise_mask:
            self.feature_dim = self.d_inp * (self.d_ob + 16)
            if static:
                self.feature_dim += d_inp
        else:
            # Original calculation
            if static:
                self.feature_dim = d_model + 16 + d_inp  # d_model + d_pe + d_inp
            else:
                self.feature_dim = d_model + 16  # d_model + d_pe
        
        # Replace single classifier with multiple task-specific classifiers
        # 1. Mortality prediction head (binary classification)
        self.mortality_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1)  # Binary classification
        ).to(DEVICE)
        
        # 2. Readmission prediction head (binary classification)
        self.readmission_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, nhid),
            nn.ReLU(),
            nn.Linear(nhid, 1)  # Binary classification
        ).to(DEVICE)
        
        # 3. PHE code prediction head (multi-label classification) - if phe_code_size is provided
        if phe_code_size is not None:
            phecode_bottleneck_dim = min(512, phe_code_size // 2)  # Create a bottleneck
            self.phecode_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, nhid),
                nn.ReLU(),
                nn.Dropout(0.3),  # Add dropout to prevent overfitting
                nn.Linear(nhid, phecode_bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(phecode_bottleneck_dim, phe_code_size)
            ).to(DEVICE)
        
        debug_print(f"[MultiTaskRaindropV2] Initialized with feature_dim: {self.feature_dim}")
        debug_print(f"[MultiTaskRaindropV2] Prediction heads: mortality, readmission, phecodes: {phe_code_size is not None}")
        logging.info(f"MultiTaskRaindropV2 initialized with feature dim: {self.feature_dim}, PHE code size: {phe_code_size}")
    
    def forward(self, src, static, times, lengths):
        """
        Forward pass for MultiTaskRaindropV2.
        
        Args:
            src: Input tensor with shape [max_len, batch_size, 2*d_inp]
                 - First half contains actual sensor readings
                 - Second half contains missing value masks (1 = missing, 0 = present)
            static: Static features with shape [batch_size, d_static]
            times: Timestamps with shape [max_len, batch_size]
            lengths: Valid sequence lengths for each sample with shape [batch_size]
            
        Returns:
            dict: Dictionary containing predictions for each task:
                 - 'orig_output': Original model output (for backward compatibility)
                 - 'mortality': Mortality prediction logits
                 - 'readmission': Readmission prediction logits
                 - 'phecodes': PHE codes prediction logits (if phe_code_size is provided)
                 - 'distance': Graph distance metric
        """

        # Check tensor devices for debugging
        debug_print(f"[MultiTaskRaindropV2] Input src device: {src.device}")
        debug_print(f"[MultiTaskRaindropV2] Static device: {None if static is None else static.device}")
        debug_print(f"[MultiTaskRaindropV2] Times device: {times.device}")
        debug_print(f"[MultiTaskRaindropV2] Self.emb weight device: {self.emb.weight.device}")
        debug_print(f"[MultiTaskRaindropV2] Self.mortality_classifier[0].weight device: {self.mortality_classifier[0].weight.device}")
        
        debug_print(f"[MultiTaskRaindropV2] Input src shape: {src.shape}, static shape: {None if static is None else static.shape}")
        debug_print(f"[MultiTaskRaindropV2] times shape: {times.shape}, lengths shape: {lengths.shape}")
        
        # Get the base model outputs
        orig_output, distance, _ = super().forward(src, static, times, lengths)
        debug_print(f"[MultiTaskRaindropV2] Base model output shape: {orig_output.shape}")
        
        # Generate predictions for each task
        mortality_logits = self.mortality_classifier(orig_output)
        debug_print(f"[MultiTaskRaindropV2] Mortality logits shape: {mortality_logits.shape}")
        
        readmission_logits = self.readmission_classifier(orig_output)
        debug_print(f"[MultiTaskRaindropV2] Readmission logits shape: {readmission_logits.shape}")
        
        # Build the result dictionary
        result = {
            'orig_output': orig_output,  # Keep original output before classification
            'mortality': mortality_logits,
            'readmission': readmission_logits,
            'distance': distance
        }
        
        # Add PHE code predictions if enabled
        if hasattr(self, 'phecode_classifier'):
            phecode_logits = self.phecode_classifier(orig_output)
            debug_print(f"[MultiTaskRaindropV2] PHE code logits shape: {phecode_logits.shape}")
            result['phecodes'] = phecode_logits
        
        return result 