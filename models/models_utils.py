import torch.nn as nn
import torch.nn.functional as F
from utils import *
from einops import *
from einops import repeat
import logging
import math
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import get_device


def get_device_info():
    device = get_device()
    if device.type == 'mps':
        return "MPS"
    elif device.type == 'cuda':
        return "CUDA"
    else:
        return "CPU"


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
