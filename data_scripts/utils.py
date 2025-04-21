
from datetime import datetime
import torch.nn as nn



LOGGING_ENABLED = True

# Create a unique log file for each run
log_file = f'logs/raindrop_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

def log_message(*args, print_to_console=False):
    """Log a message to the log file and optionally print to console
    
    Args:
        *args: Any number of arguments to log
        print_to_console: Whether to print to console as well
    """
    # Skip logging if disabled
    if not LOGGING_ENABLED:
        return
        
    # Convert all arguments to strings and join them
    message = " ".join(str(arg) for arg in args)
    
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)





class ConvNetMLP(nn.Module):
    def __init__(self, hidden_size, final_dim, kernel_size=3):
        """
        Args:
            hidden_size (int): Dimensionality of the transformer hidden states.
            final_dim (int): Desired dimensionality of the final embedding.
            kernel_size (int): Kernel size for the Conv1D layer (default: 3).
        """
        super(ConvNetMLP, self).__init__()
        # Conv1d expects input as (batch_size, channels, sequence_length)
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # to preserve sequence length
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(hidden_size, final_dim)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_length, hidden_size)
        Returns:
            torch.Tensor: Final embeddings of shape (batch_size, final_dim)
        """
        x = hidden_states.transpose(1, 2)  # shape: (batch_size, hidden_size, seq_length)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)   # shape: (batch_size, hidden_size, 1)
        x = x.squeeze(-1)  # shape: (batch_size, hidden_size)
        return self.fc(x)  # shape: (batch_size, final_dim)
            