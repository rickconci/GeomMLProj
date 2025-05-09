
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_first_sample(sample_batch, use_mask=False, plot_static=False):
    """
    Plots the time series data for the first sample in the batch.
    
    Parameters:
    -----------
    sample_batch : dict
        Dictionary containing keys:
            - 'values': numpy array of shape (B, T, V), where B is batch size,
                        T is time steps, V is number of sensors/features.
            - 'mask': numpy array of shape (B, T, V) indicating valid data (1) or missing (0).
            - 'static': numpy array of shape (B, d_static) if available.
            - 'times': numpy array of shape (B, T) providing timestamps for each measurement.
            - 'length': numpy array or list of length B indicating the valid number of time steps for each sample.
            - 'label': numpy array or list of labels for each sample.
    use_mask : bool, optional
        Whether to additionally mark missing data points using the mask (default is False).
    plot_static : bool, optional
        Whether to print the static features of the sample on the plot (default is False).
    """
    
    # Extract the first sample from the batch
    values = sample_batch['values']  # shape: B x T x V
    times = sample_batch['times']    # shape: B x T
    mask = sample_batch.get('mask', None)
    static = sample_batch.get('static', None)
    length = sample_batch['length']  # assumes this is an iterable with one integer per sample
    label = sample_batch.get('label', None)
    
    print('values:', values[3])
    print('times:', times[3])
    print('static:', static[3])
    print('length:', length[3])
    print('label:', label[3])
    
    # Choose the first sample (index 0)
    sample_values = values[3]    # shape: T x V
    sample_times = times[3]      # shape: T
    sample_length = length[3]    # valid time steps
    
    # Optionally restrict the data to the valid length
    sample_times = sample_times[:sample_length]
    sample_values = sample_values[:sample_length, :]  # shape: (sample_length, V)
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    num_sensors = sample_values.shape[1]
    
    # Loop over each sensor/feature
    for i in range(num_sensors):
        y = sample_values[:, i]
        plt.plot(sample_times, y, label=f"Sensor {i}")
        # If use_mask is True and mask data exists, mark missing values (i.e., where mask == 0)
        if use_mask and (mask is not None):
            sample_mask = mask[0][:sample_length, i]
            # Plot missing data as red circles (only plotting where mask is 0)
            missing_indices = np.where(sample_mask == 0)[0]
            if missing_indices.size > 0:
                plt.plot(sample_times[missing_indices], y[missing_indices],
                         'ro', markersize=5, label=f"Missing (Sensor {i})")
    
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    # Optionally include the label or static information in the title
    plot_title = "First Sample Time Series"
    if label is not None:
        plot_title += f" - Label: {label[0]}"
    plt.title(plot_title)
    
    # Optionally print static features
    if plot_static and (static is not None):
        sample_static = static[0]
        # Convert static features to a string and place it in a text box inside the plot.
        static_str = "Static: " + ", ".join([f"{val:.2f}" for val in sample_static])
        plt.text(0.02, 0.95, static_str, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_mask_heatmap(sample_batch):
    """
    Plots a heatmap of the mask matrix for the first sample in the batch.
    
    Parameters:
    -----------
    sample_batch : dict
        Dictionary containing key 'mask' of shape (B, T, V) and key 'length' to limit the time steps.
    """
    mask = sample_batch.get('mask', None)
    if mask is None:
        print("No mask available in the sample batch.")
        return
    
    # Extract the mask for the first sample, and limit it to its valid length.
    sample_mask = mask[3]  # shape: T x V
    length = sample_batch['length'][3]
    sample_mask = sample_mask[:length, :]
    
    # Convert from tensor to numpy array if necessary
    if isinstance(sample_mask, torch.Tensor):
        sample_mask = sample_mask.cpu().detach().numpy()
    
    # Cast to integers for proper formatting
    sample_mask_int = sample_mask.astype(int)
    
    # Create a heatmap of the mask matrix using integer formatting
    plt.figure(figsize=(10, 6))
    sns.heatmap(sample_mask_int, cmap="viridis", cbar=True, annot=True, fmt="d")
    plt.xlabel("Sensors")
    plt.ylabel("Time")
    plt.title("Heatmap of Mask Matrix for First Sample")
    plt.tight_layout()
    plt.show()
