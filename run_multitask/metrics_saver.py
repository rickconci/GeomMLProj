import os
import csv
import pytorch_lightning as pl
import logging

class MetricsSaverCallback(pl.Callback):
    """
    PyTorch Lightning callback that saves validation metrics to a CSV file at the end of training.
    """
    
    def __init__(self, csv_path, model_type, append=True):
        """
        Initialize the metrics saver callback.
        
        Args:
            csv_path (str): Path to the CSV file to save metrics to
            model_type (str): The model type (DS_only, TS_only, DS_TS_concat)
            append (bool): Whether to append to the CSV file if it exists
        """
        super().__init__()
        self.csv_path = csv_path
        self.model_type = model_type
        self.append = append
        self.best_metrics = {}
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Save the best metrics at the end of each validation epoch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        # Get all the logged metrics for this validation epoch
        logged_metrics = trainer.callback_metrics
        
        # Save only validation metrics that we care about
        metrics_to_track = [
            'val_epoch_mortality_auroc',
            'val_epoch_mortality_auprc',
            'val_epoch_readmission_auroc',
            'val_epoch_readmission_auprc',
            'val_epoch_current_phecode_macro_auc',
            'val_epoch_current_phecode_micro_auc',
            'val_epoch_current_phecode_micro_ap',
            'val_epoch_current_phecode_prec@5',
            'val_epoch_next_phecode_macro_auc',
            'val_epoch_next_phecode_micro_auc',
            'val_epoch_next_phecode_micro_ap',
            'val_epoch_next_phecode_prec@5'
        ]
        
        # Update best metrics
        for metric_name in metrics_to_track:
            if metric_name in logged_metrics:
                # Get numeric value from tensor
                value = float(logged_metrics[metric_name].cpu().detach().numpy())
                
                # Update best value
                if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
                    
        # Log best metrics so far
        if self.best_metrics:
            logging.info(f"Current best validation metrics for {self.model_type}:")
            for name, value in self.best_metrics.items():
                logging.info(f"  {name}: {value:.4f}")
                
    def on_fit_end(self, trainer, pl_module):
        """
        Save the final metrics to a CSV file.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: The PyTorch Lightning module
        """
        # Define the column names for the CSV file
        fieldnames = [
            'model_type',
            'mortality_auroc',
            'mortality_auprc',
            'readmission_auroc',
            'readmission_auprc',
            'current_phecode_macro_auc',
            'current_phecode_micro_auc',
            'current_phecode_micro_ap',
            'current_phecode_prec@5',
            'next_phecode_macro_auc',
            'next_phecode_micro_auc',
            'next_phecode_micro_ap',
            'next_phecode_prec@5'
        ]
        
        # Create a mapping from logged metric names to CSV column names
        metric_mapping = {
            'val_epoch_mortality_auroc': 'mortality_auroc',
            'val_epoch_mortality_auprc': 'mortality_auprc',
            'val_epoch_readmission_auroc': 'readmission_auroc',
            'val_epoch_readmission_auprc': 'readmission_auprc',
            'val_epoch_current_phecode_macro_auc': 'current_phecode_macro_auc',
            'val_epoch_current_phecode_micro_auc': 'current_phecode_micro_auc',
            'val_epoch_current_phecode_micro_ap': 'current_phecode_micro_ap',
            'val_epoch_current_phecode_prec@5': 'current_phecode_prec@5',
            'val_epoch_next_phecode_macro_auc': 'next_phecode_macro_auc',
            'val_epoch_next_phecode_micro_auc': 'next_phecode_micro_auc',
            'val_epoch_next_phecode_micro_ap': 'next_phecode_micro_ap',
            'val_epoch_next_phecode_prec@5': 'next_phecode_prec@5'
        }
        
        # Create a row with the metrics to save
        row = {'model_type': self.model_type}
        for log_name, csv_name in metric_mapping.items():
            if log_name in self.best_metrics:
                row[csv_name] = self.best_metrics[log_name]
            else:
                row[csv_name] = float('NaN')  # Use NaN for missing metrics
        
        # Check if the CSV file exists
        file_exists = os.path.isfile(self.csv_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        
        # Write to the CSV file
        with open(self.csv_path, mode='a' if self.append and file_exists else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if file is new or not appending
            if not file_exists or not self.append:
                writer.writeheader()
            
            # Write the row with metrics
            writer.writerow(row)
            
        logging.info(f"Saved validation metrics for {self.model_type} to {self.csv_path}") 