# Dataset configurations
dataset_configs = {
    'P12': {
        'base_path': 'data/P12',
        'start': 0,
        'variables_num': 36,
        'd_static': 9,
        'timestamp_num': 215,
        'n_class': 2,
        'split_idx': 1
    },
    'physionet': {
        'base_path': 'data/physionet',
        'start': 4,
        'variables_num': 36,
        'd_static': 9,
        'timestamp_num': 215,
        'n_class': 2,
        'split_idx': 5
    },
    'P19': {
        'base_path': 'data/P19',
        'd_static': 6,
        'variables_num': 34,
        'timestamp_num': 60,
        'n_class': 2,
        'split_idx': 1
    },
    'mimic3': {
        'base_path': 'data/mimic3',
        'start': 0,
        'd_static': 0,
        'variables_num': 16,
        'timestamp_num': 292,
        'n_class': 2,
        'split_idx': 0
    }
}

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from utils import get_data_split, getStats, getStats_static, tensorize_normalize_with_features, tensorize_normalize_mimic3

# Lightning Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, mf=None, stdf=None, ms=None, ss=None, dataset='P12'):
        self.data = data
        self.labels = labels
        self.dataset = dataset
        self.mf = mf
        self.stdf = stdf
        self.ms = ms
        self.ss = ss
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.dataset in ['P12', 'P19', 'physionet']:
            P_tensor, P_static_tensor, P_avg_interval_tensor, P_length_tensor, P_time_tensor = \
                tensorize_normalize_with_features(self.data[idx], self.mf, self.stdf, self.ms, self.ss)
        else:
            P_tensor, P_static_tensor, P_avg_interval_tensor, P_length_tensor, P_time_tensor = \
                tensorize_normalize_mimic3(self.data[idx], self.mf, self.stdf)
        
        y_tensor = torch.tensor(self.labels[idx]).float()
        
        return P_tensor, P_static_tensor, P_avg_interval_tensor, P_length_tensor, P_time_tensor, None, y_tensor


# Lightning DataModule
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size, num_workers=4, plm='bert', source='gpt'):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = dataset_configs[dataset_name]
        self.plm = plm
        self.source = source
        
        # Get split path
        if dataset_name == 'P12':
            self.split_path = '/splits/phy12_split' + str(self.config['split_idx']) + '.npy'
        elif dataset_name == 'physionet':
            self.split_path = '/splits/phy12_split' + str(self.config['split_idx']) + '.npy'
        elif dataset_name == 'P19':
            self.split_path = '/splits/phy19_split' + str(self.config['split_idx']) + '_new.npy'
        elif dataset_name == 'mimic3':
            self.split_path = ''
            
        # Set up PLM suffix
        if source == 'gpt':
            self.suffix = '_var_rep_gpt_source.pt'
        elif source == 'name':
            self.suffix = '_var_rep_name_source.pt'
        elif source == 'wiki':
            self.suffix = '_var_rep_wiki_source.pt'
        
    def prepare_data(self):
        # Download data if needed (not applicable here)
        pass
    
    def setup(self, stage=None):
        # Load data and split into train/val/test
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(
            self.config['base_path'], self.split_path, dataset=self.dataset_name
        )
        
        # Calculate statistics for normalization
        if self.dataset_name in ['P12', 'P19', 'physionet']:
            T, F = Ptrain[0]['arr'].shape
            D = len(Ptrain[0]['extended_static'])
            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            Ptrain_static_tensor = np.zeros((len(Ptrain), D))

            for i in range(len(Ptrain)):
                Ptrain_tensor[i] = Ptrain[i]['arr']
                Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

            # Calculate mean and standard deviation for normalization
            self.mf, self.stdf = getStats(Ptrain_tensor)
            self.ms, self.ss = getStats_static(Ptrain_static_tensor, dataset=self.dataset_name)
            
        elif self.dataset_name == 'mimic3':
            T, F = self.config['timestamp_num'], self.config['variables_num']
            Ptrain_tensor = np.zeros((len(Ptrain), T, F))
            for i in range(len(Ptrain)):
                Ptrain_tensor[i][:Ptrain[i][4]] = Ptrain[i][2]

            # Calculate mean and standard deviation
            self.mf, self.stdf = getStats(Ptrain_tensor)
            self.ms, self.ss = None, None
        
        # Create datasets
        self.train_dataset = TimeSeriesDataset(Ptrain, ytrain, self.mf, self.stdf, self.ms, self.ss, self.dataset_name)
        self.val_dataset = TimeSeriesDataset(Pval, yval, self.mf, self.stdf, self.ms, self.ss, self.dataset_name)
        self.test_dataset = TimeSeriesDataset(Ptest, ytest, self.mf, self.stdf, self.ms, self.ss, self.dataset_name)
        
        # Load PLM representations
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.plm_rep_tensor = torch.load(self.config['base_path'] + f'/{self.dataset_name}_{self.plm}{self.suffix}').to(device)
        
    def train_dataloader(self):
        # Perform upsampling for class balance
        train_data = self.train_dataset.data
        train_labels = self.train_dataset.labels
        
        idx_0 = np.where(train_labels == 0)[0]
        idx_1 = np.where(train_labels == 1)[0]
        
        # Upsample minority class
        expanded_idx_1 = np.concatenate([idx_1, idx_1, idx_1], axis=0)
        
        # Combine indices
        all_indices = np.concatenate([idx_0, expanded_idx_1])
        
        # Custom sampler for balanced batches
        balanced_sampler = torch.utils.data.sampler.SubsetRandomSampler(all_indices)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=balanced_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def transfer_batch_to_device(self, batch, device):
        # Override to add PLM representation to each batch
        P, P_static, P_avg_interval, P_length, P_time, _, y = batch
        
        # Move all tensors to device
        P = P.to(device)
        if P_static is not None:
            P_static = P_static.to(device)
        P_avg_interval = P_avg_interval.to(device)
        P_length = P_length.to(device)
        P_time = P_time.to(device)
        y = y.to(device)
        
        # Add PLM representation tensor to batch
        return P, P_static, P_avg_interval, P_length, P_time, self.plm_rep_tensor, y
