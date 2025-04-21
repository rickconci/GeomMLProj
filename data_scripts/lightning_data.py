import torch
import pytorch_lightning as pl
from data import MIMICContrastivePairsDataset, MIMICDemographicsLoader, MIMICClinicalEventsProcessor, MIMICDischargeNotesProcessor
from torch.utils.data import DataLoader


class MIMICDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_path='/Users/riccardoconci/Local_documents/!!MIMIC',
                 temp_dfs_path='/Users/riccardoconci/Library/Mobile Documents/com~apple~CloudDocs/HQ_2024/Projects/2024_Harvard_AIM/Courses/Lent_2025/Geometric_ML/GeomMLProject/ContrastiveRain/temp_dfs',
                 batch_size=32, 
                 num_workers=4,
                 time_steps=8,
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15,
                 random_state=42,
                 max_text_length=512,
                 collate_fn=None):
        super().__init__()
        self.data_path = data_path
        self.temp_dfs_path = temp_dfs_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.time_steps = time_steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.max_text_length = max_text_length
        
        # Use provided collate function or default if not provided.
        self.collate_fn = collate_fn if collate_fn is not None else contrastive_collate_fn
        
        # These will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.demo_loader = None
        self.event_processor = None
        
    def setup(self, stage=None):
        """
        Setup the data module.
        Load demographics, split data, and process events.
        """
        print("Loading demographics data...")
        self.demo_loader = MIMICDemographicsLoader(self.data_path, self.temp_dfs_path)
        self.demo_loader.load_demographics()
        
        print("Splitting data by subject_id...")
        self.demo_loader.split_by_subject_id(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            random_state=self.random_state
        )
        
        hadm_ids = self.demo_loader.get_hadm_ids()
        print(f"Found {len(hadm_ids)} hospital admissions with discharge notes")
        
        print("Loading event data...")
        self.event_processor = MIMICClinicalEventsProcessor(
            base_path=self.data_path,
            hadm_ids=hadm_ids,
            cache_dir=self.temp_dfs_path
        )
        self.event_processor.load_all_events()
        self.event_processor.discharge_df = self.demo_loader.discharge_df
        self.event_processor.merged_with_disch_df = self.demo_loader.merged_with_disch_df
        self.event_processor.process_events()
        self.cluster_labels = self.event_processor.cluster_labels
        
        self.disch_notes_processor = MIMICDischargeNotesProcessor(
            hadm_ids=hadm_ids,
            cache_dir=self.temp_dfs_path
        )
        
        # Create datasets for each split.
        self.train_dataset = MIMICContrastivePairsDataset(
            event_processor=self.event_processor,
            disch_notes_processor=self.disch_notes_processor,
            splits_loader=self.demo_loader,
            split='train',
            T=self.time_steps,
            cache_dir=self.temp_dfs_path
        )
        
        self.val_dataset = MIMICContrastivePairsDataset(
            event_processor=self.event_processor,
            disch_notes_processor=self.disch_notes_processor,
            splits_loader=self.demo_loader,
            split='val',
            T=self.time_steps,
            cache_dir=self.temp_dfs_path
        )
        
        self.test_dataset = MIMICContrastivePairsDataset(
            event_processor=self.event_processor,
            disch_notes_processor=self.disch_notes_processor,
            splits_loader=self.demo_loader,
            split='test',
            T=self.time_steps,
            cache_dir=self.temp_dfs_path
        )
        
        print(f"Created datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val, {len(self.test_dataset)} test")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )


def contrastive_collate_fn(batch):
    """
    Custom collate function for MIMIC datasets for contrastive learning.
    This function stacks physiologic time series into a tensor, while
    leaving dataframe-level items in lists. It also collates the discharge_chunks.
    Args: batch: List of samples (each a dict).
    Returns: A dictionary with collated data.
    """
    hadm_ids = [d['hadm_id'] for d in batch]
    physio_tensor = torch.stack([d['physio_tensor'] for d in batch])
    baseline_dfs = [d['baseline_df'] for d in batch]
    treatments_dfs = [d['treatments_df'] for d in batch]
    discharge_chunks = [d['discharge_chunks'] for d in batch] 
    
    return {
        'hadm_ids': hadm_ids,
        'physio_tensor': physio_tensor,
        'baseline_dfs': baseline_dfs,
        'treatments_dfs': treatments_dfs,
        'discharge_chunks': discharge_chunks
    }
