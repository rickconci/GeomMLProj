import dataclasses
from typing import Optional


@dataclasses.dataclass
class Config:
    # common
    mode: str                       # ds_supervised | ts_supervised | multimodal_contrastive
    data_path: str = "./MIMIC"
    temp_dfs_path: str = "temp_dfs_lite"
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 10
    lr: float = 1e-3
    hidden_dim: int = 128
    projection_dim: int = 256       # for DS encoder & contrastive head
    temperature: float = 0.1        # CL temperature
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "GeomMLProj"
    wandb_entity: Optional[str] = None
    resume: Optional[str] = None
    out_dir: str = "./runs"
    
    # model selection
    model_type: str = "kedgn"       # kedgn | raindrop_v2
    
    # raindrop specific
    d_model: int = 64               # Raindrop model dimension
    nlayers: int = 2                # Number of transformer layers for Raindrop
    num_heads: int = 2              # Number of attention heads (used by both models)
    global_structure_path: Optional[str] = None  # Path to adjacency matrix for sensor relationships
    sensor_wise_mask: bool = False  # Use sensor-wise masking for Raindrop_v2
    
    # task specific
    task_mode: str = "NEXT_24h"     # CONTRASTIVE | NEXT_24h
    
    # derived
    @property
    def self_supervised(self) -> bool:
        return self.mode == "multimodal_contrastive"
