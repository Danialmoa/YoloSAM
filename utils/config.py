from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


@dataclass
class SAMFinetuneConfig:
    device: str = "cuda"
    num_workers: int = 1
    sam_path: str = "pretrained/sam_vit_h_4b8939.pth"
    checkpoint_path: Optional[str] = None
    model_type: str = "vit_b"
    image_size: int = 1024
    
    # training
    batch_size: int = 2
    num_epochs: int = 100
    
    # loss Dice + BCE + KL divergence -> Dice = 1 - BCE - KL
    lambda_bce: float = 0.2 # for BCE loss (0.2)
    lambda_kl: float = 0.2 # for KL divergence (0.2)
    sigma: float = 1.0 # for soft label (KL divergence)
    
    # optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # wandb
    wandb_project: str = "SAM-finetune"
    wandb_name: str = "test"
    wandb_mode: str = "disabled"
    
    
    
@dataclass
class SAMDatasetConfig:
    dataset_path: str = "data/dataset"
    image_size: int = 1024
    
    # Agumentation
    percentiles: Tuple[float, float] = (0.1, 99.9)
    # rotation
    rotate_limit: float = 15
    rotate_prob: float = 0.5
    
    # scale
    scale_limit: float = 0.1
    scale_prob: float = 0.5
    
    # horizontal flip
    horizontal_flip_prob: float = 0.5
    
    # gamma
    gamma_prob: float = 0.7
    gamma_limit: Tuple[float, float] = (80, 120)
    
    # train or val
    train: bool = True
    
    # remove non-scar
    remove_nonscar: bool = True
    
    # prompt
    yolo_prompt: bool = False # if True, use yolo boxes as prompt
    
    # box prompt
    box_prompt: bool = True
    enable_direction_aug: bool = True
    enable_size_aug: bool = True

    # point prompt
    point_prompt: bool = True
    num_points: int = 3
    point_prompt_types: List[str] = field(default_factory=lambda: ['positive'])
    
    # sample size (For testing)
    sample_size: int = 100
