from dataclasses import dataclass
from typing import List, Optional, Tuple
import os


@dataclass
class SAMFinetuneConfig:
    device: str = "cuda"
    sam_path: str = "pretrained/sam_vit_h_4b8939.pth"
    checkpoint_path: Optional[str] = None
    model_type: str = "vit_b"
    image_size: int = 1024
    num_points: int = 3
    num_boxes: int = 3
    num_masks: int = 1024
    num_queries: int = 100
    
    
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
    gamma_limit: Tuple[float, float] = (0.8, 1.2)
    
    # brightness
    brightness_prob: float = 0.7
    contrast_prob: float = 0.7
    
    # train or val
    train: bool = True
    
    # remove non-scar
    remove_nonscar: bool = True
    
    # prompt
    point_prompt: bool = True
    box_prompt: bool = True
    
    # number of points
    num_points: int = 3
