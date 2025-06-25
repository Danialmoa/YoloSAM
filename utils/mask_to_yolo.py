import cv2
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Optional
from tqdm import tqdm


class MaskToYOLOConverter:
    """Convert mask images to YOLO format labels."""
    
    def __init__(self, class_id: int = 0):
        self.class_id = class_id
    
    def mask_to_bboxes(self, mask: np.ndarray, min_area: int = 100) -> List[Tuple[float, float, float, float]]:
        """
        Convert binary mask to YOLO format bounding boxes.
        
        Args:
            mask: Binary mask (0 and 255 or 0 and 1)
            min_area: Minimum area threshold for filtering small objects
            
        Returns:
            List of bounding boxes in YOLO format [x_center, y_center, width, height] (normalized 0-1)
        """
        # Ensure mask is binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        height, width = mask.shape
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small objects
            if w * h < min_area:
                continue
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            width_norm = w / width
            height_norm = h / height
            
            # Ensure coordinates are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            bboxes.append((x_center, y_center, width_norm, height_norm))
        
        return bboxes
    
    def convert_single_mask(self, mask_path: Path, output_path: Path, min_area: int = 100):
        """Convert a single mask file to YOLO label format."""
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            return
        
        # Get bounding boxes
        bboxes = self.mask_to_bboxes(mask, min_area)
        
        # Write YOLO labels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for bbox in bboxes:
                x_center, y_center, width, height = bbox
                f.write(f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def convert_dataset_inplace(
        self, 
        base_path: str, 
        min_area: int = 100,
        splits: List[str] = ['train', 'val']
    ):
        """
        Convert masks to YOLO labels in the same folder structure.
        Creates 'labels' folders alongside existing 'masks' folders.
        
        Args:
            base_path: Base path containing train/val subdirectories
            min_area: Minimum area threshold
            splits: Dataset splits to process
        """
        base_path = Path(base_path)
        
        for split in splits:
            print(f"Processing {split} split...")
            
            # Paths
            split_dir = base_path / split
            mask_dir = split_dir / 'masks'
            label_dir = split_dir / 'labels'  # Create labels in same split directory
            
            # Create labels directory
            label_dir.mkdir(parents=True, exist_ok=True)
            
            if not mask_dir.exists():
                print(f"Warning: Mask directory {mask_dir} does not exist")
                continue
            
            # Get all mask files
            mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
            
            print(f"Found {len(mask_files)} mask files in {split}")
            
            for mask_file in tqdm(mask_files, desc=f"Converting {split}"):
                # Generate label file path
                label_file = label_dir / f"{mask_file.stem}.txt"
                
                # Convert mask to YOLO label
                self.convert_single_mask(mask_file, label_file, min_area)
            
            print(f"Completed {split} split: {len(mask_files)} files processed")
            print(f"Labels saved to: {label_dir}")
        
        print(f"Dataset conversion completed!")
        print(f"Your dataset structure is now:")
        for split in splits:
            split_dir = base_path / split
            if split_dir.exists():
                print(f"  {split_dir}/")
                print(f"    images/")
                print(f"    masks/")
                print(f"    labels/  <- NEW")


def main():
    """Convert masks to YOLO labels in the same folder structure."""
    converter = MaskToYOLOConverter(class_id=0)  # 0 for 'scar' class
    
    # Convert your dataset in place
    converter.convert_dataset_inplace(
        base_path='./sample_data',
        min_area=5,  # Minimum area threshold (adjust as needed)
        splits=['train', 'val']
    )
    

if __name__ == "__main__":
    main() 