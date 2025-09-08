import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any


def get_transforms(
    mode: str = "train",
    resolution: int = 512,
    lr_flip_prob: float = 0.5,
    rotation_range: int = 10,
    contrast_jitter: float = 0.1,
    gaussian_noise: float = 0.01
) -> A.Compose:
    """
    Get data transformations for mammography images
    
    Args:
        mode: "train", "val", or "test"
        resolution: Target image resolution
        lr_flip_prob: Probability of left-right flip
        rotation_range: Maximum rotation in degrees
        contrast_jitter: Contrast jittering factor
        gaussian_noise: Standard deviation of Gaussian noise
    
    Returns:
        Albumentations transform pipeline
    """
    
    if mode == "train":
        # Training transforms with augmentation
        transforms = A.Compose([
            # Resize and crop
            A.Resize(height=resolution, width=resolution),
            
            # Augmentations
            A.HorizontalFlip(p=lr_flip_prob),
            A.Rotate(limit=rotation_range, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=contrast_jitter,
                contrast_limit=contrast_jitter,
                p=0.5
            ),
            A.GaussNoise(var_limit=gaussian_noise, p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        
    elif mode == "val":
        # Validation transforms (minimal)
        transforms = A.Compose([
            A.Resize(height=resolution, width=resolution),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        
    elif mode == "test":
        # Test transforms (no augmentation)
        transforms = A.Compose([
            A.Resize(height=resolution, width=resolution),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
        
    else:
        raise ValueError(f"Unknown transform mode: {mode}")
    
    return transforms


def get_conditional_transforms(
    mode: str = "train",
    resolution: int = 512,
    conditions: Dict[str, Any] = None
) -> A.Compose:
    """
    Get conditional transforms that respect certain conditions
    
    Args:
        mode: "train", "val", or "test"
        resolution: Target image resolution
        conditions: Dictionary with condition keys (view, laterality, etc.)
    
    Returns:
        Albumentations transform pipeline
    """
    
    base_transforms = [
        A.Resize(height=resolution, width=resolution),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ]
    
    if mode == "train" and conditions is not None:
        # Conditional augmentations
        conditional_transforms = []
        
        # Left-right flip only if laterality is updated accordingly
        if 'laterality' in conditions:
            # For training, we can flip but need to update laterality
            # This is handled in the dataset class
            conditional_transforms.append(
                A.HorizontalFlip(p=0.5)
            )
        
        # Rotation (small angles to preserve orientation)
        conditional_transforms.append(
            A.Rotate(limit=10, p=0.5)
        )
        
        # Contrast and brightness (preserve diagnostic information)
        conditional_transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            )
        )
        
        # Gaussian noise (small amount)
        conditional_transforms.append(
            A.GaussNoise(var_limit=0.01, p=0.3)
        )
        
        # Combine conditional and base transforms
        transforms = A.Compose(conditional_transforms + base_transforms)
        
    else:
        # No conditional augmentations
        transforms = A.Compose(base_transforms)
    
    return transforms


def apply_windowing(
    image: np.ndarray,
    window_center: float,
    window_width: float,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Apply windowing to mammography image
    
    Args:
        image: Input image array
        window_center: Window center value
        window_width: Window width value
        min_val: Minimum output value
        max_val: Maximum output value
    
    Returns:
        Windowed image
    """
    # Calculate window bounds
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    
    # Clip values to window
    image = np.clip(image, window_min, window_max)
    
    # Normalize to [min_val, max_val]
    image = (image - window_min) / (window_max - window_min)
    image = image * (max_val - min_val) + min_val
    
    return image


def apply_bbox_crop(
    image: np.ndarray,
    bbox: list,
    padding: int = 0
) -> np.ndarray:
    """
    Apply bounding box crop with optional padding
    
    Args:
        image: Input image array
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding around bbox
    
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    # Crop image
    cropped = image[y1:y2, x1:x2]
    
    return cropped


def get_breast_mask(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Get breast mask from mammography image
    
    Args:
        image: Input image array
        threshold: Threshold for foreground detection
    
    Returns:
        Binary mask
    """
    # Simple thresholding (can be improved with more sophisticated methods)
    mask = image > threshold
    
    # Remove small connected components
    from scipy import ndimage
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    
    return mask.astype(np.uint8)
