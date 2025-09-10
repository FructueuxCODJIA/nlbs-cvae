import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any
from skimage.filters import threshold_otsu
from scipy import ndimage

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
    """
    if mode == "train":
        transforms = A.Compose([
            A.Resize(height=resolution, width=resolution),
            A.HorizontalFlip(p=lr_flip_prob),
            A.Rotate(limit=rotation_range, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=contrast_jitter,
                contrast_limit=contrast_jitter,
                p=0.5
            ),
            A.GaussNoise(var_limit=(1.0, gaussian_noise * 255), p=0.3),  # corrigé
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    elif mode in ["val", "test"]:
        transforms = A.Compose([
            A.Resize(height=resolution, width=resolution),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unknown transform mode: {mode}")
    
    return transforms


# --- Fonctions utilitaires pour le prétraitement DICOM ---
def apply_windowing(
    image: np.ndarray,
    window_center: float,
    window_width: float,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    image = np.clip(image, window_min, window_max)
    image = (image - window_min) / (window_max - window_min)
    image = image * (max_val - min_val) + min_val
    return image

def apply_bbox_crop(
    image: np.ndarray,
    bbox: list,
    padding: int = 0
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    return image[y1:y2, x1:x2]

def get_breast_mask(image: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Return binary mask of breast region using Otsu thresholding
    """
    if threshold is None:
        threshold = threshold_otsu(image)
    mask = image > threshold
    
    # Nettoyage du masque
    mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5,5)))
    mask = ndimage.binary_fill_holes(mask)
    
    return mask.astype(np.uint8)

def apply_clahe(image: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """
    Apply CLAHE to enhance contrast
    """
    from skimage import exposure
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return exposure.equalize_adapthist(image, clip_limit=clip, nbins=256)
