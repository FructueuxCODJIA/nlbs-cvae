import numpy as np
from PIL import Image
from typing import List, Tuple
import cv2
from scipy import ndimage


def create_patches(
    image: Image.Image,
    patch_size: int = 512,
    stride: int = 256,
    min_foreground_frac: float = 0.2
) -> List[Image.Image]:
    """
    Create patches from mammography image with foreground filtering
    
    Args:
        image: Input PIL image
        patch_size: Size of patches (square)
        stride: Stride between patches
        min_foreground_frac: Minimum foreground fraction to keep patch
    
    Returns:
        List of valid patches
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Get image dimensions
    h, w = img_array.shape
    
    patches = []
    
    # Extract patches with stride
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = img_array[y:y + patch_size, x:x + patch_size]
            
            # Check foreground fraction
            if calculate_foreground_fraction(patch) >= min_foreground_frac:
                # Convert back to PIL Image
                patch_img = Image.fromarray(patch)
                patches.append(patch_img)
    
    return patches


def calculate_foreground_fraction(
    image: np.ndarray,
    threshold: float = 0.1,
    min_area: int = 100
) -> float:
    """
    Calculate fraction of image that is foreground (breast tissue)
    
    Args:
        image: Image array
        threshold: Threshold for foreground detection
        min_area: Minimum area for connected components
    
    Returns:
        Foreground fraction [0, 1]
    """
    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0
    
    # Create binary mask
    mask = image > threshold
    
    # Remove small connected components
    labeled, num_features = ndimage.label(mask)
    
    # Keep only components above minimum area
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled == i)
        if component_size < min_area:
            mask[labeled == i] = False
    
    # Calculate foreground fraction
    foreground_pixels = np.sum(mask)
    total_pixels = mask.size
    
    return foreground_pixels / total_pixels


def extract_breast_region(
    image: np.ndarray,
    threshold: float = 0.1,
    min_area: int = 1000
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract breast region from mammography image
    
    Args:
        image: Input image array
        threshold: Threshold for breast detection
        min_area: Minimum breast area
    
    Returns:
        breast_mask: Binary mask of breast region
        bbox: Bounding box (x1, y1, x2, y2)
    """
    # Create initial mask
    mask = image > threshold
    
    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    labeled, num_features = ndimage.label(mask)
    
    # Find largest component (should be the breast)
    max_area = 0
    best_component = 0
    
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled == i)
        if component_size > max_area and component_size > min_area:
            max_area = component_size
            best_component = i
    
    if best_component == 0:
        # No valid breast region found
        return np.zeros_like(mask), (0, 0, image.shape[1], image.shape[0])
    
    # Create breast mask
    breast_mask = (labeled == best_component).astype(np.uint8)
    
    # Find bounding box
    coords = np.where(breast_mask)
    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()
    
    bbox = (x1, y1, x2, y2)
    
    return breast_mask, bbox


def apply_breast_aware_cropping(
    image: Image.Image,
    target_size: int = 512,
    padding: int = 20
) -> Image.Image:
    """
    Crop image to focus on breast region while maintaining aspect ratio
    
    Args:
        image: Input PIL image
        target_size: Target size for output
        padding: Padding around breast region
    
    Returns:
        Cropped and resized image
    """
    # Convert to numpy
    img_array = np.array(image)
    
    # Extract breast region
    breast_mask, bbox = extract_breast_region(img_array)
    
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_array.shape[1], x2 + padding)
    y2 = min(img_array.shape[0], y2 + padding)
    
    # Crop to breast region
    cropped = img_array[y1:y2, x1:x2]
    
    # Resize to target size
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Convert back to PIL
    return Image.fromarray(resized)


def create_balanced_patches(
    image: Image.Image,
    patch_size: int = 512,
    stride: int = 256,
    min_foreground_frac: float = 0.2,
    max_patches: int = 8
) -> List[Image.Image]:
    """
    Create balanced patches with different foreground fractions
    
    Args:
        image: Input PIL image
        patch_size: Size of patches
        stride: Stride between patches
        min_foreground_frac: Minimum foreground fraction
        max_patches: Maximum number of patches to return
    
    Returns:
        List of patches
    """
    # Get all valid patches
    all_patches = create_patches(
        image, patch_size, stride, min_foreground_frac
    )
    
    if len(all_patches) <= max_patches:
        return all_patches
    
    # Calculate foreground fractions for all patches
    patch_fractions = []
    for patch in all_patches:
        frac = calculate_foreground_fraction(np.array(patch))
        patch_fractions.append((patch, frac))
    
    # Sort by foreground fraction
    patch_fractions.sort(key=lambda x: x[1])
    
    # Select balanced set of patches
    selected_patches = []
    
    # Add patches with low, medium, and high foreground fractions
    n_low = max_patches // 3
    n_medium = max_patches // 3
    n_high = max_patches - n_low - n_medium
    
    # Low foreground fraction patches
    selected_patches.extend([p[0] for p in patch_fractions[:n_low]])
    
    # Medium foreground fraction patches
    mid_start = len(patch_fractions) // 2 - n_medium // 2
    selected_patches.extend([p[0] for p in patch_fractions[mid_start:mid_start + n_medium]])
    
    # High foreground fraction patches
    selected_patches.extend([p[0] for p in patch_fractions[-n_high:]])
    
    return selected_patches


def validate_patch_quality(
    patch: np.ndarray,
    min_contrast: float = 0.05,
    min_std: float = 0.02
) -> bool:
    """
    Validate patch quality based on contrast and standard deviation
    
    Args:
        patch: Patch array
        min_contrast: Minimum contrast (max - min)
        min_std: Minimum standard deviation
    
    Returns:
        True if patch meets quality criteria
    """
    # Normalize to [0, 1]
    if patch.max() > 1.0:
        patch = patch.astype(np.float32) / 255.0
    
    # Calculate contrast
    contrast = patch.max() - patch.min()
    
    # Calculate standard deviation
    std = patch.std()
    
    # Check quality criteria
    if contrast < min_contrast or std < min_std:
        return False
    
    return True
