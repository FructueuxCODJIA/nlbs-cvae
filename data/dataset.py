import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import pydicom
import pydicom.pixel_data_handlers
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .transforms import get_transforms
from .utils import create_patches, calculate_foreground_fraction


class MammographyDataset(Dataset):
    """
    Dataset for mammography images with conditioning
    
    Expected CSV columns:
    - out_path: Path to image file
    - view: View position (0=CC, 1=MLO)
    - laterality: Laterality (0=L, 1=R)
    - age_bin: Age bin (0-3)
    - cancer: Cancer presence (0/1)
    - false_positive: False positive (0/1)
    - bbox: Bounding box coordinates
    - windowing: Windowing parameters
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        resolution: int = 512,
        patch_stride: int = 256,
        min_foreground_frac: float = 0.2,
        transform_mode: str = "train",
        use_patches: bool = True,
        max_patches_per_image: int = 2
    ):
        """
        Args:
            csv_path: Path to CSV file with metadata
            image_dir: Directory containing images
            resolution: Target image resolution
            patch_stride: Stride for patch extraction
            min_foreground_frac: Minimum foreground fraction for patches
            transform_mode: "train", "val", or "test"
            use_patches: Whether to extract patches or use full images
            max_patches_per_image: Maximum number of patches per image
        """
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.patch_stride = patch_stride
        self.min_foreground_frac = min_foreground_frac
        self.transform_mode = transform_mode
        self.use_patches = use_patches
        self.max_patches_per_image = max_patches_per_image
        
        # Load and standardize metadata
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self._standardize_metadata(self.metadata)
        
        # Filter valid images
        self.metadata = self._filter_valid_images()
        
        # Create patches if needed
        if self.use_patches:
            self.patches = self._create_patches()
        else:
            self.patches = None
        
        # Get transforms
        self.transforms = get_transforms(
            mode=transform_mode,
            resolution=resolution
        )
        
        print(f"Dataset initialized with {len(self)} samples")
        if self.use_patches:
            print(f"Using patches with stride {patch_stride}")
        print(f"Transform mode: {transform_mode}")
    
    def _load_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load image from path (supports DICOM and common image formats)."""
        try:
            suffix = image_path.suffix.lower()
            if suffix == ".dcm":
                ds = pydicom.dcmread(str(image_path), force=True)
                pixel_array = ds.pixel_array.astype(np.float32)
                # Normalize to [0, 255]
                if pixel_array.max() > 0:
                    pixel_array = pixel_array - pixel_array.min()
                    pixel_array = pixel_array / (pixel_array.max() + 1e-8)
                pixel_array = (pixel_array * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(pixel_array)
                if img.mode != 'L':
                    img = img.convert('L')
                return img
            else:
                with Image.open(image_path) as img:
                    img = img.convert('L') if img.mode != 'L' else img.copy()
                    return img
        except Exception:
            return None

    def _standardize_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map NLBS metadata columns to internal expected schema.
        Also supports single-column NLBS format lines like:
        "abnormal\\...\\IM-XXXX.dcmLCC6600"
        where suffix encodes laterality(L/R), view(CC/MLO), age(2 digits), cancer(0/1), false_positive(0/1).
        """
        df = df.copy()

        # Case 1: Single-column concatenated format detected
        if len(df.columns) == 1:
            col = df.columns[0]
            values = [str(v) for v in df[col].tolist() if isinstance(v, str) and v.strip()]
            # Drop accidental header duplicate rows
            values = [v for v in values if v.strip() != col.strip()]

            records = []
            for v in values:
                s = v.strip()
                pos = s.lower().rfind('.dcm')
                if pos == -1:
                    continue
                path_part = s[: pos + 4]
                suffix = s[pos + 4 :].strip()
                if len(suffix) < 5:
                    # Not enough info
                    continue
                try:
                    fp = int(suffix[-1])
                    cancer = int(suffix[-2])
                    age_val = int(suffix[-4:-2])
                    lv = suffix[:-4].upper()  # e.g., LCC, LMLO, RCC, RMLO
                    laterality_char = lv[0] if lv else 'L'
                    view_str = lv[1:] if len(lv) > 1 else 'CC'
                    laterality = 0 if laterality_char == 'L' else 1
                    view = 0 if view_str == 'CC' else 1
                    # Age binning
                    if age_val <= 45:
                        age_bin = 0
                    elif age_val <= 55:
                        age_bin = 1
                    elif age_val <= 65:
                        age_bin = 2
                    else:
                        age_bin = 3
                    # Normalize slashes to forward
                    out_path = path_part.replace('\\', '/').lstrip('/')
                    records.append(
                        {
                            'out_path': out_path,
                            'view': view,
                            'laterality': laterality,
                            'age_bin': age_bin,
                            'cancer': cancer,
                            'false_positive': fp,
                        }
                    )
                except Exception:
                    continue

            if not records:
                raise ValueError("Failed to parse single-column NLBS metadata format")
            return pd.DataFrame.from_records(records)

        # Case 2: Already standardized
        required_cols = {"out_path", "view", "laterality", "age_bin", "cancer", "false_positive"}
        if required_cols.issubset(set(df.columns)):
            return df[list(required_cols)]

        # Case 3: Map known NLBS column names
        col_map = {
            'File Path': 'out_path',
            'Image Laterality': 'laterality_text',
            'View Position': 'view_text',
            'Age': 'age',
            'Cancer': 'cancer',
            'False Positive': 'false_positive',
        }
        for src, dst in col_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        # Clean out_path: normalize backslashes and strip leading slashes
        if 'out_path' in df.columns:
            df['out_path'] = df['out_path'].astype(str).apply(lambda x: x.replace('\\', '/').lstrip('/'))

        # Laterality: L->0, R->1
        if 'laterality' not in df.columns and 'laterality_text' in df.columns:
            df['laterality'] = df['laterality_text'].map(lambda x: 0 if str(x).strip().upper().startswith('L') else 1)

        # View: CC->0, MLO->1
        if 'view' not in df.columns and 'view_text' in df.columns:
            df['view'] = df['view_text'].map(lambda x: 0 if str(x).strip().upper() == 'CC' else 1)

        # Age binning
        if 'age_bin' not in df.columns and 'age' in df.columns:
            def _age_to_bin(a):
                try:
                    a = int(a)
                except Exception:
                    return 1
                if a <= 45:
                    return 0
                if a <= 55:
                    return 1
                if a <= 65:
                    return 2
                return 3
            df['age_bin'] = df['age'].apply(_age_to_bin)

        # Ensure ints
        for col in ['view', 'laterality', 'age_bin', 'cancer', 'false_positive']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Keep only required
        keep_cols = ['out_path', 'view', 'laterality', 'age_bin', 'cancer', 'false_positive']
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns after standardization: {missing}")
        return df[keep_cols]

    def _filter_valid_images(self) -> pd.DataFrame:
        """Filter out images that don't exist or are invalid"""
        valid_images = []
        
        for _, row in self.metadata.iterrows():
            image_path = self.image_dir / row['out_path']
            
            # Check if image exists
            if not image_path.exists():
                continue
            
            # Check if image can be opened
            img = self._load_image(image_path)
            if img is None:
                continue
            if img.size[0] < self.resolution or img.size[1] < self.resolution:
                continue
            
            valid_images.append(row)
        
        return pd.DataFrame(valid_images)
    
    def _create_patches(self) -> list:
        """Create patches from images"""
        patches = []
        
        for _, row in self.metadata.iterrows():
            image_path = self.image_dir / row['out_path']
            
            try:
                img = self._load_image(image_path)
                if img is None:
                    continue
                
                # Create patches
                image_patches = create_patches(
                    img,
                    patch_size=self.resolution,
                    stride=self.patch_stride,
                    min_foreground_frac=self.min_foreground_frac
                )
                
                # Limit patches per image
                if len(image_patches) > self.max_patches_per_image:
                    indices = np.random.choice(
                        len(image_patches),
                        self.max_patches_per_image,
                        replace=False
                    )
                    image_patches = [image_patches[i] for i in indices]
                
                # Add patches with metadata
                for patch in image_patches:
                    patches.append({
                        'image': patch,
                        'metadata': row.copy()
                    })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return patches
    
    def __len__(self) -> int:
        if self.use_patches:
            return len(self.patches)
        else:
            return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a sample from the dataset
        
        Returns:
            image: Image tensor [1, H, W]
            conditions: Dictionary of condition tensors
        """
        if self.use_patches:
            # Get patch data
            patch_data = self.patches[idx]
            image = patch_data['image']
            metadata = patch_data['metadata']
        else:
            # Load full image
            row = self.metadata.iloc[idx]
            image_path = self.image_dir / row['out_path']
            
            img = self._load_image(image_path)
            if img is None:
                raise RuntimeError(f"Failed to load image: {image_path}")
            image = img
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=np.array(image))
            image = transformed['image']  # [H, W] or [C, H, W]
        
        # Ensure image is [1, H, W]
        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[0] == 3:
            # Convert RGB to grayscale
            image = image.mean(dim=0, keepdim=True)
        
        # Prepare conditions
        conditions = self._prepare_conditions(metadata)
        
        return image, conditions
    
    def _prepare_conditions(self, metadata: pd.Series) -> Dict[str, torch.Tensor]:
        """Prepare condition tensors from metadata"""
        conditions = {}
        
        # View position (0=CC, 1=MLO)
        conditions['view'] = torch.tensor(metadata['view'], dtype=torch.long)
        
        # Laterality (0=L, 1=R)
        conditions['laterality'] = torch.tensor(metadata['laterality'], dtype=torch.long)
        
        # Age bin (0-3)
        conditions['age_bin'] = torch.tensor(metadata['age_bin'], dtype=torch.long)
        
        # Cancer presence (0/1)
        conditions['cancer'] = torch.tensor(metadata['cancer'], dtype=torch.long)
        
        # False positive (0/1)
        conditions['false_positive'] = torch.tensor(metadata['false_positive'], dtype=torch.long)
        
        return conditions
    
    def get_condition_stats(self) -> Dict[str, dict]:
        """Get statistics about conditions in the dataset"""
        if self.use_patches:
            df = pd.DataFrame([p['metadata'] for p in self.patches])
        else:
            df = self.metadata
        
        stats = {}
        
        # View distribution
        stats['view'] = df['view'].value_counts().to_dict()
        
        # Laterality distribution
        stats['laterality'] = df['laterality'].value_counts().to_dict()
        
        # Age bin distribution
        stats['age_bin'] = df['age_bin'].value_counts().to_dict()
        
        # Cancer distribution
        stats['cancer'] = df['cancer'].value_counts().to_dict()
        
        # False positive distribution
        stats['false_positive'] = df['false_positive'].value_counts().to_dict()
        
        return stats
    
    def get_balanced_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        """Get a balanced sampler for training"""
        if not self.use_patches:
            raise ValueError("Balanced sampling only available with patches")
        
        # Calculate weights based on condition balance
        weights = []
        
        for patch_data in self.patches:
            metadata = patch_data['metadata']
            
            # Weight based on cancer presence (higher weight for cancer cases)
            if metadata['cancer'] == 1:
                weight = 5.0  # Higher weight for cancer cases
            else:
                weight = 1.0
            
            # Additional weight based on view/laterality balance
            view_weight = 1.0
            laterality_weight = 1.0
            
            weights.append(weight * view_weight * laterality_weight)
        
        weights = torch.tensor(weights, dtype=torch.float)
        
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
