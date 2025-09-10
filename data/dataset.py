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
    """
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        resolution: int = 512,
        patch_stride: int = 256,
        min_foreground_frac: float = 0.1,  # Réduit pour moins filtrer
        transform_mode: str = "train",
        use_patches: bool = True,
        max_patches_per_image: int = 2
    ):
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.patch_stride = patch_stride
        self.min_foreground_frac = min_foreground_frac
        self.transform_mode = transform_mode
        self.use_patches = use_patches
        self.max_patches_per_image = max_patches_per_image

        # Charger et vérifier les métadonnées
        try:
            self.metadata = pd.read_csv(csv_path)
            print(f"Initial metadata rows: {len(self.metadata)}")
        except Exception as e:
            raise ValueError(f"Failed to read CSV {csv_path}: {e}")

        self.metadata = self._standardize_metadata(self.metadata)
        print(f"Standardized metadata rows: {len(self.metadata)}")
        
        self.metadata = self._filter_valid_images()
        print(f"Filtered metadata rows: {len(self.metadata)}")
        
        if len(self.metadata) == 0:
            raise ValueError("No valid images found after filtering. Check image_dir and csv_path.")

        if self.use_patches:
            self.patches = self._create_patches()
            print(f"Patches created: {len(self.patches)}")
            if len(self.patches) == 0:
                raise ValueError("No patches created. Check image loading or patch creation parameters.")
        else:
            self.patches = None

        dataset_size = len(self)
        print(f"Dataset initialized with {dataset_size} samples")
        if dataset_size == 0:
            raise ValueError("Dataset is empty after initialization.")
        if self.use_patches:
            print(f"Using patches with stride {patch_stride}")
        print(f"Transform mode: {transform_mode}")

        self.transforms = get_transforms(
            mode=transform_mode,
            resolution=resolution
        )

    def _load_image(self, image_path: Path) -> Optional[Image.Image]:
        try:
            suffix = image_path.suffix.lower()
            if suffix == ".dcm":
                ds = pydicom.dcmread(str(image_path), force=True)
                pixel_array = ds.pixel_array.astype(np.float32)
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
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _standardize_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df.columns) == 1:
            col = df.columns[0]
            values = [str(v) for v in df[col].tolist() if isinstance(v, str) and v.strip()]
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
                    continue
                try:
                    fp = int(suffix[-1])
                    cancer = int(suffix[-2])
                    age_val = int(suffix[-4:-2])
                    lv = suffix[:-4].upper()
                    laterality_char = lv[0] if lv else 'L'
                    view_str = lv[1:] if len(lv) > 1 else 'CC'
                    laterality = 0 if laterality_char == 'L' else 1
                    view = 0 if view_str == 'CC' else 1
                    if age_val <= 45:
                        age_bin = 0
                    elif age_val <= 55:
                        age_bin = 1
                    elif age_val <= 65:
                        age_bin = 2
                    else:
                        age_bin = 3
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

        required_cols = {"out_path", "view", "laterality", "age_bin", "cancer", "false_positive"}
        if required_cols.issubset(set(df.columns)):
            return df[list(required_cols)]

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

        if 'out_path' in df.columns:
            df['out_path'] = df['out_path'].astype(str).apply(lambda x: x.replace('\\', '/').lstrip('/'))

        if 'laterality' not in df.columns and 'laterality_text' in df.columns:
            df['laterality'] = df['laterality_text'].map(lambda x: 0 if str(x).strip().upper().startswith('L') else 1)

        if 'view' not in df.columns and 'view_text' in df.columns:
            df['view'] = df['view_text'].map(lambda x: 0 if str(x).strip().upper() == 'CC' else 1)

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

        for col in ['view', 'laterality', 'age_bin', 'cancer', 'false_positive']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        keep_cols = ['out_path', 'view', 'laterality', 'age_bin', 'cancer', 'false_positive']
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Metadata missing required columns after standardization: {missing}")
        return df[keep_cols]

    def _filter_valid_images(self) -> pd.DataFrame:
        valid_images = []
        for _, row in self.metadata.iterrows():
            image_path = self.image_dir / row['out_path']
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            img = self._load_image(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue
            if img.size[0] < self.resolution or img.size[1] < self.resolution:
                print(f"Image too small: {image_path}, size={img.size}")
                continue
            valid_images.append(row)
        print(f"Valid images after filtering: {len(valid_images)}")
        return pd.DataFrame(valid_images)

    def _create_patches(self) -> list:
        patches = []
        for _, row in self.metadata.iterrows():
            image_path = self.image_dir / row['out_path']
            try:
                img = self._load_image(image_path)
                if img is None:
                    print(f"Skipping {image_path}: Failed to load image")
                    continue
                image_patches = create_patches(
                    img,
                    patch_size=self.resolution,
                    stride=self.patch_stride,
                    min_foreground_frac=self.min_foreground_frac
                )
                if not image_patches:
                    print(f"No patches for {image_path}: min_foreground_frac={self.min_foreground_frac}")
                if len(image_patches) > self.max_patches_per_image:
                    indices = np.random.choice(len(image_patches), self.max_patches_per_image, replace=False)
                    image_patches = [image_patches[i] for i in indices]
                for patch in image_patches:
                    patches.append({'image': patch, 'metadata': row.copy()})
                print(f"Processed {image_path}: {len(image_patches)} patches")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        print(f"Total patches created: {len(patches)}")
        return patches

    def __len__(self) -> int:
        return len(self.patches) if self.use_patches else len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.use_patches:
            if not self.patches:
                raise ValueError(f"No patches available for index {idx}. Check patch creation.")
            patch_data = self.patches[idx]
            image = patch_data['image']
            metadata = patch_data['metadata'].to_dict() if isinstance(patch_data['metadata'], pd.Series

        else:
            if not len(self.metadata):
                raise ValueError(f"No metadata available for index {idx}. Check CSV and image_dir.")
            row = self.metadata.iloc[idx]
            image_path = self.image_dir / row['out_path']
            img = self._load_image(image_path)
            if img is None:
                print(f"Warning: Failed to load image {image_path}, using fallback")
                image = Image.fromarray(np.zeros((self.resolution, self.resolution), dtype=np.uint8))
            else:
                image = img
            metadata = row

        if self.transforms:
            try:
                transformed = self.transforms(image=np.array(image))
                image = transformed['image']
            except Exception as e:
                print(f"Error applying transforms to image: {e}")
                image = torch.zeros((1, self.resolution, self.resolution))

        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3 and image.shape[0] == 3:
            image = image.mean(dim=0, keepdim=True)

        conditions = self._prepare_conditions(metadata)
        return image, conditions

    def _prepare_conditions(self, metadata: pd.Series) -> Dict[str, torch.Tensor]:
        conditions = {
            'view': torch.tensor(metadata.get('view', 0), dtype=torch.long),
            'laterality': torch.tensor(metadata.get('laterality', 0), dtype=torch.long),
            'age_bin': torch.tensor(metadata.get('age_bin', 1), dtype=torch.long),
            'cancer': torch.tensor(metadata.get('cancer', 0), dtype=torch.long),
            'false_positive': torch.tensor(metadata.get('false_positive', 0), dtype=torch.long),
        }
        return conditions

    def get_condition_stats(self) -> Dict[str, dict]:
        df = pd.DataFrame([p['metadata'] for p in self.patches]) if self.use_patches else self.metadata
        stats = {
            'view': df['view'].value_counts().to_dict(),
            'laterality': df['laterality'].value_counts().to_dict(),
            'age_bin': df['age_bin'].value_counts().to_dict(),
            'cancer': df['cancer'].value_counts().to_dict(),
            'false_positive': df['false_positive'].value_counts().to_dict(),
        }
        return stats

    def get_balanced_sampler(self) -> torch.utils.data.WeightedRandomSampler:
        if not self.use_patches:
            raise ValueError("Balanced sampling only available with patches")
        weights = []
        for patch_data in self.patches:
            metadata = patch_data['metadata']
            weight = 5.0 if metadata.get('cancer', 0) == 1 else 1.0
            weights.append(weight)
        weights = torch.tensor(weights, dtype=torch.float)
        return torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)