#!/usr/bin/env python3
"""
Quick dataset test script
"""

import yaml
import pandas as pd
from pathlib import Path
from data.dataset import MammographyDataset

def main():
    # Load config
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing dataset with first 10 CSV rows only...")
    
    # Read CSV and limit to first 10 rows for testing
    csv_path = config['data']['csv_path']
    df = pd.read_csv(csv_path)
    print(f"Original CSV has {len(df)} rows")
    
    # Create a small test CSV
    test_csv_path = "test_metadata.csv"
    df_small = df.head(10)
    df_small.to_csv(test_csv_path, index=False)
    print(f"Created test CSV with {len(df_small)} rows")
    
    # Test dataset creation
    try:
        ds = MammographyDataset(
            csv_path=test_csv_path,
            image_dir=config['data']['image_dir'],
            resolution=config['data']['resolution'],
            patch_stride=config['data']['patch_stride'],
            min_foreground_frac=config['data']['min_foreground_frac'],
            transform_mode='train',
            use_patches=True,
            max_patches_per_image=1  # Just 1 patch per image for speed
        )
        
        print(f"Dataset created successfully with {len(ds)} samples")
        
        if len(ds) > 0:
            print("Testing first sample...")
            sample = ds[0]
            image, conditions = sample
            print(f"Image shape: {image.shape}")
            print(f"Conditions: {conditions}")
            print("SUCCESS: Dataset is working!")
        else:
            print("WARNING: Dataset is empty")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    Path(test_csv_path).unlink(missing_ok=True)

if __name__ == "__main__":
    main()