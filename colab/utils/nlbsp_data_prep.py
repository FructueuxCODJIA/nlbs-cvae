"""
Data preparation utilities for NLBSP mammography dataset in Google Colab
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import zipfile
from typing import Dict, List, Tuple, Optional
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files, drive
from tqdm import tqdm


def upload_nlbsp_data():
    """Upload NLBSP dataset to Colab"""
    print("📤 Upload your NLBSP dataset files:")
    print("You can upload:")
    print("1. ZIP file containing all DICOM images")
    print("2. Individual files (metadata CSV + images)")
    print("3. Multiple ZIP files")
    
    uploaded = files.upload()
    
    # Process uploaded files
    for filename, content in uploaded.items():
        filepath = f'/content/data/{filename}'
        
        # Create data directory if it doesn't exist
        os.makedirs('/content/data', exist_ok=True)
        
        # Save uploaded file
        with open(filepath, 'wb') as f:
            f.write(content)
        
        print(f"✅ Uploaded: {filename}")
        
        # Extract ZIP files
        if filename.endswith('.zip'):
            print(f"📦 Extracting {filename}...")
            extract_zip_file(filepath, '/content/data')
    
    return list(uploaded.keys())


def extract_zip_file(zip_path: str, extract_to: str):
    """Extract ZIP file and organize contents"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✅ Extracted {zip_path}")
        
        # Remove the ZIP file to save space
        os.remove(zip_path)
        print(f"🗑️ Removed ZIP file to save space")
        
    except Exception as e:
        print(f"❌ Failed to extract {zip_path}: {e}")


def setup_nlbsp_from_drive(drive_path: str = "/content/drive/MyDrive/NLBS_Data"):
    """Setup NLBSP data from Google Drive"""
    print("📁 Setting up data from Google Drive...")
    
    # Mount drive if not already mounted
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    if not os.path.exists(drive_path):
        print(f"❌ Data not found at {drive_path}")
        print("Please upload your data to Google Drive first")
        return False
    
    # Create local data directory
    local_data_dir = '/content/data'
    os.makedirs(local_data_dir, exist_ok=True)
    
    # Copy or link data
    if os.path.exists(f"{drive_path}/NLBSP-metadata.csv"):
        shutil.copy2(f"{drive_path}/NLBSP-metadata.csv", f"{local_data_dir}/NLBSP-metadata.csv")
        print("✅ Copied metadata CSV")
    
    # Link image directory (faster than copying)
    if os.path.exists(f"{drive_path}/images"):
        if os.path.exists(f"{local_data_dir}/images"):
            os.remove(f"{local_data_dir}/images")
        os.symlink(f"{drive_path}/images", f"{local_data_dir}/images")
        print("✅ Linked image directory")
    
    return True


def analyze_nlbsp_dataset(csv_path: str, image_dir: str) -> Dict:
    """Analyze the NLBSP dataset and provide statistics"""
    print("🔍 Analyzing NLBSP dataset...")
    
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    stats = {
        'total_images': len(df),
        'unique_patients': df['File Path'].apply(lambda x: x.split('\\')[1]).nunique(),
        'views': df['View Position'].value_counts().to_dict(),
        'laterality': df['Image Laterality'].value_counts().to_dict(),
        'cancer_distribution': df['Cancer'].value_counts().to_dict(),
        'false_positive_distribution': df['False Positive'].value_counts().to_dict(),
        'age_stats': {
            'min': df['Age'].min(),
            'max': df['Age'].max(),
            'mean': df['Age'].mean(),
            'std': df['Age'].std()
        }
    }
    
    # Check file existence
    missing_files = []
    existing_files = 0
    
    print("📋 Dataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Unique patients: {stats['unique_patients']}")
    print(f"Views: {stats['views']}")
    print(f"Laterality: {stats['laterality']}")
    print(f"Cancer distribution: {stats['cancer_distribution']}")
    print(f"False positive distribution: {stats['false_positive_distribution']}")
    print(f"Age range: {stats['age_stats']['min']}-{stats['age_stats']['max']} (mean: {stats['age_stats']['mean']:.1f})")
    
    # Check a sample of files
    print("\n🔍 Checking file existence (sample)...")
    sample_size = min(100, len(df))
    sample_df = df.sample(n=sample_size)
    
    for _, row in tqdm(sample_df.iterrows(), total=sample_size, desc="Checking files"):
        # Convert Windows path to Unix path
        file_path = row['File Path'].replace('\\', '/')
        full_path = os.path.join(image_dir, file_path)
        
        if os.path.exists(full_path):
            existing_files += 1
        else:
            missing_files.append(file_path)
    
    print(f"✅ Found {existing_files}/{sample_size} files in sample")
    if missing_files:
        print(f"⚠️ Missing files in sample: {len(missing_files)}")
        print("First few missing files:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    
    stats['file_check'] = {
        'sample_size': sample_size,
        'existing_files': existing_files,
        'missing_files': len(missing_files)
    }
    
    return stats


def preprocess_nlbsp_metadata(csv_path: str, output_path: str) -> str:
    """Preprocess NLBSP metadata for training"""
    print("🔄 Preprocessing NLBSP metadata...")
    
    df = pd.read_csv(csv_path)
    
    # Convert Windows paths to Unix paths
    df['File Path'] = df['File Path'].str.replace('\\', '/')
    
    # Create age bins
    age_bins = [0, 50, 60, 70, 80, 100]
    age_labels = [0, 1, 2, 3, 4]  # 0: <50, 1: 50-60, 2: 60-70, 3: 70-80, 4: >80
    df['age_bin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
    
    # Convert categorical variables to numeric
    df['view_numeric'] = df['View Position'].map({'CC': 0, 'MLO': 1})
    df['laterality_numeric'] = df['Image Laterality'].map({'L': 0, 'R': 1})
    
    # Add patient ID
    df['patient_id'] = df['File Path'].apply(lambda x: x.split('/')[1])
    
    # Save preprocessed metadata
    df.to_csv(output_path, index=False)
    
    print(f"✅ Preprocessed metadata saved to: {output_path}")
    print(f"Added columns: age_bin, view_numeric, laterality_numeric, patient_id")
    
    return output_path


def create_data_splits(csv_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
    """Create train/val/test splits ensuring patient-level separation"""
    print("📊 Creating data splits...")
    
    df = pd.read_csv(csv_path)
    
    # Get unique patients
    patients = df['patient_id'].unique()
    np.random.shuffle(patients)
    
    # Calculate split sizes
    n_patients = len(patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    # Split patients
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    
    # Create splits
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    print(f"✅ Data splits created:")
    print(f"  Train: {len(train_df)} images from {len(train_patients)} patients")
    print(f"  Val: {len(val_df)} images from {len(val_patients)} patients")
    print(f"  Test: {len(test_df)} images from {len(test_patients)} patients")
    
    return splits


def visualize_sample_images(csv_path: str, image_dir: str, num_samples: int = 8):
    """Visualize sample images from the dataset"""
    print("🖼️ Visualizing sample images...")
    
    df = pd.read_csv(csv_path)
    
    # Sample diverse images
    sample_df = df.groupby(['View Position', 'Image Laterality', 'Cancer']).apply(
        lambda x: x.sample(min(1, len(x)))
    ).reset_index(drop=True)
    
    sample_df = sample_df.head(num_samples)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        if i >= num_samples:
            break
            
        # Load DICOM image
        file_path = row['File Path']
        full_path = os.path.join(image_dir, file_path)
        
        try:
            # Read DICOM
            dicom = pydicom.dcmread(full_path)
            image = dicom.pixel_array
            
            # Normalize for display
            image = (image - image.min()) / (image.max() - image.min())
            
            # Display
            axes[i].imshow(image, cmap='gray')
            
            # Title with metadata
            cancer_status = 'Cancer' if row['Cancer'] == 1 else 'Normal'
            title = f"{row['View Position']} {row['Image Laterality']} - {cancer_status}\nAge: {row['Age']}"
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error loading\n{file_path}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sample_df), num_samples):
        axes[i].axis('off')
    
    plt.suptitle('Sample NLBSP Mammograms', fontsize=16)
    plt.tight_layout()
    plt.show()


def check_dicom_properties(csv_path: str, image_dir: str, num_samples: int = 10):
    """Check DICOM properties to understand the data format"""
    print("🔍 Checking DICOM properties...")
    
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=min(num_samples, len(df)))
    
    properties = []
    
    for _, row in sample_df.iterrows():
        file_path = row['File Path']
        full_path = os.path.join(image_dir, file_path)
        
        try:
            dicom = pydicom.dcmread(full_path)
            
            props = {
                'file': file_path,
                'rows': dicom.Rows,
                'columns': dicom.Columns,
                'bits_allocated': dicom.BitsAllocated,
                'bits_stored': dicom.BitsStored,
                'pixel_representation': dicom.PixelRepresentation,
                'photometric_interpretation': dicom.PhotometricInterpretation,
                'min_pixel_value': dicom.pixel_array.min(),
                'max_pixel_value': dicom.pixel_array.max(),
                'mean_pixel_value': dicom.pixel_array.mean(),
                'std_pixel_value': dicom.pixel_array.std()
            }
            
            properties.append(props)
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    if properties:
        props_df = pd.DataFrame(properties)
        
        print("📋 DICOM Properties Summary:")
        print(f"Image dimensions: {props_df['rows'].iloc[0]} x {props_df['columns'].iloc[0]}")
        print(f"Bits allocated: {props_df['bits_allocated'].iloc[0]}")
        print(f"Bits stored: {props_df['bits_stored'].iloc[0]}")
        print(f"Photometric interpretation: {props_df['photometric_interpretation'].iloc[0]}")
        print(f"Pixel value range: {props_df['min_pixel_value'].min():.0f} - {props_df['max_pixel_value'].max():.0f}")
        print(f"Mean pixel value: {props_df['mean_pixel_value'].mean():.1f} ± {props_df['std_pixel_value'].mean():.1f}")
        
        return props_df
    
    return None


def setup_nlbsp_for_training(data_source: str = "upload") -> Tuple[str, str]:
    """Complete setup of NLBSP data for training"""
    print("🚀 Setting up NLBSP data for training...")
    
    if data_source == "upload":
        # Upload data
        uploaded_files = upload_nlbsp_data()
        csv_path = "/content/data/NLBSP-metadata.csv"
        image_dir = "/content/data"
        
    elif data_source == "drive":
        # Setup from Google Drive
        success = setup_nlbsp_from_drive()
        if not success:
            return None, None
        csv_path = "/content/data/NLBSP-metadata.csv"
        image_dir = "/content/data/images"
    
    else:
        print(f"❌ Unknown data source: {data_source}")
        return None, None
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"❌ Metadata file not found: {csv_path}")
        return None, None
    
    if not os.path.exists(image_dir):
        print(f"❌ Image directory not found: {image_dir}")
        return None, None
    
    # Analyze dataset
    stats = analyze_nlbsp_dataset(csv_path, image_dir)
    
    # Preprocess metadata
    processed_csv = "/content/data/NLBSP-metadata-processed.csv"
    preprocess_nlbsp_metadata(csv_path, processed_csv)
    
    # Create data splits
    splits = create_data_splits(processed_csv)
    
    # Save splits
    for split_name, split_df in splits.items():
        split_path = f"/content/data/NLBSP-metadata-{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"💾 Saved {split_name} split to: {split_path}")
    
    # Visualize samples
    visualize_sample_images(processed_csv, image_dir)
    
    # Check DICOM properties
    check_dicom_properties(processed_csv, image_dir)
    
    print("✅ NLBSP data setup complete!")
    print(f"📊 Dataset ready for training with {stats['total_images']} images")
    
    return processed_csv, image_dir


if __name__ == "__main__":
    # Example usage
    csv_path, image_dir = setup_nlbsp_for_training(data_source="upload")
    
    if csv_path and image_dir:
        print("🎉 Data setup successful!")
        print(f"CSV: {csv_path}")
        print(f"Images: {image_dir}")
    else:
        print("❌ Data setup failed!")