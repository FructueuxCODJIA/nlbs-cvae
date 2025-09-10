"""
Colab-specific utility functions for NLBS-CVAE
"""

import os
import torch
import shutil
import zipfile
import gdown
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from google.colab import files, drive
import matplotlib.pyplot as plt
from IPython.display import clear_output
import psutil
import GPUtil


def setup_colab_environment():
    """Setup the Colab environment for NLBS-CVAE training"""
    print("🚀 Setting up Colab environment...")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ No GPU detected. Training will be slow.")
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"💾 RAM: {ram_gb:.1f} GB")
    
    # Create directories
    directories = [
        '/content/results',
        '/content/results/checkpoints',
        '/content/results/logs',
        '/content/results/galleries',
        '/content/data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Environment setup complete!")


def mount_drive_and_setup_paths():
    """Mount Google Drive and setup data paths"""
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Check if data exists in Drive
    drive_data_path = '/content/drive/MyDrive/NLBS_Data'
    if os.path.exists(drive_data_path):
        print(f"✅ Found data in Google Drive: {drive_data_path}")
        return drive_data_path
    else:
        print(f"⚠️ No data found at {drive_data_path}")
        return None


def download_demo_data():
    """Download or create demo data for testing"""
    print("📥 Setting up demo data...")
    
    # This would download demo data from a public source
    # For now, we'll create synthetic data
    create_synthetic_demo_data()


def create_synthetic_demo_data(num_patients: int = 20):
    """Create synthetic mammography data for testing"""
    print(f"🔬 Creating synthetic data for {num_patients} patients...")
    
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    
    # Create demo directory
    demo_dir = Path('/content/data/demo')
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_rows = []
    views = ['CC', 'MLO']
    lateralities = ['L', 'R']
    ages = [45, 52, 58, 63, 67]
    
    for patient_id in range(1, num_patients + 1):
        age = np.random.choice(ages)
        cancer = np.random.choice([0, 1], p=[0.7, 0.3])
        
        for laterality in lateralities:
            for view in views:
                # Create synthetic mammogram
                image = create_synthetic_mammogram()
                
                # Create filename
                filename = f"patient_{patient_id:03d}_{laterality}_{view}.dcm"
                filepath = demo_dir / filename
                
                # Create and save DICOM
                ds = create_synthetic_dicom(image, str(filepath))
                ds.save_as(filepath)
                
                # Add to metadata
                metadata_rows.append({
                    'File Path': f'demo/{filename}',
                    'Image Laterality': laterality,
                    'View Position': view,
                    'Age': age,
                    'Cancer': cancer,
                    'False Positive': np.random.choice([0, 1], p=[0.9, 0.1])
                })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv('/content/data/metadata.csv', index=False)
    
    print(f"✅ Created {len(metadata_rows)} synthetic samples")
    return '/content/data/metadata.csv', '/content/data'


def create_synthetic_mammogram(width: int = 512, height: int = 512) -> np.ndarray:
    """Create a synthetic mammogram-like image"""
    # Create base tissue pattern
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    
    # Simulate breast tissue with varying density
    tissue = np.exp(-((x-0.5)**2 + (y-0.5)**2) * 3)
    
    # Add some texture and noise
    noise = np.random.normal(0, 0.1, (height, width))
    texture = np.sin(x * 20) * np.sin(y * 20) * 0.1
    
    # Combine and normalize
    image = tissue + texture + noise
    image = np.clip(image, 0, 1)
    
    # Convert to uint16 (typical for mammograms)
    image = (image * 65535).astype(np.uint16)
    
    return image


def create_synthetic_dicom(image_array: np.ndarray, filename: str):
    """Create a synthetic DICOM file"""
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    
    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = '1.2.3.4.5.6.7.8.9.10'
    file_meta.ImplementationClassUID = '1.2.3.4.5.6.7.8.9.10'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    # Create the main dataset
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\\0" * 128)
    
    # Add required DICOM tags
    ds.PatientName = "Demo^Patient"
    ds.PatientID = "DEMO001"
    ds.Modality = "MG"
    ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.10.11"
    ds.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.10.11.12"
    ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.10.11.12.13"
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.1.2'
    
    # Image-specific tags
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = image_array.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = image_array.tobytes()
    
    return ds


def monitor_resources():
    """Monitor GPU and RAM usage"""
    # GPU monitoring
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_utilization = gpu_memory_used / gpu_memory_total * 100
        
        print(f"🔥 GPU Memory: {gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB ({gpu_utilization:.1f}%)")
    
    # RAM monitoring
    ram = psutil.virtual_memory()
    ram_used = ram.used / 1e9
    ram_total = ram.total / 1e9
    ram_percent = ram.percent
    
    print(f"💾 RAM: {ram_used:.1f}/{ram_total:.1f} GB ({ram_percent:.1f}%)")


def cleanup_memory():
    """Clean up GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    
    print("🧹 Memory cleaned up")


def backup_to_drive(source_dir: str, backup_name: str):
    """Backup results to Google Drive"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f'/content/drive/MyDrive/{backup_name}_{timestamp}'
    
    if os.path.exists(source_dir):
        shutil.copytree(source_dir, backup_dir)
        print(f"✅ Backed up to: {backup_dir}")
        return backup_dir
    else:
        print(f"❌ Source directory not found: {source_dir}")
        return None


def create_training_summary(config: Dict[str, Any], metrics: Dict[str, float]):
    """Create a training summary report"""
    summary = f"""
# NLBS-CVAE Training Summary

## Configuration
- Model: {config['project']['name']}
- Experiment: {config['project']['experiment_id']}
- Epochs: {config['training']['num_epochs']}
- Batch Size: {config['data']['batch_size']}
- Learning Rate: {config['training']['learning_rate']}
- Image Resolution: {config['data']['resolution']}

## Results
- Best Validation Loss: {metrics.get('best_val_loss', 'N/A'):.4f}
- Final Training Loss: {metrics.get('final_train_loss', 'N/A'):.4f}
- Total Parameters: {metrics.get('total_params', 'N/A'):,}

## Hardware
- Device: {config['hardware']['device']}
- Mixed Precision: {config['training']['mixed_precision']}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return summary


def visualize_training_progress(log_dir: str):
    """Create training progress visualization"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Load TensorBoard logs
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        # Get scalar data
        train_loss = event_acc.Scalars('Train/Loss')
        val_loss = event_acc.Scalars('Val/Loss')
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training loss
        steps = [x.step for x in train_loss]
        values = [x.value for x in train_loss]
        ax1.plot(steps, values, label='Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Validation loss
        if val_loss:
            val_steps = [x.step for x in val_loss]
            val_values = [x.value for x in val_loss]
            ax2.plot(val_steps, val_values, label='Validation Loss', color='orange')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Validation Loss')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('/content/results/training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Could not create training visualization: {e}")


def download_from_url(url: str, output_path: str):
    """Download file from URL (supports Google Drive links)"""
    try:
        if 'drive.google.com' in url:
            gdown.download(url, output_path, quiet=False)
        else:
            import urllib.request
            urllib.request.urlretrieve(url, output_path)
        
        print(f"✅ Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def upload_files():
    """Upload files using Colab's file upload widget"""
    print("📤 Select files to upload:")
    uploaded = files.upload()
    
    for filename, content in uploaded.items():
        with open(f'/content/data/{filename}', 'wb') as f:
            f.write(content)
        print(f"✅ Uploaded: {filename}")
    
    return list(uploaded.keys())


def create_colab_readme():
    """Create a README for the Colab version"""
    readme_content = """# NLBS-CVAE Colab Version

This is the Google Colab adaptation of the NLBS-CVAE project for mammography generation.

## Quick Start

1. Open `notebooks/NLBS_CVAE_Training.ipynb` in Google Colab
2. Run all cells to start training with demo data
3. Monitor progress with TensorBoard
4. Results are automatically saved to Google Drive

## Features

- 🚀 Optimized for Colab GPU/TPU
- 💾 Google Drive integration
- 📊 Real-time monitoring
- 🔄 Automatic session management
- 📈 Memory optimization

## Configuration

Edit `configs/colab_training_config.yaml` to customize:
- Model architecture
- Training parameters
- Data paths
- Hardware settings

## Data Options

1. **Demo Data**: Synthetic mammograms for testing
2. **Upload**: Use Colab's file upload widget
3. **Google Drive**: Link to your Drive data

## Results

All results are automatically backed up to Google Drive:
- Model checkpoints
- Generated images
- Training logs
- Configuration files

## Support

For issues or questions, please refer to the main repository:
https://github.com/FructueuxCODJIA/nlbs-cvae
"""
    
    return readme_content