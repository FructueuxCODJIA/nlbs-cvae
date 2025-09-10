#!/usr/bin/env python3
"""
Quick setup script for NLBS-CVAE on Google Colab
Run this script to automatically set up the environment
"""

import os
import sys
import subprocess
import torch
from pathlib import Path

def check_environment():
    """Check if we're running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_colab_environment():
    """Set up the Colab environment"""
    print("🚀 Setting up NLBS-CVAE for Google Colab...")
    
    # Check if we're in Colab
    if not check_environment():
        print("⚠️ This script is designed for Google Colab")
        print("For local setup, please use the main installation instructions")
        return False
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ No GPU detected. Training will be slow.")
    
    # Create necessary directories
    directories = [
        '/content/results',
        '/content/results/checkpoints',
        '/content/results/logs',
        '/content/results/galleries',
        '/content/data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install requirements
    print("📦 Installing requirements...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 
            '/content/nlbs-cvae/colab/requirements_colab.txt'
        ], check=True, capture_output=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    # Mount Google Drive
    print("📁 Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully")
    except Exception as e:
        print(f"⚠️ Could not mount Google Drive: {e}")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Open NLBS_CVAE_Training.ipynb for training")
    print("2. Open NLBS_CVAE_Inference.ipynb for generation")
    print("3. Check the colab/README.md for detailed instructions")
    
    return True

def create_demo_data():
    """Create demo data for testing"""
    print("🔬 Creating demo data...")
    
    try:
        # Import the helper function
        sys.path.append('/content/nlbs-cvae')
        from colab.utils.colab_helpers import create_synthetic_demo_data
        
        csv_path, data_dir = create_synthetic_demo_data(num_patients=10)
        print(f"✅ Demo data created:")
        print(f"   CSV: {csv_path}")
        print(f"   Images: {data_dir}")
        
        return csv_path, data_dir
    except Exception as e:
        print(f"❌ Failed to create demo data: {e}")
        return None, None

def main():
    """Main setup function"""
    print("=" * 60)
    print("NLBS-CVAE Google Colab Setup")
    print("=" * 60)
    
    # Setup environment
    if not setup_colab_environment():
        return
    
    # Ask user if they want demo data
    print("\n" + "=" * 40)
    print("Would you like to create demo data for testing?")
    print("This will create synthetic mammogram images.")
    print("=" * 40)
    
    # In Colab, we'll create demo data by default
    create_demo_data()
    
    print("\n" + "=" * 60)
    print("Setup completed successfully! 🎉")
    print("You can now run the training or inference notebooks.")
    print("=" * 60)

if __name__ == "__main__":
    main()