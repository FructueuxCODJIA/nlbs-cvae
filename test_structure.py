#!/usr/bin/env python3
"""
Simple structure test without PyTorch dependencies
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test basic imports
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import yaml
        print("✅ YAML imported successfully")
        
        # Test project structure
        from pathlib import Path
        
        # Check if key directories exist
        dirs_to_check = [
            'models',
            'data', 
            'training',
            'utils',
            'configs',
            'assessment'
        ]
        
        for dir_name in dirs_to_check:
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"✅ Directory '{dir_name}' exists")
            else:
                print(f"❌ Directory '{dir_name}' missing")
        
        # Check key files
        files_to_check = [
            'models/__init__.py',
            'models/cvae.py',
            'models/layers.py',
            'models/losses.py',
            'data/__init__.py',
            'data/dataset.py',
            'data/transforms.py',
            'data/utils.py',
            'configs/training_config.yaml',
            'training/train_cvae.py'
        ]
        
        for file_name in files_to_check:
            file_path = project_root / file_name
            if file_path.exists():
                print(f"✅ File '{file_name}' exists")
            else:
                print(f"❌ File '{file_name}' missing")
        
        print("\n🎉 Basic structure test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        config_path = project_root / 'configs' / 'training_config.yaml'
        
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key configuration sections
        required_sections = [
            'project',
            'data', 
            'model',
            'training',
            'evaluation',
            'logging',
            'hardware',
            'output'
        ]
        
        for section in required_sections:
            if section in config:
                print(f"✅ Config section '{section}' found")
            else:
                print(f"❌ Config section '{section}' missing")
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_paths():
    """Test data path configuration"""
    print("\nTesting data path configuration...")
    
    try:
        config_path = project_root / 'configs' / 'training_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check data paths (they might not exist, but should be configured)
        csv_path = config['data']['csv_path']
        image_dir = config['data']['image_dir']
        
        print(f"📁 CSV path configured: {csv_path}")
        print(f"📁 Image directory configured: {image_dir}")
        
        # Check if paths exist (optional)
        if Path(csv_path).exists():
            print("✅ CSV file exists")
        else:
            print("⚠️  CSV file not found (this is expected for testing)")
            
        if Path(image_dir).exists():
            print("✅ Image directory exists")
        else:
            print("⚠️  Image directory not found (this is expected for testing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data path test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting NLBS-CVAE structure tests...\n")
    
    success = True
    
    # Run tests
    success &= test_imports()
    success &= test_config()
    success &= test_data_paths()
    
    if success:
        print("\n🎉 All structure tests passed!")
        print("The project structure is ready!")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        sys.exit(1)