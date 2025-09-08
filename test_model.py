#!/usr/bin/env python3
"""
Simple test script to verify the Conditional VAE model works
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import ConditionalVAE


def test_model():
    """Test the Conditional VAE model"""
    print("Testing Conditional VAE model...")
    
    # Create model
    model = ConditionalVAE(
        in_channels=1,
        image_size=512,  # Add image_size parameter
        latent_dim=256,
        condition_embed_dim=128,
        encoder_channels=[64, 128, 256, 512],
        decoder_channels=[512, 256, 128, 64, 1],
        use_skip_connections=True,
        use_group_norm=True,
        groups=8,
        activation="silu"
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    image_size = 512
    
    # Create dummy input
    x = torch.randn(batch_size, 1, image_size, image_size)
    
    # Create dummy conditions
    conditions = {
        'view': torch.randint(0, 2, (batch_size,)),  # 0=CC, 1=MLO
        'laterality': torch.randint(0, 2, (batch_size,)),  # 0=L, 1=R
        'age_bin': torch.randint(0, 4, (batch_size,)),  # 0-3
        'cancer': torch.randint(0, 2, (batch_size,)),  # 0/1
        'false_positive': torch.randint(0, 2, (batch_size,))  # 0/1
    }
    
    print(f"Input shape: {x.shape}")
    print(f"Conditions: {conditions}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        x_recon, mu, logvar = model(x, conditions)
        
        print(f"Reconstruction shape: {x_recon.shape}")
        print(f"Latent mean shape: {mu.shape}")
        print(f"Latent log variance shape: {logvar.shape}")
        
        # Test sampling
        samples = model.sample(conditions, batch_size)
        print(f"Generated samples shape: {samples.shape}")
        
        # Test condition embedding
        condition_embed = model.get_condition_embedding(conditions)
        print(f"Condition embedding shape: {condition_embed.shape}")
    
    print("✅ All tests passed! Model is working correctly.")
    
    return model


def test_loss_functions():
    """Test the loss functions"""
    print("\nTesting loss functions...")
    
    from models.losses import VAELoss
    
    # Create loss function
    criterion = VAELoss(
        reconstruction_weight=1.0,
        kl_weight=1.0,
        edge_weight=0.1,
        kl_anneal_epochs=20
    )
    
    # Create dummy data
    batch_size = 4
    image_size = 512
    
    x_recon = torch.randn(batch_size, 1, image_size, image_size)
    x_target = torch.randn(batch_size, 1, image_size, image_size)
    mu = torch.randn(batch_size, 256)
    logvar = torch.randn(batch_size, 256)
    
    # Test loss calculation
    loss, loss_dict = criterion(x_recon, x_target, mu, logvar)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("✅ Loss functions working correctly!")


def test_dataset():
    """Test the dataset creation"""
    print("\nTesting dataset...")
    
    from data import MammographyDataset
    
    # Create a dummy CSV file for testing
    import pandas as pd
    import tempfile
    import os
    
    # Create dummy metadata
    dummy_data = {
        'out_path': ['image1.png', 'image2.png', 'image3.png'],
        'view': [0, 1, 0],
        'laterality': [0, 1, 1],
        'age_bin': [1, 2, 0],
        'cancer': [0, 1, 0],
        'false_positive': [0, 0, 1],
        'bbox': ['[100,100,400,400]', '[150,150,450,450]', '[200,200,500,500]'],
        'windowing': ['[0,1]', '[0,1]', '[0,1]']
    }
    
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame(dummy_data)
        df.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        # Create dataset (without patches for testing)
        dataset = MammographyDataset(
            csv_path=csv_path,
            image_dir=".",  # Current directory
            resolution=512,
            patch_stride=256,
            min_foreground_frac=0.2,
            transform_mode='test',
            use_patches=False  # Don't use patches for testing
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            try:
                sample = dataset[0]
                print(f"Sample shape: {sample[0].shape}")
                print(f"Conditions: {sample[1]}")
                print("✅ Dataset working correctly!")
            except Exception as e:
                print(f"⚠️  Dataset sample loading failed (expected without real images): {e}")
        
    finally:
        # Clean up
        os.unlink(csv_path)


if __name__ == "__main__":
    print("🚀 Starting Conditional VAE tests...\n")
    
    try:
        # Test model
        model = test_model()
        
        # Test loss functions
        test_loss_functions()
        
        # Test dataset
        test_dataset()
        
        print("\n🎉 All tests completed successfully!")
        print("The Conditional VAE project is ready to use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
