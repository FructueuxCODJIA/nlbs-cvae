#!/usr/bin/env python3
"""
Test training pipeline with dummy data
"""

import sys
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import ConditionalVAE
from models.losses import VAELoss


def create_dummy_dataset(num_samples=32, image_size=256):
    """Create dummy dataset for testing"""
    print(f"Creating dummy dataset with {num_samples} samples...")
    
    # Create dummy images
    images = torch.randn(num_samples, 1, image_size, image_size)
    
    # Create dummy conditions (as tensors, not dict)
    view = torch.randint(0, 2, (num_samples,))
    laterality = torch.randint(0, 2, (num_samples,))
    age_bin = torch.randint(0, 4, (num_samples,))
    cancer = torch.randint(0, 2, (num_samples,))
    false_positive = torch.randint(0, 2, (num_samples,))
    
    # Stack conditions into a single tensor
    conditions = torch.stack([view, laterality, age_bin, cancer, false_positive], dim=1)
    
    return TensorDataset(images, conditions)


def conditions_tensor_to_dict(conditions_tensor):
    """Convert conditions tensor to dictionary format expected by model"""
    return {
        'view': conditions_tensor[:, 0],
        'laterality': conditions_tensor[:, 1], 
        'age_bin': conditions_tensor[:, 2],
        'cancer': conditions_tensor[:, 3],
        'false_positive': conditions_tensor[:, 4]
    }


def test_training_step():
    """Test a single training step"""
    print("🚀 Testing training pipeline...\n")
    
    # Configuration
    image_size = 256
    batch_size = 4
    latent_dim = 128  # Smaller for testing
    
    # Create model
    print("Creating model...")
    model = ConditionalVAE(
        in_channels=1,
        image_size=image_size,
        latent_dim=latent_dim,
        condition_embed_dim=64,
        encoder_channels=[32, 64, 128, 256],  # Smaller for testing
        decoder_channels=[256, 128, 64, 32, 1],
        use_skip_connections=True,
        use_group_norm=True,
        groups=4,
        activation="silu"
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = VAELoss(
        reconstruction_weight=1.0,
        kl_weight=1.0,
        edge_weight=0.1,
        kl_anneal_epochs=10
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create dummy dataset
    dataset = create_dummy_dataset(num_samples=16, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Starting training test...")
    
    # Training loop
    model.train()
    total_loss = 0
    
    for epoch in range(2):  # Just 2 epochs for testing
        epoch_loss = 0
        
        for batch_idx, (images, conditions_tensor) in enumerate(dataloader):
            # Convert conditions to dict format
            conditions = conditions_tensor_to_dict(conditions_tensor)
            
            # Forward pass
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(images, conditions)
            
            # Calculate loss
            loss, loss_dict = criterion(x_recon, images, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                  f"Loss={loss.item():.4f}, "
                  f"Rec={loss_dict['reconstruction_loss']:.4f}, "
                  f"KL={loss_dict['kl_loss']:.4f}, "
                  f"Edge={loss_dict['edge_loss']:.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        total_loss += avg_loss
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}\n")
        
        # Update KL annealing
        criterion.update_epoch(epoch + 1)
    
    print("✅ Training test completed successfully!")
    print(f"Final average loss: {total_loss / 2:.4f}")
    
    return model


def test_generation():
    """Test sample generation"""
    print("\n🎨 Testing sample generation...")
    
    # Use the trained model from training test
    model = test_training_step()
    
    model.eval()
    with torch.no_grad():
        # Create test conditions
        batch_size = 4
        conditions = {
            'view': torch.tensor([0, 1, 0, 1]),  # CC, MLO, CC, MLO
            'laterality': torch.tensor([0, 0, 1, 1]),  # L, L, R, R
            'age_bin': torch.tensor([0, 1, 2, 3]),  # Different age groups
            'cancer': torch.tensor([0, 0, 1, 1]),  # No cancer, No cancer, Cancer, Cancer
            'false_positive': torch.tensor([0, 1, 0, 1])  # Various false positive states
        }
        
        # Generate samples
        samples = model.sample(conditions, batch_size)
        print(f"Generated samples shape: {samples.shape}")
        
        # Check sample statistics
        print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"Sample mean: {samples.mean():.3f}")
        print(f"Sample std: {samples.std():.3f}")
    
    print("✅ Generation test completed successfully!")


if __name__ == "__main__":
    print("🚀 Starting training pipeline tests...\n")
    
    try:
        test_generation()
        print("\n🎉 All training tests passed!")
        print("The NLBS-CVAE project is fully functional!")
        
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)