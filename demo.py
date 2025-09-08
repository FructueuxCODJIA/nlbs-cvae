#!/usr/bin/env python3
"""
Demo script showing how to use the NLBS-CVAE project
"""

import sys
import torch
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import ConditionalVAE
from models.losses import VAELoss


def load_config():
    """Load configuration from YAML file"""
    config_path = project_root / 'configs' / 'training_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config):
    """Create model from configuration"""
    print("Creating Conditional VAE model...")
    
    model = ConditionalVAE(
        in_channels=config['data']['channels'],
        image_size=config['data']['resolution'],
        latent_dim=config['model']['latent_dim'],
        condition_embed_dim=config['model']['condition_embed_dim'],
        encoder_channels=config['model']['encoder']['channels'],
        decoder_channels=config['model']['decoder']['channels'],
        use_skip_connections=config['model']['decoder']['use_skip_connections'],
        use_group_norm=config['model']['encoder']['use_group_norm'],
        groups=config['model']['encoder']['groups'],
        activation=config['model']['encoder']['activation'],
        posterior_bias=config['model'].get('conditioning', {}).get('posterior_bias', True),
    )
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_loss_from_config(config):
    """Create loss function from configuration"""
    print("Creating VAE loss function...")
    
    criterion = VAELoss(
        reconstruction_weight=config['training']['loss']['reconstruction_weight'],
        kl_weight=config['training']['loss']['kl_weight'],
        edge_weight=config['training']['loss']['edge_weight'],
        kl_anneal_epochs=config['training']['loss']['kl_anneal_epochs']
    )
    
    print("✅ Loss function created")
    return criterion


def demo_conditional_generation():
    """Demonstrate conditional generation with different conditions"""
    print("\n🎨 Demonstrating conditional generation...")
    
    config = load_config()
    model = create_model_from_config(config)
    
    model.eval()
    with torch.no_grad():
        # Define different mammography conditions
        conditions_list = [
            {
                'name': 'Young, Left CC, No Cancer',
                'conditions': {
                    'view': torch.tensor([0]),  # CC
                    'laterality': torch.tensor([0]),  # Left
                    'age_bin': torch.tensor([0]),  # Young
                    'cancer': torch.tensor([0]),  # No cancer
                    'false_positive': torch.tensor([0])  # Not false positive
                }
            },
            {
                'name': 'Older, Right MLO, Cancer',
                'conditions': {
                    'view': torch.tensor([1]),  # MLO
                    'laterality': torch.tensor([1]),  # Right
                    'age_bin': torch.tensor([3]),  # Older
                    'cancer': torch.tensor([1]),  # Cancer
                    'false_positive': torch.tensor([0])  # Not false positive
                }
            },
            {
                'name': 'Middle-aged, Left MLO, False Positive',
                'conditions': {
                    'view': torch.tensor([1]),  # MLO
                    'laterality': torch.tensor([0]),  # Left
                    'age_bin': torch.tensor([2]),  # Middle-aged
                    'cancer': torch.tensor([0]),  # No cancer
                    'false_positive': torch.tensor([1])  # False positive
                }
            },
            {
                'name': 'Young, Right CC, Cancer',
                'conditions': {
                    'view': torch.tensor([0]),  # CC
                    'laterality': torch.tensor([1]),  # Right
                    'age_bin': torch.tensor([1]),  # Young-middle
                    'cancer': torch.tensor([1]),  # Cancer
                    'false_positive': torch.tensor([0])  # Not false positive
                }
            }
        ]
        
        print(f"Generating samples for {len(conditions_list)} different conditions...")
        
        for i, cond_info in enumerate(conditions_list):
            print(f"\n{i+1}. {cond_info['name']}")
            
            # Generate sample
            sample = model.sample(cond_info['conditions'], 1)
            
            # Print sample statistics
            print(f"   Sample shape: {sample.shape}")
            print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")
            print(f"   Mean: {sample.mean():.3f}, Std: {sample.std():.3f}")
            
            # Convert to numpy for potential visualization
            sample_np = sample.squeeze().cpu().numpy()
            print(f"   Ready for visualization: {sample_np.shape}")


def demo_latent_interpolation():
    """Demonstrate latent space interpolation"""
    print("\n🔄 Demonstrating latent space interpolation...")
    
    config = load_config()
    model = create_model_from_config(config)
    
    model.eval()
    with torch.no_grad():
        # Create two different conditions
        condition_a = {
            'view': torch.tensor([0]),  # CC
            'laterality': torch.tensor([0]),  # Left
            'age_bin': torch.tensor([0]),  # Young
            'cancer': torch.tensor([0]),  # No cancer
            'false_positive': torch.tensor([0])
        }
        
        condition_b = {
            'view': torch.tensor([1]),  # MLO
            'laterality': torch.tensor([1]),  # Right
            'age_bin': torch.tensor([3]),  # Older
            'cancer': torch.tensor([1]),  # Cancer
            'false_positive': torch.tensor([0])
        }
        
        # Sample from latent space
        latent_dim = config['model']['latent_dim']
        z_a = torch.randn(1, latent_dim)
        z_b = torch.randn(1, latent_dim)
        
        print("Interpolating between two latent points...")
        
        # Interpolate
        num_steps = 5
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z_a + alpha * z_b
            
            # Use condition A for all interpolations
            sample = model.decode(z_interp, condition_a)
            
            print(f"Step {i+1}/{num_steps} (α={alpha:.2f}): "
                  f"range=[{sample.min():.3f}, {sample.max():.3f}], "
                  f"mean={sample.mean():.3f}")


def demo_training_setup():
    """Demonstrate how to set up training"""
    print("\n🏋️ Demonstrating training setup...")
    
    config = load_config()
    
    # Create model and loss
    model = create_model_from_config(config)
    criterion = create_loss_from_config(config)
    
    # Create optimizer (as would be done in training)
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    beta1 = float(config['training']['beta1'])
    beta2 = float(config['training']['beta2'])
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    print(f"✅ Optimizer created: {type(optimizer).__name__}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Weight decay: {config['training']['weight_decay']}")
    
    # Show training configuration
    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Image resolution: {config['data']['resolution']}")
    print(f"   Latent dimension: {config['model']['latent_dim']}")
    
    print(f"\n💾 Output Configuration:")
    print(f"   Results directory: {config['output']['results_dir']}")
    if 'save_every_n_epochs' in config['output']:
        print(f"   Save every N epochs: {config['output']['save_every_n_epochs']}")
    else:
        print(f"   Save every N epochs: Not configured")


def main():
    """Main demo function"""
    print("🚀 NLBS-CVAE Project Demo")
    print("=" * 50)
    
    try:
        # Load and show configuration
        config = load_config()
        print(f"✅ Configuration loaded from: configs/training_config.yaml")
        print(f"   Project: {config['project']['name']}")
        if 'version' in config['project']:
            print(f"   Version: {config['project']['version']}")
        
        # Run demos
        demo_conditional_generation()
        demo_latent_interpolation()
        demo_training_setup()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your mammography dataset")
        print("2. Update the data paths in configs/training_config.yaml")
        print("3. Run: python training/train_cvae.py")
        print("4. Monitor training with: tensorboard --logdir results/logs")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()