#!/usr/bin/env python3
"""
Quick training test with reduced dataset
"""

import yaml
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.dataset import MammographyDataset
from models import ConditionalVAE
from models.losses import VAELoss

def main():
    # Load config
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Creating reduced dataset for quick test...")
    
    # Create small test dataset (first 50 rows)
    csv_path = config['data']['csv_path']
    df = pd.read_csv(csv_path)
    test_csv_path = "quick_test_metadata.csv"
    df_small = df.head(50)  # Just 50 images for quick test
    df_small.to_csv(test_csv_path, index=False)
    
    # Create dataset
    dataset = MammographyDataset(
        csv_path=test_csv_path,
        image_dir=config['data']['image_dir'],
        resolution=config['data']['resolution'],
        patch_stride=config['data']['patch_stride'],
        min_foreground_frac=config['data']['min_foreground_frac'],
        transform_mode='train',
        use_patches=True,
        max_patches_per_image=1
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("No samples found, exiting")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Very small batch
        shuffle=True,
        num_workers=0,  # No multiprocessing for simplicity
        pin_memory=False
    )
    
    # Create model
    device = torch.device('cpu')
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
    ).to(device)
    
    # Create loss
    criterion = VAELoss(
        reconstruction_weight=config['training']['loss']['reconstruction_weight'],
        kl_weight=config['training']['loss']['kl_weight'],
        edge_weight=config['training']['loss']['edge_weight'],
        kl_anneal_epochs=config['training']['loss']['kl_anneal_epochs']
    )
    
    print("Testing forward pass...")
    
    # Test one batch
    model.train()
    for batch_idx, (images, conditions) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images shape {images.shape}")
        
        images = images.to(device)
        conditions = {k: v.to(device) for k, v in conditions.items()}
        
        # Forward pass
        try:
            x_recon, mu, logvar = model(images, conditions)
            print(f"Reconstruction shape: {x_recon.shape}")
            print(f"Latent mu shape: {mu.shape}")
            print(f"Latent logvar shape: {logvar.shape}")
            
            # Test loss
            loss, loss_dict = criterion(x_recon, images, mu, logvar)
            print(f"Loss: {loss.item():.4f}")
            print(f"Loss components: {loss_dict}")
            
            print("SUCCESS: Forward pass and loss computation work!")
            break
            
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Clean up
    Path(test_csv_path).unlink(missing_ok=True)
    print("Quick test completed!")

if __name__ == "__main__":
    main()