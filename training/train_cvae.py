#!/usr/bin/env python3
"""
Training script for Conditional VAE on mammography images
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import ConditionalVAE
from models.losses import VAELoss
from data import MammographyDataset
from utils.training_utils import (
    setup_logging, 
    save_checkpoint, 
    load_checkpoint,
    create_optimizer,
    create_scheduler,
    set_seed
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Conditional VAE")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed"
    )
    return parser.parse_args()


def train_epoch(
    model: ConditionalVAE,
    train_loader: DataLoader,
    criterion: VAELoss,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    criterion.update_epoch(epoch)
    
    total_loss = 0.0
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_edge_loss = 0.0
    
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, conditions) in enumerate(progress_bar):
        # Move to device
        images = images.to(device)
        conditions = {k: v.to(device) for k, v in conditions.items()}
        
        # Forward pass
        optimizer.zero_grad()
        
        x_recon, mu, logvar = model(images, conditions)
        
        # Calculate loss
        loss, loss_dict = criterion(x_recon, images, mu, logvar)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Update running totals
        total_loss += loss_dict['total_loss']
        total_rec_loss += loss_dict['reconstruction_loss']
        total_kl_loss += loss_dict['kl_loss']
        total_edge_loss += loss_dict['edge_loss']
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss_dict['total_loss']:.4f}",
            'Rec': f"{loss_dict['reconstruction_loss']:.4f}",
            'KL': f"{loss_dict['kl_loss']:.4f}",
            'Edge': f"{loss_dict['edge_loss']:.4f}",
            'KL_w': f"{loss_dict['kl_weight']:.3f}"
        })
        
        # Log to tensorboard
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss', loss_dict['total_loss'], step)
            writer.add_scalar('Train/Reconstruction_Loss', loss_dict['reconstruction_loss'], step)
            writer.add_scalar('Train/KL_Loss', loss_dict['kl_loss'], step)
            writer.add_scalar('Train/Edge_Loss', loss_dict['edge_loss'], step)
            writer.add_scalar('Train/KL_Weight', loss_dict['kl_weight'], step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], step)
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_rec_loss = total_rec_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_edge_loss = total_edge_loss / num_batches
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_rec_loss,
        'kl_loss': avg_kl_loss,
        'edge_loss': avg_edge_loss
    }


def validate_epoch(
    model: ConditionalVAE,
    val_loader: DataLoader,
    criterion: VAELoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_edge_loss = 0.0
    
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, conditions) in enumerate(tqdm(val_loader, desc=f"Validation {epoch}")):
            # Move to device
            images = images.to(device)
            conditions = {k: v.to(device) for k, v in conditions.items()}
            
            # Forward pass
            x_recon, mu, logvar = model(images, conditions)
            
            # Calculate loss
            loss, loss_dict = criterion(x_recon, images, mu, logvar)
            
            # Update running totals
            total_loss += loss_dict['total_loss']
            total_rec_loss += loss_dict['reconstruction_loss']
            total_kl_loss += loss_dict['kl_loss']
            total_edge_loss += loss_dict['edge_loss']
    
    # Calculate averages
    avg_loss = total_loss / num_batches
    avg_rec_loss = total_rec_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_edge_loss = total_edge_loss / num_batches
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Reconstruction_Loss', avg_rec_loss, epoch)
    writer.add_scalar('Val/KL_Loss', avg_kl_loss, epoch)
    writer.add_scalar('Val/Edge_Loss', avg_edge_loss, epoch)
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_rec_loss,
        'kl_loss': avg_kl_loss,
        'edge_loss': avg_edge_loss
    }


def save_sample_images(
    model: ConditionalVAE,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    num_samples: int = 8
):
    """Save sample images to tensorboard"""
    model.eval()
    
    # Get a batch
    images, conditions = next(iter(val_loader))
    images = images[:num_samples].to(device)
    conditions = {k: v[:num_samples].to(device) for k, v in conditions.items()}
    
    with torch.no_grad():
        # Generate reconstructions
        x_recon, _, _ = model(images, conditions)
        
        # Generate samples
        x_gen = model.sample(conditions, num_samples)
        
        # Denormalize images (assuming normalization to [-1, 1])
        images_denorm = (images + 1) / 2
        x_recon_denorm = (x_recon + 1) / 2
        x_gen_denorm = (x_gen + 1) / 2
        
        # Create image grid
        from torchvision.utils import make_grid
        
        # Original images
        orig_grid = make_grid(images_denorm, nrow=4, normalize=False)
        writer.add_image('Images/Original', orig_grid, epoch)
        
        # Reconstructions
        recon_grid = make_grid(x_recon_denorm, nrow=4, normalize=False)
        writer.add_image('Images/Reconstruction', recon_grid, epoch)
        
        # Generated samples
        gen_grid = make_grid(x_gen_denorm, nrow=4, normalize=False)
        writer.add_image('Images/Generated', gen_grid, epoch)


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(config['project']['seed'])
    
    # Setup logging
    log_dir = Path(config['output']['results_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_dir / 'training.log')
    logger.info(f"Starting training with config: {args.config}")
    
    # Setup device (respect config hardware.device)
    dev_cfg = config.get('hardware', {}).get('device', 'auto')
    if dev_cfg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif dev_cfg == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MammographyDataset(
        csv_path=config['data']['csv_path'],
        image_dir=config['data']['image_dir'],
        resolution=config['data']['resolution'],
        patch_stride=config['data']['patch_stride'],
        min_foreground_frac=config['data']['min_foreground_frac'],
        transform_mode='train',
        use_patches=True
    )
    
    val_dataset = MammographyDataset(
        csv_path=config['data']['csv_path'],
        image_dir=config['data']['image_dir'],
        resolution=config['data']['resolution'],
        patch_stride=config['data']['patch_stride'],
        min_foreground_frac=config['data']['min_foreground_frac'],
        transform_mode='val',
        use_patches=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create model
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
    
    model = model.to(device)
    
    # Create loss function
    criterion = VAELoss(
        reconstruction_weight=config['training']['loss']['reconstruction_weight'],
        kl_weight=config['training']['loss']['kl_weight'],
        edge_weight=config['training']['loss']['edge_weight'],
        kl_anneal_epochs=config['training']['loss']['kl_anneal_epochs']
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model, 
        config['training']['optimizer'],
        config['training']['learning_rate'],
        config['training']['weight_decay'],
        config['training']['beta1'],
        config['training']['beta2']
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        config['training']['scheduler'],
        config['training']['num_epochs'],
        config['training']['warmup_epochs'],
        config['training']['min_lr']
    )
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training loop")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, writer, config
        )
        
        # Validate
        if epoch % config['evaluation']['val_every_n_epochs'] == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch, writer, config
            )
            
            # Save sample images
            if epoch % config['logging']['save_images_every_n_epochs'] == 0:
                save_sample_images(
                    model, val_loader, device, epoch, writer,
                    config['evaluation']['num_val_samples']
                )
            
            # Check if best validation loss
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save best model
                if config['training']['save_best']:
                    checkpoint_path = Path(config['output']['checkpoints_dir']) / 'best_model.pt'
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    save_checkpoint(
                        checkpoint_path,
                        model, optimizer, scheduler, epoch, best_val_loss,
                        config
                    )
        
        # Save checkpoint
        if epoch % config['training']['save_every_n_epochs'] == 0:
            checkpoint_path = Path(config['output']['checkpoints_dir']) / f'checkpoint_epoch_{epoch}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_checkpoint(
                checkpoint_path,
                model, optimizer, scheduler, epoch, best_val_loss,
                config
            )
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )
    
    # Save final model
    final_checkpoint_path = Path(config['output']['checkpoints_dir']) / 'final_model.pt'
    save_checkpoint(
        final_checkpoint_path,
        model, optimizer, scheduler, config['training']['num_epochs'] - 1,
        best_val_loss, config
    )
    
    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
