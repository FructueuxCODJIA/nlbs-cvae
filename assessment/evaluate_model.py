#!/usr/bin/env python3
"""
Evaluation script for Conditional VAE on mammography images
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import ConditionalVAE
from data import MammographyDataset
from utils.training_utils import setup_logging, load_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Conditional VAE")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1000,
        help="Number of samples to generate for evaluation"
    )
    return parser.parse_args()


def calculate_metrics(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    metrics = {}
    
    # Convert to numpy for metric calculation
    real_np = real_images.cpu().numpy()
    gen_np = generated_images.cpu().numpy()
    
    # PSNR
    mse = np.mean((real_np - gen_np) ** 2)
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    metrics['psnr'] = psnr
    
    # SSIM (simplified version)
    metrics['ssim'] = calculate_ssim(real_np, gen_np)
    
    # LPIPS (if available)
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        real_lpips = real_images.to(device)
        gen_lpips = generated_images.to(device)
        
        # Convert to RGB for LPIPS
        if real_lpips.shape[1] == 1:
            real_lpips = real_lpips.repeat(1, 3, 1, 1)
            gen_lpips = gen_lpips.repeat(1, 3, 1, 1)
        
        lpips_score = loss_fn(real_lpips, gen_lpips).mean().item()
        metrics['lpips'] = lpips_score
    except ImportError:
        metrics['lpips'] = None
    
    return metrics


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, window_size: int = 11) -> float:
    """Calculate SSIM between two images"""
    from skimage.metrics import structural_similarity
    
    # Ensure images are in the right format
    if img1.ndim == 4:  # Batch dimension
        ssim_scores = []
        for i in range(img1.shape[0]):
            score = structural_similarity(
                img1[i, 0], img2[i, 0], 
                win_size=window_size,
                data_range=1.0
            )
            ssim_scores.append(score)
        return np.mean(ssim_scores)
    else:
        return structural_similarity(
            img1, img2, 
            win_size=window_size,
            data_range=1.0
        )


def calculate_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: torch.device
) -> float:
    """Calculate FID score"""
    try:
        from pytorch_fid import fid_score
        
        # Save images temporarily for FID calculation
        temp_dir = Path("temp_fid")
        temp_dir.mkdir(exist_ok=True)
        
        real_dir = temp_dir / "real"
        gen_dir = temp_dir / "generated"
        real_dir.mkdir(exist_ok=True)
        gen_dir.mkdir(exist_ok=True)
        
        # Save real images
        for i in range(real_images.shape[0]):
            img = real_images[i, 0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            plt.imsave(real_dir / f"real_{i:04d}.png", img, cmap='gray')
        
        # Save generated images
        for i in range(generated_images.shape[0]):
            img = generated_images[i, 0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            plt.imsave(gen_dir / f"gen_{i:04d}.png", img, cmap='gray')
        
        # Calculate FID
        fid = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(gen_dir)],
            batch_size=50,
            device=device
        )
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return fid
        
    except ImportError:
        print("pytorch-fid not available, skipping FID calculation")
        return None


def generate_samples(
    model: ConditionalVAE,
    test_loader: DataLoader,
    num_samples: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Generate samples from the model"""
    model.eval()
    
    real_images = []
    generated_images = []
    conditions_list = []
    
    with torch.no_grad():
        for batch_idx, (images, conditions) in enumerate(test_loader):
            if len(real_images) >= num_samples:
                break
            
            # Move to device
            images = images.to(device)
            conditions = {k: v.to(device) for k, v in conditions.items()}
            
            # Generate samples
            x_gen = model.sample(conditions, images.shape[0])
            
            # Store results
            real_images.append(images)
            generated_images.append(x_gen)
            conditions_list.append(conditions)
    
    # Concatenate all batches
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    return {
        'real_images': real_images,
        'generated_images': generated_images,
        'conditions': conditions_list
    }


def create_evaluation_gallery(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    conditions: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    num_samples: int = 16
):
    """Create evaluation gallery"""
    gallery_dir = output_dir / "gallery"
    gallery_dir.mkdir(exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(real_images), min(num_samples, len(real_images)), replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        if i >= 16:
            break
        
        # Original image
        real_img = real_images[idx, 0].cpu().numpy()
        axes[i].imshow(real_img, cmap='gray')
        axes[i].set_title(f"Real {idx}")
        axes[i].axis('off')
        
        # Generated image
        gen_img = generated_images[idx, 0].cpu().numpy()
        axes[i+8].imshow(gen_img, cmap='gray')
        axes[i+8].set_title(f"Generated {idx}")
        axes[i+8].axis('off')
    
    plt.tight_layout()
    plt.savefig(gallery_dir / "comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual images
    for i, idx in enumerate(indices[:8]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Real image
        real_img = real_images[idx, 0].cpu().numpy()
        ax1.imshow(real_img, cmap='gray')
        ax1.set_title(f"Real Image {idx}")
        ax1.axis('off')
        
        # Generated image
        gen_img = generated_images[idx, 0].cpu().numpy()
        ax2.imshow(gen_img, cmap='gray')
        ax2.set_title(f"Generated Image {idx}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(gallery_dir / f"sample_{idx:04d}.png", dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_condition_control(
    model: ConditionalVAE,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path
):
    """Evaluate condition control by generating images with different conditions"""
    model.eval()
    
    # Get a base batch
    images, base_conditions = next(iter(test_loader))
    images = images[:4].to(device)
    base_conditions = {k: v[:4].to(device) for k, v in base_conditions.items()}
    
    # Create different condition variations
    condition_variations = []
    
    # View variations
    for view in [0, 1]:  # CC, MLO
        new_conditions = base_conditions.copy()
        new_conditions['view'] = torch.full_like(new_conditions['view'], view)
        condition_variations.append(('view', view, new_conditions))
    
    # Laterality variations
    for lat in [0, 1]:  # L, R
        new_conditions = base_conditions.copy()
        new_conditions['laterality'] = torch.full_like(new_conditions['laterality'], lat)
        condition_variations.append(('laterality', lat, new_conditions))
    
    # Age variations
    for age in [0, 1, 2, 3]:  # Age bins
        new_conditions = base_conditions.copy()
        new_conditions['age_bin'] = torch.full_like(new_conditions['age_bin'], age)
        condition_variations.append(('age_bin', age, new_conditions))
    
    # Generate samples for each variation
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # First row: original images
    for i in range(4):
        img = images[i, 0].cpu().numpy()
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis('off')
    
    # Generate variations
    for row, (condition_name, condition_value, conditions) in enumerate(condition_variations[:3]):
        with torch.no_grad():
            x_gen = model.sample(conditions, 4)
        
        for col in range(4):
            img = x_gen[col, 0].cpu().numpy()
            axes[row+1, col].imshow(img, cmap='gray')
            axes[row+1, col].set_title(f"{condition_name}={condition_value}")
            axes[row+1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "condition_control.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir / "evaluation.log")
    logger.info(f"Starting evaluation with model: {args.model_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = ConditionalVAE(
        in_channels=config['data']['channels'],
        latent_dim=config['model']['latent_dim'],
        condition_embed_dim=config['model']['condition_embed_dim'],
        encoder_channels=config['model']['encoder']['channels'],
        decoder_channels=config['model']['decoder']['channels'],
        use_skip_connections=config['model']['decoder']['use_skip_connections'],
        use_group_norm=config['model']['encoder']['use_group_norm'],
        groups=config['model']['encoder']['groups'],
        activation=config['model']['encoder']['activation']
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, model)
    model = model.to(device)
    
    # Create test dataset
    test_dataset = MammographyDataset(
        csv_path=config['data']['csv_path'],
        image_dir=config['data']['image_dir'],
        resolution=config['data']['resolution'],
        patch_stride=config['data']['patch_stride'],
        min_foreground_frac=config['data']['min_foreground_frac'],
        transform_mode='test',
        use_patches=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Generate samples
    logger.info("Generating samples for evaluation...")
    samples = generate_samples(model, test_loader, args.num_samples, device)
    
    real_images = samples['real_images']
    generated_images = samples['generated_images']
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(real_images, generated_images, device)
    
    # Calculate FID
    logger.info("Calculating FID score...")
    fid_score = calculate_fid(real_images, generated_images, device)
    if fid_score is not None:
        metrics['fid'] = fid_score
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        if value is not None:
            logger.info(f"  {metric}: {value:.4f}")
    
    # Create evaluation gallery
    logger.info("Creating evaluation gallery...")
    create_evaluation_gallery(
        real_images, generated_images, samples['conditions'], output_dir
    )
    
    # Evaluate condition control
    logger.info("Evaluating condition control...")
    evaluate_condition_control(model, test_loader, device, output_dir)
    
    # Save sample images
    logger.info("Saving sample images...")
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    for i in range(min(100, len(real_images))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Real image
        real_img = real_images[i, 0].cpu().numpy()
        ax1.imshow(real_img, cmap='gray')
        ax1.set_title("Real Image")
        ax1.axis('off')
        
        # Generated image
        gen_img = generated_images[i, 0].cpu().numpy()
        ax2.imshow(gen_img, cmap='gray')
        ax2.set_title("Generated Image")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(samples_dir / f"sample_{i:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
