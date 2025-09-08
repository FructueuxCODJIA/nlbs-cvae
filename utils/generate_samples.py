#!/usr/bin/env python3
"""
Sample generation script for Conditional VAE
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import ConditionalVAE
from utils.training_utils import setup_logging, load_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate samples from Conditional VAE")
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
        default="results/samples",
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--conditions", 
        type=str, 
        default=None,
        help="Path to conditions file (JSON/YAML) or comma-separated values"
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        default=8,
        help="Grid size for sample visualization"
    )
    return parser.parse_args()


def parse_conditions(conditions_str: str) -> Dict[str, List[int]]:
    """Parse conditions from string or file"""
    if conditions_str.endswith(('.json', '.yaml', '.yml')):
        # Load from file
        with open(conditions_str, 'r') as f:
            if conditions_str.endswith('.json'):
                import json
                conditions = json.load(f)
            else:
                conditions = yaml.safe_load(f)
    else:
        # Parse comma-separated values
        # Format: view,laterality,age_bin,cancer,false_positive
        values = [int(x.strip()) for x in conditions_str.split(',')]
        if len(values) != 5:
            raise ValueError("Expected 5 condition values: view,laterality,age_bin,cancer,false_positive")
        
        conditions = {
            'view': [values[0]],
            'laterality': [values[1]],
            'age_bin': [values[2]],
            'cancer': [values[3]],
            'false_positive': [values[4]]
        }
    
    return conditions


def create_condition_variations(
    base_conditions: Dict[str, List[int]],
    num_samples: int
) -> List[Dict[str, torch.Tensor]]:
    """Create variations of conditions for sampling"""
    conditions_list = []
    
    # Get base values
    view = base_conditions['view'][0]
    laterality = base_conditions['laterality'][0]
    age_bin = base_conditions['age_bin'][0]
    cancer = base_conditions['cancer'][0]
    false_positive = base_conditions['false_positive'][0]
    
    # Create variations
    for i in range(num_samples):
        # Add some randomness to conditions
        if np.random.random() < 0.3:  # 30% chance to vary view
            view_var = 1 - view  # Flip view
        else:
            view_var = view
            
        if np.random.random() < 0.3:  # 30% chance to vary laterality
            laterality_var = 1 - laterality  # Flip laterality
        else:
            laterality_var = laterality
            
        if np.random.random() < 0.2:  # 20% chance to vary age
            age_var = np.random.randint(0, 4)
        else:
            age_var = age_bin
            
        if np.random.random() < 0.1:  # 10% chance to vary cancer
            cancer_var = 1 - cancer
        else:
            cancer_var = cancer
            
        if np.random.random() < 0.1:  # 10% chance to vary false positive
            false_positive_var = 1 - false_positive
        else:
            false_positive_var = false_positive
        
        # Create condition tensor
        condition = {
            'view': torch.tensor([view_var]),
            'laterality': torch.tensor([laterality_var]),
            'age_bin': torch.tensor([age_var]),
            'cancer': torch.tensor([cancer_var]),
            'false_positive': torch.tensor([false_positive_var])
        }
        
        conditions_list.append(condition)
    
    return conditions_list


def generate_samples(
    model: ConditionalVAE,
    conditions: List[Dict[str, torch.Tensor]],
    device: torch.device,
    num_samples: int
) -> torch.Tensor:
    """Generate samples from the model"""
    model.eval()
    
    all_samples = []
    
    with torch.no_grad():
        for i in range(0, num_samples, len(conditions)):
            batch_conditions = conditions[i:i+len(conditions)]
            
            # Stack conditions for batch processing
            batch_size = len(batch_conditions)
            stacked_conditions = {}
            
            for key in batch_conditions[0].keys():
                stacked_conditions[key] = torch.cat([
                    cond[key] for cond in batch_conditions
                ])
            
            # Move to device
            stacked_conditions = {k: v.to(device) for k, v in stacked_conditions.items()}
            
            # Generate samples
            samples = model.sample(stacked_conditions, batch_size)
            all_samples.append(samples)
    
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    
    return all_samples


def save_samples_grid(
    samples: torch.Tensor,
    output_dir: Path,
    grid_size: int = 8,
    filename: str = "samples_grid.png"
):
    """Save samples in a grid format"""
    # Calculate grid dimensions
    n_samples = min(samples.shape[0], grid_size * grid_size)
    grid_h = grid_w = int(np.ceil(np.sqrt(n_samples)))
    
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 2, grid_h * 2))
    
    if grid_h == 1:
        axes = [axes] if grid_w == 1 else axes
    elif grid_w == 1:
        axes = [[ax] for ax in axes]
    
    for i in range(grid_h):
        for j in range(grid_w):
            idx = i * grid_w + j
            if idx < n_samples:
                img = samples[idx, 0].cpu().numpy()
                axes[i][j].imshow(img, cmap='gray')
                axes[i][j].set_title(f"Sample {idx}")
                axes[i][j].axis('off')
            else:
                axes[i][j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_individual_samples(
    samples: torch.Tensor,
    output_dir: Path,
    max_samples: int = 50
):
    """Save individual sample images"""
    samples_dir = output_dir / "individual"
    samples_dir.mkdir(exist_ok=True)
    
    n_samples = min(samples.shape[0], max_samples)
    
    for i in range(n_samples):
        img = samples[i, 0].cpu().numpy()
        
        # Convert to PIL Image and save
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.save(samples_dir / f"sample_{i:04d}.png")
        
        # Also save as numpy array
        np.save(samples_dir / f"sample_{i:04d}.npy", img)


def create_condition_analysis(
    samples: torch.Tensor,
    conditions: List[Dict[str, torch.Tensor]],
    output_dir: Path
):
    """Create analysis of how conditions affect generated samples"""
    # Convert conditions to numpy for analysis
    condition_arrays = {}
    for key in conditions[0].keys():
        condition_arrays[key] = [cond[key].item() for cond in conditions]
    
    # Create condition distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    condition_names = ['View (CC/MLO)', 'Laterality (L/R)', 'Age Bin', 'Cancer', 'False Positive']
    
    for i, (key, name) in enumerate(zip(['view', 'laterality', 'age_bin', 'cancer', 'false_positive'], condition_names)):
        values = condition_arrays[key]
        unique, counts = np.unique(values, return_counts=True)
        
        axes[i].bar(unique, counts)
        axes[i].set_title(name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    
    # Remove extra subplot
    axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / "condition_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save condition statistics
    stats = {}
    for key in condition_arrays.keys():
        values = condition_arrays[key]
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'unique_values': list(np.unique(values))
        }
    
    import json
    with open(output_dir / "condition_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)


def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir / "generation.log")
    logger.info(f"Starting sample generation with model: {args.model_path}")
    
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
    
    # Parse conditions
    if args.conditions:
        base_conditions = parse_conditions(args.conditions)
        logger.info(f"Using conditions: {base_conditions}")
    else:
        # Use default conditions
        base_conditions = {
            'view': [0],  # CC
            'laterality': [0],  # L
            'age_bin': [1],  # Age bin 1
            'cancer': [0],  # No cancer
            'false_positive': [0]  # No false positive
        }
        logger.info(f"Using default conditions: {base_conditions}")
    
    # Create condition variations
    conditions = create_condition_variations(base_conditions, args.num_samples)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    samples = generate_samples(model, conditions, device, args.num_samples)
    
    logger.info(f"Generated {samples.shape[0]} samples with shape {samples.shape[1:]}")
    
    # Save samples
    logger.info("Saving samples...")
    
    # Save grid
    save_samples_grid(samples, output_dir, args.grid_size)
    
    # Save individual samples
    save_individual_samples(samples, output_dir)
    
    # Create condition analysis
    create_condition_analysis(samples, conditions, output_dir)
    
    # Save sample metadata
    import json
    import pandas as pd
    metadata = {
        'num_samples': samples.shape[0],
        'sample_shape': list(samples.shape[1:]),
        'base_conditions': base_conditions,
        'model_path': args.model_path,
        'generation_date': str(pd.Timestamp.now())
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Sample generation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
