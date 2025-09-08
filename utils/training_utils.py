import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
import numpy as np


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    # Create logger
    logger = logging.getLogger("nlbs-cvae")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
    **kwargs
):
    """Save training checkpoint"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """Load training checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}")
    
    return checkpoint


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999
) -> optim.Optimizer:
    """Create optimizer for training"""
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2)
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    num_epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6
) -> Optional[Any]:
    """Create learning rate scheduler"""
    if scheduler_name.lower() == "none":
        return None
    elif scheduler_name.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr
        )
    elif scheduler_name.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    elif scheduler_name.lower() == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    elif scheduler_name.lower() == "cosine_with_warmup":
        # Custom cosine scheduler with warmup
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=num_epochs,
            cycle_mult=1.0,
            max_lr=optimizer.param_groups[0]['lr'],
            min_lr=min_lr,
            warmup_steps=warmup_epochs,
            gamma=1.0
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


class CosineAnnealingWarmupRestarts:
    """
    Custom cosine annealing scheduler with warmup
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 10.0,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1
    ):
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step()
        
        self.last_epoch = last_epoch
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = epoch
        
        if epoch < self.warmup_steps:
            lr = self.max_lr * (epoch / self.warmup_steps)
        else:
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1.0 + np.cos(np.pi * (epoch - self.warmup_steps) / self.cur_cycle_steps)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        if epoch == self.cur_cycle_steps - 1:
            self.cycle += 1
            self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
            self.step()


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def setup_mixed_precision(
    use_mixed_precision: bool = True,
    dtype: str = "fp16"
) -> tuple:
    """Setup mixed precision training"""
    if not use_mixed_precision:
        return None, None
    
    if dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    elif dtype == "bf16":
        scaler = None
        autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown mixed precision dtype: {dtype}")
    
    return scaler, autocast


def log_model_info(model: torch.nn.Module, logger: logging.Logger):
    """Log model information"""
    total_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Model size: {model_size:.2f} MB")
    
    # Log layer information
    logger.info("Model architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            if hasattr(module, 'weight'):
                weight_shape = module.weight.shape if module.weight is not None else "None"
                logger.info(f"  {name}: {module.__class__.__name__} - Weight: {weight_shape}")
            else:
                logger.info(f"  {name}: {module.__class__.__name__}")


def create_experiment_dir(
    base_dir: Path,
    experiment_name: str,
    create_subdirs: bool = True
) -> Path:
    """Create experiment directory structure"""
    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    if create_subdirs:
        (experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        (experiment_dir / "samples").mkdir(exist_ok=True)
        (experiment_dir / "metrics").mkdir(exist_ok=True)
    
    return experiment_dir
