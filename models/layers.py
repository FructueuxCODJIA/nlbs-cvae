import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and SiLU activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 8,
        use_group_norm: bool = True,
        activation: str = "silu"
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=False
        )
        
        if use_group_norm and out_channels >= groups:
            self.norm = nn.GroupNorm(groups, out_channels)
        else:
            self.norm = nn.BatchNorm2d(out_channels)
        
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DeconvBlock(nn.Module):
    """Deconvolutional block with GroupNorm and SiLU activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        groups: int = 8,
        use_group_norm: bool = True,
        activation: str = "silu"
    ):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, 
            output_padding=output_padding, bias=False
        )
        
        if use_group_norm and out_channels >= groups:
            self.norm = nn.GroupNorm(groups, out_channels)
        else:
            self.norm = nn.BatchNorm2d(out_channels)
        
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer for conditioning"""
    
    def __init__(self, condition_dim: int, feature_dim: int):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        
        # Project condition to gamma and beta parameters
        self.film_proj = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            condition: Condition vector [B, condition_dim]
        
        Returns:
            Modulated features [B, C, H, W]
        """
        batch_size = x.shape[0]
        
        # Project condition to film parameters
        film_params = self.film_proj(condition)  # [B, feature_dim * 2]
        gamma, beta = torch.chunk(film_params, 2, dim=1)  # [B, feature_dim] each
        
        # Reshape for broadcasting
        gamma = gamma.view(batch_size, -1, 1, 1)  # [B, C, 1, 1]
        beta = beta.view(batch_size, -1, 1, 1)   # [B, C, 1, 1]
        
        # Apply FiLM modulation
        x = gamma * x + beta
        
        return x


class ConditionEmbedder(nn.Module):
    """Embedding layer for condition vectors"""
    
    def __init__(
        self,
        condition_dim: int,
        embed_dim: int,
        age_bins: int = 4,
        age_embed_dim: int = 8
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.embed_dim = embed_dim
        self.age_bins = age_bins
        self.age_embed_dim = age_embed_dim
        
        # Age embedding
        self.age_embedding = nn.Embedding(age_bins, age_embed_dim)
        
        # Project final condition vector to embedding space
        final_condition_dim = 2 + 2 + age_embed_dim + 1 + 1  # view + laterality + age_embed + cancer + false_pos
        self.condition_proj = nn.Sequential(
            nn.Linear(final_condition_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, conditions: dict) -> torch.Tensor:
        """
        Args:
            conditions: Dictionary with condition keys:
                - view: [B] (0=CC, 1=MLO)
                - laterality: [B] (0=L, 1=R)
                - age_bin: [B] (0-3)
                - cancer: [B] (0/1)
                - false_positive: [B] (0/1)
        
        Returns:
            Condition embedding [B, embed_dim]
        """
        batch_size = conditions['view'].shape[0]
        
        # One-hot encode view and laterality
        view_onehot = F.one_hot(conditions['view'], num_classes=2).float()  # [B, 2]
        laterality_onehot = F.one_hot(conditions['laterality'], num_classes=2).float()  # [B, 2]
        
        # Age embedding
        age_embed = self.age_embedding(conditions['age_bin'])  # [B, age_embed_dim]
        
        # Cancer and false positive (already 0/1)
        cancer = conditions['cancer'].float().unsqueeze(1)  # [B, 1]
        false_positive = conditions['false_positive'].float().unsqueeze(1)  # [B, 1]
        
        # Concatenate all conditions
        condition_vector = torch.cat([
            view_onehot,           # [B, 2]
            laterality_onehot,     # [B, 2]
            age_embed,             # [B, age_embed_dim]
            cancer,                # [B, 1]
            false_positive         # [B, 1]
        ], dim=1)  # [B, 2+2+age_embed_dim+1+1]
        
        # Project to final embedding
        condition_embed = self.condition_proj(condition_vector)  # [B, embed_dim]
        
        return condition_embed


class ResidualBlock(nn.Module):
    """Residual block with FiLM conditioning"""
    
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        kernel_size: int = 3,
        groups: int = 8
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(channels, channels, kernel_size, groups=groups)
        self.conv2 = ConvBlock(channels, channels, kernel_size, groups=groups)
        self.film = FiLMLayer(condition_dim, channels)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.film(x, condition)
        x = self.conv2(x)
        x = self.film(x, condition)
        
        return x + residual
