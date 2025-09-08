import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VAELoss(nn.Module):
    """
    Combined VAE loss function:
    L = L_rec + β * KL + λ_edge * L_edge
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1.0,
        edge_weight: float = 0.1,
        kl_anneal_epochs: int = 20,
        current_epoch: int = 0
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.edge_weight = edge_weight
        self.kl_anneal_epochs = kl_anneal_epochs
        self.current_epoch = current_epoch
        
        # Edge detection kernel (Laplacian)
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
    
    def update_epoch(self, epoch: int):
        """Update current epoch for KL annealing"""
        self.current_epoch = epoch
    
    def get_kl_weight(self) -> float:
        """Get current KL weight with annealing"""
        if self.current_epoch < self.kl_anneal_epochs:
            # Linear annealing from 0 to kl_weight
            anneal_factor = self.current_epoch / self.kl_anneal_epochs
            return self.kl_weight * anneal_factor
        else:
            return self.kl_weight
    
    def reconstruction_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        L1 reconstruction loss
        
        Args:
            x_recon: Reconstructed images [B, 1, H, W]
            x_target: Target images [B, 1, H, W]
        
        Returns:
            Reconstruction loss
        """
        return F.l1_loss(x_recon, x_target, reduction='mean')
    
    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between posterior and prior
        
        Args:
            mu: Posterior mean [B, latent_dim]
            logvar: Posterior log variance [B, latent_dim]
        
        Returns:
            KL divergence loss
        """
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - var)
        var = torch.exp(logvar)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var, dim=1)
        return kl_loss.mean()
    
    def edge_preservation_loss(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Edge preservation loss using Laplacian operator
        
        Args:
            x_recon: Reconstructed images [B, 1, H, W]
            x_target: Target images [B, 1, H, W]
        
        Returns:
            Edge preservation loss
        """
        # Apply Laplacian operator to detect edges
        edges_recon = F.conv2d(x_recon, self.laplacian_kernel, padding=1)
        edges_target = F.conv2d(x_target, self.laplacian_kernel, padding=1)
        
        # L1 loss on edges
        edge_loss = F.l1_loss(edges_recon, edges_target, reduction='mean')
        
        return edge_loss
    
    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total VAE loss
        
        Args:
            x_recon: Reconstructed images [B, 1, H, W]
            x_target: Target images [B, 1, H, W]
            mu: Posterior mean [B, latent_dim]
            logvar: Posterior log variance [B, latent_dim]
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Individual losses
        rec_loss = self.reconstruction_loss(x_recon, x_target)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        edge_loss = self.edge_preservation_loss(x_recon, x_target)
        
        # Get current KL weight with annealing
        current_kl_weight = self.get_kl_weight()
        
        # Combined loss
        total_loss = (
            self.reconstruction_weight * rec_loss +
            current_kl_weight * kl_loss +
            self.edge_weight * edge_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': rec_loss.item(),
            'kl_loss': kl_loss.item(),
            'edge_loss': edge_loss.item(),
            'kl_weight': current_kl_weight,
            'reconstruction_weight': self.reconstruction_weight,
            'edge_weight': self.edge_weight
        }
        
        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features (optional for phase 2)
    """
    
    def __init__(self, feature_layers: list = [2, 7, 12, 21, 30]):
        super().__init__()
        
        # Load pretrained VGG16 and freeze
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        vgg.eval()
        
        # Extract feature layers
        self.feature_extractor = nn.ModuleList()
        features = list(vgg.features)
        
        for i in range(max(feature_layers) + 1):
            self.feature_extractor.append(features[i])
            if i in feature_layers:
                break
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            x_recon: Reconstructed images [B, 1, H, W]
            x_target: Target images [B, 1, H, W]
        
        Returns:
            Perceptual loss
        """
        # Convert grayscale to RGB (repeat single channel)
        if x_recon.shape[1] == 1:
            x_recon = x_recon.repeat(1, 3, 1, 1)
            x_target = x_target.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_recon.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_recon.device)
        
        x_recon = (x_recon - mean) / std
        x_target = (x_target - mean) / std
        
        # Extract features
        features_recon = []
        features_target = []
        
        x_recon_feat = x_recon
        x_target_feat = x_target
        
        for i, layer in enumerate(self.feature_extractor):
            x_recon_feat = layer(x_recon_feat)
            x_target_feat = layer(x_target_feat)
            
            if i in [2, 7, 12, 21, 30]:  # Feature layers
                features_recon.append(x_recon_feat)
                features_target.append(x_target_feat)
        
        # Compute L2 loss on features
        perceptual_loss = 0
        for feat_recon, feat_target in zip(features_recon, features_target):
            perceptual_loss += F.mse_loss(feat_recon, feat_target, reduction='mean')
        
        return perceptual_loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for VAE-GAN (optional for phase 2)
    """
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
    
    def forward(self, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial loss
        
        Args:
            x_recon: Reconstructed images [B, 1, H, W]
        
        Returns:
            Adversarial loss
        """
        # Get discriminator predictions
        fake_pred = self.discriminator(x_recon)
        
        # Adversarial loss: maximize log(D(G(x)))
        adversarial_loss = -torch.mean(torch.log(fake_pred + 1e-8))
        
        return adversarial_loss