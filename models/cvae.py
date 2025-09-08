import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .layers import ConvBlock, DeconvBlock, FiLMLayer, ConditionEmbedder


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for FFDM mammographic image generation

    - Encoder: strided ConvBlocks to downsample spatially
    - Latent: Gaussian with mean/logvar produced from flattened encoder features
    - Decoder: Linear projection -> reshaped feature map -> ConvTranspose2d blocks
    - Conditioning: FiLM modulation in each decoder block, and optional posterior bias

    Notes
    - Images are assumed normalized to [-1, 1]. Output activation is Tanh.
    - image_size should be divisible by 2**len(encoder_channels).
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 256,
        latent_dim: int = 256,
        condition_embed_dim: int = 128,
        encoder_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: List[int] = [512, 256, 128, 64, 1],
        use_skip_connections: bool = True,
        use_group_norm: bool = True,
        groups: int = 8,
        activation: str = "silu",
        posterior_bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.condition_embed_dim = condition_embed_dim
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.use_skip_connections = use_skip_connections
        self.use_group_norm = use_group_norm
        self.groups = groups
        self.activation = activation
        self.posterior_bias_flag = posterior_bias

        # Validate spatial compatibility
        self.num_down = len(self.encoder_channels)
        divisor = 2 ** self.num_down
        if image_size % divisor != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by 2**len(encoder_channels) ({divisor})"
            )
        self.initial_size = image_size // divisor  # H0 = W0

        # Condition embedder
        self.condition_embedder = ConditionEmbedder(
            condition_dim=14,  # view(2) + laterality(2) + age_embed(8) + cancer(1) + false_pos(1)
            embed_dim=condition_embed_dim,
        )
        if self.posterior_bias_flag:
            # Project condition embedding to 2*latent_dim to bias (mu, logvar)
            self.posterior_bias = nn.Sequential(
                nn.Linear(condition_embed_dim, condition_embed_dim),
                nn.SiLU(),
                nn.Linear(condition_embed_dim, latent_dim * 2),
            )

        # Encoder
        enc_layers = []
        in_ch = self.in_channels
        for out_ch in self.encoder_channels:
            enc_layers.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.groups,
                    use_group_norm=self.use_group_norm,
                    activation=self.activation,
                )
            )
            in_ch = out_ch
        self.encoder = nn.ModuleList(enc_layers)
        self.enc_out_ch = self.encoder_channels[-1]

        # Latent heads
        flat_dim = self.enc_out_ch * self.initial_size * self.initial_size
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder
        self.fc_z = nn.Sequential(
            nn.Linear(latent_dim, self.decoder_channels[0] * self.initial_size * self.initial_size),
            nn.SiLU(),
        )

        dec_blocks = []
        for i in range(len(self.decoder_channels) - 1):
            in_ch = self.decoder_channels[i]
            out_ch = self.decoder_channels[i + 1]
            dec_blocks.append(
                nn.ModuleDict(
                    {
                        "deconv": DeconvBlock(
                            in_ch,
                            out_ch,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            groups=self.groups,
                            use_group_norm=self.use_group_norm,
                            activation=self.activation if i < len(self.decoder_channels) - 2 else "silu",
                        ),
                        "film": FiLMLayer(self.condition_embed_dim, out_ch),
                    }
                )
            )
        self.decoder = nn.ModuleList(dec_blocks)
        self.out_act = nn.Tanh()

        # Container for skip features collected in encode
        self.skip_features: List[torch.Tensor] = []

    # ---------------------
    # Core VAE components
    # ---------------------
    def encode(self, x: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input images to latent distribution parameters (mu, logvar)."""
        _ = self.condition_embedder(conditions)  # computed for consistency; not used directly here

        self.skip_features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if self.use_skip_connections and i < len(self.encoder) - 1:
                self.skip_features.append(x)

        h = self.flatten(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.posterior_bias_flag:
            cond = self.condition_embedder(conditions)
            bias = self.posterior_bias(cond)
            b_mu, b_logvar = torch.chunk(bias, 2, dim=1)
            mu = mu + b_mu
            logvar = logvar + b_logvar

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample z ~ N(mu, sigma^2)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode latent vectors into images."""
        cond = self.condition_embedder(conditions)

        B = z.shape[0]
        x = self.fc_z(z)
        x = x.view(B, self.decoder_channels[0], self.initial_size, self.initial_size)

        # Iterate through decoder blocks
        rev_skips = list(reversed(self.skip_features)) if self.use_skip_connections else []
        for i, block in enumerate(self.decoder):
            x = block["deconv"](x)

            # Add skip features (resized) after upsampling, before FiLM
            if self.use_skip_connections and i < len(rev_skips):
                skip = rev_skips[i]
                if skip is not None:
                    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
                    # Project if channel mismatch
                    if skip.shape[1] != x.shape[1]:
                        # simple 1x1 projection on-the-fly
                        proj = skip.mean(dim=1, keepdim=True)
                        if proj.shape[1] != x.shape[1]:
                            proj = proj.repeat(1, x.shape[1], 1, 1)
                        skip = proj
                    x = x + skip

            x = block["film"](x, cond)

        x = self.out_act(x)
        return x

    def forward(self, x: torch.Tensor, conditions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, conditions)
        return x_recon, mu, logvar

    # ---------------------
    # Sampling helpers
    # ---------------------
    def sample(self, conditions: Dict[str, torch.Tensor], num_samples: int = 1) -> torch.Tensor:
        """Generate samples from the prior given conditions."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            z = torch.randn(num_samples, self.latent_dim, device=device)
            x = self.decode(z, conditions)
        return x

    def get_condition_embedding(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.condition_embedder(conditions)
