# ==============================
# --- file: fcil/replay_gan.py
# ==============================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from fed_base import ReplayStrategy
from config import GANReplayConfig


def compute_deconv_layers(target_size: int, start_size: int = 4) -> List[Tuple[int, int, int]]:
    """
    Compute the number of transposed conv layers needed to reach target_size.
    
    Returns list of (kernel_size, stride, padding) for each layer.
    For standard DCGAN-style upsampling: kernel=4, stride=2, padding=1 doubles size.
    """
    layers = []
    current_size = start_size
    while current_size < target_size:
        layers.append((4, 2, 1))  # Standard DCGAN upsampling
        current_size *= 2
    return layers


class CondGenerator(nn.Module):
    """
    Conditional DCGAN-style generator supporting multiple output sizes.
    Supports 28x28 (MNIST), 32x32 (CIFAR), 64x64, etc.
    """
    def __init__(
        self, 
        z_dim: int, 
        num_classes: int, 
        img_channels: int = 3, 
        img_size: int = 32,
        base: int = 64
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        self.embed = nn.Embedding(num_classes, z_dim)
        
        # Determine architecture based on target image size
        # We start from 4x4 and upsample
        self.start_size = 4
        num_upsamples = int(math.log2(img_size / self.start_size))
        
        # Calculate channel progression (decreasing as we upsample)
        # More upsamples = need more initial channels
        self.init_channels = base * (2 ** num_upsamples)
        
        # Initial projection: z + label_embed -> spatial features
        self.net = nn.Sequential(
            nn.Linear(z_dim + z_dim, self.init_channels * self.start_size * self.start_size),
            nn.ReLU(True),
        )
        
        # Build upsampling layers dynamically
        deconv_layers = []
        in_ch = self.init_channels
        
        for i in range(num_upsamples):
            out_ch = in_ch // 2 if i < num_upsamples - 1 else base
            deconv_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ])
            in_ch = out_ch
        
        # Final conv to get correct number of channels
        deconv_layers.extend([
            nn.Conv2d(base, img_channels, 3, 1, 1),
            nn.Tanh(),
        ])
        
        self.deconv = nn.Sequential(*deconv_layers)
        
        # Handle non-power-of-2 sizes (e.g., 28x28 for MNIST)
        self.final_size = self.start_size * (2 ** num_upsamples)
        self.needs_resize = (self.final_size != img_size)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        zy = torch.cat([z, self.embed(y)], dim=1)
        h = self.net(zy)
        h = h.view(h.size(0), self.init_channels, self.start_size, self.start_size)
        x = self.deconv(h)
        
        # Resize if needed (for non-power-of-2 sizes like 28x28)
        if self.needs_resize:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Map from [-1, 1] to [0, 1]
        x_rep = (x + 1.0) / 2.0
        return x_rep


class CondDiscriminator(nn.Module):
    """
    Conditional discriminator for various image sizes with concatenated label embedding.
    Uses ResNet18 backbone for feature extraction (designed for images resized to 224x224).
    """
    def __init__(
        self, 
        num_classes: int, 
        img_channels: int = 3, 
        embed_dim: int = 32,
        input_size: int = 224
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.img_channels = img_channels
        self.input_size = input_size
        self.embed = nn.Embedding(num_classes, embed_dim)

        # Use pretrained ResNet18 backbone for better feature extraction
        # Note: Input images should already be transformed to 224x224 by the data pipeline
        resnet18 = models.resnet18(pretrained=True)
        
        # If input is not 3-channel RGB, we need to modify the first conv layer
        if img_channels != 3:
            # Replace first conv layer to accept different channel count
            old_conv = resnet18.conv1
            resnet18.conv1 = nn.Conv2d(
                img_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize new conv weights by averaging/replicating old weights
            with torch.no_grad():
                if img_channels == 1:
                    # Average RGB weights for grayscale
                    resnet18.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    # For other channel counts, initialize randomly
                    nn.init.kaiming_normal_(resnet18.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Remove the FC layer and only take features
        self.conv = nn.Sequential(*list(resnet18.children())[:-1])

        # Freeze ResNet18 parameters
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # ResNet18 outputs 512 features before the FC layer
        feature_size = 512
        
        # Concatenate features with label embedding
        self.head = nn.Linear(feature_size + embed_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        
        # Concatenate label embedding to extracted features
        y_embed = self.embed(y)
        h = torch.cat([h, y_embed], dim=1)
        
        return self.head(h).squeeze(1)



class ClientGANReplay(ReplayStrategy):
    """
    Client-side replay:
      - Maintains a conditional GAN
      - Can sample labeled synthetic data for replay
      - Can be FedAvg-aggregated via state_dict
    """
    def __init__(
        self,
        cfg: GANReplayConfig,
        device: torch.device,
        sample_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.cfg = cfg
        self.device = device
        
        # Create generator with appropriate image size and channels
        self.G = CondGenerator(
            z_dim=cfg.z_dim, 
            num_classes=cfg.num_total_classes, 
            img_channels=cfg.img_channels,
            img_size=cfg.generator_img_size,
        ).to(device)
        
        # Discriminator works on transformed images (resized, potentially RGB-converted)
        self.D = CondDiscriminator(
            num_classes=cfg.num_total_classes, 
            img_channels=cfg.discriminator_img_channels,
            input_size=cfg.discriminator_img_size,
        ).to(device)
        
        self._sample_transform = sample_transform

        self._known_classes: List[int] = []
        self._opt_g = torch.optim.Adam(self.G.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))
        self._opt_d = torch.optim.Adam(self.D.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))

    def set_known_classes(self, known: List[int]) -> None:
        self._known_classes = list(known)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        # Only aggregate generator by default (simpler, and common in practice).
        return self.G.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.G.load_state_dict(state, strict=True)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._known_classes:
            # fallback: uniform over whole label space (still deterministic behavior)
            ys = torch.randint(0, self.cfg.num_total_classes, (n,), device=device)
        else:
            idx = torch.randint(0, len(self._known_classes), (n,), device=device)
            ys = torch.tensor(self._known_classes, device=device, dtype=torch.long)[idx]
        z = torch.randn(n, self.cfg.z_dim, device=device)
        x = self.G(z, ys)
        if self._sample_transform is not None:
            x = self._sample_transform(x)
        return x, ys

    def train_gan_on_batch(self, x_real: torch.Tensor, y_real: torch.Tensor) -> Dict[str, float]:
        """
        One GAN update (or a few) for a single real batch.
        
        Note: x_real should already be in the discriminator's expected format (e.g., RGB 224x224).
        The generator produces images in native format (e.g., grayscale 28x28 for MNIST),
        which are then transformed via _sample_transform before passing to the discriminator.
        """
        bsz = x_real.size(0)
        
        # Skip GAN training if batch size is too small for BatchNorm
        # BatchNorm requires at least 2 samples to compute statistics
        if bsz < 2:
            return {"loss_d": 0.0, "loss_g": 0.0}
        
        self.G.train()
        self.D.train()

        z = torch.randn(bsz, self.cfg.z_dim, device=self.device)
        y = y_real

        # --- D step
        for _ in range(self.cfg.gan_steps_per_batch):
            x_fake_raw = self.G(z, y).detach()
            # Transform fake images to match discriminator's expected input format
            if self._sample_transform is not None:
                x_fake = self._sample_transform(x_fake_raw)
            else:
                x_fake = x_fake_raw
            d_real = self.D(x_real, y)
            d_fake = self.D(x_fake, y)
            loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                     F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            self._opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            self._opt_d.step()

        # --- G step
        z2 = torch.randn(bsz, self.cfg.z_dim, device=self.device)
        x_fake2_raw = self.G(z2, y)
        # Transform fake images for discriminator (need gradients to flow through)
        if self._sample_transform is not None:
            x_fake2 = self._sample_transform(x_fake2_raw)
        else:
            x_fake2 = x_fake2_raw
        d_fake2 = self.D(x_fake2, y)
        loss_g = F.binary_cross_entropy_with_logits(d_fake2, torch.ones_like(d_fake2))
        self._opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        self._opt_g.step()

        return {"loss_d": float(loss_d.detach().cpu()), "loss_g": float(loss_g.detach().cpu())}

