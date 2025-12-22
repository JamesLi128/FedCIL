# ==============================
# --- file: fcil/replay_gan.py
# ==============================
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from fed_base import ReplayStrategy
from config import GANReplayConfig


class CondGenerator(nn.Module):
    """
    Tiny conditional DCGAN-ish generator for 32x32 images.
    """
    def __init__(self, z_dim: int, num_classes: int, img_channels: int = 3, base: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, z_dim)

        self.net = nn.Sequential(
            nn.Linear(z_dim + z_dim, base * 4 * 4 * 4),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),      # 16x16
            nn.BatchNorm2d(base),
            nn.ReLU(True),
            nn.ConvTranspose2d(base, img_channels, 4, 2, 1),  # 32x32
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        zy = torch.cat([z, self.embed(y)], dim=1)
        h = self.net(zy)
        h = h.view(h.size(0), -1, 4, 4)
        # entries in [-1, 1], map to [0, 1] 
        x_rep = (self.deconv(h) + 1.0) / 2.0
        return x_rep


class CondDiscriminator(nn.Module):
    """
    Conditional discriminator for 224x224 images with concatenated label embedding.
    Designed to handle images transformed for ResNet18 input size.
    """
    def __init__(self, num_classes: int, img_channels: int = 3, base: int = 64, embed_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(num_classes, embed_dim)

        # Feature extraction for 224x224 images
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        # self.conv = nn.Sequential(
        #     nn.Conv2d(img_channels, base, 4, 2, 1),        # 224 -> 112
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(base, base * 2, 4, 2, 1),            # 112 -> 56
        #     nn.BatchNorm2d(base * 2),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(base * 2, base * 4, 4, 2, 1),        # 56 -> 28
        #     nn.BatchNorm2d(base * 4),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(base * 4, base * 8, 4, 2, 1),        # 28 -> 14
        #     nn.BatchNorm2d(base * 8),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(base * 8, base * 8, 4, 2, 1),        # 14 -> 7
        #     nn.BatchNorm2d(base * 8),
        #     nn.LeakyReLU(0.2, True),
        # )
        
        # # Adaptive pooling to ensure consistent spatial size
        # self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # # Feature size after pooling: base * 8 * 4 * 4 = 512 * 16 = 8192
        # feature_size = base * 8 * 4 * 4

        # Use pretrained ResNet18 backbone for better feature extraction
        resnet18 = models.resnet18(pretrained=True)
        # Remove the FC layer and only take features
        self.conv = nn.Sequential(*list(resnet18.children())[:-1])  # Remove FC layer

        # freeze ResNet18 parameters
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # ResNet18 outputs 512 features before the FC layer
        feature_size = 512
        
        # Concatenate features with label embedding
        self.head = nn.Linear(feature_size + embed_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        # h = self.pool(h)
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
        self.G = CondGenerator(cfg.z_dim, cfg.num_total_classes, cfg.img_channels).to(device)
        self.D = CondDiscriminator(cfg.num_total_classes, cfg.img_channels).to(device)
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
            x_fake = self.G(z, y).detach()
            d_real = self.D(x_real, y)
            d_fake = self.D(x_fake, y)
            loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                     F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            self._opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            self._opt_d.step()

        # --- G step
        z2 = torch.randn(bsz, self.cfg.z_dim, device=self.device)
        x_fake2 = self.G(z2, y)
        d_fake2 = self.D(x_fake2, y)
        loss_g = F.binary_cross_entropy_with_logits(d_fake2, torch.ones_like(d_fake2))
        self._opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        self._opt_g.step()

        return {"loss_d": float(loss_d.detach().cpu()), "loss_g": float(loss_g.detach().cpu())}

