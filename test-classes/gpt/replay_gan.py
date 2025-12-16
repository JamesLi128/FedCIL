
# ==============================
# --- file: fcil/replay_gan.py
# ==============================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fed_base import ReplayStrategy


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
            nn.Linear(z_dim, base * 4 * 4 * 4),
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
        zy = z + self.embed(y)
        h = self.net(zy)
        h = h.view(h.size(0), -1, 4, 4)
        return self.deconv(h)


class CondDiscriminator(nn.Module):
    """
    Tiny conditional discriminator using label embedding concatenated as a bias.
    """
    def __init__(self, num_classes: int, img_channels: int = 3, base: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, base)

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, base, 4, 2, 1),      # 16x16
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1),          # 8x8
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),      # 4x4
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Linear(base * 4 * 4 * 4, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        # simple conditioning: add embedded label bias
        h = h + self.embed(y).repeat_interleave(h.size(1) // self.embed.embedding_dim, dim=1)
        return self.head(h).squeeze(1)


@dataclass
class GANReplayConfig:
    z_dim: int = 128
    gan_lr: float = 2e-4
    gan_steps_per_batch: int = 1
    img_channels: int = 3
    num_total_classes: int = 100  # global label space size (e.g., CIFAR-100)


class ClientGANReplay(ReplayStrategy):
    """
    Client-side replay:
      - Maintains a conditional GAN
      - Can sample labeled synthetic data for replay
      - Can be FedAvg-aggregated via state_dict
    """
    def __init__(self, cfg: GANReplayConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.G = CondGenerator(cfg.z_dim, cfg.num_total_classes, cfg.img_channels).to(device)
        self.D = CondDiscriminator(cfg.num_total_classes, cfg.img_channels).to(device)

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
        return x, ys

    def train_gan_on_batch(self, x_real: torch.Tensor, y_real: torch.Tensor) -> Dict[str, float]:
        """
        One GAN update (or a few) for a single real batch.
        """
        self.G.train()
        self.D.train()

        bsz = x_real.size(0)
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

