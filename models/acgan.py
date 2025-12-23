# ==================================
# --- file: fcil/acgan.py
# ==================================
"""
ACGAN (Auxiliary Classifier GAN) module for Federated Class-Incremental Learning.

This implementation:
- Supports multiple image sizes (28x28 MNIST, 32x32 CIFAR, etc.)
- Has an expandable auxiliary classifier for class-incremental learning
- Provides both image generation and classification in a single model
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACGANGenerator(nn.Module):
    """
    ACGAN Generator supporting multiple output sizes.
    Supports 28x28 (MNIST), 32x32 (CIFAR), 64x64, etc.
    
    Uses conditional generation with one-hot label embedding concatenated to noise.
    """
    def __init__(
        self,
        z_dim: int,
        num_classes: int,
        img_channels: int = 3,
        img_size: int = 32,
        base_channels: int = 64
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        # Input: z_dim + num_classes (one-hot label)
        self.input_dim = z_dim + num_classes
        
        # Determine architecture based on target size
        # Start from 4x4 and upsample
        self.start_size = 4
        num_upsamples = int(math.log2(img_size / self.start_size))
        if 2 ** num_upsamples * self.start_size != img_size:
            # Handle non-power-of-2 sizes (e.g., 28)
            num_upsamples = int(math.ceil(math.log2(img_size / self.start_size)))
        
        self.num_upsamples = num_upsamples
        self.init_channels = base_channels * (2 ** num_upsamples)
        
        # Project and reshape
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.init_channels * self.start_size * self.start_size),
            nn.BatchNorm1d(self.init_channels * self.start_size * self.start_size),
            nn.ReLU(True),
        )
        
        # Build upsampling layers
        layers = []
        in_ch = self.init_channels
        
        for i in range(num_upsamples):
            out_ch = in_ch // 2 if i < num_upsamples - 1 else base_channels
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ])
            in_ch = out_ch
        
        # Final conv to get correct channels
        layers.extend([
            nn.Conv2d(base_channels, img_channels, 3, 1, 1),
            nn.Tanh(),
        ])
        
        self.deconv = nn.Sequential(*layers)
        
        # Track actual output size for potential resizing
        self.actual_output_size = self.start_size * (2 ** num_upsamples)
        self.needs_resize = (self.actual_output_size != img_size)
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noise tensor of shape (batch, z_dim)
            y: Class labels of shape (batch,) - integer labels
        
        Returns:
            Generated images in range [0, 1]
        """
        batch_size = z.size(0)
        device = z.device
        
        # Create one-hot encoding
        y_onehot = torch.zeros(batch_size, self.num_classes, device=device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        
        # Concatenate noise and label
        z_y = torch.cat([z, y_onehot], dim=1)
        
        # Project and reshape
        h = self.fc(z_y)
        h = h.view(batch_size, self.init_channels, self.start_size, self.start_size)
        
        # Upsample
        x = self.deconv(h)
        
        # Resize if needed
        if self.needs_resize:
            x = F.interpolate(x, size=(self.img_size, self.img_size), 
                            mode='bilinear', align_corners=False)
        
        # Map from [-1, 1] to [0, 1]
        return (x + 1.0) / 2.0
    
    def expand_classes(self, num_new_classes: int) -> None:
        """Expand the generator to handle more classes."""
        if num_new_classes <= 0:
            return
        
        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + num_new_classes
        new_input_dim = self.z_dim + new_num_classes
        
        # Expand the first linear layer
        old_fc = self.fc[0]
        new_fc = nn.Linear(new_input_dim, old_fc.out_features)
        new_fc = new_fc.to(old_fc.weight.device)
        
        with torch.no_grad():
            # Copy weights for z_dim portion
            new_fc.weight[:, :self.z_dim].copy_(old_fc.weight[:, :self.z_dim])
            # Copy weights for existing classes
            new_fc.weight[:, self.z_dim:self.z_dim + old_num_classes].copy_(
                old_fc.weight[:, self.z_dim:]
            )
            # Initialize new class weights (small random)
            nn.init.normal_(new_fc.weight[:, self.z_dim + old_num_classes:], 0, 0.02)
            new_fc.bias.copy_(old_fc.bias)
        
        self.fc[0] = new_fc
        self.num_classes = new_num_classes
        self.input_dim = new_input_dim


class ACGANDiscriminator(nn.Module):
    """
    ACGAN Discriminator with expandable auxiliary classifier.
    
    Outputs:
        - real/fake logit
        - class logits (auxiliary classifier)
    """
    def __init__(
        self,
        num_classes: int,
        img_channels: int = 3,
        img_size: int = 32,
        base_channels: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.base_channels = base_channels
        
        # Calculate number of downsampling layers needed
        # We downsample until we reach 4x4
        num_downsamples = int(math.log2(img_size / 4))
        if num_downsamples < 1:
            num_downsamples = 1
        
        # Build conv layers
        layers = []
        in_ch = img_channels
        out_ch = base_channels
        
        # First layer (no batchnorm)
        layers.extend([
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ])
        in_ch = out_ch
        
        # Subsequent layers
        for i in range(num_downsamples - 1):
            out_ch = min(in_ch * 2, base_channels * 8)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ])
            in_ch = out_ch
        
        self.conv = nn.Sequential(*layers)
        
        # Calculate feature size after conv
        # After num_downsamples, size is img_size / 2^num_downsamples
        final_size = max(img_size // (2 ** num_downsamples), 1)
        self.feature_size = in_ch * final_size * final_size
        
        # Discriminator head (real/fake)
        self.fc_dis = nn.Linear(self.feature_size, 1)
        
        # Auxiliary classifier head (class prediction)
        self.fc_aux = nn.Linear(self.feature_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images of shape (batch, channels, height, width)
        
        Returns:
            dis_out: Real/fake logits of shape (batch,)
            aux_out: Class logits of shape (batch, num_classes)
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        
        dis_out = self.fc_dis(h).squeeze(1)
        aux_out = self.fc_aux(h)
        
        return dis_out, aux_out
    
    def expand_classes(self, num_new_classes: int) -> None:
        """Expand the auxiliary classifier to handle more classes."""
        if num_new_classes <= 0:
            return
        
        old_fc = self.fc_aux
        new_num_classes = self.num_classes + num_new_classes
        new_fc = nn.Linear(self.feature_size, new_num_classes)
        new_fc = new_fc.to(old_fc.weight.device)
        
        with torch.no_grad():
            new_fc.weight[:self.num_classes].copy_(old_fc.weight)
            new_fc.bias[:self.num_classes].copy_(old_fc.bias)
            # Initialize new class weights
            nn.init.xavier_uniform_(new_fc.weight[self.num_classes:])
            nn.init.zeros_(new_fc.bias[self.num_classes:])
        
        self.fc_aux = new_fc
        self.num_classes = new_num_classes
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return only classification logits (for inference)."""
        _, aux_out = self.forward(x)
        return aux_out


class IncrementalACGAN(nn.Module):
    """
    Full ACGAN model for class-incremental learning.
    
    Combines:
    - Generator: Produces synthetic images conditioned on class labels
    - Discriminator: Distinguishes real/fake AND classifies images
    
    The auxiliary classifier in the discriminator serves as the main
    classification model, eliminating the need for a separate classifier.
    """
    def __init__(
        self,
        z_dim: int,
        num_classes: int,
        img_channels: int = 3,
        img_size: int = 32,
        base_channels: int = 64,
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        
        self.G = ACGANGenerator(
            z_dim=z_dim,
            num_classes=num_classes,
            img_channels=img_channels,
            img_size=img_size,
            base_channels=base_channels,
        )
        
        self.D = ACGANDiscriminator(
            num_classes=num_classes,
            img_channels=img_channels,
            img_size=img_size,
            base_channels=base_channels,
        )
        
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=g_lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=d_lr, betas=(0.5, 0.999)
        )
        
        self.dis_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_loss_fn = nn.CrossEntropyLoss()
    
    def expand_classes(self, num_new_classes: int) -> None:
        """Expand both generator and discriminator for new classes."""
        self.G.expand_classes(num_new_classes)
        self.D.expand_classes(num_new_classes)
        self.num_classes += num_new_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification (uses discriminator's aux head)."""
        _, logits = self.D(x)
        return logits
    
    def train_step(
        self, 
        x_real: torch.Tensor, 
        y_real: torch.Tensor,
        train_d: bool = True,
        train_g: bool = True,
    ) -> Dict[str, float]:
        """
        Single training step for ACGAN.
        
        Args:
            x_real: Real images in range [0, 1], shape (batch, C, H, W)
            y_real: Real labels, shape (batch,)
            train_d: Whether to update discriminator
            train_g: Whether to update generator
        
        Returns:
            Dictionary with loss values
        """
        batch_size = x_real.size(0)
        device = x_real.device
        
        # Skip if batch too small for BatchNorm
        if batch_size < 2:
            return {"loss_d": 0.0, "loss_g": 0.0, "loss_aux": 0.0}
        
        metrics = {}
        
        # ============================
        # Train Discriminator
        # ============================
        if train_d:
            self.D.train()
            self.d_optimizer.zero_grad()
            
            # Real images
            dis_real, aux_real = self.D(x_real)
            loss_dis_real = self.dis_loss_fn(dis_real, torch.ones_like(dis_real))
            loss_aux_real = self.aux_loss_fn(aux_real, y_real)
            
            # Generate fake images
            z = torch.randn(batch_size, self.z_dim, device=device)
            x_fake = self.G(z, y_real).detach()
            
            # Fake images
            dis_fake, aux_fake = self.D(x_fake)
            loss_dis_fake = self.dis_loss_fn(dis_fake, torch.zeros_like(dis_fake))
            loss_aux_fake = self.aux_loss_fn(aux_fake, y_real)
            
            # Total D loss
            loss_d = loss_dis_real + loss_dis_fake + loss_aux_real + loss_aux_fake
            loss_d.backward()
            self.d_optimizer.step()
            
            metrics["loss_d"] = loss_d.item()
            metrics["loss_aux"] = (loss_aux_real.item() + loss_aux_fake.item()) / 2
        
        # ============================
        # Train Generator
        # ============================
        if train_g:
            self.G.train()
            self.g_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, self.z_dim, device=device)
            x_fake = self.G(z, y_real)
            
            # Generator wants discriminator to think fakes are real
            dis_fake, aux_fake = self.D(x_fake)
            loss_dis_g = self.dis_loss_fn(dis_fake, torch.ones_like(dis_fake))
            loss_aux_g = self.aux_loss_fn(aux_fake, y_real)
            
            loss_g = loss_dis_g + loss_aux_g
            loss_g.backward()
            self.g_optimizer.step()
            
            metrics["loss_g"] = loss_g.item()
        
        return metrics
    
    @torch.no_grad()
    def sample(self, n: int, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic samples.
        
        Args:
            n: Number of samples
            labels: Optional labels to generate. If None, samples uniformly.
        
        Returns:
            x_fake: Generated images in range [0, 1]
            y_fake: Labels used for generation
        """
        self.G.eval()
        device = next(self.G.parameters()).device
        
        if labels is None:
            labels = torch.randint(0, self.num_classes, (n,), device=device)
        else:
            labels = labels.to(device)
        
        z = torch.randn(n, self.z_dim, device=device)
        x_fake = self.G(z, labels)
        
        return x_fake, labels
    
    def generator_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return generator state dict for federated aggregation."""
        return self.G.state_dict()
    
    def load_generator_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load generator state dict."""
        self.G.load_state_dict(state, strict=True)
    
    def discriminator_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return discriminator state dict for federated aggregation."""
        return self.D.state_dict()
    
    def load_discriminator_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load discriminator state dict."""
        self.D.load_state_dict(state, strict=True)
