# ==================================
# --- file: fcil/acgan.py
# ==================================
"""
ACGAN (Auxiliary Classifier GAN) module for Federated Class-Incremental Learning.

This implementation:
- Supports multiple image sizes (28x28 MNIST, 32x32 CIFAR, etc.)
- Has an expandable auxiliary classifier for class-incremental learning
- Provides both image generation and classification in a single model
- Uses dataset-aware custom conv backbones (no pretrained ResNet)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Dataset-specific architecture configurations
# ============================================================================

@dataclass
class DatasetArchConfig:
    """Architecture configuration for a specific dataset."""
    img_size: int
    img_channels: int
    # Generator config
    gen_start_size: int  # Starting spatial size before upsampling
    gen_init_channels: int  # Number of channels after FC projection
    # Discriminator config  
    disc_feature_size: int  # Spatial size after conv backbone (before flatten)
    disc_final_channels: int  # Number of channels at final conv layer


# Pre-defined configurations for common datasets
DATASET_ARCH_CONFIGS: Dict[str, DatasetArchConfig] = {
    "mnist": DatasetArchConfig(
        img_size=28, img_channels=1,
        gen_start_size=7, gen_init_channels=256,
        disc_feature_size=3, disc_final_channels=256
    ),
    "fashion_mnist": DatasetArchConfig(
        img_size=28, img_channels=1,
        gen_start_size=7, gen_init_channels=256,
        disc_feature_size=3, disc_final_channels=256
    ),
    "emnist": DatasetArchConfig(
        img_size=28, img_channels=1,
        gen_start_size=7, gen_init_channels=256,
        disc_feature_size=3, disc_final_channels=256
    ),
    "cifar10": DatasetArchConfig(
        img_size=32, img_channels=3,
        gen_start_size=4, gen_init_channels=512,
        disc_feature_size=4, disc_final_channels=256
    ),
    "cifar100": DatasetArchConfig(
        img_size=32, img_channels=3,
        gen_start_size=4, gen_init_channels=512,
        disc_feature_size=4, disc_final_channels=256
    ),
}


def get_arch_config(dataset_name: str) -> DatasetArchConfig:
    """Get architecture config for a dataset."""
    name_lower = dataset_name.lower()
    if name_lower not in DATASET_ARCH_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_ARCH_CONFIGS.keys())}")
    return DATASET_ARCH_CONFIGS[name_lower]


# ============================================================================
# Generator
# ============================================================================

class ACGANGenerator(nn.Module):
    """
    ACGAN Generator with dataset-aware architecture.
    
    Architecture is determined by the dataset:
    - MNIST/Fashion-MNIST/EMNIST (28x28): 7x7 -> 14x14 -> 28x28
    - CIFAR-10/100 (32x32): 4x4 -> 8x8 -> 16x16 -> 32x32
    
    Output range is [-1, 1] (using Tanh), matching normalized training data.
    """
    def __init__(
        self,
        z_dim: int,
        num_classes: int,
        dataset_name: str = "mnist",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset_name = dataset_name.lower()
        
        # Get dataset-specific architecture config
        arch = get_arch_config(self.dataset_name)
        self.img_channels = arch.img_channels
        self.img_size = arch.img_size
        self.start_size = arch.gen_start_size
        self.init_channels = arch.gen_init_channels
        
        # Input: z_dim + num_classes (one-hot label)
        self.input_dim = z_dim + num_classes
        
        # Project and reshape: z+y -> feature map
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.init_channels * self.start_size * self.start_size),
            nn.ReLU(),
        )
        
        # Build dataset-specific decoder
        self.deconv = self._build_decoder(arch)
    
    def _build_decoder(self, arch: DatasetArchConfig) -> nn.Sequential:
        """Build decoder layers based on dataset architecture."""
        if self.dataset_name in ("mnist", "fashion_mnist", "emnist"):
            # 28x28 images: 7x7 -> 14x14 -> 28x28
            return nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 14x14 -> 28x28
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Final conv to get 1 channel
                nn.Conv2d(64, arch.img_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
        elif self.dataset_name in ("cifar10", "cifar100"):
            # 32x32 images: 4x4 -> 8x8 -> 16x16 -> 32x32
            return nn.Sequential(
                # 4x4 -> 8x8
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                # 8x8 -> 16x16
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                # 16x16 -> 32x32
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                # Final conv to get 3 channels
                nn.Conv2d(64, arch.img_channels, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"No decoder defined for dataset: {self.dataset_name}")
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noise tensor of shape (batch, z_dim)
            y: Class labels of shape (batch,) - integer labels
        
        Returns:
            Generated images in range [-1, 1], shape (batch, C, H, W)
        """
        batch_size = z.size(0)
        
        # Create one-hot encoding
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        
        # Concatenate noise and label
        z_y = torch.cat([z, y_onehot], dim=1)
        
        # Project and reshape
        h = self.fc(z_y)
        h = h.view(batch_size, self.init_channels, self.start_size, self.start_size)
        
        # Upsample to target size
        img = self.deconv(h)
        return img
    
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


def _build_frozen_resnet18_backbone(input_channels: int = 3) -> Tuple[nn.Module, int]:
    """
    Build a frozen pretrained ResNet18 backbone for the discriminator.
    
    Args:
        input_channels: Number of input channels (should be 3 for pretrained weights)
    
    Returns:
        Tuple of (backbone module, feature dimension)
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=True)
    
    feat_dim = backbone.fc.in_features  # 512 for ResNet18
    backbone.fc = nn.Identity()  # Remove final classification layer
    
    # Modify first conv layer if input channels != 3
    if input_channels != 3:
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            if input_channels == 1:
                # Average RGB weights for grayscale
                backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # Freeze all backbone parameters
    for p in backbone.parameters():
        p.requires_grad = False
    
    return backbone, feat_dim


class ACGANDiscriminator(nn.Module):
    """
    ACGAN Discriminator with dataset-aware custom conv backbone.
    
    Architecture is determined by the dataset:
    - MNIST/Fashion-MNIST/EMNIST (28x28): 28x28 -> 14x14 -> 7x7 -> 3x3
    - CIFAR-10/100 (32x32): 32x32 -> 16x16 -> 8x8 -> 4x4
    
    All parameters are trainable (no frozen pretrained backbone).
    Has two heads:
        - fc_dis: Real/fake discrimination head
        - fc_aux: Auxiliary classification head (expandable for incremental learning)
    """
    def __init__(
        self,
        num_classes: int,
        dataset_name: str = "mnist",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dataset_name = dataset_name.lower()
        
        # Get dataset-specific architecture config
        arch = get_arch_config(self.dataset_name)
        self.img_channels = arch.img_channels
        self.img_size = arch.img_size
        
        # Build dataset-specific conv backbone
        self.backbone = self._build_backbone(arch)
        
        # Calculate feature size after backbone
        self.feature_size = arch.disc_final_channels * arch.disc_feature_size * arch.disc_feature_size
        
        # Discriminator head (real/fake) - with label conditioning
        self.fc_dis = nn.Linear(self.feature_size, 1)
        
        # Auxiliary classifier head (class prediction)
        self.fc_aux = nn.Linear(self.feature_size, num_classes)
    
    def _build_backbone(self, arch: DatasetArchConfig) -> nn.Sequential:
        """Build convolutional backbone based on dataset architecture."""
        if self.dataset_name in ("mnist", "fashion_mnist", "emnist"):
            # 28x28 -> 14x14 -> 7x7 -> 3x3 (feature_size=3)
            return nn.Sequential(
                # 28x28 -> 14x14
                nn.Conv2d(arch.img_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # 14x14 -> 7x7
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                # 7x7 -> 3x3
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
            )
        elif self.dataset_name in ("cifar10", "cifar100"):
            # 32x32 -> 16x16 -> 8x8 -> 4x4 (feature_size=4)
            return nn.Sequential(
                # 32x32 -> 16x16
                nn.Conv2d(arch.img_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # 16x16 -> 8x8
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                # 8x8 -> 4x4
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
            )
        else:
            raise ValueError(f"No backbone defined for dataset: {self.dataset_name}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images of shape (batch, channels, height, width)
            y: Optional class labels for conditional discrimination (batch,)
        
        Returns:
            dis_out: Real/fake logits of shape (batch,)
            aux_out: Class logits of shape (batch, num_classes)
        """
        # Extract features using conv backbone
        h = self.backbone(x)
        h = h.flatten(1)  # (batch, feature_size)
        
        # Auxiliary classification
        aux_out = self.fc_aux(h)
        # Discriminator output
        dis_out = self.fc_dis(h).squeeze(1)
        
        return dis_out, aux_out
    
    def expand_classes(self, num_new_classes: int) -> None:
        if num_new_classes <= 0:
            return
        
        old_num_classes = self.num_classes
        new_num_classes = old_num_classes + num_new_classes
        
        # Expand auxiliary classifier head
        old_fc = self.fc_aux
        new_fc = nn.Linear(self.feature_size, new_num_classes)
        new_fc = new_fc.to(old_fc.weight.device)
        
        with torch.no_grad():
            new_fc.weight[:old_num_classes].copy_(old_fc.weight)
            new_fc.bias[:old_num_classes].copy_(old_fc.bias)
            # Initialize new class weights
            nn.init.xavier_uniform_(new_fc.weight[old_num_classes:])
            nn.init.zeros_(new_fc.bias[old_num_classes:])
        
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
    - Discriminator: Custom conv backbone for real/fake + classification
    
    Both generator and discriminator use dataset-aware architectures with custom
    conv backbones trained from scratch. No pretrained ResNet is used.
    
    Data is expected to be normalized to [-1, 1] (matching generator output range).
    """
    def __init__(
        self,
        z_dim: int,
        num_classes: int,
        dataset_name: str = "mnist",
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset_name = dataset_name.lower()
        
        # Get architecture config for backward compatibility
        arch = get_arch_config(self.dataset_name)
        self.img_channels = arch.img_channels
        self.img_size = arch.img_size
        
        # For backward compatibility
        self.generator_img_channels = arch.img_channels
        self.generator_img_size = arch.img_size
        self.discriminator_img_channels = arch.img_channels
        self.discriminator_img_size = arch.img_size
        
        # Create generator
        self.G = ACGANGenerator(
            z_dim=z_dim,
            num_classes=num_classes,
            dataset_name=dataset_name,
        )
        
        # Create discriminator
        self.D = ACGANDiscriminator(
            num_classes=num_classes,
            dataset_name=dataset_name,
        )
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), lr=g_lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=d_lr, betas=(0.5, 0.999)
        )
        # Only update discriminator auxiliary head during distillation
        self.distill_optimizer = torch.optim.Adam(
            self.D.fc_aux.parameters(), lr=d_lr, betas=(0.5, 0.999)
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
    
    def train_D_loss(self, x_real: torch.Tensor, y_real: torch.Tensor, x_fake_prev: torch.Tensor, y_fake_prev: torch.Tensor, x_fake_curr: torch.Tensor, y_fake_curr: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss with gradient for a batch."""
        
        # Real images
        dis_real, aux_real = self.D(x_real)
        loss_dis_real = self.dis_loss_fn(dis_real, torch.ones_like(dis_real))
        loss_aux_real = self.aux_loss_fn(aux_real, y_real)

        # Fake images (previous)
        if x_fake_prev is not None:
            dis_fake_prev, aux_fake_prev = self.D(x_fake_prev)
            loss_dis_fake_prev = self.dis_loss_fn(dis_fake_prev, torch.ones_like(dis_fake_prev))
            loss_aux_fake_prev = self.aux_loss_fn(aux_fake_prev, y_fake_prev)
        else:
            loss_dis_fake_prev = torch.tensor(0.0, device=x_real.device)
            loss_aux_fake_prev = torch.tensor(0.0, device=x_real.device)

        # Fake images (current)
        dis_fake_curr, aux_fake_curr = self.D(x_fake_curr)
        loss_dis_fake_curr = self.dis_loss_fn(dis_fake_curr, torch.zeros_like(dis_fake_curr))
        # loss_aux_fake_curr = self.aux_loss_fn(aux_fake_curr, y_fake_curr)

        # Total D loss
        loss_d_replay = loss_dis_fake_prev + loss_aux_fake_prev
        loss_d_real = loss_dis_real + loss_aux_real
        loss_d_fake = loss_dis_fake_curr
        
        loss_d = loss_d_real + loss_d_fake + loss_d_replay
        
        return loss_d, loss_aux_real.item(), loss_d_real.item(), loss_d_replay.item(), loss_d_fake.item()
    
    def train_G_loss(self, x_fake: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
        """Compute generator loss with gradient for a batch."""
        
        # Generator wants discriminator to think fakes are real
        dis_fake, aux_fake = self.D(x_fake)
        loss_dis_g = self.dis_loss_fn(dis_fake, torch.ones_like(dis_fake))
        loss_aux_g = self.aux_loss_fn(aux_fake, y_fake)
        
        loss_g = loss_dis_g + loss_aux_g
        return loss_g, loss_dis_g.item(), loss_aux_g.item()
    
    def train_step(
        self, 
        x_real: torch.Tensor, 
        y_real: torch.Tensor,
        sample_classes: List[int],
        n_replay: int = 0,
        prev_task_model: Optional[IncrementalACGAN] = None,
    ) -> Dict[str, float]:
        """
        Single training step for ACGAN.
        
        Args:
            x_real: Real images in range [-1, 1], shape (batch, C, H, W)
            y_real: Real labels, shape (batch,)
            known_classes: List of known class labels
            n_replay: Number of replay samples to generate
        
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
        self.D.train()
        self.d_optimizer.zero_grad()

        # ============================
        # A. PREPARE REPLAY DATA
        # ============================
        z_prev = torch.randn(n_replay, self.z_dim, device=device)

        # Sample random labels from known classes, uniformly
        if sample_classes:
            y_fake = torch.tensor(
                [sample_classes[i] for i in torch.randint(0, len(sample_classes), (n_replay,))],
                dtype=torch.long,
                device=device
            )
        else:
            y_fake = y_real[:n_replay]

        if prev_task_model is not None:
            with torch.no_grad():
                x_fake_prev = prev_task_model.G(z_prev, y_fake)
        else:
            x_fake_prev = None


        # ============================
        # B. GENERATE FAKE IMAGES
        # ============================
        z_curr = torch.randn(batch_size, self.z_dim, device=device)
        x_fake_curr = self.G(z_curr, y_real)


        # ============================
        # C. COMPUTE D LOSS AND UPDATE
        # ============================

        loss_d, loss_ce, loss_d_real, loss_d_replay, loss_d_fake = self.train_D_loss(
            x_real=x_real,
            y_real=y_real,
            x_fake_prev=x_fake_prev,
            y_fake_prev=y_fake,
            x_fake_curr=x_fake_curr,
            y_fake_curr=y_real,
        )

        loss_d.backward()
        self.d_optimizer.step()

        # ============================
        # LOG METRICS
        # ============================
        
        metrics["loss_d"] = loss_d.item()
        metrics["loss_ce"] = loss_ce
        metrics["loss_d_real"] = loss_d_real
        metrics["loss_d_replay"] = loss_d_replay
        metrics["loss_d_fake"] = loss_d_fake
        
        # ============================
        # Train Generator
        # ============================
        self.G.train()
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size + n_replay, self.z_dim, device=device)
        if sample_classes and n_replay > 0:
            y_gen = torch.cat([y_real, torch.tensor(
                [sample_classes[i] for i in torch.randint(0, len(sample_classes), (n_replay,))],
                dtype=torch.long,
                device=device
            )], dim=0)
        else:
            if n_replay > 0 and n_replay <= batch_size:
                y_gen = torch.cat([y_real, y_real[:n_replay]], dim=0)
            elif n_replay > 0 and n_replay > batch_size:
                y_gen = torch.cat([y_real] * (1 + n_replay // batch_size) + [y_real[:n_replay % batch_size]], dim=0)
                
        x_fake = self.G(z, y_gen)
        
        loss_g, loss_dis_g, loss_aux_g = self.train_G_loss(x_fake, y_gen)
        loss_g.backward()
        self.g_optimizer.step()
        
        metrics["loss_g"] = loss_g.item()
        metrics["loss_dis_g"] = loss_dis_g
        metrics["loss_aux_g"] = loss_aux_g
    
        return metrics
    
    def KL_distill_from(self, prev_task_model: IncrementalACGAN, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Distill knowledge from previous model using generated samples.
        Use KL divergence to match auxiliary classifier outputs.
        
        Args:
            prev_task_model: Previous IncrementalACGAN model
            x: Input images for distillation
            y: True labels for input images
        """
        self.D.train()
        self.d_optimizer.zero_grad()
        
        # Get previous model's predictions
        with torch.no_grad():
            _, prev_aux_logits = prev_task_model.D(x)
            prev_aux_probs = F.softmax(prev_aux_logits, dim=1)
            
        
        # Current model's predictions
        _, curr_aux_logits = self.D(x)
        curr_aux_probs = F.softmax(curr_aux_logits, dim=1)

        # Use KL divergence for distillation
        aux_loss = F.kl_div(curr_aux_probs.log(), prev_aux_probs, reduction='batchmean')

        loss_distill = aux_loss
        loss_distill.backward()
        self.d_optimizer.step()

    def distill_step(self, x_teacher: torch.Tensor, x_student: torch.Tensor, y_teacher: torch.Tensor = None) -> None:
        """
        Distillation step using two sets of generated samples.
        
        Args:
            x1: First set of generated images for distillation
            x2: Second set of generated images for distillation
        """
        self.D.train()
        self.distill_optimizer.zero_grad()
        
        if y_teacher == None:
            _, aux_logits_teacher = self.D(x_teacher)
            aux_probs_teacher = F.softmax(aux_logits_teacher, dim=1)
        else:
            # use onehot y_teacher to get aux_probs_teacher
            y_onehot = F.one_hot(y_teacher, num_classes=self.num_classes).float()
            aux_probs_teacher = y_onehot
        _, aux_logits_student = self.D(x_student)
        aux_probs_student = F.softmax(aux_logits_student, dim=1)

        aux_loss = F.kl_div(aux_probs_student.log(), aux_probs_teacher, reduction='batchmean')
        aux_loss.backward()
        self.distill_optimizer.step()



    def distill_c1(self, prev_local_model: IncrementalACGAN, prev_global_model: IncrementalACGAN, prev_known_classes: List[int], n_distill: int) -> None:
        """
        Distill knowledge from previous local and global models using generated samples.
        
        Args:
            prev_local_model: Previous local IncrementalACGAN model
            prev_global_model: Previous global IncrementalACGAN model
            prev_known_classes: List of known class labels in previous task
            n_distill: Number of distillation samples to generate
        """
        device = next(self.G.parameters()).device
        
        # Generate distillation samples from previous local model
        z = torch.randn(n_distill, self.z_dim, device=device)
        y_distill = torch.tensor(
            [prev_known_classes[i] for i in torch.randint(0, len(prev_known_classes), (n_distill,))],
            dtype=torch.long,
            device=device
        )
        
        with torch.no_grad():
            x_teacher = prev_local_model.G(z, y_distill)
            x_student = prev_global_model.G(z, y_distill)

        self.distill_step(x_teacher=x_teacher, x_student=x_student)

    def distill_c2(self, prev_global_model: IncrementalACGAN, x_real: torch.Tensor, y_real: torch.Tensor) -> None:
        """
        Distill knowledge from previous global model using real samples.
        
        Args:
            prev_global_model: Previous global IncrementalACGAN model
            x_real: Real images for distillation
            y_real: True labels for real images
        """
        device = next(self.G.parameters()).device
        
        x_real = x_real.to(device)
        y_real = y_real.to(device)

        # Sanity check: skip if y_real contains classes not known by prev_global_model
        max_label = y_real.max().item()
        if max_label >= prev_global_model.num_classes:
            # Skip distillation if labels are out of bounds for prev_global_model
            return

        z = torch.randn(x_real.size(0), self.z_dim, device=device)
        with torch.no_grad():
            x_student = prev_global_model.G(z, y_real)

        self.distill_step(x_teacher=x_real, x_student=x_student)

    def distill_c3(self, prev_global_model: IncrementalACGAN, prev_known_classes: List[int], n_distill: int) -> None:
        """
        Distill knowledge from previous global model using generated samples.
        
        Args:
            prev_global_model: Previous global IncrementalACGAN model
            prev_known_classes: List of known class labels in previous task
            n_distill: Number of distillation samples to generate
        """
        try:
            device = next(self.G.parameters()).device
            
            # Validate that prev_known_classes are within bounds for prev_global_model
            if prev_known_classes:
                max_prev_class = max(prev_known_classes)
                if max_prev_class >= prev_global_model.num_classes:
                    raise ValueError(
                        f"Class index out of bounds: max class in prev_known_classes ({max_prev_class}) "
                        f">= prev_global_model.num_classes ({prev_global_model.num_classes})"
                    )
            
            z = torch.randn(n_distill, self.z_dim, device=device)
            y_distill = torch.tensor(
                [prev_known_classes[i] for i in torch.randint(0, len(prev_known_classes), (n_distill,))],
                dtype=torch.long,
                device=device
            )
            x_student = prev_global_model.G(z, y_distill)

            self.distill_step(x_teacher=None, x_student=x_student, y_teacher=y_distill)
        
        except (RuntimeError, IndexError, ValueError) as e:
            print(f"\n{'='*80}")
            print(f"ERROR in distill_c3:")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"  prev_known_classes: {prev_known_classes}")
            print(f"  prev_global_model.num_classes: {prev_global_model.num_classes}")
            print(f"  self.num_classes: {self.num_classes}")
            print(f"  n_distill: {n_distill}")
            print(f"{'='*80}\n")
            import sys
            sys.exit(1)
        
        

    @torch.no_grad()
    def sample(self, n: int, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic samples.
        
        Args:
            n: Number of samples
            labels: Optional labels to generate. If None, samples uniformly.
        
        Returns:
            x_fake: Generated images in range [-1, 1]
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
