"""
Dataset registry for supporting multiple datasets in FCIL.

This module provides a centralized registry for dataset specifications,
including image properties, normalization statistics, and torchvision dataset classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type, Callable
from torchvision import datasets


@dataclass
class DatasetSpec:
    """Specification for a dataset including all properties needed for training."""
    
    name: str
    num_classes: int
    original_img_size: int  # Original image size (e.g., 32 for CIFAR, 28 for MNIST)
    img_channels: int  # Number of input channels (1 for grayscale, 3 for RGB)
    mean: Tuple[float, ...]  # Normalization mean per channel
    std: Tuple[float, ...]  # Normalization std per channel
    dataset_cls: Type  # torchvision dataset class
    default_classes_per_task: int = 2
    requires_grayscale_to_rgb: bool = False  # Whether to convert grayscale to RGB
    
    def __post_init__(self):
        # Validate channel count matches mean/std length
        if len(self.mean) != self.img_channels:
            raise ValueError(f"mean length ({len(self.mean)}) must match img_channels ({self.img_channels})")
        if len(self.std) != self.img_channels:
            raise ValueError(f"std length ({len(self.std)}) must match img_channels ({self.img_channels})")


# -------------------------
# Dataset Specifications
# -------------------------

CIFAR10_SPEC = DatasetSpec(
    name="cifar10",
    num_classes=10,
    original_img_size=32,
    img_channels=3,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2470, 0.2430, 0.2616),
    dataset_cls=datasets.CIFAR10,
    default_classes_per_task=2,
)

CIFAR100_SPEC = DatasetSpec(
    name="cifar100",
    num_classes=100,
    original_img_size=32,
    img_channels=3,
    mean=(0.5071, 0.4867, 0.4408),
    std=(0.2675, 0.2565, 0.2761),
    dataset_cls=datasets.CIFAR100,
    default_classes_per_task=10,
)

MNIST_SPEC = DatasetSpec(
    name="mnist",
    num_classes=10,
    original_img_size=28,
    img_channels=1,
    mean=(0.5,),  # Normalize to [0, 1] range then center
    std=(0.5,),
    dataset_cls=datasets.MNIST,
    default_classes_per_task=2,
    requires_grayscale_to_rgb=False,  # Using custom conv backbone, no RGB conversion needed
)

FASHION_MNIST_SPEC = DatasetSpec(
    name="fashion_mnist",
    num_classes=10,
    original_img_size=28,
    img_channels=1,
    mean=(0.5,),  # Normalize to [0, 1] range then center
    std=(0.5,),
    dataset_cls=datasets.FashionMNIST,
    default_classes_per_task=2,
    requires_grayscale_to_rgb=False,  # Using custom conv backbone, no RGB conversion needed
)

EMNIST_SPEC = DatasetSpec(
    name="emnist",
    num_classes=47,  # Using 'balanced' split
    original_img_size=28,
    img_channels=1,
    mean=(0.5,),  # Normalize to [0, 1] range then center
    std=(0.5,),
    dataset_cls=datasets.EMNIST,
    default_classes_per_task=5,
    requires_grayscale_to_rgb=False,  # Using custom conv backbone, no RGB conversion needed
)


# -------------------------
# Registry
# -------------------------

DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "cifar10": CIFAR10_SPEC,
    "cifar100": CIFAR100_SPEC,
    "mnist": MNIST_SPEC,
    "fashion_mnist": FASHION_MNIST_SPEC,
    "emnist": EMNIST_SPEC,
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """
    Get dataset specification by name.
    
    Args:
        name: Dataset name (case-insensitive)
        
    Returns:
        DatasetSpec for the requested dataset
        
    Raises:
        ValueError: If dataset is not found in registry
    """
    name_lower = name.lower()
    if name_lower not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name_lower]


def register_dataset(spec: DatasetSpec) -> None:
    """
    Register a new dataset specification.
    
    Args:
        spec: DatasetSpec to register
    """
    DATASET_REGISTRY[spec.name.lower()] = spec


def list_available_datasets() -> list[str]:
    """Return list of available dataset names."""
    return list(DATASET_REGISTRY.keys())
