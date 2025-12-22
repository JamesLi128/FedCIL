"""Data utilities for task streams and related helpers.

Supports multiple datasets including CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, etc.
Uses the dataset registry for centralized dataset specifications.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from config import TaskInfo
from dataset_registry import get_dataset_spec, DatasetSpec


# -------------------------
# Helper functions
# -------------------------
def chunks(seq: Sequence[int], n: int) -> Iterable[List[int]]:
	for i in range(0, len(seq), n):
		yield list(seq[i : i + n])


def filter_by_class(ds, class_ids: List[int]) -> Subset:
	mask = [i for i, y in enumerate(ds.targets) if int(y) in class_ids]
	return Subset(ds, mask)


def split_evenly(ds: Subset, num_parts: int, g: torch.Generator) -> List[Subset]:
	idx = torch.randperm(len(ds), generator=g)
	splits = torch.tensor_split(idx, num_parts)
	return [Subset(ds, split.tolist()) for split in splits if len(split) > 0]


def split_dirichlet(ds: Subset, num_parts: int, alpha: float, g: torch.Generator) -> List[Subset]:
	if num_parts <= 0:
		return []

	# Use deterministic numpy Dirichlet sampling seeded from the passed generator.
	seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g).item())
	rng = np.random.default_rng(seed)
	idx = torch.randperm(len(ds), generator=g).tolist()
	proportions = rng.dirichlet([alpha] * num_parts)
	counts = rng.multinomial(len(idx), proportions)

	splits: List[Subset] = []
	start = 0
	for count in counts:
		end = start + int(count)
		part_idx = idx[start:end]
		if part_idx:
			splits.append(Subset(ds, part_idx))
		start = end
	return splits


def build_transforms(
	spec: DatasetSpec, 
	target_img_size: int,
	convert_to_rgb: bool = False
) -> Tuple[transforms.Compose, transforms.Compose]:
	"""
	Build train and test transforms for a dataset.
	
	Args:
		spec: Dataset specification containing mean/std normalization values
		target_img_size: Target image size (e.g., 224 for ResNet)
		convert_to_rgb: Whether to convert grayscale to RGB (for 1-channel datasets)
		
	Returns:
		Tuple of (train_transform, test_transform)
	"""
	# Build normalization based on output channels
	if convert_to_rgb and spec.img_channels == 1:
		# When converting grayscale to RGB, use same stats repeated 3 times
		# But actually, for pretrained models, use ImageNet-like stats
		norm_mean = (0.485, 0.456, 0.406)
		norm_std = (0.229, 0.224, 0.225)
	else:
		norm_mean = spec.mean
		norm_std = spec.std
	
	# Base transforms
	train_transforms_list = [
		transforms.Resize((target_img_size, target_img_size)),
	]
	test_transforms_list = [
		transforms.Resize((target_img_size, target_img_size)),
	]
	
	# Add grayscale to RGB conversion if needed
	if convert_to_rgb and spec.img_channels == 1:
		train_transforms_list.append(transforms.Grayscale(num_output_channels=3))
		test_transforms_list.append(transforms.Grayscale(num_output_channels=3))
	
	# Data augmentation for training (adjust for different datasets)
	if spec.img_channels == 3 or convert_to_rgb:
		train_transforms_list.extend([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(target_img_size, padding=4),
		])
	else:
		# For grayscale without conversion, simpler augmentation
		train_transforms_list.extend([
			transforms.RandomRotation(10),
			transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
		])
	
	# Convert to tensor and normalize
	train_transforms_list.extend([
		transforms.ToTensor(),
		transforms.Normalize(norm_mean, norm_std),
	])
	
	test_transforms_list.extend([
		transforms.ToTensor(),
		transforms.Normalize(norm_mean, norm_std),
	])
	
	return transforms.Compose(train_transforms_list), transforms.Compose(test_transforms_list)


def make_gan_sample_transform(
	spec: DatasetSpec,
	target_img_size: int,
	generator_channels: int,
	convert_to_rgb: bool = False
):
	"""
	Transform synthetic GAN outputs to match training normalization.
	
	Args:
		spec: Dataset specification
		target_img_size: Target image size for the model
		generator_channels: Number of channels the generator outputs
		convert_to_rgb: Whether classifier expects RGB input
		
	Returns:
		Transform function for GAN samples
	"""
	# Determine normalization based on final output format
	if convert_to_rgb and spec.img_channels == 1:
		norm_mean = (0.485, 0.456, 0.406)
		norm_std = (0.229, 0.224, 0.225)
		final_channels = 3
	else:
		norm_mean = spec.mean
		norm_std = spec.std
		final_channels = spec.img_channels
	
	mean = torch.tensor(norm_mean).view(1, final_channels, 1, 1)
	std = torch.tensor(norm_std).view(1, final_channels, 1, 1)

	def _transform(x: torch.Tensor) -> torch.Tensor:
		# Resize to target size
		x = F.interpolate(x, size=(target_img_size, target_img_size), mode="bilinear", align_corners=False)
		
		# Convert grayscale to RGB if needed
		if convert_to_rgb and generator_channels == 1:
			x = x.repeat(1, 3, 1, 1)
		
		# Normalize
		return (x - mean.to(x)) / std.to(x)

	return _transform


# -------------------------
# Generic Task Stream
# -------------------------
class TaskStream:
	"""Pre-builds task loaders for an incremental class stream. Supports multiple datasets."""

	def __init__(
		self,
		dataset: str,
		data_root: str,
		img_size: int,
		num_clients: int,
		alpha: float,
		batch_size: int,
		eval_batch_size: int,
		classes_per_task: int,
		num_workers: int,
		seed: int,
		device: torch.device,
		convert_grayscale_to_rgb: bool = True,
	) -> None:
		"""
		Initialize a task stream for federated class-incremental learning.
		
		Args:
			dataset: Dataset name (e.g., 'mnist', 'cifar10', 'cifar100')
			data_root: Root directory for dataset storage
			img_size: Target image size (e.g., 224 for ResNet)
			num_clients: Number of federated clients
			alpha: Dirichlet concentration parameter for non-IID splits
			batch_size: Training batch size
			eval_batch_size: Evaluation batch size
			classes_per_task: Number of classes per incremental task
			num_workers: DataLoader workers
			seed: Random seed for reproducibility
			device: Target device
			convert_grayscale_to_rgb: Whether to convert grayscale datasets to RGB
		"""
		self.spec = get_dataset_spec(dataset)
		self.img_size = img_size
		self.convert_to_rgb = convert_grayscale_to_rgb and self.spec.img_channels == 1

		# Build transforms
		train_tf, test_tf = build_transforms(self.spec, img_size, self.convert_to_rgb)
		
		# Load dataset
		self.train_ds, self.test_ds = self._load_dataset(data_root, train_tf, test_tf)
		self.num_classes = self.spec.num_classes

		class_order = list(range(self.num_classes))
		gen = torch.Generator().manual_seed(seed)

		self.tasks: List[Tuple[TaskInfo, Dict[int, DataLoader], DataLoader, List[int]]] = []
		seen: List[int] = []
		for tid, cls_chunk in enumerate(chunks(class_order, classes_per_task)):
			task_classes = cls_chunk
			new_info = TaskInfo(task_id=tid, new_classes=task_classes)

			train_subset = filter_by_class(self.train_ds, task_classes)
			per_client = split_dirichlet(train_subset, num_clients, alpha, gen)
			loaders: Dict[int, DataLoader] = {}
			for cid, subset in enumerate(per_client):
				if len(subset) == 0:
					continue
				loaders[cid] = DataLoader(
					subset,
					batch_size=batch_size,
					shuffle=True,
					num_workers=num_workers,
					pin_memory=(device.type == "cuda"),
					persistent_workers=False,
				)

			for c in task_classes:
				if c not in seen:
					seen.append(c)
			eval_subset = filter_by_class(self.test_ds, seen)
			eval_loader = DataLoader(
				eval_subset,
				batch_size=eval_batch_size,
				shuffle=False,
				num_workers=num_workers,
				pin_memory=(device.type == "cuda"),
				persistent_workers=False,
			)

			self.tasks.append((new_info, loaders, eval_loader, list(seen)))
	
	def _load_dataset(
		self, 
		data_root: str, 
		train_tf: transforms.Compose, 
		test_tf: transforms.Compose
	) -> Tuple[Dataset, Dataset]:
		"""Load the dataset using the specification."""
		ds_cls = self.spec.dataset_cls
		
		# Handle special cases for certain datasets
		if self.spec.name == "emnist":
			train_ds = ds_cls(root=data_root, split='balanced', train=True, download=True, transform=train_tf)
			test_ds = ds_cls(root=data_root, split='balanced', train=False, download=True, transform=test_tf)
		else:
			train_ds = ds_cls(root=data_root, train=True, download=True, transform=train_tf)
			test_ds = ds_cls(root=data_root, train=False, download=True, transform=test_tf)
		
		return train_ds, test_ds

	def __iter__(self):
		for task_info, loaders, _, _ in self.tasks:
			yield task_info, loaders

	def __len__(self) -> int:
		return len(self.tasks)

	def eval_loader(self, task_id: int) -> DataLoader:
		return self.tasks[task_id][2]

	def seen_classes(self, task_id: int) -> List[int]:
		return self.tasks[task_id][3]

	def gan_sample_transform(self):
		"""Get the transform for GAN-generated samples."""
		# Generator produces images with original dataset channels
		generator_channels = self.spec.img_channels
		return make_gan_sample_transform(
			self.spec, 
			self.img_size, 
			generator_channels,
			self.convert_to_rgb
		)
	
	@property
	def input_channels(self) -> int:
		"""Number of input channels expected by the model (after any RGB conversion)."""
		return 3 if self.convert_to_rgb else self.spec.img_channels
	
	@property
	def generator_channels(self) -> int:
		"""Number of channels the GAN generator should produce."""
		return self.spec.img_channels
	
	@property
	def generator_img_size(self) -> int:
		"""Image size the GAN generator should produce (original dataset size)."""
		return self.spec.original_img_size


# -------------------------
# Backward compatibility alias
# -------------------------
class CIFARTaskStream(TaskStream):
	"""Backward-compatible alias for TaskStream. Use TaskStream for new code."""
	pass
