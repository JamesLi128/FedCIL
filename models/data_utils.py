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
	
	For consistency in continual/incremental learning, train and test transforms
	are identical - no data augmentation is applied. This ensures that the model
	sees data from the same distribution during training and evaluation.
	
	Images are normalized to [-1, 1] range to match GAN generator output (Tanh).
	
	Args:
		spec: Dataset specification containing mean/std normalization values
		target_img_size: Target image size (original dataset size, e.g., 28 for MNIST)
		convert_to_rgb: Whether to convert grayscale to RGB (deprecated, no longer used)
		
	Returns:
		Tuple of (train_transform, test_transform) - both are identical
	"""
	# Build transform pipeline - keep original size, normalize to [-1, 1]
	transforms_list = []
	
	# Only resize if target size differs from original
	if target_img_size != spec.original_img_size:
		transforms_list.append(transforms.Resize((target_img_size, target_img_size)))
	
	# Convert to tensor (scales to [0, 1]) and normalize to [-1, 1]
	transforms_list.extend([
		transforms.ToTensor(),
		transforms.Normalize(spec.mean, spec.std),
	])
	
	tf = transforms.Compose(transforms_list)
	# Return same transform for both train and test
	return tf, tf


def make_gan_sample_transform(
	spec: DatasetSpec,
	target_img_size: int,
	generator_channels: int,
	convert_to_rgb: bool = False
):
	"""
	Transform synthetic GAN outputs to match training normalization.
	
	GAN generator outputs images in [-1, 1] range (Tanh activation).
	Training data is also normalized to [-1, 1] with mean=0.5, std=0.5.
	So no transformation is needed if sizes match.
	
	Args:
		spec: Dataset specification
		target_img_size: Target image size for the model
		generator_channels: Number of channels the generator outputs
		convert_to_rgb: Deprecated, no longer used
		
	Returns:
		Transform function for GAN samples (identity if sizes match)
	"""
	def _transform(x: torch.Tensor) -> torch.Tensor:
		# Only resize if needed
		if x.shape[-1] != target_img_size or x.shape[-2] != target_img_size:
			x = F.interpolate(x, size=(target_img_size, target_img_size), mode="bilinear", align_corners=False)
		# GAN output is already in [-1, 1], same as training data normalization
		return x

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
		img_size: Optional[int],  # If None, uses original dataset size
		num_clients: int,
		alpha: float,
		batch_size: int,
		eval_batch_size: int,
		classes_per_task: int,
		num_workers: int,
		seed: int,
		device: torch.device,
		convert_grayscale_to_rgb: bool = False,  # Deprecated, kept for backward compatibility
	) -> None:
		"""
		Initialize a task stream for federated class-incremental learning.
		
		Args:
			dataset: Dataset name (e.g., 'mnist', 'cifar10', 'cifar100')
			data_root: Root directory for dataset storage
			img_size: Target image size. If None, uses original dataset size (recommended).
			num_clients: Number of federated clients
			alpha: Dirichlet concentration parameter for non-IID splits
			batch_size: Training batch size
			eval_batch_size: Evaluation batch size
			classes_per_task: Number of classes per incremental task
			num_workers: DataLoader workers
			seed: Random seed for reproducibility
			device: Target device
			convert_grayscale_to_rgb: Deprecated, no longer used
		"""
		self.spec = get_dataset_spec(dataset)
		# Use original dataset size if not specified
		self.img_size = img_size if img_size is not None else self.spec.original_img_size
		self.convert_to_rgb = False  # No longer convert to RGB

		# Build transforms
		train_tf, test_tf = build_transforms(self.spec, self.img_size, self.convert_to_rgb)
		
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
		"""Number of input channels expected by the model."""
		return self.spec.img_channels
	
	@property
	def generator_channels(self) -> int:
		"""Number of channels the GAN generator should produce."""
		return self.spec.img_channels
	
	@property
	def generator_img_size(self) -> int:
		"""Image size the GAN generator should produce (original dataset size)."""
		return self.spec.original_img_size
	
	@property
	def dataset_name(self) -> str:
		"""Name of the dataset."""
		return self.spec.name


# -------------------------
# Backward compatibility alias
# -------------------------
class CIFARTaskStream(TaskStream):
	"""Backward-compatible alias for TaskStream. Use TaskStream for new code."""
	pass
