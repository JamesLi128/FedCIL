"""Data utilities for CIFAR task streams and related helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import TaskInfo

# -------------------------
# Constants
# -------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2430, 0.2616)


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


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
	train_tf = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(img_size, padding=4),
			transforms.ToTensor(),
			transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
		]
	)
	test_tf = transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
		]
	)
	return train_tf, test_tf


def make_gan_sample_transform(img_size: int):
	"""Transform synthetic GAN outputs to match training normalization."""

	mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
	std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)

	def _transform(x: torch.Tensor) -> torch.Tensor:
		x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
		return (x - mean.to(x)) / std.to(x)

	return _transform


# -------------------------
# CIFAR task stream
# -------------------------
class CIFARTaskStream:
	"""Pre-builds task loaders for an incremental class stream."""

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
	) -> None:
		self.img_size = img_size

		train_tf, test_tf = build_transforms(img_size)
		ds_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
		self.train_ds = ds_cls(root=data_root, train=True, download=True, transform=train_tf)
		self.test_ds = ds_cls(root=data_root, train=False, download=True, transform=test_tf)
		self.num_classes = 10 if dataset == "cifar10" else 100

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
					persistent_workers=False,  # Ensure workers terminate for cleanup
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
				persistent_workers=False,  # Ensure workers terminate for cleanup
			)

			self.tasks.append((new_info, loaders, eval_loader, list(seen)))

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
		return make_gan_sample_transform(self.img_size)
