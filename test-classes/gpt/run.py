"""
Minimal runnable entrypoint for ExampleFedAvgWithGANReplay.

Features:
- CLI hyperparameter overrides (no edits to config.py needed).
- Progress bar with ETA + per-round logging.
- TensorBoard logging for loss/accuracy (real-time if tensorboard is running).
- Supports CIFAR-10/100 stored under ~/data with upsampled inputs for ResNet18.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from config import ClientConfig, ServerConfig, TaskInfo
from example_method import ExampleFedAvgWithGANReplay


# -------------------------
# Constants
# -------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# -------------------------
# Helpers
# -------------------------
def log_section(title: str, items: Dict[str, object]) -> None:
	print(f"\n[{title}]")
	for key in sorted(items.keys()):
		print(f"{key:20}: {items[key]}")


def log_configuration(
	args: argparse.Namespace,
	client_cfg: ClientConfig,
	server_cfg: ServerConfig,
	device: torch.device,
	log_dir: Path,
	total_tasks: int,
	total_rounds: int,
	total_classes: int,
) -> None:
	print("=" * 60)
	print("Initialized FCIL run with: double-check TensorBoard for live metrics.")
	log_section("CLI Args", vars(args))
	log_section("ClientConfig", asdict(client_cfg))
	log_section("ServerConfig", asdict(server_cfg))
	log_section(
		"Derived",
		{
			"device": device,
			"log_dir": log_dir,
			"total_tasks": total_tasks,
			"total_rounds": total_rounds,
			"total_classes": total_classes,
		},
	)
	print("=" * 60)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run FCIL FedAvg+GAN on CIFAR")
	parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
	parser.add_argument("--data-root", default=str(Path.home() / "data"))
	parser.add_argument("--cache-dir", default=str(Path.home() / ".cache"), help="Where torchvision caches models")
	parser.add_argument("--log-dir", default="~/scratch/logs/", help="TensorBoard log directory")
	parser.add_argument("--num-clients", type=int, default=5)
	parser.add_argument("--clients-per-round", type=int, default=3)
	parser.add_argument("--global-rounds", type=int, default=5)
	parser.add_argument("--local-epochs", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--eval-batch-size", type=int, default=512)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=0.0)
	parser.add_argument("--replay-ratio", type=float, default=0.5)
	parser.add_argument("--max-grad-norm", type=float, default=None)
	parser.add_argument("--classes-per-task", type=int, default=2, help="How many classes appear in each new task")
	parser.add_argument("--eval-every", type=int, default=1, help="Rounds between evals (per task)")
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--server-opt-steps", type=int, default=0, help="Optional server-side optimization steps")
	parser.add_argument("--server-opt-lr", type=float, default=1e-4)
	parser.add_argument("--server-replay-batch", type=int, default=128)
	parser.add_argument("--img-size", type=int, default=224, help="Resize input for ResNet18")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
	return parser.parse_args()


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


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


def format_seconds(sec: float) -> str:
	sec = max(sec, 0.0)
	m, s = divmod(int(sec), 60)
	h, m = divmod(m, 60)
	return f"{h:02d}:{m:02d}:{s:02d}"


def make_gan_sample_transform(img_size: int):
	mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
	std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)

	def _transform(x: torch.Tensor) -> torch.Tensor:
		x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
		return (x - mean.to(x)) / std.to(x)

	return _transform


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
	model.eval()
	correct, total = 0, 0
	for x, y in loader:
		x = x.to(device)
		y = y.to(device)
		logits = model(x)
		pred = logits.argmax(dim=1)
		correct += (pred == y).sum().item()
		total += y.numel()
	return float(correct) / max(total, 1)


class CIFARTaskStream:
	"""Pre-builds task loaders for an incremental class stream."""

	def __init__(
		self,
		dataset: str,
		data_root: str,
		img_size: int,
		num_clients: int,
		batch_size: int,
		eval_batch_size: int,
		classes_per_task: int,
		num_workers: int,
		seed: int,
		device: torch.device,
	) -> None:
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
			per_client = split_evenly(train_subset, num_clients, gen)
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


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	cache_dir = Path(args.cache_dir).expanduser()
	cache_dir.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))

	log_dir = Path(args.log_dir).expanduser()
	log_dir.mkdir(parents=True, exist_ok=True)

	device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
	torch.backends.cudnn.benchmark = True

	data_root = str(Path(args.data_root).expanduser())
	stream = CIFARTaskStream(
		dataset=args.dataset,
		data_root=data_root,
		img_size=args.img_size,
		num_clients=args.num_clients,
		batch_size=args.batch_size,
		eval_batch_size=args.eval_batch_size,
		classes_per_task=args.classes_per_task,
		num_workers=args.num_workers,
		seed=args.seed,
		device=device,
	)

	total_classes = stream.num_classes
	num_init = min(args.classes_per_task, total_classes)

	client_cfg = ClientConfig(
		local_epochs=args.local_epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		batch_size=args.batch_size,
		replay_ratio=args.replay_ratio,
		max_grad_norm=args.max_grad_norm,
	)
	server_cfg = ServerConfig(
		global_rounds=args.global_rounds,
		clients_per_round=args.clients_per_round,
		server_opt_steps=args.server_opt_steps,
		server_opt_lr=args.server_opt_lr,
		server_replay_batch=args.server_replay_batch,
	)

	gan_sample_transform = make_gan_sample_transform(args.img_size)

	algo = ExampleFedAvgWithGANReplay(
		num_clients=args.num_clients,
		num_init_classes=num_init,
		total_num_classes=total_classes,
		client_cfg=client_cfg,
		server_cfg=server_cfg,
		device=device,
		sample_transform=gan_sample_transform,
	)

	total_rounds = len(stream) * server_cfg.global_rounds
	log_configuration(
		args=args,
		client_cfg=client_cfg,
		server_cfg=server_cfg,
		device=device,
		log_dir=log_dir,
		total_tasks=len(stream),
		total_rounds=total_rounds,
		total_classes=total_classes,
	)

	writer = SummaryWriter(str(log_dir))
	pbar = tqdm(total=total_rounds, dynamic_ncols=True)
	start_time = time.time()
	global_step = 0
	seen_classes: List[int] = []

	def round_logger(task: TaskInfo, round_idx: int, updates, metrics):
		nonlocal global_step, seen_classes
		if round_idx == 0:
			for c in task.new_classes:
				if c not in seen_classes:
					seen_classes.append(c)

		acc = None
		if (round_idx + 1) % args.eval_every == 0 or (round_idx + 1) == server_cfg.global_rounds:
			acc = evaluate(algo.server.model, stream.eval_loader(task.task_id), device)

		completed = pbar.n + 1
		elapsed = time.time() - start_time
		eta = (elapsed / completed) * (total_rounds - completed)
		pbar.set_description(
			f"Task {task.task_id + 1}/{len(stream)} | Global Round {round_idx + 1}/{server_cfg.global_rounds}"
		)
		postfix = {
			"test acc": f"{acc:.3f}" if acc is not None else "-",
			"CE loss": f"{metrics.get('loss_ce', 0.0):.4f}",
			"GAN D": f"{metrics.get('gan_loss_d', 0.0):.4f}",
			"GAN G": f"{metrics.get('gan_loss_g', 0.0):.4f}",
			"elapsed": format_seconds(elapsed),
			"eta": format_seconds(eta),
		}
		pbar.set_postfix(postfix)
		pbar.update(1)

		step = global_step
		for k, v in metrics.items():
			writer.add_scalar(f"train/{k}", v, step)
		if acc is not None:
			writer.add_scalar("eval/accuracy", acc, step)
		writer.add_scalar("time/eta_sec", eta, step)
		writer.add_scalar("time/elapsed_sec", elapsed, step)
		global_step += 1

	algo.run(stream, round_hook=round_logger)

	pbar.close()
	writer.close()
	total_time = time.time() - start_time
	print(f"Finished training in {format_seconds(total_time)}. Logs at {log_dir}")


if __name__ == "__main__":
	main()
