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
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils import CIFARTaskStream
from config import ClientConfig, ServerConfig, TaskInfo
from example_method import ExampleFedAvgWithGANReplay


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
	parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration for client splits")
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


def format_seconds(sec: float) -> str:
	sec = max(sec, 0.0)
	m, s = divmod(int(sec), 60)
	h, m = divmod(m, 60)
	return f"{h:02d}:{m:02d}:{s:02d}"


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
		alpha=args.alpha,
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

	gan_sample_transform = stream.gan_sample_transform()

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
