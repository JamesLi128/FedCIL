"""
Minimal runnable entrypoint for ExampleFedAvgWithGANReplay.

Features:
- Hydra/OmegaConf configuration management.
- Progress bar with ETA + per-round logging.
- TensorBoard logging for loss/accuracy (real-time if tensorboard is running).
- Supports CIFAR-10/100 stored under ~/data with upsampled inputs for ResNet18.
- Supports multi-run sweeps via Hydra for hyperparameter search within single SLURM allocation.

Usage:
    # Single run with defaults
    python run.py --config-name=cifar10

    # Override parameters
    python run.py --config-name=cifar10 training.lr=0.01 federated.num_clients=10

    # Multi-run sweep (useful for sbatch)
    python run.py --config-name=cifar10 -m training.lr=0.001,0.01,0.1 federated.alpha=0.1,1.0
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

# Ensure CUDA device ordering is stable before importing torch
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Get logger for this module (Hydra will configure it)
log = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from data_utils import CIFARTaskStream
from config import ClientConfig, ServerConfig, TaskInfo
from example_method import ExampleFedAvgWithGANReplay


# -------------------------
# Helpers
# -------------------------
def log_section(title: str, items: Dict[str, object]) -> None:
	log.info(f"[{title}]")
	for key in sorted(items.keys()):
		log.info(f"  {key:20}: {items[key]}")


def log_configuration(
	cfg: DictConfig,
	device: torch.device,
	log_dir: Path,
	total_tasks: int,
	total_rounds: int,
	total_classes: int,
) -> None:
	log.info("=" * 60)
	log.info("Initialized FCIL run with: double-check TensorBoard for live metrics.")
	log.info("[Configuration]")
	for line in OmegaConf.to_yaml(cfg).split('\n'):
		log.info(f"  {line}")
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
	log.info("=" * 60)


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


@hydra.main(version_base=None, config_path="../config", config_name="cifar10")
def main(cfg: DictConfig) -> None:
	# Resolve interpolations for logging
	OmegaConf.resolve(cfg)
	
	set_seed(cfg.system.seed)

	cache_dir = Path(cfg.system.cache_dir).expanduser()
	cache_dir.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
	os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

	# Hydra sets working directory; use hydra.runtime.output_dir for logs
	log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
	log_dir.mkdir(parents=True, exist_ok=True)

	device = torch.device("cpu" if cfg.system.cpu or not torch.cuda.is_available() else "cuda")
	torch.backends.cudnn.benchmark = True

	data_root = str(Path(cfg.dataset.data_root).expanduser())
	stream = CIFARTaskStream(
		dataset=cfg.dataset.name,
		data_root=data_root,
		img_size=cfg.dataset.img_size,
		num_clients=cfg.federated.num_clients,
		alpha=cfg.federated.alpha,
		batch_size=cfg.training.batch_size,
		eval_batch_size=cfg.training.eval_batch_size,
		classes_per_task=cfg.dataset.classes_per_task,
		num_workers=cfg.system.num_workers,
		seed=cfg.system.seed,
		device=device,
	)

	total_classes = stream.num_classes
	num_init = min(cfg.dataset.classes_per_task, total_classes)

	client_cfg = ClientConfig(
		local_epochs=cfg.training.local_epochs,
		lr=cfg.training.lr,
		weight_decay=cfg.training.weight_decay,
		batch_size=cfg.training.batch_size,
		replay_ratio=cfg.replay.ratio,
		max_grad_norm=cfg.training.max_grad_norm,
		classification_head_type=cfg.model.classification_head_type,
		hidden_dim=cfg.model.hidden_dim
	)
	server_cfg = ServerConfig(
		global_rounds=cfg.training.global_rounds,
		clients_per_round=cfg.federated.clients_per_round,
		server_opt_steps=cfg.server.opt_steps,
		server_opt_lr=cfg.server.opt_lr,
		server_replay_batch=cfg.server.replay_batch,
	)

	gan_sample_transform = stream.gan_sample_transform()

	algo = ExampleFedAvgWithGANReplay(
		num_clients=cfg.federated.num_clients,
		num_init_classes=num_init,
		total_num_classes=total_classes,
		client_cfg=client_cfg,
		server_cfg=server_cfg,
		device=device,
		sample_transform=gan_sample_transform,
	)

	total_rounds = len(stream) * server_cfg.global_rounds
	log_configuration(
		cfg=cfg,
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
	
	# Store config values needed in closure
	eval_every = cfg.logging.eval_every
	max_concurrent_clients = cfg.system.max_concurrent_clients

	def round_logger(task: TaskInfo, round_idx: int, updates, metrics):
		nonlocal global_step, seen_classes
		if round_idx == 0:
			for c in task.new_classes:
				if c not in seen_classes:
					seen_classes.append(c)

		acc = None
		if (round_idx + 1) % eval_every == 0 or (round_idx + 1) == server_cfg.global_rounds:
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
			writer.add_scalar(f"task_{task.task_id + 1}/eval_accuracy", acc, step)
		if "loss_ce" in metrics:
			writer.add_scalar("train/loss_ce", metrics["loss_ce"], step)
		if "gan_loss_d" in metrics:
			writer.add_scalar("train/gan_loss_d", metrics["gan_loss_d"], step)
		if "gan_loss_g" in metrics:
			writer.add_scalar("train/gan_loss_g", metrics["gan_loss_g"], step)
		writer.add_scalar("time/eta_sec", eta, step)
		writer.add_scalar("time/elapsed_sec", elapsed, step)
		
		# Log to file as well
		acc_str = f"{acc:.3f}" if acc is not None else "-"
		# Create text-based progress bar
		bar_width = 20
		progress = completed / total_rounds
		filled = int(bar_width * progress)
		bar = "█" * filled + "░" * (bar_width - filled)
		pct = progress * 100
		log.info(
			f"[{bar}] {pct:5.1f}% | Task {task.task_id + 1}/{len(stream)} | Round {round_idx + 1}/{server_cfg.global_rounds} | "
			f"Acc: {acc_str} | CE: {metrics.get('loss_ce', 0.0):.4f} | "
			f"GAN_D: {metrics.get('gan_loss_d', 0.0):.4f} | GAN_G: {metrics.get('gan_loss_g', 0.0):.4f} | "
			f"Elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)}"
		)
		global_step += 1

	algo.run_concurrent(
		stream=stream,
		max_concurrent_clients=max_concurrent_clients,
		round_hook=round_logger,
	)

	pbar.close()
	writer.close()
	total_time = time.time() - start_time
	log.info(f"Finished training in {format_seconds(total_time)}. Logs at {log_dir}")
	
	# Save final config to output directory
	config_path = log_dir / "config.yaml"
	OmegaConf.save(cfg, config_path)
	log.info(f"Configuration saved to {config_path}")


if __name__ == "__main__":
	main()
