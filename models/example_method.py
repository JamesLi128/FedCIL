# ==================================
# --- file: fcil/example_method.py
# ==================================
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fed_base import (
    BaseClient, BaseServer, BaseFCILAlgorithm,
    ClientConfig, ServerConfig, TaskInfo,
    ClientUpdate, IncrementalNet,
)
from replay_gan import ClientGANReplay
from config import GANReplayConfig
from copy import deepcopy


def build_frozen_pretrained_resnet18_backbone() -> tuple[nn.Module, int]:
    """
    Returns (backbone, feat_dim). The backbone outputs a feature vector.
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        from torchvision.models import resnet18
        m = resnet18(pretrained=True)

    feat_dim = m.fc.in_features
    m.fc = nn.Identity()

    # freeze all parameters except fc
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True

    return m, feat_dim


class FedAvgGANClient(BaseClient):
    """
    Example client:
      - trains incremental classifier on (real + replay) batches
      - trains local GAN on real data (simple baseline)
    """
    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: IncrementalNet,
        replay: Optional[ClientGANReplay],
        cfg: ClientConfig,
    ) -> None:
        super().__init__(client_id, device, model, replay, cfg)

    def fit_one_task(self, task: TaskInfo, train_loader) -> ClientUpdate:
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        total_samples = 0
        metrics = {"loss_ce": 0.0, "steps": 0.0, "gan_loss_d": 0.0, "gan_loss_g": 0.0}

        for _ in range(self.cfg.local_epochs):
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                bsz = x.size(0)
                total_samples += bsz

                # --- client-side replay (from global/local generator)
                if self.replay is not None and self.known_classes:
                    n_rep = int(bsz * self.cfg.replay_ratio)
                    if n_rep > 0:
                        x_rep, y_rep = self.replay.sample(n_rep, self.device)
                        x_mix = torch.cat([x, x_rep], dim=0)
                        y_mix = torch.cat([y, y_rep], dim=0)
                    else:
                        x_mix, y_mix = x, y
                else:
                    x_mix, y_mix = x, y

                logits = self.model(x_mix)
                loss = F.cross_entropy(logits, y_mix)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                opt.step()

                metrics["loss_ce"] += float(loss.detach().cpu())
                metrics["steps"] += 1.0

                # --- train GAN on REAL batch (simple baseline)
                if self.replay is not None:
                    gan_m = self.replay.train_gan_on_batch(x_real=x, y_real=y)
                    metrics["gan_loss_d"] += gan_m["loss_d"]
                    metrics["gan_loss_g"] += gan_m["loss_g"]

        # average metrics
        if metrics["steps"] > 0:
            metrics["loss_ce"] /= metrics["steps"]
            metrics["gan_loss_d"] /= metrics["steps"]
            metrics["gan_loss_g"] /= metrics["steps"]

        upd = ClientUpdate(
            client_id=self.client_id,
            num_samples=total_samples,
            model_state={k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            gen_state={k: v.detach().cpu() for k, v in self.replay.state_dict().items()} if self.replay is not None else None,
            metrics=metrics,
        )
        return upd


class FedAvgGANServer(BaseServer):
    """
    Example server:
      - FedAvg model aggregation (in BaseServer.aggregate)
      - FedAvg generator aggregation (generator only)
      - optional server-side optimization using global generator (BaseServer.server_optimize_step)
    """
    pass


class ExampleFedAvgWithGANReplay(BaseFCILAlgorithm):
    """
    Example "method" class that:
      - expands the classifier when new classes arrive
      - updates server/client known_classes
      - runs FedAvg + replay hooks
    """
    def __init__(
        self,
        num_clients: int,
        num_init_classes: int,
        total_num_classes: int,
        client_cfg: ClientConfig,
        server_cfg: ServerConfig,
        device: torch.device,
        sample_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        backbone, feat_dim = build_frozen_pretrained_resnet18_backbone()
        model = IncrementalNet(backbone, feat_dim, num_init_classes, classification_head_type=client_cfg.classification_head_type, hidden_dim=client_cfg.hidden_dim)

        # Global generator on server
        gan_cfg = GANReplayConfig(num_total_classes=total_num_classes, gan_lr=client_cfg.gan_lr, gan_weight_decay=client_cfg.gan_weight_decay)
        server_replay = ClientGANReplay(gan_cfg, device=device, sample_transform=sample_transform)

        server = FedAvgGANServer(model=model, replay=server_replay, cfg=server_cfg, device=device)

        clients: Dict[int, FedAvgGANClient] = {}
        for cid in range(num_clients):
            # each client gets its own model copy and its own replay module
            c_model = deepcopy(model).to(device)
            c_replay = ClientGANReplay(gan_cfg, device=device, sample_transform=sample_transform)
            clients[cid] = FedAvgGANClient(
                client_id=cid,
                device=device,
                model=c_model,
                replay=c_replay,
                cfg=client_cfg,
            )

        super().__init__(server=server, clients=clients)

        self.total_num_classes = total_num_classes
        self.current_num_classes = num_init_classes
        self.server.set_known_classes(list(range(num_init_classes)))

    def _on_new_task(self, task: TaskInfo) -> None:
        # Expand classifier heads if new classes exceed current size
        if task.new_classes:
            max_new = max(task.new_classes)
            needed = (max_new + 1) - self.current_num_classes
            if needed > 0:
                # expand server model
                self.server.model.expand_classes(needed)
                # expand client models to match
                for c in self.clients.values():
                    c.model.expand_classes(needed)
                self.current_num_classes += needed

        # Update known classes on server (+ clients will get it via payload)
        known = sorted(set(self.server.known_classes).union(task.new_classes))
        self.server.set_known_classes(known)
