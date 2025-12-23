# ==================================
# --- file: fcil/acgan_method.py
# ==================================
"""
ACGAN-based Federated Class-Incremental Learning.

This module provides:
- ACGANClient: A client that uses ACGAN for both classification and generation
- ACGANServer: Server for aggregating ACGAN models
- FedAvgWithACGAN: Algorithm orchestrator for ACGAN-based FL
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fed_base import (
    BaseClient, BaseServer, BaseFCILAlgorithm, ReplayStrategy,
    ClientConfig, ServerConfig, TaskInfo,
    ClientUpdate, ServerPayload, fedavg, _to_cpu_state,
)
from acgan import IncrementalACGAN
from config import ACGANConfig


class ACGANReplayStrategy(ReplayStrategy):
    """
    Replay strategy that wraps an IncrementalACGAN for generating synthetic samples.
    
    This allows the ACGAN to be used with the existing replay infrastructure.
    Since the ACGAN now uses dataset-aware architectures with no resizing needed,
    no sample transform is required.
    """
    def __init__(
        self,
        acgan: IncrementalACGAN,
    ):
        self.acgan = acgan
        self._known_classes: List[int] = []
    
    def set_known_classes(self, known: List[int]) -> None:
        self._known_classes = list(known)
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return generator state for federated aggregation."""
        return self.acgan.generator_state_dict()
    
    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load generator state."""
        self.acgan.load_generator_state_dict(state)
    
    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample synthetic images from known classes."""
        if not self._known_classes:
            # Fallback: uniform over all classes
            labels = torch.randint(0, self.acgan.num_classes, (n,), device=device)
        else:
            idx = torch.randint(0, len(self._known_classes), (n,), device=device)
            labels = torch.tensor(self._known_classes, device=device, dtype=torch.long)[idx]
        
        x, y = self.acgan.sample(n, labels)
        # Generator output is already in [-1, 1], matching training data format
        return x, y


class ACGANClient(BaseClient):
    """
    Client that uses ACGAN for both classification and image generation.
    
    Key differences from FedAvgGANClient:
    - Uses ACGAN's discriminator for classification (no separate IncrementalNet)
    - Single model handles both tasks, simplifying the architecture
    - Generator and discriminator are jointly trained with auxiliary classification loss
    - Uses dataset-aware custom conv backbones (no pretrained ResNet)
    
    The model attribute is an IncrementalACGAN instead of IncrementalNet.
    """
    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: IncrementalACGAN,  # Note: model is now IncrementalACGAN
        cfg: ClientConfig,
    ) -> None:
        # We don't call super().__init__ because model type is different
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        self.cfg = cfg
        self.prev_known_classes: List[int] = []
        self.known_classes: List[int] = []
        
        # Create replay wrapper for consistency with other code
        self.replay = ACGANReplayStrategy(model)
    
    def set_known_classes(self, known: List[int]) -> None:
        self.prev_known_classes = self.known_classes
        self.known_classes = list(known)
        self.replay.set_known_classes(known)
    
    def load_server_payload(self, payload: ACGANServerPayload) -> None:
        """Load state from server payload."""
        self.set_known_classes(payload.known_classes)
        
        # Load generator state
        if payload.global_gen_state is not None:
            gen_state = {k: v.to(self.device) for k, v in payload.global_gen_state.items()}
            self.model.load_generator_state_dict(gen_state)
        
        # Load discriminator state
        if payload.global_dis_state is not None:
            dis_state = {k: v.to(self.device) for k, v in payload.global_dis_state.items()}
            self.model.load_discriminator_state_dict(dis_state)
    
    def fit_one_task(self, task: TaskInfo, train_loader, prev_global_model: Optional[IncrementalACGAN] = None) -> ACGANClientUpdate:
        """
        Train ACGAN on local data for one task.
        
        The ACGAN is trained with:
        - Discriminator loss: real/fake + auxiliary classification
        - Generator loss: fool discriminator + correct auxiliary classification
        
        Replay samples from the generator are mixed in for rehearsal.
        """
        self.model.G.train()
        self.model.D.train()
        
        total_samples = 0
        metrics = {
            "loss_ce": 0.0,  # Auxiliary classification loss
            "loss_d": 0.0,
            "loss_g": 0.0,
            "steps": 0.0,
        }
        
        for _ in range(self.cfg.local_epochs):
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                bsz = x.size(0)
                total_samples += bsz
                
                # Mix in replay samples for rehearsal
                if self.known_classes and self.cfg.replay_ratio > 0:
                    n_rep = int(bsz * self.cfg.replay_ratio)
                else:
                    n_rep = 0
                
                # Train ACGAN
                step_metrics = self.model.train_step(x, y, n_replay=n_rep, sample_classes=self.prev_known_classes)

                # Distillation from previous global model if provided
                if prev_global_model is not None:
                    self.model.KL_distill_from(prev_global_model, x, y)
                
                metrics["loss_d"] += step_metrics.get("loss_d", 0.0)
                metrics["loss_g"] += step_metrics.get("loss_g", 0.0)
                metrics["loss_ce"] += step_metrics.get("loss_aux", 0.0)
                metrics["steps"] += 1.0
        
        # Average metrics
        if metrics["steps"] > 0:
            metrics["loss_d"] /= metrics["steps"]
            metrics["loss_g"] /= metrics["steps"]
            metrics["loss_ce"] /= metrics["steps"]
        
        # Rename for consistency with other clients
        metrics["gan_loss_d"] = metrics.pop("loss_d")
        metrics["gan_loss_g"] = metrics.pop("loss_g")
        
        return ACGANClientUpdate(
            client_id=self.client_id,
            num_samples=total_samples,
            gen_state={k: v.detach().cpu() for k, v in self.model.generator_state_dict().items()},
            dis_state={k: v.detach().cpu() for k, v in self.model.discriminator_state_dict().items()},
            metrics=metrics,
        )


class ACGANClientUpdate:
    """Update from an ACGAN client."""
    def __init__(
        self,
        client_id: int,
        num_samples: int,
        gen_state: Dict[str, torch.Tensor],
        dis_state: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
    ):
        self.client_id = client_id
        self.num_samples = num_samples
        self.gen_state = gen_state
        self.dis_state = dis_state
        self.metrics = metrics
        
        # For compatibility with aggregate_client_metrics
        self.model_state = dis_state  # discriminator is the "model" for classification


class ACGANServerPayload:
    """Payload sent from server to clients."""
    def __init__(
        self,
        global_gen_state: Optional[Dict[str, torch.Tensor]],
        global_dis_state: Optional[Dict[str, torch.Tensor]],
        known_classes: List[int],
    ):
        self.global_gen_state = global_gen_state
        self.global_dis_state = global_dis_state
        self.known_classes = known_classes


class ACGANServer(BaseServer):
    """
    Server for ACGAN-based federated learning.
    
    Aggregates both generator and discriminator from clients.
    The server model is an IncrementalACGAN.
    """
    def __init__(
        self,
        model: IncrementalACGAN,
        cfg: ServerConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.cfg = cfg
        self.known_classes: List[int] = []
        self._round: int = 0

        # Store previous model state for distillation
        # self.prev_model = None
        # Store previous known classes
        # self.prev_known_classes: List[int] = []
        
        # Replay wrapper for server-side optimization
        self.replay = ACGANReplayStrategy(model)
    
    def set_known_classes(self, known: List[int]) -> None:
        self.known_classes = list(known)
        self.replay.set_known_classes(known)
    
    def broadcast(self) -> ACGANServerPayload:
        """Create payload to send to clients."""
        return ACGANServerPayload(
            global_gen_state=_to_cpu_state(self.model.generator_state_dict()),
            global_dis_state=_to_cpu_state(self.model.discriminator_state_dict()),
            known_classes=list(self.known_classes),
        )
    
    def aggregate(self, updates: List[ACGANClientUpdate]) -> None:
        """Aggregate generator and discriminator states using FedAvg."""
        if not updates:
            return
        
        # Aggregate generator
        gen_states = [(_to_cpu_state(u.gen_state), u.num_samples) for u in updates]
        new_gen = fedavg(gen_states)
        new_gen = {k: v.to(self.device) for k, v in new_gen.items()}
        self.model.load_generator_state_dict(new_gen)
        
        # Aggregate discriminator
        dis_states = [(_to_cpu_state(u.dis_state), u.num_samples) for u in updates]
        new_dis = fedavg(dis_states)
        new_dis = {k: v.to(self.device) for k, v in new_dis.items()}
        self.model.load_discriminator_state_dict(new_dis)
        
    def server_optimize_step(self) -> None:
        """
        Optional server-side optimization using synthetic samples.
        
        This trains the discriminator's auxiliary classifier on generated samples
        to potentially improve class boundaries.
        """
        if self.cfg.server_opt_steps <= 0 or not self.known_classes:
            return
        
        self.model.D.train()
        # Optimize all discriminator parameters (no frozen backbone anymore)
        opt = torch.optim.Adam(self.model.D.parameters(), lr=self.cfg.server_opt_lr)
        
        for _ in range(self.cfg.server_opt_steps):
            x_syn, y_syn = self.model.sample(self.cfg.server_replay_batch)
            # No transform needed - generator output is already in correct format
            _, logits = self.model.D(x_syn)
            loss = F.cross_entropy(logits, y_syn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


class FedAvgWithACGAN(BaseFCILAlgorithm):
    """
    Federated Class-Incremental Learning using ACGAN.
    
    This algorithm:
    - Uses ACGAN for both classification and replay generation
    - Both generator and discriminator use dataset-aware custom conv backbones
    - Works directly with original dataset size (e.g., 28x28 for MNIST, 32x32 for CIFAR)
    - Expands the ACGAN when new classes arrive
    - Aggregates both generator and discriminator across clients
    """
    def __init__(
        self,
        num_clients: int,
        num_init_classes: int,
        total_num_classes: int,
        client_cfg: ClientConfig,
        server_cfg: ServerConfig,
        device: torch.device,
        # ACGAN-specific configuration
        z_dim: int = 100,
        dataset_name: str = "mnist",
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
    ) -> None:
        # Create the global ACGAN model
        model = IncrementalACGAN(
            z_dim=z_dim,
            num_classes=num_init_classes,
            dataset_name=dataset_name,
            g_lr=g_lr,
            d_lr=d_lr,
        )
        
        # Create server
        server = ACGANServer(model=model, cfg=server_cfg, device=device)
        
        # Create clients
        clients: Dict[int, ACGANClient] = {}
        for cid in range(num_clients):
            c_model = deepcopy(model).to(device)
            # Reset optimizers for each client
            c_model.g_optimizer = torch.optim.Adam(
                c_model.G.parameters(), lr=g_lr, betas=(0.5, 0.999)
            )
            c_model.d_optimizer = torch.optim.Adam(
                c_model.D.parameters(), lr=d_lr, betas=(0.5, 0.999)
            )
            clients[cid] = ACGANClient(
                client_id=cid,
                device=device,
                model=c_model,
                cfg=client_cfg,
            )
        
        # Initialize base class
        # Note: We're using ACGANServer which has a different interface,
        # so we need to handle this carefully
        self.server = server
        self.clients = clients
        
        self.total_num_classes = total_num_classes
        self.current_num_classes = num_init_classes
        self.server.set_known_classes(list(range(num_init_classes)))
    
    def _on_new_task(self, task: TaskInfo) -> None:
        """Handle new task: expand classifier heads if needed."""
        if task.new_classes:
            max_new = max(task.new_classes)
            needed = (max_new + 1) - self.current_num_classes
            if needed > 0:
                # Expand server model
                self.server.model.expand_classes(needed)
                # Expand client models
                for c in self.clients.values():
                    c.model.expand_classes(needed)
                self.current_num_classes += needed
        
        # Update known classes
        known = sorted(set(self.server.known_classes).union(task.new_classes))
        self.server.set_known_classes(known)
    
    def run(
        self,
        stream,
        round_hook: Optional[Callable] = None,
    ) -> None:
        """Run federated training."""
        prev_global_model = None
        for task, per_client_loaders in stream:
            self._on_new_task(task)
            
            for r in range(self.server.cfg.global_rounds):
                payload = self.server.broadcast()
                
                available = sorted(per_client_loaders.keys())
                chosen = available[:min(self.server.cfg.clients_per_round, len(available))]
                
                updates: List[ACGANClientUpdate] = []
                for cid in chosen:
                    c = self.clients[cid]
                    c.load_server_payload(payload)
                    upd = c.fit_one_task(task, per_client_loaders[cid], prev_global_model=prev_global_model)
                    updates.append(upd)
                
                self.server.aggregate(updates)
                self.server.server_optimize_step()

                with torch.no_grad():
                    self.prev_global_model = deepcopy(self.server.model)
                    for param in self.prev_global_model.parameters():
                        param.requires_grad = False
                    self.prev_global_model.eval()
                
                if round_hook is not None:
                    # Convert to standard ClientUpdate for metrics
                    metrics = {}
                    total_samples = sum(u.num_samples for u in updates)
                    if total_samples > 0:
                        for key in ["loss_ce", "gan_loss_d", "gan_loss_g"]:
                            val = sum(u.metrics.get(key, 0) * u.num_samples for u in updates if u.metrics)
                            metrics[key] = val / total_samples
                    round_hook(task, r, updates, metrics)
    
    def run_concurrent(
        self,
        stream,
        round_hook: Optional[Callable] = None,
        max_concurrent_clients: Optional[int] = None,
    ) -> None:
        """Run with concurrent client training (sequential for now)."""
        # For ACGAN, we use sequential training
        self.run(stream, round_hook)
