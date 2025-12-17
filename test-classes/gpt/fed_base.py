# =========================
# --- file: fcil/fed_base.py
# =========================
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn

from config import *



def _to_cpu_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state.items()}


def fedavg(states: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """
    FedAvg over a list of (state_dict, num_samples).
    Assumes all state_dicts share the same keys.
    """
    if not states:
        raise ValueError("fedavg got empty states")

    keys = list(states[0][0].keys())
    total = sum(n for _, n in states)
    if total <= 0:
        raise ValueError("Total samples must be > 0")

    avg: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for sd, n in states:
            w = (n / total)
            term = sd[k].float() * w
            acc = term if acc is None else (acc + term)
        avg[k] = acc
    return avg


def aggregate_client_metrics(updates: List[ClientUpdate]) -> Dict[str, float]:
    """
    Weighted average of metrics using client sample counts. Skips missing metrics.
    """
    agg: Dict[str, float] = {}
    weight: Dict[str, float] = {}

    for upd in updates:
        if upd.metrics is None:
            continue
        for k, v in upd.metrics.items():
            agg[k] = agg.get(k, 0.0) + v * upd.num_samples
            weight[k] = weight.get(k, 0.0) + upd.num_samples

    for k, w in weight.items():
        agg[k] /= max(w, 1e-9)
    return agg


# -------------------------
# Incremental model base
# -------------------------
class IncrementalClassifier(nn.Module):
    """
    Minimal expandable linear head for class-incremental learning.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def expand(self, num_new_classes: int) -> None:
        if num_new_classes <= 0:
            return
        old_fc: nn.Linear = self.fc
        new_num = self.num_classes + num_new_classes
        new_fc = nn.Linear(self.in_features, new_num)
        
        # Move new layer to same device as old layer
        new_fc = new_fc.to(old_fc.weight.device)

        # copy old weights/bias
        with torch.no_grad():
            new_fc.weight[: self.num_classes].copy_(old_fc.weight)
            new_fc.bias[: self.num_classes].copy_(old_fc.bias)

        self.num_classes = new_num
        self.fc = new_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class IncrementalNet(nn.Module):
    """
    Backbone + (expandable) classifier head.
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = IncrementalClassifier(feat_dim, num_classes)

    def expand_classes(self, num_new: int) -> None:
        self.classifier.expand(num_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return self.classifier(feat)


# -------------------------
# Replay strategy interfaces
# -------------------------
class ReplayStrategy(ABC):
    @abstractmethod
    def set_known_classes(self, known: List[int]) -> None:
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        ...

    @abstractmethod
    def sample(self, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (x_synth, y_synth) for replay.
        """
        ...


# -------------------------
# Client / Server base
# -------------------------
class TaskStream(Protocol):
    """
    Minimal protocol: your code should yield tasks; each task provides
    per-client dataloaders for the *current* task data.
    """
    def __iter__(self) -> Iterable[Tuple[TaskInfo, Dict[int, torch.utils.data.DataLoader]]]:
        ...


class BaseClient(ABC):
    def __init__(
        self,
        client_id: int,
        device: torch.device,
        model: IncrementalNet,
        replay: Optional[ReplayStrategy],
        cfg: ClientConfig,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        self.replay = replay
        self.cfg = cfg
        self.known_classes: List[int] = []

    def set_known_classes(self, known: List[int]) -> None:
        self.known_classes = list(known)
        if self.replay is not None:
            self.replay.set_known_classes(self.known_classes)

    def load_server_payload(self, payload: ServerPayload) -> None:
        self.set_known_classes(payload.known_classes)
        # Move payload state to device before loading
        model_state = {k: v.to(self.device) for k, v in payload.global_model_state.items()}
        self.model.load_state_dict(model_state, strict=True)
        if self.replay is not None and payload.global_gen_state is not None:
            gen_state = {k: v.to(self.device) for k, v in payload.global_gen_state.items()}
            self.replay.load_state_dict(gen_state)

    @abstractmethod
    def fit_one_task(
        self,
        task: TaskInfo,
        train_loader: torch.utils.data.DataLoader,
    ) -> ClientUpdate:
        ...


class BaseServer(ABC):
    def __init__(
        self,
        model: IncrementalNet,
        replay: Optional[ReplayStrategy],
        cfg: ServerConfig,
        device: torch.device,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.replay = replay
        self.cfg = cfg

        self.known_classes: List[int] = []
        self._round: int = 0

    def set_known_classes(self, known: List[int]) -> None:
        self.known_classes = list(known)
        if self.replay is not None:
            self.replay.set_known_classes(self.known_classes)

    def broadcast(self) -> ServerPayload:
        gen_state = self.replay.state_dict() if self.replay is not None else None
        return ServerPayload(
            global_model_state=_to_cpu_state(self.model.state_dict()),
            global_gen_state=_to_cpu_state(gen_state) if gen_state is not None else None,
            known_classes=list(self.known_classes),
        )

    def aggregate(self, updates: List[ClientUpdate]) -> None:
        # FedAvg model
        model_states = [(_to_cpu_state(u.model_state), u.num_samples) for u in updates]
        new_model = fedavg(model_states)
        # Move aggregated state to device before loading
        new_model = {k: v.to(self.device) for k, v in new_model.items()}
        self.model.load_state_dict(new_model, strict=True)

        # FedAvg generator (optional)
        if self.replay is not None:
            gen_updates = [(u.gen_state, u.num_samples) for u in updates if u.gen_state is not None]
            if gen_updates:
                gen_states = [(_to_cpu_state(sd), n) for sd, n in gen_updates]  # type: ignore[arg-type]
                new_gen = fedavg(gen_states)
                # Move aggregated generator state to device before loading
                new_gen = {k: v.to(self.device) for k, v in new_gen.items()}
                self.replay.load_state_dict(new_gen)

    def server_optimize_step(self) -> None:
        """
        Optional: server-side "optimization" using server-side replay samples
        from the (global) generator. Keeps things simple + explicit.
        """
        if self.replay is None or self.cfg.server_opt_steps <= 0 or not self.known_classes:
            return

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.server_opt_lr)

        for _ in range(self.cfg.server_opt_steps):
            x_syn, y_syn = self.replay.sample(self.cfg.server_replay_batch, self.device)
            logits = self.model(x_syn)
            loss = F.cross_entropy(logits, y_syn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


class BaseFCILAlgorithm(ABC):
    """
    Orchestrates: for each task -> for each FL round -> broadcast/train/aggregate
    """
    def __init__(self, server: BaseServer, clients: Dict[int, BaseClient]):
        self.server = server
        self.clients = clients

    def run(
        self,
        stream: TaskStream,
        round_hook: Optional[
            Callable[[TaskInfo, int, List[ClientUpdate], Dict[str, float]], None]
        ] = None,
    ) -> None:
        for task, per_client_loaders in stream:
            # Update known classes, and expand heads if needed.
            self._on_new_task(task)

            for r in range(self.server.cfg.global_rounds):
                payload = self.server.broadcast()

                # sample clients (simple: intersect with available loaders)
                available = sorted(per_client_loaders.keys())
                chosen = available[: min(self.server.cfg.clients_per_round, len(available))]

                updates: List[ClientUpdate] = []
                for cid in chosen:
                    c = self.clients[cid]
                    c.load_server_payload(payload)
                    upd = c.fit_one_task(task, per_client_loaders[cid])
                    updates.append(upd)

                self.server.aggregate(updates)
                self.server.server_optimize_step()

                if round_hook is not None:
                    metrics = aggregate_client_metrics(updates)
                    round_hook(task, r, updates, metrics)

    def run_concurrent(
        self,
        stream: TaskStream,
        round_hook: Optional[Callable] = None,
        max_concurrent_clients: Optional[int] = None,
    ) -> None:
        # Pre-allocate distinct CUDA streams if on GPU
        # Note: Use a pool of streams roughly equal to max_workers
        self.streams = [torch.cuda.Stream() for _ in range(max_concurrent_clients or 4)]

        for task, per_client_loaders in stream:
            self._on_new_task(task)

            for r in range(self.server.cfg.global_rounds):
                payload = self.server.broadcast()

                available = sorted(per_client_loaders.keys())
                chosen = available[: min(self.server.cfg.clients_per_round, len(available))]
                if not chosen:
                    continue

                # Define the worker function with Stream handling
                def _train_client(cid_idx_tuple):
                    idx, cid = cid_idx_tuple
                    c = self.clients[cid]
                    
                    # Assign a stream based on thread index to avoid creating new streams constantly
                    # or just create a new one: s = torch.cuda.Stream()
                    s = self.streams[idx % len(self.streams)]
                    
                    with torch.cuda.stream(s):
                        c.load_server_payload(payload)
                        update = c.fit_one_task(task, per_client_loaders[cid])
                    
                    # Synchronization is crucial before returning results to main thread
                    s.synchronize() 
                    return idx, update

                max_workers = len(chosen) if max_concurrent_clients is None else max(
                    1, min(max_concurrent_clients, len(chosen))
                )

                updates: List[ClientUpdate] = [None] * len(chosen) # Pre-allocate to preserve order
                
                if max_workers == 1:
                    # Sequential path (no stream overhead needed usually, but safe to keep)
                    for i, cid in enumerate(chosen):
                        _, upd = _train_client((i, cid))
                        updates[i] = upd
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        # Pass index to help manage streams and sort results
                        args = [(i, cid) for i, cid in enumerate(chosen)]
                        results = pool.map(_train_client, args)
                        
                        # pool.map preserves order, so we can just listify
                        for idx, upd in results:
                            updates[idx] = upd

                self.server.aggregate(updates)
                self.server.server_optimize_step()

                if round_hook is not None:
                    metrics = aggregate_client_metrics(updates)
                    round_hook(task, r, updates, metrics)

    def run_multi_gpu(
        self,
        stream: TaskStream,
        round_hook: Optional[
            Callable[[TaskInfo, int, List[ClientUpdate], Dict[str, float]], None]
        ] = None,
        num_gpus: Optional[int] = None,
        backend: str = "nccl",
    ) -> None:
        """
        Multi-GPU federated training using DistributedDataParallel.
        
        Args:
            stream: TaskStream containing tasks and per-client loaders
            round_hook: Optional callback for each round
            num_gpus: Number of GPUs to use. If None, uses all available GPUs
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU/GPU)
        """
        
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        
        if num_gpus <= 1:
            # Fallback to single-GPU training
            return self.run(stream, round_hook)
        
        # Spawn processes for each GPU
        spawn(self._run_multi_gpu_worker, args=(stream, round_hook, num_gpus, backend), nprocs=num_gpus)


    def _run_multi_gpu_worker(
        self,
        rank: int,
        stream: TaskStream,
        round_hook: Optional[
            Callable[[TaskInfo, int, List[ClientUpdate], Dict[str, float]], None]
        ],
        num_gpus: int,
        backend: str,
    ) -> None:
        """
        Worker function for each GPU process in DistributedDataParallel.
        
        Args:
            rank: Process rank (GPU ID)
            stream: TaskStream
            round_hook: Optional round callback
            num_gpus: Total number of GPUs
            backend: Distributed backend
        """
        import os
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # Set environment variables for distributed training
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        
        # Initialize distributed process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=num_gpus,
            init_method="env://",
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        # Wrap server model with DDP
        self.server.model = DDP(
            self.server.model.to(device),
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,  # Set to False if all params are used
        )
        
        # Wrap each client model with DDP
        for cid, client in self.clients.items():
            if hasattr(client, "model"):
                client.model = DDP(
                    client.model.to(device),
                    device_ids=[rank],
                    output_device=rank,
                    find_unused_parameters=True,
                )
        
        try:
            # Run the standard training loop
            for task, per_client_loaders in stream:
                # Update known classes, and expand heads if needed.
                self._on_new_task(task)
                
                for r in range(self.server.cfg.global_rounds):
                    payload = self.server.broadcast()
                    
                    # Sample clients (simple: intersect with available loaders)
                    available = sorted(per_client_loaders.keys())
                    chosen = available[: min(self.server.cfg.clients_per_round, len(available))]
                    
                    updates: List[ClientUpdate] = []
                    for cid in chosen:
                        c = self.clients[cid]
                        c.load_server_payload(payload)
                        upd = c.fit_one_task(task, per_client_loaders[cid])
                        updates.append(upd)
                    
                    # Synchronize gradients across processes
                    self.server.aggregate(updates)
                    
                    # Synchronize server state across all GPUs
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    
                    self.server.server_optimize_step()
                    
                    # Only rank 0 process logs and calls round_hook to avoid duplicates
                    if rank == 0:
                        if round_hook is not None:
                            metrics = aggregate_client_metrics(updates)
                            round_hook(task, r, updates, metrics)
        
        finally:
            # Cleanup: Destroy process group
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


    @abstractmethod
    def _on_new_task(self, task: TaskInfo) -> None:
        ...

