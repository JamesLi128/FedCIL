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
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

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
        """
        Client training with batched processing.
        
        Note: True multiprocessing parallelism with CUDA on a single GPU has
        known issues (ConnectionResetError, CUDA tensor sharing problems).
        This method now uses sequential training which is reliable.
        
        For true parallelism, use run_multi_gpu with multiple GPUs instead.
        
        Args:
            stream: Task stream providing per-client data loaders
            round_hook: Optional callback after each round
            max_concurrent_clients: (Deprecated) Previously used for parallelism,
                                   now ignored as we use sequential training.
        """
        for task, per_client_loaders in stream:
            self._on_new_task(task)

            for r in range(self.server.cfg.global_rounds):
                payload = self.server.broadcast()

                available = sorted(per_client_loaders.keys())
                chosen = available[: min(self.server.cfg.clients_per_round, len(available))]
                if not chosen:
                    continue

                # Sequential client training (reliable approach)
                updates: List[ClientUpdate] = []
                for cid in chosen:
                    client = self.clients[cid]
                    client.load_server_payload(payload)
                    upd = client.fit_one_task(task, per_client_loaders[cid])
                    updates.append(upd)

                self.server.aggregate(updates)
                self.server.server_optimize_step()

                if round_hook is not None:
                    metrics = aggregate_client_metrics(updates)
                    round_hook(task, r, updates, metrics)

    def run_pipelined(
        self,
        stream: TaskStream,
        round_hook: Optional[Callable] = None,
        prefetch_factor: int = 2,
    ) -> None:
        """
        Pipelined client training that overlaps data transfer with computation.
        
        This is a simpler alternative to full multiprocessing that can still
        provide speedups by overlapping CPU-side data loading/preparation with
        GPU computation.
        
        Key optimization: While one client is training on GPU, the next client's
        data is being prepared and transferred asynchronously.
        
        Args:
            stream: Task stream
            round_hook: Optional callback
            prefetch_factor: How many clients' data to prefetch (default 2)
        """
        use_cuda = next(iter(self.clients.values())).device.type == "cuda"
        device = next(iter(self.clients.values())).device
        
        for task, per_client_loaders in stream:
            self._on_new_task(task)

            for r in range(self.server.cfg.global_rounds):
                payload = self.server.broadcast()

                available = sorted(per_client_loaders.keys())
                chosen = available[: min(self.server.cfg.clients_per_round, len(available))]
                if not chosen:
                    continue

                updates: List[ClientUpdate] = []
                
                if use_cuda and len(chosen) > 1:
                    # Pipelined execution with async data transfer
                    # Create separate streams for data transfer and computation
                    compute_stream = torch.cuda.Stream()
                    transfer_stream = torch.cuda.Stream()
                    
                    # Prefetch first client's data
                    prefetch_events = {}
                    prefetch_data = {}
                    
                    for i, cid in enumerate(chosen):
                        client = self.clients[cid]
                        client.load_server_payload(payload)
                        
                        # If we have prefetched data for this client, wait for it
                        if cid in prefetch_events:
                            compute_stream.wait_event(prefetch_events[cid])
                        
                        # Train this client on compute stream
                        with torch.cuda.stream(compute_stream):
                            upd = client.fit_one_task(task, per_client_loaders[cid])
                            updates.append(upd)
                        
                        # Prefetch next client's data while current is training
                        # (This is limited because DataLoader prefetch is internal,
                        # but we can at least overlap the payload loading)
                        if i + 1 < len(chosen):
                            next_cid = chosen[i + 1]
                            with torch.cuda.stream(transfer_stream):
                                # Pre-warm the next client
                                next_client = self.clients[next_cid]
                                next_client.load_server_payload(payload)
                                event = transfer_stream.record_event()
                                prefetch_events[next_cid] = event
                    
                    # Sync at the end
                    torch.cuda.synchronize()
                else:
                    # Sequential fallback
                    for cid in chosen:
                        client = self.clients[cid]
                        client.load_server_payload(payload)
                        upd = client.fit_one_task(task, per_client_loaders[cid])
                        updates.append(upd)

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
    ) -> None:
        """
        Executes federated training using DistributedDataParallel (DDP).
        
        Each GPU processes one or more clients' silo datasets in parallel.
        No manual data partitioning needed since loaders are already silo-partitioned.
        """
        # 1. Strict GPU Check
        num_gpus = torch.cuda.device_count()
        print(f"[Master] Detected {num_gpus} GPUs.")
        if num_gpus < 2:
            raise RuntimeError(
                f"DDP requires at least 2 GPUs, but only {num_gpus} were detected. "
                "Check your CUDA_VISIBLE_DEVICES or MIG configuration."
            )

        # 2. Prepare Stream (Generators cannot be pickled, so we listify)
        try:
            stream_data = list(stream)
            print(f"[Master] Stream captured with {len(stream_data)} tasks.")
        except Exception as e:
            raise RuntimeError(f"Could not convert TaskStream to list for spawning: {e}")

        # 3. Spawn Processes
        mp.spawn(
            self._ddp_worker,
            args=(stream_data, round_hook, num_gpus),
            nprocs=num_gpus,
            join=True
        )


    def _ddp_worker(
        self,
        rank: int,
        stream_data: List,
        round_hook: Optional[Callable],
        world_size: int,
    ) -> None:
        """
        Worker process for DDP.
        
        Each rank (GPU) processes clients in parallel.
        Only rank 0 performs aggregation and optimization.
        """
        # 1. Setup Distributed Environment
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # Ensure server and clients know about the local device
        self.server.device = device
        self.server.model.to(device)
        if self.server.replay is not None and hasattr(self.server.replay, "to"):
            try:
                self.server.replay.to(device)  # type: ignore[attr-defined]
            except Exception:
                pass
        for c in self.clients.values():
            c.device = device
            c.model.to(device)
            if c.replay is not None and hasattr(c.replay, "to"):
                try:
                    c.replay.to(device)  # type: ignore[attr-defined]
                except Exception:
                    pass
        
        if rank == 0:
            print(f"[Rank {rank}] Initialized on {device}")

        try:
            for task, per_client_loaders in stream_data:
                # Update known classes and model structure
                self._on_new_task(task)
                
                # Sync processes to ensure model structure is consistent across all ranks
                dist.barrier()

                for r in range(self.server.cfg.global_rounds):
                    payload = self.server.broadcast()

                    # Select clients for this round
                    available = sorted(per_client_loaders.keys())
                    chosen = available[: min(self.server.cfg.clients_per_round, len(available))]

                    # --- DISTRIBUTE CLIENTS ACROSS GPUs ---
                    # Each GPU processes a subset of the chosen clients
                    # This avoids redundant processing and enables true parallelization
                    clients_per_gpu = (len(chosen) + world_size - 1) // world_size
                    start_idx = rank * clients_per_gpu
                    end_idx = min(start_idx + clients_per_gpu, len(chosen))
                    gpu_chosen = chosen[start_idx:end_idx]

                    if rank == 0:
                        print(f"[Round {r}] Total clients: {len(chosen)}, GPU distribution: {clients_per_gpu} per GPU")
                        print(f"[Rank {rank}] Processing clients: {gpu_chosen}")

                    updates: List[ClientUpdate] = []

                    for cid in gpu_chosen:
                        client = self.clients[cid]

                        # Load payload (weights from server)
                        client.load_server_payload(payload)

                        # Train client locally on its GPU; no DDP wrapping so gradients stay local
                        upd = client.fit_one_task(task, per_client_loaders[cid])
                        updates.append(upd)

                    # --- SYNCHRONIZE BEFORE AGGREGATION ---
                    # All ranks wait here before rank 0 aggregates
                    dist.barrier()

                    # --- AGGREGATION AND SERVER OPTIMIZATION (RANK 0 ONLY) ---
                    # Gather all client updates from all ranks to rank 0
                    # NOTE: This is a simple gather; in production, you might use 
                    # torch.distributed.all_gather, but for FL we typically gather to rank 0
                    if world_size > 1:
                        all_updates = [None] * world_size
                        dist.all_gather_object(all_updates, updates)
                        if rank == 0:
                            flat_updates = [u for upd_list in all_updates if upd_list for u in upd_list]
                            self.server.aggregate(flat_updates)
                    else:
                        if rank == 0:
                            self.server.aggregate(updates)

                    if rank == 0:
                        self.server.server_optimize_step()

                    # --- BROADCAST UPDATED WEIGHTS FROM RANK 0 ---
                    # Ensure all ranks have the same server weights before next round
                    dist.barrier()
                    for param in self.server.model.parameters():
                        dist.broadcast(param.data, src=0)

                    # --- LOGGING (RANK 0 ONLY) ---
                    if rank == 0 and round_hook is not None:
                        if world_size > 1:
                            all_updates = [None] * world_size
                            dist.all_gather_object(all_updates, updates)
                            flat_updates = [u for upd_list in all_updates if upd_list for u in upd_list]
                            metrics = aggregate_client_metrics(flat_updates)
                            round_hook(task, r, flat_updates, metrics)
                        else:
                            metrics = aggregate_client_metrics(updates)
                            round_hook(task, r, updates, metrics)

                    dist.barrier()

        except Exception as e:
            print(f"[Rank {rank}] Error during training: {e}")
            import traceback
            traceback.print_exc()
        finally:
            dist.destroy_process_group()



    @abstractmethod
    def _on_new_task(self, task: TaskInfo) -> None:
        ...

