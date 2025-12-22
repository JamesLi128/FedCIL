from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

# -------------------------
# Configs / payloads
# -------------------------
@dataclass
class ClientConfig:
    local_epochs: int = 1
    lr: float = 1e-3
    gan_lr: float = 2e-4
    weight_decay: float = 0.0
    gan_weight_decay: float = 0.0
    batch_size: int = 64
    replay_ratio: float = 0.5              # replay samples per real batch (0..1)
    max_grad_norm: Optional[float] = None,  # gradient clipping
    classification_head_type: str = "singlehead"  # "singlehead" or "multilayer"
    hidden_dim: Optional[int] = None        # for multilayer classifier only


@dataclass
class ServerConfig:
    global_rounds: int = 10
    clients_per_round: int = 10
    server_opt_steps: int = 0              # server-side optimization steps
    server_opt_lr: float = 1e-4
    server_replay_batch: int = 128         # synthetic batch size for server steps


@dataclass
class TaskInfo:
    task_id: int
    new_classes: List[int]                 # the classes introduced in this task


@dataclass
class ClientUpdate:
    client_id: int
    num_samples: int
    model_state: Dict[str, torch.Tensor]
    gen_state: Optional[Dict[str, torch.Tensor]] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class ServerPayload:
    # What the server broadcasts each round
    global_model_state: Dict[str, torch.Tensor]
    global_gen_state: Optional[Dict[str, torch.Tensor]]
    known_classes: List[int]

@dataclass
class GANReplayConfig:
    z_dim: int = 128
    label_embed_dim: int = 128
    gan_lr: float = 2e-4
    gan_weight_decay: float = 0.0005
    gan_steps_per_batch: int = 1
    img_channels: int = 3
    num_total_classes: int = 100  # global label space size (e.g., CIFAR-100)