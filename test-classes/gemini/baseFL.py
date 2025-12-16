import copy
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ==========================================
# 1. Base Model Wrapper
# ==========================================
class BaseFederatedModel(nn.Module, ABC):
    """
    Wraps the Vision Backbone (Classifier) and the Generator.
    """
    def __init__(self):
        super().__init__()
        # 1. The main vision task model
        self.classifier = self.build_classifier()
        # 2. The generative model for replay
        self.generator = self.build_generator()

    @abstractmethod
    def build_classifier(self):
        """Return the backbone (e.g., ResNet)."""
        pass

    @abstractmethod
    def build_generator(self):
        """Return the generative model (e.g., GAN/VAE)."""
        pass

    def get_weights(self):
        """Helper to get state_dict for federation."""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        """Helper to load state_dict."""
        self.load_state_dict(weights)


# ==========================================
# 2. Base Client
# ==========================================
class BaseClient(ABC):
    def __init__(self, client_id, model, device='cuda'):
        self.client_id = client_id
        self.model = model  # This is a BaseFederatedModel
        self.device = device
        self.local_data = None

    def receive_global_weights(self, global_weights):
        """Load global model parameters."""
        self.model.set_weights(global_weights)

    @abstractmethod
    def train_step(self, task_id):
        """
        1. Train Generator on current local data.
        2. Generate replay data using the Generator.
        3. Train Classifier on (Current Real Data + Generated Replay Data).
        """
        pass

    def upload_weights(self):
        """Return local weights to server."""
        return self.model.get_weights()


# ==========================================
# 3. Base Server
# ==========================================
class BaseServer(ABC):
    def __init__(self, global_model, device='cuda'):
        self.global_model = global_model
        self.device = device

    def aggregate(self, client_weights_list):
        """Standard FedAvg: Average weights from selected clients."""
        avg_weights = copy.deepcopy(client_weights_list[0])
        for k in avg_weights.keys():
            for i in range(1, len(client_weights_list)):
                avg_weights[k] += client_weights_list[i][k]
            avg_weights[k] = torch.div(avg_weights[k], len(client_weights_list))
        
        self.global_model.set_weights(avg_weights)
        return avg_weights

    @abstractmethod
    def server_optimization_step(self):
        """
        Your custom step:
        1. Use the global generative model (aggregated from clients).
        2. Generate synthetic data (global replay).
        3. Optimize the global classifier before next round.
        """
        pass