import torch.optim as optim
import torch.nn.functional as F
from baseFL import *

# ==========================================
# Concrete Model
# ==========================================
class ResNetGANModel(BaseFederatedModel):
    def build_classifier(self):
        # Example: Simple ConvNet or Pretrained ResNet
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 26 * 26, 100) # Assuming 28x28 input for simplicity
        )

    def build_generator(self):
        # Example: Simple Generator (Latent -> Image)
        return nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 3*28*28), nn.Sigmoid()
        )

# ==========================================
# Concrete Client (Local Replay)
# ==========================================
class ReplayClient(BaseClient):
    def train_step(self, dataloader, task_id):
        self.model.to(self.device)
        self.model.train()
        
        # Optimizers
        clf_opt = optim.SGD(self.model.classifier.parameters(), lr=0.01)
        gen_opt = optim.Adam(self.model.generator.parameters(), lr=0.001)

        # 1. Train Generator (Simplified GAN training)
        # In a real project, you would need a Discriminator here too.
        # For simplicity, we assume a VAE-like reconstruction or just generator updates.
        for real_x, _ in dataloader:
            real_x = real_x.to(self.device)
            z = torch.randn(real_x.size(0), 64).to(self.device)
            fake_x = self.model.generator(z)
            # Dummy loss for generator example
            gen_loss = F.mse_loss(fake_x.view(-1, 3*28*28), real_x.view(-1, 3*28*28))
            
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

        # 2. Train Classifier with Replay
        for real_x, real_y in dataloader:
            real_x, real_y = real_x.to(self.device), real_y.to(self.device)

            # Generate Replay Data (Pseudo-rehearsal)
            z = torch.randn(32, 64).to(self.device)
            with torch.no_grad():
                replay_x = self.model.generator(z).view(-1, 3, 28, 28)
                # In real FCIL, you need a way to assign labels to replay data 
                # (e.g., using the previous model copy)
                replay_y = torch.randint(0, task_id * 10, (32,)).to(self.device) # Placeholder

            # Combine
            combined_x = torch.cat((real_x, replay_x))
            combined_y = torch.cat((real_y, replay_y))

            # Optimization
            outputs = self.model.classifier(combined_x)
            clf_loss = F.cross_entropy(outputs, combined_y)
            
            clf_opt.zero_grad()
            clf_loss.backward()
            clf_opt.step()

# ==========================================
# Concrete Server (Global Optimization)
# ==========================================
class GenerativeServer(BaseServer):
    def server_optimization_step(self):
        """
        The 'Server Replay' step you requested.
        After aggregating weights, the server refines the model using 
        the aggregated Generator.
        """
        self.global_model.to(self.device)
        self.global_model.train()
        optimizer = optim.SGD(self.global_model.classifier.parameters(), lr=0.01)

        print("Server: Performing generative replay optimization...")
        
        # 1. Generate synthetic data using the GLOBAL generator
        # (which now contains knowledge aggregated from all clients)
        z = torch.randn(100, 64).to(self.device) # Batch of 100
        with torch.no_grad():
            generated_data = self.global_model.generator(z).view(-1, 3, 28, 28)
            # Use the classifier itself (teacher-forcing) or stored prototypes to label these
            # For simplicity, we predict using the current model to reinforce current knowledge
            outputs = self.global_model.classifier(generated_data)
            pseudo_labels = outputs.argmax(dim=1)

        # 2. Optimize the global classifier on this synthetic data
        # This helps consolidate knowledge at the server level
        pred = self.global_model.classifier(generated_data)
        loss = F.cross_entropy(pred, pseudo_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return self.global_model.get_weights()

# ==========================================
# Usage Flow (Simulation)
# ==========================================
if __name__ == "__main__":
    # Setup
    global_model = ResNetGANModel()
    server = GenerativeServer(global_model)
    client1 = ReplayClient(client_id=1, model=copy.deepcopy(global_model))
    
    # --- Round 1 ---
    # 1. Client trains
    # client1.train_step(dataloader=..., task_id=0)
    w1 = client1.upload_weights()
    
    # 2. Server Aggregates (FedAvg)
    server.aggregate([w1]) # In reality, list of multiple client weights
    
    # 3. Server Optimization Step (Unique to your project)
    final_global_weights = server.server_optimization_step()
    
    # 4. Broadcast back
    client1.receive_global_weights(final_global_weights)