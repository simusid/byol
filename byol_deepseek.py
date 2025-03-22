import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import os

# Define the ResNet-18 encoder adjusted for CIFAR10
class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# Projection and Prediction MLPs
class ProjectionMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.layers(x)

class PredictionMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.layers(x)

# Online and Target Networks
class OnlineNetwork(nn.Module):
    def __init__(self, encoder, projector, predictor):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        q = self.predictor(z)
        return q

class TargetNetwork(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        return z

# Data Augmentations
class ContrastiveTransformations:
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]

# Prepare CIFAR10 Dataset
def prepare_data():
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=ContrastiveTransformations(augmentation, 2))
    return train_dataset

# BYOL Loss Function
def byol_loss(q1, q2, z1, z2):
    q1 = F.normalize(q1, dim=1)
    q2 = F.normalize(q2, dim=1)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    loss = 2 - 2 * ( (q1 * z2.detach()).sum(dim=1).mean() + (q2 * z1.detach()).sum(dim=1).mean() )
    return loss / 2

# Training Loop
def train_byol(train_loader, online_net, target_net, optimizer, epochs, device):
    total_steps = epochs * len(train_loader)
    current_step = 0
    tau_initial = 0.996

    for epoch in range(epochs):
        online_net.train()
        total_loss = 0.0

        for views in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            v1, v2 = views
            v1, v2 = v1.to(device), v2.to(device)

            # Online network predictions
            q1 = online_net(v1)
            q2 = online_net(v2)

            # Target network projections
            with torch.no_grad():
                z1 = target_net(v1)
                z2 = target_net(v2)

            # Compute loss
            loss = byol_loss(q1, q2, z1, z2)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update for target network
            current_step += 1
            tau = 1 - (1 - tau_initial) * ( (math.cos(math.pi * current_step / total_steps) + 1) / 2 )
            with torch.no_grad():
                for online_p, target_p in zip(online_net.encoder.parameters(), target_net.encoder.parameters()):
                    target_p.data = tau * target_p.data + (1 - tau) * online_p.data
                for online_p, target_p in zip(online_net.projector.parameters(), target_net.projector.parameters()):
                    target_p.data = tau * target_p.data + (1 - tau) * online_p.data

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_dataset = prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

    # Initialize networks
    online_encoder = ResNet18Encoder().to(device)
    online_projector = ProjectionMLP().to(device)
    online_predictor = PredictionMLP().to(device)
    online_net = OnlineNetwork(online_encoder, online_projector, online_predictor).to(device)

    target_encoder = ResNet18Encoder().to(device)
    target_projector = ProjectionMLP().to(device)
    target_net = TargetNetwork(target_encoder, target_projector).to(device)

    # Initialize target network with online weights
    target_net.encoder.load_state_dict(online_net.encoder.state_dict())
    target_net.projector.load_state_dict(online_net.projector.state_dict())
    for param in target_net.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = optim.Adam(online_net.parameters(), lr=3e-4)

    # Train BYOL
    train_byol(train_loader, online_net, target_net, optimizer, epochs=100, device=device)

    # Save the encoder for downstream tasks
    torch.save(online_net.encoder.state_dict(), 'byol_encoder.pth')

if __name__ == "__main__":
    main()
