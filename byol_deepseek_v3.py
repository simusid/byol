import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 512
learning_rate = 3e-4
weight_decay = 1e-6
temperature = 0.2
hidden_dim = 2048
projection_dim = 256
moving_average_decay = 0.996
epochs = 200

# Data augmentation for BYOL
class BYOLTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform_prime(x)

# Load CIFAR-10 dataset
train_transform = BYOLTransform()
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Encoder network (ResNet-18)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(resnet.fc.in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.projection(h)

# Predictor network
class Predictor(nn.Module):
    def __init__(self, input_dim=projection_dim, hidden_dim=hidden_dim, output_dim=projection_dim):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# BYOL model
class BYOL(nn.Module):
    def __init__(self):
        super(BYOL, self).__init__()
        self.online_encoder = Encoder()
        self.online_predictor = Predictor()
        
        self.target_encoder = Encoder()
        
        # Initialize target encoder with online encoder weights
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        
        # Initialize target projection with online projection weights
        for param_online, param_target in zip(self.online_encoder.projection.parameters(), self.target_encoder.projection.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
    
    def update_target_network(self, decay=moving_average_decay):
        with torch.no_grad():
            for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                param_target.data = decay * param_target.data + (1 - decay) * param_online.data
            
            for param_online, param_target in zip(self.online_encoder.projection.parameters(), self.target_encoder.projection.parameters()):
                param_target.data = decay * param_target.data + (1 - decay) * param_online.data
    
    def forward(self, x1, x2):
        # Online network forward
        online_proj_one = self.online_encoder(x1)
        online_proj_two = self.online_encoder(x2)
        
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        
        # Target network forward
        with torch.no_grad():
            target_proj_one = self.target_encoder(x2)
            target_proj_two = self.target_encoder(x1)
        
        return online_pred_one, online_pred_two, target_proj_one, target_proj_two

# Loss function
def byol_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

# Initialize model and optimizer
model = BYOL().to(device)
optimizer = optim.Adam(list(model.online_encoder.parameters()) + list(model.online_predictor.parameters()), 
                      lr=learning_rate, weight_decay=weight_decay)

# Training and evaluation tracking
train_losses = []
test_losses = []

def train(epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    
    for (x1, x2), _ in progress_bar:
        x1, x2 = x1.to(device), x2.to(device)
        
        optimizer.zero_grad()
        
        online_pred_one, online_pred_two, target_proj_one, target_proj_two = model(x1, x2)
        
        loss_one = byol_loss(online_pred_one, target_proj_one)
        loss_two = byol_loss(online_pred_two, target_proj_two)
        loss = loss_one + loss_two
        
        loss.backward()
        optimizer.step()
        
        model.update_target_network()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch+1}, Train Loss: {avg_loss:.4f}')

def test():
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            
            online_pred_one, online_pred_two, target_proj_one, target_proj_two = model(x1, x2)
            
            loss_one = byol_loss(online_pred_one, target_proj_one)
            loss_two = byol_loss(online_pred_two, target_proj_two)
            loss = loss_one + loss_two
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    test_losses.append(avg_loss)
    print(f'Test Loss: {avg_loss:.4f}')

# Training loop
for epoch in range(epochs):
    train(epoch)
    if (epoch + 1) % 10 == 0:
        test()

# Function to get embedding feature vector
def get_embedding(image):
    """
    Takes an image tensor (C, H, W) and returns the embedding feature vector
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)
        # Get the projection from the online encoder
        embedding = model.online_encoder(image)
        # Return as numpy array on CPU
        return embedding.squeeze().cpu().numpy()

# Function to plot training and testing metrics
def plot_metrics():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if test_losses:
        x = np.arange(9, len(train_losses), 10)  # Test losses are recorded every 10 epochs
        plt.plot(x, test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Save the model
torch.save(model.state_dict(), 'byol_cifar10.pth')

# Example usage
if __name__ == "__main__":
    # Plot the training curves
    plot_metrics()
    
    # Example of getting an embedding
    image, _ = test_dataset[0]  # Get first test image
    embedding = get_embedding(image)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Sample embedding values: {embedding[:10]}")  # Print first 10 values
